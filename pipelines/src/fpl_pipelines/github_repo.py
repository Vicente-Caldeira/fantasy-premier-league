"""Commit-pinned, content-addressed snapshots of data published in GitHub repos.

One tree call resolves a branch to a commit and lists every file with its git blob
SHA. Only blobs not stored yet are downloaded, each verified against its SHA, and a
daily manifest maps the selected files to blobs, pinning exactly which upstream state
the data came from. Layout under `bronze/<source>/`:

    manifests/snapshot_date=YYYY-MM-DD/manifest.json
    blobs/<sha[:2]>/<sha>.csv
"""

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote

import dagster as dg

from fpl_core.http import RetryingHttpClient
from fpl_pipelines.lake import Lake
from fpl_pipelines.snapshots import take_snapshot

API_URL = "https://api.github.com"
RAW_URL = "https://raw.githubusercontent.com"


@dataclass(frozen=True)
class FileRef:
    season: str  # "2026-27"
    dataset: str
    path: str
    sha: str
    gameweek: int | None = None


def git_blob_sha(content: bytes) -> str:
    """The SHA git assigns a file's content, used to verify downloads."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content, usedforsecurity=False).hexdigest()


class GitHubResource(dg.ConfigurableResource):  # type: ignore[type-arg]
    """GitHub API + raw content. A token is optional; it raises the API rate limit."""

    api_url: str = API_URL
    raw_url: str = RAW_URL
    token: str | None = os.getenv("GITHUB_TOKEN")

    def api(self) -> RetryingHttpClient:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return RetryingHttpClient(self.api_url, headers=headers)

    def raw(self) -> RetryingHttpClient:
        return RetryingHttpClient(self.raw_url)


@dataclass(frozen=True)
class RepoSource:
    name: str  # lake folder, e.g. "fpl_elo_insights"
    repo: str  # "owner/name"
    branch: str
    select: Callable[[list[dict[str, Any]]], list[FileRef]]

    def blob_path(self, sha: str) -> str:
        return f"bronze/{self.name}/blobs/{sha[:2]}/{sha}.csv"

    def manifest_path(self, snapshot_date: str) -> str:
        return f"bronze/{self.name}/manifests/snapshot_date={snapshot_date}/manifest.json"

    def latest_manifest(self, lake: Lake) -> dict[str, Any]:
        days = lake.list(f"bronze/{self.name}/manifests")
        if not days:
            raise dg.Failure(f"No {self.name} manifest yet; materialise its snapshot first")
        manifest: dict[str, Any] = json.loads(lake.read_bytes(f"{days[-1]}/manifest.json"))
        return manifest

    def files(self, lake: Lake, season: str, dataset: str) -> list[FileRef]:
        refs = [FileRef(**f) for f in self.latest_manifest(lake)["files"]]
        return [r for r in refs if r.season == season and r.dataset == dataset]

    def read(self, lake: Lake, ref: FileRef) -> bytes:
        return lake.read_bytes(self.blob_path(ref.sha))

    def snapshot(
        self, context: dg.AssetExecutionContext, github: GitHubResource, lake: Lake
    ) -> dg.MaterializeResult[None]:
        def fetch() -> bytes:
            with github.api() as api:
                commit = api.get(f"/repos/{self.repo}/commits/{self.branch}").json()["sha"]
                tree = api.get(f"/repos/{self.repo}/git/trees/{commit}?recursive=1").json()
            if tree.get("truncated"):
                raise dg.Failure("GitHub returned a truncated tree; files would be missed")
            refs = self.select(tree["tree"])
            missing = {r.sha: r for r in refs if not lake.exists(self.blob_path(r.sha))}

            def download(ref: FileRef) -> None:
                content = raw.get(f"/{self.repo}/{commit}/{quote(ref.path)}").content
                if git_blob_sha(content) != ref.sha:
                    raise dg.Failure(f"{ref.path}: content does not match git SHA {ref.sha}")
                lake.write_bytes(self.blob_path(ref.sha), content)

            with github.raw() as raw, ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(download, missing.values()))
            context.log.info("Commit %s: %d files, %d new blobs", commit, len(refs), len(missing))
            manifest = {"repo": self.repo, "commit": commit, "files": [asdict(r) for r in refs]}
            return json.dumps(manifest, indent=1).encode()

        return take_snapshot(context, lake, self.manifest_path(context.partition_key), fetch)
