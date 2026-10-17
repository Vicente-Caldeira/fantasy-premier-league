"""FPL-Elo-Insights ingestion: content-addressed bronze, season-partitioned silver.

Bronze (daily): resolve upstream `main` to a commit, list its tree, download only blobs
not stored yet (verified against their git SHA), and write a manifest that maps each
file to its blob. A manifest pins exactly which upstream state a day's data came from.

Silver (per season): rebuild a season's tables from the latest manifest.
"""

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

import dagster as dg
import polars as pl

from fpl_core.http import RetryingHttpClient
from fpl_pipelines.lake import Lake
from fpl_pipelines.snapshots import current_season, daily_snapshots, season_label, take_snapshot
from fpl_pipelines.sources import fpl_elo_insights as src

BRONZE = ["bronze", "fpl_elo_insights"]
SILVER = ["silver", "fpl_elo_insights"]
MANIFESTS = "bronze/fpl_elo_insights/manifests"
MATCHES_PER_SEASON = 380

elo_seasons = dg.StaticPartitionsDefinition(
    [season_label(y) for y in range(src.FIRST_SEASON, int(current_season()[:4]) + 1)]
)


class GitHubResource(dg.ConfigurableResource):  # type: ignore[type-arg]
    """GitHub API + raw content. A token is optional; it raises the API rate limit."""

    api_url: str = src.API_URL
    raw_url: str = src.RAW_URL
    token: str | None = os.getenv("GITHUB_TOKEN")

    def api(self) -> RetryingHttpClient:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return RetryingHttpClient(self.api_url, headers=headers)

    def raw(self) -> RetryingHttpClient:
        return RetryingHttpClient(self.raw_url)


def blob_path(sha: str) -> str:
    return f"bronze/fpl_elo_insights/blobs/{sha[:2]}/{sha}.csv"


def manifest_path(snapshot_date: str) -> str:
    return f"{MANIFESTS}/snapshot_date={snapshot_date}/manifest.json"


def latest_manifest(lake: Lake) -> dict[str, Any]:
    days = lake.list(MANIFESTS)
    if not days:
        raise dg.Failure("No FPL-Elo-Insights manifest yet; materialise repo_snapshot first")
    manifest: dict[str, Any] = json.loads(lake.read_bytes(f"{days[-1]}/manifest.json"))
    return manifest


@dg.asset(key_prefix=BRONZE, partitions_def=daily_snapshots, group_name="fpl_elo_insights_bronze")
def repo_snapshot(
    context: dg.AssetExecutionContext, github: GitHubResource, lake: Lake
) -> dg.MaterializeResult[None]:
    """Manifest of the upstream commit's files, plus any new content-addressed blobs."""

    def fetch() -> bytes:
        with github.api() as api:
            commit = api.get(f"/repos/{src.REPO}/commits/main").json()["sha"]
            tree = api.get(f"/repos/{src.REPO}/git/trees/{commit}?recursive=1").json()
        if tree.get("truncated"):
            raise dg.Failure("GitHub returned a truncated tree; files would be missed")
        refs = src.select_files(tree["tree"])
        missing = [r for r in refs if not lake.exists(blob_path(r.sha))]

        def download(ref: src.FileRef) -> None:
            content = raw.get(src.raw_path(commit, ref.path)).content
            if src.git_blob_sha(content) != ref.sha:
                raise dg.Failure(f"{ref.path}: content does not match git SHA {ref.sha}")
            lake.write_bytes(blob_path(ref.sha), content)

        with github.raw() as raw, ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, missing))
        context.log.info("Commit %s: %d files, %d new blobs", commit, len(refs), len(missing))
        manifest = {"repo": src.REPO, "commit": commit, "files": [asdict(r) for r in refs]}
        return json.dumps(manifest, indent=1).encode()

    return take_snapshot(context, lake, manifest_path(context.partition_key), fetch)


def _season_files(lake: Lake, season: str, dataset: str) -> list[src.FileRef]:
    refs = [src.FileRef(**f) for f in latest_manifest(lake)["files"]]
    return [r for r in refs if r.season == season and r.dataset == dataset]


def silver_asset(name: str) -> Callable[[Callable[..., pl.DataFrame]], dg.AssetsDefinition]:
    return dg.asset(
        name=name,
        key_prefix=SILVER,
        partitions_def=elo_seasons,
        group_name="fpl_elo_insights_silver",
        io_manager_key="parquet_io_manager",
        metadata={"partition_column": "season"},
        deps=[dg.AssetDep(repo_snapshot, partition_mapping=dg.AllPartitionMapping())],
    )


@silver_asset("matches")
def matches(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """Premier League matches with team-level stats and Elo (one row per fixture)."""
    season = context.partition_key
    refs = _season_files(lake, season, "matches")
    return pl.concat([src.parse_matches(lake.read_bytes(blob_path(r.sha)), season) for r in refs])


@silver_asset("player_match_stats")
def player_match_stats(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """Opta-style per-player match stats: touches, chances created, xG/xA/xGOT, duels."""
    season = context.partition_key
    refs = _season_files(lake, season, "playermatchstats")
    return pl.concat(
        [
            src.parse_player_match_stats(lake.read_bytes(blob_path(r.sha)), season, r.gameweek)
            for r in refs
        ]
    )


@dg.asset_check(asset=matches, blocking=True, partitions_def=elo_seasons)
def full_fixture_list(matches: pl.DataFrame) -> dg.AssetCheckResult:
    duplicates = matches.height - matches["match_id"].n_unique()
    self_matches = matches.filter(pl.col("home_team_code") == pl.col("away_team_code")).height
    unscored = matches.filter(pl.col("finished") & pl.col("home_score").is_null()).height
    teams = matches["home_team_code"].n_unique()
    return dg.AssetCheckResult(
        passed=matches.height == MATCHES_PER_SEASON
        and duplicates == self_matches == unscored == 0
        and teams == 20,
        metadata={
            "rows": matches.height,
            "duplicates": duplicates,
            "self_matches": self_matches,
            "finished_without_score": unscored,
            "teams": teams,
        },
    )


@dg.asset_check(asset=matches, partitions_def=elo_seasons)
def elo_coverage(matches: pl.DataFrame) -> dg.AssetCheckResult:
    """Warn when Elo ratings are missing; upstream stopped filling them in 2026-27."""
    missing = matches.filter(pl.col("home_elo").is_null() | pl.col("away_elo").is_null()).height
    return dg.AssetCheckResult(
        passed=missing == 0,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={"fixtures": matches.height, "missing_elo": missing},
    )


@dg.asset_check(
    asset=player_match_stats,
    blocking=True,
    partitions_def=elo_seasons,
    additional_ins={"matches": dg.AssetIn(key=matches.key)},
)
def player_rows_are_valid(
    player_match_stats: pl.DataFrame, matches: pl.DataFrame
) -> dg.AssetCheckResult:
    duplicates = player_match_stats.height - player_match_stats.n_unique(["player_id", "match_id"])
    orphans = player_match_stats.join(matches, on="match_id", how="anti").height
    bad_minutes = player_match_stats.filter(~pl.col("minutes_played").is_between(0, 130)).height
    counts = [c for c, t in player_match_stats.schema.items() if t == pl.Int64]
    negative = player_match_stats.filter(pl.any_horizontal(pl.col(counts) < 0)).height
    return dg.AssetCheckResult(
        passed=duplicates == orphans == bad_minutes == negative == 0,
        metadata={
            "rows": player_match_stats.height,
            "duplicates": duplicates,
            "orphan_match_ids": orphans,
            "bad_minutes": bad_minutes,
            "negative_counts": negative,
        },
    )
