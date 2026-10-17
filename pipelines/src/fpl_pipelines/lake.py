"""Object storage for the data lake: a local directory in dev, a GCS bucket when hosted.

Layout (all paths relative to `base_url`):
    bronze/<source>/<dataset>/snapshot_date=YYYY-MM-DD/<dataset>.json.gz   immutable raw
    silver/<table>/snapshot_date=YYYY-MM-DD/data.parquet                   typed tables
"""

import gzip
import json
from typing import Any

import dagster as dg
import fsspec
import polars as pl
from fsspec import AbstractFileSystem


class Lake(dg.ConfigurableResource):  # type: ignore[type-arg]
    base_url: str = "data"

    def _fs(self) -> tuple[AbstractFileSystem, str]:
        fs, root = fsspec.core.url_to_fs(self.base_url)
        return fs, root.rstrip("/")

    def uri(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path}"

    def exists(self, path: str) -> bool:
        fs, root = self._fs()
        return bool(fs.exists(f"{root}/{path}"))

    def write_bytes(self, path: str, data: bytes) -> None:
        fs, root = self._fs()
        full = f"{root}/{path}"
        fs.makedirs(full.rsplit("/", 1)[0], exist_ok=True)
        with fs.open(full, "wb") as f:
            f.write(data)

    def read_bytes(self, path: str) -> bytes:
        fs, root = self._fs()
        with fs.open(f"{root}/{path}", "rb") as f:
            data: bytes = f.read()
        return data

    def write_json_gz(self, path: str, payload: Any) -> int:  # noqa: ANN401 - raw JSON
        # mtime=0 makes identical payloads byte-identical, so content hashes are stable.
        data = gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0)
        self.write_bytes(path, data)
        return len(data)

    def read_json_gz(self, path: str) -> Any:  # noqa: ANN401 - raw JSON
        return json.loads(gzip.decompress(self.read_bytes(path)))


class PartitionedParquetIOManager(dg.ConfigurableIOManager):
    """Stores each partition of a Polars DataFrame asset as one Parquet file."""

    lake: dg.ResourceDependency[Lake]

    def _path(self, context: dg.OutputContext | dg.InputContext) -> str:
        table = "/".join(context.asset_key.path)
        return f"{table}/snapshot_date={context.asset_partition_key}/data.parquet"

    def handle_output(self, context: dg.OutputContext, obj: pl.DataFrame) -> None:
        path = self._path(context)
        fs, root = self.lake._fs()
        full = f"{root}/{path}"
        fs.makedirs(full.rsplit("/", 1)[0], exist_ok=True)
        with fs.open(full, "wb") as f:
            obj.write_parquet(f)
        context.add_output_metadata(
            {"uri": self.lake.uri(path), "rows": obj.height, "columns": obj.width}
        )

    def load_input(self, context: dg.InputContext) -> pl.DataFrame:
        fs, root = self.lake._fs()
        with fs.open(f"{root}/{self._path(context)}", "rb") as f:
            return pl.read_parquet(f)
