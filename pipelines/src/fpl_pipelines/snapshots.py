"""Partitioning shared by every source.

Daily snapshots capture sources that only expose *current* state (FPL prices and news,
upcoming-match odds). A snapshot can only be taken on its own date, so materialising a
past partition fails instead of storing today's data under yesterday's label.

Season partitions cover sources that publish one file per season.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime

import dagster as dg

from fpl_pipelines.lake import Lake

FIRST_SNAPSHOT_DATE = "2026-10-17"
FIRST_SEASON = 2016

daily_snapshots = dg.DailyPartitionsDefinition(
    start_date=FIRST_SNAPSHOT_DATE, timezone="UTC", end_offset=1
)


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season(today: date | None = None) -> str:
    """Seasons run August to May; July belongs to the season about to start."""
    today = today or datetime.now(UTC).date()
    return season_label(today.year if today.month >= 7 else today.year - 1)


seasons = dg.StaticPartitionsDefinition(
    [season_label(y) for y in range(FIRST_SEASON, int(current_season()[:4]) + 1)]
)


def assert_live(snapshot_date: str) -> None:
    today = datetime.now(UTC).date().isoformat()
    if snapshot_date != today:
        raise dg.Failure(
            f"Snapshot {snapshot_date} can only be taken on that date (today is {today}): "
            "the source has no history, so a backfill would mislabel current data."
        )


def take_snapshot(
    context: dg.AssetExecutionContext,
    lake: Lake,
    path: str,
    fetch: Callable[[], bytes],
    *,
    refresh_today: bool = False,
) -> dg.MaterializeResult[None]:
    """Write `fetch()` to `path` once; later runs of the partition reuse it.

    `refresh_today`: for sources that change during the day (fixtures: results, moved
    kickoffs), a re-run on the snapshot's own date replaces it with the current state.
    Past partitions are never rewritten.
    """
    today = datetime.now(UTC).date().isoformat()
    if lake.exists(path) and not (refresh_today and context.partition_key == today):
        context.log.info("Snapshot %s already exists, reusing it", path)
        return dg.MaterializeResult(metadata={"uri": lake.uri(path), "reused": True})
    assert_live(context.partition_key)
    data = fetch()
    lake.write_bytes(path, data)
    return dg.MaterializeResult(
        metadata={
            "uri": lake.uri(path),
            "bytes": len(data),
            "fetched_at": datetime.now(UTC).isoformat(),
            "reused": False,
        }
    )


def latest_version(lake: Lake, prefix: str) -> str | None:
    versions = lake.list(prefix)
    return versions[-1] if versions else None


def store_version(
    lake: Lake, prefix: str, data: bytes, extension: str
) -> dg.MaterializeResult[None]:
    """For files re-published in place: keep a new immutable version only on change."""
    latest = latest_version(lake, prefix)
    if latest is not None and lake.read_bytes(latest) == data:
        return dg.MaterializeResult(metadata={"uri": lake.uri(latest), "changed": False})
    # Microseconds keep versions distinct and lexicographically ordered by fetch time.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = f"{prefix}/fetched_at={stamp}.{extension}"
    lake.write_bytes(path, data)
    return dg.MaterializeResult(
        metadata={"uri": lake.uri(path), "bytes": len(data), "changed": True}
    )
