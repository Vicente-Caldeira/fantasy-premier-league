"""Pre-deadline refresh: predictions made with the freshest team news.

The daily snapshot is taken at 06:00 UTC, so for a Friday-evening deadline the players'
status, chance of playing and the crowd's transfers are about 12 hours old by the time
they matter. Minutes are the largest source of prediction error, and the crowd signal
is the strongest single feature, so both are worth refreshing near the deadline.

`pre_deadline_snapshot` re-runs today's daily snapshot once per gameweek, between
`LEAD` and `LEAD - WINDOW` before the deadline (`bootstrap_static` and `fixtures` refresh
on a same-day re-run; past days are never rewritten).
`warehouse_after_pre_deadline_snapshot` then rebuilds the warehouse and the models
(expected points, plans) on that snapshot.
"""

import io
from datetime import UTC, datetime, timedelta

import dagster as dg
import polars as pl
from dagster._core.definitions.unresolved_asset_job_definition import (
    UnresolvedAssetJobDefinition,
)

from fpl_pipelines.lake import Lake

LEAD = timedelta(hours=2)  # start of the window, before the deadline
WINDOW = timedelta(hours=1)  # the sensor may fire until LEAD - WINDOW before it
GAMEWEEKS = "silver/fpl/gameweeks"
TAG = "fpl/pre_deadline"


def next_deadline(gameweeks: pl.DataFrame, now: datetime) -> tuple[str, int, datetime] | None:
    """(season, gameweek, deadline) of the first deadline after `now`, if any."""
    upcoming = gameweeks.filter(pl.col("deadline_time") > now).sort("deadline_time")
    if upcoming.is_empty():
        return None
    season, gameweek, deadline = upcoming.select("season", "id", "deadline_time").row(0)
    return season, gameweek, deadline


def is_due(deadline: datetime, now: datetime) -> bool:
    return deadline - LEAD <= now < deadline - LEAD + WINDOW


def latest_gameweeks(lake: Lake) -> pl.DataFrame | None:
    """The newest snapshot's gameweeks (deadlines), or None before the first snapshot."""
    partitions = lake.list(GAMEWEEKS)
    if not partitions:
        return None
    return pl.read_parquet(io.BytesIO(lake.read_bytes(f"{partitions[-1]}/data.parquet")))


def pre_deadline_request(
    gameweeks: pl.DataFrame | None, now: datetime, cursor: str | None
) -> dg.RunRequest | dg.SkipReason:
    """The run to request now (today's snapshot partition), or why not."""
    if gameweeks is None:
        return dg.SkipReason("no gameweeks snapshot yet")
    found = next_deadline(gameweeks, now)
    if found is None:
        return dg.SkipReason("no upcoming deadline")
    season, gameweek, deadline = found
    key = f"{season} GW{gameweek}"
    if cursor == key:
        return dg.SkipReason(f"{key} already refreshed")
    if not is_due(deadline, now):
        return dg.SkipReason(f"{key} deadline {deadline:%Y-%m-%d %H:%M} UTC not within {LEAD}")
    return dg.RunRequest(run_key=key, partition_key=now.date().isoformat(), tags={TAG: key})


def build_sensors(
    snapshot_job: UnresolvedAssetJobDefinition, warehouse_job: UnresolvedAssetJobDefinition
) -> list[dg.SensorDefinition]:
    @dg.sensor(job=snapshot_job, minimum_interval_seconds=600)
    def pre_deadline_snapshot(
        context: dg.SensorEvaluationContext, lake: Lake
    ) -> dg.RunRequest | dg.SkipReason:
        """Re-snapshot FPL once per gameweek, about 2 h before its deadline."""
        request = pre_deadline_request(latest_gameweeks(lake), datetime.now(UTC), context.cursor)
        if isinstance(request, dg.RunRequest):
            context.update_cursor(request.run_key)
        return request

    @dg.run_status_sensor(
        run_status=dg.DagsterRunStatus.SUCCESS,
        monitored_jobs=[snapshot_job],
        request_job=warehouse_job,
    )
    def warehouse_after_pre_deadline_snapshot(
        context: dg.RunStatusSensorContext,
    ) -> dg.RunRequest | dg.SkipReason:
        """After a pre-deadline snapshot, rebuild the warehouse and the models on it."""
        key = context.dagster_run.tags.get(TAG)
        if key is None:
            return dg.SkipReason("not a pre-deadline snapshot")
        return dg.RunRequest(run_key=f"warehouse {key}", tags={TAG: key})

    return [pre_deadline_snapshot, warehouse_after_pre_deadline_snapshot]
