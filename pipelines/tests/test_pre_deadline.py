"""Pre-deadline refresh: fire once per gameweek, about two hours before its deadline."""

from datetime import UTC, datetime, timedelta

import dagster as dg
import polars as pl

from fpl_pipelines import pre_deadline
from fpl_pipelines.definitions import defs
from fpl_pipelines.lake import Lake

DEADLINE = datetime(2026, 11, 6, 18, 30, tzinfo=UTC)


def gameweeks():
    return pl.DataFrame({
        "id": [10, 11],
        "season": ["2026-27", "2026-27"],
        "deadline_time": [DEADLINE - timedelta(days=7), DEADLINE],
    })  # fmt: skip


def test_fires_inside_the_window_for_todays_partition():
    now = DEADLINE - timedelta(minutes=110)
    request = pre_deadline.pre_deadline_request(gameweeks(), now, cursor=None)
    assert isinstance(request, dg.RunRequest)
    assert request.partition_key == "2026-11-06"
    assert request.tags[pre_deadline.TAG] == "2026-27 GW11"


def test_skips_outside_the_window_and_once_refreshed():
    for minutes in (125, 59, 10):  # too early, too late, after the window
        now = DEADLINE - timedelta(minutes=minutes)
        assert isinstance(pre_deadline.pre_deadline_request(gameweeks(), now, None), dg.SkipReason)
    now = DEADLINE - timedelta(minutes=90)
    skip = pre_deadline.pre_deadline_request(gameweeks(), now, cursor="2026-27 GW11")
    assert isinstance(skip, dg.SkipReason)


def test_no_snapshot_or_no_upcoming_deadline_skips():
    assert isinstance(pre_deadline.pre_deadline_request(None, DEADLINE, None), dg.SkipReason)
    after = DEADLINE + timedelta(days=1)
    assert isinstance(pre_deadline.pre_deadline_request(gameweeks(), after, None), dg.SkipReason)


def test_reads_the_newest_gameweeks_snapshot(tmp_path):
    lake = Lake(base_url=str(tmp_path))
    for day, deadline in (("2026-11-01", DEADLINE), ("2026-11-02", DEADLINE + timedelta(hours=1))):
        frame = gameweeks().with_columns(pl.lit(deadline).alias("deadline_time"))
        path = tmp_path / pre_deadline.GAMEWEEKS / f"snapshot_date={day}" / "data.parquet"
        path.parent.mkdir(parents=True)
        frame.write_parquet(path)
    latest = pre_deadline.latest_gameweeks(lake)
    assert latest is not None
    assert latest["deadline_time"][0] == DEADLINE + timedelta(hours=1)
    assert pre_deadline.latest_gameweeks(Lake(base_url=str(tmp_path / "empty"))) is None


def test_sensors_are_registered():
    names = {s.name for s in defs.sensors or []}
    assert {"pre_deadline_snapshot", "warehouse_after_pre_deadline_snapshot"} <= names
