"""End-to-end materialisation of the FPL snapshot job against real captured payloads."""

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import dagster as dg
import polars as pl
import pytest

from fpl_pipelines import snapshots
from fpl_pipelines.assets import fpl_api
from fpl_pipelines.definitions import daily_snapshot_job, defs
from fpl_pipelines.lake import Lake, PartitionedParquetIOManager
from fpl_pipelines.resources import FplApiResource

FIXTURES = Path(__file__).parent / "fixtures"
SUMMARY = json.loads(
    (
        Path(__file__).parents[2] / "packages/fpl_core/tests/fixtures/element_summary.json"
    ).read_text()
)


def load_gz(name):
    return json.loads(gzip.decompress((FIXTURES / name).read_bytes()))


class FakeClient:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def bootstrap_static(self):
        self.calls.append("bootstrap_static")
        return load_gz("bootstrap_static.json.gz")

    def fixtures(self):
        self.calls.append("fixtures")
        return load_gz("fixtures.json.gz")

    def element_summaries(self, ids):
        self.calls.append("element_summaries")
        # The captured history belongs to a goalkeeper, so goalkeepers get it (re-keyed so
        # (element, fixture) stays unique) and outfielders get none: official points stay
        # consistent with each player's position.
        keepers = {
            e["id"]
            for e in load_gz("bootstrap_static.json.gz")["elements"]
            if e["element_type"] == 1
        }
        history = SUMMARY["history"]
        return {
            i: {"history": [{**m, "element": i} for m in history] if i in keepers else []}
            for i in ids
        }


class FakeFplApi(FplApiResource):
    def client(self):
        return FakeClient(CALLS)


CALLS: list[str] = []
TODAY = datetime.now(UTC).date().isoformat()
ALL_ASSETS = [*dg.load_assets_from_modules([fpl_api])]
ALL_CHECKS = [*dg.load_asset_checks_from_modules([fpl_api])]


@pytest.fixture
def lake(tmp_path):
    return Lake(base_url=str(tmp_path))


def run(lake, partition_key=TODAY):
    CALLS.clear()
    return dg.materialize(
        [*ALL_ASSETS, *ALL_CHECKS],
        partition_key=partition_key,
        resources={
            "fpl_api": FakeFplApi(),
            "lake": lake,
            "parquet_io_manager": PartitionedParquetIOManager(lake=lake),
        },
        raise_on_error=False,
    )


def read_table(lake, name):
    return pl.read_parquet(
        Path(lake.base_url) / f"silver/fpl/{name}/snapshot_date={TODAY}/data.parquet"
    )


def test_snapshot_job_materialises_bronze_and_silver(lake):
    result = run(lake)
    assert result.success

    for dataset in ("bootstrap_static", "fixtures", "element_summaries"):
        assert lake.exists(fpl_api.bronze_path(dataset, TODAY))

    players = read_table(lake, "players")
    assert players.height == 667
    assert players["season"].unique().to_list() == ["2026-27"]
    assert players.schema["selected_by_percent"] == pl.Float64
    assert "pulse_id" not in read_table(lake, "teams").columns
    keepers = players.filter(pl.col("element_type") == 1).height
    assert read_table(lake, "player_matches").height == keepers * len(SUMMARY["history"])


def test_all_quality_checks_pass_on_real_data(lake):
    result = run(lake)
    checks = {c.check_name: c.passed for c in result.get_asset_check_evaluations()}
    assert checks == {
        "twenty_unique_teams": True,
        "thirty_eight_gameweeks": True,
        "players_are_valid": True,
        "full_season_of_fixtures": True,
        "one_row_per_player_fixture": True,
        "points_match_official_scoring": True,
    }


def test_scoring_check_flags_points_that_disagree(lake, monkeypatch):
    tampered = {**SUMMARY, "history": [{**m, "total_points": 99} for m in SUMMARY["history"]]}
    monkeypatch.setitem(globals(), "SUMMARY", tampered)
    result = run(lake)
    check = next(
        c
        for c in result.get_asset_check_evaluations()
        if c.check_name == "points_match_official_scoring"
    )
    assert not check.passed
    assert check.metadata["mismatches"].value > 0


def test_rerun_reuses_existing_snapshot_without_calling_api(lake):
    run(lake)
    first = lake.read_bytes(fpl_api.bronze_path("bootstrap_static", TODAY))
    result = run(lake)
    assert result.success
    assert CALLS == ["fixtures"]  # only fixtures refresh during the day
    assert lake.read_bytes(fpl_api.bronze_path("bootstrap_static", TODAY)) == first


def test_rerun_today_refreshes_fixtures_with_latest_results(lake, monkeypatch):
    run(lake)
    payload = load_gz("fixtures.json.gz")
    payload[0] = {**payload[0], "finished_provisional": True, "team_h_score": 4}
    monkeypatch.setattr(FakeClient, "fixtures", lambda self: payload)
    assert run(lake).success
    stored = json.loads(gzip.decompress(lake.read_bytes(fpl_api.bronze_path("fixtures", TODAY))))
    assert stored[0]["team_h_score"] == 4
    table = read_table(lake, "fixtures_table")
    assert table.filter(pl.col("id") == payload[0]["id"])["team_h_score"].item() == 4


def test_fixtures_refresh_schedule_targets_today():
    schedule = defs.get_schedule_def("fixtures_refresh_schedule")
    context = dg.build_schedule_context(
        scheduled_execution_time=datetime.now(UTC), repository_def=defs.get_repository_def()
    )
    run_request = next(iter(schedule.evaluate_tick(context).run_requests or []))
    assert run_request.partition_key == TODAY


def test_snapshot_guard_rejects_any_date_but_today():
    with pytest.raises(dg.Failure, match="can only be taken on that date"):
        snapshots.assert_live("2026-01-01")
    snapshots.assert_live(TODAY)


def test_backfilling_a_past_snapshot_is_refused(lake):
    result = run(lake, partition_key=snapshots.FIRST_SNAPSHOT_DATE)
    if snapshots.FIRST_SNAPSHOT_DATE == TODAY:
        pytest.skip("first snapshot date is today")
    assert not result.success
    assert not lake.exists(fpl_api.bronze_path("bootstrap_static", snapshots.FIRST_SNAPSHOT_DATE))


def test_blocking_check_stops_bad_data_reaching_downstream(lake, monkeypatch):
    payload = load_gz("bootstrap_static.json.gz")
    payload["teams"] = payload["teams"][:19]
    monkeypatch.setattr(FakeClient, "bootstrap_static", lambda self: payload)
    result = run(lake)
    failed = [c.check_name for c in result.get_asset_check_evaluations() if not c.passed]
    assert "twenty_unique_teams" in failed


def test_schedule_targets_todays_partition():
    schedule = defs.get_schedule_def("daily_snapshot_schedule")
    tick = datetime.now(UTC).replace(hour=6, minute=0, second=0, microsecond=0)
    context = dg.build_schedule_context(scheduled_execution_time=tick)
    run_request = next(iter(schedule.evaluate_tick(context).run_requests or []))
    assert run_request.partition_key == tick.date().isoformat()
    assert schedule.job_name == daily_snapshot_job.name


def test_every_check_shares_its_assets_partitioning():
    # A check on a partitioned asset that isn't partitioned itself loads every partition
    # and crashes once a second one exists. (dbt tests on unpartitioned models are fine.)
    graph = defs.resolve_asset_graph()
    mismatched = [
        key.name
        for key in graph.asset_check_keys
        if graph.get(key).partitions_def != graph.get(key.asset_key).partitions_def
    ]
    assert mismatched == []
