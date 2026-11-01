"""Dagster definitions for the FPL platform."""

import os
from datetime import UTC, datetime

import dagster as dg
from dagster_dbt import DbtCliResource

from fpl_pipelines.assets import (
    football_data,
    fpl_api,
    fpl_elo_insights,
    models,
    understat,
    vaastav,
    warehouse,
)
from fpl_pipelines.github_repo import GitHubResource
from fpl_pipelines.lake import Lake, PartitionedParquetIOManager
from fpl_pipelines.resources import FplApiResource
from fpl_pipelines.snapshots import current_season

ASSET_MODULES = [fpl_api, football_data, fpl_elo_insights, vaastav, understat, warehouse, models]

lake = Lake(base_url=os.getenv("FPL_LAKE_URL", "data"))

daily_snapshot_job = dg.define_asset_job(
    "daily_snapshot",
    selection=dg.AssetSelection.groups(
        "fpl_bronze",
        "fpl_silver",
        "football_data_bronze",
        "football_data_silver",
        "fpl_elo_insights_bronze",
        "vaastav_bronze",
    )
    - dg.AssetSelection.assets(football_data.season_results, football_data.matches),
)

# Intra-day: results land and kickoffs move between daily snapshots. Refresh today's
# fixtures snapshot, then the match mart and the fixture expectations built on it
# (staging/intermediate models are views, so only the mart table needs rebuilding).
fixtures_refresh_job = dg.define_asset_job(
    "fixtures_refresh",
    selection=dg.AssetSelection.assets(fpl_api.fixtures, fpl_api.fixtures_table),
)
matches_refresh_job = dg.define_asset_job(
    "matches_refresh",
    selection=dg.AssetSelection.keys("fct_matches", "gold/team_strength/fixture_lambdas"),
)


@dg.schedule(job=fixtures_refresh_job, cron_schedule="15 */2 * * *", execution_timezone="UTC")
def fixtures_refresh_schedule() -> dg.RunRequest:
    """Every two hours: today's fixtures partition, re-fetched."""
    return dg.RunRequest(partition_key=datetime.now(UTC).date().isoformat())


matches_refresh_schedule = dg.ScheduleDefinition(
    job=matches_refresh_job, cron_schedule="35 */2 * * *", execution_timezone="UTC"
)

season_refresh_job = dg.define_asset_job(
    "season_refresh",
    selection=dg.AssetSelection.assets(football_data.season_results, football_data.matches),
)


elo_insights_refresh_job = dg.define_asset_job(
    "elo_insights_refresh", selection=dg.AssetSelection.groups("fpl_elo_insights_silver")
)


# Finished seasons don't change: backfill once, re-run only if upstream fixes data.
vaastav_history_job = dg.define_asset_job(
    "vaastav_history", selection=dg.AssetSelection.groups("vaastav_silver")
)


understat_refresh_job = dg.define_asset_job(
    "understat_refresh", selection=dg.AssetSelection.groups("understat_bronze", "understat_silver")
)


@dg.schedule(job=understat_refresh_job, cron_schedule="15 8 * * 1", execution_timezone="UTC")
def understat_refresh_schedule() -> dg.RunRequest:
    """Mondays 08:15 UTC: the current season's league JSON, plus one request per match
    played since the last run."""
    return dg.RunRequest(partition_key=current_season())


# Rebuild the marts, then re-run the backtests on them.
warehouse_job = dg.define_asset_job(
    "warehouse_build",
    selection=dg.AssetSelection.assets(warehouse.warehouse) | dg.AssetSelection.groups("models"),
)

# 07:00 UTC: after the daily snapshot (06:00) and the current-season refreshes (06:30).
warehouse_schedule = dg.ScheduleDefinition(
    job=warehouse_job, cron_schedule="0 7 * * *", execution_timezone="UTC"
)


@dg.schedule(job=elo_insights_refresh_job, cron_schedule="30 6 * * *", execution_timezone="UTC")
def elo_insights_refresh_schedule() -> dg.RunRequest:
    """Daily 06:30 UTC, after the snapshot: rebuild the current season's tables."""
    return dg.RunRequest(partition_key=current_season())


@dg.schedule(job=season_refresh_job, cron_schedule="0 8 * * 1", execution_timezone="UTC")
def season_refresh_schedule() -> dg.RunRequest:
    """Mondays 08:00 UTC: pick up the weekend's results for the current season."""
    return dg.RunRequest(partition_key=current_season())


defs = dg.Definitions(
    assets=dg.load_assets_from_modules(ASSET_MODULES),
    asset_checks=dg.load_asset_checks_from_modules(ASSET_MODULES),
    jobs=[
        daily_snapshot_job,
        season_refresh_job,
        elo_insights_refresh_job,
        vaastav_history_job,
        understat_refresh_job,
        warehouse_job,
        fixtures_refresh_job,
        matches_refresh_job,
    ],
    schedules=[
        # 06:00 UTC: after overnight price changes, before most news breaks.
        dg.build_schedule_from_partitioned_job(daily_snapshot_job, hour_of_day=6, minute_of_hour=0),
        season_refresh_schedule,
        elo_insights_refresh_schedule,
        understat_refresh_schedule,
        warehouse_schedule,
        fixtures_refresh_schedule,
        matches_refresh_schedule,
    ],
    resources={
        "fpl_api": FplApiResource(),
        "football_data": football_data.FootballDataResource(),
        "github": GitHubResource(),
        "understat": understat.UnderstatResource(),
        "dbt": DbtCliResource(project_dir=warehouse.dbt_project),
        "lake": lake,
        "parquet_io_manager": PartitionedParquetIOManager(lake=lake),
    },
)
