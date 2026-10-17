"""Dagster definitions for the FPL platform."""

import os

import dagster as dg

from fpl_pipelines.assets import football_data, fpl_api, fpl_elo_insights
from fpl_pipelines.lake import Lake, PartitionedParquetIOManager
from fpl_pipelines.resources import FplApiResource
from fpl_pipelines.snapshots import current_season

ASSET_MODULES = [fpl_api, football_data, fpl_elo_insights]

lake = Lake(base_url=os.getenv("FPL_LAKE_URL", "data"))

daily_snapshot_job = dg.define_asset_job(
    "daily_snapshot",
    selection=dg.AssetSelection.groups(
        "fpl_bronze",
        "fpl_silver",
        "football_data_bronze",
        "football_data_silver",
        "fpl_elo_insights_bronze",
    )
    - dg.AssetSelection.assets(football_data.season_results, football_data.matches),
)

season_refresh_job = dg.define_asset_job(
    "season_refresh",
    selection=dg.AssetSelection.assets(football_data.season_results, football_data.matches),
)


elo_insights_refresh_job = dg.define_asset_job(
    "elo_insights_refresh", selection=dg.AssetSelection.groups("fpl_elo_insights_silver")
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
    jobs=[daily_snapshot_job, season_refresh_job, elo_insights_refresh_job],
    schedules=[
        # 06:00 UTC: after overnight price changes, before most news breaks.
        dg.build_schedule_from_partitioned_job(daily_snapshot_job, hour_of_day=6, minute_of_hour=0),
        season_refresh_schedule,
        elo_insights_refresh_schedule,
    ],
    resources={
        "fpl_api": FplApiResource(),
        "football_data": football_data.FootballDataResource(),
        "github": fpl_elo_insights.GitHubResource(),
        "lake": lake,
        "parquet_io_manager": PartitionedParquetIOManager(lake=lake),
    },
)
