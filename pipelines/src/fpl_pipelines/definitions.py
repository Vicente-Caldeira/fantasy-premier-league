"""Dagster definitions for the FPL platform."""

import os

import dagster as dg

from fpl_pipelines.assets import fpl_api
from fpl_pipelines.lake import Lake, PartitionedParquetIOManager
from fpl_pipelines.resources import FplApiResource

lake = Lake(base_url=os.getenv("FPL_LAKE_URL", "data"))

fpl_snapshot_job = dg.define_asset_job(
    "fpl_snapshot",
    selection=dg.AssetSelection.groups("fpl_bronze", "fpl_silver"),
)

defs = dg.Definitions(
    assets=dg.load_assets_from_modules([fpl_api]),
    asset_checks=dg.load_asset_checks_from_modules([fpl_api]),
    jobs=[fpl_snapshot_job],
    schedules=[
        # 06:00 UTC: after overnight price changes, before most news breaks.
        dg.build_schedule_from_partitioned_job(fpl_snapshot_job, hour_of_day=6, minute_of_hour=0)
    ],
    resources={
        "fpl_api": FplApiResource(),
        "lake": lake,
        "parquet_io_manager": PartitionedParquetIOManager(lake=lake),
    },
)
