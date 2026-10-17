"""Model evaluation assets, downstream of the warehouse marts."""

import io
import os
from datetime import UTC, datetime
from pathlib import Path

import dagster as dg

from fpl_models.baselines import BASELINES
from fpl_models.data import load_player_matches, player_gameweeks
from fpl_models.leaderboard import EVALUATION_SEASONS, PRIMARY_METRIC, leaderboard, to_markdown
from fpl_pipelines.lake import Lake


@dg.asset(
    key=["gold", "backtests", "leaderboard"],
    deps=[dg.AssetKey("fct_player_matches")],
    group_name="models",
)
def backtest_leaderboard(
    context: dg.AssetExecutionContext, lake: Lake
) -> dg.MaterializeResult[None]:
    """Walk-forward backtest of every registered predictor, ranked by the primary metric.

    Each run is also logged to MLflow when MLFLOW_TRACKING_URI is set.
    """
    warehouse = Path(os.getenv("FPL_WAREHOUSE", "data/warehouse.duckdb"))
    board = leaderboard(
        BASELINES, player_gameweeks(load_player_matches(warehouse)), EVALUATION_SEASONS
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = f"gold/backtests/leaderboard/run_at={stamp}/leaderboard.parquet"
    buffer = io.BytesIO()
    board.write_parquet(buffer)
    lake.write_bytes(path, buffer.getvalue())

    champion = board.row(0, named=True)
    context.log.info(
        "Champion %s: %s=%.3f", champion["predictor"], PRIMARY_METRIC, champion[PRIMARY_METRIC]
    )
    return dg.MaterializeResult(
        metadata={
            "uri": lake.uri(path),
            "champion": champion["predictor"],
            PRIMARY_METRIC: champion[PRIMARY_METRIC],
            "seasons": ", ".join(EVALUATION_SEASONS),
            "leaderboard": dg.MetadataValue.md(to_markdown(board)),
        }
    )
