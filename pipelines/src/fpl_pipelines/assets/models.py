"""Model evaluation assets, downstream of the warehouse marts."""

import io
import os
from datetime import UTC, datetime
from pathlib import Path

import dagster as dg
import polars as pl

from fpl_models import team_strength
from fpl_models.baselines import BASELINES
from fpl_models.data import load_player_matches, player_gameweeks
from fpl_models.leaderboard import EVALUATION_SEASONS, PRIMARY_METRIC, leaderboard, to_markdown
from fpl_pipelines.lake import Lake


def _warehouse() -> Path:
    return Path(os.getenv("FPL_WAREHOUSE", "data/warehouse.duckdb"))


def _write_parquet(lake: Lake, path: str, frame: pl.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    lake.write_bytes(path, buffer.getvalue())
    return lake.uri(path)


TEAM_SUMMARY_COLUMNS = (
    "model",
    "horizon",
    "goals_deviance",
    "clean_sheet_log_loss",
    "outcome_log_loss",
)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


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
    board = leaderboard(
        BASELINES, player_gameweeks(load_player_matches(_warehouse())), EVALUATION_SEASONS
    )
    uri = _write_parquet(
        lake, f"gold/backtests/leaderboard/run_at={_stamp()}/leaderboard.parquet", board
    )

    champion = board.row(0, named=True)
    context.log.info(
        "Champion %s: %s=%.3f", champion["predictor"], PRIMARY_METRIC, champion[PRIMARY_METRIC]
    )
    return dg.MaterializeResult(
        metadata={
            "uri": uri,
            "champion": champion["predictor"],
            PRIMARY_METRIC: champion[PRIMARY_METRIC],
            "seasons": ", ".join(EVALUATION_SEASONS),
            "leaderboard": dg.MetadataValue.md(to_markdown(board)),
        }
    )


@dg.asset(
    key=["gold", "backtests", "team_strength"],
    deps=[dg.AssetKey("fct_matches")],
    group_name="models",
)
def team_strength_leaderboard(lake: Lake) -> dg.MaterializeResult[None]:
    """Match-level walk-forward backtest of team-strength models, incl. planning horizons."""
    board = team_strength.leaderboard(team_strength.load_matches(_warehouse()))
    uri = _write_parquet(
        lake, f"gold/backtests/team_strength/run_at={_stamp()}/leaderboard.parquet", board
    )
    best = board.filter(pl.col("horizon") == 0).row(0, named=True)
    return dg.MaterializeResult(
        metadata={
            "uri": uri,
            "champion": best["model"],
            team_strength.PRIMARY_METRIC: best[team_strength.PRIMARY_METRIC],
            "leaderboard": dg.MetadataValue.md(
                to_markdown(
                    board,
                    [
                        "model",
                        "horizon",
                        "goals_deviance",
                        "clean_sheet_log_loss",
                        "outcome_log_loss",
                    ],
                )
            ),
        }
    )


@dg.asset(
    key=["gold", "team_strength", "fixture_lambdas"],
    deps=[dg.AssetKey("fct_matches")],
    group_name="models",
)
def fixture_lambdas(lake: Lake) -> dg.MaterializeResult[None]:
    """Expected goals, outcome and clean-sheet probabilities for every unplayed fixture."""
    now = datetime.now(UTC)
    expectations = team_strength.fixture_expectations(
        team_strength.load_matches(_warehouse()), as_of=now
    )
    uri = _write_parquet(
        lake, f"gold/team_strength/fixture_lambdas/as_of={_stamp()}/data.parquet", expectations
    )
    sources = dict(expectations.group_by("source").len().iter_rows()) if expectations.height else {}
    return dg.MaterializeResult(
        metadata={
            "uri": uri,
            "fixtures": expectations.height,
            **{f"from_{k}": v for k, v in sources.items()},
        }
    )
