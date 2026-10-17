"""Run predictors through the backtest, rank them, and log each run to MLflow.

    uv run python -m fpl_models.leaderboard --seasons 2023-24 2024-25 2025-26

MLflow logging is on when MLFLOW_TRACKING_URI is set (e.g. http://localhost:5050).
"""

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from fpl_models.backtest import Predictor, run_backtest
from fpl_models.baselines import BASELINES
from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks
from fpl_models.metrics import evaluate

PRIMARY_METRIC = "regulars_mae"
EVALUATION_SEASONS = ("2023-24", "2024-25", "2025-26")
EXPERIMENT = "backtests"


def _log_to_mlflow(name: str, seasons: Sequence[str], metrics: dict[str, float]) -> None:
    import mlflow  # noqa: PLC0415 - optional dependency path, only when tracking

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"predictor": name, "seasons": ",".join(seasons)})
        mlflow.log_metrics(metrics)


def leaderboard(
    predictors: Sequence[Predictor], gameweeks: pl.DataFrame, seasons: Sequence[str]
) -> pl.DataFrame:
    rows = []
    for predictor in predictors:
        started = time.perf_counter()
        result = run_backtest(predictor, gameweeks, seasons)
        metrics = evaluate(result.predictions, gameweeks)
        metrics["seconds"] = round(time.perf_counter() - started, 2)
        if os.getenv("MLFLOW_TRACKING_URI"):
            _log_to_mlflow(predictor.name, seasons, metrics)
        rows.append({"predictor": predictor.name, **metrics})
    return pl.DataFrame(rows).sort(PRIMARY_METRIC)


SUMMARY_COLUMNS = ("predictor", PRIMARY_METRIC, "mae", "captain_points", "top11_points", "spearman")


def to_markdown(board: pl.DataFrame, columns: Sequence[str] = SUMMARY_COLUMNS) -> str:
    """Leaderboard as a Markdown table (floats to 3 decimals)."""

    def cell(value: object) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += [
        "| " + " | ".join(cell(row[c]) for c in columns) + " |"
        for row in board.select(columns).iter_rows(named=True)
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=list(EVALUATION_SEASONS))
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    args = parser.parse_args()
    gameweeks = player_gameweeks(load_player_matches(args.warehouse))
    board = leaderboard(BASELINES, gameweeks, args.seasons)
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(board.drop("rows", "gameweeks"))


if __name__ == "__main__":
    main()
