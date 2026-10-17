"""Walk-forward backtesting that cannot leak the future.

For every gameweek of the evaluation seasons, in deadline order:
  1. `history` = every player-gameweek whose deadline is strictly earlier, with outcomes;
  2. `targets` = that gameweek's players with CONTEXT columns only (outcomes removed);
  3. the model predicts expected points for each target row.

Leakage is prevented structurally, not by convention: a model never receives a frame
that contains the outcomes it is asked to predict.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import polars as pl

from fpl_models.data import CONTEXT_COLUMNS, OUTCOME_COLUMNS


class Predictor(Protocol):
    @property
    def name(self) -> str: ...

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.Series:
        """Expected points for each row of `targets`, in the same order."""
        ...


class LeakageError(AssertionError):
    """History contained a row at or after the deadline being predicted."""


@dataclass(frozen=True)
class Backtest:
    predictor: str
    predictions: pl.DataFrame  # context + outcomes + `predicted`


def run_backtest(predictor: Predictor, gameweeks: pl.DataFrame, seasons: Sequence[str]) -> Backtest:
    deadlines = (
        gameweeks.filter(pl.col("season").is_in(list(seasons)))
        .select("season", "gameweek", "deadline_utc")
        .unique()
        .sort("deadline_utc")
    )
    frames = []
    for season, gameweek, deadline in deadlines.iter_rows():
        history = gameweeks.filter(pl.col("deadline_utc") < deadline)
        if history.height and history["deadline_utc"].max() >= deadline:  # pragma: no cover
            raise LeakageError(f"{season} GW{gameweek}: history reaches the deadline")
        actual = gameweeks.filter((pl.col("season") == season) & (pl.col("gameweek") == gameweek))
        targets = actual.select(CONTEXT_COLUMNS)
        predicted = predictor.predict(history, targets)
        if predicted.len() != targets.height:
            raise ValueError(
                f"{predictor.name}: {predicted.len()} predictions for {targets.height}"
            )
        frames.append(
            actual.select(*CONTEXT_COLUMNS, *OUTCOME_COLUMNS).with_columns(
                predicted.cast(pl.Float64).fill_null(0.0).alias("predicted")
            )
        )
    return Backtest(predictor.name, pl.concat(frames))
