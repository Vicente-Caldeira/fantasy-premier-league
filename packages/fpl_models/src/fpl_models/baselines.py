"""Simple predictors that every real model must beat.

Each uses only `history` (strictly before the deadline), so all are leakage-free by
construction of the backtest.
"""

from dataclasses import dataclass

import polars as pl


def _lookup(targets: pl.DataFrame, stats: pl.DataFrame, on: str, column: str) -> pl.Series:
    """`stats[column]` for each target row, in target order (predictions are positional)."""
    return targets.join(stats, on=on, how="left", maintain_order="left")[column]


def _by_player(targets: pl.DataFrame, stats: pl.DataFrame, column: str) -> pl.Series:
    return _lookup(targets, stats, "player_code", column).fill_null(0.0)


@dataclass(frozen=True)
class GlobalMean:
    """Average points of all player-gameweeks so far. The floor."""

    name: str = "global_mean"

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.Series:
        mean = history["points"].mean() if history.height else 0.0
        return pl.Series([mean] * targets.height, dtype=pl.Float64)


@dataclass(frozen=True)
class RecentForm:
    """Mean points over the player's last `window` gameweeks (like FPL's "form")."""

    window: int = 4

    @property
    def name(self) -> str:
        return f"form_last_{self.window}"

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.Series:
        recent = (
            history.sort("deadline_utc")
            .group_by("player_code")
            .agg(pl.col("points").tail(self.window).mean().alias("form"))
        )
        return _by_player(targets, recent, "form")


@dataclass(frozen=True)
class SeasonPointsPerGameweek:
    """Season-to-date mean points per gameweek; last season's before the first deadline."""

    name: str = "season_ppg"

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.Series:
        season = targets["season"][0]
        current = history.filter(pl.col("season") == season)
        source = (
            current
            if current.height
            else history.filter(pl.col("season") == history["season"].max())
        )
        ppg = source.group_by("player_code").agg(pl.col("points").mean().alias("ppg"))
        return _by_player(targets, ppg, "ppg")


@dataclass(frozen=True)
class ExponentialForm:
    """Exponentially weighted mean of all past gameweeks, half-life in gameweeks.

    Shrinks towards the positional mean for players with little history.
    """

    half_life: float = 6.0
    prior_weight: float = 2.0

    @property
    def name(self) -> str:
        return f"ewm_hl{self.half_life:g}"

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.Series:
        decay = 0.5 ** (1 / self.half_life)
        weighted = (
            history.with_columns(
                (
                    pl.col("deadline_utc").rank("dense", descending=True).over("player_code") - 1
                ).alias("age")
            )
            .with_columns((decay ** pl.col("age")).alias("w"))
            .group_by("player_code")
            .agg(
                (pl.col("w") * pl.col("points")).sum().alias("wp"),
                pl.col("w").sum().alias("ws"),
            )
        )
        prior = history.group_by("position").agg(pl.col("points").mean().alias("prior"))
        # Prior comes from the target's position this season, not from history rows.
        wp = _lookup(targets, weighted, "player_code", "wp").fill_null(0.0)
        ws = _lookup(targets, weighted, "player_code", "ws").fill_null(0.0)
        p = _lookup(targets, prior, "position", "prior").fill_null(0.0)
        return (wp + self.prior_weight * p) / (ws + self.prior_weight)


BASELINES = (
    GlobalMean(),
    SeasonPointsPerGameweek(),
    RecentForm(window=4),
    ExponentialForm(half_life=6.0),
)
