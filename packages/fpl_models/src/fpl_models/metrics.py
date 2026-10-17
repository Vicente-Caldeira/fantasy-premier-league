"""Evaluation metrics: accuracy, and the decisions an FPL manager actually makes.

Accuracy is reported on every player and on *regulars*: players who played in any of
their previous three gameweeks. That is decided from pre-deadline information only,
so the subset never peeks at the outcome. Fringe players who never play make error
metrics look better than they are.
"""

import polars as pl

TOP_N = 11


def regulars(predictions: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    """Rows whose player had minutes in any of his previous three gameweeks."""
    played = (
        history.sort("deadline_utc")
        .with_columns(
            pl.col("minutes")
            .gt(0)
            .cast(pl.Int8)
            .rolling_sum(window_size=3, min_samples=1)
            .shift(1)
            .over("player_code")
            .alias("recent_appearances")
        )
        .select("season", "gameweek", "player_code", "recent_appearances")
    )
    return predictions.join(played, on=["season", "gameweek", "player_code"], how="left").filter(
        pl.col("recent_appearances").fill_null(0) > 0
    )


def _errors(frame: pl.DataFrame, prefix: str) -> dict[str, float]:
    """MAE, RMSE and bias; NaN when the subset is empty rather than a misleading 0."""
    if frame.height == 0:
        return {f"{prefix}{m}": float("nan") for m in ("mae", "rmse", "bias")}
    error = pl.col("predicted") - pl.col("points")
    row = frame.select(
        mae=error.abs().mean(), rmse=(error**2).mean().sqrt(), bias=error.mean()
    ).row(0, named=True)
    return {f"{prefix}{name}": float(value) for name, value in row.items()}


def evaluate(predictions: pl.DataFrame, gameweeks: pl.DataFrame) -> dict[str, float]:
    decisions = predictions.group_by("season", "gameweek").agg(
        pl.col("points").sort_by("predicted", descending=True).first().alias("captain_points"),
        pl.col("points").max().alias("captain_points_oracle"),
        pl.col("points")
        .sort_by("predicted", descending=True)
        .head(TOP_N)
        .sum()
        .alias("top11_points"),
        pl.col("points").sort(descending=True).head(TOP_N).sum().alias("top11_points_oracle"),
        pl.corr("predicted", "points", method="spearman").alias("spearman"),
    )
    means = decisions.select(pl.exclude("season", "gameweek").mean()).row(0, named=True)
    return {
        **_errors(predictions, ""),
        **_errors(regulars(predictions, gameweeks), "regulars_"),
        **{name: float(value) for name, value in means.items()},
        "gameweeks": float(decisions.height),
        "rows": float(predictions.height),
    }
