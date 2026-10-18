"""E5: walk-forward isotonic calibration of v2 predictions, by position.

On the tuning seasons, v2's high predictions run ~5% hot for DEF, GKP and FWD. An
isotonic map (monotone, predicted -> actual points) is refitted every REFIT gameweeks
on all predictions whose gameweek was played before the deadline (2020-21 is warm-up
history only) and applied to the next block. Variants:

    iso_position   one map per position
    iso_all        one map for everyone
    half_position  halfway between raw and the per-position map (shrinkage)

Scored like every other experiment (paired bootstrap over gameweeks vs raw v2).

    uv run python -m fpl_models.e5_predictions   # once: data/experiments/e5/predictions.parquet
    uv run python -m fpl_models.e5
"""

import polars as pl
from sklearn.isotonic import IsotonicRegression

from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks
from fpl_models.e5_predictions import OUT as PREDICTIONS
from fpl_models.experiments import TUNING_SEASONS, compare, per_gameweek

REFIT = 4


def _fit(history: pl.DataFrame) -> IsotonicRegression:
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(history["predicted"].to_numpy(), history["points"].to_numpy())
    return model


def calibrate(predictions: pl.DataFrame, by_position: bool, shrink: float = 1.0) -> pl.DataFrame:
    """Calibrated `predicted` for the tuning seasons, from earlier rows only."""
    target = predictions.filter(pl.col("season").is_in(TUNING_SEASONS))
    deadlines = target.select("deadline_utc").unique().sort("deadline_utc")["deadline_utc"]
    out = []
    for start in range(0, len(deadlines), REFIT):
        block = deadlines[start : start + REFIT]
        history = predictions.filter(pl.col("deadline_utc") < block[0])
        rows = target.filter(pl.col("deadline_utc").is_in(block.implode()))
        groups = rows.partition_by("position") if by_position else [rows]
        for group in groups:
            past = (
                history.filter(pl.col("position") == group["position"][0])
                if by_position
                else history
            )
            mapped = _fit(past).predict(group["predicted"].to_numpy())
            raw = group["predicted"].to_numpy()
            out.append(group.with_columns(predicted=pl.Series(raw + shrink * (mapped - raw))))
    return pl.concat(out)


def main() -> None:
    predictions = pl.read_parquet(PREDICTIONS)
    truth = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    raw = predictions.filter(pl.col("season").is_in(TUNING_SEASONS))
    base = per_gameweek(raw, truth)
    variants = {
        "iso_position": calibrate(predictions, by_position=True),
        "iso_all": calibrate(predictions, by_position=False),
        "half_position": calibrate(predictions, by_position=True, shrink=0.5),
    }
    rows = [{"variant": n, **compare(per_gameweek(v, truth), base)} for n, v in variants.items()]
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, float_precision=4):
        print(pl.DataFrame(rows))
        # Does it fix the top end? Predictions of 4.5+ by position, raw vs per-position map.
        for name, frame in (("raw", raw), ("iso_position", variants["iso_position"])):
            print(name)
            print(
                frame.filter(pl.col("predicted") >= 4.5)
                .group_by("position")
                .agg(pl.len(), pl.col("predicted").mean(), pl.col("points").mean())
                .sort("position")
            )


if __name__ == "__main__":
    main()
