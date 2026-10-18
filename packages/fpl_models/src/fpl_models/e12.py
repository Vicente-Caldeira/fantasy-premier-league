"""E12: is point-in-time availability worth more than the live tool's multiplier?

The live tool multiplies every expected event by FPL's `chance_of_playing_next_round`
(and zeroes status `u`) after the model. That adjustment was never backtested, because
availability had no history. vaastav's git history now provides it (`availability`):

    raw         v2 walk-forward predictions (E8's cached rows, fixed data)
    posthoc     raw x the live multiplier, from the snapshot before each deadline

The model variants (availability as features, round 8) are scored the same way by
`experiments`.

    uv run python -m fpl_models.e12
"""

from pathlib import Path

import polars as pl

from fpl_models import availability, experiments, player_model
from fpl_models.backtest import run_backtest
from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks

E8_ROWS = Path("data/experiments/e8/rows.parquet")
KEY = ["season", "gameweek", "player_code"]


def raw_predictions(rows: pl.DataFrame) -> pl.DataFrame:
    """Per player-gameweek predicted and actual points from E8's fixture rows."""
    return rows.group_by(KEY).agg(
        pl.col("xp").sum().alias("predicted"), pl.col("total_points").sum().alias("points")
    )


def multiplier(table: pl.DataFrame) -> pl.DataFrame:
    """The live tool's availability factor (`player_model.availability`) per snapshot."""
    chance = pl.col("chance_of_playing_next_round") / 100
    return table.select(
        *KEY,
        pl.when(pl.col("status") == "u")
        .then(0.0)
        .when(chance.is_not_null())
        .then(chance)
        .otherwise(1.0)
        .alias("factor"),
    )


SIM_FACTORS = Path("data/experiments/e12/sim_factors.parquet")  # read by `e1 --availability`


def sim_factors(table: pl.DataFrame) -> pl.DataFrame:
    """season, gameweek, player_code, this_week, later: the live multiplier for the
    season simulator (players who left the club count for nothing all horizon)."""
    return (
        multiplier(table)
        .join(table.select(*KEY, "status"), on=KEY, how="left")
        .select(
            *KEY,
            pl.col("factor").alias("this_week"),
            pl.when(pl.col("status") == "u").then(0.0).otherwise(1.0).alias("later"),
        )
    )


def by_gameweek(factors: pl.DataFrame, season: str) -> dict[int, pl.DataFrame]:
    """gameweek -> (player_code, this_week, later) for one season, as `play` takes it."""
    season_rows = factors.filter(pl.col("season") == season)
    return {
        gameweek: frame.select("player_code", "this_week", "later")
        for (gameweek,), frame in season_rows.group_by("gameweek")
    }


def posthoc(predictions: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    joined = predictions.join(multiplier(table), on=KEY, how="left")
    return joined.with_columns(pl.col("predicted") * pl.col("factor").fill_null(1.0)).drop("factor")


OUT = Path("data/experiments/e12")


def run_variant(name: str) -> None:
    """Backtest one round-8 config on the tuning seasons; keep row-level predictions so
    it can be scored raw and with the live multiplier on top."""
    gameweeks = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    features = experiments.load_features(DEFAULT_WAREHOUSE)
    model = player_model.ComponentModel(features, config=experiments.EXPERIMENTS[name], name=name)
    predictions = run_backtest(model, gameweeks, experiments.TUNING_SEASONS).predictions
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.select(*KEY, "predicted", "points").write_parquet(OUT / f"{name}_rows.parquet")
    print(f"{name}: done", flush=True)


def score() -> None:
    """Every variant raw and x multiplier, against production raw and production x multiplier."""
    gameweeks = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    deadlines = gameweeks.select("season", "gameweek", "deadline_utc").unique()
    table = availability.features(availability.align(availability.load("data"), deadlines))
    base_rows = pl.read_parquet(OUT / "r8_base_rows.parquet")
    raw = experiments.per_gameweek(base_rows, gameweeks)
    post = experiments.per_gameweek(posthoc(base_rows, table), gameweeks)
    rows = [{"variant": "r8_base x multiplier", "vs": "raw", **experiments.compare(post, raw)}]
    for file in sorted(OUT.glob("r8_*_rows.parquet")):
        name = file.name.removesuffix("_rows.parquet")
        if name == "r8_base":
            continue
        predictions = pl.read_parquet(file)
        variants = {name: predictions, f"{name} x multiplier": posthoc(predictions, table)}
        for label, frame in variants.items():
            per_gw = experiments.per_gameweek(frame, gameweeks)
            for against, base in (("raw", raw), ("posthoc", post)):
                rows.append({"variant": label, "vs": against, **experiments.compare(per_gw, base)})
    with pl.Config(tbl_rows=30, tbl_cols=-1, tbl_width_chars=250, float_precision=4):
        print(pl.DataFrame(rows))


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="backtest one round-8 config")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.variant:
        run_variant(args.variant)
        return
    if args.score:
        score()
        return
    gameweeks = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    deadlines = gameweeks.select("season", "gameweek", "deadline_utc").unique()
    table = availability.features(availability.align(availability.load("data"), deadlines))
    OUT.mkdir(parents=True, exist_ok=True)
    sim_factors(table).write_parquet(SIM_FACTORS)
    raw = raw_predictions(pl.read_parquet(E8_ROWS))
    raw.write_parquet(OUT / "raw_rows.parquet")
    adjusted = posthoc(raw, table)
    flagged = adjusted.join(raw, on=KEY, suffix="_raw").filter(
        pl.col("predicted") != pl.col("predicted_raw")
    )
    print(f"{flagged.height} of {raw.height} player-gameweeks changed by the multiplier")
    base = experiments.per_gameweek(raw, gameweeks)
    cand = experiments.per_gameweek(adjusted, gameweeks)
    with pl.Config(float_precision=4):
        print("posthoc vs raw:", experiments.compare(cand, base))
    base.write_parquet(OUT / "raw.parquet")
    cand.write_parquet(OUT / "posthoc.parquet")


if __name__ == "__main__":
    main()
