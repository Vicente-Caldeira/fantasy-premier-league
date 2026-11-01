"""E19: Kumo Tabular alone, and blended with production, in the walk-forward backtest.

    uv run python -m fpl_models.e19              # tuning seasons
    uv run python -m fpl_models.e19 --held-out   # once, for the accepted blend

Kumo's predictions and production's (`r4_xg_1`) come from the same `run_backtest`
harness. Blends mix the two per player-gameweek, and every result is a paired bootstrap
vs production. The season simulation is `fpl_models.e1` with the `v4kumo50` config.
"""

import argparse
from pathlib import Path

import polars as pl

from fpl_models import experiments, kumo, player_model
from fpl_models.backtest import Predictor, run_backtest
from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks
from fpl_models.leaderboard import EVALUATION_SEASONS

KEYS = ["season", "gameweek", "player_code"]


def predictions(
    name: str, predictor: Predictor, gameweeks: pl.DataFrame, seasons: tuple[str, ...], out: Path
) -> pl.DataFrame:
    """Run (or reuse) one predictor's walk-forward predictions."""
    path = out / f"{name}_predictions.parquet"
    if path.exists():
        return pl.read_parquet(path)
    result = run_backtest(predictor, gameweeks, seasons).predictions
    result.write_parquet(path)
    return result


def calibration(label: str, frame: pl.DataFrame) -> str:
    """Mean predicted vs actual: overall, £10m+ and each gameweek's top 20 predicted."""
    ranked = frame.with_columns(
        rank=pl.col("predicted").rank("ordinal", descending=True).over("season", "gameweek")
    )
    premium, top = frame.filter(pl.col("price") >= 100), ranked.filter(pl.col("rank") <= 20)
    means = [
        part.select(pl.col("predicted").mean(), pl.col("points").mean()).row(0)
        for part in (frame, premium, top)
    ]
    (p, a), (pp, pa), (tp, ta) = means
    return (
        f"{label:<22} mean {p:.3f} vs {a:.3f} | £10m+ {pp:.2f} vs {pa:.2f} | "
        f"top 20 {tp:.2f} vs {ta:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out", action="store_true", help="final, one-off evaluation")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--size", default=kumo.PRODUCTION.size)
    parser.add_argument("--context-rows", type=int, default=kumo.PRODUCTION.context_rows)
    parser.add_argument("--context-stride", type=int, default=kumo.PRODUCTION.context_stride)
    parser.add_argument("--seed", type=int, default=kumo.PRODUCTION.seed)
    args = parser.parse_args()
    seasons = EVALUATION_SEASONS if args.held_out else experiments.TUNING_SEASONS
    results = experiments.RESULTS_DIR / ("held_out" if args.held_out else "")
    out = results / "e19"
    out.mkdir(parents=True, exist_ok=True)

    features = experiments.load_features(DEFAULT_WAREHOUSE)
    gameweeks = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    config = kumo.KumoConfig(
        size=args.size,
        context_rows=args.context_rows,
        context_stride=args.context_stride,
        seed=args.seed,
    )
    production = player_model.ComponentModel(
        features, config=player_model.PRODUCTION, name="r4_xg_1"
    )
    base = predictions("r4_xg_1", production, gameweeks, seasons, out)
    stride = f"x{config.context_stride}" if config.context_stride > 1 else ""
    tag = f"kumo_{config.size}_{config.context_rows // 1000}k{stride}_s{config.seed}"
    alone = predictions(tag, kumo.KumoModel(features, config, name=tag), gameweeks, seasons, out)
    base_per = experiments.per_gameweek(base, gameweeks)

    joined = base.join(alone.select(*KEYS, pl.col("predicted").alias("kumo")), on=KEYS)
    if joined.height != base.height:
        raise ValueError("Kumo and production predicted different player-gameweeks")
    print(f"corr(production, Kumo) = {joined.select(pl.corr('predicted', 'kumo')).item():.3f}")
    rows = []
    print(calibration("production", base))
    for weight in (1.0, *args.weights):
        name = f"{tag}_w{int(weight * 100)}"
        blended = joined.with_columns(
            predicted=kumo.blend(joined["predicted"], joined["kumo"], weight)
        ).drop("kumo")
        per = experiments.per_gameweek(blended, gameweeks)
        per.write_parquet(out / f"{name}.parquet")
        print(calibration(name, blended))
        rows.append({"experiment": name, **experiments.compare(per, base_per)})
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
