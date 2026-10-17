"""Player-model experiments, scored on the tuning seasons only.

    uv run python -m fpl_models.experiments                 # every experiment
    uv run python -m fpl_models.experiments --only l2_1 lr03

Settings may only be chosen on TUNING_SEASONS; the evaluation seasons in
`leaderboard.EVALUATION_SEASONS` are scored once per accepted change. Each run is
logged to MLflow (experiment `player_model_tuning`) when MLFLOW_TRACKING_URI is set.
"""

import argparse
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl

from fpl_models import player_features, player_model, team_strength
from fpl_models.backtest import run_backtest
from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks
from fpl_models.metrics import TOP_N, evaluate, regulars

TUNING_SEASONS = ("2021-22", "2022-23")
RESULTS_DIR = Path("data/experiments")
BOOTSTRAP_SAMPLES = 2000
BASE = player_model.ModelConfig()

# Round 1: one change at a time against the current champion.
EXPERIMENTS: dict[str, player_model.ModelConfig] = {
    "baseline": BASE,
    "recency_180d": replace(BASE, recency_half_life_days=180),
    "recency_365d": replace(BASE, recency_half_life_days=365),
    "recency_730d": replace(BASE, recency_half_life_days=730),
    "seasons_2": replace(BASE, train_seasons=2),
    "seasons_6": replace(BASE, train_seasons=6),
    "lr03_iter400": replace(BASE, learning_rate=0.03, max_iter=400),
    "leaves_15": replace(BASE, max_leaf_nodes=15),
    "leaves_63": replace(BASE, max_leaf_nodes=63),
    "min_leaf_100": replace(BASE, min_samples_leaf=100),
    "l2_1": replace(BASE, l2_regularization=1.0),
    "multi_half_life": replace(BASE, features=player_features.ALL_FEATURES),
    # Crowd signal (see player_features): lagged is live-safe; current is skew-prone.
    "crowd_lagged": replace(BASE, features=BASE.features + player_features.CROWD_LAGGED),
    "crowd_current": replace(BASE, features=BASE.features + player_features.CROWD_CURRENT),
    # Round 2 candidates (inputs and targets).
    "schedule": replace(BASE, features=BASE.features + player_features.SCHEDULE_FEATURES),
    # Not "opta": Opta detail starts in 2024-25, so it is empty on the tuning seasons.
    # Evaluating it needs a different split (see docs/model-experiments.md).
    "denoise_targets": replace(BASE, denoise_targets=True),
    "reconcile": replace(BASE, reconcile=True),
    "crowd_share": replace(BASE, features=BASE.features + player_features.CROWD_SHARE),
    # Combination of round 1's significant settings (each improved MAE, CI below 0).
    "tuned": replace(
        BASE,
        recency_half_life_days=180,
        l2_regularization=1.0,
        min_samples_leaf=100,
        learning_rate=0.03,
        max_iter=400,
    ),
    "tuned_crowd_share": replace(
        BASE,
        recency_half_life_days=180,
        l2_regularization=1.0,
        min_samples_leaf=100,
        learning_rate=0.03,
        max_iter=400,
        features=BASE.features + player_features.CROWD_SHARE,
    ),
}


def per_gameweek(predictions: pl.DataFrame, gameweeks: pl.DataFrame) -> pl.DataFrame:
    """Per-gameweek sufficient statistics, so differences can be bootstrapped."""
    errors = (
        regulars(predictions, gameweeks)
        .group_by("season", "gameweek")
        .agg(
            (pl.col("predicted") - pl.col("points")).abs().sum().alias("abs_error"),
            pl.len().alias("regular_rows"),
        )
    )
    decisions = predictions.group_by("season", "gameweek").agg(
        pl.col("points").sort_by("predicted", descending=True).first().alias("captain"),
        pl.col("points").sort_by("predicted", descending=True).head(TOP_N).sum().alias("top11"),
    )
    return errors.join(decisions, on=["season", "gameweek"]).sort("season", "gameweek")


def compare(candidate: pl.DataFrame, baseline: pl.DataFrame, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap over gameweeks: candidate - baseline with a 95% interval.

    For MAE, lower is better: a negative difference whose interval excludes 0 is a
    real improvement.
    """
    joined = candidate.join(baseline, on=["season", "gameweek"], suffix="_base")
    n = joined.height
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))

    def stat(column: str, per: str | None = None) -> np.ndarray:
        a, b = joined[column].to_numpy(), joined[f"{column}_base"].to_numpy()
        if per is None:
            return a[idx].mean(axis=1) - b[idx].mean(axis=1)
        wa, wb = joined[per].to_numpy(), joined[f"{per}_base"].to_numpy()
        return a[idx].sum(axis=1) / wa[idx].sum(axis=1) - b[idx].sum(axis=1) / wb[idx].sum(axis=1)  # type: ignore[no-any-return]

    out = {}
    for name, column, per in (
        ("regulars_mae", "abs_error", "regular_rows"),
        ("captain_points", "captain", None),
        ("top11_points", "top11", None),
    ):
        draws = stat(column, per)
        out[f"{name}_delta"] = float(np.mean(draws))
        out[f"{name}_ci_low"] = float(np.quantile(draws, 0.025))
        out[f"{name}_ci_high"] = float(np.quantile(draws, 0.975))
    return out


def run(names: list[str], warehouse: Path) -> pl.DataFrame:
    matches = team_strength.load_matches(warehouse)
    features = player_features.build_features(
        player_features.load_fixture_rows(warehouse),
        player_features.match_lambdas(matches),
        player_features.schedule_features(matches),
    )
    gameweeks = player_gameweeks(load_player_matches(warehouse))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        config = EXPERIMENTS[name]
        model = player_model.ComponentModel(features, config=config, name=name)
        predictions = run_backtest(model, gameweeks, TUNING_SEASONS).predictions
        per_gameweek(predictions, gameweeks).write_parquet(RESULTS_DIR / f"{name}.parquet")
        result = {"experiment": name, **evaluate(predictions, gameweeks)}
        if os.getenv("MLFLOW_TRACKING_URI"):
            import mlflow  # noqa: PLC0415 - optional, only when tracking

            mlflow.set_experiment("player_model_tuning")
            with mlflow.start_run(run_name=name):
                params = {k: v for k, v in asdict(config).items() if k != "features"}
                mlflow.log_params({**params, "n_features": len(config.features)})
                mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, float)})
        rows.append(result)
        print(f"{name:<16} regulars_mae {result['regulars_mae']:.4f}", flush=True)
    return pl.DataFrame(rows)


def significance(names: list[str], baseline: str = "baseline") -> pl.DataFrame:
    """Each experiment vs the baseline, from the saved per-gameweek results."""
    base = pl.read_parquet(RESULTS_DIR / f"{baseline}.parquet")
    rows = [
        {"experiment": n, **compare(pl.read_parquet(RESULTS_DIR / f"{n}.parquet"), base)}
        for n in names
        if n != baseline
    ]
    return pl.DataFrame(rows).sort("regulars_mae_delta")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    parser.add_argument("--compare-only", action="store_true", help="reuse saved results")
    args = parser.parse_args()
    if not args.compare_only:
        run(args.only, args.warehouse)
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, float_precision=4, tbl_rows=50):
        print(significance(args.only))


if __name__ == "__main__":
    main()
