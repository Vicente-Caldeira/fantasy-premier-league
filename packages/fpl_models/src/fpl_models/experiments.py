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

from fpl_models import (
    availability,
    penalties,
    player_features,
    player_model,
    team_strength,
    understat,
)
from fpl_models.backtest import run_backtest
from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks
from fpl_models.leaderboard import EVALUATION_SEASONS
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
    "denoise_targets": replace(BASE, denoise_weight=1.0),
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

# Round 3: everything that helped, then remove one piece at a time (ablation).
TUNED = replace(
    BASE, recency_half_life_days=180, l2_regularization=1.0, min_samples_leaf=100,
    learning_rate=0.03, max_iter=400,
)  # fmt: skip
ALL_IN = replace(
    TUNED, denoise_weight=1.0, reconcile=True,
    features=BASE.features + player_features.CROWD_SHARE + player_features.SCHEDULE_FEATURES,
)  # fmt: skip
EXPERIMENTS |= {
    "all_in": ALL_IN,
    "all_no_denoise": replace(ALL_IN, denoise_weight=0.0),
    "all_no_reconcile": replace(ALL_IN, reconcile=False),
    "all_no_schedule": replace(ALL_IN, features=BASE.features + player_features.CROWD_SHARE),
    "all_no_crowd": replace(ALL_IN, features=BASE.features + player_features.SCHEDULE_FEATURES),
    # Round 4: no reconciliation (it hurt in combination); blend of goals and xG targets.
    "r4_xg_1": replace(ALL_IN, reconcile=False, denoise_weight=1.0),
    "r4_xg_half": replace(ALL_IN, reconcile=False, denoise_weight=0.5),
    "r4_xg_0": replace(ALL_IN, reconcile=False, denoise_weight=0.0),
    "all_untuned": replace(
        ALL_IN,
        recency_half_life_days=None,
        l2_regularization=0.0,
        min_samples_leaf=20,
        learning_rate=0.05,
        max_iter=200,
    ),
}

# Round 5 (E3a): last season's penalty taking, against production (= r4_xg_1).
EXPERIMENTS |= {
    "r5_penalties": replace(
        player_model.PRODUCTION,
        features=player_model.PRODUCTION.features + penalties.PENALTY_FEATURES,
    ),
    "r5_penalty_share": replace(
        player_model.PRODUCTION,
        features=(*player_model.PRODUCTION.features, "penalty_share_prev"),
    ),
}

# Round 6 (E9): E8 found elite attackers' goals heavily under-predicted (Haaland
# 2022-23: 12.3 expected, 36 scored); min_samples_leaf=100 was chosen on MAE, which
# barely sees the tail. Less shrinkage, against production.
EXPERIMENTS |= {
    "r6_leaf30": replace(player_model.PRODUCTION, min_samples_leaf=30),
    "r6_iter800": replace(player_model.PRODUCTION, max_iter=800),
}

# Round 7 (E11): congestion across all competitions (cup data from 2025-26 only, so it
# is evaluated on 2026-27 and, separately labelled, 2025-26; see `fpl_models.e11`).
EXPERIMENTS |= {
    "r7_congestion": replace(
        player_model.PRODUCTION,
        features=player_model.PRODUCTION.features + player_features.CONGESTION_FEATURES,
    ),
}

# Round 8 (E12): point-in-time status/news and set-piece order (vaastav's archive),
# against production and against production x the live availability multiplier.
_PROD = player_model.PRODUCTION.features
EXPERIMENTS |= {
    "r8_base": player_model.PRODUCTION,  # same runner, so the comparison is like for like
    "r8_avail": replace(
        player_model.PRODUCTION, features=_PROD + availability.AVAILABILITY_FEATURES
    ),
    "r8_pens": replace(player_model.PRODUCTION, features=_PROD + availability.SET_PIECE_FEATURES),
    "r8_both": replace(
        player_model.PRODUCTION,
        features=_PROD + availability.AVAILABILITY_FEATURES + availability.SET_PIECE_FEATURES,
    ),
}


# Round 9 (E15): predictions are compressed at the top (elite attackers' goals ~10%
# low). Production stops early (~150 of 400 rounds at learning rate 0.03).
EXPERIMENTS |= {
    "r9_full_400": replace(player_model.PRODUCTION, early_stopping=False),
    "r9_full_800": replace(player_model.PRODUCTION, early_stopping=False, max_iter=800),
    "r9_lr10": replace(player_model.PRODUCTION, learning_rate=0.1),
    # E16: per-90 rates given an appearance x expected minutes.
    "r10_conditional": replace(player_model.PRODUCTION, conditional_rates=True),
    "r10_conditional_full": replace(
        player_model.PRODUCTION, conditional_rates=True, early_stopping=False
    ),
    # E17: structural goals/assists (shrunk per-90 rate x fixture x expected minutes).
    "r12_structural": replace(player_model.PRODUCTION, structural_weight=1.0),
    "r12_blend": replace(player_model.PRODUCTION, structural_weight=0.5),
    "r12_blend25": replace(player_model.PRODUCTION, structural_weight=0.25),
    # E18: premium attackers out-score their xG (+14%) and pure-xG targets remove
    # finishing skill; blend actual goals back in (re-test on the fixed xG data).
    "r13_xg_07": replace(player_model.PRODUCTION, denoise_weight=0.7),
    "r13_xg_05": replace(player_model.PRODUCTION, denoise_weight=0.5),
}


# Round 11 (E11b): players' form and load across all competitions (cup stats from
# 2025-26; evaluated like E11, see `fpl_models.e11b`). Ablation: windows up to 14 days
# and a 4-match half-life vs. all windows (adding 28 days, season, 15-match half-life).
EXPERIMENTS |= {
    "r11_fatigue": replace(
        player_model.PRODUCTION,
        features=_PROD + player_features.FATIGUE_SHORT + player_features.FATIGUE_LONG,
    ),
    "r11_form": replace(
        player_model.PRODUCTION,
        features=_PROD + player_features.FORM_SHORT + player_features.FORM_LONG,
    ),
    "r11_both_short": replace(
        player_model.PRODUCTION,
        features=_PROD + player_features.FATIGUE_SHORT + player_features.FORM_SHORT,
    ),
    "r11_both_all": replace(
        player_model.PRODUCTION, features=_PROD + player_features.ALL_COMP_FEATURES
    ),
}


# Round 20 (E20): Understat per-match xG/xA before FPL's (2022-23), for features and
# denoised targets; and Understat shots, key passes, npxG, xGChain, xGBuildup per 90.
_US = player_features.UNDERSTAT_FEATURES
EXPERIMENTS |= {
    "r20_xg_fill": replace(player_model.PRODUCTION, xg_source="fill"),
    "r20_xg_understat": replace(player_model.PRODUCTION, xg_source="understat"),
    "r20_us_feats": replace(player_model.PRODUCTION, features=_PROD + _US),
    "r20_fill_us_feats": replace(player_model.PRODUCTION, xg_source="fill", features=_PROD + _US),
    "r20_understat_us_feats": replace(
        player_model.PRODUCTION, xg_source="understat", features=_PROD + _US
    ),
}


# Rounds 21-22 (E21, E22): data that only exists from 2024-25, so they use the late split
# (`LATE_TUNING`, then `LATE_HELD_OUT` once) instead of the 2021-23 tuning seasons.
LATE_TUNING = ("2024-25",)
LATE_HELD_OUT = ("2025-26",)
EXPERIMENTS |= {
    "r21_base": player_model.PRODUCTION,
    "r22_base": player_model.PRODUCTION,  # E22's runner keeps row-level predictions
    "r21_opta": replace(player_model.PRODUCTION, features=_PROD + player_features.OPTA_FEATURES),
    "r22_avail": replace(
        player_model.PRODUCTION, features=_PROD + availability.AVAILABILITY_FEATURES
    ),
    "r22_avail_fresh": replace(
        player_model.PRODUCTION, features=_PROD + availability.FRESH_FEATURES
    ),
}


def load_features(warehouse: Path, xg_source: str = "fpl") -> pl.DataFrame:
    """Every candidate feature: fixture rows (with Understat's per-match stats and the
    chosen xG source, E20), team context, schedule, penalties and, once
    `fpl_models.availability` has built the snapshots, point-in-time availability."""
    matches = team_strength.load_matches(warehouse)
    rows = understat.with_understat(
        player_features.load_fixture_rows(warehouse), warehouse, xg_source
    )
    features = player_features.build_features(
        rows,
        player_features.match_lambdas(matches),
        player_features.schedule_features(matches),
    )
    cups = player_features.load_cup_matches(warehouse)
    congestion = player_features.congestion_features(matches, cups)
    features = player_features.add_congestion(features, congestion)
    all_comp = player_features.all_competition_features(
        features, player_features.load_cup_player_matches(warehouse), cups
    )
    features = player_features.add_all_competition(features, all_comp)
    features = penalties.add_penalties(features, penalties.season_table(*penalties.load(warehouse)))
    snapshots = Path(os.getenv("FPL_LAKE_URL", "data")) / availability.SNAPSHOT
    if not snapshots.exists():
        return features
    deadlines = features.select("season", "gameweek", "deadline_utc").unique()
    table = availability.features(availability.align(pl.read_parquet(snapshots), deadlines))
    return availability.add_availability(features, table)


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


def run(
    names: list[str],
    warehouse: Path,
    *,
    held_out: bool = False,
    seasons: tuple[str, ...] | None = None,
    results: Path | None = None,
) -> pl.DataFrame:
    """Backtest each config. `held_out` scores EVALUATION_SEASONS: run it once, for the
    final candidate only, after every choice has been made on the tuning seasons.
    `seasons` and `results` override both (E21-E22's late split)."""
    seasons = seasons or (EVALUATION_SEASONS if held_out else TUNING_SEASONS)
    results = results or (RESULTS_DIR / "held_out" if held_out else RESULTS_DIR)
    features: dict[str, pl.DataFrame] = {}  # by xg source
    gameweeks = player_gameweeks(load_player_matches(warehouse))
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        config = EXPERIMENTS[name]
        if config.xg_source not in features:
            features[config.xg_source] = load_features(warehouse, config.xg_source)
        model = player_model.ComponentModel(features[config.xg_source], config=config, name=name)
        predictions = run_backtest(model, gameweeks, seasons).predictions
        per_gameweek(predictions, gameweeks).write_parquet(results / f"{name}.parquet")
        result = {"experiment": name, **evaluate(predictions, gameweeks)}
        if os.getenv("MLFLOW_TRACKING_URI"):
            import mlflow  # noqa: PLC0415 - optional, only when tracking

            mlflow.set_experiment("player_model_held_out" if held_out else "player_model_tuning")
            with mlflow.start_run(run_name=name):
                params = {k: v for k, v in asdict(config).items() if k != "features"}
                mlflow.log_params({**params, "n_features": len(config.features)})
                mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, float)})
        rows.append(result)
        print(f"{name:<16} regulars_mae {result['regulars_mae']:.4f}", flush=True)
    return pl.DataFrame(rows)


def significance(
    names: list[str],
    baseline: str = "baseline",
    *,
    held_out: bool = False,
    results: Path | None = None,
) -> pl.DataFrame:
    """Each experiment vs the baseline, from the saved per-gameweek results."""
    results = results or (RESULTS_DIR / "held_out" if held_out else RESULTS_DIR)
    base = pl.read_parquet(results / f"{baseline}.parquet")
    rows = [
        {"experiment": n, **compare(pl.read_parquet(results / f"{n}.parquet"), base)}
        for n in names
        if n != baseline
    ]
    return pl.DataFrame(rows).sort("regulars_mae_delta")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    parser.add_argument("--compare-only", action="store_true", help="reuse saved results")
    parser.add_argument("--held-out", action="store_true", help="final, one-off evaluation")
    parser.add_argument("--seasons", nargs="+", help="override the seasons (late split)")
    parser.add_argument("--results", type=Path, help="override the results directory")
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()
    seasons = tuple(args.seasons) if args.seasons else None
    if not args.compare_only:
        run(
            args.only, args.warehouse, held_out=args.held_out, seasons=seasons, results=args.results
        )
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, float_precision=4, tbl_rows=50):
        print(significance(args.only, args.baseline, held_out=args.held_out, results=args.results))


if __name__ == "__main__":
    main()
