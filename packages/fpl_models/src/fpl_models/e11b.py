"""E11b: does a player's form and load across ALL competitions improve predictions?

Cup and European player stats exist only from 2025-26 (FPL-Elo-Insights), so, as in
E11, the primary window is the current season, 2026-27 (gameweeks played so far), and
2025-26 is reported separately and clearly labelled, because it is a held-out season.

Configs (see `experiments`, round 11) against production (`r4_xg_1`):
    r11_fatigue      minutes load and share over 7/14/28 days and season, last-match
                     minutes, days since the last appearance
    r11_form         all-competition per-90 xG, xA, shots, box touches, chances created
                     over 14/28 days, season, and EWM with 4- and 15-match half-lives
    r11_both_short   both, short horizons only (<= 14 days, 4-match half-life)
    r11_both_all     both, all horizons (the ablation pair: do long windows add?)

Reported per season: regulars MAE, captain and top-11 points (paired bootstrap per
gameweek), minutes-class log loss and absolute points error for all fixtures and for
players who played a cup or European match in the 6 days before the fixture.

    uv run python -m fpl_models.e11b [--seasons ...] [--configs ...]
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from fpl_models import e11, experiments, player_model
from fpl_models.data import (
    CONTEXT_COLUMNS,
    DEFAULT_WAREHOUSE,
    load_player_matches,
    player_gameweeks,
)

OUT = Path("data/experiments/e11b")
BASELINE = "r4_xg_1"
CANDIDATES = ("r11_fatigue", "r11_form", "r11_both_short", "r11_both_all")


def record(name: str, features: pl.DataFrame, gameweeks: pl.DataFrame, season: str) -> pl.DataFrame:
    """Walk-forward fixture rows (minutes probabilities, expected points) for one season."""
    model = player_model.ComponentModel(features, config=experiments.EXPERIMENTS[name], name=name)
    deadlines = (
        gameweeks.filter(pl.col("season") == season)
        .select("season", "gameweek", "deadline_utc")
        .unique()
        .sort("deadline_utc")
    )
    frames = []
    for _, gameweek, deadline in deadlines.iter_rows():
        history = gameweeks.filter(pl.col("deadline_utc") < deadline)
        targets = gameweeks.filter((pl.col("season") == season) & (pl.col("gameweek") == gameweek))
        rows, events = model.fixture_events(history, targets.select(CONTEXT_COLUMNS))
        frames.append(
            rows.select("season", "gameweek", "player_code", "team_key", "price", "minutes")
            .with_columns(
                p_play=events["p_play"], p_60=events["p_60"], p_1_59=events["p_1_59"],
                xp=player_model.expected_points(rows, events),
            )
        )  # fmt: skip
    return pl.concat(frames)


def midweek_flags(features: pl.DataFrame) -> pl.DataFrame:
    """(season, gameweek, player_code) -> played a cup match in the 6 days before."""
    return features.group_by("season", "gameweek", "player_code").agg(
        (pl.col("eval_cup_minutes_prev_6d").fill_null(0) > 0).any().alias("midweek")
    )


def points_error(rows: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Absolute error of each player's gameweek expected points."""
    predicted = rows.group_by("season", "gameweek", "player_code").agg(pl.col("xp").sum())
    return predicted.join(truth, on=["season", "gameweek", "player_code"]).with_columns(
        error=(pl.col("xp") - pl.col("points")).abs()
    )


def report(
    season: str, results: dict[str, pl.DataFrame], gameweeks: pl.DataFrame, flags: pl.DataFrame
) -> None:
    truth = gameweeks.filter(pl.col("season") == season).select(
        "season", "gameweek", "player_code", "points"
    )

    def per_gw(rows: pl.DataFrame) -> pl.DataFrame:
        predicted = rows.group_by("season", "gameweek", "player_code").agg(
            pl.col("xp").sum().alias("predicted")
        )
        return experiments.per_gameweek(
            truth.join(predicted, on=["season", "gameweek", "player_code"], how="left")
            .with_columns(pl.col("predicted").fill_null(0.0)),
            gameweeks,
        )  # fmt: skip

    base = results[BASELINE].join(flags, on=["season", "gameweek", "player_code"], how="left")
    base_err = points_error(results[BASELINE], truth).join(
        flags, on=["season", "gameweek", "player_code"], how="left"
    )
    n_gws = base["gameweek"].n_unique()
    n_mid = int(base["midweek"].fill_null(value=False).sum())
    print(f"\n{season}: {n_gws} gameweeks, {base.height} fixture rows, {n_mid} after a cup game")
    for name in CANDIDATES:
        if name not in results:
            continue
        cand = results[name].join(flags, on=["season", "gameweek", "player_code"], how="left")
        diff = experiments.compare(per_gw(results[name]), per_gw(results[BASELINE]))
        print(f"  {name} - {BASELINE}")
        for metric in ("regulars_mae", "captain_points", "top11_points"):
            d, lo, hi = (diff[f"{metric}_{k}"] for k in ("delta", "ci_low", "ci_high"))
            print(f"    {metric:<15} {d:+.4f} (95% CI {lo:+.4f}, {hi:+.4f})")
        mid = pl.col("midweek").fill_null(value=False)
        for label, mask in (("all fixtures", pl.lit(value=True)), ("after a cup game", mid)):
            b, c = base.filter(mask), cand.filter(mask)
            print(f"    minutes log loss, {label:<17} n={b.height:6d}  "
                  f"{e11.minutes_log_loss(b):.4f} -> {e11.minutes_log_loss(c):.4f}")  # fmt: skip
        cand_err = points_error(results[name], truth).join(
            base_err.select("season", "gameweek", "player_code", "midweek",
                            pl.col("error").alias("base_error")),
            on=["season", "gameweek", "player_code"],
        )  # fmt: skip
        sub = cand_err.filter(pl.col("midweek").fill_null(value=False))
        if sub.height:
            delta = (sub["error"] - sub["base_error"]).to_numpy()
            rng = np.random.default_rng(0)
            boot = rng.choice(delta, size=(5000, delta.size)).mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            print(f"    points abs error after a cup game (n={delta.size}, rows not independent):"
                  f" {delta.mean():+.4f} (95% CI {lo:+.4f}, {hi:+.4f})")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=["2026-27", "2025-26"])
    parser.add_argument("--configs", nargs="+", default=list(CANDIDATES), help="record these")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gameweeks = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    features = experiments.load_features(DEFAULT_WAREHOUSE)
    flags = midweek_flags(features)
    for season in args.seasons:
        results = {}
        for name in [BASELINE, *CANDIDATES]:
            # The baseline is E11's (same data, same features); reuse its cache.
            cache = (e11.OUT if name == BASELINE else OUT) / f"{name}_{season}.parquet"
            if name in args.configs and not args.report_only and not cache.exists():
                record(name, features, gameweeks, season).write_parquet(cache)
            if cache.exists():
                results[name] = pl.read_parquet(cache)
        if BASELINE in results:
            report(season, results, gameweeks, flags)


if __name__ == "__main__":
    main()
