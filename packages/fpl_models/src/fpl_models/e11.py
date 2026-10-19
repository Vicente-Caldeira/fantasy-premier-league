"""E11: does congestion across all competitions (cups, Europe) improve predictions?

Cup and European fixtures exist only from 2025-26 (FPL-Elo-Insights), so the tuning
seasons (2021-23) can't test them. The primary window is the current season, 2026-27
(gameweeks played so far); 2025-26 is reported separately and clearly labelled, because
it is one of the held-out seasons.

For each config: walk-forward fixture predictions, then
    points    regulars MAE, captain and top-11 points (paired bootstrap per gameweek)
    minutes   log loss of the minutes class (0 / 1-59 / 60+), overall and for premium
              players (>= 7.5m) at clubs playing in Europe, where rotation shows

    uv run python -m fpl_models.e11 [--seasons 2026-27 2025-26] [--configs ...]
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from fpl_models import experiments, player_model
from fpl_models.data import (
    CONTEXT_COLUMNS,
    DEFAULT_WAREHOUSE,
    load_player_matches,
    player_gameweeks,
)

OUT = Path("data/experiments/e11")
BASELINE = "r4_xg_1"
CANDIDATE = "r7_congestion"
PREMIUM = 75  # tenths


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
            rows.select("season", "gameweek", "player_code", "team_key", "price", "minutes",
                        "team_days_to_europe", "team_days_since_europe")
            .with_columns(
                p_play=events["p_play"], p_60=events["p_60"], p_1_59=events["p_1_59"],
                xp=player_model.expected_points(rows, events),
            )
        )  # fmt: skip
    return pl.concat(frames)


def minutes_log_loss(rows: pl.DataFrame) -> float:
    minutes = rows["minutes"].fill_null(0).to_numpy()
    p = np.where(
        minutes == 0,
        1 - rows["p_play"].to_numpy(),
        np.where(minutes >= 60, rows["p_60"].to_numpy(), rows["p_1_59"].to_numpy()),
    )
    return float(-np.mean(np.log(np.clip(p, 1e-6, 1))))


def report(season: str, results: dict[str, pl.DataFrame], gameweeks: pl.DataFrame) -> None:
    def per_gw(rows: pl.DataFrame) -> pl.DataFrame:
        predicted = rows.group_by("season", "gameweek", "player_code").agg(
            pl.col("xp").sum().alias("predicted")
        )
        truth = gameweeks.filter(pl.col("season") == season).select(
            "season", "gameweek", "player_code", "points"
        )
        return experiments.per_gameweek(
            truth.join(
                predicted, on=["season", "gameweek", "player_code"], how="left"
            ).with_columns(pl.col("predicted").fill_null(0.0)),
            gameweeks,
        )

    base, cand = results[BASELINE], results[CANDIDATE]
    diff = experiments.compare(per_gw(cand), per_gw(base))
    print(f"\n{season}: {CANDIDATE} - {BASELINE} ({base['gameweek'].n_unique()} gameweeks)")
    for metric in ("regulars_mae", "captain_points", "top11_points"):
        d, lo, hi = (diff[f"{metric}_{k}"] for k in ("delta", "ci_low", "ci_high"))
        print(f"  {metric:<15} {d:+.4f} (95% CI {lo:+.4f}, {hi:+.4f})")
    europe = (pl.col("team_days_to_europe") <= 4) | (pl.col("team_days_since_europe") <= 4)
    segments = {
        "all fixtures": pl.lit(value=True),
        "premium, Europe within 4 days": europe.fill_null(value=False)
        & (pl.col("price") >= PREMIUM),
        "premium, no Europe within 4 days": ~europe.fill_null(value=False)
        & (pl.col("price") >= PREMIUM),
    }
    for label, mask in segments.items():
        b, c = base.filter(mask), cand.filter(mask)
        print(f"  minutes log loss, {label:<34} n={b.height:6d}  "
              f"{minutes_log_loss(b):.4f} -> {minutes_log_loss(c):.4f}")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=["2026-27", "2025-26"])
    parser.add_argument("--config", choices=[BASELINE, CANDIDATE], help="record one config only")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gameweeks = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    names = [args.config] if args.config else [BASELINE, CANDIDATE]
    features = None
    for season in args.seasons:
        results = {}
        for name in [BASELINE, CANDIDATE]:
            cache = OUT / f"{name}_{season}.parquet"
            if name in names and not cache.exists():
                if features is None:
                    features = experiments.load_features(DEFAULT_WAREHOUSE)
                record(name, features, gameweeks, season).write_parquet(cache)
            if cache.exists():
                results[name] = pl.read_parquet(cache)
        if len(results) == 2:
            report(season, results, gameweeks)


if __name__ == "__main__":
    main()
