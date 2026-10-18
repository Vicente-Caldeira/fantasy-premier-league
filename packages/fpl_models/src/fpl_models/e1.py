"""E1: many season paths per setting, to measure how noisy season simulation is.

One deterministic path per season swings by about 120 points, which hides real
differences. Here each config gets PATHS paths per tuning season. Each path starts from
a different reasonable squad (seeded noise on expected points at the first deadline
only) and then plays the season normally. Paths with the same seed are paired across
configs, because they start from the same noise draws.

    uv run python -m fpl_models.e1 prepare   # predictions per config x season, cached
    uv run python -m fpl_models.e1 play      # PATHS paths each, in parallel
    uv run python -m fpl_models.e1 analyse
"""

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from fpl_models import experiments, optimizer, player_features, player_model, team_strength
from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks
from fpl_models.season_sim import SeasonSimulator, Week, play

OUT = Path("data/experiments/e1")
PATHS = 20
CONFIGS = {"v1": experiments.EXPERIMENTS["baseline"], "v2": player_model.PRODUCTION}
SETTINGS = optimizer.Settings(time_limit_seconds=30)
DETECT = 50  # points of season-total difference we want to be able to detect


def weeks_path(config: str, season: str) -> Path:
    return OUT / f"weeks_{config}_{season}.pkl"


def _prepare(job: tuple[str, str]) -> str:
    config, season = job
    matches = team_strength.load_matches(DEFAULT_WAREHOUSE)
    features = player_features.build_features(
        player_features.load_fixture_rows(DEFAULT_WAREHOUSE),
        player_features.match_lambdas(matches),
        player_features.schedule_features(matches),
    )
    truth = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    sim = SeasonSimulator(features, truth, matches, CONFIGS[config], settings=SETTINGS)
    weeks = sim.prepare(season)
    weeks_path(config, season).write_bytes(pickle.dumps(weeks))
    return f"prepared {config} {season}: {len(weeks)} gameweeks"


def _play(job: tuple[str, str, int]) -> str:
    config, season, seed = job
    cached = weeks_path(config, season).read_bytes()
    weeks: list[Week] = pickle.loads(cached)  # noqa: S301 - our own cache file
    result, records = play(season, weeks, SETTINGS, seed=seed)
    path = OUT / "paths" / f"{config}_{season}_{seed:02d}.json"
    path.write_text(json.dumps([asdict(r) for r in records]))
    return f"{config} {season} seed {seed}: {result.total} points, {sum(result.hits)} hits"


def totals() -> pl.DataFrame:
    rows = []
    for file in sorted((OUT / "paths").glob("*.json")):
        config, season, seed = file.stem.rsplit("_", 2)
        records = json.loads(file.read_text())
        rows.append({
            "config": config, "season": season, "seed": int(seed),
            "points": sum(r["points"] for r in records), "hits": sum(r["hits"] for r in records),
        })  # fmt: skip
    return pl.DataFrame(rows)


def analyse() -> None:
    t = totals()
    with pl.Config(tbl_rows=20, float_precision=1):
        print(
            t.group_by("config", "season")
            .agg(
                pl.len().alias("paths"),
                pl.col("points").mean().alias("mean"),
                pl.col("points").std().alias("sd"),
                pl.col("points").min().alias("min"),
                pl.col("points").max().alias("max"),
                pl.col("hits").mean().alias("hits"),
            )
            .sort("season", "config")
        )
    wide = t.pivot(on="config", index=["season", "seed"], values="points").drop_nulls()
    diff = (wide["v2"] - wide["v1"]).to_numpy()
    rng = np.random.default_rng(0)
    boot = rng.choice(diff, size=(10_000, diff.size)).mean(axis=1)
    spread = t.group_by("config", "season").agg(pl.col("points").std().alias("sd"))
    sd_unpaired = float(np.mean(spread["sd"].to_numpy()))
    print(f"\nv2 - v1 per path (both seasons, paired by seed, n={diff.size}):")
    low, high = np.quantile(boot, [0.025, 0.975])
    sd_diff = diff.std(ddof=1)
    print(f"  mean {diff.mean():+.1f}, 95% CI ({low:+.1f}, {high:+.1f}), sd {sd_diff:.1f}")
    z = 1.96 + 0.84  # 5% two-sided, 80% power
    print(f"Paths per setting to detect {DETECT} points (80% power): "
          f"paired {int(np.ceil((z * sd_diff / DETECT) ** 2))}, "
          f"unpaired {int(np.ceil(2 * (z * sd_unpaired / DETECT) ** 2))}")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "play", "analyse"])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    (OUT / "paths").mkdir(parents=True, exist_ok=True)
    seasons = experiments.TUNING_SEASONS
    if args.stage == "prepare":
        jobs = [(c, s) for c in CONFIGS for s in seasons]
        with ProcessPoolExecutor(len(jobs)) as pool:
            for line in pool.map(_prepare, jobs):
                print(line, flush=True)
    elif args.stage == "play":
        plays = [(c, s, seed) for seed in range(PATHS) for c in CONFIGS for s in seasons]
        with ProcessPoolExecutor(args.workers) as pool:
            for line in pool.map(_play, plays):
                print(line, flush=True)
    else:
        analyse()


if __name__ == "__main__":
    main()
