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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fpl_models import experiments, optimizer, player_model, prices, team_strength
from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks
from fpl_models.season_sim import SeasonSimulator, Week, play

OUT = Path("data/experiments/e1")
PRICE_MOVES = Path("data/experiments/e10/moves.parquet")  # written by `fpl_models.prices`
PATHS = 20
# "v2fix": production prepared after the 2022-23 zero-xG fix ("v2" paths predate it).
CONFIGS = {
    "v1": experiments.EXPERIMENTS["baseline"],
    "v2": player_model.PRODUCTION,
    "v2fix": player_model.PRODUCTION,
    "v2pen": experiments.EXPERIMENTS["r5_penalties"],
    "v3blend": experiments.EXPERIMENTS["r12_blend"],  # E17: half structural attack rates
}
SETTINGS = optimizer.Settings(time_limit_seconds=30)
DETECT = 50  # points of season-total difference we want to be able to detect


def weeks_path(config: str, season: str, horizon: int = 3) -> Path:
    suffix = "" if horizon == 3 else f"h{horizon}"
    return OUT / f"weeks_{config}{suffix}_{season}.pkl"


def _prepare(job: tuple[str, str, int]) -> str:
    config, season, horizon = job
    matches = team_strength.load_matches(DEFAULT_WAREHOUSE)
    features = experiments.load_features(DEFAULT_WAREHOUSE)
    truth = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    sim = SeasonSimulator(
        features, truth, matches, CONFIGS[config], horizon=horizon, settings=SETTINGS
    )
    weeks = sim.prepare(season)
    weeks_path(config, season, horizon).write_bytes(pickle.dumps(weeks))
    return f"prepared {config} horizon {horizon} {season}: {len(weeks)} gameweeks"


@dataclass(frozen=True)
class Job:
    config: str
    season: str
    horizon: int
    seed: int
    hit_penalty: float
    label: str
    wildcards: tuple[int, ...] = ()
    wildcard_threshold: float | None = None
    value_weight: float = 0.0  # E10: points per 0.1m of expected selling-value change


def path_file(label: str, season: str, seed: int) -> Path:
    return OUT / "paths" / f"{label}_{season}_{seed:02d}.json"


def _play(job: Job) -> str:
    config, season, horizon, seed, label = job.config, job.season, job.horizon, job.seed, job.label
    cached = weeks_path(config, season, horizon).read_bytes()
    weeks: list[Week] = pickle.loads(cached)  # noqa: S301 - our own cache file
    settings = optimizer.Settings(
        time_limit_seconds=30, hit_penalty=job.hit_penalty, value_weight=job.value_weight
    )
    moves = None
    if job.value_weight:
        moves = prices.by_gameweek(pl.read_parquet(PRICE_MOVES), season)
    result, records = play(
        season, weeks, settings, seed=seed, wildcards=job.wildcards,
        wildcard_threshold=job.wildcard_threshold, price_moves=moves,
    )  # fmt: skip
    path = path_file(label, season, seed)
    path.write_text(json.dumps([asdict(r) for r in records]))
    return f"{label} {season} seed {seed}: {result.total} points, {sum(result.hits)} hits"


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


def compare_labels(labels: list[str], against: str) -> None:
    """Unpaired: difference in mean season total per path, bootstrapped by path."""
    t = totals()
    rng = np.random.default_rng(0)
    base = t.filter(pl.col("config") == against)
    for label in labels:
        other = t.filter(pl.col("config") == label)
        gains, draws = [], np.zeros(10_000)
        for season in sorted(other["season"].unique()):
            a = other.filter(pl.col("season") == season)["points"].to_numpy()
            b = base.filter(pl.col("season") == season)["points"].to_numpy()
            gains.append(a.mean() - b.mean())
            draws += rng.choice(a, (10_000, a.size)).mean(1) - rng.choice(b, (10_000, b.size)).mean(
                1
            )
        n = other.height // max(other["season"].n_unique(), 1)
        low, high = np.quantile(draws / len(gains), [0.025, 0.975])
        hits = float(np.mean(other["hits"].to_numpy()))
        print(f"{label:<14} vs {against}: {np.mean(gains):+.1f} per season (95% CI {low:+.1f}, "
              f"{high:+.1f}); {n} paths per season; hits/season {hits:.1f}")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "play", "analyse", "compare"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--hit-penalty", type=float, default=float(optimizer.HIT_COST))
    parser.add_argument("--seeds", type=int, nargs=2, default=(0, PATHS), metavar=("FROM", "TO"))
    parser.add_argument("--label", help="name for these paths (default: the config)")
    parser.add_argument("--wildcards", type=int, nargs="*", default=[], help="gameweeks")
    parser.add_argument("--wildcard-threshold", type=float)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--value-weight", type=float, default=0.0, help="E10, points per 0.1m")
    parser.add_argument("--labels", nargs="+", default=[])
    parser.add_argument("--against", default="v2")
    parser.add_argument("--seasons", nargs="+", default=list(experiments.TUNING_SEASONS))
    args = parser.parse_args()
    (OUT / "paths").mkdir(parents=True, exist_ok=True)
    seasons = args.seasons
    if args.stage == "prepare":
        jobs = [(c, s, args.horizon) for c in args.configs for s in seasons]
        with ProcessPoolExecutor(len(jobs)) as pool:
            for line in pool.map(_prepare, jobs):
                print(line, flush=True)
    elif args.stage == "play":
        plays = [
            Job(
                c,
                s,
                args.horizon,
                seed,
                args.hit_penalty,
                args.label or c,
                tuple(args.wildcards),
                args.wildcard_threshold,
                args.value_weight,
            )
            for seed in range(*args.seeds)
            for c in args.configs
            for s in seasons
        ]
        if args.skip_existing:  # resume an interrupted run
            plays = [j for j in plays if not path_file(j.label, j.season, j.seed).exists()]
        with ProcessPoolExecutor(args.workers) as pool:
            for line in pool.map(_play, plays):
                print(line, flush=True)
    elif args.stage == "compare":
        compare_labels(args.labels, args.against)
    else:
        analyse()


if __name__ == "__main__":
    main()
