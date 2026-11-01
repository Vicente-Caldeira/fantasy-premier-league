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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl

from fpl_models import e12, experiments, kumo, optimizer, player_model, prices, team_strength
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
    "v3blend25": experiments.EXPERIMENTS["r12_blend25"],
    "v4kumo50": player_model.PRODUCTION,  # E19: half Kumo Tabular (see KUMO)
    "v5defcon": experiments.EXPERIMENTS["r24_defcon_opta"],  # E24
    "v2now": player_model.PRODUCTION,  # production re-prepared with E23-E24's code and data
    "v6bps": experiments.EXPERIMENTS["r23_bps_bonus"],  # E23
}
KUMO = {"v4kumo50": kumo.KumoConfig(weight=0.5)}
SETTINGS = optimizer.Settings(time_limit_seconds=30)
DETECT = 50  # points of season-total difference we want to be able to detect


def weeks_path(config: str, season: str, horizon: int = 3) -> Path:
    suffix = "" if horizon == 3 else f"h{horizon}"
    return OUT / f"weeks_{config}{suffix}_{season}.pkl"


def _prepare(job: tuple[str, str, int]) -> str:
    config, season, horizon = job
    matches = team_strength.load_matches(DEFAULT_WAREHOUSE)
    cfg = CONFIGS[config]
    features = experiments.load_features(DEFAULT_WAREHOUSE, cfg.xg_source, cfg.defcon_source)
    truth = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    sim = SeasonSimulator(
        features, truth, matches, CONFIGS[config], horizon=horizon, settings=SETTINGS,
        kumo=KUMO.get(config),
    )  # fmt: skip
    weeks = sim.prepare(season)
    weeks_path(config, season, horizon).write_bytes(pickle.dumps(weeks))
    return f"prepared {config} horizon {horizon} {season}: {len(weeks)} gameweeks"


# Replicated protocol. A season total moves by ~20 points from prediction changes far
# smaller than any real model difference (per-player shifts of sd 0.1 in expected points),
# beyond the path-to-path noise a 40-path CI measures. So each arm is played on several
# replicates, each with a persistent per-player shift N(0, PERTURB_SD) seeded by the
# replicate (the same shift for both arms), and the CI resamples replicates as well as
# paths (`compare_replicated`).
PERTURB_SD = 0.1


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
    start_noise: float = 0.3  # first-squad noise (0: the model's own squad)
    availability: bool = False  # E12: the live tool's status multiplier, point in time
    extra_bank: int = 0  # E13: tenths added to the bank after the first squad
    perturb_seed: int | None = None  # replicated protocol: per-player shift of expected points
    perturb_sd: float = PERTURB_SD


def perturb(weeks: list[Week], seed: int, sd: float = PERTURB_SD) -> list[Week]:
    """Each player's expected points shifted by the same N(0, sd) draw in every week."""
    codes = sorted({c for w in weeks for c in w.xp["player_code"].to_list()})
    shift = dict(zip(codes, np.random.default_rng(seed).normal(0, sd, len(codes)), strict=True))
    bump = pl.col("player_code").replace_strict(shift, default=0.0, return_dtype=pl.Float64)
    return [
        replace(
            w, xp=w.xp.with_columns((pl.col("expected_points") + bump).alias("expected_points"))
        )
        for w in weeks
    ]


def path_file(label: str, season: str, seed: int) -> Path:
    return OUT / "paths" / f"{label}_{season}_{seed:02d}.json"


def _play(job: Job) -> str:
    config, season, horizon, seed, label = job.config, job.season, job.horizon, job.seed, job.label
    cached = weeks_path(config, season, horizon).read_bytes()
    weeks: list[Week] = pickle.loads(cached)  # noqa: S301 - our own cache file
    if job.perturb_seed is not None:
        weeks = perturb(weeks, job.perturb_seed, job.perturb_sd)
    settings = optimizer.Settings(
        time_limit_seconds=30, hit_penalty=job.hit_penalty, value_weight=job.value_weight
    )
    moves = None
    if job.value_weight:
        moves = prices.by_gameweek(pl.read_parquet(PRICE_MOVES), season)
    factors = None
    if job.availability:
        factors = e12.by_gameweek(pl.read_parquet(e12.SIM_FACTORS), season)
    result, records = play(
        season, weeks, settings, seed=seed, start_noise=job.start_noise,
        wildcards=job.wildcards, wildcard_threshold=job.wildcard_threshold, price_moves=moves,
        availability=factors, extra_bank=job.extra_bank,
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


def replicate_label(label: str, replicate: int | None) -> str:
    return label if replicate is None else f"{label}R{replicate}"


def compare_replicated(
    label: str, against: str, seasons: list[str], replicates: int, draws: int = 10_000
) -> tuple[float, float, float]:
    """Mean difference in season total over paired replicates, averaged over seasons; the
    95% CI resamples replicates, then paths within each. Returns (mean, low, high)."""
    t = totals()
    rng = np.random.default_rng(0)
    gains, boot = [], np.zeros(draws)
    for season in seasons:
        arms = [
            [
                t.filter(
                    (pl.col("config") == replicate_label(name, r)) & (pl.col("season") == season)
                )["points"].to_numpy()
                for r in range(replicates)
            ]
            for name in (label, against)
        ]
        diffs = [a.mean() - b.mean() for a, b in zip(*arms, strict=True)]
        gains.append(float(np.mean(diffs)))
        pick = rng.integers(0, replicates, (draws, replicates))
        season_boot = np.zeros(draws)
        for r in range(replicates):
            a, b = arms[0][r], arms[1][r]
            resampled = rng.choice(a, (draws, a.size)).mean(1) - rng.choice(
                b, (draws, b.size)
            ).mean(1)
            season_boot += (pick == r).sum(axis=1) * resampled
        boot += season_boot / replicates
        print(f"{season}: {label} vs {against}, {replicates} replicates: {gains[-1]:+.1f} "
              f"(replicates {', '.join(f'{d:+.1f}' for d in diffs)})")  # fmt: skip
    mean = float(np.mean(gains))
    low, high = (float(x) for x in np.quantile(boot / len(seasons), [0.025, 0.975]))
    print(f"{label} vs {against} over {', '.join(seasons)}: {mean:+.1f} per season "
          f"(95% CI {low:+.1f}, {high:+.1f}), {replicates} replicates")  # fmt: skip
    return mean, low, high


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
    parser.add_argument("--start-noise", type=float, default=0.3, help="0: no noise")
    parser.add_argument("--availability", action="store_true", help="E12 status multiplier")
    parser.add_argument("--extra-bank", type=int, default=0, help="E13, tenths after GW1")
    parser.add_argument(
        "--replicates", type=int, default=0, help="play/compare K perturbed replicates per arm"
    )
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
        replicates: list[int | None] = list(range(args.replicates)) or [None]
        plays = [
            Job(
                c,
                s,
                args.horizon,
                seed,
                args.hit_penalty,
                replicate_label(args.label or c, r),
                tuple(args.wildcards),
                args.wildcard_threshold,
                args.value_weight,
                args.start_noise,
                args.availability,
                args.extra_bank,
                r,
            )
            for r in replicates
            for seed in range(*args.seeds)
            for c in args.configs
            for s in seasons
        ]
        if args.skip_existing:  # resume an interrupted run
            plays = [j for j in plays if not path_file(j.label, j.season, j.seed).exists()]
        with ProcessPoolExecutor(args.workers) as pool:
            for line in pool.map(_play, plays):
                print(line, flush=True)
    elif args.stage == "compare" and args.replicates:
        for label in args.labels:
            compare_replicated(label, args.against, seasons, args.replicates)
    elif args.stage == "compare":
        compare_labels(args.labels, args.against)
    else:
        analyse()


if __name__ == "__main__":
    main()
