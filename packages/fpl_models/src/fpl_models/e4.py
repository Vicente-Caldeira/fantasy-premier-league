"""E4a: triple captain, bench boost and free hit timing, on the E1 v2 season paths.

2025-26 rules: each chip twice a season, once per half (GW1-19, GW20-38), at most one
chip per gameweek. A rule decides at each deadline from information available then:
play the chip the first time its signal reaches T, or in the half's last gameweek if
still unused. Signals (all from expected points):

    triple captain  the captain's expected points          (extra = captain's points)
    bench boost     the bench's expected points             (extra = bench's points)
    free hit        best one-week squad xP - current XI xP  (extra = realised difference)

Triple captain and bench boost don't change a path, so they are rescored from the
recorded squads. A free hit squad doesn't depend on the path either (built from scratch,
reverts next week), so it is solved once per gameweek. Approximation: the free hit
budget is 100.0m, not the path's own squad value (which drifts by a few million).
"Hindsight" picks each half's best gameweek after the fact: an upper bound.

    uv run python -m fpl_models.e4
"""

import argparse
import json
import pickle
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import polars as pl

from fpl_models import e1, optimizer
from fpl_models.experiments import TUNING_SEASONS
from fpl_models.season_sim import Week, auto_subs, score

HALF = 19
THRESHOLDS = {
    "triple_captain": (6.0, 7.0, 8.0, 9.0, 10.0, 99.0),
    "bench_boost": (6.0, 8.0, 10.0, 12.0, 14.0, 99.0),
    "free_hit": (4.0, 8.0, 12.0, 16.0, 20.0, 99.0),
}  # 99 = "always the last gameweek of the half"


def free_hit_squads(weeks: list[Week]) -> dict[int, tuple[float, int]]:
    """Per gameweek: (expected points of the best one-week XI, its realised score)."""
    out = {}
    for week in weeks:
        xp = week.xp.filter(pl.col("gameweek") == week.gameweek)
        plan = optimizer.optimise(
            week.pool, xp, settings=optimizer.Settings(time_limit_seconds=30, bench_weight=0.0)
        )
        gw = plan.gameweeks[0]
        position = dict(zip(week.pool["player_code"], week.pool["position"], strict=True))
        values = dict(zip(xp["player_code"], xp["expected_points"], strict=True))
        bench = sorted((p for p in gw.squad if p not in gw.lineup), key=lambda p: -values[p])
        out[week.gameweek] = (
            gw.expected_points,
            score(gw, bench, week.actual, week.minutes, position),
        )
    return out


def signals(config: str = "v2", seasons: Sequence[str] = TUNING_SEASONS) -> pl.DataFrame:
    """Per path-gameweek: each chip's signal and the extra points it would have scored."""
    rows = []
    for season in seasons:
        weeks: list[Week] = pickle.loads(e1.weeks_path(config, season).read_bytes())  # noqa: S301
        by_gw = {w.gameweek: w for w in weeks}
        free_hit = free_hit_squads(weeks)
        for file in sorted((e1.OUT / "paths").glob(f"{config}_{season}_*.json")):
            seed = int(file.stem.rsplit("_", 1)[1])
            for r in json.loads(file.read_text()):
                week = by_gw[r["gameweek"]]
                position = dict(zip(week.pool["player_code"], week.pool["position"], strict=True))
                xp = {p: v for p, g, v in week.xp.iter_rows() if g == week.gameweek}
                actual, minutes = week.actual, week.minutes
                captain = r["captain"] if minutes.get(r["captain"], 0) > 0 else r["vice_captain"]
                played = auto_subs(r["lineup"], r["bench"], minutes, position)
                base = optimizer.Gameweek(
                    gameweek=week.gameweek, squad=r["squad"], lineup=r["lineup"],
                    captain=r["captain"], vice_captain=r["vice_captain"], transfers_in=[],
                    transfers_out=[], free_transfers=0, hits=0, expected_points=0.0,
                )  # fmt: skip
                scored = score(replace(base), r["bench"], actual, minutes, position)
                lineup_xp = sum(xp.get(p, 0.0) for p in r["lineup"]) + xp.get(r["captain"], 0.0)
                fh_xp, fh_points = free_hit[week.gameweek]
                rows.append({
                    "season": season, "seed": seed, "gameweek": week.gameweek,
                    "half": 1 if week.gameweek <= HALF else 2,
                    "triple_captain_signal": xp.get(r["captain"], 0.0),
                    "triple_captain_extra": actual.get(captain, 0) if captain in played else 0,
                    "bench_boost_signal": sum(xp.get(p, 0.0) for p in r["bench"]),
                    "bench_boost_extra": sum(actual.get(p, 0) for p in r["squad"])
                    - sum(actual.get(p, 0) for p in played),
                    "free_hit_signal": fh_xp - lineup_xp,
                    "free_hit_extra": fh_points - scored,
                })  # fmt: skip
    return pl.DataFrame(rows)


def play_rule(path: pl.DataFrame, chip: str, threshold: float) -> tuple[int, ...]:
    """Gameweeks where the rule plays `chip` on one path (one per half)."""
    used = []
    for _, half in path.sort("gameweek").group_by("half", maintain_order=True):
        signal = half[f"{chip}_signal"].to_numpy()
        hit = np.flatnonzero(signal >= threshold)
        used.append(int(half["gameweek"][int(hit[0]) if hit.size else half.height - 1]))
    return tuple(used)


def evaluate(data: pl.DataFrame) -> pl.DataFrame:
    """Season gain per chip and threshold (mean over paths, 95% CI by path bootstrap)."""
    rng = np.random.default_rng(0)
    rows = []
    paths = data.partition_by("season", "seed", as_dict=True)
    for chip, grid in THRESHOLDS.items():
        for threshold in (*grid, "hindsight"):
            gains = []
            for path in paths.values():
                if threshold == "hindsight":
                    best = path.group_by("half").agg(pl.col(f"{chip}_extra").max())
                    gains.append(float(best[f"{chip}_extra"].sum()))
                    continue
                weeks = play_rule(path, chip, float(threshold))
                gains.append(
                    float(path.filter(pl.col("gameweek").is_in(weeks))[f"{chip}_extra"].sum())
                )
            g = np.array(gains)
            draws = rng.choice(g, (10_000, g.size)).mean(axis=1)
            rows.append({
                "chip": chip, "threshold": str(threshold), "season_gain": g.mean(),
                "ci_low": np.quantile(draws, 0.025), "ci_high": np.quantile(draws, 0.975),
            })  # fmt: skip
    return pl.DataFrame(rows)


def combined(data: pl.DataFrame, best: dict[str, float]) -> pl.DataFrame:
    """All three chips at their best thresholds, at most one chip per gameweek: chips are
    placed in order of average extra; a chip whose gameweek is taken moves to its next
    qualifying gameweek, else the half's last free one."""
    order = sorted(best, key=lambda c: -float(np.mean(data[f"{c}_extra"].to_numpy())))
    out = []
    moved = dict.fromkeys(order, 0)
    for (season, seed), path in data.partition_by("season", "seed", as_dict=True).items():
        total, taken = 0.0, set[int]()
        for chip in order:
            for _, half in path.sort("gameweek").group_by("half", maintain_order=True):
                qualifying = half.filter(pl.col(f"{chip}_signal") >= best[chip])
                free = half.filter(~pl.col("gameweek").is_in(list(taken)))
                free_qualifying = free.filter(pl.col(f"{chip}_signal") >= best[chip])
                row = free_qualifying.head(1) if free_qualifying.height else free.tail(1)
                if qualifying.height and qualifying["gameweek"][0] in taken:
                    moved[chip] += 1
                taken.add(int(row["gameweek"][0]))
                total += float(row[f"{chip}_extra"].to_numpy()[0])
        out.append({"season": season, "seed": seed, "gain": total})
    print(f"chip order {order}; times moved by a clash: {moved}")
    return pl.DataFrame(out)


# Chosen on the tuning seasons (docs/model-experiments.md, E4a), fixed before any
# out-of-sample check.
PRE_REGISTERED = {"triple_captain": 10.0, "bench_boost": 10.0, "free_hit": 20.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=list(TUNING_SEASONS))
    args = parser.parse_args()
    out_of_sample = tuple(args.seasons) != TUNING_SEASONS
    data = signals(seasons=args.seasons)
    name = "e4_signals_oos" if out_of_sample else "e4_signals"
    data.write_parquet(e1.OUT / f"{name}.parquet")
    result = evaluate(data)
    if out_of_sample:
        with pl.Config(tbl_rows=40, float_precision=1):
            print(result)
        both = combined(data, PRE_REGISTERED)
        gain = both["gain"].to_numpy()
        print(f"pre-registered {PRE_REGISTERED}, seasons {args.seasons}: all three chips "
              f"{gain.mean():+.1f} per season (sd {gain.std(ddof=1):.1f})")  # fmt: skip
        return
    with pl.Config(tbl_rows=40, float_precision=1):
        print(result)
    best = {
        chip: float(
            group.filter(pl.col("threshold") != "hindsight").sort("season_gain")["threshold"][-1]
        )
        for (chip,), group in result.group_by("chip")
    }
    print(f"best thresholds (chosen on these same paths: optimistic): {best}")
    both = combined(data, best)
    gain = both["gain"].to_numpy()
    print(f"all three chips: {gain.mean():+.1f} points per season (sd {gain.std(ddof=1):.1f})")


if __name__ == "__main__":
    main()
