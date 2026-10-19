"""E2: risk-aware captaincy, rescored on the E1 season paths.

Each E1 path records the squad, XI and bench chosen every gameweek. Here only the
captain and vice-captain are re-chosen, by each rule from information available at
the deadline, and the gameweek is rescored (auto-subs, vice fallback). Transfers and
lineups stay identical, so differences come from the captain choice alone.

Rules: score = expected points + w x P(haul >= 10), for several w (w = 0 is the
current behaviour); and "p90", the highest simulated 90th percentile.

    uv run python -m fpl_models.e2 [--config v2]
"""

import argparse
import json
import pickle
from dataclasses import replace

import numpy as np
import polars as pl

from fpl_models import e1, optimizer
from fpl_models.experiments import TUNING_SEASONS
from fpl_models.season_sim import Week, score

WEIGHTS = (0.0, 2.0, 5.0, 10.0, 20.0)
BOOTSTRAP = 10_000


def _captains(lineup: list[int], value: dict[int, float]) -> tuple[int, int]:
    ranked = sorted(lineup, key=lambda p: -value.get(p, 0.0))
    return ranked[0], ranked[1]


def rescore(config: str) -> pl.DataFrame:
    """Per path-gameweek realised points under every rule."""
    rows = []
    for season in TUNING_SEASONS:
        weeks: list[Week] = pickle.loads(e1.weeks_path(config, season).read_bytes())  # noqa: S301
        by_gw = {w.gameweek: w for w in weeks}
        for file in sorted((e1.OUT / "paths").glob(f"{config}_{season}_*.json")):
            seed = int(file.stem.rsplit("_", 1)[1])
            for record in json.loads(file.read_text()):
                week = by_gw[record["gameweek"]]
                position = dict(zip(week.pool["player_code"], week.pool["position"], strict=True))
                xp = {p: v for p, g, v in week.xp.iter_rows() if g == week.gameweek}
                upside = {r[0]: r[1:] for r in week.upside.iter_rows()}  # p90, p_haul
                base = optimizer.Gameweek(
                    gameweek=week.gameweek, squad=record["squad"], lineup=record["lineup"],
                    captain=record["captain"], vice_captain=record["vice_captain"],
                    transfers_in=[], transfers_out=[], free_transfers=0, hits=0,
                    expected_points=0.0,
                )  # fmt: skip
                rules: dict[str, dict[int, float]] = {
                    f"w{w:g}": {p: xp.get(p, 0.0) + w * upside.get(p, (0, 0))[1] for p in xp}
                    for w in WEIGHTS
                }
                rules["p90"] = {p: upside.get(p, (0, 0))[0] + 1e-3 * xp.get(p, 0.0) for p in xp}
                row = {
                    "season": season, "seed": seed, "gameweek": week.gameweek,
                    "recorded": record["points"] + optimizer.HIT_COST * record["hits"],
                }  # fmt: skip
                for name, value in rules.items():
                    captain, vice = _captains(record["lineup"], value)
                    chosen = replace(base, captain=captain, vice_captain=vice)
                    row[name] = score(chosen, record["bench"], week.actual, week.minutes, position)
                rows.append(row)
    return pl.DataFrame(rows)


def analyse(points: pl.DataFrame) -> pl.DataFrame:
    """Each rule vs w0: per-path season-total gain, bootstrapped over gameweeks (all
    paths share a gameweek's real outcomes, so gameweeks are the independent unit)."""
    rules = [c for c in points.columns if c not in ("season", "seed", "gameweek", "recorded", "w0")]
    per_gw = points.group_by("season", "gameweek").agg(
        (pl.col(r) - pl.col("w0")).mean().alias(r) for r in rules
    )
    gameweeks_per_season = per_gw.height / per_gw["season"].n_unique()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, per_gw.height, size=(BOOTSTRAP, per_gw.height))
    out = []
    for rule in rules:
        gain = per_gw[rule].to_numpy()
        draws = gain[idx].mean(axis=1) * gameweeks_per_season
        out.append({
            "rule": rule, "season_gain": gain.mean() * gameweeks_per_season,
            "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "gw_better": float((gain > 0).mean()), "gw_worse": float((gain < 0).mean()),
        })  # fmt: skip
    return pl.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v2")
    args = parser.parse_args()
    points = rescore(args.config)
    points.write_parquet(e1.OUT / f"e2_{args.config}.parquet")
    # Sanity check: w0 is the optimiser's own rule, so it should reproduce the paths.
    match = float(np.mean(points["w0"].to_numpy() == points["recorded"].to_numpy()))
    print(f"{points.height} path-gameweeks; w0 reproduces the recorded points in {match:.1%}")
    print(points.select(pl.exclude("season", "seed", "gameweek", "recorded")).mean())
    with pl.Config(float_precision=2):
        print(analyse(points))


if __name__ == "__main__":
    main()
