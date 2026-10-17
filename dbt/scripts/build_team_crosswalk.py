"""Derive `seeds/teams.csv`: one row per club, mapping every source's identifier to it.

Sources name clubs differently ("Man United", "Manchester United", FPL code 1).
Rather than hand-typing a mapping, every football-data and Understat match is lined
up with the FPL-coded fixture list on (season, date, home goals, away goals), and
each name/id is assigned the FPL team code it co-occurs with most. The script fails
if any assignment is ambiguous, so a wrong mapping cannot be written silently.

Clubs only seen before FPL-coded data begins (2016-17) get no FPL code.

    uv run python dbt/scripts/build_team_crosswalk.py [--lake data]
"""

import argparse
import glob
import re
from pathlib import Path

import polars as pl

SEED = Path(__file__).parents[1] / "seeds" / "teams.csv"
# A winner needs at least 4x the votes of the runner-up.
MIN_MARGIN = 4


def read(lake: str, table: str) -> pl.DataFrame:
    return pl.concat(
        [pl.read_parquet(f) for f in sorted(glob.glob(f"{lake}/{table}/*/data.parquet"))]
    )


def fpl_fixtures(lake: str) -> pl.DataFrame:
    """FPL-coded results: vaastav (2016-17 .. 2025-26) + the latest FPL API snapshot."""
    vaastav = read(lake, "silver/vaastav/player_matches").filter(pl.col("was_home"))
    history = vaastav.select(
        "season",
        pl.col("kickoff_time").dt.date().alias("date"),
        pl.col("team_code").alias("home_code"),
        pl.col("opponent_team_code").alias("away_code"),
        pl.col("team_h_score").alias("home_goals"),
        pl.col("team_a_score").alias("away_goals"),
    ).unique()
    fixtures = pl.read_parquet(
        sorted(glob.glob(f"{lake}/silver/fpl/fixtures_table/*/data.parquet"))[-1]
    )
    teams = pl.read_parquet(sorted(glob.glob(f"{lake}/silver/fpl/teams/*/data.parquet"))[-1])
    code = dict(zip(teams["id"], teams["code"], strict=True))
    current = fixtures.filter(pl.col("finished")).select(
        "season",
        pl.col("kickoff_time").dt.date().alias("date"),
        pl.col("team_h").replace_strict(code).alias("home_code"),
        pl.col("team_a").replace_strict(code).alias("away_code"),
        pl.col("team_h_score").alias("home_goals"),
        pl.col("team_a_score").alias("away_goals"),
    )
    return pl.concat([history, current], how="vertical_relaxed")


def vote(matches: pl.DataFrame, reference: pl.DataFrame, source: str) -> dict[str, int]:
    keys = ["season", "date", "home_goals", "away_goals"]
    joined = matches.join(reference, on=keys, how="inner")
    votes = (
        pl.concat(
            [
                joined.select(pl.col("home_id").alias("id"), pl.col("home_code").alias("code")),
                joined.select(pl.col("away_id").alias("id"), pl.col("away_code").alias("code")),
            ]
        )
        .group_by("id", "code")
        .len()
    )
    mapping = {}
    for (team_id,), group in votes.group_by("id"):
        ranked = group.sort("len", descending=True)["len"].to_list()
        if len(ranked) > 1 and ranked[0] < MIN_MARGIN * ranked[1]:
            raise SystemExit(f"{source}: ambiguous mapping for {team_id!r}: {group.rows()}")
        mapping[team_id] = group.sort("len", descending=True)["code"][0]
    if len(set(mapping.values())) != len(mapping):
        raise SystemExit(f"{source}: two identifiers mapped to the same FPL code")
    return mapping


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("'", "")).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data")
    lake = parser.parse_args().lake
    reference = fpl_fixtures(lake)

    fd = read(lake, "silver/football_data/matches").select(
        "season",
        pl.col("match_date").alias("date"),
        pl.col("home_team").alias("home_id"),
        pl.col("away_team").alias("away_id"),
        "home_goals",
        "away_goals",
    )
    us = read(lake, "silver/understat/matches").filter(pl.col("is_result"))
    us_names = dict(
        pl.concat(
            [
                us.select(pl.col("home_team_id").alias("id"), pl.col("home_team").alias("name")),
                us.select(pl.col("away_team_id").alias("id"), pl.col("away_team").alias("name")),
            ]
        )
        .unique()
        .iter_rows()
    )
    us = us.select(
        "season",
        pl.col("match_date").alias("date"),
        pl.col("home_team_id").cast(pl.String).alias("home_id"),
        pl.col("away_team_id").cast(pl.String).alias("away_id"),
        "home_goals",
        "away_goals",
    )

    fd_to_code = vote(fd, reference, "football-data")
    us_to_code = vote(us, reference, "understat")
    code_to_fd = {c: n for n, c in fd_to_code.items()}
    code_to_us = {c: int(i) for i, c in us_to_code.items()}

    teams = pl.read_parquet(sorted(glob.glob(f"{lake}/silver/fpl/teams/*/data.parquet"))[-1])
    fpl_names = {r["code"]: (r["name"], r["short_name"]) for r in teams.iter_rows(named=True)}

    rows = []
    for code in sorted(set(code_to_fd) | set(code_to_us)):
        name = fpl_names.get(code, (code_to_fd.get(code) or us_names[code_to_us[code]], None))
        rows.append(
            {
                "team_key": slug(code_to_fd.get(code) or name[0]),
                "name": name[0],
                "short_name": name[1],
                "fpl_team_code": code,
                "football_data_name": code_to_fd.get(code),
                "understat_team_id": code_to_us.get(code),
                "understat_name": us_names.get(code_to_us.get(code, -1)),
            }
        )
    # Clubs Understat saw only before FPL-coded data starts (2014-15, 2015-16).
    for team_id, name in sorted(us_names.items()):
        if str(team_id) not in us_to_code:
            rows.append(
                {
                    "team_key": slug(name),
                    "name": name,
                    "short_name": None,
                    "fpl_team_code": None,
                    "football_data_name": None,
                    "understat_team_id": team_id,
                    "understat_name": name,
                }
            )
    seed = pl.DataFrame(rows, infer_schema_length=None).sort("team_key")
    if seed["team_key"].n_unique() != seed.height:
        raise SystemExit("team_key collision")
    seed.write_csv(SEED)
    print(f"wrote {seed.height} teams to {SEED}")


if __name__ == "__main__":
    main()
