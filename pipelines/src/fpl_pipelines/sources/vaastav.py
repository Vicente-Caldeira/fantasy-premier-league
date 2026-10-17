"""vaastav/Fantasy-Premier-League: historical FPL data, one folder per season.

Used for seasons before our own daily FPL snapshots began (2016-17 .. 2025-26):

    data/<season>/gws/merged_gw.csv   one row per player per fixture (points, stats, price)
    data/<season>/players_raw.csv     end-of-season player list (position, team, stable code)

Quirks handled here:
  * 2016-17 .. 2018-19 are Windows-1252 encoded; later seasons UTF-8.
  * Position and team columns only exist from 2020-21; they come from players_raw.
  * 2024-25 includes Assistant Manager entries (element_type 5), which are not players.
  * 2019-20 rounds jump from 29 to 39: FPL voided GW30-38 after the COVID suspension.
    Fixture 275, postponed from GW29 to GW39, appears twice; the GW29 rows are
    zero-minute placeholders. 2025-26 has a few exact duplicate rows.
  * `xP` is scraped after matches and leaks the outcome, so it is never ingested.
  * Points follow each season's own rules (e.g. defensive contributions from 2025-26).

The code is MIT licensed but the data belongs to FPL/Understat, so it is never committed.
"""

import re
from typing import Any

import polars as pl

from fpl_pipelines.github_repo import FileRef
from fpl_pipelines.sources.csv_contract import (
    ContractError,
    boolean,
    optional,
    read_strings,
    required,
    typed,
    value,
)

REPO = "vaastav/Fantasy-Premier-League"
BRANCH = "master"
FIRST_SEASON = 2016
LAST_SEASON = 2025  # 2025-26; later seasons come from our own FPL API snapshots
ASSISTANT_MANAGER = 5

_FILES = {
    "merged_gw": re.compile(r"^data/(\d{4}-\d{2})/gws/merged_gw\.csv$"),
    "players_raw": re.compile(r"^data/(\d{4}-\d{2})/players_raw\.csv$"),
}


def select_files(tree: list[dict[str, Any]]) -> list[FileRef]:
    refs = []
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        for dataset, pattern in _FILES.items():
            if (m := pattern.match(entry["path"])) and (
                FIRST_SEASON <= int(m.group(1)[:4]) <= LAST_SEASON
            ):
                refs.append(FileRef(m.group(1), dataset, entry["path"], entry["sha"]))
    return sorted(refs, key=lambda r: (r.season, r.dataset))


PLAYER_COUNTS = ("id", "code", "element_type", "team", "team_code", "now_cost", "total_points")
PLAYER_STRINGS = ("web_name", "first_name", "second_name")

MATCH_COUNTS = (
    "element", "fixture", "round", "opponent_team", "team_h_score", "team_a_score", "minutes",
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "total_points", "value", "selected", "transfers_in", "transfers_out", "transfers_balance",
)  # fmt: skip
MATCH_FLOATS = ("influence", "creativity", "threat", "ict_index")
# Added in later seasons; null before.
MATCH_OPTIONAL_COUNTS = (
    "starts", "tackles", "recoveries", "clearances_blocks_interceptions", "defensive_contribution",
)  # fmt: skip
MATCH_OPTIONAL_FLOATS = (
    "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
)  # fmt: skip


def parse_players(data: bytes, season: str) -> pl.DataFrame:
    raw = read_strings(data)
    counts = {c: required(raw, c) for c in PLAYER_COUNTS}
    names = raw.select(value(raw, c, required=True).alias(c) for c in PLAYER_STRINGS)
    players = pl.concat([typed(raw, {}, counts), names], how="horizontal")
    return players.filter(pl.col("element_type") != ASSISTANT_MANAGER).select(
        pl.lit(season).alias("season"), pl.all()
    )


def parse_player_matches(data: bytes, season: str) -> pl.DataFrame:
    raw = read_strings(data)
    floats = {c: required(raw, c) for c in MATCH_FLOATS}
    floats |= {c: optional(raw, c) for c in MATCH_OPTIONAL_FLOATS}
    counts = {c: required(raw, c) for c in MATCH_COUNTS}
    counts |= {c: optional(raw, c) for c in MATCH_OPTIONAL_COUNTS}
    meta = raw.select(
        value(raw, "kickoff_time", required=True)
        .str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=True, time_zone="UTC")
        .alias("kickoff_time"),
        boolean(raw, "was_home").alias("was_home"),
    )
    if meta["was_home"].null_count():
        raise ValueError("was_home must be True or False")
    return pl.concat([meta, typed(raw, floats, counts)], how="horizontal").select(
        pl.lit(season).alias("season"), pl.all()
    )


def deduplicate(matches: pl.DataFrame) -> pl.DataFrame:
    """One row per (element, fixture).

    Exact copies are dropped. A rescheduled fixture keeps the row from when it was
    actually played (the latest kickoff); two *played* rows for one fixture can't be
    explained by rescheduling, so they fail instead of being guessed at.
    """
    unique = matches.unique(maintain_order=True)
    keys = ["element", "fixture"]
    played_twice = (
        unique.filter(pl.col("minutes") > 0).group_by(keys).len().filter(pl.col("len") > 1)
    )
    if played_twice.height:
        raise ContractError(f"{played_twice.height} player-fixtures have two played rows")
    return unique.sort("kickoff_time", descending=True).unique(keys, keep="first")


def enrich(matches: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Attach position, the team played for, and stable cross-season codes.

    `players_raw` is an end-of-season snapshot, so its team is wrong for every match a
    player played before a mid-season transfer. The team is instead derived from the
    fixture: home players' opponent is the away side and vice versa. FPL team ids are
    season-specific (1..20) and are mapped to stable team codes.
    """
    sides = matches.group_by("fixture").agg(
        pl.col("opponent_team").filter(~pl.col("was_home")).first().alias("home_team"),
        pl.col("opponent_team").filter(pl.col("was_home")).first().alias("away_team"),
    )
    team_codes = players.select("team", "team_code").unique()
    to_code = dict(zip(team_codes["team"], team_codes["team_code"], strict=True))
    return (
        matches.join(sides, on="fixture", how="left")
        .with_columns(
            pl.when(pl.col("was_home"))
            .then(pl.col("home_team"))
            .otherwise(pl.col("away_team"))
            .alias("team")
        )
        .with_columns(
            pl.col("team").replace_strict(to_code, default=None).alias("team_code"),
            pl.col("opponent_team")
            .replace_strict(to_code, default=None)
            .alias("opponent_team_code"),
        )
        .drop("home_team", "away_team")
        .join(
            players.select(
                pl.col("id").alias("element"), pl.col("code").alias("player_code"), "element_type"
            ),
            on="element",
            how="inner",  # Assistant Managers are absent from `players`
        )
        .sort("kickoff_time", "fixture", "element")
    )
