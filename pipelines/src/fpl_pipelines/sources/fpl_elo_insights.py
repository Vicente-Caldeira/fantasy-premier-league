"""FPL-Elo-Insights (github.com/olbauday/FPL-Elo-Insights): Opta-style match stats.

The repo updates twice daily; bronze snapshots it via `fpl_pipelines.github_repo`.

Layouts:
    2024-2025       data/2024-2025/<dataset>/GW<n>/<dataset>.csv
    2025-2026 ..    data/<season>/By Tournament/Premier League/GW<n>/<dataset>.csv
                    data/<season>/By Tournament/<cup>/GW<n>/matches.csv   (`cup_matches`)

Cup files (European competitions and domestic cups, 2025-26 onwards) list only matches
involving a Premier League club, grouped by the league gameweek they fall in; foreign
clubs have no team code.

The repo has no license, so its data is ingested for personal use and never committed.
"""

import re
from typing import Any

import polars as pl

from fpl_pipelines.github_repo import FileRef
from fpl_pipelines.sources.csv_contract import (
    ContractError,
    optional,
    read_strings,
    required,
    typed,
    value,
)

REPO = "olbauday/FPL-Elo-Insights"
BRANCH = "main"
FIRST_SEASON = 2024
DATASETS = ("matches", "playermatchstats")
CUPS = ("Champions League", "Europa League", "Conference League", "EFL Cup", "FA Cup")
CUP_MATCHES = "cup_matches"

_LEGACY = re.compile(r"^data/(\d{4})-(\d{4})/(\w+)/GW(\d+)/(\w+)\.csv$")
_CURRENT = re.compile(r"^data/(\d{4})-(\d{4})/By Tournament/Premier League/GW(\d+)/(\w+)\.csv$")
_CUP = re.compile(r"^data/(\d{4})-(\d{4})/By Tournament/([^/]+)/GW(\d+)/matches\.csv$")


def _season(start: str, end: str) -> str:
    return f"{start}-{end[2:]}"


def select_files(tree: list[dict[str, Any]]) -> list[FileRef]:
    """Premier League per-gameweek files for the datasets we ingest, plus cup matches."""
    refs = []
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path = entry["path"]
        if m := _CURRENT.match(path):
            start, end, gw, name = m.groups()
            dataset = name
        elif (m := _CUP.match(path)) and m.group(3) in CUPS:
            start, end, _, gw = m.groups()
            dataset = CUP_MATCHES
        elif (m := _LEGACY.match(path)) and m.group(3) == m.group(5):
            start, end, dataset, gw, _ = m.groups()
        else:
            continue
        if dataset in (*DATASETS, CUP_MATCHES) and int(start) >= FIRST_SEASON:
            refs.append(FileRef(_season(start, end), dataset, path, entry["sha"], int(gw)))
    return sorted(refs, key=lambda r: (r.season, r.dataset, r.gameweek or 0))


# --- parsing --------------------------------------------------------------------------
# Columns we depend on. Required ones exist in every season; optional ones were added
# later (defensive contributions in 2025-26, physical stats in 2026-27) and are null
# before that. Everything else stays in bronze.

PLAYER_KEYS = {"player_id": pl.Int64, "match_id": pl.String}
PLAYER_COUNTS = (
    "minutes_played", "start_min", "finish_min", "goals", "assists", "penalties_scored",
    "penalties_missed", "total_shots", "shots_on_target", "big_chances_missed", "touches",
    "touches_opposition_box", "chances_created", "successful_dribbles", "accurate_passes",
    "final_third_passes", "accurate_crosses", "accurate_long_balls", "tackles", "tackles_won",
    "interceptions", "recoveries", "blocks", "clearances", "headed_clearances", "dribbled_past",
    "duels_won", "duels_lost", "ground_duels_won", "aerial_duels_won", "was_fouled",
    "fouls_committed", "offsides", "saves", "goals_conceded", "team_goals_conceded",
    "sweeper_actions", "high_claim",
)  # fmt: skip
PLAYER_FLOATS = ("xg", "xa", "xgot", "xgot_faced", "goals_prevented")
PLAYER_OPTIONAL_COUNTS = (
    "defensive_contributions", "dispossessed", "saves_inside_box", "corners", "number_of_sprints",
)  # fmt: skip
PLAYER_OPTIONAL_FLOATS = (
    "distance_covered", "running_distance", "sprinting_distance", "walking_distance", "top_speed",
)  # fmt: skip

MATCH_KEYS = {"match_id": pl.String, "gameweek": pl.Int64}
# Per-side team stats: canonical name -> source suffix (home_<suffix> / away_<suffix>).
MATCH_SIDE_COUNTS = {
    "team_code": "team", "score": "score", "total_shots": "total_shots",
    "shots_on_target": "shots_on_target", "shots_inside_box": "shots_inside_box",
    "big_chances": "big_chances", "big_chances_missed": "big_chances_missed",
    "touches_in_opposition_box": "touches_in_opposition_box", "corners": "corners",
    "passes": "passes", "accurate_passes": "accurate_passes", "keeper_saves": "keeper_saves",
    "tackles_won": "tackles_won", "interceptions": "interceptions", "clearances": "clearances",
    "blocks": "blocks", "fouls_committed": "fouls_committed", "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
}  # fmt: skip
MATCH_SIDE_FLOATS = {
    "elo": "team_elo", "possession": "possession", "xg": "expected_goals_xg",
    "non_penalty_xg": "non_penalty_xg", "xg_open_play": "xg_open_play",
    "xg_set_play": "xg_set_play", "xgot": "xg_on_target_xgot",
}  # fmt: skip


def _utc(raw: pl.Expr) -> pl.Expr:
    """Kickoff times come in three shapes; all are normalised to UTC.

    2024-25 `2024-08-17 14:00:00` (one row without seconds), 2025-26
    `2025-08-15T19:00:00+00:00`, 2026-27 `2026-09-05T16:30:00`. Naive values are UTC
    (verified against FPL kickoff times).
    """
    text = raw.str.replace(" ", "T", literal=True)
    text = pl.when(text.str.len_chars() == 16).then(text + ":00").otherwise(text)
    return (
        pl.when(text.str.contains(r"[+-]\d{2}:\d{2}$"))
        .then(text.str.to_datetime("%Y-%m-%dT%H:%M:%S%z", strict=False).dt.convert_time_zone("UTC"))
        .otherwise(
            text.str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False).dt.replace_time_zone("UTC")
        )
    )


def parse_player_match_stats(data: bytes, season: str, gameweek: int) -> pl.DataFrame:
    raw = read_strings(data)
    keys = [value(raw, k, required=True).alias(k) for k in PLAYER_KEYS]
    floats = {c: required(raw, c) for c in PLAYER_FLOATS}
    floats |= {c: optional(raw, c) for c in PLAYER_OPTIONAL_FLOATS}
    counts = {c: required(raw, c) for c in PLAYER_COUNTS}
    counts |= {c: optional(raw, c) for c in PLAYER_OPTIONAL_COUNTS}
    stats = typed(raw, floats, counts)
    ids = raw.select(keys).with_columns(pl.col("player_id").cast(pl.Float64).cast(pl.Int64))
    return pl.concat([ids, stats], how="horizontal").select(
        pl.lit(season).alias("season"), pl.lit(gameweek).alias("gameweek"), pl.all()
    )


def parse_matches(data: bytes, season: str) -> pl.DataFrame:
    raw = read_strings(data)
    floats: dict[str, pl.Expr] = {}
    counts: dict[str, pl.Expr] = {"gameweek": required(raw, "gameweek")}
    for side in ("home", "away"):
        counts |= {
            f"{side}_{n}": required(raw, f"{side}_{s}") for n, s in MATCH_SIDE_COUNTS.items()
        }
        floats |= {
            f"{side}_{n}": required(raw, f"{side}_{s}") for n, s in MATCH_SIDE_FLOATS.items()
        }
    stats = typed(raw, floats, counts)
    meta = raw.select(
        value(raw, "match_id", required=True).alias("match_id"),
        _utc(value(raw, "kickoff_time", required=True)).alias("kickoff_time"),
        (value(raw, "finished", required=True) == "True").alias("finished"),
        value(raw, "tournament", required=False).alias("tournament"),
    )
    # Every fixture has a kickoff time, so a null here means an unrecognised format.
    unparsed = meta.filter(pl.col("kickoff_time").is_null()).height
    if unparsed:
        raise ContractError(f"{unparsed} kickoff times could not be parsed")
    return pl.concat([meta, stats], how="horizontal").select(
        pl.lit(season).alias("season"), pl.all()
    )


def cup_name(path: str) -> str:
    """The competition of a cup_matches file, from its path."""
    if not (m := _CUP.match(path)):
        raise ContractError(f"not a cup matches file: {path}")
    return m.group(3)


def parse_cup_matches(data: bytes, season: str, competition: str) -> pl.DataFrame:
    """One row per cup match: kickoff and the FPL team codes of the Premier League sides
    (null for a foreign or lower-league club).

    Upstream leaves some rows without a kickoff time (2025-26 European knockout rounds)
    and lists undecided ties as placeholders ("arsenal-vs-newcastle-united/manchester-
    city"). Neither can be placed in time, so both are dropped: congestion from those
    matches is undercounted.
    """
    raw = read_strings(data)
    out = raw.select(
        pl.lit(season).alias("season"),
        pl.lit(competition).alias("competition"),
        value(raw, "match_id", required=True).alias("match_id"),
        _utc(value(raw, "kickoff_time", required=True)).alias("kickoff_time"),
        (value(raw, "finished", required=True) == "True").alias("finished"),
        optional(raw, "home_team").cast(pl.Int64).alias("home_team_code"),
        optional(raw, "away_team").cast(pl.Int64).alias("away_team_code"),
    )
    return out.filter(
        pl.col("kickoff_time").is_not_null() & ~pl.col("match_id").str.contains("/", literal=True)
    )
