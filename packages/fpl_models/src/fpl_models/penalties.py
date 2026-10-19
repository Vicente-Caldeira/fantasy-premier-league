"""Penalty-taking features from last season's Understat totals (E3a).

Understat publishes penalty xG only as season totals (xG - npxG, about 0.76 per
attempt), so a fixture in season S sees season S-1, which is known before S starts:

    penalty_xg_prev90    the player's penalty xG per 90 last season (any club)
    penalty_share_prev   his share of his club's penalty xG last season, if he is at
                         the same club now (0 otherwise: a new club's order is unknown)

Mid-season changes of penalty taker are not seen. Understat has no FPL identifier,
so players are matched by name within the same club and season (`crosswalk`).
"""

import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import duckdb
import polars as pl

PENALTY_FEATURES = ("penalty_xg_prev90", "penalty_share_prev")
MIN_SIMILARITY = 0.85
SUBSET = 0.97
MIN_MINUTES = 270  # below this, last season's per-90 rate is mostly noise


def _normalise(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(plain.lower().replace("-", " ").replace(".", " ").split())


def similarity(a: str, b: str) -> float:
    """Best of direct and token-sorted ratios ("Son Heung-Min" vs "Heung-Min Son").
    A name whose every token appears in the other scores SUBSET: FPL uses full legal
    names ("Bruno Miguel Borges Fernandes"), Understat common ones."""
    a, b = _normalise(a), _normalise(b)
    short, long = sorted((a.split(), b.split()), key=len)
    if len(short) >= 2 and set(short) <= set(long):
        return SUBSET
    direct = SequenceMatcher(None, a, b).ratio()
    tokens = SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split())))
    return max(direct, tokens.ratio())


def crosswalk(understat: pl.DataFrame, fpl: pl.DataFrame) -> pl.DataFrame:
    """(season, understat_player_id) -> player_code. `understat` has season,
    understat_player_id, player_name, team_keys (list); `fpl` has season, player_code,
    team_key, full_name, web_name. A match needs MIN_SIMILARITY and a clear winner."""
    by_club: dict[tuple[str, str], list[tuple[int, str, str]]] = {}
    for season, code, team, full, web in fpl.select(
        "season", "player_code", "team_key", "full_name", "web_name"
    ).iter_rows():
        by_club.setdefault((season, team), []).append((code, full, web))
    rows = []
    for season, player_id, name, teams in understat.select(
        "season", "understat_player_id", "player_name", "team_keys"
    ).iter_rows():
        scored = sorted(
            (
                (max(similarity(name, full), similarity(name, web)), code)
                for team in teams
                for code, full, web in by_club.get((season, team), [])
            ),
            reverse=True,
        )
        if not scored or scored[0][0] < MIN_SIMILARITY:
            continue
        if len(scored) > 1 and scored[1][0] > scored[0][0] - 0.02 and scored[1][1] != scored[0][1]:
            continue  # ambiguous
        rows.append({"season": season, "understat_player_id": player_id,
                     "player_code": scored[0][1], "similarity": scored[0][0]})  # fmt: skip
    return pl.DataFrame(rows).unique(["season", "player_code"], keep="first")


def load(warehouse: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Understat season totals (with team keys) and FPL players per season."""
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        understat = connection.sql(
            "select season, understat_player_id, player_name, team_title, minutes, penalty_xg"
            " from fct_understat_player_seasons"
        ).pl()
        teams = connection.sql("select understat_name, team_key from dim_teams").pl()
        fpl = connection.sql(
            "select season, player_code, team_key, first_name || ' ' || second_name as full_name,"
            " web_name from dim_player_seasons"
        ).pl()
    names = dict(zip(teams["understat_name"], teams["team_key"], strict=True))
    understat = understat.with_columns(
        team_keys=pl.col("team_title")
        .str.split(",")
        .list.eval(pl.element().replace_strict(names, default=None))
    ).drop("team_title")
    return understat, fpl


def season_table(understat: pl.DataFrame, fpl: pl.DataFrame) -> pl.DataFrame:
    """Per (season, player_code): penalty xG per 90 and share of his club's penalty xG."""
    single_club = pl.col("team_keys").list.len() == 1
    club_totals = (
        understat.filter(single_club)
        .with_columns(team_key=pl.col("team_keys").list.first())
        .group_by("season", "team_key")
        .agg(pl.col("penalty_xg").sum().alias("club_penalty_xg"))
    )
    return (
        understat.join(crosswalk(understat, fpl), on=["season", "understat_player_id"])
        .with_columns(team_key=pl.when(single_club).then(pl.col("team_keys").list.first()))
        .join(club_totals, on=["season", "team_key"], how="left")
        .select(
            "season",
            "player_code",
            pl.col("team_key").alias("penalty_team_key"),
            pl.when(pl.col("minutes") >= MIN_MINUTES)
            .then(pl.col("penalty_xg") / pl.col("minutes") * 90)
            .otherwise(None)
            .alias("penalty_xg_prev90"),
            (pl.col("penalty_xg") / pl.col("club_penalty_xg"))
            .fill_nan(None)
            .alias("penalty_share"),
        )
    )


def _next_season(season: str) -> str:
    start = int(season[:4]) + 1
    return f"{start}-{(start + 1) % 100:02d}"


def add_penalties(rows: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Join last season's penalty features onto fixture rows (season, player_code,
    team_key)."""
    prev = table.with_columns(pl.col("season").map_elements(_next_season, pl.String))
    joined = rows.join(prev, on=["season", "player_code"], how="left", maintain_order="left")
    same_club = pl.col("penalty_team_key") == pl.col("team_key")
    return joined.with_columns(
        penalty_share_prev=pl.when(same_club).then(pl.col("penalty_share")).otherwise(0.0),
        penalty_xg_prev90=pl.col("penalty_xg_prev90").fill_null(0.0),
    ).drop("penalty_team_key", "penalty_share")


def main() -> None:
    """Crosswalk coverage report: share of Understat penalty xG and minutes matched."""
    from fpl_models.data import DEFAULT_WAREHOUSE  # noqa: PLC0415

    understat, fpl = load(DEFAULT_WAREHOUSE)
    walk = crosswalk(understat, fpl)
    matched = understat.join(walk, on=["season", "understat_player_id"], how="left")
    with pl.Config(tbl_rows=20):
        print(
            matched.group_by("season")
            .agg(
                pl.len().alias("players"),
                pl.col("player_code").is_not_null().mean().alias("matched"),
                (pl.col("minutes").filter(pl.col("player_code").is_not_null()).sum()
                 / pl.col("minutes").sum()).alias("minutes_matched"),
                (pl.col("penalty_xg").filter(pl.col("player_code").is_not_null()).sum()
                 / pl.col("penalty_xg").sum()).alias("penalty_xg_matched"),
            )
            .sort("season")
        )  # fmt: skip
        missed = matched.filter(pl.col("player_code").is_null())
        print(
            missed.sort("penalty_xg", "minutes", descending=True)
            .select("season", "player_name", "team_keys", "minutes", "penalty_xg")
            .head(20)
        )


if __name__ == "__main__":
    main()
