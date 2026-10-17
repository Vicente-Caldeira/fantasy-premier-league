"""Load player-gameweek data from the warehouse.

The unit of prediction is a player's FPL points for one gameweek, summed over every
fixture he has in it (double and triple gameweeks included). Each gameweek's
*deadline* is 90 minutes before its first kickoff; models may only use information
from matches that kicked off before it.
"""

from datetime import timedelta
from pathlib import Path

import duckdb
import polars as pl

DEADLINE_BEFORE_KICKOFF = timedelta(minutes=90)
DEFAULT_WAREHOUSE = Path("data/warehouse.duckdb")

# Known before the deadline: who, where, against whom, at what price.
CONTEXT_COLUMNS = (
    "season",
    "gameweek",
    "deadline_utc",
    "player_code",
    "element",
    "position",
    "team_key",
    "n_fixtures",
    "opponent_team_keys",
    "is_home_any",
    "price",
)
# Only known after the matches: never passed to a model for the gameweek it predicts.
OUTCOME_COLUMNS = ("points", "minutes", "goals_scored", "assists", "clean_sheets", "bonus")


def load_player_matches(warehouse: Path = DEFAULT_WAREHOUSE) -> pl.DataFrame:
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        connection.execute("set TimeZone = 'UTC'")
        return connection.sql(
            """
            select season, gameweek, kickoff_utc, player_code, element, position,
                team_key, opponent_team_key, is_home, is_provisional, price,
                total_points, minutes, goals_scored, assists, clean_sheets, bonus
            from fct_player_matches
            """
        ).pl()


def player_gameweeks(matches: pl.DataFrame) -> pl.DataFrame:
    """Aggregate fixtures to one row per (season, gameweek, player), with deadlines.

    Gameweeks containing provisional (unfinished) fixtures are dropped: their
    outcomes are not final yet.
    """
    deadlines = (
        matches.group_by("season", "gameweek")
        .agg(
            (pl.col("kickoff_utc").min() - DEADLINE_BEFORE_KICKOFF).alias("deadline_utc"),
            pl.col("is_provisional").any().alias("has_provisional"),
        )
        .filter(~pl.col("has_provisional"))
        .drop("has_provisional")
    )
    return (
        matches.join(deadlines, on=["season", "gameweek"], how="inner")
        .group_by("season", "gameweek", "deadline_utc", "player_code")
        .agg(
            pl.col("element").first(),
            pl.col("position").first(),
            pl.col("team_key").first(),
            pl.len().alias("n_fixtures"),
            pl.col("opponent_team_key").sort().alias("opponent_team_keys"),
            pl.col("is_home").any().alias("is_home_any"),
            pl.col("price").first(),
            pl.col("total_points").sum().alias("points"),
            pl.col("minutes").sum(),
            pl.col("goals_scored").sum(),
            pl.col("assists").sum(),
            pl.col("clean_sheets").sum(),
            pl.col("bonus").sum(),
        )
        .sort("deadline_utc", "player_code")
    )
