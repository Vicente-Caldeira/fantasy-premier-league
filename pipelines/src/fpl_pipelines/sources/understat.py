"""Understat: team-level expected goals for every Premier League match since 2014-15.

`/getLeagueData/EPL/<year>` returns, per season:
    dates    every match: result, xG per side, Understat's win/draw/loss forecast
    teams    each team's match log: xG/xGA, non-penalty xG, PPDA, deep completions, xPts
    players  season totals per player

Kickoff times are NOT reliable: in 2025-26, 343 of 380 matched UTC, 31 (May) were an
hour ahead and 6 on the October clock-change day an hour behind. The match *date* is
reliable, so it is the join key, and the time is kept only as `kickoff_reported`.

Understat has no official API or published terms for reuse: requests are infrequent
(weekly for the current season) and the data is never committed.
"""

from datetime import datetime

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

BASE_URL = "https://understat.com"
FIRST_SEASON = 2014


def league_path(season: str) -> str:
    """`2025-26` -> `/getLeagueData/EPL/2025` (Understat names seasons by start year)."""
    return f"/getLeagueData/EPL/{season[:4]}"


def request_headers(season: str) -> dict[str, str]:
    # The endpoint only answers XHR requests made from the league page.
    return {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE_URL}/league/EPL/{season[:4]}",
    }


class Contract(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class Side(Contract):
    id: int
    title: str


class HomeAway(Contract):
    h: float | None
    a: float | None


class Forecast(Contract):
    w: float
    d: float
    l: float  # noqa: E741 - upstream field name


class Match(Contract):
    id: int
    isResult: bool  # noqa: N815 - upstream field name
    h: Side
    a: Side
    goals: HomeAway
    xG: HomeAway  # noqa: N815
    datetime: datetime
    forecast: Forecast | None = None


class Pressing(Contract):
    """PPDA components: opponent passes allowed, and our defensive actions."""

    att: int
    def_: int = Field(alias="def")  # `def` is a Python keyword


class TeamMatch(Contract):
    h_a: str
    xG: float  # noqa: N815
    xGA: float  # noqa: N815
    npxG: float  # noqa: N815
    npxGA: float  # noqa: N815
    ppda: Pressing
    ppda_allowed: Pressing
    deep: int
    deep_allowed: int
    scored: int
    missed: int
    xpts: float
    result: str
    date: datetime


class Team(Contract):
    id: int
    title: str
    history: list[TeamMatch]


class PlayerSeason(Contract):
    id: int
    player_name: str
    team_title: str
    position: str
    games: int
    time: int
    goals: int
    npg: int
    assists: int
    shots: int
    key_passes: int
    yellow_cards: int
    red_cards: int
    xG: float  # noqa: N815
    npxG: float  # noqa: N815
    xA: float  # noqa: N815
    xGChain: float  # noqa: N815
    xGBuildup: float  # noqa: N815


class League(Contract):
    teams: dict[str, Team]
    players: list[PlayerSeason]
    dates: list[Match]


def matches_table(league: League, season: str) -> pl.DataFrame:
    rows = [
        {
            "season": season,
            "match_id": m.id,
            "match_date": m.datetime.date(),
            "kickoff_reported": m.datetime,
            "is_result": m.isResult,
            "home_team_id": m.h.id,
            "home_team": m.h.title,
            "away_team_id": m.a.id,
            "away_team": m.a.title,
            "home_goals": int(m.goals.h) if m.goals.h is not None else None,
            "away_goals": int(m.goals.a) if m.goals.a is not None else None,
            "home_xg": m.xG.h,
            "away_xg": m.xG.a,
            "forecast_home_win": m.forecast.w if m.forecast else None,
            "forecast_draw": m.forecast.d if m.forecast else None,
            "forecast_away_win": m.forecast.l if m.forecast else None,
        }
        for m in league.dates
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def team_matches_table(league: League, season: str, matches: pl.DataFrame) -> pl.DataFrame:
    """One row per team per played match, linked to `match_id` via date and team."""
    rows = [
        {
            "season": season,
            "team_id": team.id,
            "team": team.title,
            "is_home": tm.h_a == "h",
            "kickoff_reported": tm.date,
            "xg": tm.xG,
            "xga": tm.xGA,
            "npxg": tm.npxG,
            "npxga": tm.npxGA,
            "ppda_passes": tm.ppda.att,
            "ppda_defensive_actions": tm.ppda.def_,
            "ppda_allowed_passes": tm.ppda_allowed.att,
            "ppda_allowed_defensive_actions": tm.ppda_allowed.def_,
            "deep": tm.deep,
            "deep_allowed": tm.deep_allowed,
            "scored": tm.scored,
            "conceded": tm.missed,
            "xpts": tm.xpts,
            "result": tm.result,
        }
        for team in league.teams.values()
        for tm in team.history
    ]
    df = pl.DataFrame(rows, infer_schema_length=None)
    sides = pl.concat(
        [
            matches.select(
                "match_id", "kickoff_reported", pl.col(f"{side}_team_id").alias("team_id")
            ).with_columns(pl.lit(side == "home").alias("is_home"))
            for side in ("home", "away")
        ]
    )
    return df.join(sides, on=["kickoff_reported", "team_id", "is_home"], how="left").select(
        "season", "match_id", pl.exclude("season", "match_id")
    )


def players_table(league: League, season: str) -> pl.DataFrame:
    rows = [
        {"season": season, **p.model_dump(include=set(PlayerSeason.model_fields))}
        for p in league.players
    ]
    return pl.DataFrame(rows, infer_schema_length=None).rename(
        {
            "id": "player_id",
            "xG": "xg",
            "npxG": "npxg",
            "xA": "xa",
            "xGChain": "xg_chain",
            "xGBuildup": "xg_buildup",
            "time": "minutes",
        }
    )
