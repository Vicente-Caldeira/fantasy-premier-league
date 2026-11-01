"""Understat: team-level expected goals for every Premier League match since 2014-15.

`/getLeagueData/EPL/<year>` returns, per season:
    dates    every match: result, xG per side, Understat's win/draw/loss forecast
    teams    each team's match log: xG/xGA, non-penalty xG, PPDA, deep completions, xPts
    players  season totals per player

`/getMatchData/<match_id>` returns one match's rosters (per player: minutes, goals,
shots, key passes, xG, xA, xGChain, xGBuildup) and its shots (with xG and situation,
so penalty xG is known per player-match). FPL's own xG starts in 2022-23; this goes
back to 2014-15.

Kickoff times are NOT reliable: in 2025-26, 343 of 380 matched UTC, 31 (May) were an
hour ahead and 6 on the October clock-change day an hour behind. The match *date* is
reliable, so it is the join key, and the time is kept only as `kickoff_reported`.

Understat has no official API or published terms for reuse: requests are infrequent
(weekly for the current season) and the data is never committed.
"""

import json
from datetime import datetime

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

BASE_URL = "https://understat.com"
FIRST_SEASON = 2014


def league_path(season: str) -> str:
    """`2025-26` -> `/getLeagueData/EPL/2025` (Understat names seasons by start year)."""
    return f"/getLeagueData/EPL/{season[:4]}"


def match_path(match_id: int) -> str:
    return f"/getMatchData/{match_id}"


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


class RosterEntry(Contract):
    """One player's line in a match: Understat sends every number as a string."""

    player_id: int
    player: str
    team_id: int
    h_a: str
    position: str
    time: int
    goals: int
    own_goals: int
    shots: int
    key_passes: int
    assists: int
    xG: float  # noqa: N815
    xA: float  # noqa: N815
    xGChain: float  # noqa: N815
    xGBuildup: float  # noqa: N815
    yellow_card: int
    red_card: int


class Shot(Contract):
    player_id: int
    xG: float  # noqa: N815
    situation: str  # OpenPlay, FromCorner, SetPiece, DirectFreekick, Penalty
    result: str


class Sides(Contract):
    h: dict[str, RosterEntry]
    a: dict[str, RosterEntry]


class ShotSides(Contract):
    h: list[Shot]
    a: list[Shot]


class MatchData(Contract):
    rosters: Sides
    shots: ShotSides


def strip_match_payload(data: bytes) -> dict[str, object]:
    """Validate one `/getMatchData` response and drop `tmpl`, the page's rendered HTML
    (pure presentation, most of the bytes)."""
    MatchData.model_validate_json(data)  # reject a broken payload before storing it
    payload: dict[str, object] = json.loads(data)
    payload.pop("tmpl", None)
    return payload


def player_matches_table(
    payloads: dict[str, dict[str, object]], season: str, matches: pl.DataFrame
) -> pl.DataFrame:
    """One row per player per match (`payloads`: match id -> stripped match data).
    Non-penalty xG is xG minus the xG of the player's penalty shots in that match."""
    dates = dict(zip(matches["match_id"], matches["match_date"], strict=True))
    rows = []
    for match_id, raw in payloads.items():
        data = MatchData.model_validate(raw)
        penalty_xg: dict[int, float] = {}
        for shot in (*data.shots.h, *data.shots.a):
            if shot.situation == "Penalty":
                penalty_xg[shot.player_id] = penalty_xg.get(shot.player_id, 0.0) + shot.xG
        for entry in (*data.rosters.h.values(), *data.rosters.a.values()):
            pen = penalty_xg.get(entry.player_id, 0.0)
            rows.append({
                "season": season, "match_id": int(match_id), "match_date": dates.get(int(match_id)),
                "understat_player_id": entry.player_id, "player_name": entry.player,
                "team_id": entry.team_id, "is_home": entry.h_a == "h", "position": entry.position,
                "minutes": entry.time, "goals": entry.goals, "own_goals": entry.own_goals,
                "shots": entry.shots, "key_passes": entry.key_passes, "assists": entry.assists,
                "xg": entry.xG, "npxg": entry.xG - pen, "penalty_xg": pen, "xa": entry.xA,
                "xg_chain": entry.xGChain, "xg_buildup": entry.xGBuildup,
                "yellow_cards": entry.yellow_card, "red_cards": entry.red_card,
            })  # fmt: skip
    return pl.DataFrame(rows, schema=PLAYER_MATCH_SCHEMA).sort(
        "match_id", "is_home", "understat_player_id", descending=[False, True, False]
    )


PLAYER_MATCH_SCHEMA = {
    "season": pl.String, "match_id": pl.Int64, "match_date": pl.Date,
    "understat_player_id": pl.Int64, "player_name": pl.String, "team_id": pl.Int64,
    "is_home": pl.Boolean, "position": pl.String, "minutes": pl.Int64, "goals": pl.Int64,
    "own_goals": pl.Int64, "shots": pl.Int64, "key_passes": pl.Int64, "assists": pl.Int64,
    "xg": pl.Float64, "npxg": pl.Float64, "penalty_xg": pl.Float64, "xa": pl.Float64,
    "xg_chain": pl.Float64, "xg_buildup": pl.Float64, "yellow_cards": pl.Int64,
    "red_cards": pl.Int64,
}  # fmt: skip
