"""Understat parsing and ingestion, with a synthetic season (the data isn't ours to commit)."""

import json
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path

import dagster as dg
import httpx
import polars as pl
import pytest
from pydantic import ValidationError

from fpl_core.http import RetryingHttpClient
from fpl_pipelines.assets import understat as assets
from fpl_pipelines.lake import Lake, PartitionedParquetIOManager
from fpl_pipelines.sources import understat as src

SEASON = "2020-21"
TEAMS = {i: f"Team {i}" for i in range(71, 91)}


def team_entry(is_home, kickoff, goals, xgs):
    (scored, conceded), (xg, xga) = goals, xgs
    return {
        "h_a": "h" if is_home else "a", "xG": xg, "xGA": xga, "npxG": xg, "npxGA": xga,
        "ppda": {"att": 200, "def": 20}, "ppda_allowed": {"att": 150, "def": 25},
        "deep": 5, "deep_allowed": 4, "scored": scored, "missed": conceded, "xpts": 1.5,
        "result": "w" if scored > conceded else "l" if scored < conceded else "d",
        "date": kickoff, "wins": 0, "draws": 0, "loses": 0, "pts": 0,
    }  # fmt: skip


def league_payload(played=380):
    """A double round-robin; the first `played` fixtures have results."""
    start = datetime(2020, 9, 12, 14, 0)
    dates, history = [], {t: [] for t in TEAMS}
    for i, (home, away) in enumerate(permutations(TEAMS, 2)):
        kickoff = (start + timedelta(hours=3 * i)).strftime("%Y-%m-%d %H:%M:%S")
        done = i < played
        dates.append({
            "id": str(10000 + i), "isResult": done,
            "h": {"id": str(home), "title": TEAMS[home], "short_title": "H"},
            "a": {"id": str(away), "title": TEAMS[away], "short_title": "A"},
            "goals": {"h": "2" if done else None, "a": "1" if done else None},
            "xG": {"h": "1.5" if done else None, "a": "0.75" if done else None},
            "datetime": kickoff,
            "forecast": {"w": "0.5", "d": "0.3", "l": "0.2"} if done else None,
        })  # fmt: skip
        if done:
            history[home].append(team_entry(True, kickoff, (2, 1), (1.5, 0.75)))
            history[away].append(team_entry(False, kickoff, (1, 2), (0.75, 1.5)))
    teams = {str(t): {"id": str(t), "title": TEAMS[t], "history": h} for t, h in history.items()}
    players = [{
        "id": "1250", "player_name": "A Player", "games": "38", "time": "3392", "goals": "29",
        "xG": "27.7", "assists": "18", "xA": "15.8", "shots": "130", "key_passes": "89",
        "yellow_cards": "1", "red_cards": "0", "position": "F M", "team_title": "Team 71",
        "npg": "20", "npxG": "20.8", "xGChain": "48.5", "xGBuildup": "16.2",
    }]  # fmt: skip
    return {"teams": teams, "players": players, "dates": dates}


# --- source ---------------------------------------------------------------------------


def test_league_path_uses_the_start_year():
    assert src.league_path("2025-26") == "/getLeagueData/EPL/2025"


def test_tables_parse_and_link_team_logs_to_matches():
    league = src.League.model_validate(league_payload())
    matches = src.matches_table(league, SEASON)
    teams = src.team_matches_table(league, SEASON, matches)
    assert matches.height == 380
    assert teams.height == 760
    assert teams["match_id"].null_count() == 0
    assert teams.select("ppda_passes", "ppda_defensive_actions").row(0) == (200, 20)
    assert matches["match_date"].dtype == pl.Date


def test_unplayed_matches_have_no_score_or_forecast():
    league = src.League.model_validate(league_payload(played=10))
    matches = src.matches_table(league, SEASON).filter(~pl.col("is_result"))
    assert matches["home_goals"].null_count() == matches.height
    assert matches["forecast_home_win"].null_count() == matches.height


def test_players_table_renames_to_snake_case():
    players = src.players_table(src.League.model_validate(league_payload()), SEASON)
    assert {"player_id", "xg", "npxg", "xa", "xg_chain", "minutes"} <= set(players.columns)


def test_broken_payload_is_rejected():
    payload = league_payload()
    del payload["dates"][0]["xG"]
    with pytest.raises(ValidationError):
        src.League.model_validate(payload)


# --- ingestion ------------------------------------------------------------------------

PAYLOAD: dict[str, bytes] = {}


class FakeUnderstat(assets.UnderstatResource):
    def client(self, season):
        def handler(request):
            assert request.headers["X-Requested-With"] == "XMLHttpRequest"
            return httpx.Response(200, content=PAYLOAD["body"])

        return RetryingHttpClient(
            "https://understat.test",
            headers=src.request_headers(season),
            transport=httpx.MockTransport(handler),
        )


@pytest.fixture
def lake(tmp_path):
    PAYLOAD["body"] = json.dumps(league_payload()).encode()
    return Lake(base_url=str(tmp_path))


def run(lake):
    return dg.materialize(
        [*dg.load_assets_from_modules([assets]), *dg.load_asset_checks_from_modules([assets])],
        partition_key=SEASON,
        resources={
            "understat": FakeUnderstat(),
            "lake": lake,
            "parquet_io_manager": PartitionedParquetIOManager(lake=lake),
        },
        raise_on_error=False,
    )


def checks(result):
    return {c.check_name: c.passed for c in result.get_asset_check_evaluations()}


def test_season_ingests_and_passes_checks(lake):
    result = run(lake)
    assert result.success
    assert checks(result) == {"complete_fixture_list": True, "team_logs_agree_with_matches": True}
    base = Path(lake.base_url) / "silver/understat"
    assert pl.read_parquet(base / f"team_matches/season={SEASON}/data.parquet").height == 760


def test_unchanged_payload_is_stored_once(lake):
    run(lake)
    run(lake)
    assert len(lake.list(assets.season_prefix(SEASON))) == 1


def test_team_log_disagreeing_with_match_xg_is_blocked(lake):
    payload = league_payload()
    first_team = next(iter(payload["teams"].values()))
    first_team["history"][0]["xG"] = 9.9
    PAYLOAD["body"] = json.dumps(payload).encode()
    assert checks(run(lake))["team_logs_agree_with_matches"] is False


def test_finished_season_with_unplayed_matches_is_blocked(lake):
    PAYLOAD["body"] = json.dumps(league_payload(played=300)).encode()
    assert checks(run(lake))["complete_fixture_list"] is False
