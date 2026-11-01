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
from fpl_pipelines.snapshots import current_season
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


def roster_entry(player_id, team_id, h_a, **stats):
    goals, xg, minutes = stats.get("goals", 0), stats.get("xg", 0.1), stats.get("minutes", 90)
    return {
        "id": str(player_id * 10), "player_id": str(player_id), "player": f"P{player_id}",
        "team_id": str(team_id), "h_a": h_a, "position": "FW", "time": str(minutes),
        "goals": str(goals), "own_goals": "0", "shots": "2", "key_passes": "1",
        "assists": "0", "xG": str(xg), "xA": "0.2", "xGChain": "0.4", "xGBuildup": "0.1",
        "yellow_card": "0", "red_card": "0", "roster_in": "0", "roster_out": "0",
        "positionOrder": "1",
    }  # fmt: skip


def match_payload(home=71, away=72, home_goals=2):
    """A 2-1 (by default): the home striker scores twice, once from a 0.76 xG penalty."""
    rosters = {
        "h": {"1": roster_entry(1, home, "h", goals=home_goals, xg=1.26),
              "2": roster_entry(2, home, "h", minutes=20)},
        "a": {"3": roster_entry(3, away, "a", goals=1, xg=0.75)},
    }  # fmt: skip
    shot = {"id": "9", "minute": "10", "X": "0.9", "Y": "0.5", "player": "P1", "h_a": "h",
            "player_id": "1", "season": "2020", "shotType": "RightFoot", "match_id": "1",
            "result": "Goal", "lastAction": "None"}  # fmt: skip
    shots = {
        "h": [
            {**shot, "xG": "0.76", "situation": "Penalty"},
            {**shot, "xG": "0.5", "situation": "OpenPlay"},
        ],
        "a": [{**shot, "player_id": "3", "h_a": "a", "xG": "0.75", "situation": "OpenPlay"}],
    }
    return {"rosters": rosters, "shots": shots, "tmpl": {"home": "<div>", "away": "<div>"}}


# --- source ---------------------------------------------------------------------------


def test_match_payload_drops_the_page_html_and_splits_penalty_xg():
    stripped = src.strip_match_payload(json.dumps(match_payload()).encode())
    assert "tmpl" not in stripped
    matches = pl.DataFrame({"match_id": [10000], "match_date": [datetime(2020, 9, 12).date()]})
    table = src.player_matches_table({"10000": stripped}, SEASON, matches)
    assert table.height == 3
    striker = table.filter(pl.col("understat_player_id") == 1).row(0, named=True)
    assert striker["penalty_xg"] == pytest.approx(0.76)
    assert striker["npxg"] == pytest.approx(0.5)
    assert striker["match_date"] == datetime(2020, 9, 12).date()
    assert table["is_home"].to_list() == [True, True, False]


def test_player_matches_table_is_typed_when_empty():
    table = src.player_matches_table({}, SEASON, pl.DataFrame({"match_id": [], "match_date": []}))
    assert table.height == 0
    assert table.schema["xg"] == pl.Float64


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
REQUESTS: list[str] = []


class FakeUnderstat(assets.UnderstatResource):
    def client(self, season):
        def handler(request):
            assert request.headers["X-Requested-With"] == "XMLHttpRequest"
            if request.url.path.startswith("/getMatchData/"):
                REQUESTS.append(request.url.path)
                return httpx.Response(200, content=PAYLOAD["match"])
            return httpx.Response(200, content=PAYLOAD["body"])

        return RetryingHttpClient(
            "https://understat.test",
            headers=src.request_headers(season),
            transport=httpx.MockTransport(handler),
        )


@pytest.fixture
def lake(tmp_path):
    PAYLOAD["body"] = json.dumps(league_payload()).encode()
    PAYLOAD["match"] = json.dumps(match_payload()).encode()
    REQUESTS.clear()
    return Lake(base_url=str(tmp_path))


def run(lake, season=SEASON):
    return dg.materialize(
        [*dg.load_assets_from_modules([assets]), *dg.load_asset_checks_from_modules([assets])],
        partition_key=season,
        resources={
            "understat": FakeUnderstat(request_interval_seconds=0),
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
    assert checks(result) == {
        "complete_fixture_list": True,
        "team_logs_agree_with_matches": True,
        "rosters_agree_with_scores": True,
    }
    base = Path(lake.base_url) / "silver/understat"
    assert pl.read_parquet(base / f"team_matches/season={SEASON}/data.parquet").height == 760
    players = pl.read_parquet(base / f"player_matches/season={SEASON}/data.parquet")
    assert players.height == 3 * 380
    assert len(REQUESTS) == 380


def test_match_data_fetches_only_new_matches(lake):
    season = current_season()  # a season in progress may be partly played
    PAYLOAD["body"] = json.dumps(league_payload(played=100)).encode()
    assert run(lake, season).success
    assert len(REQUESTS) == 100
    PAYLOAD["body"] = json.dumps(league_payload(played=110)).encode()
    assert run(lake, season).success
    assert len(REQUESTS) == 110  # ten new matches, ten requests
    assert len(lake.list(assets.match_data_prefix(season))) == 2


def test_rosters_disagreeing_with_the_score_are_blocked(lake):
    PAYLOAD["match"] = json.dumps(match_payload(home_goals=3)).encode()
    assert checks(run(lake))["rosters_agree_with_scores"] is False


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
