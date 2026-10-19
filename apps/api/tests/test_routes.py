from datetime import UTC, datetime, timedelta

import polars as pl
from fastapi.testclient import TestClient

from fpl_api.main import create_app
from fpl_api.store import Snapshot, Store, club_ratings, price_table, season_stats

SHAPE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
CLUBS = [f"club{i}" for i in range(10)]
KICKOFF = datetime(2026, 10, 24, 15, tzinfo=UTC)


def _players() -> pl.DataFrame:
    rows, code = [], 1
    for position, n in SHAPE.items():
        for i in range(n * 2):
            rows.append({
                "season": "2026-27", "player_code": code, "element": code,
                "web_name": f"{position}{i}", "position": position,
                "team_key": CLUBS[code % len(CLUBS)], "price": 45 + i,
                "selected_by_percent": 1.5, "status": "a", "news": None,
                "chance_of_playing_next_round": None, "transfers_in_event": 10 * code,
                "transfers_out_event": 5, "points": 10, "minutes": 270, "appearances": 3,
                "starts": 3, "goals": 1, "assists": 0, "xg": 0.8, "xa": 0.2,
                "clean_sheets": 1, "bonus": 2, "saves": 0, "defensive_contribution": 5,
                "form": 3.3, "net_transfers": 10 * code - 5, "p_rise": 0.1, "p_fall": 0.0,
            })  # fmt: skip
            code += 1
    return pl.DataFrame(rows)


def snapshot() -> Snapshot:
    players = _players()
    expected = pl.DataFrame([
        {"player_code": p, "gameweek": gw, "expected_points": float(p % 7) + gw / 10,
         "p_play": 0.9, "p_60": 0.8, "e_goals_scored": 0.2, "e_assists": 0.1,
         "p_clean_sheet": 0.3, "availability": 1.0, "sim_p10": 1.0, "sim_p90": 8.0,
         "p_haul": 0.1}
        for p in players["player_code"] for gw in (8, 9)
    ])  # fmt: skip
    fixtures = pl.DataFrame([
        {"match_key": f"m{gw}{i}", "season": "2026-27", "gameweek": gw,
         "kickoff_utc": KICKOFF + timedelta(days=7 * (gw - 8)),
         "home_team_key": CLUBS[2 * i], "away_team_key": CLUBS[2 * i + 1],
         "lambda_home": 1.5, "lambda_away": 1.0, "home_win": 0.5, "draw": 0.25,
         "away_win": 0.25, "over_2_5": 0.5, "home_clean_sheet": 0.37,
         "away_clean_sheet": 0.22, "source": "ratings"}
        for gw in (8, 9) for i in range(5)
    ])  # fmt: skip
    results = pl.DataFrame({
        "match_key": ["r1"], "gameweek": [7], "kickoff_utc": [KICKOFF - timedelta(days=7)],
        "home_team_key": ["club0"], "away_team_key": ["club1"], "home_goals": [2],
        "away_goals": [1], "home_xg": [1.4], "away_xg": [0.9],
    })  # fmt: skip
    clubs = pl.DataFrame({
        "team_key": CLUBS, "name": [c.title() for c in CLUBS],
        "short_name": [f"C{i}" for i in range(10)], "fpl_team_code": list(range(10)),
        "attack": [1.0 + i / 20 for i in range(10)], "defence": [1.0] * 10,
    })  # fmt: skip
    gameweeks = pl.DataFrame({
        "gameweek": [7, 8, 9], "deadline_utc": [KICKOFF - timedelta(days=d) for d in (8, 1, -6)],
        "finished": [False] * 3, "is_current": [True, False, False],
        "is_next": [False, True, False],
    })  # fmt: skip
    history = pl.DataFrame({
        "season": ["2026-27"], "gameweek": [7], "kickoff_utc": [KICKOFF - timedelta(days=7)],
        "player_code": [1], "element": [1], "opponent_team_key": ["club1"], "is_home": [True],
        "minutes": [90], "total_points": [6], "goals_scored": [0], "assists": [1],
        "expected_goals": [0.1], "expected_assists": [0.4], "bonus": [1], "bps": [20],
        "clean_sheets": [1], "price": [45], "selected": [1000],
    })  # fmt: skip
    past = pl.DataFrame({
        "player_code": [1], "season": ["2025-26"], "points": [120], "minutes": [3000],
        "goals": [0], "assists": [2], "clean_sheets": [12], "max_price": [45],
    })  # fmt: skip
    return Snapshot(
        season="2026-27", gameweeks=gameweeks, clubs=clubs, players=players,
        expected=expected, fixtures=fixtures, results=results, history=history,
        past_seasons=past, predictions_as_of="20261018T000000Z",
    )  # fmt: skip


class FixedStore(Store):
    def __init__(self) -> None:
        self._fixed = snapshot()

    def get(self) -> Snapshot:
        return self._fixed


client = TestClient(create_app(FixedStore()))


def squad() -> list[int]:
    players = _players()
    picks: list[int] = []
    for position, n in SHAPE.items():
        picks += players.filter(pl.col("position") == position)["player_code"].head(n).to_list()
    return picks


def test_meta_reports_next_gameweek_and_horizon():
    meta = client.get("/api/meta").json()
    assert meta["next_gameweek"] == 8
    assert meta["horizon"] == [8, 9]


def test_players_carry_expected_points_per_horizon_gameweek_in_millions():
    players = client.get("/api/players").json()
    first = next(p for p in players if p["code"] == 1)
    assert first["xp"] == [1.8, 1.9]
    assert first["xp_total"] == 3.7
    assert first["price"] == 4.5


def test_player_detail_has_fixtures_with_home_away_and_history():
    detail = client.get("/api/players/1").json()
    assert detail["upcoming"][0]["opponents"] == [{"team_key": "club0", "is_home": False}]
    assert detail["matches"][0]["points"] == 6
    assert detail["past_seasons"][0]["max_price"] == 4.5
    assert client.get("/api/players/9999").status_code == 404


def test_clubs_have_record_and_fixtures_from_their_side():
    clubs = {c["key"]: c for c in client.get("/api/clubs").json()}
    away = clubs["club1"]
    assert away["record"] == {"played": 1, "won": 0, "drawn": 0, "lost": 1, "goals_for": 1,
                              "goals_against": 2, "xg_for": 0.9, "xg_against": 1.4}  # fmt: skip
    first = away["fixtures"][0]
    assert (first["is_home"], first["goals_for"], first["clean_sheet"]) == (False, 1.0, 0.22)


def test_fixtures_list_results_then_upcoming():
    fixtures = client.get("/api/fixtures").json()
    assert fixtures[0]["home_goals"] == 2
    assert fixtures[-1]["lambda_home"] == 1.5


def test_evaluate_picks_a_legal_lineup_and_captain_each_gameweek():
    plan = client.post("/api/squad/evaluate", json={"squad": squad(), "horizon": 2}).json()
    assert [g["gameweek"] for g in plan["gameweeks"]] == [8, 9]
    for week in plan["gameweeks"]:
        assert len(week["lineup"]) == 11
        assert week["captain"] in week["lineup"]
        assert week["transfers_in"] == []


def test_plan_rejects_unknown_players():
    bad = [*squad()[:14], 9999]
    assert client.post("/api/plan", json={"squad": bad}).status_code == 422


def test_derived_tables():
    history = pl.DataFrame({
        "season": ["2026-27"] * 4, "gameweek": [1, 2, 1, 2], "player_code": [1, 1, 2, 2],
        "price": [50, 51, 60, 59], "transfers_balance": [100, 0, -100, 0],
        "total_points": [2, 8, 1, 1], "minutes": [90] * 4, "starts": [1] * 4,
        "goals_scored": [0, 1, 0, 0], "assists": [0] * 4, "expected_goals": [0.1] * 4,
        "expected_assists": [0.0] * 4, "clean_sheets": [0] * 4, "bonus": [0] * 4,
        "saves": [0] * 4, "defensive_contribution": [0] * 4,
    })  # fmt: skip
    changes = price_table(history).sort("player_code", "gameweek")
    assert changes["change"].to_list() == [1, None, -1, None]
    assert changes["crowd_net_transfer_share"].to_list() == [0.5, 0.0, -0.5, 0.0]
    stats = season_stats(history, "2026-27").sort("player_code")
    assert stats["points"].to_list() == [10, 2]
    assert stats["form"].to_list() == [2.5, 0.5]  # mean over the last 4 gameweeks


def test_club_ratings_are_multipliers_around_one():
    kickoffs = [KICKOFF - timedelta(days=d) for d in range(10, 0, -1)]
    matches = pl.DataFrame({
        "kickoff_utc": kickoffs, "home_team_key": ["a", "b"] * 5, "away_team_key": ["b", "a"] * 5,
        "home_goals": [3, 0] * 5, "away_goals": [0, 2] * 5, "home_xg": [2.5, 0.5] * 5,
        "away_xg": [0.4, 1.8] * 5, "home_promoted": [False] * 10, "away_promoted": [False] * 10,
    })  # fmt: skip
    ratings = {r["team_key"]: r for r in club_ratings(matches, KICKOFF).iter_rows(named=True)}
    assert ratings["a"]["attack"] > 1 > ratings["b"]["attack"]
    assert ratings["a"]["defence"] < 1 < ratings["b"]["defence"]
