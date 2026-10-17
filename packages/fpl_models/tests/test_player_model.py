from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.stats import poisson

from fpl_core import scoring
from fpl_models import player_features as pf
from fpl_models import player_model as pm

START = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)


def fixture_rows(n_players=6, gameweeks=8, season="2024-25", double_gw=4):
    """Synthetic played fixture rows; gameweek `double_gw` has two fixtures."""
    rng = np.random.default_rng(0)
    rows = []
    for gw in range(1, gameweeks + 1):
        fixtures = 2 if gw == double_gw else 1
        for f in range(fixtures):
            kickoff = START + timedelta(days=7 * gw + 2 * f)
            for p in range(n_players):
                minutes = int(rng.choice([0, 30, 90]))
                rows.append({
                    "season": season, "gameweek": gw, "kickoff_utc": kickoff,
                    "match_key": f"{season}-{gw}-{f}-{p % 2}", "player_code": p,
                    "position": pf.POSITIONS[p % 4], "team_key": f"t{p % 2}", "is_home": p % 2 == 0,
                    "price": 50 + p, "is_provisional": False, "minutes": minutes,
                    "starts": int(minutes >= 60), "goals_scored": int(rng.random() < 0.2),
                    "assists": int(rng.random() < 0.2), "bonus": int(rng.integers(0, 4)),
                    "bps": int(rng.integers(0, 40)), "saves": int(rng.integers(0, 5)),
                    "yellow_cards": int(rng.random() < 0.1),
                    "total_points": int(rng.integers(0, 12)),
                    "expected_goals": float(rng.random()), "expected_assists": float(rng.random()),
                    "tackles": int(rng.integers(0, 6)),
                    "clearances_blocks_interceptions": int(rng.integers(0, 8)),
                    "recoveries": int(rng.integers(0, 8)),
                })  # fmt: skip
    return pf.add_deadlines(pl.DataFrame(rows))


def lambdas(rows):
    keys = rows["match_key"].unique()
    return pl.DataFrame({"match_key": keys, "lambda_home": 1.5, "lambda_away": 1.1})


# --- features: the leakage proof ------------------------------------------------------


@pytest.mark.parametrize("cutoff", [2, 4, 5, 8])
def test_features_do_not_change_when_later_data_is_removed(cutoff):
    rows = fixture_rows()
    full = pf.build_features(rows, lambdas(rows))
    truncated_rows = rows.filter(pl.col("gameweek") <= cutoff)
    truncated = pf.build_features(truncated_rows, lambdas(truncated_rows))
    columns = ["player_code", "match_key", *pf.FEATURES]
    expected = (
        full.filter(pl.col("gameweek") == cutoff).select(columns).sort("player_code", "match_key")
    )
    actual = (
        truncated.filter(pl.col("gameweek") == cutoff)
        .select(columns)
        .sort("player_code", "match_key")
    )
    assert expected.equals(actual)


def test_gameweek_features_ignore_that_gameweeks_results():
    rows = fixture_rows()
    base = pf.build_features(rows, lambdas(rows)).filter(pl.col("gameweek") == 5)
    changed_rows = rows.with_columns(
        pl.when(pl.col("gameweek") == 5)
        .then(99)
        .otherwise(pl.col("goals_scored"))
        .alias("goals_scored")
    )
    changed = pf.build_features(changed_rows, lambdas(changed_rows)).filter(pl.col("gameweek") == 5)
    assert base.select(pf.FEATURES).equals(changed.select(pf.FEATURES))


def test_both_double_gameweek_fixtures_share_pre_deadline_state():
    rows = fixture_rows(double_gw=4)
    gw4 = pf.build_features(rows, lambdas(rows)).filter(pl.col("gameweek") == 4)
    per_player = gw4.group_by("player_code").agg(pl.col("minutes_l1").n_unique())
    assert (per_player["minutes_l1"] == 1).all()
    assert (gw4["n_fixtures"] == 2).all()


def test_first_gameweek_has_no_history():
    rows = fixture_rows()
    first = pf.build_features(rows, lambdas(rows)).filter(pl.col("gameweek") == 1)
    assert first["minutes_l3"].null_count() == first.height


def test_latest_state_is_the_input_for_the_next_gameweek():
    rows = fixture_rows()
    upcoming = pf.build_features(rows, lambdas(rows)).filter(pl.col("gameweek") == 8)
    state = pf.latest_player_state(rows.filter(pl.col("gameweek") <= 7))
    joined = (
        upcoming.select("player_code", "minutes_l3")
        .unique()
        .join(state.select("player_code", pl.col("minutes_l3").alias("state")), on="player_code")
    )
    assert joined["minutes_l3"].to_list() == pytest.approx(joined["state"].to_list())


def test_defensive_contribution_label_uses_position_thresholds():
    rows = pl.DataFrame({
        "position": ["DEF", "DEF", "MID", "MID", "GKP"], "tackles": [5, 5, 4, 4, 9],
        "clearances_blocks_interceptions": [5, 4, 4, 4, 9], "recoveries": [0, 9, 4, 3, 9],
    })  # fmt: skip
    hit = pf._with_defensive_stats(rows)["defcon_hit"].to_list()
    assert hit == [True, False, True, False, False]


# --- points assembly ------------------------------------------------------------------


def test_expected_floor_div_matches_poisson():
    mean = np.array([1.3])
    k = np.arange(60)
    assert pm.expected_floor_div(mean, 2)[0] == pytest.approx(
        (poisson.pmf(k, 1.3) * (k // 2)).sum()
    )


def events(**overrides):
    base = {"p_play": 1.0, "p_60": 1.0, "p_1_59": 0.0, "e_goals_scored": 0.0, "e_assists": 0.0,
            "e_bonus": 0.0, "e_saves": 0.0, "e_yellow_cards": 0.0, "p_defcon": 0.0}  # fmt: skip
    return pl.DataFrame([base | overrides])


@pytest.mark.parametrize(
    ("position", "goal_points"), [("GKP", 10), ("DEF", 6), ("MID", 5), ("FWD", 4)]
)
def test_a_certain_goal_is_worth_the_positions_goal_points(position, goal_points):
    rows = pl.DataFrame({"season": ["2025-26"], "position": [position], "opponent_lambda": [50.0]})
    without = pm.expected_points(rows, events())[0]
    with_goal = pm.expected_points(rows, events(e_goals_scored=1.0))[0]
    assert with_goal - without == pytest.approx(goal_points)


def test_clean_sheet_value_follows_the_opponents_expected_goals():
    rows = pl.DataFrame({"season": ["2025-26"], "position": ["DEF"], "opponent_lambda": [1.0]})
    points = pm.expected_points(rows, events())[0]
    expected = (
        2
        + np.exp(-1.0) * scoring.CLEAN_SHEET_POINTS[scoring.Position.DEF]
        - pm.expected_floor_div(np.array([1.0]), 2)[0]
    )
    assert points == pytest.approx(expected)


def test_defensive_contribution_only_counts_under_its_rules():
    rows = pl.DataFrame(
        {
            "season": ["2024-25", "2025-26"],
            "position": ["DEF", "DEF"],
            "opponent_lambda": [50.0, 50.0],
        }
    )
    both = pl.concat([events(p_defcon=0.5), events(p_defcon=0.5)])
    points = pm.expected_points(rows, both).to_list()
    assert points[1] - points[0] == pytest.approx(0.5 * scoring.DEFENSIVE_CONTRIBUTION_POINTS)


# --- model ----------------------------------------------------------------------------


def test_masked_estimator_survives_an_all_missing_column():
    x = np.column_stack([np.arange(50.0), np.full(50, np.nan)])
    model = pm.Masked(pm._regressor()).fit(x, np.arange(50) % 3)
    prediction = model.predict(np.array([[3.0, 7.0]]))
    assert np.isfinite(prediction).all()


def test_component_model_trains_only_on_earlier_deadlines():
    rows = fixture_rows(n_players=8, gameweeks=10)
    features = pf.build_features(rows, lambdas(rows))
    model = pm.ComponentModel(features, refit_every=100)
    targets = (
        features.filter(pl.col("gameweek") == 9)
        .select("season", "gameweek", "deadline_utc", "player_code")
        .unique("player_code")
    )
    seen = {}
    original = pm.fit_components

    def spy(train, defcon_train):
        seen["max"] = train["deadline_utc"].max()
        return original(train, defcon_train)

    pm.fit_components = spy
    try:
        predicted = model.predict(features.filter(pl.col("gameweek") < 9), targets)
    finally:
        pm.fit_components = original
    assert seen["max"] < targets["deadline_utc"][0]
    assert predicted.len() == targets.height


# --- live predictions -----------------------------------------------------------------


def squads():
    return pl.DataFrame({
        "player_code": [1, 2, 3], "web_name": ["A", "B", "C"], "position": ["MID", "DEF", "FWD"],
        "team_key": ["home_fc", "away_fc", "away_fc"], "price": [80, 50, 60],
        "status": ["a", "d", "u"], "chance_of_playing_next_round": [None, 50, 0],
    })  # fmt: skip


def fixture_expectations():
    return pl.DataFrame({
        "match_key": ["m1", "m2"], "season": ["2026-27", "2026-27"], "gameweek": [9, 10],
        "kickoff_utc": [START, START + timedelta(days=7)],
        "home_team_key": ["home_fc", "away_fc"], "away_team_key": ["away_fc", "home_fc"],
        "lambda_home": [2.0, 1.2], "lambda_away": [0.8, 1.4],
    })  # fmt: skip


def test_upcoming_rows_take_each_sides_lambdas():
    rows = pm.upcoming_fixture_rows(squads(), fixture_expectations(), [9])
    home = rows.filter(pl.col("player_code") == 1).row(0, named=True)
    away = rows.filter(pl.col("player_code") == 2).row(0, named=True)
    assert (home["is_home"], home["team_lambda"], home["opponent_lambda"]) == (True, 2.0, 0.8)
    assert (away["is_home"], away["team_lambda"], away["opponent_lambda"]) == (False, 0.8, 2.0)
    assert home["p_clean_sheet"] == pytest.approx(np.exp(-0.8))


def test_availability_scales_next_gameweek_and_zeroes_departed_players():
    rows = pm.upcoming_fixture_rows(squads(), fixture_expectations(), [9, 10])
    factor = rows.with_columns(pm.availability(rows, next_gameweek=9))
    by = {
        (r["player_code"], r["gameweek"]): r["availability"] for r in factor.iter_rows(named=True)
    }
    assert by[(1, 9)] == by[(1, 10)] == 1.0  # available
    assert (by[(2, 9)], by[(2, 10)]) == (0.5, 1.0)  # doubtful: next gameweek only
    assert by[(3, 9)] == by[(3, 10)] == 0.0  # unavailable (left the club)
