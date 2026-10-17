import numpy as np
import polars as pl
import pytest

from fpl_models import player_model as pm
from fpl_models import simulator as sim


def fixture(n_per_team=11, lam_home=1.8, lam_away=1.0):
    """One match, two teams of 11, every player a nailed starter."""
    rows, events = [], []
    for team, opp, lam in (("h", "a", lam_home), ("a", "h", lam_away)):
        for i in range(n_per_team):
            position = "GKP" if i == 0 else "DEF" if i < 5 else "MID" if i < 9 else "FWD"
            share = {"GKP": 0.0, "DEF": 0.03, "MID": 0.1, "FWD": 0.2}[position]
            rows.append({"match_key": "m", "team_key": team, "opponent_team_key": opp,
                         "position": position, "team_lambda": lam,
                         "opponent_lambda": lam_away if team == "h" else lam_home,
                         "season": "2025-26"})  # fmt: skip
            events.append({"p_play": 1.0, "p_60": 1.0, "p_1_59": 0.0,
                           "e_goals_scored": share * lam, "e_assists": share * lam * 0.7,
                           "e_bonus": 0.4, "e_saves": 3.0 if position == "GKP" else 0.0,
                           "e_yellow_cards": 0.1, "p_defcon": 0.1})  # fmt: skip
    return pl.DataFrame(rows), pl.DataFrame(events)


def test_players_goals_add_up_to_team_goals_in_every_draw():
    rows, events = fixture()
    result = sim.simulate(rows, events, sim.SimulationSettings(draws=500))
    for column in np.unique(result.team_index):
        players = result.goals[:, result.team_index == column].sum(axis=1)
        assert np.array_equal(players + result.own_goals[:, column], result.team_goals[:, column])


def test_clean_sheet_probability_matches_poisson_for_nailed_defenders():
    rows, events = fixture()
    result = sim.simulate(rows, events, sim.SimulationSettings(draws=20000, own_goal_rate=0.0))
    # Home goalkeeper: clean sheet when the away side scores none.
    away_goals = result.team_goals[:, result.team_index[11]]
    expected_lambda = events.filter(rows["team_key"] == "a")["e_goals_scored"].sum()
    assert (away_goals == 0).mean() == pytest.approx(np.exp(-expected_lambda), abs=0.02)


def test_simulated_means_match_the_analytic_expected_points():
    rows, events = fixture()
    result = sim.simulate(rows, events, sim.SimulationSettings(draws=20000))
    analytic = pm.expected_points(rows, events).to_numpy()
    assert result.points.mean(axis=0) == pytest.approx(analytic, abs=0.25)


def test_teammates_returns_are_positively_correlated():
    rows, events = fixture()
    p = sim.simulate(rows, events, sim.SimulationSettings(draws=20000)).points
    forwards = [9, 10]
    assert np.corrcoef(p[:, forwards[0]], p[:, forwards[1]])[0, 1] > 0.02
    # A home attacker and the away goalkeeper move in opposite directions.
    assert np.corrcoef(p[:, 9], p[:, 11])[0, 1] < 0


def test_summary_has_quantiles_in_order():
    rows, events = fixture()
    s = sim.simulate(rows, events, sim.SimulationSettings(draws=2000)).summary()
    assert (s["sim_p10"] <= s["sim_p50"]).all()
    assert (s["sim_p50"] <= s["sim_p90"]).all()
