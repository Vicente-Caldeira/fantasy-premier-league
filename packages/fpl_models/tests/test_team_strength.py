from datetime import UTC, datetime, timedelta
from itertools import permutations

import numpy as np
import polars as pl
import pytest

from fpl_models import team_strength as ts

START = datetime(2022, 8, 6, 14, 0, tzinfo=UTC)
TEAMS = ["strong", "mid_a", "mid_b", "weak"]


def season_matches(season, start, goals, promoted=()):
    """Double round-robin, 2 matches per gameweek; `goals(home, away)` -> (hg, ag)."""
    rows = []
    for i, (home, away) in enumerate(permutations(TEAMS, 2)):
        hg, ag = goals(home, away)
        rows.append({
            "match_key": f"{season}-{home}-{away}", "season": season, "gameweek": i // 2 + 1,
            "kickoff_utc": start + timedelta(days=7 * (i // 2)), "home_team_key": home,
            "away_team_key": away, "is_played": True, "home_goals": hg, "away_goals": ag,
            "home_xg": float(hg), "away_xg": float(ag), "p_home_win": None, "p_draw": None,
            "p_away_win": None, "p_over_2_5": None,
            "home_promoted": home in promoted, "away_promoted": away in promoted,
        })  # fmt: skip
    return pl.DataFrame(
        rows,
        schema_overrides={
            c: pl.Float64 for c in ("p_home_win", "p_draw", "p_away_win", "p_over_2_5")
        },
    )


def strong_beats_everyone(home, away):
    if home == "strong":
        return 3, 0
    if away == "strong":
        return 0, 3
    return 1, 1


@pytest.fixture
def matches():
    seasons = [
        season_matches("2022-23", START, strong_beats_everyone),
        season_matches("2023-24", START + timedelta(days=365), strong_beats_everyone),
    ]
    return ts.with_deadlines(pl.concat(seasons))


# --- probabilities --------------------------------------------------------------------


def test_score_matrix_is_a_distribution():
    assert ts.score_matrix(1.6, 1.1).sum() == pytest.approx(1.0, abs=1e-9)
    assert ts.score_matrix(6.0, 6.0).sum() == pytest.approx(1.0, abs=1e-8)


def test_clean_sheet_is_probability_opponent_scores_zero():
    p = ts.outcome_probabilities(1.4, 0.9)
    assert p["home_clean_sheet"] == pytest.approx(np.exp(-0.9))
    assert p["home_win"] + p["draw"] + p["away_win"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(("lh", "la"), [(1.5, 1.1), (2.6, 0.6), (0.8, 1.9)])
def test_odds_inversion_recovers_the_lambdas(lh, la):
    p = ts.outcome_probabilities(lh, la)
    recovered = ts.lambdas_from_odds(p["home_win"], p["draw"], p["away_win"], p["over_2_5"])
    assert recovered == pytest.approx((lh, la), abs=1e-3)


# --- models ---------------------------------------------------------------------------


def test_ratings_rank_the_strong_team_first(matches):
    fitted = ts.Ratings(l2=0.1).fit(matches, START + timedelta(days=800))
    assert max(fitted.attack, key=fitted.attack.get) == "strong"
    assert max(fitted.defence, key=fitted.defence.get) == "strong"


def test_ratings_learn_a_promoted_penalty():
    def promoted_lose(home, away):
        return (0, 2) if home == "weak" else (2, 0) if away == "weak" else (1, 1)

    history = pl.concat(
        [
            season_matches(
                f"{2018 + i}-{19 + i}",
                START - timedelta(days=365 * (4 - i)),
                promoted_lose,
                promoted={"weak"},
            )
            for i in range(4)
        ]
    )
    fitted = ts.Ratings(l2=1e3, lookback_days=5 * 365).fit(history, START)
    assert fitted.promoted_attack < 0
    assert fitted.promoted_defence < 0  # conceding more = lower defence


def test_unseen_team_gets_average_strength(matches):
    target = matches.head(1).with_columns(home_team_key=pl.lit("newcomer"))
    as_of = START + timedelta(days=800)
    lambdas = ts.Ratings().fit(matches, as_of).lambdas(target.select(ts.CONTEXT_COLUMNS))
    assert lambdas["lambda_home"][0] > 0


def test_market_uses_odds_and_falls_back_without_them(matches):
    p = ts.outcome_probabilities(2.2, 0.7)
    priced = matches.head(2).with_columns(
        p_home_win=pl.Series([p["home_win"], None]),
        p_draw=pl.Series([p["draw"], None]),
        p_away_win=pl.Series([p["away_win"], None]),
        p_over_2_5=pl.Series([p["over_2_5"], None]),
    )
    lambdas = ts.MarketOdds().predict(matches, priced.select(ts.CONTEXT_COLUMNS))
    assert lambdas.row(0) == pytest.approx((2.2, 0.7), abs=1e-3)
    league = ts.LeagueAverage().predict(matches, priced.select(ts.CONTEXT_COLUMNS))
    assert lambdas.row(1) == league.row(1)


def test_production_model_prefers_market_and_falls_back_to_ratings(matches):
    p = ts.outcome_probabilities(2.2, 0.7)
    targets = matches.head(2).with_columns(
        p_home_win=pl.Series([p["home_win"], None]), p_draw=pl.Series([p["draw"], None]),
        p_away_win=pl.Series([p["away_win"], None]), p_over_2_5=pl.Series([p["over_2_5"], None]),
        deadline_utc=pl.lit(START + timedelta(days=800)),
    ).select(ts.CONTEXT_COLUMNS)  # fmt: skip
    combined = ts.MarketWithRatingsFallback().predict(matches, targets)
    rated = ts.Ratings().predict(matches, targets)
    assert combined.row(0) == pytest.approx((2.2, 0.7), abs=1e-3)
    assert combined.row(1) == pytest.approx(rated.row(1))


def test_blend_is_the_geometric_mean(matches):
    class Fixed:
        def __init__(self, value):
            self.value, self.name = value, f"fixed{value}"

        def predict(self, history, targets):
            return pl.DataFrame(
                {
                    "lambda_home": [self.value] * targets.height,
                    "lambda_away": [1.0] * targets.height,
                }
            )

    blended = ts.Blend(Fixed(4.0), Fixed(1.0)).predict(matches, matches.head(1))
    assert blended["lambda_home"][0] == pytest.approx(2.0)


# --- backtest -------------------------------------------------------------------------


class Spy:
    name = "spy"

    def __init__(self):
        self.calls = []

    def predict(self, history, targets):
        self.calls.append((history, targets))
        return pl.DataFrame(
            {"lambda_home": [1.0] * targets.height, "lambda_away": [1.0] * targets.height}
        )


def test_history_is_strictly_before_the_deadline(matches):
    spy = Spy()
    ts.run_match_backtest(spy, matches, ["2023-24"])
    for history, targets in spy.calls:
        assert history["kickoff_utc"].max() < targets["deadline_utc"][0]
        assert not set(ts.OUTCOME_COLUMNS) & set(targets.columns)


def test_horizon_moves_the_cutoff_back_and_withholds_odds(matches):
    spy = Spy()
    ts.run_match_backtest(spy, matches.with_columns(p_home_win=pl.lit(0.5)), ["2023-24"], horizon=3)
    history, targets = spy.calls[5]
    gameweek = targets["gameweek"][0]
    assert history.filter(pl.col("season") == "2023-24")["gameweek"].max() == gameweek - 4
    assert targets["p_home_win"].null_count() == targets.height


def test_evaluation_on_a_certain_outcome():
    predictions = pl.DataFrame(
        {"lambda_home": [3.0], "lambda_away": [0.2], "home_goals": [3], "away_goals": [0]}
    )
    metrics = ts.evaluate_matches(predictions)
    p_home_win = ts.outcome_probabilities(3.0, 0.2)["home_win"]
    assert metrics["outcome_log_loss"] == pytest.approx(-np.log(p_home_win))
    # Home: 3 goals vs lambda 3 -> 0. Away: 0 vs 0.2 -> 2 * 0.2 = 0.4. Mean 0.2.
    assert metrics["goals_deviance"] == pytest.approx(0.2)
