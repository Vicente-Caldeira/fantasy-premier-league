from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from fpl_models import baselines, metrics
from fpl_models.backtest import run_backtest
from fpl_models.data import CONTEXT_COLUMNS, OUTCOME_COLUMNS, player_gameweeks
from fpl_models.leaderboard import leaderboard, to_markdown

START = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)


def match_rows(points_by_player, *, gameweeks=6, seasons=("2023-24", "2024-25")):
    """Synthetic fct_player_matches rows: one fixture per player per gameweek."""
    rows = []
    for s, season in enumerate(seasons):
        for gw in range(1, gameweeks + 1):
            kickoff = START + timedelta(days=365 * s + 7 * gw)
            for player, pts in points_by_player.items():
                rows.append({
                    "season": season, "gameweek": gw, "kickoff_utc": kickoff,
                    "player_code": player, "element": player, "position": "MID",
                    "team_key": "a", "opponent_team_key": "b", "is_home": True,
                    "is_provisional": False, "price": 50, "total_points": pts(gw),
                    "minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 0,
                    "bonus": 0,
                })  # fmt: skip
    return pl.DataFrame(rows)


@pytest.fixture
def gameweeks():
    return player_gameweeks(match_rows({1: lambda gw: 10, 2: lambda gw: 2, 3: lambda gw: gw}))


# --- data -----------------------------------------------------------------------------


def test_double_gameweek_points_are_summed_and_deadline_precedes_first_kickoff():
    rows = match_rows({1: lambda gw: 3}, gameweeks=1, seasons=("2024-25",))
    second = rows.with_columns(
        pl.col("kickoff_utc") + timedelta(days=2), total_points=pl.lit(5, pl.Int64)
    )
    gw = player_gameweeks(pl.concat([rows, second]))
    assert gw.select("points", "n_fixtures").row(0) == (8, 2)
    assert gw["deadline_utc"][0] == START + timedelta(days=7) - timedelta(minutes=90)


def test_gameweeks_with_provisional_fixtures_are_excluded():
    rows = match_rows({1: lambda gw: 3}, gameweeks=2, seasons=("2024-25",))
    rows = rows.with_columns(is_provisional=pl.col("gameweek") == 2)
    assert player_gameweeks(rows)["gameweek"].to_list() == [1]


# --- backtest -------------------------------------------------------------------------


class Spy:
    """Records what the backtest hands a model."""

    name = "spy"

    def __init__(self):
        self.calls = []

    def predict(self, history, targets):
        self.calls.append((history, targets))
        return pl.Series([0.0] * targets.height)


def test_history_is_strictly_before_the_deadline(gameweeks):
    spy = Spy()
    run_backtest(spy, gameweeks, ["2024-25"])
    for history, targets in spy.calls:
        assert history["deadline_utc"].max() < targets["deadline_utc"][0]


def test_targets_never_contain_outcomes(gameweeks):
    spy = Spy()
    run_backtest(spy, gameweeks, ["2024-25"])
    for _, targets in spy.calls:
        assert set(targets.columns) == set(CONTEXT_COLUMNS)
        assert not set(targets.columns) & set(OUTCOME_COLUMNS)


def test_first_gameweek_of_a_season_sees_previous_seasons_only(gameweeks):
    spy = Spy()
    run_backtest(spy, gameweeks, ["2024-25"])
    history, targets = spy.calls[0]
    assert targets["gameweek"][0] == 1
    assert history["season"].unique().to_list() == ["2023-24"]


def test_predictions_stay_aligned_with_their_players(gameweeks):
    # Player 1 always scores 10, player 2 always 2, player 3 scores the gameweek number.
    # Form over 2 GWs for player 3 in 2024-25: (6+5)/2, (1+6)/2, (2+1)/2, ... -> mean 3.5.
    result = run_backtest(baselines.RecentForm(window=2), gameweeks, ["2024-25"])
    by_player = result.predictions.group_by("player_code").agg(pl.col("predicted").mean())
    assert dict(by_player.iter_rows()) == {1: 10.0, 2: 2.0, 3: pytest.approx(3.5)}


def test_wrong_number_of_predictions_is_rejected(gameweeks):
    class Broken:
        name = "broken"

        def predict(self, history, targets):
            return pl.Series([1.0])

    with pytest.raises(ValueError, match="predictions for"):
        run_backtest(Broken(), gameweeks, ["2024-25"])


# --- baselines ------------------------------------------------------------------------


def test_season_ppg_uses_last_season_before_the_first_deadline(gameweeks):
    history = gameweeks.filter(pl.col("season") == "2023-24")
    targets = gameweeks.filter((pl.col("season") == "2024-25") & (pl.col("gameweek") == 1))
    predicted = baselines.SeasonPointsPerGameweek().predict(
        history, targets.select(CONTEXT_COLUMNS)
    )
    assert predicted.to_list() == [10.0, 2.0, 3.5]


def test_unknown_player_gets_the_positional_prior_from_ewm(gameweeks):
    history = gameweeks.filter(pl.col("season") == "2023-24")
    newcomer = history.head(1).with_columns(player_code=pl.lit(99)).select(CONTEXT_COLUMNS)
    predicted = baselines.ExponentialForm().predict(history, newcomer)
    assert predicted[0] == pytest.approx(history["points"].mean())


def test_ewm_weights_recent_gameweeks_more(gameweeks):
    history = gameweeks.filter(pl.col("player_code") == 3)  # points rise with gameweek
    target = history.tail(1).select(CONTEXT_COLUMNS)
    ewm = baselines.ExponentialForm(half_life=1, prior_weight=0).predict(history, target)[0]
    assert ewm > history["points"].mean()


# --- metrics --------------------------------------------------------------------------


def test_decision_metrics_on_a_known_ranking():
    predictions = pl.DataFrame({
        "season": ["s"] * 3, "gameweek": [1] * 3, "player_code": [1, 2, 3],
        "predicted": [9.0, 5.0, 1.0], "points": [2, 12, 4], "minutes": [90, 90, 90],
    })  # fmt: skip
    result = metrics.evaluate(predictions, predictions.with_columns(deadline_utc=pl.lit(START)))
    assert result["captain_points"] == 2
    assert result["captain_points_oracle"] == 12
    assert result["top11_points"] == result["top11_points_oracle"] == 18
    assert result["mae"] == pytest.approx((7 + 7 + 3) / 3)


def test_regulars_are_chosen_from_previous_gameweeks_only():
    history = pl.DataFrame({
        "season": ["s"] * 3, "gameweek": [1, 2, 3], "player_code": [1, 1, 1],
        "deadline_utc": [START + timedelta(days=7 * i) for i in range(3)],
        "minutes": [0, 0, 90],
    })  # fmt: skip
    predictions = history.with_columns(predicted=pl.lit(0.0), points=pl.lit(0))
    # Played only in GW3, so he is not a "regular" for GW3 itself.
    assert metrics.regulars(predictions, history).height == 0


def test_leaderboard_ranks_by_primary_metric(gameweeks):
    board = leaderboard(baselines.BASELINES, gameweeks, ["2024-25"])
    assert board["regulars_mae"].is_sorted()
    assert set(board["predictor"]) == {b.name for b in baselines.BASELINES}


def test_leaderboard_renders_as_markdown(gameweeks):
    board = leaderboard(baselines.BASELINES, gameweeks, ["2024-25"])
    lines = to_markdown(board).splitlines()
    assert lines[0].startswith("| predictor | regulars_mae")
    assert len(lines) == 2 + len(baselines.BASELINES)
