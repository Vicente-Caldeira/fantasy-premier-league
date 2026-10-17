import pytest
from hypothesis import given
from hypothesis import strategies as st

from fpl_core.scoring import MatchStats, Position, defensive_contribution_points, points


def test_did_not_play_scores_zero():
    assert points(Position.MID, MatchStats(minutes=0, bonus=0)) == 0


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(1, 1), (59, 1), (60, 2), (90, 2)],
)
def test_appearance_points(minutes, expected):
    # FWD gets no clean sheet points, so this isolates appearance points.
    assert points(Position.FWD, MatchStats(minutes=minutes)) == expected


@pytest.mark.parametrize(
    ("position", "goal_points"),
    [(Position.GKP, 10), (Position.DEF, 6), (Position.MID, 5), (Position.FWD, 4)],
)
def test_goal_points_by_position(position, goal_points):
    base = MatchStats(minutes=90, goals_conceded=2)
    scored = base.model_copy(update={"goals_scored": 1})
    assert points(position, scored) - points(position, base) == goal_points


@pytest.mark.parametrize(
    ("position", "cs_points"),
    [(Position.GKP, 4), (Position.DEF, 4), (Position.MID, 1), (Position.FWD, 0)],
)
def test_clean_sheet_requires_sixty_minutes(position, cs_points):
    assert points(position, MatchStats(minutes=90)) == 2 + cs_points
    assert points(position, MatchStats(minutes=59)) == 1


def test_defender_concedes_one_point_per_two_goals():
    assert points(Position.DEF, MatchStats(minutes=90, goals_conceded=1)) == 2
    assert points(Position.DEF, MatchStats(minutes=90, goals_conceded=3)) == 1
    assert points(Position.DEF, MatchStats(minutes=90, goals_conceded=4)) == 0


def test_goalkeeper_saves_and_penalty_save():
    stats = MatchStats(minutes=90, goals_conceded=1, saves=7, penalties_saved=1)
    assert points(Position.GKP, stats) == 2 + 2 + 5


def test_outfield_saves_do_not_score():
    assert points(Position.DEF, MatchStats(minutes=90, goals_conceded=1, saves=3)) == 2


def test_defender_defcon_uses_cbit_without_recoveries():
    below = MatchStats(minutes=90, clearances_blocks_interceptions=6, tackles=3, recoveries=10)
    at = below.model_copy(update={"tackles": 4})
    assert defensive_contribution_points(Position.DEF, below) == 0
    assert defensive_contribution_points(Position.DEF, at) == 2


def test_midfielder_defcon_includes_recoveries():
    stats = MatchStats(minutes=90, clearances_blocks_interceptions=4, tackles=3, recoveries=5)
    assert defensive_contribution_points(Position.MID, stats) == 2
    assert (
        defensive_contribution_points(Position.MID, stats.model_copy(update={"recoveries": 4})) == 0
    )


def test_goalkeeper_never_earns_defcon():
    stats = MatchStats(minutes=90, clearances_blocks_interceptions=30)
    assert defensive_contribution_points(Position.GKP, stats) == 0


def test_defcon_is_capped_at_two():
    stats = MatchStats(minutes=90, clearances_blocks_interceptions=40)
    assert defensive_contribution_points(Position.DEF, stats) == 2


def test_negative_events():
    stats = MatchStats(
        minutes=90, goals_conceded=1, yellow_cards=1, red_cards=1, own_goals=1, penalties_missed=1
    )
    assert points(Position.FWD, stats) == 2 - 1 - 3 - 2 - 2


@given(
    position=st.sampled_from(Position),
    minutes=st.integers(1, 90),
    goals=st.integers(0, 4),
    bonus=st.integers(0, 3),
)
def test_scoring_a_goal_never_reduces_points(position, minutes, goals, bonus):
    stats = MatchStats(minutes=minutes, goals_scored=goals, goals_conceded=1, bonus=bonus)
    more = stats.model_copy(update={"goals_scored": goals + 1})
    assert points(position, more) > points(position, stats)
