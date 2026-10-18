from fpl_models import optimizer
from fpl_models import season_sim as ss

POSITION = {1: "GKP", 2: "DEF", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "MID", 8: "MID",
            9: "FWD", 10: "FWD", 11: "FWD", 12: "GKP", 13: "DEF", 14: "MID", 15: "FWD"}  # fmt: skip
LINEUP = list(range(1, 12))  # 1-3-4-3
BENCH = [12, 13, 14, 15]  # GK first is skipped for outfielders


def gameweek(captain=9, vice=10):
    return optimizer.Gameweek(
        gameweek=1, squad=LINEUP + BENCH, lineup=LINEUP, captain=captain, vice_captain=vice,
        transfers_in=[], transfers_out=[], free_transfers=1, hits=0, expected_points=0.0,
    )  # fmt: skip


def played(*absent):
    return {p: (0 if p in absent else 90) for p in POSITION}


def test_non_playing_starter_is_replaced_by_first_valid_bench_player():
    team = ss.auto_subs(LINEUP, [13, 14, 15], played(5), POSITION)
    assert 5 not in team
    assert 13 in team  # a defender may replace a midfielder: 2 DEF + ... formation stays valid


def test_substitution_keeps_a_valid_formation():
    # Only 3 defenders start; if one doesn't play he must be replaced by a defender.
    team = ss.auto_subs(LINEUP, [14, 15, 13], played(2), POSITION)
    assert 13 in team
    assert 14 not in team
    assert 15 not in team


def test_goalkeeper_only_replaced_by_goalkeeper():
    team = ss.auto_subs(LINEUP, [13, 12], played(1), POSITION)
    assert 12 in team
    assert 13 not in team


def test_captain_doubled_and_vice_takes_over_when_captain_misses():
    actual = {p: 2 for p in POSITION} | {9: 10, 10: 6}
    assert (
        ss.score(gameweek(), [12, 13, 14, 15], actual, played(), POSITION)
        == 11 * 2 - 4 + 10 + 6 + 10
    )
    # Captain (9) doesn't play: vice-captain (10) is doubled; 15 comes off the bench.
    actual_missing = actual | {9: 0}
    expected = (
        sum(actual_missing[p] for p in LINEUP if p != 9) + actual_missing[15] + actual_missing[10]
    )
    assert ss.score(gameweek(), [12, 13, 14, 15], actual_missing, played(9), POSITION) == expected
