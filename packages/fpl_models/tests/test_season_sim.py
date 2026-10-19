import numpy as np
import polars as pl

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


def synthetic_players(seed=1):
    """Five per position, with cheap enablers so a valid squad always exists."""
    rng = np.random.default_rng(seed)
    rows = []
    for position in ("GKP", "DEF", "MID", "FWD"):
        for i in range(8):
            price = int(rng.integers(40, 46) if i < 5 else rng.integers(50, 120))
            rows.append({"player_code": len(rows), "position": position,
                         "team_key": f"club{len(rows) % 10}", "price": price,
                         "base": price / 20 + rng.normal(0, 0.5)})  # fmt: skip
    return pl.DataFrame(rows)


def xp(players, gameweeks):
    return pl.DataFrame([
        {"player_code": p, "gameweek": gw, "expected_points": max(b, 0.0)}
        for p, b in zip(players["player_code"], players["base"], strict=True)
        for gw in gameweeks
    ])  # fmt: skip


def synthetic_weeks(n_weeks=3):
    players = synthetic_players()
    weeks = []
    for gw in range(1, n_weeks + 1):
        upcoming = tuple(range(gw, min(gw + 2, n_weeks) + 1))
        actual = {p: int(b) for p, b in zip(players["player_code"], players["base"], strict=True)}
        weeks.append(
            ss.Week(
                gameweek=gw,
                pool=players.select("player_code", "position", "team_key", "price"),
                xp=xp(players, upcoming),
                upside=pl.DataFrame(
                    {"player_code": players["player_code"], "sim_p90": 0.0, "p_haul": 0.0}
                ),
                actual=actual,
                minutes=dict.fromkeys(actual, 90),
            )
        )
    return weeks


def test_play_records_every_gameweek_and_seeds_change_the_start():
    weeks = synthetic_weeks()
    settings = optimizer.Settings(time_limit_seconds=10)
    result, records = ss.play("2021-22", weeks, settings, seed=0)
    assert len(result.points) == len(records) == len(weeks)
    assert all(len(r.squad) == 15 and len(r.lineup) == 11 for r in records)
    assert all(r.captain in r.lineup for r in records)
    starts = {
        tuple(sorted(ss.play("2021-22", weeks, settings, seed=s)[1][0].squad)) for s in range(4)
    }
    assert len(starts) > 1  # different seeds, different starting squads


def test_wildcard_is_played_once_per_half_and_takes_no_hits():
    weeks = synthetic_weeks(n_weeks=4)
    settings = optimizer.Settings(time_limit_seconds=10)
    _, records = ss.play("2021-22", weeks, settings, seed=0, wildcards=(2, 3))
    chips = [r.chip for r in records]
    assert chips == [None, "wildcard", None, None]  # GW3 is in the same half: no second
    assert records[1].hits == 0
    # Threshold rule: an impossible bar never plays it; a zero bar plays it at once.
    _, never = ss.play("2021-22", weeks, settings, seed=0, wildcard_threshold=1e9)
    _, eager = ss.play("2021-22", weeks, settings, seed=0, wildcard_threshold=0.0)
    assert all(r.chip is None for r in never)
    assert [r.chip for r in eager] == [None, "wildcard", None, None]
