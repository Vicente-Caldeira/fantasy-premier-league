import polars as pl

from fpl_models import e4


def path(signal):
    gws = list(range(1, 39))
    return pl.DataFrame({"gameweek": gws, "half": [1 if g <= e4.HALF else 2 for g in gws],
                         "bench_boost_signal": signal})  # fmt: skip


def test_rule_plays_first_qualifying_week_in_each_half():
    signal = [0.0] * 38
    signal[4], signal[9], signal[24] = 12.0, 15.0, 11.0  # GW5, GW10, GW25
    assert e4.play_rule(path(signal), "bench_boost", 10.0) == (5, 25)


def test_rule_falls_back_to_the_last_gameweek_of_the_half():
    assert e4.play_rule(path([0.0] * 38), "bench_boost", 10.0) == (19, 38)
