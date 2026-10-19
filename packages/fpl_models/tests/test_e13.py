from datetime import UTC, datetime, timedelta

import polars as pl

from fpl_models import e13, e13_analysis


def test_captures_are_one_a_day_after_the_price_changes_inside_the_study_period():
    cdx = [["20230801010000"], ["20230801090000"], ["20230801130000"], ["20230802050000"],
           ["20230601010000"], ["20220801010000"]]  # fmt: skip
    assert e13.chosen_captures(cdx) == ["20230801130000", "20230802050000"]


def test_buying_before_a_rise_keeps_half_and_selling_before_a_fall_keeps_all():
    assert e13_analysis.buy_gain(paid=50, at_deadline=52) == 1
    assert e13_analysis.buy_gain(paid=50, at_deadline=51) == 0  # one rise: nothing to sell
    assert e13_analysis.buy_gain(paid=50, at_deadline=49) == -1
    assert e13_analysis.sell_gain(sold=80, at_deadline=79) == 1
    assert e13_analysis.sell_gain(sold=80, at_deadline=82) == -1


def test_pairs_stay_within_one_transfer_window():
    t0 = datetime(2024, 8, 20, 10, tzinfo=UTC)
    frame = pl.DataFrame({
        "player_code": [1, 1, 1],
        "captured_at": [t0 + timedelta(hours=h) for h in (0, 24, 48)],
        "next_gameweek": [2, 2, 3], "price": [50, 51, 51], "season": ["2024-25"] * 3,
    })  # fmt: skip
    out = e13_analysis.pairs(frame)
    assert out["change"].to_list() == [1]  # the second pair crosses a deadline
