from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from fpl_models import player_features as pf

DAY0 = datetime(2025, 9, 1, 15, tzinfo=UTC)


def day(d: float) -> datetime:
    return DAY0 + timedelta(days=d)


def league_rows() -> pl.DataFrame:
    """One player at club "a": league games on days 0, 7 and 14 (deadlines 1.5 days
    before each kickoff), 90 minutes, 0.2 xG each."""
    return pl.DataFrame({
        "match_key": ["m0", "m7", "m14"], "player_code": [1] * 3, "season": ["2025-26"] * 3,
        "team_key": ["a"] * 3, "kickoff_utc": [day(0), day(7), day(14)],
        "deadline_utc": [day(-1.5), day(5.5), day(12.5)], "minutes": [90, 90, 90],
        "expected_goals": [0.2] * 3, "expected_assists": [0.1] * 3, "total_shots": [2] * 3,
        "touches_opposition_box": [5] * 3, "chances_created": [1] * 3,
    })  # fmt: skip


def cup_players() -> pl.DataFrame:
    """A cup game on day 10 (before the day-14 deadline) and one on day 13, between the
    deadline and kickoff, which must not leak into the day-14 features."""
    return pl.DataFrame({
        "season": ["2025-26"] * 2, "kickoff_utc": [day(10), day(13)], "player_code": [1, 1],
        "minutes": [90, 60], "xg": [1.0, 5.0], "xa": [0.0, 0.0], "total_shots": [4, 9],
        "touches_opposition_box": [8, 20], "chances_created": [0, 0],
    })  # fmt: skip


def cups() -> pl.DataFrame:
    return pl.DataFrame({
        "season": ["2025-26"] * 2, "competition": ["Champions League"] * 2,
        "kickoff_utc": [day(10), day(13)], "team_key": ["a", "a"],
    })  # fmt: skip


def features() -> dict[str, float | None]:
    out = pf.all_competition_features(league_rows(), cup_players(), cups())
    return out.filter(pl.col("match_key") == "m14").row(0, named=True)


def test_windows_count_league_and_cup_minutes_before_the_deadline_only():
    f = features()
    assert f["ac_minutes_7d"] == 180  # day 7 league + day 10 cup; day 13 is after the deadline
    assert f["ac_minutes_14d"] == 270  # + day 0
    assert f["ac_minutes_season"] == 270
    assert f["ac_minutes_share_7d"] == 1.0  # 180 of 2 club matches x 90
    assert f["ac_last_minutes"] == 90
    assert f["ac_days_since_appearance"] == 4  # day 10 -> day 14
    assert f["eval_cup_minutes_prev_6d"] == 90


def test_rates_pool_league_and_cup_appearances():
    f = features()
    assert f["ac_xg_per90_14d"] == pytest.approx((0.2 + 0.2 + 1.0) / 270 * 90)
    assert f["ac_box_touches_per90_season"] == pytest.approx((5 + 5 + 8) / 270 * 90)
    # The EWM puts more weight on the latest appearance (the day-10 cup game).
    assert f["ac_xg_per90_hl4"] > f["ac_xg_per90_14d"]
    assert f["ac_xg_per90_hl15"] < f["ac_xg_per90_hl4"]


def test_no_history_and_no_coverage_give_nulls():
    out = pf.all_competition_features(league_rows(), cup_players(), cups())
    first = out.filter(pl.col("match_key") == "m0").row(0, named=True)
    assert first["ac_minutes_7d"] == 0
    assert first["ac_days_since_appearance"] is None
    assert first["ac_xg_per90_14d"] is None  # under MIN_WINDOW_MINUTES
    uncovered = pf.all_competition_features(league_rows(), cup_players().head(0), cups())
    assert uncovered["ac_minutes_7d"].is_null().all()
