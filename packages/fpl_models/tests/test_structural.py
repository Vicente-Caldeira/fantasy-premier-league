from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from fpl_models import structural


def _rows(n_weeks: int = 40) -> pl.DataFrame:
    """Three forwards: a cheap 0.2/90 one, a premium 0.8/90 one, and a newcomer."""
    rng = np.random.default_rng(0)
    start = datetime(2021, 8, 13, tzinfo=UTC)
    players = ((1, 55, 0.2, 900), (2, 125, 0.8, 900), (3, 125, 0.8, 0))
    rows = []
    for week in range(n_weeks):
        for code, price, rate, exposure in players:
            rows.append({
                "player_code": code, "position": "FWD", "team_key": "A", "match_key": f"m{week}",
                "deadline_utc": start + timedelta(days=7 * week), "price": price,
                "minutes": 90, "minutes_exposure_l10": exposure, "team_lambda": 1.5,
                "expected_goals_per90": rate, "goals_scored_per90": rate,
                "expected_goals": rate, "goals_scored": int(rng.poisson(rate)),
            })  # fmt: skip
    return pl.DataFrame(rows)


def test_rates_follow_the_player_and_fall_back_to_the_price_prior():
    rows = _rows()
    model = structural.fit(rows, "goals_scored")
    cheap, premium, newcomer = model.rate(rows.filter(pl.col("match_key") == "m0"))
    assert cheap < premium
    # No minutes yet: the newcomer gets the prior, which rises with price.
    prior = model.prior_rate(rows.filter(pl.col("player_code") == 3))[0]
    assert newcomer == prior * model.calibration["FWD"]
    assert model.prior["FWD"][1] > 0


def test_rates_scale_with_this_fixtures_team_lambda():
    rows = _rows()
    model = structural.fit(rows, "goals_scored")
    week = rows.filter(pl.col("match_key") == "m0")
    easy = model.rate(week.with_columns(team_lambda=pl.lit(3.0)))
    assert np.allclose(easy, 2 * model.rate(week))
