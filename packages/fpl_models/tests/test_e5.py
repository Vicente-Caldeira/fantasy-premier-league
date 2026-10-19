from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from fpl_models import e5


def predictions(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s, season in enumerate(("2020-21", "2021-22", "2022-23")):
        for gw in range(1, 11):
            deadline = datetime(2020 + s, 8, 1, tzinfo=UTC) + timedelta(days=7 * gw)
            for p in range(40):
                predicted = float(rng.gamma(2, 1.2))
                rows.append({"season": season, "gameweek": gw, "deadline_utc": deadline,
                             "player_code": p, "position": ("GKP", "DEF", "MID", "FWD")[p % 4],
                             "predicted": predicted,
                             "points": int(rng.poisson(0.9 * predicted))})  # fmt: skip
    return pl.DataFrame(rows)


def test_calibration_uses_only_earlier_gameweeks():
    base = predictions()
    calibrated = e5.calibrate(base, by_position=True)
    cutoff = calibrated["deadline_utc"].sort()[25]
    # Rewrite every outcome from the cutoff on: earlier calibrated values can't move.
    changed = base.with_columns(
        pl.when(pl.col("deadline_utc") >= cutoff)
        .then(0)
        .otherwise(pl.col("points"))
        .alias("points")
    )
    again = e5.calibrate(changed, by_position=True)
    key = ["season", "gameweek", "player_code"]
    before = calibrated.filter(pl.col("deadline_utc") <= cutoff).sort(key)["predicted"]
    after = again.filter(pl.col("deadline_utc") <= cutoff).sort(key)["predicted"]
    assert np.allclose(before.to_numpy(), after.to_numpy())


def test_calibration_corrects_a_systematic_overprediction():
    calibrated = e5.calibrate(predictions(), by_position=False)
    raw = predictions().filter(pl.col("season") != "2020-21")
    raw_bias = (raw["predicted"] - raw["points"]).mean()
    new_bias = (calibrated["predicted"] - calibrated["points"]).mean()
    assert abs(new_bias) < abs(raw_bias) / 2
