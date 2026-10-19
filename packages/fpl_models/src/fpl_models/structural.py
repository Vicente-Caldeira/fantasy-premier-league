"""Structural attacking model (E17): goals and assists from shrunk per-90 rates.

The tree models compress the top end (elite attackers' goals ~10% low): a player's
expected goals are learned as one target from many correlated features, and trees
average extreme players with similar-looking rows. The strong public models instead
build them explicitly:

    rate     per-90 rate given an appearance: xG blended with goals (goals alone
             before FPL published xG), over his recent minutes
    shrink   towards a prior from position and price (FPL prices quality), with
             strength k (in 90s) chosen on the newest training rows
    fixture  x team lambda in this match / the team's recent average lambda
    minutes  x expected minutes from the minutes model
    calib    x a per-position constant making training totals match

All parameters are fitted on training rows only, whose features are pre-deadline.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

POSITIONS = ("GKP", "DEF", "MID", "FWD")
SHRINK_GRID = (1.0, 2.0, 4.0, 8.0, 16.0)  # prior weight, in 90s of minutes
XG_WEIGHT = 0.7  # finishing skill is real but noisy: lean on xG
VALIDATION_SHARE = 0.2
TEAM_FIXTURES = 10

TARGETS = {
    # target: (expected per-90 column, actual per-90 column, expected target column)
    "goals_scored": ("expected_goals_per90", "goals_scored_per90", "expected_goals"),
    "assists": ("expected_assists_per90", "assists_per90", "expected_assists"),
}


def observed_rate(rows: pl.DataFrame, target: str) -> np.ndarray:
    expected, actual, _ = TARGETS[target]
    xg = rows[expected].to_numpy().astype(np.float64)
    goals = rows[actual].fill_null(0.0).to_numpy().astype(np.float64)
    return np.where(np.isnan(xg), goals, XG_WEIGHT * xg + (1 - XG_WEIGHT) * goals)


def exposure_90s(rows: pl.DataFrame) -> np.ndarray:
    return rows["minutes_exposure_l10"].fill_null(0.0).to_numpy().astype(np.float64) / 90


@dataclass(frozen=True)
class AttackModel:
    target: str
    k: float
    # log prior rate = a[position] + b[position] * log(price)
    prior: dict[str, tuple[float, float]]
    calibration: dict[str, float]
    team_lambda: dict[str, float]  # recent average lambda per team
    league_lambda: float

    def prior_rate(self, rows: pl.DataFrame) -> np.ndarray:
        price = np.log(rows["price"].cast(pl.Float64).to_numpy())
        a = np.array([self.prior.get(p, (np.log(0.05), 0.0))[0] for p in rows["position"]])
        b = np.array([self.prior.get(p, (np.log(0.05), 0.0))[1] for p in rows["position"]])
        result: np.ndarray = np.exp(a + b * price)
        return result

    def rate(self, rows: pl.DataFrame, k: float | None = None) -> np.ndarray:
        """Shrunk per-90 rate given an appearance, adjusted to this fixture."""
        k = self.k if k is None else k
        n = exposure_90s(rows)
        observed = np.nan_to_num(observed_rate(rows, self.target))
        shrunk = (observed * n + self.prior_rate(rows) * k) / (n + k)
        team = np.array([self.team_lambda.get(t, self.league_lambda) for t in rows["team_key"]])
        fixture = rows["team_lambda"].fill_null(self.league_lambda).to_numpy() / team
        calib = np.array([self.calibration.get(p, 1.0) for p in rows["position"]])
        result: np.ndarray = shrunk * fixture * calib
        return result


def _fit_prior(rows: pl.DataFrame, target: str) -> dict[str, tuple[float, float]]:
    """Per position, log(per-90 rate) ~ log(price), weighted by minutes played."""
    expected_col = TARGETS[target][2]
    prior = {}
    for position in POSITIONS:
        group = rows.filter((pl.col("position") == position) & (pl.col("minutes") > 0))
        if group.height < 50:
            continue
        # Bin by price so the fit is on rates, not on single (mostly zero) matches.
        label = group[expected_col].cast(pl.Float64).fill_null(group[target].cast(pl.Float64))
        binned = (
            group.with_columns(label=label, band=(pl.col("price") // 5) * 5)
            .group_by("band")
            .agg(pl.col("label").sum(), (pl.col("minutes").sum() / 90).alias("n90"))
            .filter(pl.col("n90") > 20)
        )
        if binned.height < 2:
            continue
        x = np.log(binned["band"].cast(pl.Float64).to_numpy() + 2.5)
        y = np.log(np.maximum(binned["label"].to_numpy() / binned["n90"].to_numpy(), 1e-4))
        w = binned["n90"].to_numpy()
        b, a = np.polyfit(x, y, 1, w=np.sqrt(w))
        prior[position] = (float(a), float(b))
    return prior


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-9)
    log_term = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
    return float(np.mean(2 * (log_term - (y - mu))))


def fit(train: pl.DataFrame, target: str) -> AttackModel:
    """Fit prior, team averages, shrinkage strength and calibration on training rows."""
    played = train.filter(pl.col("minutes") > 0).sort("deadline_utc")
    recent = (
        train.sort("deadline_utc")
        .group_by("team_key", "match_key")
        .agg(pl.col("team_lambda").first(), pl.col("deadline_utc").first())
        .sort("deadline_utc")
        .group_by("team_key")
        .agg(pl.col("team_lambda").tail(TEAM_FIXTURES).mean())
    )
    team_lambda = dict(zip(recent["team_key"], recent["team_lambda"], strict=True))
    league = float(np.nanmean(list(team_lambda.values()))) if team_lambda else 1.35
    base = AttackModel(target, 4.0, _fit_prior(played, target), {}, team_lambda, league)

    # Shrinkage strength on the newest rows, scored on goals-per-appearance with
    # exposure = minutes/90 (the rate model's own likelihood).
    cut = int(played.height * (1 - VALIDATION_SHARE))
    valid = played[cut:]
    y = valid[target].cast(pl.Float64).to_numpy()
    exposure = valid["minutes"].to_numpy() / 90
    best_k = min(SHRINK_GRID, key=lambda k: _poisson_deviance(y, base.rate(valid, k) * exposure))

    # Calibration: make predicted totals match actual totals per position.
    model = AttackModel(target, best_k, base.prior, {}, team_lambda, league)
    mu = model.rate(played) * played["minutes"].to_numpy() / 90
    actual = played[target].cast(pl.Float64).to_numpy()
    calibration = {}
    for position in POSITIONS:
        mask = (played["position"] == position).to_numpy()
        if mu[mask].sum() > 0:
            calibration[position] = float(actual[mask].sum() / mu[mask].sum())
    return AttackModel(target, best_k, base.prior, calibration, team_lambda, league)
