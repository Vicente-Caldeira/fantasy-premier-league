"""Team strength: expected goals (lambda) for both sides of every fixture.

Lambdas drive everything downstream: clean-sheet probability is P(opponent scores 0)
= exp(-lambda_opponent), and a team's expected goals are what its players share.

Models (all scored by the match-level walk-forward backtest below):
  * LeagueAverage: home/away mean goals over the last two seasons. The floor.
  * MarketOdds:    lambdas whose independent-Poisson outcome and over/under-2.5
                   probabilities best reproduce the margin-free *opening* odds
                   (closing odds are struck after the FPL deadline: leakage).
  * Ratings:       log lambda = mu + home + attack - defence (+ promoted effects),
                   fitted by time-weighted Poisson likelihood on a goals/xG blend,
                   with L2 shrinkage. Works for fixtures far ahead, without odds.
  * Blend:         geometric mean of market and ratings lambdas.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import duckdb
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize
from scipy.stats import poisson

# Tail mass beyond 25 goals is <= 1.3e-9 per side for lambda <= 6 (the odds solver's
# upper bound). 10 goals would have dropped 4% of the mass at lambda = 6.
MAX_GOALS = 25
DEADLINE_BEFORE_KICKOFF = timedelta(minutes=90)
Floats = NDArray[np.float64]

# Known before the deadline. Goals, xG and closing odds are outcomes / post-deadline.
CONTEXT_COLUMNS = (
    "match_key",
    "season",
    "gameweek",
    "deadline_utc",
    "kickoff_utc",
    "home_team_key",
    "away_team_key",
    "home_promoted",
    "away_promoted",
    "p_home_win",
    "p_draw",
    "p_away_win",
    "p_over_2_5",
)
OUTCOME_COLUMNS = ("home_goals", "away_goals", "home_xg", "away_xg")


# --- data -----------------------------------------------------------------------------


def load_matches(warehouse: Path) -> pl.DataFrame:
    """Matches since 2016-17 with gameweek deadlines and promoted-team flags."""
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        connection.execute("set TimeZone = 'UTC'")
        matches = connection.sql(
            """
            select match_key, season, gameweek, kickoff_utc, home_team_key, away_team_key,
                is_played, home_goals, away_goals, home_xg, away_xg,
                p_home_win, p_draw, p_away_win, p_over_2_5
            from fct_matches
            """
        ).pl()
    return with_promoted(with_deadlines(matches))


def with_deadlines(matches: pl.DataFrame) -> pl.DataFrame:
    deadlines = (
        matches.filter(pl.col("gameweek").is_not_null())
        .group_by("season", "gameweek")
        .agg((pl.col("kickoff_utc").min() - DEADLINE_BEFORE_KICKOFF).alias("deadline_utc"))
    )
    return matches.join(deadlines, on=["season", "gameweek"], how="left", maintain_order="left")


def with_promoted(matches: pl.DataFrame) -> pl.DataFrame:
    """A team is promoted in a season if it wasn't in the league the season before."""
    seasons = matches.select("season", pl.col("home_team_key").alias("team")).unique()
    ordered = sorted(seasons["season"].unique().to_list())
    previous = dict(zip(ordered[1:], ordered[:-1], strict=False))
    present = set(seasons.iter_rows())

    def promoted(season: str, team: str) -> bool:
        before = previous.get(season)
        return before is not None and (before, team) not in present

    flags = {(s, t): promoted(s, t) for s, t in present}
    return matches.with_columns(
        pl.struct("season", "home_team_key")
        .map_elements(lambda r: flags[(r["season"], r["home_team_key"])], return_dtype=pl.Boolean)
        .alias("home_promoted"),
        pl.struct("season", "away_team_key")
        .map_elements(lambda r: flags[(r["season"], r["away_team_key"])], return_dtype=pl.Boolean)
        .alias("away_promoted"),
    )


# --- probabilities --------------------------------------------------------------------


def score_matrix(lambda_home: float, lambda_away: float) -> Floats:
    goals = np.arange(MAX_GOALS + 1)
    matrix: Floats = np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))
    return matrix


def outcome_probabilities(lambda_home: float, lambda_away: float) -> dict[str, float]:
    m = score_matrix(lambda_home, lambda_away)
    total = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    return {
        "home_win": float(np.tril(m, -1).sum()),
        "draw": float(np.trace(m)),
        "away_win": float(np.triu(m, 1).sum()),
        "over_2_5": float(m[total > 2.5].sum()),
        "home_clean_sheet": float(np.exp(-lambda_away)),
        "away_clean_sheet": float(np.exp(-lambda_home)),
    }


def lambdas_from_odds(
    p_home: float, p_draw: float, p_away: float, p_over: float | None
) -> tuple[float, float]:
    """Poisson lambdas that best reproduce margin-free outcome and over/under odds."""
    targets = [p_home, p_draw, p_away] + ([p_over] if p_over is not None else [])

    def residuals(x: Floats) -> Floats:
        p = outcome_probabilities(x[0], x[1])
        model = [p["home_win"], p["draw"], p["away_win"]] + ([p["over_2_5"]] if p_over else [])
        return np.asarray(model) - np.asarray(targets)

    fit = least_squares(residuals, x0=[1.5, 1.2], bounds=([0.05, 0.05], [6.0, 6.0]))
    return float(fit.x[0]), float(fit.x[1])


# --- models ---------------------------------------------------------------------------


class TeamModel(Protocol):
    @property
    def name(self) -> str: ...

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
        """`lambda_home`, `lambda_away` for each target row, in target order."""
        ...


def _frame(home: Sequence[float], away: Sequence[float]) -> pl.DataFrame:
    return pl.DataFrame({"lambda_home": list(home), "lambda_away": list(away)}, schema={
        "lambda_home": pl.Float64, "lambda_away": pl.Float64
    })  # fmt: skip


@dataclass(frozen=True)
class LeagueAverage:
    seasons: int = 2
    name: str = "league_average"

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
        recent = sorted(history["season"].unique().to_list())[-self.seasons :]
        played = history.filter(pl.col("season").is_in(recent))
        home, away = played["home_goals"].mean(), played["away_goals"].mean()
        return _frame([float(home)] * targets.height, [float(away)] * targets.height)  # type: ignore[arg-type]


@dataclass(frozen=True)
class MarketOdds:
    """Lambdas implied by opening odds; league average where a match has no odds."""

    name: str = "market_odds"
    fallback: LeagueAverage = field(default_factory=LeagueAverage)

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
        base = self.fallback.predict(history, targets)
        home, away = [], []
        for row, b in zip(targets.iter_rows(named=True), base.iter_rows(named=True), strict=True):
            if row["p_home_win"] is None:
                home.append(b["lambda_home"])
                away.append(b["lambda_away"])
                continue
            lh, la = lambdas_from_odds(
                row["p_home_win"], row["p_draw"], row["p_away_win"], row["p_over_2_5"]
            )
            home.append(lh)
            away.append(la)
        return _frame(home, away)


@dataclass(frozen=True)
class Ratings:
    """Time-weighted Poisson attack/defence ratings on a goals/xG blend.

    `xg_weight` blends the target: 0 = goals only, 1 = Understat xG only.
    """

    # Tuned on 2021-22 and 2022-23 (goals deviance); see ADR 0009.
    half_life_days: float = 180.0
    xg_weight: float = 0.5
    l2: float = 1.0
    lookback_days: int = 3 * 365

    @property
    def name(self) -> str:
        return f"ratings_hl{self.half_life_days:g}_xg{self.xg_weight:g}"

    def fit(self, history: pl.DataFrame, as_of: datetime) -> "FittedRatings":
        played = history.filter(
            pl.col("home_goals").is_not_null()
            & (pl.col("kickoff_utc") >= as_of - timedelta(days=self.lookback_days))
        )
        teams = sorted(set(played["home_team_key"]) | set(played["away_team_key"]))
        index = {t: i for i, t in enumerate(teams)}
        h = np.array([index[t] for t in played["home_team_key"]])
        a = np.array([index[t] for t in played["away_team_key"]])
        hp = played["home_promoted"].cast(pl.Float64).to_numpy()
        ap = played["away_promoted"].cast(pl.Float64).to_numpy()

        def target(goals: str, xg: str) -> Floats:
            g = played[goals].cast(pl.Float64).to_numpy()
            x = played[xg].cast(pl.Float64).fill_null(played[goals].cast(pl.Float64)).to_numpy()
            return (1 - self.xg_weight) * g + self.xg_weight * x

        yh, ya = target("home_goals", "home_xg"), target("away_goals", "away_xg")
        age_days = played.select(
            (pl.lit(as_of) - pl.col("kickoff_utc")).dt.total_seconds() / 86_400
        ).to_series()
        w = 0.5 ** (age_days.to_numpy().astype(np.float64) / self.half_life_days)
        n = len(teams)

        def unpack(x: Floats) -> tuple[float, float, Floats, Floats, float, float]:
            return x[0], x[1], x[2 : 2 + n], x[2 + n : 2 + 2 * n], x[-2], x[-1]

        def objective(x: Floats) -> tuple[float, Floats]:
            mu, home, att, dfn, att_p, def_p = unpack(x)
            log_h = mu + home + att[h] - dfn[a] + att_p * hp - def_p * ap
            log_a = mu + att[a] - dfn[h] + att_p * ap - def_p * hp
            lh, la = np.exp(log_h), np.exp(log_a)
            nll = np.sum(w * (lh - yh * log_h)) + np.sum(w * (la - ya * log_a))
            nll += self.l2 * (np.sum(att**2) + np.sum(dfn**2))
            rh, ra = w * (lh - yh), w * (la - ya)  # d nll / d log lambda
            grad = np.zeros_like(x)
            grad[0] = rh.sum() + ra.sum()
            grad[1] = rh.sum()
            g_att, g_def = np.zeros(n), np.zeros(n)
            np.add.at(g_att, h, rh)
            np.add.at(g_att, a, ra)
            np.add.at(g_def, a, -rh)
            np.add.at(g_def, h, -ra)
            grad[2 : 2 + n] = g_att + 2 * self.l2 * att
            grad[2 + n : 2 + 2 * n] = g_def + 2 * self.l2 * dfn
            grad[-2] = np.sum(rh * hp) + np.sum(ra * ap)
            grad[-1] = -np.sum(rh * ap) - np.sum(ra * hp)
            return float(nll), grad

        x0 = np.zeros(2 * n + 4)
        x0[0] = np.log(max(float(np.mean(np.concatenate([yh, ya]))), 0.1))
        result = minimize(objective, x0, jac=True, method="L-BFGS-B")
        mu, home, att, dfn, att_p, def_p = unpack(result.x)
        return FittedRatings(
            mu, home, dict(zip(teams, att, strict=True)), dict(zip(teams, dfn, strict=True)),
            att_p, def_p,
        )  # fmt: skip

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
        as_of: datetime = targets["deadline_utc"].min()  # type: ignore[assignment]
        fitted = self.fit(history, as_of)
        return fitted.lambdas(targets)


@dataclass(frozen=True)
class FittedRatings:
    mu: float
    home: float
    attack: dict[str, float]
    defence: dict[str, float]
    promoted_attack: float
    promoted_defence: float

    def lambdas(self, targets: pl.DataFrame) -> pl.DataFrame:
        home, away = [], []
        for r in targets.iter_rows(named=True):
            # Unseen teams (newly promoted with no recent history) start at average.
            ah, aa = (
                self.attack.get(r["home_team_key"], 0.0),
                self.attack.get(r["away_team_key"], 0.0),
            )
            dh, da = (
                self.defence.get(r["home_team_key"], 0.0),
                self.defence.get(r["away_team_key"], 0.0),
            )
            ph, pa = float(r["home_promoted"]), float(r["away_promoted"])
            p_att, p_def = self.promoted_attack, self.promoted_defence
            home.append(np.exp(self.mu + self.home + ah - da + p_att * ph - p_def * pa))
            away.append(np.exp(self.mu + aa - dh + p_att * pa - p_def * ph))
        return _frame(home, away)


@dataclass(frozen=True)
class MarketWithRatingsFallback:
    """Production model: market lambdas where opening odds exist, ratings elsewhere.

    The market wins whenever it has priced a match; ratings cover the fixtures weeks
    ahead that bookmakers haven't priced yet.
    """

    ratings: "Ratings" = field(default_factory=Ratings)
    name: str = "market_else_ratings"

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
        market = MarketOdds().predict(history, targets)
        rated = self.ratings.predict(history, targets)
        use_market = targets["p_home_win"].is_not_null()
        return pl.DataFrame(
            {
                side: market[side].zip_with(use_market, rated[side])
                for side in ("lambda_home", "lambda_away")
            }
        )


@dataclass(frozen=True)
class Blend:
    """Geometric mean of two models' lambdas (equal weight in log space)."""

    first: TeamModel
    second: TeamModel
    weight: float = 0.5

    @property
    def name(self) -> str:
        return f"blend_{self.first.name}_{self.second.name}_w{self.weight:g}"

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
        a = self.first.predict(history, targets)
        b = self.second.predict(history, targets)
        w = self.weight
        return a.select(
            ((pl.col("lambda_home").log() * w) + (b["lambda_home"].log() * (1 - w))).exp(),
            ((pl.col("lambda_away").log() * w) + (b["lambda_away"].log() * (1 - w))).exp(),
        )


# --- backtest -------------------------------------------------------------------------


def run_match_backtest(
    model: TeamModel, matches: pl.DataFrame, seasons: Sequence[str], horizon: int = 0
) -> pl.DataFrame:
    """Predict every played match of `seasons` from strictly pre-deadline history.

    `horizon` = k predicts each gameweek from information as of the deadline k
    gameweeks earlier (planning ahead). For k > 0 odds are withheld: bookmakers
    don't price fixtures that far out.
    """
    evaluated = matches.filter(pl.col("season").is_in(list(seasons)) & pl.col("is_played"))
    deadlines = matches.select("deadline_utc").drop_nulls().unique().sort("deadline_utc")
    ordered = deadlines["deadline_utc"].to_list()
    frames = []
    for (season, gameweek), _ in evaluated.group_by("season", "gameweek", maintain_order=True):
        targets = evaluated.filter((pl.col("season") == season) & (pl.col("gameweek") == gameweek))
        deadline = targets["deadline_utc"].min()
        cutoff = ordered[max(ordered.index(deadline) - horizon, 0)]
        history = matches.filter(pl.col("is_played") & (pl.col("kickoff_utc") < cutoff))
        context = targets.select(CONTEXT_COLUMNS).with_columns(deadline_utc=pl.lit(cutoff))
        if horizon:
            context = context.with_columns(
                pl.lit(None, pl.Float64).alias(c)
                for c in ("p_home_win", "p_draw", "p_away_win", "p_over_2_5")
            )
        predicted = model.predict(history, context)
        if predicted.height != targets.height:
            raise ValueError(f"{model.name}: {predicted.height} predictions for {targets.height}")
        frames.append(pl.concat([targets, predicted], how="horizontal"))
    return pl.concat(frames)


def evaluate_matches(predictions: pl.DataFrame) -> dict[str, float]:
    """Log losses (outcome, clean sheets, over 2.5), outcome Brier and goals deviance."""
    eps = 1e-12
    outcome_ll, outcome_brier, cs_ll, over_ll, deviance = [], [], [], [], []
    for r in predictions.iter_rows(named=True):
        lh, la, hg, ag = r["lambda_home"], r["lambda_away"], r["home_goals"], r["away_goals"]
        p = outcome_probabilities(lh, la)
        actual = "home_win" if hg > ag else "away_win" if ag > hg else "draw"
        outcome_ll.append(-np.log(max(p[actual], eps)))
        outcome_brier.append(
            sum((p[k] - (k == actual)) ** 2 for k in ("home_win", "draw", "away_win"))
        )
        for side, conceded in (("home_clean_sheet", ag), ("away_clean_sheet", hg)):
            q = min(max(p[side], eps), 1 - eps)
            cs_ll.append(-np.log(q if conceded == 0 else 1 - q))
        q = min(max(p["over_2_5"], eps), 1 - eps)
        over_ll.append(-np.log(q if hg + ag > 2.5 else 1 - q))
        for y, lam in ((hg, lh), (ag, la)):
            deviance.append(2 * ((y * np.log(y / lam) if y > 0 else 0.0) - (y - lam)))
    return {
        "outcome_log_loss": float(np.mean(outcome_ll)),
        "outcome_brier": float(np.mean(outcome_brier)),
        "clean_sheet_log_loss": float(np.mean(cs_ll)),
        "over_2_5_log_loss": float(np.mean(over_ll)),
        "goals_deviance": float(np.mean(deviance)),
        "matches": float(predictions.height),
    }


# --- leaderboard ----------------------------------------------------------------------

PRIMARY_METRIC = "goals_deviance"
EVALUATION_SEASONS = ("2023-24", "2024-25", "2025-26")
MODELS: tuple[TeamModel, ...] = (
    LeagueAverage(),
    Ratings(),
    MarketOdds(),
    MarketWithRatingsFallback(),
)
HORIZON_MODELS: tuple[TeamModel, ...] = (LeagueAverage(), Ratings())
HORIZONS = (1, 3, 6)


def leaderboard(matches: pl.DataFrame, seasons: Sequence[str] = EVALUATION_SEASONS) -> pl.DataFrame:
    """Next-gameweek scores for every model, plus planning-horizon scores (no odds)."""
    rows = [
        {
            "model": model.name,
            "horizon": 0,
            **evaluate_matches(run_match_backtest(model, matches, seasons)),
        }
        for model in MODELS
    ]
    rows += [
        {
            "model": model.name,
            "horizon": k,
            **evaluate_matches(run_match_backtest(model, matches, seasons, k)),
        }
        for model in HORIZON_MODELS
        for k in HORIZONS
    ]
    return pl.DataFrame(rows).sort("horizon", PRIMARY_METRIC)


def fixture_expectations(
    matches: pl.DataFrame, as_of: datetime, model: TeamModel | None = None
) -> pl.DataFrame:
    """Lambdas and probabilities for every unplayed fixture, using data before `as_of`.

    `source` says whether a fixture's lambdas come from the market or the ratings.
    """
    model = model or MarketWithRatingsFallback()
    history = matches.filter(pl.col("is_played") & (pl.col("kickoff_utc") < as_of))
    upcoming = matches.filter(~pl.col("is_played")).with_columns(deadline_utc=pl.lit(as_of))
    if upcoming.height == 0:
        return pl.DataFrame()
    predicted = model.predict(history, upcoming.select(CONTEXT_COLUMNS))
    probabilities = pl.DataFrame(
        [outcome_probabilities(lh, la) for lh, la in predicted.iter_rows()]
    )
    return pl.concat(
        [
            upcoming.select(
                "match_key", "season", "gameweek", "kickoff_utc", "home_team_key", "away_team_key"
            ),
            predicted,
            probabilities,
        ],
        how="horizontal",
    ).with_columns(
        pl.when(upcoming["p_home_win"].is_not_null())
        .then(pl.lit("market"))
        .otherwise(pl.lit("ratings"))
        .alias("source"),
        pl.lit(as_of).alias("as_of"),
    )
