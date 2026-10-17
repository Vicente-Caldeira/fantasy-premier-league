"""Component model: expected FPL points from predicted scoring events.

Per player-fixture, gradient-boosted models predict
    minutes class (0 / 1-59 / 60+)   -> appearance points, and P(60+) for clean sheets
    goals, assists, bonus, saves,    -> Poisson regressions (expected counts)
    yellow cards
    defensive contribution           -> P(reaching the CBIT / CBIRT threshold)
and the team-strength model supplies the opponent's expected goals, giving
    P(clean sheet) = P(60+) * exp(-lambda_opponent)
    E[goals-conceded penalty] = P(60+) * E[floor(G / 2)],  G ~ Poisson(lambda_opponent).
Events are valued with `fpl_core.scoring` constants, under the rules of the season
being predicted (defensive contributions only exist from 2025-26), and summed over a
player's fixtures in the gameweek.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from fpl_core import scoring
from fpl_models.player_features import FEATURES, POSITIONS, feature_matrix, latest_player_state

DEFCON_FROM_SEASON = "2025-26"
MAX_COUNT = 20
GOAL_POINTS = {p: scoring.GOAL_POINTS[scoring.Position[p]] for p in POSITIONS}
CLEAN_SHEET_POINTS = {p: scoring.CLEAN_SHEET_POINTS[scoring.Position[p]] for p in POSITIONS}
CONCEDING = {p.name for p in scoring.CONCEDING_POSITIONS}
CATEGORICAL = [FEATURES.index("position_code")]
COUNT_TARGETS = ("goals_scored", "assists", "bonus", "saves", "yellow_cards")


class Masked:
    """Estimator wrapper that masks feature columns empty in the training data.

    HistGradientBoosting handles missing values, but crashes on a column that is
    missing in *every* training row (e.g. xG in seasons before it existed). Those
    columns are zeroed at fit time and identically at predict time.
    """

    def __init__(
        self, estimator: HistGradientBoostingClassifier | HistGradientBoostingRegressor
    ) -> None:
        self.estimator = estimator
        self.empty: np.ndarray | None = None

    def _mask(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[:, self.empty] = 0.0
        return x

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Masked":
        self.empty = np.isnan(x).all(axis=0)
        self.estimator.fit(self._mask(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        result: np.ndarray = self.estimator.predict(self._mask(x))
        return result

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        result: np.ndarray = self.estimator.predict_proba(self._mask(x))
        return result

    @property
    def classes_(self) -> np.ndarray:
        classes: np.ndarray = self.estimator.classes_
        return classes


def _regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=200, learning_rate=0.05, categorical_features=CATEGORICAL,
        random_state=0,
    )  # fmt: skip


def _classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, categorical_features=CATEGORICAL, random_state=0
    )


def expected_floor_div(mean: np.ndarray, divisor: int) -> np.ndarray:
    """E[floor(X / divisor)] for X ~ Poisson(mean), per element."""
    k = np.arange(MAX_COUNT * divisor + 1)
    pmf = poisson.pmf(k[None, :], np.asarray(mean, dtype=np.float64)[:, None])
    result: np.ndarray = (pmf * (k // divisor)[None, :]).sum(axis=1)
    return result


@dataclass
class FittedComponents:
    minutes: Masked
    counts: dict[str, Masked]
    defcon: Masked | None  # None: no CBIT-labelled history yet

    def components(self, rows: pl.DataFrame) -> pl.DataFrame:
        """Expected events per fixture row."""
        x = feature_matrix(rows)
        p_minutes = self.minutes.predict_proba(x)
        classes = list(self.minutes.classes_)
        out = {
            "p_play": 1 - p_minutes[:, classes.index(0)],
            "p_60": p_minutes[:, classes.index(2)],
            "p_1_59": p_minutes[:, classes.index(1)],
            **{f"e_{t}": model.predict(x) for t, model in self.counts.items()},
            "p_defcon": (self.defcon.predict_proba(x)[:, 1] if self.defcon else np.zeros(len(x))),
        }
        return pl.DataFrame(out)


def _minutes_class(minutes: pl.Series) -> np.ndarray:
    return np.where(minutes.to_numpy() == 0, 0, np.where(minutes.to_numpy() >= 60, 2, 1))


def fit_components(train: pl.DataFrame, defcon_train: pl.DataFrame) -> FittedComponents:
    """`defcon_train`: CBIT-labelled outfield rows, which exist only in some seasons."""
    x = feature_matrix(train)
    minutes = Masked(_classifier()).fit(x, _minutes_class(train["minutes"]))
    counts = {}
    for target in COUNT_TARGETS:
        rows = train.filter(pl.col("position") == "GKP") if target == "saves" else train
        counts[target] = Masked(_regressor()).fit(feature_matrix(rows), rows[target].to_numpy())
    defcon = None
    if defcon_train.height:
        labels = defcon_train["defcon_hit"].cast(pl.Int8).to_numpy()
        defcon = Masked(_classifier()).fit(feature_matrix(defcon_train), labels)
    return FittedComponents(minutes, counts, defcon)


def expected_points(rows: pl.DataFrame, events: pl.DataFrame) -> pl.Series:
    """Value expected events per fixture with the scoring rules of each row's season."""
    frame = pl.concat(
        [rows.select("season", "position", "opponent_lambda"), events], how="horizontal"
    )
    position = frame["position"].to_list()
    goal_pts = np.array([GOAL_POINTS[p] for p in position])
    cs_pts = np.array([CLEAN_SHEET_POINTS[p] for p in position])
    conceding = np.array([p in CONCEDING for p in position])
    is_gk = np.array([p == "GKP" for p in position])
    defcon_on = (frame["season"] >= DEFCON_FROM_SEASON).to_numpy() & ~is_gk
    lam_opp = frame["opponent_lambda"].to_numpy()
    p60, p1_59 = frame["p_60"].to_numpy(), frame["p_1_59"].to_numpy()

    points = (
        p1_59 * scoring.SHORT_PLAY_POINTS
        + p60 * scoring.LONG_PLAY_POINTS
        + frame["e_goals_scored"].to_numpy() * goal_pts
        + frame["e_assists"].to_numpy() * scoring.ASSIST_POINTS
        + p60 * np.exp(-lam_opp) * cs_pts
        - np.where(conceding, p60 * expected_floor_div(lam_opp, 2), 0.0)
        + np.where(
            is_gk, expected_floor_div(frame["e_saves"].to_numpy(), scoring.SAVES_PER_POINT), 0.0
        )
        + np.where(
            defcon_on, frame["p_defcon"].to_numpy() * scoring.DEFENSIVE_CONTRIBUTION_POINTS, 0.0
        )
        + frame["e_bonus"].to_numpy()
        + frame["e_yellow_cards"].to_numpy() * scoring.YELLOW_CARD_POINTS
    )
    return pl.Series("expected_points", points)


@dataclass
class ComponentModel:
    """Backtest predictor: refits the components every `refit_every` gameweeks.

    `features` holds every fixture row with pre-deadline features (see
    player_features). Training uses only rows whose deadline is strictly before the
    gameweek being predicted; the target rows' outcomes are never read.
    """

    features: pl.DataFrame
    refit_every: int = 4
    train_seasons: int = 4
    name: str = "components_gbm"
    _fitted: FittedComponents | None = field(default=None, repr=False)
    _fitted_at: int = field(default=-(10**9), repr=False)

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.Series:
        deadline = targets["deadline_utc"][0]
        season, gameweek = targets["season"][0], targets["gameweek"][0]
        step = self._step(deadline)
        if self._fitted is None or step - self._fitted_at >= self.refit_every:
            seasons = sorted(history["season"].unique().to_list())[-self.train_seasons :]
            before = self.features.filter(pl.col("deadline_utc") < deadline)
            train = before.filter(pl.col("season").is_in(seasons))
            # DefCon labels exist only in some seasons, so use all labelled history.
            defcon_train = before.filter(
                pl.col("defcon_hit").is_not_null() & (pl.col("position") != "GKP")
            )
            self._fitted, self._fitted_at = fit_components(train, defcon_train), step
        rows = self.features.filter((pl.col("season") == season) & (pl.col("gameweek") == gameweek))
        points = expected_points(rows, self._fitted.components(rows))
        per_player = (
            rows.select("player_code")
            .with_columns(points)
            .group_by("player_code")
            .agg(pl.col("expected_points").sum())
        )
        return targets.join(per_player, on="player_code", how="left", maintain_order="left")[
            "expected_points"
        ].fill_null(0.0)

    def _step(self, deadline: object) -> int:
        order = self.features.select("deadline_utc").unique().sort("deadline_utc")["deadline_utc"]
        return int((order < deadline).sum())


# --- live predictions -----------------------------------------------------------------

EVENT_COLUMNS = (
    "p_play", "p_60", "p_1_59", *(f"e_{t}" for t in COUNT_TARGETS), "p_defcon",
)  # fmt: skip


def upcoming_fixture_rows(
    squads: pl.DataFrame, fixtures: pl.DataFrame, gameweeks: list[int]
) -> pl.DataFrame:
    """One row per squad player per upcoming fixture of his club in `gameweeks`.

    `fixtures` are team-strength fixture expectations (match_key, gameweek, teams,
    lambda_home, lambda_away).
    """
    selected = fixtures.filter(pl.col("gameweek").is_in(gameweeks))
    sides = []
    for side, other, own, opponent in (
        ("home_team_key", "away_team_key", "lambda_home", "lambda_away"),
        ("away_team_key", "home_team_key", "lambda_away", "lambda_home"),
    ):
        sides.append(
            selected.select(
                "match_key",
                "season",
                "gameweek",
                "kickoff_utc",
                pl.col(side).alias("team_key"),
                pl.col(other).alias("opponent_team_key"),
                pl.lit(side == "home_team_key").alias("is_home"),
                pl.col(own).alias("team_lambda"),
                pl.col(opponent).alias("opponent_lambda"),
            )
        )
    rows = pl.concat(sides).join(
        squads.select("player_code", "web_name", "position", "team_key", "price", "status",
                      "chance_of_playing_next_round"),
        on="team_key",
    )  # fmt: skip
    return rows.with_columns(
        (-pl.col("opponent_lambda")).exp().alias("p_clean_sheet"),
        pl.len().over("player_code", "gameweek").alias("n_fixtures"),
        pl.col("position")
        .replace_strict({p: i for i, p in enumerate(POSITIONS)}, return_dtype=pl.Int8)
        .alias("position_code"),
    )


def availability(rows: pl.DataFrame, next_gameweek: int) -> pl.Series:
    """Multiplier on every expected event, from FPL's current availability flags.

    Availability has no history (FPL only exposes the current status), so it can't
    be learned; it is applied here instead. `chance_of_playing_next_round` scales the
    next gameweek; players flagged unavailable (`u`, e.g. left the club) are zeroed
    for every gameweek.
    """
    chance = rows["chance_of_playing_next_round"].cast(pl.Float64) / 100
    is_next = rows["gameweek"] == next_gameweek
    return (
        pl.select(
            pl.when(rows["status"] == "u")
            .then(0.0)
            .when(is_next & chance.is_not_null())
            .then(chance)
            .otherwise(1.0)
        )
        .to_series()
        .alias("availability")
    )


def predict_upcoming(
    features: pl.DataFrame,
    squads: pl.DataFrame,
    fixtures: pl.DataFrame,
    gameweeks: list[int],
    *,
    train_seasons: int = 4,
) -> pl.DataFrame:
    """Expected points per player for each of `gameweeks` (deadlines still to come).

    `features` are all played fixture rows (from `build_features`): the training data,
    and the history each player's latest state is computed from. Trains on the latest
    `train_seasons` seasons (DefCon on all labelled history).
    """
    if not gameweeks:
        raise ValueError("no upcoming gameweeks to predict")
    seasons = sorted(features["season"].unique().to_list())[-train_seasons:]
    defcon_train = features.filter(
        pl.col("defcon_hit").is_not_null() & (pl.col("position") != "GKP")
    )
    fitted = fit_components(features.filter(pl.col("season").is_in(seasons)), defcon_train)

    rows = upcoming_fixture_rows(squads, fixtures, gameweeks).join(
        latest_player_state(features), on="player_code", how="left"
    )
    factor = availability(rows, gameweeks[0])
    events = fitted.components(rows).with_columns(pl.col(EVENT_COLUMNS) * factor)
    points = expected_points(rows, events)
    per_fixture = pl.concat([rows, events, factor.to_frame()], how="horizontal").with_columns(
        points
    )
    return (
        per_fixture.group_by("player_code", "web_name", "position", "team_key", "price", "gameweek")
        .agg(
            pl.col("expected_points").sum(),
            pl.len().alias("fixtures"),
            pl.col("opponent_team_key").sort().alias("opponents"),
            pl.col("p_play").mean(),
            pl.col("p_60").mean(),
            pl.col("e_goals_scored").sum(),
            pl.col("e_assists").sum(),
            (pl.col("p_60") * pl.col("p_clean_sheet")).mean().alias("p_clean_sheet"),
            pl.col("availability").min(),
        )
        .sort("gameweek", "expected_points", descending=[False, True])
    )
