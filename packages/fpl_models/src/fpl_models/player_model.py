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

from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from fpl_core import scoring
from fpl_models import simulator
from fpl_models.player_features import (
    CROWD_ALL,
    FEATURES,
    POSITIONS,
    SCHEDULE_FEATURES,
    add_schedule,
    feature_matrix,
    latest_player_state,
)

DEFCON_FROM_SEASON = "2025-26"
MAX_COUNT = 20
GOAL_POINTS = {p: scoring.GOAL_POINTS[scoring.Position[p]] for p in POSITIONS}
CLEAN_SHEET_POINTS = {p: scoring.CLEAN_SHEET_POINTS[scoring.Position[p]] for p in POSITIONS}
CONCEDING = {p.name for p in scoring.CONCEDING_POSITIONS}
POSITION_FEATURE = "position_code"
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

    def fit(
        self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "Masked":
        self.empty = np.isnan(x).all(axis=0)
        self.estimator.fit(self._mask(x), y, sample_weight=sample_weight)
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


@dataclass(frozen=True)
class ModelConfig:
    """Everything an experiment may vary. Defaults are the original champion."""

    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    # Training-row weight halves every this many days (None: all rows equal).
    recency_half_life_days: float | None = None
    train_seasons: int = 4
    features: tuple[str, ...] = FEATURES
    # Train goals/assists on a blend of actual and expected (xG/xA, where FPL has them,
    # 2022-23+): 1.0 = expected only (least noise), 0.0 = actual only. Pure xG
    # under-rewards finishers who consistently beat it, so a blend may be best.
    denoise_weight: float = 0.0
    # Rescale players' expected goals/assists so each team's add up to its expected
    # goals (live GW8: player sums ranged from 57% to 126% of team lambda).
    reconcile: bool = False

    def gbm_params(self) -> dict[str, object]:
        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "categorical_features": [self.features.index(POSITION_FEATURE)],
            "random_state": 0,
        }


# Chosen on the tuning seasons (docs/model-experiments.md, rounds 1-4 + season simulation).
PRODUCTION = ModelConfig(
    recency_half_life_days=180,
    l2_regularization=1.0,
    min_samples_leaf=100,
    learning_rate=0.03,
    max_iter=400,
    denoise_weight=1.0,
    features=(
        *FEATURES,
        "crowd_ownership",
        "crowd_net_transfer_share",
        *SCHEDULE_FEATURES,
    ),
)


def _regressor(config: ModelConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="poisson", **config.gbm_params())


def _classifier(config: ModelConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**config.gbm_params())


def sample_weights(rows: pl.DataFrame, config: ModelConfig) -> np.ndarray | None:
    """Exponential decay by age relative to the newest training row."""
    if config.recency_half_life_days is None:
        return None
    newest = rows["deadline_utc"].max()
    age = rows.select(
        (pl.lit(newest) - pl.col("deadline_utc")).dt.total_seconds() / 86_400
    ).to_series()
    weights: np.ndarray = 0.5 ** (age.to_numpy() / config.recency_half_life_days)
    return weights


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
    features: tuple[str, ...] = FEATURES
    # Team goals scored / assisted by players, per unit of team lambda (from training).
    team_rates: dict[str, float] | None = None

    def components(self, rows: pl.DataFrame) -> pl.DataFrame:
        """Expected events per fixture row."""
        x = feature_matrix(rows, self.features)
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


def fit_components(
    train: pl.DataFrame, defcon_train: pl.DataFrame, config: ModelConfig | None = None
) -> FittedComponents:
    """`defcon_train`: CBIT-labelled outfield rows, which exist only in some seasons."""
    config = config or ModelConfig()

    def fit(model: Masked, rows: pl.DataFrame, y: np.ndarray) -> Masked:
        return model.fit(feature_matrix(rows, config.features), y, sample_weights(rows, config))

    minutes = fit(Masked(_classifier(config)), train, _minutes_class(train["minutes"]))
    denoised = {"goals_scored": "expected_goals", "assists": "expected_assists"}
    counts = {}
    for target in COUNT_TARGETS:
        rows = train.filter(pl.col("position") == "GKP") if target == "saves" else train
        label = rows[target].cast(pl.Float64)
        if config.denoise_weight and target in denoised:
            expected = rows[denoised[target]].cast(pl.Float64).fill_null(label)
            label = config.denoise_weight * expected + (1 - config.denoise_weight) * label
        counts[target] = fit(Masked(_regressor(config)), rows, label.to_numpy())
    defcon = None
    if defcon_train.height:
        labels = defcon_train["defcon_hit"].cast(pl.Int8).to_numpy()
        defcon = fit(Masked(_classifier(config)), defcon_train, labels)
    return FittedComponents(minutes, counts, defcon, config.features, team_rates(train))


def team_rates(train: pl.DataFrame) -> dict[str, float]:
    """Player goals and assists per unit of team lambda, over training team-fixtures.

    Below 1 for goals because own goals count for the team but no player; well below 1
    for assists because not every goal has an FPL assist.
    """
    per_team = train.group_by("match_key", "team_key").agg(
        pl.col("goals_scored").sum(), pl.col("assists").sum(), pl.col("team_lambda").first()
    )
    lam = float(per_team["team_lambda"].sum())
    return {
        "e_goals_scored": float(per_team["goals_scored"].sum()) / lam,
        "e_assists": float(per_team["assists"].sum()) / lam,
    }


def reconcile(rows: pl.DataFrame, events: pl.DataFrame, rates: dict[str, float]) -> pl.DataFrame:
    """Scale each team-fixture's player expectations to rate x team lambda."""
    frame = pl.concat(
        [rows.select("match_key", "team_key", "team_lambda"), events], how="horizontal"
    )
    scaled = frame.with_columns(
        (
            pl.col(col)
            * rate
            * pl.col("team_lambda")
            / pl.col(col).sum().over("match_key", "team_key").clip(lower_bound=1e-9)
        ).alias(col)
        for col, rate in rates.items()
    )
    return scaled.select(events.columns)


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
    config: ModelConfig = field(default_factory=ModelConfig)
    refit_every: int = 4
    name: str = "components_gbm"
    _fitted: FittedComponents | None = field(default=None, repr=False)
    _fitted_at: int = field(default=-(10**9), repr=False)

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.Series:
        deadline = targets["deadline_utc"][0]
        season, gameweek = targets["season"][0], targets["gameweek"][0]
        step = self._step(deadline)
        if self._fitted is None or step - self._fitted_at >= self.refit_every:
            seasons = sorted(history["season"].unique().to_list())[-self.config.train_seasons :]
            before = self.features.filter(pl.col("deadline_utc") < deadline)
            train = before.filter(pl.col("season").is_in(seasons))
            # DefCon labels exist only in some seasons, so use all labelled history.
            defcon_train = before.filter(
                pl.col("defcon_hit").is_not_null() & (pl.col("position") != "GKP")
            )
            self._fitted = fit_components(train, defcon_train, self.config)
            self._fitted_at = step
        rows = self.features.filter((pl.col("season") == season) & (pl.col("gameweek") == gameweek))
        events = self._fitted.components(rows)
        if self.config.reconcile and self._fitted.team_rates:
            events = reconcile(rows, events, self._fitted.team_rates)
        points = expected_points(rows, events)
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


def with_live_crowd(rows: pl.DataFrame, squads: pl.DataFrame, next_gameweek: int) -> pl.DataFrame:
    """Crowd signal for the open transfer window, from FPL's live counters.

    `transfers_in_event` / `transfers_out_event` count the window now open (verified:
    hours after the GW7 deadline they summed to 0.4M vs GW7's final 15.1M). A player's
    share of all activity so far is window-invariant, so it matches training data.
    Only the next gameweek has a window; later rows get nulls.
    """
    net = pl.col("transfers_in_event") - pl.col("transfers_out_event")
    live = squads.select(
        "player_code",
        (pl.col("selected_by_percent").cast(pl.Float64) / 100).alias("crowd_ownership"),
        (net / net.abs().sum().clip(lower_bound=1)).alias("crowd_net_transfer_share"),
    )
    columns = ("crowd_ownership", "crowd_net_transfer_share")
    joined = rows.drop([c for c in columns if c in rows.columns]).join(
        live, on="player_code", how="left", maintain_order="left"
    )
    is_next = pl.col("gameweek") == next_gameweek
    return joined.with_columns(pl.when(is_next).then(pl.col(c)).alias(c) for c in columns)


def predict_upcoming(  # noqa: PLR0913 - extras are keyword-only
    features: pl.DataFrame,
    squads: pl.DataFrame,
    fixtures: pl.DataFrame,
    gameweeks: list[int],
    *,
    config: ModelConfig | None = None,
    schedule: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Expected points per player for each of `gameweeks` (deadlines still to come).

    `features` are all played fixture rows (from `build_features`): the training data,
    and the history each player's latest state is computed from. Trains on the latest
    `config.train_seasons` seasons (DefCon on all labelled history). `schedule`
    (`player_features.schedule_features`) is required when the config uses it.
    """
    if not gameweeks:
        raise ValueError("no upcoming gameweeks to predict")
    config = config or ModelConfig()
    seasons = sorted(features["season"].unique().to_list())[-config.train_seasons :]
    train = features.filter(pl.col("season").is_in(seasons))
    defcon_train = features.filter(
        pl.col("defcon_hit").is_not_null() & (pl.col("position") != "GKP")
    )
    fitted = fit_components(train, defcon_train, config)
    rows = upcoming_fixture_rows(squads, fixtures, gameweeks).join(
        latest_player_state(features), on="player_code", how="left"
    )
    rows = with_live_crowd(rows, squads, next_gameweek=gameweeks[0])
    if schedule is not None:
        # The latest state carries the *last played* fixture's schedule: replace it.
        rows = add_schedule(rows.drop(SCHEDULE_FEATURES, strict=False), schedule)

    crowd = [c for c in config.features if c in CROWD_ALL]
    if crowd and len(gameweeks) > 1:
        # The crowd signal exists only for the next deadline's open transfer window.
        # Later gameweeks use a planning model trained without it.
        no_crowd = replace(config, features=tuple(c for c in config.features if c not in crowd))
        planning = fit_components(train, defcon_train, no_crowd)
        order = rows.with_row_index("_row")
        is_next = pl.col("gameweek") == gameweeks[0]
        nxt, later = order.filter(is_next), order.filter(~is_next)
        events = (
            pl.concat(
                [
                    fitted.components(nxt).with_columns(nxt["_row"]),
                    planning.components(later).with_columns(later["_row"]),
                ]
            )
            .sort("_row")
            .drop("_row")
        )
    else:
        events = fitted.components(rows)
    factor = availability(rows, gameweeks[0])
    events = events.with_columns(pl.col(EVENT_COLUMNS) * factor)
    if config.reconcile and fitted.team_rates:
        # After availability: an absent player's share of team goals moves to teammates.
        events = reconcile(rows, events, fitted.team_rates)
    points = expected_points(rows, events)
    per_fixture = pl.concat([rows, events, factor.to_frame()], how="horizontal").with_columns(
        points
    )
    settings = simulator.SimulationSettings(defcon_on=rows["season"][0] >= DEFCON_FROM_SEASON)
    distributions = simulator.per_player_gameweek(simulator.simulate(rows, events, settings), rows)
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
        .join(distributions.drop("sim_mean"), on=["player_code", "gameweek"], how="left")
        .sort("gameweek", "expected_points", descending=[False, True])
    )
