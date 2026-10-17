"""Player-fixture features, built from information available before each deadline.

Rolling statistics are computed per player over *previous gameweeks only*: stats are
summed to player-gameweeks, smoothed in deadline order, then shifted by one gameweek.
So both fixtures of a double gameweek see the same pre-deadline state, and nothing
from the gameweek being predicted (or later) can reach its features. A test proves
this by truncating the data and checking that features are unchanged.

Team context comes from the team-strength model: each side's expected goals (lambda)
implied by the fixture's opening odds, which are known before the deadline.
"""

from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from fpl_models import team_strength

DEADLINE_BEFORE_KICKOFF = timedelta(minutes=90)
EWM_HALF_LIFE_GWS = 8.0
POSITIONS = ("GKP", "DEF", "MID", "FWD")

# Per-gameweek sums smoothed into per-90 rates.
RATE_STATS = (
    "goals_scored", "expected_goals", "assists", "expected_assists", "bonus", "bps",
    "saves", "yellow_cards", "cbit", "cbirt", "total_points",
)  # fmt: skip

FEATURES = (
    "position_code", "is_home", "price", "n_fixtures",
    "team_lambda", "opponent_lambda", "p_clean_sheet",
    "minutes_l1", "minutes_l3", "minutes_l6", "appear_rate_l6", "played60_rate_l6",
    "starts_rate_l6", "minutes_exposure_l10", "gws_since_appearance", "gws_in_history",
    *(f"{s}_per90" for s in RATE_STATS), "points_ewm",
)  # fmt: skip

TARGETS = (
    "minutes",
    "goals_scored",
    "assists",
    "bonus",
    "saves",
    "yellow_cards",
    "defcon_hit",
)


def load_fixture_rows(warehouse: Path) -> pl.DataFrame:
    """Played player-fixture rows with outcomes, match context and deadlines."""
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        connection.execute("set TimeZone = 'UTC'")
        rows = connection.sql(
            """
            select pm.season, pm.gameweek, pm.kickoff_utc, pm.match_key, pm.player_code,
                pm.position, pm.team_key, pm.is_home, pm.price, pm.is_provisional,
                pm.minutes, pm.starts, pm.goals_scored, pm.assists, pm.bonus, pm.bps,
                pm.saves, pm.yellow_cards, pm.total_points, pm.expected_goals,
                pm.expected_assists, pm.tackles, pm.clearances_blocks_interceptions,
                pm.recoveries
            from fct_player_matches as pm
            """
        ).pl()
    return add_deadlines(rows).filter(~pl.col("is_provisional"))


def add_deadlines(rows: pl.DataFrame) -> pl.DataFrame:
    deadlines = rows.group_by("season", "gameweek").agg(
        (pl.col("kickoff_utc").min() - DEADLINE_BEFORE_KICKOFF).alias("deadline_utc")
    )
    return rows.join(deadlines, on=["season", "gameweek"], how="left", maintain_order="left")


def match_lambdas(matches: pl.DataFrame) -> pl.DataFrame:
    """Opening-odds lambdas per match (pre-deadline information), league mean if unpriced."""
    rows = []
    fallback = matches.filter(pl.col("is_played"))
    home_mean = float(fallback["home_goals"].mean() or 1.5)  # type: ignore[arg-type]
    away_mean = float(fallback["away_goals"].mean() or 1.2)  # type: ignore[arg-type]
    for r in matches.iter_rows(named=True):
        if r["p_home_win"] is None:
            lh, la = home_mean, away_mean
        else:
            lh, la = team_strength.lambdas_from_odds(
                r["p_home_win"], r["p_draw"], r["p_away_win"], r["p_over_2_5"]
            )
        rows.append({"match_key": r["match_key"], "lambda_home": lh, "lambda_away": la})
    return pl.DataFrame(rows)


def add_team_context(rows: pl.DataFrame, lambdas: pl.DataFrame) -> pl.DataFrame:
    """Each row's team and opponent expected goals, and clean-sheet probability."""
    return (
        rows.join(lambdas, on="match_key", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("is_home"))
            .then(pl.col("lambda_home"))
            .otherwise(pl.col("lambda_away"))
            .alias("team_lambda"),
            pl.when(pl.col("is_home"))
            .then(pl.col("lambda_away"))
            .otherwise(pl.col("lambda_home"))
            .alias("opponent_lambda"),
        )
        .with_columns((-pl.col("opponent_lambda")).exp().alias("p_clean_sheet"))
        .drop("lambda_home", "lambda_away")
    )


def _with_defensive_stats(rows: pl.DataFrame) -> pl.DataFrame:
    cbit = pl.col("tackles") + pl.col("clearances_blocks_interceptions")
    cbirt = cbit + pl.col("recoveries")
    hit = (
        pl.when(pl.col("position") == "GKP")
        .then(False)
        .when(pl.col("position") == "DEF")
        .then(cbit >= 10)
        .otherwise(cbirt >= 12)
    )
    return rows.with_columns(cbit.alias("cbit"), cbirt.alias("cbirt"), hit.alias("defcon_hit"))


ROLLED = tuple(c for c in FEATURES if c not in (
    "position_code", "is_home", "price", "n_fixtures",
    "team_lambda", "opponent_lambda", "p_clean_sheet",
))  # fmt: skip


def _player_state(rows: pl.DataFrame) -> pl.DataFrame:
    """Per (player, gameweek): rolling state *including* that gameweek (unshifted)."""
    # Nulls (stat not recorded that season) must stay null rather than sum to 0.
    sums = [
        pl.when(pl.col(s).is_null().all()).then(None).otherwise(pl.col(s).sum()).alias(s)
        for s in ("minutes", *RATE_STATS)
    ]
    per_gw = (
        _with_defensive_stats(rows)
        .group_by("player_code", "season", "gameweek", "deadline_utc")
        .agg(
            *sums,
            pl.len().alias("fixtures"),
            (pl.col("minutes") > 0).sum().alias("appearances"),
            (pl.col("minutes") >= 60).sum().alias("played60"),
            pl.col("starts").sum().alias("starts"),
        )
        .sort("player_code", "deadline_utc")
        .with_columns(pl.int_range(pl.len()).over("player_code").alias("gw_index"))
    )

    def per_fixture(col: str, n: int) -> pl.Expr:
        window = pl.col(col).rolling_sum(n, min_samples=1)
        return (window / pl.col("fixtures").rolling_sum(n, min_samples=1)).over("player_code")

    alpha = 1 - 0.5 ** (1 / EWM_HALF_LIFE_GWS)

    def ewm(col: str) -> pl.Expr:
        return pl.col(col).ewm_mean(alpha=alpha, ignore_nulls=True).over("player_code")

    last_appearance = pl.when(pl.col("appearances") > 0).then(pl.col("gw_index"))
    return per_gw.with_columns(
        per_fixture("minutes", 1).alias("minutes_l1"),
        per_fixture("minutes", 3).alias("minutes_l3"),
        per_fixture("minutes", 6).alias("minutes_l6"),
        per_fixture("appearances", 6).alias("appear_rate_l6"),
        per_fixture("played60", 6).alias("played60_rate_l6"),
        per_fixture("starts", 6).alias("starts_rate_l6"),
        pl.col("minutes")
        .rolling_sum(10, min_samples=1)
        .over("player_code")
        .alias("minutes_exposure_l10"),
        *[(ewm(s) / ewm("minutes") * 90).alias(f"{s}_per90") for s in RATE_STATS],
        ewm("total_points").alias("points_ewm"),
        (pl.col("gw_index") - last_appearance.forward_fill().over("player_code")).alias(
            "gws_since_appearance"
        ),
        (pl.col("gw_index") + 1).alias("gws_in_history"),
    ).select("player_code", "deadline_utc", *ROLLED)


def player_history_features(rows: pl.DataFrame) -> pl.DataFrame:
    """Per (player, gameweek): the state *before* that gameweek (shifted by one)."""
    state = _player_state(rows)
    return state.select(
        "player_code",
        "deadline_utc",
        *[pl.col(c).shift(1).over("player_code") for c in ROLLED],
    )


def latest_player_state(rows: pl.DataFrame) -> pl.DataFrame:
    """Each player's state after his last gameweek: the features for the next one."""
    return (
        _player_state(rows)
        .group_by("player_code")
        .agg(pl.all().sort_by("deadline_utc").last())
        .drop("deadline_utc")
    )


def build_features(rows: pl.DataFrame, lambdas: pl.DataFrame) -> pl.DataFrame:
    """Played fixture rows with pre-deadline features and outcome targets."""
    rows = _with_defensive_stats(add_team_context(rows, lambdas))
    history = player_history_features(rows)
    position_code = pl.col("position").replace_strict(
        {p: i for i, p in enumerate(POSITIONS)}, return_dtype=pl.Int8
    )
    n_fixtures = pl.len().over("player_code", "season", "gameweek").alias("n_fixtures")
    return rows.with_columns(position_code.alias("position_code"), n_fixtures).join(
        history, on=["player_code", "deadline_utc"], how="left", maintain_order="left"
    )


def feature_matrix(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(pl.col(FEATURES).cast(pl.Float64)).to_numpy()
