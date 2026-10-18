"""Model evaluation assets, downstream of the warehouse marts."""

import io
import os
from datetime import UTC, datetime
from pathlib import Path

import dagster as dg
import duckdb
import polars as pl

from fpl_models import optimizer, player_features, player_model, team_strength
from fpl_models.backtest import Predictor
from fpl_models.baselines import BASELINES
from fpl_models.data import load_player_matches, next_gameweeks, player_gameweeks
from fpl_models.leaderboard import EVALUATION_SEASONS, PRIMARY_METRIC, leaderboard, to_markdown
from fpl_models.plan import describe, latest_expected_points
from fpl_pipelines.lake import Lake


def _warehouse() -> Path:
    return Path(os.getenv("FPL_WAREHOUSE", "data/warehouse.duckdb"))


def _write_parquet(lake: Lake, path: str, frame: pl.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    lake.write_bytes(path, buffer.getvalue())
    return lake.uri(path)


HORIZON = 5  # gameweeks ahead for live predictions and plans

TEAM_SUMMARY_COLUMNS = (
    "model",
    "horizon",
    "goals_deviance",
    "clean_sheet_log_loss",
    "outcome_log_loss",
)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dg.asset(
    key=["gold", "backtests", "leaderboard"],
    deps=[dg.AssetKey("fct_player_matches")],
    group_name="models",
)
def backtest_leaderboard(
    context: dg.AssetExecutionContext, lake: Lake
) -> dg.MaterializeResult[None]:
    """Walk-forward backtest of every registered predictor, ranked by the primary metric.

    Each run is also logged to MLflow when MLFLOW_TRACKING_URI is set.
    """
    warehouse = _warehouse()
    matches = team_strength.load_matches(warehouse)
    features = player_features.build_features(
        player_features.load_fixture_rows(warehouse),
        player_features.match_lambdas(matches),
        player_features.schedule_features(matches),
    )
    predictors: list[Predictor] = [
        *BASELINES,
        player_model.ComponentModel(features, name="component_model_v1"),
        player_model.ComponentModel(
            features, config=player_model.PRODUCTION, name="component_model_v2"
        ),
    ]
    board = leaderboard(
        predictors, player_gameweeks(load_player_matches(warehouse)), EVALUATION_SEASONS
    )
    uri = _write_parquet(
        lake, f"gold/backtests/leaderboard/run_at={_stamp()}/leaderboard.parquet", board
    )

    champion = board.row(0, named=True)
    context.log.info(
        "Champion %s: %s=%.3f", champion["predictor"], PRIMARY_METRIC, champion[PRIMARY_METRIC]
    )
    return dg.MaterializeResult(
        metadata={
            "uri": uri,
            "champion": champion["predictor"],
            PRIMARY_METRIC: champion[PRIMARY_METRIC],
            "seasons": ", ".join(EVALUATION_SEASONS),
            "leaderboard": dg.MetadataValue.md(to_markdown(board)),
        }
    )


@dg.asset(
    key=["gold", "backtests", "team_strength"],
    deps=[dg.AssetKey("fct_matches")],
    group_name="models",
)
def team_strength_leaderboard(lake: Lake) -> dg.MaterializeResult[None]:
    """Match-level walk-forward backtest of team-strength models, incl. planning horizons."""
    board = team_strength.leaderboard(team_strength.load_matches(_warehouse()))
    uri = _write_parquet(
        lake, f"gold/backtests/team_strength/run_at={_stamp()}/leaderboard.parquet", board
    )
    best = board.filter(pl.col("horizon") == 0).row(0, named=True)
    return dg.MaterializeResult(
        metadata={
            "uri": uri,
            "champion": best["model"],
            team_strength.PRIMARY_METRIC: best[team_strength.PRIMARY_METRIC],
            "leaderboard": dg.MetadataValue.md(
                to_markdown(
                    board,
                    [
                        "model",
                        "horizon",
                        "goals_deviance",
                        "clean_sheet_log_loss",
                        "outcome_log_loss",
                    ],
                )
            ),
        }
    )


@dg.asset(
    key=["gold", "team_strength", "fixture_lambdas"],
    deps=[dg.AssetKey("fct_matches")],
    group_name="models",
)
def fixture_lambdas(lake: Lake) -> dg.MaterializeResult[None]:
    """Expected goals, outcome and clean-sheet probabilities for every unplayed fixture."""
    now = datetime.now(UTC)
    expectations = team_strength.fixture_expectations(
        team_strength.load_matches(_warehouse()), as_of=now
    )
    uri = _write_parquet(
        lake, f"gold/team_strength/fixture_lambdas/as_of={_stamp()}/data.parquet", expectations
    )
    sources = dict(expectations.group_by("source").len().iter_rows()) if expectations.height else {}
    return dg.MaterializeResult(
        metadata={
            "uri": uri,
            "fixtures": expectations.height,
            **{f"from_{k}": v for k, v in sources.items()},
        }
    )


@dg.asset(
    key=["gold", "player_points", "expected_points"],
    deps=[
        dg.AssetKey("fct_player_matches"),
        dg.AssetKey("current_squads"),
        dg.AssetKey("dim_gameweeks"),
        fixture_lambdas,
    ],
    group_name="models",
)
def expected_points(lake: Lake) -> dg.MaterializeResult[None]:
    """Expected points per player for each of the next five gameweeks (by FPL deadline).

    Component models trained on all history, applied to each squad player's latest
    state and the current team-strength fixture expectations, adjusted for FPL's
    availability flags.
    """
    warehouse = _warehouse()
    matches = team_strength.load_matches(warehouse)
    schedule = player_features.schedule_features(matches)
    features = player_features.build_features(
        player_features.load_fixture_rows(warehouse),
        player_features.match_lambdas(matches),
        schedule,
    )
    now = datetime.now(UTC)
    fixtures = team_strength.fixture_expectations(matches, as_of=now)
    gameweeks = next_gameweeks(warehouse, now, HORIZON)
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        squads = connection.sql("select * from current_squads").pl()
    predictions = player_model.predict_upcoming(
        features, squads, fixtures, gameweeks, config=player_model.PRODUCTION, schedule=schedule
    )

    path = f"gold/player_points/expected_points/as_of={_stamp()}/data.parquet"
    uri = _write_parquet(lake, path, predictions)
    next_gw = predictions.filter(pl.col("gameweek") == predictions["gameweek"].min())
    top = next_gw.head(10).select("web_name", "position", "team_key", "price", "expected_points")
    return dg.MaterializeResult(
        metadata={
            "uri": uri,
            "players": predictions["player_code"].n_unique(),
            "gameweeks": ", ".join(str(g) for g in predictions["gameweek"].unique().sort()),
            "top_next_gameweek": dg.MetadataValue.md(to_markdown(top, top.columns)),
        }
    )


@dg.asset(
    key=["gold", "plans", "wildcard"],
    deps=[expected_points, dg.AssetKey("current_squads")],
    group_name="models",
)
def wildcard_plan(lake: Lake) -> dg.MaterializeResult[None]:
    """The best squad from scratch and its lineups/captains/transfers over the horizon.

    Solved as a mixed-integer program on the latest expected points; every plan is
    checked against FPL's rules independently of the solver.
    """
    warehouse = _warehouse()
    gameweeks = next_gameweeks(warehouse, datetime.now(UTC), HORIZON)
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        squads = connection.sql("select * from current_squads").pl()
    xp = latest_expected_points(os.getenv("FPL_LAKE_URL", "data"), gameweeks)
    plan = optimizer.optimise(
        squads.select("player_code", "position", "team_key", "price"),
        xp.select("player_code", "gameweek", "expected_points"),
    )
    rows = [
        {
            "gameweek": gw.gameweek,
            "player_code": p,
            "in_lineup": p in gw.lineup,
            "is_captain": p == gw.captain,
            "is_vice_captain": p == gw.vice_captain,
            "bought": p in gw.transfers_in,
            "hits": gw.hits,
            "expected_points": gw.expected_points,
        }
        for gw in plan.gameweeks
        for p in gw.squad
    ]
    path = f"gold/plans/wildcard/as_of={_stamp()}/data.parquet"
    uri = _write_parquet(lake, path, pl.DataFrame(rows))
    return dg.MaterializeResult(
        metadata={
            "uri": uri,
            "status": plan.status,
            "expected_points": round(plan.total_expected_points, 1),
            "plan": dg.MetadataValue.md(f"```\n{describe(plan, squads, None)}\n```"),
        }
    )
