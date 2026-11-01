"""Understat ingestion (2014-15 onwards), versioned on change: one league JSON per season,
and every played match's rosters and shots (one request per match, fetched once)."""

import json
import time
from collections.abc import Callable

import dagster as dg
import polars as pl

from fpl_core.http import RetryingHttpClient
from fpl_pipelines.lake import Lake
from fpl_pipelines.snapshots import current_season, latest_version, season_label, store_version
from fpl_pipelines.sources import understat as src

BRONZE = ["bronze", "understat"]
SILVER = ["silver", "understat"]
MATCHES_PER_SEASON = 380
XG_TOLERANCE = 1e-3

understat_seasons = dg.StaticPartitionsDefinition(
    [season_label(y) for y in range(src.FIRST_SEASON, int(current_season()[:4]) + 1)]
)


class UnderstatResource(dg.ConfigurableResource):  # type: ignore[type-arg]
    base_url: str = src.BASE_URL
    # Match data is one request per match: pace them (a backfill is ~380 per season).
    request_interval_seconds: float = 1.0

    def client(self, season: str) -> RetryingHttpClient:
        return RetryingHttpClient(self.base_url, headers=src.request_headers(season))


def season_prefix(season: str) -> str:
    return f"bronze/understat/league/season={season}"


def match_data_prefix(season: str) -> str:
    return f"bronze/understat/match_data/season={season}"


@dg.asset(key_prefix=BRONZE, partitions_def=understat_seasons, group_name="understat_bronze")
def league(
    context: dg.AssetExecutionContext, understat: UnderstatResource, lake: Lake
) -> dg.MaterializeResult[None]:
    """Raw league JSON for one season, stored as a new version only when it changes."""
    season = context.partition_key
    with understat.client(season) as client:
        data = client.get(src.league_path(season)).content
    src.League.model_validate_json(data)  # reject a broken payload before storing it
    return store_version(lake, season_prefix(season), data, "json")


def _league(lake: Lake, season: str) -> src.League:
    latest = latest_version(lake, season_prefix(season))
    if latest is None:
        raise dg.Failure(f"No Understat bronze for {season}")
    return src.League.model_validate_json(lake.read_bytes(latest))


def silver_asset(
    ins: dict[str, dg.AssetIn] | None = None,
) -> Callable[[Callable[..., pl.DataFrame]], dg.AssetsDefinition]:
    return dg.asset(
        key_prefix=SILVER,
        partitions_def=understat_seasons,
        group_name="understat_silver",
        io_manager_key="parquet_io_manager",
        metadata={"partition_column": "season"},
        deps=[league],
        ins=ins,
    )


@silver_asset()
def matches(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """Every match: result, xG per side and Understat's pre-match forecast."""
    season = context.partition_key
    return src.matches_table(_league(lake, season), season)


@silver_asset(ins={"matches": dg.AssetIn(key=["silver", "understat", "matches"])})
def team_matches(
    context: dg.AssetExecutionContext, lake: Lake, matches: pl.DataFrame
) -> pl.DataFrame:
    """Per team per match: xG/xGA, non-penalty xG, PPDA, deep completions, xPts."""
    season = context.partition_key
    return src.team_matches_table(_league(lake, season), season, matches)


@dg.asset(
    key_prefix=SILVER,
    partitions_def=understat_seasons,
    group_name="understat_silver",
    io_manager_key="parquet_io_manager",
    metadata={"partition_column": "season"},
    deps=[league],
)
def player_seasons(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """Season totals per player: xG, npxG, xA, xGChain, xGBuildup, key passes."""
    season = context.partition_key
    return src.players_table(_league(lake, season), season)


@dg.asset_check(asset=matches, blocking=True, partitions_def=understat_seasons)
def complete_fixture_list(
    context: dg.AssetCheckExecutionContext, matches: pl.DataFrame
) -> dg.AssetCheckResult:
    finished = context.partition_key != current_season()
    played = matches.filter(pl.col("is_result"))
    incomplete = played.filter(pl.col("home_xg").is_null() | pl.col("home_goals").is_null()).height
    metadata = {
        "rows": matches.height,
        "played": played.height,
        "duplicates": matches.height - matches["match_id"].n_unique(),
        "played_without_xg_or_score": incomplete,
        "teams": matches["home_team_id"].n_unique(),
    }
    passed = (
        matches.height == MATCHES_PER_SEASON
        and (played.height == MATCHES_PER_SEASON or not finished)
        and metadata["duplicates"] == incomplete == 0
        and metadata["teams"] == 20
    )
    return dg.AssetCheckResult(passed=passed, metadata=metadata)


@dg.asset_check(
    asset=team_matches,
    blocking=True,
    partitions_def=understat_seasons,
    additional_ins={"matches": dg.AssetIn(key=matches.key)},
)
def team_logs_agree_with_matches(
    team_matches: pl.DataFrame, matches: pl.DataFrame
) -> dg.AssetCheckResult:
    """Two rows per played match, each matching that side's goals and xG."""
    played = matches.filter(pl.col("is_result"))
    sides = pl.concat(
        [
            played.select(
                "match_id",
                pl.lit(side == "home").alias("is_home"),
                pl.col(f"{side}_goals").alias("goals"),
                pl.col(f"{side}_xg").alias("match_xg"),
            )
            for side in ("home", "away")
        ]
    )
    joined = team_matches.join(sides, on=["match_id", "is_home"], how="inner")
    metadata = {
        "rows": team_matches.height,
        "expected_rows": 2 * played.height,
        "unlinked": team_matches["match_id"].null_count(),
        "goal_mismatches": joined.filter(pl.col("scored") != pl.col("goals")).height,
        "xg_mismatches": joined.filter(
            (pl.col("xg") - pl.col("match_xg")).abs() > XG_TOLERANCE
        ).height,
    }
    return dg.AssetCheckResult(
        passed=metadata["rows"] == metadata["expected_rows"] == joined.height
        and metadata["unlinked"] == metadata["goal_mismatches"] == metadata["xg_mismatches"] == 0,
        metadata=metadata,
    )


@dg.asset(
    key_prefix=BRONZE,
    partitions_def=understat_seasons,
    group_name="understat_bronze",
    ins={"matches": dg.AssetIn(key=matches.key)},
)
def match_data(
    context: dg.AssetExecutionContext,
    understat: UnderstatResource,
    lake: Lake,
    matches: pl.DataFrame,
) -> dg.MaterializeResult[None]:
    """Every played match's rosters and shots, one JSON per season keyed by match id.

    Only played matches missing from the latest version are fetched, so the weekly
    refresh costs one request per new match. A new version is stored only on change.
    """
    season = context.partition_key
    prefix = match_data_prefix(season)
    latest = latest_version(lake, prefix)
    stored: dict[str, object] = json.loads(lake.read_bytes(latest)) if latest else {}
    played = matches.filter(pl.col("is_result"))["match_id"].sort().to_list()
    missing = [m for m in played if str(m) not in stored]
    with understat.client(season) as client:
        for i, match_id in enumerate(missing):
            if i:
                time.sleep(understat.request_interval_seconds)
            stored[str(match_id)] = src.strip_match_payload(
                client.get(src.match_path(match_id)).content
            )
    data = json.dumps(stored, sort_keys=True, separators=(",", ":")).encode()
    result = store_version(lake, prefix, data, "json")
    return dg.MaterializeResult(
        metadata={**(result.metadata or {}), "fetched": len(missing), "matches": len(stored)}
    )


@dg.asset(
    key_prefix=SILVER,
    partitions_def=understat_seasons,
    group_name="understat_silver",
    io_manager_key="parquet_io_manager",
    metadata={"partition_column": "season"},
    deps=[match_data],
    ins={"matches": dg.AssetIn(key=matches.key)},
)
def player_matches(
    context: dg.AssetExecutionContext, lake: Lake, matches: pl.DataFrame
) -> pl.DataFrame:
    """Per player per match: minutes, goals, shots, key passes, xG, npxG, xA, xGChain."""
    season = context.partition_key
    latest = latest_version(lake, match_data_prefix(season))
    if latest is None:
        raise dg.Failure(f"No Understat match data for {season}")
    return src.player_matches_table(json.loads(lake.read_bytes(latest)), season, matches)


@dg.asset_check(
    asset=player_matches,
    blocking=True,
    partitions_def=understat_seasons,
    additional_ins={"matches": dg.AssetIn(key=matches.key)},
)
def rosters_agree_with_scores(
    player_matches: pl.DataFrame, matches: pl.DataFrame
) -> dg.AssetCheckResult:
    """Every played match has both rosters, and each side's goals (its players' goals plus
    the opponents' own goals) equal the match score."""
    played = matches.filter(pl.col("is_result"))
    side = player_matches.group_by("match_id", "is_home").agg(
        pl.col("goals").sum(), pl.col("own_goals").sum()
    )
    scored = side.join(
        side.select("match_id", pl.col("is_home").not_(), pl.col("own_goals").alias("opp_og")),
        on=["match_id", "is_home"],
        how="left",
    ).with_columns(scored=pl.col("goals") + pl.col("opp_og").fill_null(0))
    expected = pl.concat(
        [
            played.select(
                "match_id",
                pl.lit(side == "home").alias("is_home"),
                pl.col(f"{side}_goals").alias("score"),
            )
            for side in ("home", "away")
        ]
    )
    joined = expected.join(scored, on=["match_id", "is_home"], how="left")
    metadata = {
        "rows": player_matches.height,
        "played_matches": played.height,
        "sides_missing": joined["scored"].null_count(),
        "score_mismatches": joined.filter(pl.col("scored") != pl.col("score")).height,
    }
    return dg.AssetCheckResult(
        passed=metadata["sides_missing"] == metadata["score_mismatches"] == 0, metadata=metadata
    )
