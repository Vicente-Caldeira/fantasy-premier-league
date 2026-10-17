"""FPL-Elo-Insights ingestion: commit-pinned bronze, season-partitioned silver.

Bronze (daily) snapshots the upstream repo via `fpl_pipelines.github_repo`; silver
rebuilds a season's tables from the latest manifest.
"""

from collections.abc import Callable

import dagster as dg
import polars as pl

from fpl_pipelines.github_repo import GitHubResource, RepoSource
from fpl_pipelines.lake import Lake
from fpl_pipelines.snapshots import current_season, daily_snapshots, season_label
from fpl_pipelines.sources import fpl_elo_insights as src

BRONZE = ["bronze", "fpl_elo_insights"]
SILVER = ["silver", "fpl_elo_insights"]
MATCHES_PER_SEASON = 380

SOURCE = RepoSource("fpl_elo_insights", src.REPO, src.BRANCH, src.select_files)

elo_seasons = dg.StaticPartitionsDefinition(
    [season_label(y) for y in range(src.FIRST_SEASON, int(current_season()[:4]) + 1)]
)


@dg.asset(key_prefix=BRONZE, partitions_def=daily_snapshots, group_name="fpl_elo_insights_bronze")
def repo_snapshot(
    context: dg.AssetExecutionContext, github: GitHubResource, lake: Lake
) -> dg.MaterializeResult[None]:
    """Manifest of the upstream commit's files, plus any new content-addressed blobs."""
    return SOURCE.snapshot(context, github, lake)


def silver_asset(name: str) -> Callable[[Callable[..., pl.DataFrame]], dg.AssetsDefinition]:
    return dg.asset(
        name=name,
        key_prefix=SILVER,
        partitions_def=elo_seasons,
        group_name="fpl_elo_insights_silver",
        io_manager_key="parquet_io_manager",
        metadata={"partition_column": "season"},
        deps=[dg.AssetDep(repo_snapshot, partition_mapping=dg.AllPartitionMapping())],
    )


@silver_asset("matches")
def matches(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """Premier League matches with team-level stats and Elo (one row per fixture)."""
    season = context.partition_key
    refs = SOURCE.files(lake, season, "matches")
    return pl.concat([src.parse_matches(SOURCE.read(lake, r), season) for r in refs])


@silver_asset("player_match_stats")
def player_match_stats(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """Opta-style per-player match stats: touches, chances created, xG/xA/xGOT, duels."""
    season = context.partition_key
    refs = SOURCE.files(lake, season, "playermatchstats")
    return pl.concat(
        [src.parse_player_match_stats(SOURCE.read(lake, r), season, r.gameweek or 0) for r in refs]
    )


@dg.asset_check(asset=matches, blocking=True, partitions_def=elo_seasons)
def full_fixture_list(matches: pl.DataFrame) -> dg.AssetCheckResult:
    duplicates = matches.height - matches["match_id"].n_unique()
    self_matches = matches.filter(pl.col("home_team_code") == pl.col("away_team_code")).height
    unscored = matches.filter(pl.col("finished") & pl.col("home_score").is_null()).height
    teams = matches["home_team_code"].n_unique()
    return dg.AssetCheckResult(
        passed=matches.height == MATCHES_PER_SEASON
        and duplicates == self_matches == unscored == 0
        and teams == 20,
        metadata={
            "rows": matches.height,
            "duplicates": duplicates,
            "self_matches": self_matches,
            "finished_without_score": unscored,
            "teams": teams,
        },
    )


@dg.asset_check(asset=matches, partitions_def=elo_seasons)
def elo_coverage(matches: pl.DataFrame) -> dg.AssetCheckResult:
    """Warn when Elo ratings are missing; upstream stopped filling them in 2026-27."""
    missing = matches.filter(pl.col("home_elo").is_null() | pl.col("away_elo").is_null()).height
    return dg.AssetCheckResult(
        passed=missing == 0,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={"fixtures": matches.height, "missing_elo": missing},
    )


@dg.asset_check(
    asset=player_match_stats,
    blocking=True,
    partitions_def=elo_seasons,
    additional_ins={"matches": dg.AssetIn(key=matches.key)},
)
def player_rows_are_valid(
    player_match_stats: pl.DataFrame, matches: pl.DataFrame
) -> dg.AssetCheckResult:
    duplicates = player_match_stats.height - player_match_stats.n_unique(["player_id", "match_id"])
    orphans = player_match_stats.join(matches, on="match_id", how="anti").height
    bad_minutes = player_match_stats.filter(~pl.col("minutes_played").is_between(0, 130)).height
    counts = [c for c, t in player_match_stats.schema.items() if t == pl.Int64]
    negative = player_match_stats.filter(pl.any_horizontal(pl.col(counts) < 0)).height
    return dg.AssetCheckResult(
        passed=duplicates == orphans == bad_minutes == negative == 0,
        metadata={
            "rows": player_match_stats.height,
            "duplicates": duplicates,
            "orphan_match_ids": orphans,
            "bad_minutes": bad_minutes,
            "negative_counts": negative,
        },
    )
