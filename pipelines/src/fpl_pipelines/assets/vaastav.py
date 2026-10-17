"""vaastav FPL history: commit-pinned bronze, season-partitioned silver (2016-17 .. 2025-26)."""

from collections.abc import Callable

import dagster as dg
import polars as pl

from fpl_core.scoring import MatchStats, Position, points
from fpl_pipelines.github_repo import GitHubResource, RepoSource
from fpl_pipelines.lake import Lake
from fpl_pipelines.snapshots import daily_snapshots, season_label
from fpl_pipelines.sources import vaastav as src

BRONZE = ["bronze", "vaastav"]
SILVER = ["silver", "vaastav"]
FIXTURES_PER_SEASON = 380
# First season scored under today's rules (defensive contributions, 10-point GK goals).
CURRENT_RULES_FROM = "2025-26"

SOURCE = RepoSource("vaastav", src.REPO, src.BRANCH, src.select_files)

history_seasons = dg.StaticPartitionsDefinition(
    [season_label(y) for y in range(src.FIRST_SEASON, src.LAST_SEASON + 1)]
)


@dg.asset(key_prefix=BRONZE, partitions_def=daily_snapshots, group_name="vaastav_bronze")
def repo_snapshot(
    context: dg.AssetExecutionContext, github: GitHubResource, lake: Lake
) -> dg.MaterializeResult[None]:
    """Manifest of the upstream commit's season files, plus any new blobs."""
    return SOURCE.snapshot(context, github, lake)


def _read_one(lake: Lake, season: str, dataset: str) -> bytes:
    refs = SOURCE.files(lake, season, dataset)
    if len(refs) != 1:
        raise dg.Failure(f"Expected one {dataset} file for {season}, found {len(refs)}")
    return SOURCE.read(lake, refs[0])


def silver_asset(
    ins: dict[str, dg.AssetIn] | None = None,
) -> Callable[[Callable[..., pl.DataFrame]], dg.AssetsDefinition]:
    return dg.asset(
        key_prefix=SILVER,
        partitions_def=history_seasons,
        group_name="vaastav_silver",
        io_manager_key="parquet_io_manager",
        metadata={"partition_column": "season"},
        deps=[dg.AssetDep(repo_snapshot, partition_mapping=dg.AllPartitionMapping())],
        ins=ins,
    )


@silver_asset()
def players(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """End-of-season player list: position, team and the stable cross-season `code`."""
    season = context.partition_key
    return src.parse_players(_read_one(lake, season, "players_raw"), season)


@silver_asset(ins={"players": dg.AssetIn(key=["silver", "vaastav", "players"])})
def player_matches(
    context: dg.AssetExecutionContext, lake: Lake, players: pl.DataFrame
) -> pl.DataFrame:
    """One row per player per fixture, with position and stable player/team codes."""
    season = context.partition_key
    raw = src.parse_player_matches(_read_one(lake, season, "merged_gw"), season)
    matches = src.deduplicate(raw)
    context.add_output_metadata({"duplicates_removed": raw.height - matches.height})
    return src.enrich(matches, players)


@dg.asset_check(asset=players, blocking=True, partitions_def=history_seasons)
def one_row_per_player(players: pl.DataFrame) -> dg.AssetCheckResult:
    duplicate_ids = players.height - players["id"].n_unique()
    duplicate_codes = players.height - players["code"].n_unique()
    teams = players["team_code"].n_unique()
    return dg.AssetCheckResult(
        passed=duplicate_ids == duplicate_codes == 0 and teams == 20,
        metadata={
            "rows": players.height,
            "duplicate_ids": duplicate_ids,
            "duplicate_codes": duplicate_codes,
            "teams": teams,
        },
    )


@dg.asset_check(
    asset=player_matches,
    blocking=True,
    partitions_def=history_seasons,
    additional_ins={"players": dg.AssetIn(key=players.key)},
)
def complete_and_consistent(
    player_matches: pl.DataFrame, players: pl.DataFrame
) -> dg.AssetCheckResult:
    """Every fixture present, no duplicates, every player and opponent resolved."""
    fixtures = player_matches["fixture"].n_unique()
    duplicates = player_matches.height - player_matches.n_unique(["element", "fixture"])
    unknown_opponents = player_matches["opponent_team_code"].null_count()
    bad_minutes = player_matches.filter(~pl.col("minutes").is_between(0, 130)).height
    return dg.AssetCheckResult(
        passed=fixtures == FIXTURES_PER_SEASON
        and duplicates == unknown_opponents == bad_minutes == 0,
        metadata={
            "rows": player_matches.height,
            "fixtures": fixtures,
            "players": player_matches["element"].n_unique(),
            "duplicates": duplicates,
            "unknown_opponents": unknown_opponents,
            "bad_minutes": bad_minutes,
        },
    )


@dg.asset_check(asset=player_matches, partitions_def=history_seasons)
def points_match_current_rules(
    context: dg.AssetCheckExecutionContext, player_matches: pl.DataFrame
) -> dg.AssetCheckResult:
    """Recompute points with `fpl_core.scoring` for seasons scored under today's rules.

    Earlier seasons used different rules, so they are reported as not applicable.
    """
    if context.partition_key < CURRENT_RULES_FROM:
        return dg.AssetCheckResult(passed=True, metadata={"applicable": False})
    fields = set(MatchStats.model_fields)
    rows = player_matches.to_dicts()
    mismatches = sum(
        points(Position(r["element_type"]), MatchStats(**{k: r[k] for k in fields}))
        != r["total_points"]
        for r in rows
    )
    return dg.AssetCheckResult(
        passed=mismatches == 0,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={"applicable": True, "checked": len(rows), "mismatches": mismatches},
    )
