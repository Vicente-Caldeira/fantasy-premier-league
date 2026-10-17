"""FPL API ingestion: daily raw snapshots (bronze) and typed tables (silver).

The API only exposes *current* state: prices, ownership, injury news and form have no
history endpoint. Each daily partition is therefore a snapshot that can only be taken on
its own date; materialising a past partition fails instead of storing today's data under
yesterday's label.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import dagster as dg
import polars as pl

from fpl_core.fpl_api import BootstrapStatic, ElementSummary, Fixture
from fpl_core.fpl_api.schemas import Contract
from fpl_core.scoring import MatchStats, Position, points
from fpl_pipelines.lake import Lake
from fpl_pipelines.resources import FplApiResource

FIRST_SNAPSHOT_DATE = "2026-10-17"
snapshot_partitions = dg.DailyPartitionsDefinition(
    start_date=FIRST_SNAPSHOT_DATE, timezone="UTC", end_offset=1
)

BRONZE = ["bronze", "fpl_api"]
SILVER = ["silver", "fpl"]


def bronze_path(dataset: str, snapshot_date: str) -> str:
    return f"bronze/fpl_api/{dataset}/snapshot_date={snapshot_date}/{dataset}.json.gz"


def _assert_live(snapshot_date: str) -> None:
    today = datetime.now(UTC).date().isoformat()
    if snapshot_date != today:
        raise dg.Failure(
            f"Snapshot {snapshot_date} can only be taken on that date (today is {today}): "
            "the FPL API has no history, so a backfill would mislabel current data."
        )


def _snapshot(
    context: dg.AssetExecutionContext,
    lake: Lake,
    dataset: str,
    fetch: Any,  # noqa: ANN401
) -> dg.MaterializeResult[None]:
    path = bronze_path(dataset, context.partition_key)
    if lake.exists(path):
        # Bronze is immutable: re-running a partition reuses the snapshot already taken.
        context.log.info("Snapshot %s already exists, reusing it", path)
        return dg.MaterializeResult(metadata={"uri": lake.uri(path), "reused": True})
    _assert_live(context.partition_key)
    payload = fetch()
    size = lake.write_json_gz(path, payload)
    return dg.MaterializeResult(
        metadata={
            "uri": lake.uri(path),
            "bytes": size,
            "fetched_at": datetime.now(UTC).isoformat(),
            "reused": False,
        }
    )


@dg.asset(key_prefix=BRONZE, partitions_def=snapshot_partitions, group_name="fpl_bronze")
def bootstrap_static(
    context: dg.AssetExecutionContext, fpl_api: FplApiResource, lake: Lake
) -> dg.MaterializeResult[None]:
    """Raw `/bootstrap-static/`: teams, gameweeks, positions, every player's state."""
    with fpl_api.client() as client:
        return _snapshot(context, lake, "bootstrap_static", client.bootstrap_static)


@dg.asset(key_prefix=BRONZE, partitions_def=snapshot_partitions, group_name="fpl_bronze")
def fixtures(
    context: dg.AssetExecutionContext, fpl_api: FplApiResource, lake: Lake
) -> dg.MaterializeResult[None]:
    """Raw `/fixtures/`: every fixture this season, with per-fixture event stats."""
    with fpl_api.client() as client:
        return _snapshot(context, lake, "fixtures", client.fixtures)


@dg.asset(
    key_prefix=BRONZE,
    partitions_def=snapshot_partitions,
    group_name="fpl_bronze",
    deps=[bootstrap_static],
)
def element_summaries(
    context: dg.AssetExecutionContext, fpl_api: FplApiResource, lake: Lake
) -> dg.MaterializeResult[None]:
    """Raw `/element-summary/{id}/` for every player listed in the same day's bootstrap."""
    bootstrap = lake.read_json_gz(bronze_path("bootstrap_static", context.partition_key))
    ids = [element["id"] for element in bootstrap["elements"]]

    def fetch() -> dict[str, Any]:
        with fpl_api.client() as client:
            return {str(k): v for k, v in client.element_summaries(ids).items()}

    return _snapshot(context, lake, "element_summaries", fetch)


def _table(models: Sequence[Contract], **constants: object) -> pl.DataFrame:
    rows = [{**m.declared(), **constants} for m in models]
    return pl.DataFrame(rows, infer_schema_length=None)


def _bootstrap(lake: Lake, snapshot_date: str) -> BootstrapStatic:
    raw = lake.read_json_gz(bronze_path("bootstrap_static", snapshot_date))
    return BootstrapStatic.model_validate(raw)


def silver_asset(
    deps: Sequence[dg.AssetsDefinition],
) -> Callable[[Callable[..., pl.DataFrame]], dg.AssetsDefinition]:
    return dg.asset(
        key_prefix=SILVER,
        partitions_def=snapshot_partitions,
        group_name="fpl_silver",
        io_manager_key="parquet_io_manager",
        deps=deps,
    )


@silver_asset(deps=[bootstrap_static])
def teams(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    bootstrap = _bootstrap(lake, context.partition_key)
    return _table(bootstrap.teams, season=bootstrap.season, snapshot_date=context.partition_key)


@silver_asset(deps=[bootstrap_static])
def gameweeks(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    bootstrap = _bootstrap(lake, context.partition_key)
    return _table(bootstrap.events, season=bootstrap.season, snapshot_date=context.partition_key)


@silver_asset(deps=[bootstrap_static])
def players(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """Point-in-time player state: price, ownership, status and news as of the snapshot."""
    bootstrap = _bootstrap(lake, context.partition_key)
    return _table(bootstrap.elements, season=bootstrap.season, snapshot_date=context.partition_key)


@silver_asset(deps=[bootstrap_static, fixtures])
def fixtures_table(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    season = _bootstrap(lake, context.partition_key).season
    raw = lake.read_json_gz(bronze_path("fixtures", context.partition_key))
    models = [Fixture.model_validate(f) for f in raw]
    return _table(models, season=season, snapshot_date=context.partition_key)


@silver_asset(deps=[bootstrap_static, element_summaries])
def player_matches(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """One row per player per fixture played this season (double gameweeks = two rows)."""
    season = _bootstrap(lake, context.partition_key).season
    raw = lake.read_json_gz(bronze_path("element_summaries", context.partition_key))
    models = [m for s in raw.values() for m in ElementSummary.model_validate(s).history]
    return _table(models, season=season, snapshot_date=context.partition_key)


def _check(passed: bool, **metadata: int) -> dg.AssetCheckResult:
    return dg.AssetCheckResult(passed=passed, metadata=metadata)


@dg.asset_check(asset=teams, blocking=True)
def twenty_unique_teams(teams: pl.DataFrame) -> dg.AssetCheckResult:
    return _check(teams.height == 20 and teams["id"].n_unique() == 20, rows=teams.height)


@dg.asset_check(asset=gameweeks, blocking=True)
def thirty_eight_gameweeks(gameweeks: pl.DataFrame) -> dg.AssetCheckResult:
    ids = sorted(gameweeks["id"].to_list())
    return _check(ids == list(range(1, 39)), rows=gameweeks.height)


@dg.asset_check(asset=players, blocking=True, additional_ins={"teams": dg.AssetIn(key=teams.key)})
def players_are_valid(players: pl.DataFrame, teams: pl.DataFrame) -> dg.AssetCheckResult:
    duplicate_ids = players.height - players["id"].n_unique()
    unknown_teams = players.filter(~pl.col("team").is_in(teams["id"].implode())).height
    bad_positions = players.filter(~pl.col("element_type").is_between(1, 4)).height
    bad_prices = players.filter(pl.col("now_cost") <= 0).height
    return _check(
        duplicate_ids == unknown_teams == bad_positions == bad_prices == 0,
        rows=players.height,
        duplicate_ids=duplicate_ids,
        unknown_teams=unknown_teams,
        bad_positions=bad_positions,
        bad_prices=bad_prices,
    )


@dg.asset_check(asset=fixtures_table, blocking=True)
def full_season_of_fixtures(fixtures_table: pl.DataFrame) -> dg.AssetCheckResult:
    # Each pair of teams meets exactly twice, once at each home ground.
    pairs = fixtures_table.select("team_h", "team_a").unique().height
    return _check(fixtures_table.height == 380 and pairs == 380, rows=fixtures_table.height)


@dg.asset_check(asset=player_matches, blocking=True)
def one_row_per_player_fixture(player_matches: pl.DataFrame) -> dg.AssetCheckResult:
    duplicates = player_matches.height - player_matches.select("element", "fixture").n_unique()
    return _check(duplicates == 0, rows=player_matches.height, duplicates=duplicates)


@dg.asset_check(
    asset=player_matches,
    additional_ins={
        "players": dg.AssetIn(key=players.key),
        "fixtures_table": dg.AssetIn(key=fixtures_table.key),
    },
)
def points_match_official_scoring(
    player_matches: pl.DataFrame, players: pl.DataFrame, fixtures_table: pl.DataFrame
) -> dg.AssetCheckResult:
    """Recompute points from raw stats with `fpl_core.scoring` for finished fixtures.

    A mismatch means FPL changed a rule, a player changed position mid-season (we use
    the current position), or upstream data is wrong. Warns rather than blocks.
    """
    finished = (
        player_matches.join(
            players.select(pl.col("id").alias("element"), "element_type"), on="element"
        )
        .join(
            fixtures_table.filter(pl.col("finished")).select(pl.col("id").alias("fixture")),
            on="fixture",
        )
        .to_dicts()
    )
    fields = set(MatchStats.model_fields)
    mismatches = sum(
        points(Position(row["element_type"]), MatchStats(**{k: row[k] for k in fields}))
        != row["total_points"]
        for row in finished
    )
    return dg.AssetCheckResult(
        passed=mismatches == 0,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={"checked": len(finished), "mismatches": mismatches},
    )
