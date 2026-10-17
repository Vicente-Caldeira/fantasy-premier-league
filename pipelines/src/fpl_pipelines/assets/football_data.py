"""football-data.co.uk ingestion: seasonal results + odds, and daily upcoming-match odds.

Season files are re-published as matches are played, so bronze keeps a new version only
when the content actually changes. Upcoming fixtures (pre-match odds) are daily
snapshots, like the FPL API.
"""

from datetime import UTC, date, datetime

import dagster as dg
import polars as pl

from fpl_core.http import RetryingHttpClient
from fpl_pipelines.lake import Lake
from fpl_pipelines.snapshots import current_season, daily_snapshots, seasons, take_snapshot
from fpl_pipelines.sources import football_data as fd

BRONZE = ["bronze", "football_data"]
SILVER = ["silver", "football_data"]
MATCHES_PER_SEASON = 380


class FootballDataResource(dg.ConfigurableResource):  # type: ignore[type-arg]
    base_url: str = fd.BASE_URL

    def client(self) -> RetryingHttpClient:
        return RetryingHttpClient(self.base_url)


def season_prefix(season: str) -> str:
    return f"bronze/football_data/{fd.PREMIER_LEAGUE}/season={season}"


def upcoming_path(snapshot_date: str) -> str:
    return f"bronze/football_data/fixtures/snapshot_date={snapshot_date}/fixtures.csv"


def latest_season_version(lake: Lake, season: str) -> str | None:
    versions = lake.list(season_prefix(season))
    return versions[-1] if versions else None


@dg.asset(key_prefix=BRONZE, partitions_def=seasons, group_name="football_data_bronze")
def season_results(
    context: dg.AssetExecutionContext, football_data: FootballDataResource, lake: Lake
) -> dg.MaterializeResult[None]:
    """Raw season CSV. A new immutable version is stored only when the content changes."""
    season = context.partition_key
    with football_data.client() as client:
        data = client.get(fd.season_path(season)).content

    latest = latest_season_version(lake, season)
    if latest is not None and lake.read_bytes(latest) == data:
        return dg.MaterializeResult(metadata={"uri": lake.uri(latest), "changed": False})

    # Microseconds keep versions distinct and lexicographically ordered by fetch time.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = f"{season_prefix(season)}/fetched_at={stamp}.csv"
    lake.write_bytes(path, data)
    return dg.MaterializeResult(
        metadata={"uri": lake.uri(path), "bytes": len(data), "changed": True}
    )


@dg.asset(key_prefix=BRONZE, partitions_def=daily_snapshots, group_name="football_data_bronze")
def upcoming_fixtures(
    context: dg.AssetExecutionContext, football_data: FootballDataResource, lake: Lake
) -> dg.MaterializeResult[None]:
    """Raw `fixtures.csv`: pre-match odds for every upcoming match across leagues."""
    with football_data.client() as client:
        return take_snapshot(
            context,
            lake,
            upcoming_path(context.partition_key),
            lambda: client.get(fd.UPCOMING_PATH).content,
        )


@dg.asset(
    key_prefix=SILVER,
    partitions_def=seasons,
    group_name="football_data_silver",
    io_manager_key="parquet_io_manager",
    metadata={"partition_column": "season"},
    deps=[season_results],
)
def matches(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """Premier League results, match stats and opening/closing odds, one row per match."""
    season = context.partition_key
    latest = latest_season_version(lake, season)
    if latest is None:
        raise dg.Failure(f"No bronze version for season {season}")
    return fd.parse_matches(lake.read_bytes(latest), season)


@dg.asset(
    key_prefix=SILVER,
    partitions_def=daily_snapshots,
    group_name="football_data_silver",
    io_manager_key="parquet_io_manager",
    deps=[upcoming_fixtures],
)
def upcoming_matches(context: dg.AssetExecutionContext, lake: Lake) -> pl.DataFrame:
    """Upcoming Premier League matches with pre-match odds, as of the snapshot date."""
    snapshot_date = context.partition_key
    season = current_season(date.fromisoformat(snapshot_date))
    df = fd.parse_matches(lake.read_bytes(upcoming_path(snapshot_date)), season)
    return df.with_columns(pl.lit(date.fromisoformat(snapshot_date)).alias("snapshot_date"))


def _overround_violations(df: pl.DataFrame) -> int:
    """Rows whose implied 1X2 probabilities don't sum to a plausible bookmaker margin."""
    total = pl.sum_horizontal(1 / pl.col(c) for c in ("odds_home", "odds_draw", "odds_away"))
    return df.filter(pl.col("odds_home").is_not_null() & ~total.is_between(1.0, 1.25)).height


def _bad_prices(df: pl.DataFrame) -> int:
    odds = [c for c in df.columns if "odds" in c and not c.endswith("_source")]
    return df.filter(pl.any_horizontal(pl.col(odds) <= 1.0)).height


@dg.asset_check(asset=matches, blocking=True, partitions_def=seasons)
def season_is_complete_and_consistent(
    context: dg.AssetCheckExecutionContext, matches: pl.DataFrame
) -> dg.AssetCheckResult:
    finished_season = context.partition_key != current_season()
    expected_rows = matches.height == MATCHES_PER_SEASON if finished_season else True
    duplicates = matches.height - matches.n_unique(["home_team", "away_team"])
    played = matches.filter(pl.col("home_goals").is_not_null())
    wrong_results = played.filter(
        pl.col("result")
        != pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
    ).height
    teams = matches["home_team"].n_unique()
    metadata = {
        "rows": matches.height,
        "duplicates": duplicates,
        "wrong_results": wrong_results,
        "teams": teams,
        "bad_prices": _bad_prices(matches),
        "overround_violations": _overround_violations(matches),
    }
    passed = (
        expected_rows
        and duplicates == wrong_results == metadata["bad_prices"] == 0
        and metadata["overround_violations"] == 0
        and teams == 20
    )
    return dg.AssetCheckResult(passed=passed, metadata=metadata)


@dg.asset_check(asset=upcoming_matches, blocking=True, partitions_def=daily_snapshots)
def upcoming_odds_are_valid(upcoming_matches: pl.DataFrame) -> dg.AssetCheckResult:
    self_matches = upcoming_matches.filter(pl.col("home_team") == pl.col("away_team")).height
    metadata = {
        "rows": upcoming_matches.height,
        "self_matches": self_matches,
        "bad_prices": _bad_prices(upcoming_matches),
        "overround_violations": _overround_violations(upcoming_matches),
    }
    return dg.AssetCheckResult(
        passed=self_matches == metadata["bad_prices"] == metadata["overround_violations"] == 0,
        metadata=metadata,
    )
