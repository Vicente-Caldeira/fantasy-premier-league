"""football-data.co.uk parsing and ingestion, against real files captured on 2026-10-17."""

import gzip
from datetime import UTC, date, datetime
from pathlib import Path

import dagster as dg
import httpx
import polars as pl
import pytest

from fpl_core.http import RetryingHttpClient
from fpl_pipelines.assets import football_data as assets
from fpl_pipelines.lake import Lake, PartitionedParquetIOManager
from fpl_pipelines.snapshots import current_season
from fpl_pipelines.sources import football_data as fd

FIXTURES = Path(__file__).parent / "fixtures" / "football_data"
TODAY = datetime.now(UTC).date().isoformat()


def raw(name):
    return gzip.decompress((FIXTURES / f"{name}.csv.gz").read_bytes())


# --- parsing -------------------------------------------------------------------------


def test_season_path():
    assert fd.season_path("2016-17") == "/mmz4281/1617/E0.csv"
    assert fd.season_path("2026-27") == "/mmz4281/2627/E0.csv"


def test_2016_17_uses_betbrain_and_pinnacle_and_two_digit_years():
    df = fd.parse_matches(raw("E0_1617"), "2016-17")
    assert df.height == 380
    assert df["match_date"].min() == date(2016, 8, 13)
    assert df["odds_1x2_source"].unique().to_list() == ["betbrain_avg"]
    assert df["close_odds_1x2_source"].unique().to_list() == ["pinnacle"]
    assert df["close_odds_total_source"].null_count() == 380
    assert df["home_xg"].null_count() == 380


def test_modern_season_uses_market_averages():
    df = fd.parse_matches(raw("E0_2425"), "2024-25")
    assert df.height == 380
    for group in ("odds_1x2", "odds_total", "close_odds_1x2", "close_odds_total"):
        assert df[f"{group}_source"].unique().to_list() == ["market_avg"]
    assert df.schema["odds_home"] == pl.Float64
    assert df.schema["home_goals"] == pl.Int64


def test_current_season_has_expected_goals():
    df = fd.parse_matches(raw("E0_2627"), "2026-27")
    assert df["home_xg"].null_count() == 0
    assert df.columns == list(fd.COLUMN_ORDER)


def test_upcoming_fixtures_keep_only_premier_league_without_results():
    df = fd.parse_matches(raw("fixtures"), "2026-27")
    assert df.height > 0
    assert df["division"].unique().to_list() == ["E0"]
    assert df["home_goals"].null_count() == df.height


def test_blank_trailing_rows_are_dropped():
    data = raw("E0_2627") + b",,,,,,,,\n,,,,\n"
    assert (
        fd.parse_matches(data, "2026-27").height
        == fd.parse_matches(raw("E0_2627"), "2026-27").height
    )


def test_malformed_numbers_fail_instead_of_becoming_null():
    lines = raw("E0_2627").decode("utf-8-sig").splitlines()
    header = lines[0].split(",")
    row = lines[1].split(",")
    row[header.index("FTHG")] = "two"
    data = "\n".join([lines[0], ",".join(row)]).encode()
    with pytest.raises(pl.exceptions.PolarsError):
        fd.parse_matches(data, "2026-27")


@pytest.mark.parametrize(
    ("today", "season"),
    [
        (date(2026, 10, 17), "2026-27"),
        (date(2027, 5, 30), "2026-27"),
        (date(2027, 7, 1), "2027-28"),
    ],
)
def test_current_season(today, season):
    assert current_season(today) == season


# --- ingestion -----------------------------------------------------------------------

SERVED: dict[str, bytes] = {}


class FakeFootballData(assets.FootballDataResource):
    def client(self):
        def handler(request):
            body = SERVED.get(request.url.path)
            return httpx.Response(200, content=body) if body else httpx.Response(404)

        return RetryingHttpClient("https://example.test", transport=httpx.MockTransport(handler))


@pytest.fixture
def lake(tmp_path):
    SERVED.clear()
    SERVED[fd.season_path("2024-25")] = raw("E0_2425")
    SERVED[fd.season_path(current_season())] = raw("E0_2627")
    SERVED[fd.UPCOMING_PATH] = raw("fixtures")
    return Lake(base_url=str(tmp_path))


def run(lake, selection, partition_key):
    return dg.materialize(
        [*dg.load_assets_from_modules([assets]), *dg.load_asset_checks_from_modules([assets])],
        selection=selection,
        partition_key=partition_key,
        resources={
            "football_data": FakeFootballData(),
            "lake": lake,
            "parquet_io_manager": PartitionedParquetIOManager(lake=lake),
        },
        raise_on_error=False,
    )


SEASON_ASSETS = [assets.season_results, assets.matches]


def check_results(result):
    return {c.check_name: c.passed for c in result.get_asset_check_evaluations()}


def test_finished_season_ingests_and_passes_checks(lake):
    result = run(lake, SEASON_ASSETS, "2024-25")
    assert result.success
    assert check_results(result) == {"season_is_complete_and_consistent": True}
    table = pl.read_parquet(
        Path(lake.base_url) / "silver/football_data/matches/season=2024-25/data.parquet"
    )
    assert table.height == 380


def test_unchanged_season_file_is_not_stored_twice(lake):
    run(lake, SEASON_ASSETS, "2024-25")
    run(lake, SEASON_ASSETS, "2024-25")
    assert len(lake.list(assets.season_prefix("2024-25"))) == 1


def test_changed_season_file_adds_a_version_and_silver_uses_it(lake):
    season = current_season()
    first = fd.parse_matches(raw("E0_2627"), season).height
    SERVED[fd.season_path(season)] = b"\n".join(raw("E0_2627").splitlines()[:-1])
    run(lake, SEASON_ASSETS, season)
    SERVED[fd.season_path(season)] = raw("E0_2627")
    run(lake, SEASON_ASSETS, season)
    assert len(lake.list(assets.season_prefix(season))) == 2
    table = pl.read_parquet(
        Path(lake.base_url) / f"silver/football_data/matches/season={season}/data.parquet"
    )
    assert table.height == first


def test_incomplete_finished_season_is_blocked(lake):
    SERVED[fd.season_path("2024-25")] = b"\n".join(raw("E0_2425").splitlines()[:-5])
    result = run(lake, SEASON_ASSETS, "2024-25")
    assert check_results(result) == {"season_is_complete_and_consistent": False}


def test_upcoming_fixtures_snapshot(lake):
    result = run(lake, [assets.upcoming_fixtures, assets.upcoming_matches], TODAY)
    assert result.success
    assert check_results(result) == {"upcoming_odds_are_valid": True}
    assert lake.exists(assets.upcoming_path(TODAY))


def test_checks_only_load_their_own_partition(lake):
    # Regression: unpartitioned checks loaded every partition of the asset and crashed
    # as soon as a second partition existed.
    assert run(lake, SEASON_ASSETS, "2024-25").success
    result = run(lake, SEASON_ASSETS, current_season())
    assert result.success
    assert check_results(result) == {"season_is_complete_and_consistent": True}
