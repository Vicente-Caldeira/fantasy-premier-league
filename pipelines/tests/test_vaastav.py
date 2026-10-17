"""vaastav history parsing and ingestion, with synthetic files (the data is FPL's)."""

import csv
import io
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path

import dagster as dg
import httpx
import polars as pl
import pytest

from fpl_core.http import RetryingHttpClient
from fpl_pipelines.assets import vaastav as assets
from fpl_pipelines.github_repo import GitHubResource, git_blob_sha
from fpl_pipelines.lake import Lake, PartitionedParquetIOManager
from fpl_pipelines.sources import vaastav as src
from fpl_pipelines.sources.csv_contract import ContractError

TODAY = datetime.now(UTC).date().isoformat()
COMMIT = "a" * 40


def to_csv(rows, encoding="utf-8"):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode(encoding)


def player(id_, team, element_type=3, web_name="Player"):
    return {
        "id": id_, "code": 1000 + id_, "element_type": element_type, "team": team,
        "team_code": 50 + team, "now_cost": 55, "total_points": 10, "web_name": web_name,
        "first_name": "A", "second_name": web_name,
    }  # fmt: skip


def match(element, fixture, opponent, *, kickoff="2025-08-15T19:00:00Z", **overrides):
    row = {c: "0" for c in src.MATCH_COUNTS} | {c: "0.0" for c in src.MATCH_FLOATS}
    row |= {"element": element, "fixture": fixture, "opponent_team": opponent, "round": 1}
    row |= {"kickoff_time": kickoff, "was_home": "True", "minutes": "90", "total_points": "2"}
    row |= {"xP": "9.9", "name": "Player"}  # xP leaks the outcome and must be ignored
    return row | {k: str(v) for k, v in overrides.items()}


def season_files(season):
    """20 teams, one player each, full double round-robin (380 fixtures)."""
    players = [player(t, t) for t in range(1, 21)]
    rows = []
    for fixture, (home, away) in enumerate(permutations(range(1, 21), 2), start=1):
        rows.append(match(home, fixture, away))
        rows.append(match(away, fixture, home, was_home="False"))
    return {
        f"data/{season}/players_raw.csv": to_csv(players),
        f"data/{season}/gws/merged_gw.csv": to_csv(rows),
    }


# --- source ---------------------------------------------------------------------------


def test_select_files_keeps_history_seasons_only():
    tree = [
        {"type": "blob", "sha": "1", "path": "data/2016-17/gws/merged_gw.csv"},
        {"type": "blob", "sha": "2", "path": "data/2016-17/players_raw.csv"},
        {"type": "blob", "sha": "3", "path": "data/2016-17/gws/gw1.csv"},
        {"type": "blob", "sha": "4", "path": "data/2026-27/gws/merged_gw.csv"},
        {"type": "blob", "sha": "5", "path": "data/2015-16/players_raw.csv"},
    ]
    assert [(r.season, r.dataset) for r in src.select_files(tree)] == [
        ("2016-17", "merged_gw"),
        ("2016-17", "players_raw"),
    ]


def test_windows_1252_files_decode():
    data = to_csv([player(1, 1, web_name="Sánchez")], encoding="cp1252")
    with pytest.raises(UnicodeDecodeError):
        data.decode("utf-8")
    assert src.parse_players(data, "2016-17")["web_name"].item() == "Sánchez"


def test_assistant_managers_are_not_players():
    data = to_csv([player(1, 1), player(2, 1, element_type=src.ASSISTANT_MANAGER)])
    assert src.parse_players(data, "2024-25")["id"].to_list() == [1]


def test_xp_is_never_ingested():
    df = src.parse_player_matches(to_csv([match(1, 1, 2)]), "2025-26")
    assert "xP" not in df.columns
    assert df["expected_goals"].item() is None  # optional before 2022-23


def test_postponed_fixture_keeps_the_row_from_when_it_was_played():
    placeholder = match(1, 275, 2, kickoff="2020-03-11T19:30:00Z", minutes=0, round=29)
    played = match(1, 275, 2, kickoff="2020-06-17T19:15:00Z", minutes=90, round=39)
    deduped = src.deduplicate(src.parse_player_matches(to_csv([placeholder, played]), "2019-20"))
    assert deduped.select("round", "minutes").rows() == [(39, 90)]


def test_exact_duplicates_are_dropped():
    rows = src.parse_player_matches(to_csv([match(1, 1, 2), match(1, 1, 2)]), "2025-26")
    assert src.deduplicate(rows).height == 1


def test_two_played_rows_for_one_fixture_fail():
    rows = [
        match(1, 1, 2, kickoff="2025-08-15T19:00:00Z"),
        match(1, 1, 2, kickoff="2025-08-16T19:00:00Z"),
    ]
    with pytest.raises(ContractError, match="two played rows"):
        src.deduplicate(src.parse_player_matches(to_csv(rows), "2025-26"))


def test_enrich_maps_season_ids_to_stable_codes():
    players = src.parse_players(to_csv([player(7, 3), player(8, 4)]), "2020-21")
    matches = src.parse_player_matches(to_csv([match(7, 1, 4)]), "2020-21")
    row = src.enrich(matches, players).row(0, named=True)
    assert (row["player_code"], row["team_code"], row["opponent_team_code"]) == (1007, 53, 54)


# --- ingestion ------------------------------------------------------------------------

FILES: dict[str, bytes] = {}


def github_handler(request):
    path = request.url.path
    if path.endswith("/commits/master"):
        return httpx.Response(200, json={"sha": COMMIT})
    if "/git/trees/" in path:
        tree = [{"path": p, "type": "blob", "sha": git_blob_sha(b)} for p, b in FILES.items()]
        return httpx.Response(200, json={"truncated": False, "tree": tree})
    return httpx.Response(200, content=FILES[path.split(f"/{COMMIT}/", 1)[1]])


class FakeGitHub(GitHubResource):
    def api(self):
        return RetryingHttpClient("https://api.test", transport=httpx.MockTransport(github_handler))

    def raw(self):
        return RetryingHttpClient("https://raw.test", transport=httpx.MockTransport(github_handler))


@pytest.fixture
def lake(tmp_path):
    FILES.clear()
    FILES.update(season_files("2020-21"))
    return Lake(base_url=str(tmp_path))


def run(lake, selection, partition_key):
    return dg.materialize(
        [*dg.load_assets_from_modules([assets]), *dg.load_asset_checks_from_modules([assets])],
        selection=selection,
        partition_key=partition_key,
        resources={
            "github": FakeGitHub(),
            "lake": lake,
            "parquet_io_manager": PartitionedParquetIOManager(lake=lake),
        },
        raise_on_error=False,
    )


def test_history_season_builds_and_passes_checks(lake):
    assert run(lake, [assets.repo_snapshot], TODAY).success
    result = run(lake, [assets.players, assets.player_matches], "2020-21")
    assert result.success
    assert {c.check_name: c.passed for c in result.get_asset_check_evaluations()} == {
        "one_row_per_player": True,
        "complete_and_consistent": True,
        "points_match_current_rules": True,
    }
    table = pl.read_parquet(
        Path(lake.base_url) / "silver/vaastav/player_matches/season=2020-21/data.parquet"
    )
    assert table.height == 760
    assert table["opponent_team_code"].null_count() == 0
