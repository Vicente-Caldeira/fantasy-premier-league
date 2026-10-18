"""FPL-Elo-Insights parsing and ingestion.

The upstream repo has no license, so these tests use synthetic files that follow its
real layout and column contract instead of copies of its data.
"""

import csv
import io
import json
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from urllib.parse import unquote

import dagster as dg
import httpx
import polars as pl
import pytest

from fpl_core.http import RetryingHttpClient
from fpl_pipelines.assets import fpl_elo_insights as assets
from fpl_pipelines.github_repo import GitHubResource, git_blob_sha
from fpl_pipelines.lake import Lake, PartitionedParquetIOManager
from fpl_pipelines.snapshots import current_season
from fpl_pipelines.sources import fpl_elo_insights as src

TODAY = datetime.now(UTC).date().isoformat()
TEAM_CODES = [3, 7, 8, 14, 43, 1, 4, 6, 11, 13, 17, 20, 21, 31, 36, 39, 54, 90, 91, 94]
COMMIT = "c" * 40


def to_csv(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def match_row(gameweek, home, away, *, kickoff="2026-09-05T16:30:00", finished=True):
    row = {"gameweek": f"{gameweek}.0", "kickoff_time": kickoff, "match_id": f"m-{home}-{away}"}
    row["finished"] = str(finished)
    for side, code in (("home", home), ("away", away)):
        row[f"{side}_team"] = f"{code}.0"
        for suffix in src.MATCH_SIDE_COUNTS.values():
            row.setdefault(f"{side}_{suffix}", "1.0" if finished else "")
        for suffix in src.MATCH_SIDE_FLOATS.values():
            row.setdefault(f"{side}_{suffix}", "1.5" if finished else "")
    row["tournament"] = "prem"
    return row


def player_row(player_id, match_id, **overrides):
    row = {"player_id": f"{player_id}.0", "match_id": match_id}
    row |= {c: "1.0" for c in src.PLAYER_COUNTS} | {c: "0.25" for c in src.PLAYER_FLOATS}
    row["minutes_played"] = "90.0"
    return row | overrides


def season_files():
    """A full double round-robin: 38 gameweeks of 10 fixtures, two players per match."""
    fixtures = list(permutations(TEAM_CODES, 2))
    per_gw = {gw: fixtures[(gw - 1) * 10 : gw * 10] for gw in range(1, 39)}
    matches = {gw: [match_row(gw, h, a) for h, a in fx] for gw, fx in per_gw.items()}
    players = {
        gw: [player_row(i, m["match_id"]) for m in ms for i in (int(m["home_team"][:-2]), 1000)]
        for gw, ms in matches.items()
    }
    return matches, players


def current_layout(season, dataset, gw, tournament="Premier League"):
    folder = f"{season[:4]}-{int(season[:4]) + 1}"
    return f"data/{folder}/By Tournament/{tournament}/GW{gw}/{dataset}.csv"


def cup_row(match_id, home, away, kickoff="2026-09-16T19:00:00+00:00", finished=True):
    return {"gameweek": "4.0", "kickoff_time": kickoff, "home_team": home, "away_team": away,
            "finished": str(finished), "match_id": match_id}  # fmt: skip


CUP_ROWS = [
    cup_row("ucl-arsenal-vs-psv", "3.0", ""),
    cup_row("ucl-inter-vs-liverpool", "", "14.0"),
    cup_row("ucl-knockout-without-kickoff", "8.0", "", kickoff=""),
    cup_row("ucl-arsenal-vs-psv/inter", "3.0", "", finished=False),
]


class FakeGitHub:
    def __init__(self, files):
        self.files = files  # path -> bytes
        self.raw_requests: list[str] = []

    def tree(self):
        entries = [
            {"path": p, "type": "blob", "sha": git_blob_sha(b)} for p, b in self.files.items()
        ]
        entries.append(
            {
                "path": "data/2026-2027/By Tournament/Friendlies/GW1/matches.csv",
                "type": "blob",
                "sha": "0" * 40,
            }
        )
        return {"sha": COMMIT, "truncated": False, "tree": entries}

    def handler(self, request):
        path = unquote(request.url.path)
        if path.endswith("/commits/main"):
            return httpx.Response(200, json={"sha": COMMIT})
        if "/git/trees/" in path:
            return httpx.Response(200, json=self.tree())
        self.raw_requests.append(path)
        file_path = path.split(f"/{COMMIT}/", 1)[1]
        return httpx.Response(200, content=self.files[file_path])


GITHUB = FakeGitHub({})


class FakeGitHubResource(GitHubResource):
    def api(self):
        return RetryingHttpClient("https://api.test", transport=httpx.MockTransport(GITHUB.handler))

    def raw(self):
        return RetryingHttpClient("https://raw.test", transport=httpx.MockTransport(GITHUB.handler))


@pytest.fixture
def lake(tmp_path):
    season = current_season()
    matches, players = season_files()
    files = {current_layout(season, "matches", gw): to_csv(rows) for gw, rows in matches.items()}
    files |= {
        current_layout(season, "playermatchstats", gw): to_csv(r) for gw, r in players.items()
    }
    files[current_layout(season, "matches", 4, "Champions League")] = to_csv(CUP_ROWS)
    GITHUB.files = files
    GITHUB.raw_requests = []
    return Lake(base_url=str(tmp_path))


def run(lake, selection, partition_key):
    return dg.materialize(
        [*dg.load_assets_from_modules([assets]), *dg.load_asset_checks_from_modules([assets])],
        selection=selection,
        partition_key=partition_key,
        resources={
            "github": FakeGitHubResource(),
            "lake": lake,
            "parquet_io_manager": PartitionedParquetIOManager(lake=lake),
        },
        raise_on_error=False,
    )


def checks(result):
    return {c.check_name: c.passed for c in result.get_asset_check_evaluations()}


# --- source ---------------------------------------------------------------------------


def test_git_blob_sha_matches_git():
    # `printf 'hello\n' | git hash-object --stdin`
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_select_files_handles_both_layouts_and_ignores_the_rest():
    tree = [
        {"type": "blob", "sha": "a", "path": "data/2024-2025/matches/GW3/matches.csv"},
        {
            "type": "blob",
            "sha": "b",
            "path": "data/2024-2025/playermatchstats/GW3/playermatchstats.csv",
        },
        {
            "type": "blob",
            "sha": "c",
            "path": "data/2026-2027/By Tournament/Premier League/GW7/matches.csv",
        },
        {
            "type": "blob",
            "sha": "d",
            "path": "data/2026-2027/By Tournament/EFL Cup/GW7/matches.csv",
        },
        {
            "type": "blob",
            "sha": "d2",
            "path": "data/2026-2027/By Tournament/Friendlies/GW7/matches.csv",
        },
        {
            "type": "blob",
            "sha": "e",
            "path": "data/2026-2027/By Tournament/Premier League/GW7/shots.csv",
        },
        {"type": "blob", "sha": "f", "path": "data/2024-2025/matches/matches.csv"},
        {"type": "tree", "sha": "g", "path": "data/2026-2027/By Tournament"},
    ]
    refs = src.select_files(tree)
    assert [(r.season, r.dataset, r.gameweek) for r in refs] == [
        ("2024-25", "matches", 3),
        ("2024-25", "playermatchstats", 3),
        ("2026-27", "cup_matches", 7),
        ("2026-27", "matches", 7),
    ]
    assert src.cup_name(refs[2].path) == "EFL Cup"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-08-17 14:00:00", datetime(2024, 8, 17, 14, 0, tzinfo=UTC)),
        ("2025-04-16 11:30", datetime(2025, 4, 16, 11, 30, tzinfo=UTC)),
        ("2025-08-15T19:00:00+00:00", datetime(2025, 8, 15, 19, 0, tzinfo=UTC)),
        ("2025-08-16T15:00:00+01:00", datetime(2025, 8, 16, 14, 0, tzinfo=UTC)),
        ("2026-09-05T16:30:00", datetime(2026, 9, 5, 16, 30, tzinfo=UTC)),
    ],
)
def test_every_kickoff_format_is_normalised_to_utc(raw, expected):
    df = src.parse_matches(to_csv([match_row(1, 3, 7, kickoff=raw)]), "2026-27")
    assert df["kickoff_time"].item() == expected


def test_unrecognised_kickoff_format_fails():
    with pytest.raises(src.ContractError, match="kickoff"):
        src.parse_matches(to_csv([match_row(1, 3, 7, kickoff="17/08/2024 14:00")]), "2024-25")


def test_unfinished_fixture_has_null_stats():
    df = src.parse_matches(to_csv([match_row(30, 3, 7, finished=False)]), "2026-27")
    assert not df["finished"].item()
    assert df["home_score"].item() is None


def test_player_counts_are_integers_and_optional_columns_default_to_null():
    df = src.parse_player_match_stats(to_csv([player_row(5, "m-3-7")]), "2024-25", 3)
    assert df.schema["goals"] == pl.Int64
    assert df.schema["xg"] == pl.Float64
    assert df["player_id"].item() == 5
    assert df["defensive_contributions"].item() is None
    assert df["gameweek"].item() == 3


def test_fractional_count_is_rejected():
    with pytest.raises(src.ContractError, match="goals"):
        src.parse_player_match_stats(to_csv([player_row(5, "m", goals="1.5")]), "2026-27", 1)


def test_missing_required_column_is_rejected():
    row = player_row(5, "m")
    del row["touches"]
    with pytest.raises(src.ContractError, match="touches"):
        src.parse_player_match_stats(to_csv([row]), "2026-27", 1)


# --- ingestion ------------------------------------------------------------------------


def test_snapshot_downloads_verified_blobs_and_pins_the_commit(lake):
    result = run(lake, [assets.repo_snapshot], TODAY)
    assert result.success
    manifest = json.loads(lake.read_bytes(assets.SOURCE.manifest_path(TODAY)))
    assert manifest["commit"] == COMMIT
    assert (
        len(manifest["files"]) == 77
    )  # 38 gameweeks x 2 datasets + a cup file; friendlies ignored
    assert all(lake.exists(assets.SOURCE.blob_path(f["sha"])) for f in manifest["files"])


def test_only_missing_blobs_are_downloaded(lake):
    already = next(iter(GITHUB.files.values()))
    lake.write_bytes(assets.SOURCE.blob_path(git_blob_sha(already)), already)
    run(lake, [assets.repo_snapshot], TODAY)
    assert len(GITHUB.raw_requests) == 76


def test_corrupted_download_fails_the_snapshot(lake, monkeypatch):
    original = GITHUB.handler

    def corrupt_raw_files(request):
        response = original(request)
        is_raw_file = f"/{COMMIT}/" in unquote(request.url.path)
        return httpx.Response(200, content=response.content + b"x") if is_raw_file else response

    monkeypatch.setattr(GITHUB, "handler", corrupt_raw_files)
    result = run(lake, [assets.repo_snapshot], TODAY)
    assert not result.success
    assert not lake.exists(assets.SOURCE.manifest_path(TODAY))


def test_season_tables_build_and_pass_checks(lake):
    run(lake, [assets.repo_snapshot], TODAY)
    result = run(lake, [assets.matches, assets.player_match_stats], current_season())
    assert result.success
    assert checks(result) == {
        "full_fixture_list": True,
        "elo_coverage": True,
        "player_rows_are_valid": True,
    }
    base = Path(lake.base_url) / "silver/fpl_elo_insights"
    assert pl.read_parquet(base / f"matches/season={current_season()}/data.parquet").height == 380
    assert (
        pl.read_parquet(base / f"player_match_stats/season={current_season()}/data.parquet").height
        == 760
    )


def test_orphan_player_rows_are_blocked(lake):
    path = current_layout(current_season(), "playermatchstats", 1)
    GITHUB.files[path] = to_csv([player_row(9, "no-such-match")])
    run(lake, [assets.repo_snapshot], TODAY)
    result = run(lake, [assets.matches, assets.player_match_stats], current_season())
    assert checks(result)["player_rows_are_valid"] is False


def test_cup_matches_drop_untimed_and_undecided_ties():
    df = src.parse_cup_matches(to_csv(CUP_ROWS), "2026-27", "Champions League")
    assert df["match_id"].to_list() == ["ucl-arsenal-vs-psv", "ucl-inter-vs-liverpool"]
    assert df["home_team_code"].to_list() == [3, None]  # foreign clubs have no code
    assert df["away_team_code"].to_list() == [None, 14]
    assert df["kickoff_time"][0] == datetime(2026, 9, 16, 19, 0, tzinfo=UTC)


def test_cup_matches_build_and_pass_checks(lake):
    run(lake, [assets.repo_snapshot], TODAY)
    result = run(lake, [assets.cup_matches], current_season())
    assert result.success
    assert checks(result) == {"cup_matches_involve_league_clubs": True}
    base = Path(lake.base_url) / "silver/fpl_elo_insights"
    table = pl.read_parquet(base / f"cup_matches/season={current_season()}/data.parquet")
    assert table["competition"].unique().to_list() == ["Champions League"]
    assert table.height == 2
