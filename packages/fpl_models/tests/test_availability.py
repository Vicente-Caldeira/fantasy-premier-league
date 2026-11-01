import subprocess
from datetime import UTC, datetime, timedelta

import polars as pl

from fpl_models import availability, e12
from fpl_models.season_sim import apply_availability


def _snapshots(rows):
    columns = ["season", "commit_time", "player_code", "status", "news", "news_added",
               "chance_of_playing_next_round", "chance_of_playing_this_round",
               "penalties_order", "corners_and_indirect_freekicks_order",
               "direct_freekicks_order", "now_cost"]  # fmt: skip
    return pl.DataFrame(rows, schema=columns, orient="row").with_columns(
        pl.col("commit_time").dt.replace_time_zone("UTC")
    )


def _t(day, hour=12):
    return datetime(2022, 8, day, hour)


DEADLINES = pl.DataFrame({
    "season": ["2022-23", "2022-23"], "gameweek": [1, 2],
    "deadline_utc": [datetime(2022, 8, 5, 18, tzinfo=UTC), datetime(2022, 8, 12, 18, tzinfo=UTC)],
})  # fmt: skip


def test_each_gameweek_sees_only_the_last_snapshot_committed_before_its_deadline():
    snapshots = _snapshots([
        ("2022-23", _t(3), 10, "a", "", None, None, None, 1.0, None, None, 50),
        ("2022-23", _t(4), 10, "d", "Knock - 75% chance of playing", "2022-08-04T09:00:00Z",
         75.0, 75.0, 1.0, None, None, 50),
        # Committed exactly at GW2's deadline: too late for GW2 (and after GW1's).
        ("2022-23", datetime(2022, 8, 12, 18), 10, "i", "Hamstring injury", None,
         0.0, 0.0, 1.0, None, None, 50),
    ])  # fmt: skip
    aligned = availability.align(snapshots, DEADLINES).sort("gameweek")
    assert aligned["status"].to_list() == ["d", "d"]
    table = availability.features(aligned).sort("gameweek")
    row = table.row(0, named=True)
    assert row["avail_chance"] == 0.75
    assert row["avail_doubt"] == 1
    assert row["avail_out"] == 0
    assert row["avail_news_injury"] == 1
    assert 1.3 < row["avail_news_days"] < 1.4  # 09:00 Aug 4 -> 18:00 Aug 5
    assert row["penalty_first"] == 1


def test_unlisted_set_piece_order_is_zero_only_where_orders_are_published():
    snapshots = _snapshots([
        ("2022-23", _t(4), 10, "a", "", None, None, None, 1.0, 2.0, None, 50),
        ("2022-23", _t(4), 11, "a", "", None, None, None, None, None, None, 50),
        ("2021-22", _t(4), 12, "a", "", None, None, None, None, None, None, 50),
    ])  # fmt: skip
    deadlines = pl.concat(
        [DEADLINES.head(1), DEADLINES.head(1).with_columns(season=pl.lit("2021-22"))]
    )
    table = availability.features(availability.align(snapshots, deadlines)).sort("player_code")
    assert table["penalty_order"].to_list() == [1.0, 0.0, None]
    assert table["set_piece_taker"].to_list() == [1, 0, None]
    assert table["avail_chance"].to_list() == [1.0, 1.0, 1.0]  # no flag = fit


def test_live_multiplier_scales_the_coming_week_and_zeroes_players_who_left():
    table = pl.DataFrame({
        "season": ["2022-23"] * 3, "gameweek": [1] * 3, "player_code": [1, 2, 3],
        "status": ["d", "u", "a"], "chance_of_playing_next_round": [50.0, 0.0, None],
    })  # fmt: skip
    factors = e12.sim_factors(table).sort("player_code")
    assert factors["this_week"].to_list() == [0.5, 0.0, 1.0]
    assert factors["later"].to_list() == [1.0, 0.0, 1.0]
    xp = pl.DataFrame({"player_code": [1, 1, 2, 2, 3], "gameweek": [1, 2, 1, 2, 1],
                       "expected_points": [4.0, 4.0, 6.0, 6.0, 2.0]})  # fmt: skip
    scaled = apply_availability(xp, e12.by_gameweek(factors, "2022-23")[1], gameweek=1)
    assert scaled["expected_points"].to_list() == [2.0, 4.0, 0.0, 0.0, 2.0]
    assert scaled.columns == xp.columns


def test_core_insights_reads_the_last_commit_before_each_deadline():
    t0 = datetime(2025, 11, 1, tzinfo=UTC)
    commits = [("a", t0), ("b", t0 + timedelta(hours=10)), ("c", t0 + timedelta(hours=30))]
    deadlines = [t0 + timedelta(hours=12), t0 + timedelta(hours=40), t0 - timedelta(hours=1)]
    assert availability.last_before(commits, deadlines) == {"b", "c"}  # none before the 3rd
    assert availability.last_before(commits, [t0 + timedelta(hours=10)]) == {"a"}  # strictly


def test_core_insights_snapshot_takes_each_players_highest_gameweek_row(tmp_path):
    repo = tmp_path / "core"
    path = repo / availability.core_path("2025-26")
    path.parent.mkdir(parents=True)
    path.write_text(
        "id,status,chance_of_playing_next_round,news,gw\n"
        "7,a,,,10\n7,d,50,Knock,11\n8,i,0,Hamstring,11\n9,a,,,11\n"
    )

    def git(*args):
        command = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args]
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout  # noqa: S603

    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "x")
    elements = pl.DataFrame({"season": ["2025-26"] * 2, "element": [7, 8], "player_code": [70, 80]})
    sha = git("rev-parse", "HEAD").strip()
    snap = availability.read_core_snapshot(repo, sha, "2025-26", elements)
    assert snap.sort("code").select("code", "status", "chance_of_playing_next_round").rows() == [
        (70, "d", 50.0),
        (80, "i", 0.0),
    ]  # element 9 isn't in the warehouse, so it's dropped
