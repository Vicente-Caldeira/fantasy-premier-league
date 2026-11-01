"""E20: Understat per-match stats joined to FPL fixture rows, and the xG source."""

import polars as pl
import pytest

from fpl_models import understat


def fixture_rows():
    return pl.DataFrame({
        "season": ["2021-22", "2021-22", "2022-23"],
        "match_key": ["m1", "m1", "m2"],
        "player_code": [10, 20, 10],
        "expected_goals": [None, None, 0.4],
        "expected_assists": [None, None, 0.1],
        "minutes": [90, 0, 90],
    })  # fmt: skip


def understat_rows():
    return pl.DataFrame({
        "season": ["2021-22", "2022-23"], "match_key": ["m1", "m2"], "team_key": ["ars", "ars"],
        "understat_player_id": [7, 7], "player_name": ["Bukayo Saka", "Bukayo Saka"],
        "minutes": [90, 90], "xg": [0.3, 0.5], "xa": [0.2, 0.3], "shots": [3, 4],
        "key_passes": [2, 1], "npxg": [0.3, 0.5], "xg_chain": [0.6, 0.7], "xg_buildup": [0.1, 0.2],
    })  # fmt: skip


def fpl_players():
    return pl.DataFrame({
        "season": ["2021-22", "2022-23", "2021-22"], "player_code": [10, 10, 20],
        "team_key": ["ars", "ars", "ars"],
        "full_name": ["Bukayo Ayoyinka Saka", "Bukayo Ayoyinka Saka", "Aaron Ramsdale"],
        "web_name": ["Saka", "Saka", "Ramsdale"],
    })  # fmt: skip


def joined():
    walk = understat.crosswalk(understat_rows(), fpl_players())
    return understat.add_understat(fixture_rows(), understat_rows(), walk)


def test_crosswalk_matches_names_within_club_and_season():
    walk = understat.crosswalk(understat_rows(), fpl_players())
    assert walk.sort("season")["player_code"].to_list() == [10, 10]


def test_understat_stats_join_by_match_and_player():
    rows = joined()
    assert rows["us_xg"].to_list() == [0.3, 0.0, 0.5]  # player 20 didn't play: 0, as FPL
    assert rows["us_shots"].to_list() == [3.0, 0.0, 4.0]


def test_a_player_who_played_but_isnt_matched_stays_null():
    rows = fixture_rows().with_columns(pl.Series("minutes", [90, 30, 90]))
    walk = understat.crosswalk(understat_rows(), fpl_players())
    assert understat.add_understat(rows, understat_rows(), walk)["us_xg"][1] is None
    assert rows.height == fixture_rows().height


@pytest.mark.parametrize(
    ("source", "expected"),
    [("fpl", [None, None, 0.4]), ("fill", [0.3, 0.0, 0.4]), ("understat", [0.3, 0.0, 0.5])],
)
def test_xg_source_picks_fpl_or_understat(source, expected):
    rows = understat.apply_xg_source(joined(), source)
    assert rows["expected_goals"].to_list() == expected


def test_unknown_xg_source_is_rejected():
    with pytest.raises(ValueError, match="xg source"):
        understat.apply_xg_source(joined(), "opta")


def test_no_understat_data_adds_null_columns():
    rows = understat.add_understat(fixture_rows(), pl.DataFrame(), pl.DataFrame())
    assert rows["us_xg"].null_count() == rows.height
