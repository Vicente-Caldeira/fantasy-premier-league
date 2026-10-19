import polars as pl

from fpl_models import penalties


def test_similarity_handles_order_accents_and_legal_names():
    assert penalties.similarity("Son Heung-Min", "Heung-Min Son") >= penalties.SUBSET
    assert penalties.similarity("Raúl Jiménez", "Raul Jimenez") >= penalties.SUBSET
    assert penalties.similarity("Bruno Fernandes", "Bruno Miguel Borges Fernandes") == (
        penalties.SUBSET
    )
    assert penalties.similarity("Harry Kane", "Harry Maguire") < penalties.MIN_SIMILARITY


def test_crosswalk_matches_within_club_and_skips_ambiguous_names():
    understat = pl.DataFrame({
        "season": ["2021-22"] * 3, "understat_player_id": [1, 2, 3],
        "player_name": ["Harry Kane", "Jota", "Mo Salah"],
        "team_keys": [["tottenham"], ["arsenal"], ["tottenham"]],
    })  # fmt: skip
    fpl = pl.DataFrame({
        "season": ["2021-22"] * 4, "player_code": [10, 20, 21, 30],
        "team_key": ["tottenham", "arsenal", "arsenal", "liverpool"],
        "full_name": ["Harry Kane", "Diogo Jota", "Jota Silva", "Mohamed Salah"],
        "web_name": ["Kane", "Jota", "Jota", "Salah"],
    })  # fmt: skip
    walk = penalties.crosswalk(understat, fpl)
    # Kane matches; two "Jota"s at one club is ambiguous; Salah isn't at that club.
    assert walk.select("understat_player_id", "player_code").rows() == [(1, 10)]


def test_features_use_last_season_and_only_count_share_at_the_same_club():
    table = pl.DataFrame({
        "season": ["2021-22", "2021-22"], "player_code": [10, 11],
        "penalty_team_key": ["tottenham", "chelsea"],
        "penalty_xg_prev90": [0.2, 0.1], "penalty_share": [0.9, 0.5],
    })  # fmt: skip
    rows = pl.DataFrame({
        "season": ["2022-23", "2022-23", "2021-22"], "player_code": [10, 11, 10],
        "team_key": ["tottenham", "arsenal", "tottenham"],
    })  # fmt: skip
    out = penalties.add_penalties(rows, table)
    assert out["penalty_share_prev"].to_list() == [0.9, 0.0, 0.0]  # moved club; no S-1 data
    assert out["penalty_xg_prev90"].to_list() == [0.2, 0.1, 0.0]
    assert out.height == rows.height
