"""E24: defensive-contribution counts before 2025-26 from Opta."""

import polars as pl
import pytest

from fpl_models import player_features as pf


def rows():
    return pl.DataFrame({
        "season": ["2024-25", "2025-26", "2025-26"],
        "tackles": [None, 3, 2],
        "clearances_blocks_interceptions": [None, 6, 4],
        "recoveries": [None, 5, 3],
        "tackles_won": [2, 2, 2],
        "interceptions": [1, 1, 1],
        "blocks": [1, 1, 1],
        "clearances": [5, 4, 2],
        "opta_recoveries": [4, 5, 3],
    })  # fmt: skip


def test_tackle_factor_is_fitted_on_2025_26():
    assert pf.opta_tackle_factor(rows()) == pytest.approx(5 / 4)


def test_opta_fills_only_where_fpl_has_no_counts():
    filled = pf.fill_defensive_from_opta(rows(), tackle_factor=1.25)
    assert filled["tackles"].to_list() == [2.5, 3, 2]
    assert filled["clearances_blocks_interceptions"].to_list() == [7, 6, 4]
    assert filled["recoveries"].to_list() == [4, 5, 3]


def test_rows_without_opta_stay_null():
    no_opta = rows().with_columns(pl.lit(None, pl.Int64).alias("tackles_won"))
    filled = pf.fill_defensive_from_opta(no_opta, tackle_factor=1.25)
    assert filled["tackles"][0] is None
