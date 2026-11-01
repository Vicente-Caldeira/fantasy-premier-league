"""Understat per-match xG and xA for FPL fixture rows (E20).

FPL's own xG and xA start in 2022-23. Before that, every fixture row has no xG target
for the component model and no `expected_goals_per90` / `expected_assists_per90`
feature. Understat covers every Premier League match since 2014-15
(`fct_understat_player_matches`). It has no FPL identifier, so players are matched by
name within club and season (`penalties.crosswalk`, as in E3a).

`XG_SOURCES`, for `expected_goals` / `expected_assists` (features and targets):
    fpl        FPL's only (production before E20)
    fill       FPL's where they exist, Understat's before
    understat  Understat's wherever the player is matched, FPL's otherwise

`add_understat` also adds Understat-only stats (`STATS`: shots, key passes, npxG,
xGChain, xGBuildup) as `us_*` columns, for per-90 features.
"""

from pathlib import Path

import duckdb
import polars as pl

from fpl_models import penalties

XG_SOURCES = ("fpl", "fill", "understat")
STATS = ("us_shots", "us_key_passes", "us_npxg", "us_xg_chain", "us_xg_buildup")
_COLUMNS = {
    "xg": "us_xg", "xa": "us_xa", "shots": "us_shots", "key_passes": "us_key_passes",
    "npxg": "us_npxg", "xg_chain": "us_xg_chain", "xg_buildup": "us_xg_buildup",
}  # fmt: skip


def load(warehouse: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Understat player-matches, and FPL players per season (for the crosswalk). Empty
    when the warehouse has no Understat player-matches yet."""
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        tables = {t for (t,) in connection.sql("show tables").fetchall()}
        fpl = connection.sql(
            "select season, player_code, team_key, first_name || ' ' || second_name as full_name,"
            " web_name from dim_player_seasons"
        ).pl()
        if "fct_understat_player_matches" not in tables:
            return pl.DataFrame(), fpl
        understat = connection.sql(
            "select season, match_key, team_key, understat_player_id, player_name, minutes,"
            " xg, xa, shots, key_passes, npxg, xg_chain, xg_buildup"
            " from fct_understat_player_matches"
        ).pl()
    return understat, fpl


def crosswalk(understat: pl.DataFrame, fpl: pl.DataFrame) -> pl.DataFrame:
    """(season, understat_player_id) -> player_code, from each player-season's clubs."""
    seasons = understat.group_by("season", "understat_player_id").agg(
        pl.col("player_name").first(), pl.col("team_key").unique().alias("team_keys")
    )
    return penalties.crosswalk(seasons, fpl)


def add_understat(rows: pl.DataFrame, understat: pl.DataFrame, walk: pl.DataFrame) -> pl.DataFrame:
    """Join Understat's per-match stats onto FPL fixture rows by (match_key, player_code),
    as `us_xg`, `us_xa` and `STATS`. Unmatched rows get nulls."""
    if understat.is_empty():
        return rows.with_columns(pl.lit(None, pl.Float64).alias(c) for c in _COLUMNS.values())
    per_match = (
        understat.join(walk.select("season", "understat_player_id", "player_code"),
                       on=["season", "understat_player_id"], how="inner")
        .group_by("match_key", "player_code")
        .agg(pl.col(c).sum().cast(pl.Float64).alias(alias) for c, alias in _COLUMNS.items())
    )  # fmt: skip
    return rows.join(per_match, on=["match_key", "player_code"], how="left", maintain_order="left")


def apply_xg_source(rows: pl.DataFrame, source: str) -> pl.DataFrame:
    """Set `expected_goals` / `expected_assists` from `source` (one of `XG_SOURCES`)."""
    if source not in XG_SOURCES:
        raise ValueError(f"xg source {source!r} not in {XG_SOURCES}")
    if source == "fpl":
        return rows
    pairs = (("expected_goals", "us_xg"), ("expected_assists", "us_xa"))
    if source == "fill":
        return rows.with_columns(pl.coalesce(fpl, us).alias(fpl) for fpl, us in pairs)
    return rows.with_columns(pl.coalesce(us, fpl).alias(fpl) for fpl, us in pairs)


def with_understat(rows: pl.DataFrame, warehouse: Path, source: str = "fpl") -> pl.DataFrame:
    """`rows` (`player_features.load_fixture_rows`) with Understat's stats and xG source."""
    understat, fpl = load(warehouse)
    walk = crosswalk(understat, fpl) if not understat.is_empty() else pl.DataFrame()
    return apply_xg_source(add_understat(rows, understat, walk), source)


def main() -> None:
    """Coverage: share of FPL minutes with a matched Understat row, and how FPL's and
    Understat's xG compare where both exist (2022-23 onwards)."""
    from fpl_models.data import DEFAULT_WAREHOUSE  # noqa: PLC0415
    from fpl_models.player_features import load_fixture_rows  # noqa: PLC0415

    rows = with_understat(load_fixture_rows(DEFAULT_WAREHOUSE), DEFAULT_WAREHOUSE)
    played = rows.filter(pl.col("minutes") > 0)
    with pl.Config(tbl_rows=20, float_precision=3):
        print(
            played.group_by("season")
            .agg(
                pl.len().alias("played_rows"),
                pl.col("us_xg").is_not_null().mean().alias("matched"),
                (pl.col("minutes").filter(pl.col("us_xg").is_not_null()).sum()
                 / pl.col("minutes").sum()).alias("minutes_matched"),
                pl.col("expected_goals").sum().alias("fpl_xg"),
                pl.col("us_xg").sum().alias("us_xg"),
                pl.corr("expected_goals", "us_xg").alias("corr_xg"),
                pl.corr("expected_assists", "us_xa").alias("corr_xa"),
                pl.col("goals_scored").sum().alias("goals"),
            )
            .sort("season")
        )  # fmt: skip


if __name__ == "__main__":
    main()
