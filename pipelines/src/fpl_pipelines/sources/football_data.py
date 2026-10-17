"""football-data.co.uk: Premier League results, match stats and bookmaker odds.

Column names changed over the years, so every canonical column lists its source
columns in order of preference:

    2016-17 .. 2018-19   Betbrain averages (BbAv*), Pinnacle closing 1X2 only (PSC*)
    2019-20 .. 2025-26   market averages (Avg*), closing averages (AvgC*)
    2026-27 ..           adds expected goals (HxG/AxG); Pinnacle no longer published

`*_source` columns record which source column filled each odds group, so a model can
tell a market average from a single bookmaker's price.
"""

import io
from dataclasses import dataclass

import polars as pl

BASE_URL = "https://www.football-data.co.uk"
UPCOMING_PATH = "/fixtures.csv"
PREMIER_LEAGUE = "E0"


def season_path(season: str) -> str:
    """`2026-27` -> `/mmz4281/2627/E0.csv`."""
    code = f"{season[2:4]}{season[5:7]}"
    return f"/mmz4281/{code}/{PREMIER_LEAGUE}.csv"


@dataclass(frozen=True)
class Column:
    name: str
    dtype: type[pl.DataType]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class OddsGroup:
    """Columns that must come from the same source so prices stay comparable."""

    label: str
    names: tuple[str, ...]
    sources: dict[str, tuple[str, ...]]  # source label -> one source column per name


MATCH_COLUMNS = (
    Column("division", pl.String, ("Div",)),
    Column("kickoff_time_uk", pl.String, ("Time",)),
    Column("home_team", pl.String, ("HomeTeam",)),
    Column("away_team", pl.String, ("AwayTeam",)),
    Column("home_goals", pl.Int64, ("FTHG",)),
    Column("away_goals", pl.Int64, ("FTAG",)),
    Column("result", pl.String, ("FTR",)),
    Column("home_xg", pl.Float64, ("HxG",)),
    Column("away_xg", pl.Float64, ("AxG",)),
    Column("home_shots", pl.Int64, ("HS",)),
    Column("away_shots", pl.Int64, ("AS",)),
    Column("home_shots_on_target", pl.Int64, ("HST",)),
    Column("away_shots_on_target", pl.Int64, ("AST",)),
    Column("home_corners", pl.Int64, ("HC",)),
    Column("away_corners", pl.Int64, ("AC",)),
    Column("referee", pl.String, ("Referee",)),
)

ODDS_GROUPS = (
    OddsGroup(
        "odds_1x2",
        ("odds_home", "odds_draw", "odds_away"),
        {"market_avg": ("AvgH", "AvgD", "AvgA"), "betbrain_avg": ("BbAvH", "BbAvD", "BbAvA")},
    ),
    OddsGroup(
        "odds_total",
        ("odds_over_2_5", "odds_under_2_5"),
        {"market_avg": ("Avg>2.5", "Avg<2.5"), "betbrain_avg": ("BbAv>2.5", "BbAv<2.5")},
    ),
    OddsGroup(
        "odds_ah",
        ("ah_line", "ah_odds_home", "ah_odds_away"),
        {
            "market_avg": ("AHh", "AvgAHH", "AvgAHA"),
            "betbrain_avg": ("BbAHh", "BbAvAHH", "BbAvAHA"),
        },
    ),
    OddsGroup(
        "close_odds_1x2",
        ("close_odds_home", "close_odds_draw", "close_odds_away"),
        {"market_avg": ("AvgCH", "AvgCD", "AvgCA"), "pinnacle": ("PSCH", "PSCD", "PSCA")},
    ),
    OddsGroup(
        "close_odds_total",
        ("close_odds_over_2_5", "close_odds_under_2_5"),
        {"market_avg": ("AvgC>2.5", "AvgC<2.5"), "pinnacle": ("PC>2.5", "PC<2.5")},
    ),
    OddsGroup(
        "close_odds_ah",
        ("close_ah_line", "close_ah_odds_home", "close_ah_odds_away"),
        {"market_avg": ("AHCh", "AvgCAHH", "AvgCAHA")},
    ),
)

COLUMN_ORDER = (
    "season",
    "match_date",
    *(c.name for c in MATCH_COLUMNS),
    *(n for g in ODDS_GROUPS for n in (*g.names, f"{g.label}_source")),
)


def _clean(source: str) -> pl.Expr:
    return pl.col(source).str.strip_chars().replace("", None)


def _cast(expr: pl.Expr, dtype: type[pl.DataType]) -> pl.Expr:
    # strict=True: a malformed number fails the parse instead of silently becoming null.
    return expr if dtype is pl.String else expr.cast(dtype, strict=True)


def _date(raw: pl.Expr) -> pl.Expr:
    # 2016-17 uses two-digit years; parsing "13/08/16" with %Y would give year 16 AD.
    return (
        pl.when(raw.str.len_chars() == 8)
        .then(raw.str.to_date("%d/%m/%y", strict=True))
        .otherwise(raw.str.to_date("%d/%m/%Y", strict=True))
    )


def parse_matches(data: bytes, season: str) -> pl.DataFrame:
    """Parse a football-data CSV (season file or `fixtures.csv`) into canonical columns.

    Rows from other divisions and blank trailing rows are dropped.
    """
    raw = pl.read_csv(
        io.BytesIO(data.removeprefix(b"\xef\xbb\xbf")),
        infer_schema=False,
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
    )
    raw = raw.filter((_clean("Div") == PREMIER_LEAGUE) & _clean("HomeTeam").is_not_null())
    present = set(raw.columns)

    exprs: list[pl.Expr] = [
        pl.lit(season).alias("season"),
        _date(_clean("Date")).alias("match_date"),
    ]
    for column in MATCH_COLUMNS:
        source = next((s for s in column.sources if s in present), None)
        value = _cast(_clean(source), column.dtype) if source else pl.lit(None, column.dtype)
        exprs.append(value.alias(column.name))

    for group in ODDS_GROUPS:
        chosen = next(
            ((label, cols) for label, cols in group.sources.items() if set(cols) <= present),
            None,
        )
        for i, name in enumerate(group.names):
            value = _cast(_clean(chosen[1][i]), pl.Float64) if chosen else pl.lit(None, pl.Float64)
            exprs.append(value.alias(name))
        exprs.append(
            pl.lit(chosen[0] if chosen else None, pl.String).alias(f"{group.label}_source")
        )

    return raw.select(exprs).select(COLUMN_ORDER)
