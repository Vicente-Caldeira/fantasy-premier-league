"""Helpers for parsing CSVs against a column contract.

Files are read as strings and every column is cast explicitly: required columns must
exist, optional ones (added in later seasons) become null when absent, numbers are cast
strictly, and count columns must be integral.
"""

import io

import polars as pl


class ContractError(ValueError):
    """A required column is missing or a value breaks the contract."""


def decode(data: bytes) -> str:
    """UTF-8, falling back to Windows-1252 for older files saved by Excel."""
    data = data.removeprefix(b"\xef\xbb\xbf")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252")


def read_strings(data: bytes) -> pl.DataFrame:
    return pl.read_csv(io.StringIO(decode(data)), infer_schema=False)


def value(raw: pl.DataFrame, source: str, *, required: bool) -> pl.Expr:
    if source not in raw.columns:
        if required:
            raise ContractError(f"missing required column {source!r}")
        return pl.lit(None, pl.String)
    return pl.col(source).str.strip_chars().replace("", None)


def required(raw: pl.DataFrame, source: str) -> pl.Expr:
    return value(raw, source, required=True).cast(pl.Float64, strict=True)


def optional(raw: pl.DataFrame, source: str) -> pl.Expr:
    return value(raw, source, required=False).cast(pl.Float64, strict=True)


def boolean(raw: pl.DataFrame, source: str) -> pl.Expr:
    text = value(raw, source, required=True)
    return pl.when(text == "True").then(True).when(text == "False").then(False)


def typed(
    raw: pl.DataFrame, floats: dict[str, pl.Expr], counts: dict[str, pl.Expr]
) -> pl.DataFrame:
    """Parse numbers as floats, reject fractional counts, then cast counts to Int64."""
    parsed = raw.select(**floats, **counts)
    fractional = [
        name for name in counts if parsed.filter(pl.col(name) != pl.col(name).round()).height
    ]
    if fractional:
        raise ContractError(f"fractional values in count columns: {fractional}")
    return parsed.with_columns(pl.col(list(counts)).cast(pl.Int64))
