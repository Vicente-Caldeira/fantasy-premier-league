"""Point-in-time availability and set-piece order from vaastav's git history (E12).

FPL exposes status, news, chance of playing and set-piece order only as current
state. vaastav/Fantasy-Premier-League recommits `data/{season}/players_raw.csv` (a dump
of bootstrap-static `elements`) roughly once a gameweek, so its git history is an
archive of that state from 2019-20; the set-piece orders exist from 2023-24.

Each gameweek uses the latest snapshot *committed* before its deadline (a commit is
made after the scrape, so this can't see the future):

    avail_chance       chance_of_playing_next_round / 100 (no flag = 1.0)
    avail_doubt        status d (doubtful)
    avail_out          status i / s / u / n (injured, suspended, unavailable, not in squad)
    avail_news_injury  news mentions an injury or illness
    avail_news_days    days since the news was added (null without news)
    penalty_first      first-choice penalty taker
    penalty_order      penalty order (0 = not listed)
    set_piece_taker    first or second on corners/indirect free kicks, or first on directs

E22 adds a fresher archive: FPL-Core-Insights (formerly FPL-Elo-Insights) rewrites
`playerstats.csv` (bootstrap-static state plus a `gw` label) two or three times a day,
so its git history has a snapshot ~5-6 h before most deadlines (vaastav's: ~45 h), from
2024-25 GW20 onwards. Only the last commit before each deadline is read, and from it
each player's row with the highest `gw` (the labelling changed during 2025-26, and that
rule holds under both schemes). Its `id` is the season's element, mapped to
`player_code`. Both archives go into one snapshot table; each gameweek takes the latest
snapshot from either.

Usage: `python -m fpl_models.availability` clones both repositories (blob-less) into a
cache directory, writes `silver/vaastav/player_snapshots/data.parquet` to the lake and
prints coverage.
"""

import io
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

REPO = "https://github.com/vaastav/Fantasy-Premier-League.git"
CORE_REPO = "https://github.com/olbauday/FPL-Core-Insights.git"
CORE_FIRST_SEASON = "2024-25"
FIRST_SEASON = "2019-20"
SNAPSHOT = Path("silver/vaastav/player_snapshots/data.parquet")
AVAILABILITY_FEATURES = (
    "avail_chance", "avail_doubt", "avail_out", "avail_news_injury", "avail_news_days",
)  # fmt: skip
SET_PIECE_FEATURES = ("penalty_first", "penalty_order", "set_piece_taker")
# E22: the same flags, null when the snapshot is older than MAX_FRESH_HOURS before the
# deadline, so the trees learn only from near-deadline status (the live tool's view).
MAX_FRESH_HOURS = 24.0
FRESH_FEATURES = tuple(f"{f}_fresh" for f in AVAILABILITY_FEATURES)
COLUMNS = {
    "code": pl.Int64, "status": pl.String, "news": pl.String, "news_added": pl.String,
    "chance_of_playing_next_round": pl.Float64, "chance_of_playing_this_round": pl.Float64,
    "penalties_order": pl.Float64, "corners_and_indirect_freekicks_order": pl.Float64,
    "direct_freekicks_order": pl.Float64, "now_cost": pl.Int64,
}  # fmt: skip
INJURY = (
    r"(?i)injur|knock|strain|sprain|ill(ness)?\b|hamstring|ankle|knee|groin|calf|thigh"
    r"|muscle|fitness|surgery|fracture|concussion|back problem|foot"
)
OUT_STATUSES = ("i", "s", "u", "n")
ORDERS = ("penalties_order", "corners_and_indirect_freekicks_order", "direct_freekicks_order")


def _git(repo: Path, *args: str, attempts: int = 3) -> str:
    """Run git; retried, since a blob-less clone fetches each file over the network."""
    for attempt in range(attempts):
        try:
            # Fixed program and arguments built here, never user input.
            return subprocess.run(  # noqa: S603
                ["git", "-C", str(repo), *args],  # noqa: S607
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except subprocess.CalledProcessError:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def clone(cache: Path, url: str = REPO, name: str = "vaastav") -> Path:
    """Blob-less clone (history and trees only; file contents fetched on demand). An
    existing clone is fetched, so the current season's newest commits are included."""
    repo = cache / name
    if not repo.exists():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603
            ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, str(repo)],  # noqa: S607
            check=True,
        )
    else:
        _git(repo, "fetch", "-q", "origin")
    return repo


def season_commits(repo: Path, season: str) -> list[tuple[str, datetime]]:
    """(commit, committer time) of every change to the season's players_raw.csv."""
    log = _git(
        repo, "log", "--diff-filter=AM", "--format=%H %cI", "--", f"data/{season}/players_raw.csv"
    )
    return [
        (sha, datetime.fromisoformat(ts)) for sha, ts in (line.split() for line in log.splitlines())
    ]


def read_snapshot(repo: Path, sha: str, season: str) -> pl.DataFrame:
    raw = _git(repo, "show", f"{sha}:data/{season}/players_raw.csv")
    frame = pl.read_csv(io.StringIO(raw), infer_schema=False)
    return frame.select(
        pl.col(c).cast(t, strict=False) if c in frame.columns else pl.lit(None, t).alias(c)
        for c, t in COLUMNS.items()
    )


def ingest(repo: Path, seasons: list[str]) -> pl.DataFrame:
    frames = []
    for season in seasons:
        for sha, committed in season_commits(repo, season):
            frames.append(
                read_snapshot(repo, sha, season).with_columns(
                    season=pl.lit(season), commit_time=pl.lit(committed).dt.convert_time_zone("UTC")
                )
            )
    return pl.concat(frames).rename({"code": "player_code"})


def core_path(season: str) -> str:
    """`2025-26` -> `data/2025-2026/playerstats.csv` (2024-25 sits one folder deeper)."""
    folder = f"data/{season[:4]}-{int(season[:4]) + 1}"
    return (
        f"{folder}/playerstats/playerstats.csv"
        if season == "2024-25"
        else f"{folder}/playerstats.csv"
    )


def core_commits(repo: Path, season: str) -> list[tuple[str, datetime]]:
    log = _git(repo, "log", "--format=%H %cI", "origin/HEAD", "--", core_path(season))
    return [
        (sha, datetime.fromisoformat(ts)) for sha, ts in (line.split() for line in log.splitlines())
    ]


def last_before(commits: list[tuple[str, datetime]], deadlines: list[datetime]) -> set[str]:
    """The last commit strictly before each deadline (none if the history starts later)."""
    picked = set()
    for deadline in deadlines:
        before = [(t, sha) for sha, t in commits if t < deadline]
        if before:
            picked.add(max(before)[1])
    return picked


def read_core_snapshot(repo: Path, sha: str, season: str, elements: pl.DataFrame) -> pl.DataFrame:
    """Each player's highest-`gw` row, in vaastav's snapshot columns. `elements` maps
    (season, element) to player_code."""
    raw = _git(repo, "show", f"{sha}:{core_path(season)}")
    frame = pl.read_csv(io.StringIO(raw), infer_schema=False).with_columns(
        pl.col("gw").cast(pl.Int64), pl.col("id").cast(pl.Int64).alias("element")
    )
    latest = frame.sort("gw").unique("element", keep="last")
    coded = latest.join(elements.filter(pl.col("season") == season), on="element", how="inner")
    coded = coded.with_columns(code=pl.col("player_code"))
    return coded.select(
        pl.col(c).cast(t, strict=False) if c in coded.columns else pl.lit(None, t).alias(c)
        for c, t in COLUMNS.items()
    )


def ingest_core(repo: Path, deadlines: pl.DataFrame, elements: pl.DataFrame) -> pl.DataFrame:
    """Snapshots from FPL-Core-Insights: one per deadline (the last commit before it)."""
    frames = []
    seasons = sorted(s for s in deadlines["season"].unique().to_list() if s >= CORE_FIRST_SEASON)
    for season in seasons:
        commits = core_commits(repo, season)
        times = dict(commits)
        season_deadlines = deadlines.filter(pl.col("season") == season)["deadline_utc"].to_list()
        for sha in sorted(last_before(commits, season_deadlines), key=times.__getitem__):
            frames.append(
                read_core_snapshot(repo, sha, season, elements).with_columns(
                    season=pl.lit(season),
                    commit_time=pl.lit(times[sha]).dt.convert_time_zone("UTC"),
                )
            )
    return pl.concat(frames).rename({"code": "player_code"})


def align(snapshots: pl.DataFrame, deadlines: pl.DataFrame) -> pl.DataFrame:
    """Per (season, gameweek, player): the latest snapshot committed before the deadline."""
    times = snapshots.select("season", "commit_time").unique().sort("commit_time")
    picked = (
        deadlines.select("season", "gameweek", pl.col("deadline_utc").dt.convert_time_zone("UTC"))
        .sort("deadline_utc")
        .join_asof(
            times,
            left_on="deadline_utc",
            right_on="commit_time",
            by="season",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,  # both sides sorted globally, so within each season
        )
        .drop_nulls("commit_time")
    )
    return picked.join(snapshots, on=["season", "commit_time"]).with_columns(
        snapshot_age_hours=(pl.col("deadline_utc") - pl.col("commit_time")).dt.total_minutes() / 60
    )


def features(aligned: pl.DataFrame) -> pl.DataFrame:
    news_added = pl.col("news_added").str.to_datetime(time_zone="UTC", strict=False)
    has_news = pl.col("news").fill_null("").str.len_chars() > 0
    corners = pl.col("corners_and_indirect_freekicks_order")
    # Unlisted players have no order: 0 where the snapshot publishes orders at all,
    # null where it doesn't (2019-20).
    published = pl.col("penalties_order").is_not_null().any().over("season", "commit_time")
    aligned = aligned.with_columns(
        pl.when(published).then(pl.col(c).fill_null(0)).alias(c) for c in ORDERS
    )
    return (
        aligned.select(
            "season",
            "gameweek",
            "player_code",
            "snapshot_age_hours",
            pl.col("chance_of_playing_next_round")
            .fill_null(100.0)
            .truediv(100)
            .alias("avail_chance"),
            (pl.col("status") == "d").cast(pl.Int8).alias("avail_doubt"),
            pl.col("status").is_in(OUT_STATUSES).cast(pl.Int8).alias("avail_out"),
            (has_news & pl.col("news").fill_null("").str.contains(INJURY))
            .cast(pl.Int8)
            .alias("avail_news_injury"),
            pl.when(has_news)
            .then((pl.col("deadline_utc") - news_added).dt.total_hours() / 24)
            .alias("avail_news_days"),
            (pl.col("penalties_order") == 1).cast(pl.Int8).alias("penalty_first"),
            pl.col("penalties_order").alias("penalty_order"),
            (corners.is_between(1, 2) | (pl.col("direct_freekicks_order") == 1))
            .cast(pl.Int8)
            .alias("set_piece_taker"),
            # Raw columns for the post-hoc multiplier the live tool applies.
            "status",
            "chance_of_playing_next_round",
        )
        .unique(["season", "gameweek", "player_code"], keep="last")
        .with_columns(
            pl.when(pl.col("snapshot_age_hours") <= MAX_FRESH_HOURS)
            .then(pl.col(f))
            .alias(f"{f}_fresh")
            for f in AVAILABILITY_FEATURES
        )
    )


def add_availability(rows: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Join the point-in-time features onto fixture rows (season, gameweek, player_code).
    Rows without a snapshot (before 2019-20, or a gameweek with none yet) stay null:
    the boosted trees treat missing as its own branch."""
    keep = [
        "season", "gameweek", "player_code",
        *AVAILABILITY_FEATURES, *SET_PIECE_FEATURES, *FRESH_FEATURES,
    ]  # fmt: skip
    return rows.join(
        table.select(keep), on=["season", "gameweek", "player_code"], how="left",
        maintain_order="left",
    )  # fmt: skip


def load(lake: str) -> pl.DataFrame:
    return pl.read_parquet(Path(lake) / SNAPSHOT)


def main() -> None:
    from fpl_models.data import (  # noqa: PLC0415
        DEFAULT_WAREHOUSE,
        load_player_matches,
        player_gameweeks,
    )

    lake = os.getenv("FPL_LAKE_URL", "data")
    cache = Path(os.getenv("FPL_VAASTAV_CACHE", "data/cache"))
    repo = clone(cache)
    core = clone(cache, CORE_REPO, "core_insights")
    deadlines = (
        player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
        .filter(pl.col("season") >= FIRST_SEASON)
        .select("season", "gameweek", "deadline_utc")
        .unique()
    )
    seasons = sorted(deadlines["season"].unique().to_list())
    vaastav = ingest(repo, [s for s in seasons if s <= "2025-26"])
    with duckdb.connect(str(DEFAULT_WAREHOUSE), read_only=True) as connection:
        elements = connection.sql(
            "select distinct season, element, player_code from fct_player_matches"
        ).pl()
    fresh = ingest_core(core, deadlines, elements)
    snapshots = pl.concat(
        [vaastav.with_columns(source=pl.lit("vaastav")),
         fresh.with_columns(source=pl.lit("core_insights"))],
        how="diagonal_relaxed",
    )  # fmt: skip
    (Path(lake) / SNAPSHOT).parent.mkdir(parents=True, exist_ok=True)
    snapshots.write_parquet(Path(lake) / SNAPSHOT)
    aligned = align(snapshots, deadlines)
    per_gw = aligned.select("season", "gameweek", "snapshot_age_hours", "source").unique(
        ["season", "gameweek"]
    )
    coverage = (
        deadlines.join(per_gw, on=["season", "gameweek"], how="left")
        .group_by("season")
        .agg(
            pl.len().alias("gameweeks"),
            pl.col("snapshot_age_hours").is_not_null().mean().alias("any_snapshot"),
            (pl.col("snapshot_age_hours") <= 48).mean().alias("within_48h"),
            pl.col("snapshot_age_hours").median().alias("median_age_h"),
            (pl.col("source") == "core_insights").mean().alias("core_insights"),
        )
    )
    with pl.Config(tbl_rows=20):
        print(coverage.sort("season"))
    print(f"{snapshots.height} snapshot rows from {snapshots['commit_time'].n_unique()} commits")


if __name__ == "__main__":
    main()
