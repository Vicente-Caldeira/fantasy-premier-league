"""Everything the API serves, loaded once and refreshed when its sources change.

Sources (all read-only):
    warehouse   current squads (prices, ownership, news), gameweeks, clubs, played
                matches and player-match history
    gold lake   the latest expected points per player and gameweek, and the latest
                fixture expectations (lambdas, outcome and clean-sheet probabilities)

Derived here: per-player season statistics and form, club attack/defence ratings,
and next-deadline price-move probabilities (`fpl_models.prices`).
"""

import glob
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from fpl_models import prices, team_strength

FORM_GAMEWEEKS = 4


def _latest(lake: str, path: str) -> str:
    files = sorted(glob.glob(f"{lake}/{path}/*/data.parquet"))
    if not files:
        raise FileNotFoundError(f"no {path} in {lake}: run the warehouse_build job first")
    return files[-1]


def season_stats(player_matches: pl.DataFrame, season: str) -> pl.DataFrame:
    """Per player: this season's totals and form (mean points over recent gameweeks)."""
    played = player_matches.filter(pl.col("season") == season)
    per_gw = played.group_by("player_code", "gameweek").agg(pl.col("total_points").sum())
    last = int(per_gw["gameweek"].max() or 0)  # type: ignore[arg-type]
    form = (
        per_gw.filter(pl.col("gameweek") > last - FORM_GAMEWEEKS)
        .group_by("player_code")
        .agg((pl.col("total_points").sum() / FORM_GAMEWEEKS).alias("form"))
    )
    totals = played.group_by("player_code").agg(
        pl.col("total_points").sum().alias("points"),
        pl.col("minutes").sum(),
        (pl.col("minutes") > 0).sum().alias("appearances"),
        pl.col("starts").sum(),
        pl.col("goals_scored").sum().alias("goals"),
        pl.col("assists").sum(),
        pl.col("expected_goals").sum().alias("xg"),
        pl.col("expected_assists").sum().alias("xa"),
        pl.col("clean_sheets").sum(),
        pl.col("bonus").sum(),
        pl.col("saves").sum(),
        pl.col("defensive_contribution").sum(),
    )
    return totals.join(form, on="player_code", how="left").with_columns(
        pl.col("form").fill_null(0.0)
    )


def price_table(player_matches: pl.DataFrame) -> pl.DataFrame:
    """Historical per player-gameweek: net transfer share at the deadline and the price
    change by the next gameweek, in the shape `prices.fit` takes."""
    per_gw = (
        player_matches.group_by("season", "gameweek", "player_code")
        .agg(pl.col("price").mean(), pl.col("transfers_balance").max())
        .with_columns(
            crowd_net_transfer_share=pl.col("transfers_balance")
            / pl.col("transfers_balance").abs().sum().over("season", "gameweek").clip(1)
        )
        .sort("player_code", "season", "gameweek")
    )
    return per_gw.with_columns(
        change=pl.col("price").shift(-1).over("player_code", "season") - pl.col("price")
    )


def live_price_moves(history: pl.DataFrame, squads: pl.DataFrame) -> pl.DataFrame:
    """P(rise) and P(fall) by the next deadline from this gameweek's net transfers."""
    rise, fall = prices.fit(history)
    net = (squads["transfers_in_event"] - squads["transfers_out_event"]).cast(pl.Float64)
    share = (net / max(float(net.abs().sum()), 1.0)).to_numpy()
    return pl.DataFrame({
        "player_code": squads["player_code"],
        "net_transfers": net.cast(pl.Int64),
        "p_rise": rise.predict(share),
        "p_fall": fall.predict(share),
    })  # fmt: skip


def club_ratings(matches: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Attack and defence as goal multipliers vs the league average (1.0): attack 1.3
    scores 30% more than average, defence 0.8 concedes 20% fewer."""
    fitted = team_strength.Ratings().fit(matches.filter(pl.col("kickoff_utc") < as_of), as_of)
    keys = sorted(fitted.attack)
    return pl.DataFrame({
        "team_key": keys,
        "attack": [float(np.exp(fitted.attack[k])) for k in keys],
        "defence": [float(np.exp(-fitted.defence[k])) for k in keys],
    })  # fmt: skip


@dataclass(frozen=True)
class Snapshot:
    """One consistent view of all sources."""

    season: str
    gameweeks: pl.DataFrame  # gameweek, deadline_utc, finished, is_current, is_next
    clubs: pl.DataFrame  # team_key, name, short_name, fpl_team_code, attack, defence
    players: pl.DataFrame  # current squads + season stats + price moves
    expected: pl.DataFrame  # gold expected points (player x upcoming gameweek)
    fixtures: pl.DataFrame  # upcoming fixtures with expectations
    results: pl.DataFrame  # this season's played matches
    history: pl.DataFrame  # player-match rows, this and last season
    past_seasons: pl.DataFrame  # player_code, season, points, minutes, goals, assists
    predictions_as_of: str

    @property
    def horizon(self) -> list[int]:
        return sorted(self.expected["gameweek"].unique().to_list())


def load(warehouse: Path, lake: str) -> Snapshot:
    with duckdb.connect(str(warehouse), read_only=True) as c:
        season = str(c.sql("select max(season) from dim_gameweeks").fetchone()[0])  # type: ignore[index]
        gameweeks = c.sql(
            "select gameweek, deadline_utc, finished, is_current, is_next from dim_gameweeks"
            " where season = ? order by gameweek",
            params=[season],
        ).pl()
        clubs = c.sql("select team_key, name, short_name, fpl_team_code from dim_teams").pl()
        squads = c.sql("select * from current_squads").pl()
        previous = f"{int(season[:4]) - 1}-{season[2:4]}"
        history = c.sql(
            "select * from fct_player_matches where season in (?, ?)", params=[season, previous]
        ).pl()
        price_history = c.sql(
            "select season, gameweek, player_code, price, transfers_balance"
            " from fct_player_matches where season >= '2018-19'"
        ).pl()
        past_seasons = c.sql(
            """select player_code, season, sum(total_points) as points, sum(minutes) as minutes,
                sum(goals_scored) as goals, sum(assists) as assists,
                sum(clean_sheets) as clean_sheets, max(price) as max_price
            from fct_player_matches group by player_code, season"""
        ).pl()
    matches = team_strength.load_matches(warehouse)
    now = datetime.now(UTC)
    expected_file = _latest(lake, "gold/player_points/expected_points")
    fixtures_file = _latest(lake, "gold/team_strength/fixture_lambdas")
    expected = pl.read_parquet(expected_file)
    fixtures = pl.read_parquet(fixtures_file).filter(pl.col("season") == season)
    results = matches.filter((pl.col("season") == season) & pl.col("is_played")).select(
        "match_key", "gameweek", "kickoff_utc", "home_team_key", "away_team_key",
        "home_goals", "away_goals", "home_xg", "away_xg",
    )  # fmt: skip
    players = (
        squads.join(season_stats(history, season), on="player_code", how="left")
        .join(live_price_moves(price_table(price_history), squads), on="player_code", how="left")
        .with_columns(pl.col("form", "points", "minutes").fill_null(0))
    )
    return Snapshot(
        season=season,
        gameweeks=gameweeks,
        clubs=clubs.filter(pl.col("team_key").is_in(squads["team_key"].unique().implode())).join(
            club_ratings(matches, now), on="team_key", how="left"
        ),
        players=players,
        expected=expected,
        fixtures=fixtures,
        results=results,
        history=history,
        past_seasons=past_seasons,
        predictions_as_of=Path(expected_file).parent.name.removeprefix("as_of="),
    )


class Store:
    """Thread-safe holder of the latest snapshot; reloads when a source file changes."""

    def __init__(self, warehouse: Path, lake: str) -> None:
        self.warehouse, self.lake = warehouse, lake
        self._lock = threading.Lock()
        self._snapshot: Snapshot | None = None
        self._version: tuple[float, str, str] | None = None

    def _source_version(self) -> tuple[float, str, str]:
        return (
            self.warehouse.stat().st_mtime,
            _latest(self.lake, "gold/player_points/expected_points"),
            _latest(self.lake, "gold/team_strength/fixture_lambdas"),
        )

    def get(self) -> Snapshot:
        with self._lock:
            version = self._source_version()
            if self._snapshot is None or version != self._version:
                try:
                    self._snapshot = load(self.warehouse, self.lake)
                    self._version = version
                except duckdb.IOException:
                    if self._snapshot is None:  # warehouse busy (a rebuild) and nothing yet
                        raise
            return self._snapshot
