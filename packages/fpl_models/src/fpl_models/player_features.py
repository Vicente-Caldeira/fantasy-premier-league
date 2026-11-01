"""Player-fixture features, built from information available before each deadline.

Rolling statistics are computed per player over *previous gameweeks only*: stats are
summed to player-gameweeks, smoothed in deadline order, then shifted by one gameweek.
So both fixtures of a double gameweek see the same pre-deadline state, and nothing
from the gameweek being predicted (or later) can reach its features. A test proves
this by truncating the data and checking that features are unchanged.

Team context comes from the team-strength model: each side's expected goals (lambda)
implied by the fixture's opening odds, which are known before the deadline.
"""

from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from fpl_models import team_strength

DEADLINE_BEFORE_KICKOFF = timedelta(minutes=90)
EWM_HALF_LIFE_GWS = 8.0
# Extra half-lives: short (recent form) and long (underlying ability) rates.
EXTRA_HALF_LIVES = (4.0, 16.0)
POSITIONS = ("GKP", "DEF", "MID", "FWD")

# Per-gameweek sums smoothed into per-90 rates.
RATE_STATS = (
    "goals_scored", "expected_goals", "assists", "expected_assists", "bonus", "bps",
    "saves", "yellow_cards", "cbit", "cbirt", "total_points",
)  # fmt: skip

FEATURES = (
    "position_code", "is_home", "price", "n_fixtures",
    "team_lambda", "opponent_lambda", "p_clean_sheet",
    "minutes_l1", "minutes_l3", "minutes_l6", "appear_rate_l6", "played60_rate_l6",
    "starts_rate_l6", "minutes_exposure_l10", "gws_since_appearance", "gws_in_history",
    *(f"{s}_per90" for s in RATE_STATS), "points_ewm",
)  # fmt: skip

EXTRA_FEATURES = tuple(
    f"{s}_per90_hl{hl:g}" for hl in EXTRA_HALF_LIVES for s in RATE_STATS
) + tuple(f"points_ewm_hl{hl:g}" for hl in EXTRA_HALF_LIVES)
ALL_FEATURES = FEATURES + EXTRA_FEATURES

# Crowd signal: what managers did before a deadline. A gameweek's transfer counts are
# transfers made *before* its deadline (gameweek 1 shows none in every season), so
# they're pre-deadline information. Two variants:
#   lagged:  the previous gameweek's complete window. Safe to use live.
#   current: this gameweek's window. Complete in training but only partial when
#            predicting days before a deadline (train/serve skew); measured only.
CROWD_LAGGED = ("crowd_log_owners_prev", "crowd_net_transfer_rate_prev")
CROWD_CURRENT = ("crowd_log_owners", "crowd_net_transfer_rate")
# Window-invariant variant: a player's net transfers as a share of all transfer
# activity in the gameweek so far. Its meaning barely depends on how far into the
# window it is measured, so complete-window training data matches partial live data.
CROWD_SHARE = ("crowd_ownership", "crowd_net_transfer_share")
CROWD_ALL = (*CROWD_CURRENT, *CROWD_SHARE, *CROWD_LAGGED)

# Opta detail (FPL-Elo-Insights, 2024-25 onwards; null before): per-90, EWM.
OPTA_STATS = (
    "touches_opposition_box", "chances_created", "total_shots", "opta_xgot",
    "big_chances_missed",
)  # fmt: skip
OPTA_FEATURES = tuple(f"{s}_per90" for s in OPTA_STATS)

# Understat detail (2014-15 onwards; null unless `understat.with_understat` added the
# columns): per-90, EWM. Shots and key passes stand in for Opta before 2024-25.
UNDERSTAT_STATS = ("us_shots", "us_key_passes", "us_npxg", "us_xg_chain", "us_xg_buildup")
UNDERSTAT_FEATURES = tuple(f"{s}_per90" for s in UNDERSTAT_STATS)

# Schedule: known in advance from the fixture list, so pre-deadline information.
SCHEDULE_FEATURES = (
    "team_rest_days", "opponent_rest_days", "team_matches_prev_14d", "opponent_matches_prev_14d",
)  # fmt: skip
# All competitions (E11): league plus cup and European matches, 2025-26 onwards.
CONGESTION_FEATURES = (
    "team_all_rest_days", "team_all_days_to_next", "team_all_matches_prev_14d",
    "team_days_since_europe", "team_days_to_europe",
)  # fmt: skip
# A player's form and load across all competitions (E11b), 2025-26 onwards. "Short":
# windows up to 14 days and EWM with a 4-match half-life; "long" adds 28 days,
# season-to-date and a 15-match half-life.
ALL_COMP_STATS = ("xg", "xa", "shots", "box_touches", "chances_created")
ALL_COMP_HALF_LIVES = {"short": 4.0, "long": 15.0}
ALL_COMP_WINDOWS = {"short": ("7d", "14d"), "long": ("28d", "season")}
FATIGUE_SHORT = (
    "ac_minutes_7d", "ac_minutes_14d", "ac_minutes_share_7d", "ac_minutes_share_14d",
    "ac_last_minutes", "ac_days_since_appearance",
)  # fmt: skip
FATIGUE_LONG = (
    "ac_minutes_28d", "ac_minutes_season", "ac_minutes_share_28d", "ac_minutes_share_season",
)  # fmt: skip
FORM_SHORT = tuple(
    [f"ac_{s}_per90_hl4" for s in ALL_COMP_STATS] + [f"ac_{s}_per90_14d" for s in ALL_COMP_STATS]
)
FORM_LONG = tuple(
    [f"ac_{s}_per90_hl15" for s in ALL_COMP_STATS]
    + [f"ac_{s}_per90_{w}" for w in ("28d", "season") for s in ALL_COMP_STATS]
)
ALL_COMP_FEATURES = FATIGUE_SHORT + FATIGUE_LONG + FORM_SHORT + FORM_LONG

TARGETS = (
    "minutes",
    "goals_scored",
    "assists",
    "bonus",
    "saves",
    "yellow_cards",
    "defcon_hit",
)


def load_fixture_rows(warehouse: Path) -> pl.DataFrame:
    """Played player-fixture rows with outcomes, match context and deadlines."""
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        connection.execute("set TimeZone = 'UTC'")
        rows = connection.sql(
            """
            select pm.season, pm.gameweek, pm.kickoff_utc, pm.match_key, pm.player_code,
                pm.position, pm.team_key, pm.is_home, pm.price, pm.is_provisional,
                pm.minutes, pm.starts, pm.goals_scored, pm.assists, pm.bonus, pm.bps,
                pm.saves, pm.yellow_cards, pm.total_points, pm.expected_goals,
                pm.expected_assists, pm.tackles, pm.clearances_blocks_interceptions,
                pm.recoveries, pm.selected, pm.transfers_balance, pm.opponent_team_key,
                pm.touches_opposition_box, pm.chances_created, pm.total_shots,
                pm.opta_xgot, pm.big_chances_missed
            from fct_player_matches as pm
            """
        ).pl()
    return add_deadlines(rows).filter(~pl.col("is_provisional"))


def add_deadlines(rows: pl.DataFrame) -> pl.DataFrame:
    deadlines = rows.group_by("season", "gameweek").agg(
        (pl.col("kickoff_utc").min() - DEADLINE_BEFORE_KICKOFF).alias("deadline_utc")
    )
    return rows.join(deadlines, on=["season", "gameweek"], how="left", maintain_order="left")


def match_lambdas(matches: pl.DataFrame) -> pl.DataFrame:
    """Opening-odds lambdas per match (pre-deadline information), league mean if unpriced."""
    rows = []
    fallback = matches.filter(pl.col("is_played"))
    home_mean = float(fallback["home_goals"].mean() or 1.5)  # type: ignore[arg-type]
    away_mean = float(fallback["away_goals"].mean() or 1.2)  # type: ignore[arg-type]
    for r in matches.iter_rows(named=True):
        if r["p_home_win"] is None:
            lh, la = home_mean, away_mean
        else:
            lh, la = team_strength.lambdas_from_odds(
                r["p_home_win"], r["p_draw"], r["p_away_win"], r["p_over_2_5"]
            )
        rows.append({"match_key": r["match_key"], "lambda_home": lh, "lambda_away": la})
    return pl.DataFrame(rows)


def add_team_context(rows: pl.DataFrame, lambdas: pl.DataFrame) -> pl.DataFrame:
    """Each row's team and opponent expected goals, and clean-sheet probability."""
    return (
        rows.join(lambdas, on="match_key", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("is_home"))
            .then(pl.col("lambda_home"))
            .otherwise(pl.col("lambda_away"))
            .alias("team_lambda"),
            pl.when(pl.col("is_home"))
            .then(pl.col("lambda_away"))
            .otherwise(pl.col("lambda_home"))
            .alias("opponent_lambda"),
        )
        .with_columns((-pl.col("opponent_lambda")).exp().alias("p_clean_sheet"))
        .drop("lambda_home", "lambda_away")
    )


def _with_defensive_stats(rows: pl.DataFrame) -> pl.DataFrame:
    cbit = pl.col("tackles") + pl.col("clearances_blocks_interceptions")
    cbirt = cbit + pl.col("recoveries")
    hit = (
        pl.when(pl.col("position") == "GKP")
        .then(False)
        .when(pl.col("position") == "DEF")
        .then(cbit >= 10)
        .otherwise(cbirt >= 12)
    )
    return rows.with_columns(cbit.alias("cbit"), cbirt.alias("cbirt"), hit.alias("defcon_hit"))


ROLLED = tuple(c for c in ALL_FEATURES + OPTA_FEATURES + UNDERSTAT_FEATURES if c not in (
    "position_code", "is_home", "price", "n_fixtures",
    "team_lambda", "opponent_lambda", "p_clean_sheet",
))  # fmt: skip


def _player_state(rows: pl.DataFrame) -> pl.DataFrame:
    """Per (player, gameweek): rolling state *including* that gameweek (unshifted)."""
    # Nulls (stat not recorded that season) must stay null rather than sum to 0.
    rows = rows.with_columns(
        pl.lit(None, pl.Float64).alias(s) for s in UNDERSTAT_STATS if s not in rows.columns
    )
    sums = [
        pl.when(pl.col(s).is_null().all()).then(None).otherwise(pl.col(s).sum()).alias(s)
        for s in ("minutes", *RATE_STATS, *OPTA_STATS, *UNDERSTAT_STATS)
    ]
    per_gw = (
        _with_defensive_stats(rows)
        .group_by("player_code", "season", "gameweek", "deadline_utc")
        .agg(
            *sums,
            pl.len().alias("fixtures"),
            (pl.col("minutes") > 0).sum().alias("appearances"),
            (pl.col("minutes") >= 60).sum().alias("played60"),
            pl.col("starts").sum().alias("starts"),
        )
        .sort("player_code", "deadline_utc")
        .with_columns(pl.int_range(pl.len()).over("player_code").alias("gw_index"))
    )

    def per_fixture(col: str, n: int) -> pl.Expr:
        window = pl.col(col).rolling_sum(n, min_samples=1)
        return (window / pl.col("fixtures").rolling_sum(n, min_samples=1)).over("player_code")

    def ewm(col: str, half_life: float = EWM_HALF_LIFE_GWS) -> pl.Expr:
        alpha = 1 - 0.5 ** (1 / half_life)
        return pl.col(col).ewm_mean(alpha=alpha, ignore_nulls=True).over("player_code")

    extra_rates = [
        (ewm(s, hl) / ewm("minutes", hl) * 90).alias(f"{s}_per90_hl{hl:g}")
        for hl in EXTRA_HALF_LIVES
        for s in RATE_STATS
    ]
    extra_form = [ewm("total_points", hl).alias(f"points_ewm_hl{hl:g}") for hl in EXTRA_HALF_LIVES]

    last_appearance = pl.when(pl.col("appearances") > 0).then(pl.col("gw_index"))
    return per_gw.with_columns(
        per_fixture("minutes", 1).alias("minutes_l1"),
        per_fixture("minutes", 3).alias("minutes_l3"),
        per_fixture("minutes", 6).alias("minutes_l6"),
        per_fixture("appearances", 6).alias("appear_rate_l6"),
        per_fixture("played60", 6).alias("played60_rate_l6"),
        per_fixture("starts", 6).alias("starts_rate_l6"),
        pl.col("minutes")
        .rolling_sum(10, min_samples=1)
        .over("player_code")
        .alias("minutes_exposure_l10"),
        *[(ewm(s) / ewm("minutes") * 90).alias(f"{s}_per90") for s in RATE_STATS],
        ewm("total_points").alias("points_ewm"),
        *extra_rates,
        *extra_form,
        *[(ewm(s) / ewm("minutes") * 90).alias(f"{s}_per90") for s in OPTA_STATS],
        *[(ewm(s) / ewm("minutes") * 90).alias(f"{s}_per90") for s in UNDERSTAT_STATS],
        (pl.col("gw_index") - last_appearance.forward_fill().over("player_code")).alias(
            "gws_since_appearance"
        ),
        (pl.col("gw_index") + 1).alias("gws_in_history"),
    ).select("player_code", "deadline_utc", *ROLLED)


def player_history_features(rows: pl.DataFrame) -> pl.DataFrame:
    """Per (player, gameweek): the state *before* that gameweek (shifted by one)."""
    state = _player_state(rows)
    return state.select(
        "player_code",
        "deadline_utc",
        *[pl.col(c).shift(1).over("player_code") for c in ROLLED],
    )


def latest_player_state(rows: pl.DataFrame) -> pl.DataFrame:
    """Each player's state after his last gameweek: the features for the next one."""
    return (
        _player_state(rows)
        .group_by("player_code")
        .agg(pl.all().sort_by("deadline_utc").last())
        .drop("deadline_utc")
    )


def schedule_features(matches: pl.DataFrame) -> pl.DataFrame:
    """Per (match_key, team_key): days since the team's previous league match, and how
    many league matches it played in the 14 days before kickoff.

    Built from every scheduled fixture (played or not): kickoff times are published in
    advance, so this is pre-deadline information. European and cup matches aren't in
    the league schedule yet, so congestion from them is not captured here.
    """
    sides = (
        pl.concat(
            [
                matches.select(
                    "match_key", "kickoff_utc", pl.col(f"{side}_team_key").alias("team_key")
                )
                for side in ("home", "away")
            ]
        )
        .drop_nulls("kickoff_utc")
        .sort("team_key", "kickoff_utc")
    )
    previous = pl.col("kickoff_utc").shift(1).over("team_key")
    recent = sides.rolling("kickoff_utc", period="14d", closed="left", group_by="team_key").agg(
        pl.len().alias("matches_prev_14d")
    )
    return (
        sides.with_columns(
            ((pl.col("kickoff_utc") - previous).dt.total_hours() / 24).alias("rest_days")
        )
        .join(recent, on=["team_key", "kickoff_utc"], how="left")
        .select("match_key", "team_key", "rest_days", pl.col("matches_prev_14d").cast(pl.Int64))
    )


EUROPEAN = ("Champions League", "Europa League", "Conference League")
# After a season's last recorded cup match, cup coverage is unknown (upstream lacks the
# 2025-26 knockout rounds), so later league rows get nulls rather than "no cup games".
COVERAGE_GRACE = pl.duration(days=7)


def load_cup_matches(warehouse: Path) -> pl.DataFrame:
    """Cup and European matches per Premier League club (empty if not ingested)."""
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        connection.execute("set TimeZone = 'UTC'")
        tables = {t for (t,) in connection.sql("show tables").fetchall()}
        if "fct_cup_matches" not in tables:
            return pl.DataFrame(
                schema={"season": pl.String, "competition": pl.String,
                        "kickoff_utc": pl.Datetime("us", "UTC"), "team_key": pl.String}
            )  # fmt: skip
        return connection.sql(
            "select season, competition, kickoff_utc, team_key from fct_cup_matches"
        ).pl()


def congestion_features(matches: pl.DataFrame, cups: pl.DataFrame) -> pl.DataFrame:
    """Per (match_key, team_key) league fixture, from the schedule in *all* competitions:

        all_rest_days         days since the team's previous match, league or cup
        all_days_to_next      days until its next match, league or cup
        all_matches_prev_14d  matches in any competition in the 14 days before
        days_since_europe     days since its last European match
        days_to_europe        days until its next European match

    Cup kickoffs are scheduled in advance, so this is pre-deadline information. Seasons
    without cup data (before 2025-26) and league fixtures after a season's last recorded
    cup match (plus a week) are null: the model can't tell "no cup games" from "no data".
    """
    league = pl.concat(
        [
            matches.select(
                "match_key", "season", "kickoff_utc", pl.col(f"{side}_team_key").alias("team_key")
            )
            for side in ("home", "away")
        ]
    ).with_columns(competition=pl.lit("league"))
    cup = cups.select(
        pl.lit(None, pl.String).alias("match_key"), "season", "kickoff_utc", "team_key",
        "competition",
    )  # fmt: skip
    sides = (
        pl.concat([league, cup.cast(league.schema)])
        .drop_nulls("kickoff_utc")
        .sort("team_key", "kickoff_utc")
    )
    kickoff = pl.col("kickoff_utc")
    europe = pl.when(pl.col("competition").is_in(EUROPEAN)).then(kickoff)

    def days(later: pl.Expr, earlier: pl.Expr) -> pl.Expr:
        return (later - earlier).dt.total_hours() / 24

    recent = sides.rolling("kickoff_utc", period="14d", closed="left", group_by="team_key").agg(
        pl.len().alias("all_matches_prev_14d")
    )
    covered = cups.group_by("season").agg(
        (pl.col("kickoff_utc").max() + COVERAGE_GRACE).alias("coverage_end")
    )
    features = ("all_rest_days", "all_days_to_next", "all_matches_prev_14d",
                "days_since_europe", "days_to_europe")  # fmt: skip
    return (
        sides.with_columns(
            all_rest_days=days(kickoff, kickoff.shift(1)).over("team_key"),
            all_days_to_next=days(kickoff.shift(-1), kickoff).over("team_key"),
            # Within the season: the previous season's coverage may be incomplete.
            days_since_europe=days(kickoff, europe.shift(1).forward_fill()).over(
                "team_key", "season"
            ),
            days_to_europe=days(europe.shift(-1).backward_fill(), kickoff).over(
                "team_key", "season"
            ),
        )
        .join(recent.unique(["team_key", "kickoff_utc"]), on=["team_key", "kickoff_utc"])
        .filter(pl.col("competition") == "league")
        .join(covered, on="season", how="left")
        .with_columns(
            pl.when(pl.col("kickoff_utc") <= pl.col("coverage_end")).then(pl.col(c)).alias(c)
            for c in features
        )
        .select("match_key", "team_key", *features)
    )


def add_congestion(rows: pl.DataFrame, congestion: pl.DataFrame) -> pl.DataFrame:
    """Join the team's all-competition congestion (`congestion_features`) as team_*."""
    renamed = congestion.rename({c: f"team_{c}" for c in congestion.columns[2:]})
    return rows.join(renamed, on=["match_key", "team_key"], how="left", maintain_order="left")


# Canonical stat -> (league fixture-row column, cup appearance column).
_ALL_COMP_SOURCES = {
    "xg": ("expected_goals", "xg"),
    "xa": ("expected_assists", "xa"),
    "shots": ("total_shots", "total_shots"),
    "box_touches": ("touches_opposition_box", "touches_opposition_box"),
    "chances_created": ("chances_created", "chances_created"),
}
_WINDOW_DAYS = {"7d": 7, "14d": 14, "28d": 28}
# After a season's last recorded cup appearance, coverage is unknown: null, not zero.
ALL_COMP_COVERAGE_GRACE = pl.duration(days=10)
MIN_WINDOW_MINUTES = 45  # a per-90 rate over fewer minutes is mostly noise


def load_cup_player_matches(warehouse: Path) -> pl.DataFrame:
    """Players' cup and European appearances (empty if not ingested)."""
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        connection.execute("set TimeZone = 'UTC'")
        tables = {t for (t,) in connection.sql("show tables").fetchall()}
        if "fct_cup_player_matches" not in tables:
            return pl.DataFrame(
                schema={"season": pl.String, "kickoff_utc": pl.Datetime("us", "UTC"),
                        "player_code": pl.Int64, "minutes": pl.Int64,
                        **{cup: pl.Float64 for _, cup in _ALL_COMP_SOURCES.values()}}
            )  # fmt: skip
        return connection.sql(
            "select season, kickoff_utc, player_code, minutes, xg, xa, total_shots,"
            " touches_opposition_box, chances_created from fct_cup_player_matches"
        ).pl()


def _asof(
    targets: pl.DataFrame, table: pl.DataFrame, at: str, by: list[str], suffix: str
) -> pl.DataFrame:
    """Join `table`'s state as of strictly before each target's `at` time."""
    right = table.rename({c: f"{c}{suffix}" for c in table.columns if c not in by})
    return targets.sort(at).join_asof(
        right.sort(f"kickoff_utc{suffix}"),
        left_on=at,
        right_on=f"kickoff_utc{suffix}",
        by=by,
        strategy="backward",
        allow_exact_matches=False,
    )


def all_competition_features(
    rows: pl.DataFrame, cup_players: pl.DataFrame, cups: pl.DataFrame
) -> pl.DataFrame:
    """Per (match_key, player_code): form and load over league *and* cup matches played
    before the deadline (E11b).

        ac_minutes_{7d,14d,28d,season}        minutes in all competitions
        ac_minutes_share_{...}                those minutes / (90 x the club's matches)
        ac_last_minutes                       minutes in his club's last match (league
                                              or a cup match he played)
        ac_days_since_appearance              days from his last appearance to kickoff
        ac_<stat>_per90_{14d,28d,season}      per-90 rates over the window
        ac_<stat>_per90_hl{4,15}              EWM per-90 over appearances (half-life in
                                              matches), stats xG, xA, shots, box
                                              touches, chances created

    League xG/xA are FPL's (Opta), the rest Opta via FPL-Elo-Insights (2024-25+); cup
    stats exist from 2025-26. Outside cup coverage every feature is null: the model
    can't tell "no cup games" from "no data". `eval_cup_minutes_prev_6d` (cup minutes in
    the 6 days before kickoff, known at the deadline) is for evaluation, not a feature.
    """
    stats = list(_ALL_COMP_SOURCES)
    league = rows.select(
        "player_code", "season", "kickoff_utc",
        pl.col("minutes").cast(pl.Float64),
        *[pl.col(src).cast(pl.Float64).alias(s) for s, (src, _) in _ALL_COMP_SOURCES.items()],
        pl.lit(value=False).alias("is_cup"),
    )  # fmt: skip
    cup = cup_players.select(
        "player_code", "season", "kickoff_utc",
        pl.col("minutes").cast(pl.Float64),
        *[pl.col(src).cast(pl.Float64).alias(s) for s, (_, src) in _ALL_COMP_SOURCES.items()],
        pl.lit(value=True).alias("is_cup"),
    )  # fmt: skip
    matches = (
        pl.concat([league, cup.cast(league.schema)])
        .with_columns(pl.col(*stats).fill_null(0.0))
        .sort("player_code", "kickoff_utc")
    )
    cum = matches.with_columns(
        *[pl.col(c).cum_sum().over("player_code").alias(f"c_{c}") for c in ["minutes", *stats]],
        *[
            pl.col(c).cum_sum().over("player_code", "season").alias(f"s_{c}")
            for c in ["minutes", *stats]
        ],
        pl.when(pl.col("is_cup")).then(pl.col("minutes")).otherwise(0.0)
        .cum_sum().over("player_code").alias("c_cup_minutes"),
    )  # fmt: skip
    apps = matches.filter(pl.col("minutes") > 0)
    ewm = apps.with_columns(
        *[
            (
                pl.col(s).ewm_mean(half_life=h).over("player_code")
                / pl.col("minutes").ewm_mean(half_life=h).over("player_code")
                * 90
            ).alias(f"ac_{s}_per90_hl{h:g}")
            for h in ALL_COMP_HALF_LIVES.values()
            for s in stats
        ]
    ).select("player_code", "kickoff_utc", pl.col("^ac_.*$"))

    # The club's matches in all competitions, for minutes shares.
    club = (
        pl.concat(
            [
                rows.select("team_key", "season", "kickoff_utc").unique(),
                cups.select("team_key", "season", "kickoff_utc").drop_nulls(),
            ]
        )
        .unique()
        .sort("team_key", "kickoff_utc")
        .with_columns(
            n=pl.int_range(1, pl.len() + 1).over("team_key"),
            sn=pl.int_range(1, pl.len() + 1).over("team_key", "season"),
        )
    )

    targets = rows.select(
        "match_key", "player_code", "season", "team_key", "kickoff_utc", "deadline_utc"
    ).unique(["match_key", "player_code"])
    targets = targets.with_columns(
        *[(pl.col("deadline_utc") - pl.duration(days=d)).alias(f"at_{w}")
          for w, d in _WINDOW_DAYS.items()],
        (pl.col("kickoff_utc") - pl.duration(days=6)).alias("at_6d_before_kickoff"),
    )  # fmt: skip
    cum_cols = ["kickoff_utc", "minutes", "c_minutes", "c_cup_minutes", *[f"c_{s}" for s in stats]]
    t = _asof(targets, cum.select("player_code", *cum_cols), "deadline_utc", ["player_code"], "_dl")
    season_cols = ["kickoff_utc", "s_minutes", *[f"s_{s}" for s in stats]]
    t = _asof(
        t, cum.select("player_code", "season", *season_cols),
        "deadline_utc", ["player_code", "season"], "_season",
    )  # fmt: skip
    for w in _WINDOW_DAYS:
        t = _asof(
            t, cum.select("player_code", "kickoff_utc", "c_minutes", *[f"c_{s}" for s in stats]),
            f"at_{w}", ["player_code"], f"_{w}",
        )  # fmt: skip
    t = _asof(
        t, cum.select("player_code", "kickoff_utc", "c_cup_minutes"),
        "at_6d_before_kickoff", ["player_code"], "_6d",
    )  # fmt: skip
    t = _asof(t, ewm, "deadline_utc", ["player_code"], "_app")
    for w in [*_WINDOW_DAYS, "season"]:
        at = "deadline_utc" if w == "season" else f"at_{w}"
        clubs = club.select("team_key", "kickoff_utc", "n")
        if w == "season":
            joined = _asof(
                t.select("match_key", "player_code", "team_key", "season", "deadline_utc"),
                club.select("team_key", "season", "kickoff_utc", "sn"),
                "deadline_utc",
                ["team_key", "season"],
                "_club",
            ).select(
                "match_key", "player_code", pl.col("sn_club").fill_null(0).alias("club_season")
            )
        else:
            now = _asof(
                t.select("match_key", "player_code", "team_key", "deadline_utc"),
                clubs.select("team_key", "kickoff_utc", "n"),
                "deadline_utc",
                ["team_key"],
                "_now",
            )
            then = _asof(
                t.select("match_key", "player_code", "team_key", at),
                clubs.select("team_key", "kickoff_utc", "n"),
                at,
                ["team_key"],
                "_then",
            )
            joined = now.join(then, on=["match_key", "player_code"]).select(
                "match_key", "player_code",
                (pl.col("n_now").fill_null(0) - pl.col("n_then").fill_null(0)).alias(f"club_{w}"),
            )  # fmt: skip
        t = t.join(joined, on=["match_key", "player_code"], how="left")

    def window(col: str, w: str) -> pl.Expr:
        if w == "season":
            return pl.col(f"s_{col}_season").fill_null(0.0)
        return pl.col(f"c_{col}_dl").fill_null(0.0) - pl.col(f"c_{col}_{w}").fill_null(0.0)

    def per90(stat: str, w: str) -> pl.Expr:
        minutes = window("minutes", w)
        return pl.when(minutes >= MIN_WINDOW_MINUTES).then(window(stat, w) / minutes * 90)

    def share(w: str) -> pl.Expr:
        club_matches = pl.col(f"club_{w}")
        return pl.when(club_matches > 0).then(window("minutes", w) / (90 * club_matches))

    features = t.select(
        "match_key", "player_code", "season", "deadline_utc",
        *[window("minutes", w).alias(f"ac_minutes_{w}") for w in [*_WINDOW_DAYS, "season"]],
        *[share(w).alias(f"ac_minutes_share_{w}") for w in [*_WINDOW_DAYS, "season"]],
        pl.col("minutes_dl").alias("ac_last_minutes"),
        ((pl.col("kickoff_utc") - pl.col("kickoff_utc_app")).dt.total_hours() / 24)
        .alias("ac_days_since_appearance"),
        *[per90(s, w).alias(f"ac_{s}_per90_{w}") for w in ("14d", "28d", "season") for s in stats],
        *[pl.col(f"ac_{s}_per90_hl{h:g}_app").alias(f"ac_{s}_per90_hl{h:g}")
          for h in ALL_COMP_HALF_LIVES.values() for s in stats],
        (pl.col("c_cup_minutes_dl").fill_null(0.0) - pl.col("c_cup_minutes_6d").fill_null(0.0))
        .alias("eval_cup_minutes_prev_6d"),
    )  # fmt: skip
    covered = cup_players.group_by("season").agg(
        (pl.col("kickoff_utc").max() + ALL_COMP_COVERAGE_GRACE).alias("coverage_end")
    )
    names = [c for c in features.columns if c.startswith(("ac_", "eval_"))]
    return (
        features.join(covered, on="season", how="left")
        .with_columns(
            pl.when(pl.col("deadline_utc") <= pl.col("coverage_end")).then(pl.col(c)).alias(c)
            for c in names
        )
        .select("match_key", "player_code", *names)
    )


def add_all_competition(rows: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Join `all_competition_features` onto fixture rows."""
    return rows.join(features, on=["match_key", "player_code"], how="left", maintain_order="left")


def add_schedule(rows: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    team = schedule.rename(
        {"rest_days": "team_rest_days", "matches_prev_14d": "team_matches_prev_14d"}
    )
    opponent = schedule.rename({
        "team_key": "opponent_team_key", "rest_days": "opponent_rest_days",
        "matches_prev_14d": "opponent_matches_prev_14d",
    })  # fmt: skip
    return rows.join(team, on=["match_key", "team_key"], how="left", maintain_order="left").join(
        opponent, on=["match_key", "opponent_team_key"], how="left", maintain_order="left"
    )


def crowd_features(rows: pl.DataFrame) -> pl.DataFrame:
    """Per (player, gameweek): ownership and net transfers, current and lagged."""
    per_gw = (
        rows.group_by("player_code", "season", "gameweek", "deadline_utc")
        .agg(pl.col("selected").max(), pl.col("transfers_balance").max())
        .sort("player_code", "deadline_utc")
        .with_columns(
            pl.col("selected").cast(pl.Float64).log1p().alias("crowd_log_owners"),
            (pl.col("transfers_balance") / (pl.col("selected") + 1000)).alias(
                "crowd_net_transfer_rate"
            ),
            (
                pl.col("transfers_balance")
                / pl.col("transfers_balance")
                .abs()
                .sum()
                .over("season", "gameweek")
                .clip(lower_bound=1)
            ).alias("crowd_net_transfer_share"),
            # Fraction of managers owning him: every manager owns exactly 15 players, so
            # managers = all ownership / 15. Stable as FPL grows, unlike raw counts.
            (
                pl.col("selected")
                / (pl.col("selected").sum().over("season", "gameweek") / 15).clip(lower_bound=1)
            ).alias("crowd_ownership"),
        )
    )
    return per_gw.select(
        "player_code",
        "deadline_utc",
        *CROWD_CURRENT,
        "crowd_net_transfer_share",
        "crowd_ownership",
        *[pl.col(c).shift(1).over("player_code").alias(f"{c}_prev") for c in CROWD_CURRENT],
    )


def build_features(
    rows: pl.DataFrame, lambdas: pl.DataFrame, schedule: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Played fixture rows with pre-deadline features and outcome targets.

    `schedule` (from `schedule_features`) adds rest-day and congestion features.
    """
    rows = _with_defensive_stats(add_team_context(rows, lambdas))
    if schedule is not None:
        rows = add_schedule(rows, schedule)
    history = player_history_features(rows).join(
        crowd_features(rows), on=["player_code", "deadline_utc"], how="left"
    )
    position_code = pl.col("position").replace_strict(
        {p: i for i, p in enumerate(POSITIONS)}, return_dtype=pl.Int8
    )
    n_fixtures = pl.len().over("player_code", "season", "gameweek").alias("n_fixtures")
    return rows.with_columns(position_code.alias("position_code"), n_fixtures).join(
        history, on=["player_code", "deadline_utc"], how="left", maintain_order="left"
    )


def feature_matrix(frame: pl.DataFrame, features: tuple[str, ...] = FEATURES) -> np.ndarray:
    return frame.select(pl.col(features).cast(pl.Float64)).to_numpy()
