"""E8: where does v2's error come from? An exact decomposition of squared error.

v2 predicts points as a sum of components (appearance, goals, assists, clean sheet,
conceding, saves, bonus, cards, DefCon). For every regular player-gameweek of the
tuning seasons, each component's points error splits into two parts:

    minutes   predicted - (predicted rescaled to the minutes actually played)
    returns   (rescaled prediction) - actual           # error given the minutes

plus "other" (actual points no component predicts: penalties, red cards, own goals,
and the clean-sheet/conceding approximations). The total error e is the sum of these
parts, so the mean squared error splits exactly as MSE = sum_c cov(e_c, e) - each
part's share of the MSE (shares can be negative when a part offsets the others).

Second, error by player segment (new to the league, back from absence, promoted
team, position, price): where the absolute error is concentrated.

    uv run python -m fpl_models.e8   # writes data/experiments/e8/rows.parquet
"""

from pathlib import Path

import numpy as np
import polars as pl

from fpl_core import scoring
from fpl_models import player_features, player_model, team_strength
from fpl_models.data import (
    CONTEXT_COLUMNS,
    DEFAULT_WAREHOUSE,
    load_player_matches,
    player_gameweeks,
)
from fpl_models.experiments import TUNING_SEASONS
from fpl_models.metrics import regulars

OUT = Path("data/experiments/e8")
OUTCOMES = ("minutes", "goals_scored", "assists", "bonus", "saves", "yellow_cards")


def record(features: pl.DataFrame, gameweeks: pl.DataFrame) -> pl.DataFrame:
    """Walk-forward v2 fixture rows of the tuning seasons, with events and outcomes."""
    model = player_model.ComponentModel(features, config=player_model.PRODUCTION, name="v2")
    deadlines = (
        gameweeks.filter(pl.col("season").is_in(TUNING_SEASONS))
        .select("season", "gameweek", "deadline_utc")
        .unique()
        .sort("deadline_utc")
    )
    frames = []
    for season, gameweek, deadline in deadlines.iter_rows():
        history = gameweeks.filter(pl.col("deadline_utc") < deadline)
        targets = gameweeks.filter((pl.col("season") == season) & (pl.col("gameweek") == gameweek))
        rows, events = model.fixture_events(history, targets.select(CONTEXT_COLUMNS))
        frames.append(
            pl.concat([rows, events], how="horizontal").with_columns(
                player_model.expected_points(rows, events).alias("xp")
            )
        )
    return pl.concat(frames)


def components(rows: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Per fixture row: predicted, minutes-rescaled and actual points of each component."""
    conceded = matches.select("match_key", home_conceded="away_goals", away_conceded="home_goals")
    rows = rows.join(conceded, on="match_key", how="left").with_columns(
        conceded=pl.when(pl.col("is_home")).then("home_conceded").otherwise("away_conceded")
    )
    minutes = rows["minutes"].to_numpy()
    long_play = minutes >= scoring.LONG_PLAY_MINUTES
    short = (minutes > 0) & ~long_play
    # Expected minutes from the minutes model, valued at each class's average minutes.
    mean_long = float(minutes[long_play].mean())
    mean_short = float(minutes[short].mean())
    p60, p1_59 = rows["p_60"].to_numpy(), rows["p_1_59"].to_numpy()
    expected_minutes = p60 * mean_long + p1_59 * mean_short
    scale = np.divide(minutes, expected_minutes, out=np.zeros_like(p60), where=expected_minutes > 0)

    position = rows["position"].to_list()
    goal = np.array([player_model.GOAL_POINTS[p] for p in position])
    cs = np.array([player_model.CLEAN_SHEET_POINTS[p] for p in position])
    concedes = np.array([p in player_model.CONCEDING for p in position], dtype=float)
    keeper = np.array([p == "GKP" for p in position])
    defcon_on = (rows["season"] >= player_model.DEFCON_FROM_SEASON).to_numpy() & ~keeper
    lam = rows["opponent_lambda"].to_numpy()
    gc = rows["conceded"].fill_null(0).to_numpy()
    p_cs = np.exp(-lam)
    floor_half = player_model.expected_floor_div(lam, scoring.GOALS_CONCEDED_PER_PENALTY)
    saves_points = player_model.expected_floor_div(
        rows["e_saves"].to_numpy(), scoring.SAVES_PER_POINT
    )
    defcon = defcon_on * scoring.DEFENSIVE_CONTRIBUTION_POINTS
    p_defcon = rows["p_defcon"].to_numpy()
    played = (minutes > 0) / np.maximum(rows["p_play"].to_numpy(), 1e-9)
    hit = rows["defcon_hit"].fill_null(value=False).cast(pl.Float64).to_numpy()

    def counted(event: str, points: np.ndarray | float, actual: str) -> tuple[np.ndarray, ...]:
        pred = rows[f"e_{event}"].to_numpy() * points
        return pred, pred * scale, rows[actual].fill_null(0).to_numpy() * points

    parts = {
        "appearance": (
            p1_59 * scoring.SHORT_PLAY_POINTS + p60 * scoring.LONG_PLAY_POINTS,
            short * scoring.SHORT_PLAY_POINTS + long_play * scoring.LONG_PLAY_POINTS,
            short * scoring.SHORT_PLAY_POINTS + long_play * scoring.LONG_PLAY_POINTS,
        ),
        "goals": counted("goals_scored", goal, "goals_scored"),
        "assists": counted("assists", scoring.ASSIST_POINTS, "assists"),
        "clean_sheet": (p60 * p_cs * cs, long_play * p_cs * cs, long_play * (gc == 0) * cs),
        "conceding": (
            -concedes * p60 * floor_half,
            -concedes * long_play * floor_half,
            -concedes * long_play * (gc // scoring.GOALS_CONCEDED_PER_PENALTY),
        ),
        "saves": (keeper * saves_points, keeper * saves_points * scale, keeper * (
            rows["saves"].fill_null(0).to_numpy() // scoring.SAVES_PER_POINT
        )),
        "bonus": counted("bonus", 1.0, "bonus"),
        "cards": counted("yellow_cards", scoring.YELLOW_CARD_POINTS, "yellow_cards"),
        "defcon": (defcon * p_defcon, defcon * p_defcon * played, defcon * hit),
    }  # fmt: skip
    out = rows.select("season", "gameweek", "player_code", "position", "xp", "total_points")
    actual_sum = np.zeros(rows.height)
    for name, (pred, rescaled, actual) in parts.items():
        out = out.with_columns(
            pl.Series(f"{name}_minutes", pred - rescaled),
            pl.Series(f"{name}_returns", rescaled - actual),
        )
        actual_sum += actual
    return out.with_columns(other=pl.Series(-(rows["total_points"].to_numpy() - actual_sum)))


def decompose(parts: pl.DataFrame) -> pl.DataFrame:
    """MSE share of each error part, per player-gameweek (double gameweeks summed)."""
    error_cols = [c for c in parts.columns if c.endswith(("_minutes", "_returns")) or c == "other"]
    per_gw = parts.group_by("season", "gameweek", "player_code").agg(
        pl.col("xp", "total_points", *error_cols).sum()
    )
    e = (per_gw["xp"] - per_gw["total_points"]).to_numpy()
    check = sum(per_gw[c].to_numpy() for c in error_cols)
    if not np.allclose(check, e, atol=1e-6):
        raise ValueError("parts must add up to the total error")
    mse = float(np.mean(e**2))
    rows = [
        {"part": c, "mse_share": float(np.mean(per_gw[c].to_numpy() * e)) / mse,
         "mean": float(per_gw[c].mean()), "mae_alone": float(per_gw[c].abs().mean())}  # type: ignore[arg-type]
        for c in error_cols
    ]  # fmt: skip
    print(f"{per_gw.height} regular player-gameweeks; MAE {np.abs(e).mean():.4f}, MSE {mse:.3f}")
    table = pl.DataFrame(rows).with_columns(
        group=pl.col("part").str.replace(r"^.*_(minutes|returns)$", "$1")
    )
    print(table.group_by("group").agg(pl.col("mse_share").sum()).sort("mse_share"))
    return table.sort("mse_share", descending=True)


def segments(rows: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Absolute error by pre-deadline player segment."""
    context = features.select(
        "season", "gameweek", "player_code", "gws_in_history", "gws_since_appearance",
        "price", "is_promoted",
    ) if "is_promoted" in features.columns else features.select(
        "season", "gameweek", "player_code", "gws_in_history", "gws_since_appearance", "price",
    )  # fmt: skip
    per_gw = rows.group_by("season", "gameweek", "player_code", "position").agg(
        pl.col("xp", "total_points").sum()
    )
    frame = per_gw.join(
        context.unique(["season", "gameweek", "player_code"]),
        on=["season", "gameweek", "player_code"],
        how="left",
    ).with_columns(abs_error=(pl.col("xp") - pl.col("total_points")).abs())
    total = float(frame["abs_error"].sum())
    out = []
    flags = {
        "new (<6 GWs of history)": pl.col("gws_in_history") < 6,
        "back from absence (3+ GWs out)": pl.col("gws_since_appearance") >= 3,
        "premium (price >= 9.0m)": pl.col("price") >= 90,
        "high prediction (xp >= 5)": pl.col("xp") >= 5,
        **{f"position {p}": pl.col("position") == p for p in player_features.POSITIONS},
    }
    for name, flag in flags.items():
        seg = frame.filter(flag.fill_null(False))
        if seg.is_empty():
            continue
        out.append({
            "segment": name, "rows_share": seg.height / frame.height,
            "error_share": float(seg["abs_error"].sum()) / total,
            "mae": float(seg["abs_error"].mean()),  # type: ignore[arg-type]
            "bias": float((seg["xp"] - seg["total_points"]).mean()),  # type: ignore[arg-type]
        })  # fmt: skip
    return pl.DataFrame(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matches = team_strength.load_matches(DEFAULT_WAREHOUSE)
    features = player_features.build_features(
        player_features.load_fixture_rows(DEFAULT_WAREHOUSE),
        player_features.match_lambdas(matches),
        player_features.schedule_features(matches),
    )
    gameweeks = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    cache = OUT / "rows.parquet"
    if not cache.exists():
        record(features, gameweeks).write_parquet(cache)
    rows = pl.read_parquet(cache)
    keep = regulars(rows.select("season", "gameweek", "player_code").unique(), gameweeks).select(
        "season", "gameweek", "player_code"
    )
    rows = rows.join(keep, on=["season", "gameweek", "player_code"])
    with pl.Config(tbl_rows=40, float_precision=3, tbl_width_chars=200):
        print(decompose(components(rows, matches)))
        print(segments(rows, features))


if __name__ == "__main__":
    main()
