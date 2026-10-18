"""Expected price moves by the next deadline (E10).

FPL changes prices overnight from net transfers. At a deadline, the gameweek's net
transfers so far (as a share of all transfer activity, `crowd_net_transfer_share`) are
known, and they persist: players bought by more than 0.5% of all activity rise by the
next deadline 40% of the time, heavily sold ones fall 31% of the time.

P(rise) and P(fall) are isotonic in that share (more buying never makes a rise less
likely), fitted per season on earlier seasons only. The value of holding a player
through the gameweek is in tenths of selling value: FPL keeps half of a rise (rounded
down) and all of a fall, so

    move = 0.5 * P(rise) - P(fall)
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

KEY = ["season", "gameweek", "player_code"]


def price_changes(features: pl.DataFrame) -> pl.DataFrame:
    """Per player-gameweek: net transfer share at the deadline and the price change by
    his next gameweek of the same season (tenths; null at a season's end)."""
    per_gw = (
        features.group_by(KEY)
        .agg(pl.col("price").first(), pl.col("crowd_net_transfer_share").first())
        .sort("player_code", "season", "gameweek")
    )
    return per_gw.with_columns(
        change=pl.col("price").shift(-1).over("player_code", "season") - pl.col("price")
    )


def fit(train: pl.DataFrame) -> tuple[IsotonicRegression, IsotonicRegression]:
    labelled = train.drop_nulls(["change", "crowd_net_transfer_share"])
    share = labelled["crowd_net_transfer_share"].to_numpy()
    change = labelled["change"].to_numpy()
    rise = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(share, change > 0)
    fall = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(share, change < 0)
    return rise, fall


def expected_moves(changes: pl.DataFrame, seasons: list[str]) -> pl.DataFrame:
    """Walk-forward: each season's moves from a fit on all earlier seasons."""
    frames = []
    for season in seasons:
        rise, fall = fit(changes.filter(pl.col("season") < season))
        target = changes.filter(pl.col("season") == season).drop_nulls("crowd_net_transfer_share")
        share = target["crowd_net_transfer_share"].to_numpy()
        p_rise, p_fall = rise.predict(share), fall.predict(share)
        frames.append(
            target.select(KEY).with_columns(
                p_rise=pl.Series(p_rise),
                p_fall=pl.Series(p_fall),
                move=pl.Series(0.5 * p_rise - p_fall),
            )
        )
    return pl.concat(frames)


def by_gameweek(moves: pl.DataFrame, season: str) -> dict[int, dict[int, float]]:
    """gameweek -> {player_code: move} for one season, as `season_sim.play` takes it."""
    out: dict[int, dict[int, float]] = {}
    for gameweek, player, move in (
        moves.filter(pl.col("season") == season).select("gameweek", "player_code", "move")
    ).iter_rows():
        out.setdefault(gameweek, {})[player] = move
    return out


def main() -> None:
    """Calibration report and the moves table the season simulator reads."""
    from fpl_models import experiments  # noqa: PLC0415 - heavy, CLI only
    from fpl_models.data import DEFAULT_WAREHOUSE  # noqa: PLC0415

    changes = price_changes(experiments.load_features(DEFAULT_WAREHOUSE))
    seasons = sorted(s for s in changes["season"].unique() if s >= "2018-19")
    moves = expected_moves(changes, seasons)
    scored = moves.join(changes, on=KEY).drop_nulls("change")
    rise, fall = scored["change"].to_numpy() > 0, scored["change"].to_numpy() < 0
    for name, p, y in (("rise", scored["p_rise"], rise), ("fall", scored["p_fall"], fall)):
        bins = np.quantile(p, [0, 0.9, 0.99, 0.999, 1])
        for lo, hi in pairwise(bins):
            mask = (p.to_numpy() >= lo) & (p.to_numpy() <= hi)
            print(f"P({name}) in [{lo:.2f}, {hi:.2f}]: predicted {p.to_numpy()[mask].mean():.3f}"
                  f" actual {y[mask].mean():.3f} (n={mask.sum()})")  # fmt: skip
    out = Path("data/experiments/e10")
    out.mkdir(parents=True, exist_ok=True)
    moves.write_parquet(out / "moves.parquet")
    print(f"wrote {moves.height} rows for {seasons}")


if __name__ == "__main__":
    main()
