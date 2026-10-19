"""E13 analysis: within-week price-change predictability and the value of timing.

1. Calibration: P(rise) / P(fall) within the next ~24 h from the transfer window so far
   (net transfers as a share of all activity, and per point of ownership), fitted on
   earlier seasons only (2024-25 from 2023-24; 2025-26 from both).
2. Timing: the transfers v2's held-out season-simulation paths actually made, executed
   (a) at the deadline (what the simulator does), (b) at the start of the window,
   (c) at the first capture where the predictor says the incoming player is likely to
   rise or the outgoing one to fall (else at the deadline), and (d) at the best capture
   in hindsight (an upper bound). Gains are in selling value at the deadline: buying
   before a rise keeps half of it (FPL's selling rule), selling before a fall saves all
   of it. The *decisions* are the deadline's, so this isolates execution timing.
"""

import glob
import json
import re
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from fpl_models.e13 import SILVER

PATHS = Path("data/experiments/e1/paths")
SEASONS = ("2023-24", "2024-25", "2025-26")
FEATURES = ("share", "per_owner")
RISE_MONOTONE = [1, 1]
PAIR_HOURS = (12, 36)  # next capture within this range (a price-change night in between)


def with_features(captures: pl.DataFrame) -> pl.DataFrame:
    """Window-invariant transfer pressure at each capture."""
    return captures.with_columns(
        share=pl.col("net_transfers")
        / pl.col("net_transfers").abs().sum().over("captured_at").clip(lower_bound=1),
        per_owner=pl.col("net_transfers")
        / (pl.col("selected_by_percent").clip(lower_bound=0.1) * 1e4),
    )


def pairs(captures: pl.DataFrame) -> pl.DataFrame:
    """Each capture with the player's next capture in the same transfer window."""
    ordered = captures.sort("player_code", "captured_at")
    nxt = [pl.col(c).shift(-1).over("player_code") for c in ("captured_at", "price")]
    nxt_gw = pl.col("next_gameweek").shift(-1).over("player_code")
    return (
        ordered.with_columns(
            nxt[0].alias("next_at"), nxt[1].alias("next_price"), nxt_gw.alias("next_gw")
        )
        .with_columns(
            hours=(pl.col("next_at") - pl.col("captured_at")).dt.total_seconds() / 3600,
            change=pl.col("next_price") - pl.col("price"),
        )
        .filter(
            (pl.col("next_gw") == pl.col("next_gameweek")) & pl.col("hours").is_between(*PAIR_HOURS)
        )
    )


def fit(
    train: pl.DataFrame, *, monotone: bool = True
) -> tuple[HistGradientBoostingClassifier, ...]:
    """Rise and fall classifiers; `monotone` constrains more buying to never lower
    P(rise) (and more selling P(fall))."""
    x = train.select(FEATURES).to_numpy()
    up = RISE_MONOTONE if monotone else None
    down = [-c for c in RISE_MONOTONE] if monotone else None
    rise = HistGradientBoostingClassifier(max_iter=200, monotonic_cst=up, random_state=0)
    fall = HistGradientBoostingClassifier(max_iter=200, monotonic_cst=down, random_state=0)
    return (
        rise.fit(x, (train["change"] > 0).to_numpy()),
        fall.fit(x, (train["change"] < 0).to_numpy()),
    )


def predict(
    models: tuple[HistGradientBoostingClassifier, ...], frame: pl.DataFrame
) -> pl.DataFrame:
    x = frame.select(FEATURES).to_numpy()
    rise, fall = models
    return frame.with_columns(
        p_rise=pl.Series(rise.predict_proba(x)[:, 1]),
        p_fall=pl.Series(fall.predict_proba(x)[:, 1]),
    )


def brier(scored: pl.DataFrame) -> float:
    rise = np.mean((scored["p_rise"].to_numpy() - (scored["change"] > 0).to_numpy()) ** 2)
    fall = np.mean((scored["p_fall"].to_numpy() - (scored["change"] < 0).to_numpy()) ** 2)
    return float(rise + fall)


def calibration(scored: pl.DataFrame) -> None:
    for name, label in (("rise", scored["change"] > 0), ("fall", scored["change"] < 0)):
        p = scored[f"p_{name}"].to_numpy()
        y = label.to_numpy()
        brier = float(np.mean((p - y) ** 2))
        base = float(np.mean((y.mean() - y) ** 2))
        print(f"P({name}) base rate {y.mean():.4f}, Brier {brier:.5f} vs {base:.5f} "
              f"(skill {1 - brier / base:.2f})")  # fmt: skip
        for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
            m = (p >= lo) & (p < hi)
            if m.sum():
                print(f"  [{lo:.2f}, {hi:.2f}) n={m.sum():7d} predicted {p[m].mean():.3f} "
                      f"actual {y[m].mean():.3f}")  # fmt: skip


def path_transfers(label: str = "v2fix") -> pl.DataFrame:
    """(season, seed, gameweek, player_in, player_out) from the simulated paths; a
    transfer's legs are paired in order, which is all the timing rules need."""
    rows = []
    for file in glob.glob(str(PATHS / f"{label}_*.json")):
        match = re.match(rf".*/{label}_(\d{{4}}-\d{{2}})_(\d+)\.json", file)
        if not match or match.group(1) not in SEASONS:
            continue
        season, seed = match.group(1), int(match.group(2))
        records = json.loads(Path(file).read_text())
        for before, after in pairwise(records):
            ins = sorted(set(after["squad"]) - set(before["squad"]))
            outs = sorted(set(before["squad"]) - set(after["squad"]))
            for p_in, p_out in zip(ins, outs, strict=True):
                rows.append({"season": season, "seed": seed, "gameweek": after["gameweek"],
                             "player_in": p_in, "player_out": p_out})  # fmt: skip
    return pl.DataFrame(rows)


def buy_gain(paid: int, at_deadline: int) -> int:
    """Selling value at the deadline minus cost, relative to buying at the deadline."""
    rise = at_deadline - paid
    return rise // 2 if rise > 0 else rise


def sell_gain(sold: int, at_deadline: int) -> int:
    """Selling earlier: a later fall is avoided in full; a later rise is forgone by half
    (assuming he's worth more than he was bought for)."""
    change = at_deadline - sold
    return -change if change < 0 else -(change // 2)


def timing_value(
    scored: pl.DataFrame, transfers: pl.DataFrame, threshold: float = 0.5
) -> pl.DataFrame:
    """Per transfer: gains (tenths) of rules b, c, d over executing at the deadline."""
    window: dict[tuple[str, int], dict[int, list[tuple[float, int, float, float, str]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for season, gw, code, at, price, p_rise, p_fall, status in scored.select(
        "season",
        "next_gameweek",
        "player_code",
        "captured_at",
        "price",
        "p_rise",
        "p_fall",
        "status",
    ).iter_rows():
        window[(season, gw)][code].append((at.timestamp(), price, p_rise, p_fall, status))
    out = []
    for season, seed, gw, p_in, p_out in transfers.iter_rows():
        caps_in = sorted(window[(season, gw)].get(p_in, []))
        caps_out = sorted(window[(season, gw)].get(p_out, []))
        times = sorted({c[0] for c in caps_in} & {c[0] for c in caps_out})
        if len(times) < 2:
            continue
        cin = {c[0]: c for c in caps_in}
        cout = {c[0]: c for c in caps_out}
        deadline = times[-1]

        def gain(t: float, cin: dict = cin, cout: dict = cout, end: float = deadline) -> int:  # type: ignore[type-arg]
            return buy_gain(cin[t][1], cin[end][1]) + sell_gain(cout[t][1], cout[end][1])

        rule_c = next(
            (t for t in times if cin[t][2] > threshold or cout[t][3] > threshold), deadline
        )
        out.append({
            "season": season, "seed": seed, "gameweek": gw,
            "b_window_start": gain(times[0]), "c_predictor": gain(rule_c),
            "d_hindsight": max(gain(t) for t in times),
            "c_early": rule_c != deadline,
            # News risk of executing early: available then, flagged by the deadline.
            "c_news_risk": rule_c != deadline and cin[rule_c][4] == "a" and cin[deadline][4] != "a",
            "b_news_risk": cin[times[0]][4] == "a" and cin[deadline][4] != "a",
        })  # fmt: skip
    return pl.DataFrame(out)


def main() -> None:
    captures = with_features(pl.read_parquet(SILVER))
    labelled = pairs(captures)
    print(f"{captures['captured_at'].n_unique()} captures, {labelled.height} labelled pairs")
    rates = labelled.group_by("season").agg(
        pl.len(),
        (pl.col("change") > 0).mean().alias("rise"),
        (pl.col("change") < 0).mean().alias("fall"),
    )
    print(rates.sort("season"))
    # Model choice on 2024-25 (fit on 2023-24); 2025-26 stays an untouched test.
    first, validation = (labelled.filter(pl.col("season") == s) for s in SEASONS[:2])
    scores = {
        monotone: brier(predict(fit(first, monotone=monotone), validation))
        for monotone in (True, False)
    }
    monotone = min(scores, key=scores.__getitem__)
    print(f"validation Brier (rise + fall): {scores}; monotone={monotone}")
    scored_frames, all_scored = {}, []
    for season in SEASONS:
        earlier = labelled.filter(pl.col("season") < season)
        target = captures.filter(pl.col("season") == season)
        # The first season has no earlier data: fit on itself (in-sample, flagged).
        models = fit(earlier if earlier.height else labelled.filter(pl.col("season") == season),
                     monotone=monotone)  # fmt: skip
        all_scored.append(predict(models, target))
        if earlier.height:
            scored_frames[season] = predict(models, labelled.filter(pl.col("season") == season))
    for season, frame in scored_frames.items():
        print(f"\nWalk-forward calibration {season} (next ~24 h):")
        calibration(frame)

    scored = pl.concat(all_scored)
    transfers = path_transfers()
    print(f"\n{transfers.height} transfers on {transfers.select('season', 'seed').n_unique()} "
          "v2 held-out paths")  # fmt: skip
    value = timing_value(scored, transfers)
    per_path = value.group_by("season", "seed").agg(
        pl.col("b_window_start", "c_predictor", "d_hindsight").sum(),
        pl.len().alias("transfers"),
        pl.col("c_early").mean().alias("c_early_share"),
        pl.col("c_news_risk").sum(),
        pl.col("b_news_risk").sum(),
    )
    summary = (
        per_path.group_by("season")
        .agg(
            pl.col("transfers").mean(),
            (pl.col("b_window_start", "c_predictor", "d_hindsight") / 10).mean().name.suffix("_m"),
            pl.col("c_early_share").mean(),
            pl.col("c_news_risk", "b_news_risk").mean(),
        )
        .sort("season")
    )
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        print("\nSelling-value gain per season vs executing at the deadline (£m):")
        print(summary)
        for col in ("b_window_start", "c_predictor", "d_hindsight"):
            v = per_path[col].to_numpy() / 10
            boot = np.random.default_rng(0).choice(v, (5000, v.size)).mean(1)
            print(f"{col}: {v.mean():+.2f}m per season (95% CI {np.quantile(boot, 0.025):+.2f}, "
                  f"{np.quantile(boot, 0.975):+.2f})")  # fmt: skip
