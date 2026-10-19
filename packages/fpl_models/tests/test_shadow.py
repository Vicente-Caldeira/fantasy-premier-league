import polars as pl
import pytest

from fpl_models import optimizer, shadow

SHAPE = {"GKP": 3, "DEF": 7, "MID": 7, "FWD": 5}


def players():
    """A pool where the first squad-shape players per position are cheap starters."""
    rows, code = [], 0
    for position, n in SHAPE.items():
        for i in range(n):
            rows.append({"player_code": code, "position": position,
                         "team_key": f"club{code % 8}", "price": 50 + 5 * i})  # fmt: skip
            code += 1
    return pl.DataFrame(rows)


def starting_squad(pool):
    squad = []
    for position, n in optimizer.SQUAD_SHAPE.items():
        squad += pool.filter(pl.col("position") == position).head(n)["player_code"].to_list()
    return sorted(squad)


def state(pool, **kwargs):
    squad = starting_squad(pool)
    price = dict(zip(pool["player_code"], pool["price"], strict=True))
    defaults = {"season": "2026-27", "squad": squad, "paid": {p: price[p] for p in squad},
                "bank": 0, "free_transfers": 1}  # fmt: skip
    return shadow.State(**(defaults | kwargs))


def xp(pool, boost=None, gameweeks=(8,)):
    rows = [{"player_code": p, "gameweek": g, "expected_points": 2.0 + (boost or {}).get(p, 0.0)}
            for p in pool["player_code"] for g in gameweeks]  # fmt: skip
    return pl.DataFrame(rows)


def test_plan_makes_the_obvious_free_transfer_and_replanning_replaces_it():
    pool = players()
    start = state(pool, bank=100)
    best = int(pool.filter(pl.col("position") == "FWD")["player_code"][-1])  # not owned
    planned = shadow.plan_next(start, pool, xp(pool, {best: 6.0}))
    assert planned.pending.transfers_in == [best]
    assert planned.pending.hits == 0
    assert planned.pending.captain == best
    assert planned.squad == start.squad  # nothing committed before the deadline
    again = shadow.plan_next(planned, pool, xp(pool))
    assert again.pending.transfers_in == []  # the new plan replaces the old one


def test_commit_applies_transfers_scores_the_week_and_rolls_free_transfers():
    pool = players()
    start = state(pool, bank=100)
    best = int(pool.filter(pl.col("position") == "FWD")["player_code"][-1])
    planned = shadow.plan_next(start, pool, xp(pool, {best: 6.0}))
    sold = planned.pending.transfers_out[0]
    prices = dict(zip(pool["player_code"], pool["price"], strict=True)) | {best: 71}  # rose
    actual = dict.fromkeys(pool["player_code"], 2) | {best: 10}
    minutes = dict.fromkeys(pool["player_code"], 90)
    position = dict(zip(pool["player_code"], pool["position"], strict=True))
    done = shadow.commit(planned, prices, actual, minutes, position)
    assert best in done.squad
    assert sold not in done.squad
    assert done.paid[best] == 71
    assert done.bank == 100 + start.paid[sold] - 71
    assert done.free_transfers == 1  # 1 - 1 used + 1
    assert done.history[-1].points == 10 * 2 + 2 * 10  # captain doubled, ten others
    assert done.pending is None
    with pytest.raises(ValueError, match="no pending"):
        shadow.commit(done, prices, actual, minutes, position)


def test_from_path_replays_bank_purchase_prices_and_free_transfers():
    pool = players()
    squad = starting_squad(pool)
    spare = int(pool.filter(pl.col("position") == "FWD")["player_code"][-1])
    out = max(p for p in squad if p < spare and pool["position"][p] == "FWD")
    price1 = dict(zip(pool["player_code"], pool["price"], strict=True))
    price2 = price1 | {out: price1[out] + 2}  # rose twice: sell keeps 1
    later = sorted([*(p for p in squad if p != out), spare])
    records = [
        {"gameweek": 1, "squad": squad, "lineup": squad[:11], "bench": squad[11:],
         "captain": squad[0], "vice_captain": squad[1], "hits": 0, "points": 50},
        {"gameweek": 2, "squad": later, "lineup": later[:11], "bench": later[11:],
         "captain": later[0], "vice_captain": later[1], "hits": 0, "points": 60},
    ]  # fmt: skip
    result = shadow.from_path("2026-27", records, {1: price1, 2: price2})
    bank1 = optimizer.BUDGET - sum(price1[p] for p in squad)
    assert result.bank == bank1 + price1[out] + 1 - price2[spare]
    assert result.paid[spare] == price2[spare]
    assert result.free_transfers == 1  # GW2: 1 available, 1 used, +1
    assert result.total == 110
    assert result.history[-1].transfers_in == [spare]
    assert shadow.from_json(shadow.to_json(result)) == result
