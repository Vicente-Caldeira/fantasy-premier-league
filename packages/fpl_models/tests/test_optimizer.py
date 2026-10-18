from collections import Counter

import numpy as np
import polars as pl
import pytest

from fpl_models import optimizer as opt

PER_POSITION = {"GKP": 4, "DEF": 10, "MID": 10, "FWD": 6}


def pool(seed=0, clubs=10):
    """Synthetic players: code, position, club, price; xP roughly rises with price."""
    rng = np.random.default_rng(seed)
    rows, code = [], 0
    for position, n in PER_POSITION.items():
        for i in range(n):
            # Like real FPL, each position has cheap enablers, so a valid 100.0m squad
            # always exists whatever the seed (15 x <= 5.0m = 75.0m).
            cheap = i < opt.SQUAD_SHAPE[position]
            price = int(rng.integers(40, 51) if cheap else rng.integers(40, 120))
            rows.append({"player_code": code, "position": position,
                         "team_key": f"club{code % clubs}", "price": price,
                         "base": price / 20 + rng.normal(0, 0.5)})  # fmt: skip
            code += 1
    return pl.DataFrame(rows)


def xp(players, gameweeks=(1,), override=None):
    rows = [
        {"player_code": p, "gameweek": gw, "expected_points": max(b, 0.0)}
        for p, b in zip(players["player_code"], players["base"], strict=True)
        for gw in gameweeks
    ]
    frame = pl.DataFrame(rows)
    for (p, gw), value in (override or {}).items():
        frame = frame.with_columns(
            pl.when((pl.col("player_code") == p) & (pl.col("gameweek") == gw))
            .then(value)
            .otherwise(pl.col("expected_points"))
            .alias("expected_points")
        )
    return frame


def solve(players, points, **kwargs):
    return opt.optimise(players.drop("base"), points, **kwargs)


def cheapest_valid_squad(players):
    """15 cheapest by position, respecting the club limit."""
    squad, clubs = [], Counter()
    for position, n in opt.SQUAD_SHAPE.items():
        group = players.filter(pl.col("position") == position).sort("price")
        for r in group.iter_rows(named=True):
            if len([p for p in squad if p[1] == position]) == n:
                break
            if clubs[r["team_key"]] < opt.MAX_PER_CLUB:
                squad.append((r["player_code"], position))
                clubs[r["team_key"]] += 1
    return [code for code, _ in squad]


# --- squad selection ------------------------------------------------------------------


def test_from_scratch_squad_follows_every_rule():
    players = pool()
    plan = solve(players, xp(players))
    gw = plan.gameweeks[0]
    info = {r["player_code"]: r for r in players.iter_rows(named=True)}
    assert Counter(info[p]["position"] for p in gw.squad) == opt.SQUAD_SHAPE
    assert sum(info[p]["price"] for p in gw.squad) <= opt.BUDGET
    assert max(Counter(info[p]["team_key"] for p in gw.squad).values()) <= opt.MAX_PER_CLUB
    assert plan.status == "optimal"


def test_captain_is_the_best_starter():
    players = pool()
    points = xp(players)
    gw = solve(players, points).gameweeks[0]
    by_player = dict(zip(points["player_code"], points["expected_points"], strict=True))
    assert gw.captain == max(gw.lineup, key=by_player.get)


def test_club_limit_binds_when_one_club_is_dominant():
    # Two players per position in club "stars", all far better than everyone else:
    # the solver would take all eight, but only three are allowed.
    players = pool()
    stars = players.group_by("position").agg(pl.col("player_code").head(2)).explode("player_code")
    players = players.with_columns(
        pl.when(pl.col("player_code").is_in(stars["player_code"].implode()))
        .then(pl.lit("stars"))
        .otherwise(pl.col("team_key"))
        .alias("team_key"),
        pl.when(pl.col("player_code").is_in(stars["player_code"].implode()))
        .then(pl.col("base") + 10)
        .otherwise(pl.col("base"))
        .alias("base"),
    )
    gw = solve(players, xp(players)).gameweeks[0]
    clubs = Counter(players.filter(pl.col("player_code").is_in(gw.squad))["team_key"])
    assert clubs["stars"] == opt.MAX_PER_CLUB


def test_lineup_respects_formation_even_when_defenders_dominate():
    players = pool().with_columns(
        pl.when(pl.col("position") == "DEF")
        .then(pl.col("base") + 5)
        .otherwise(pl.col("base"))
        .alias("base")
    )
    gw = solve(players, xp(players)).gameweeks[0]
    starters = Counter(players.filter(pl.col("player_code").is_in(gw.lineup))["position"])
    assert starters["DEF"] == 5
    assert starters["MID"] >= 2
    assert starters["FWD"] >= 1
    assert starters["GKP"] == 1


def test_pruning_does_not_change_the_optimum():
    players = pool(seed=3)
    points = xp(players, gameweeks=(1, 2, 3))
    pruned = solve(players, points)
    full = solve(players, points, prune=False)
    assert pruned.objective == pytest.approx(full.objective)


# --- transfers ------------------------------------------------------------------------


def upgrade_setup(gain):
    """Current squad = cheapest valid one; one bench-quality midfielder can be upgraded."""
    players = pool(seed=1)
    squad = cheapest_valid_squad(players)
    weakest_mid = next(
        p for p in squad if players.filter(pl.col("player_code") == p)["position"][0] == "MID"
    )
    star = players.filter((pl.col("position") == "MID") & ~pl.col("player_code").is_in(squad)).sort(
        "price"
    )["player_code"][0]
    flat = players.with_columns(base=pl.lit(1.0))
    points = xp(flat, override={(star, 1): 1.0 + gain})
    return flat, points, squad, weakest_mid, star


def test_a_transfer_is_free_with_a_free_transfer():
    players, points, squad, _, star = upgrade_setup(gain=1.0)
    gw = solve(players, points, current_squad=squad, bank=500, free_transfers=1).gameweeks[0]
    assert star in gw.squad
    assert gw.hits == 0


def test_a_hit_is_taken_only_when_it_pays():
    players, _, squad, _, star = upgrade_setup(gain=0.0)
    worth_it = solve(
        players,
        xp(players, override={(star, 1): 9.0}),
        current_squad=squad,
        bank=500,
        free_transfers=0,
    )
    not_worth = solve(
        players,
        xp(players, override={(star, 1): 2.5}),
        current_squad=squad,
        bank=500,
        free_transfers=0,
    )
    assert star in worth_it.gameweeks[0].squad
    assert worth_it.gameweeks[0].hits == 1
    assert star not in not_worth.gameweeks[0].squad
    assert not_worth.gameweeks[0].hits == 0


def test_free_transfers_are_used_before_any_hit():
    # Regression: the solver once paid hits while holding free transfers, to bank them.
    players = pool(seed=2)
    squad = cheapest_valid_squad(players)
    plan = solve(
        players, xp(players, gameweeks=(1, 2)), current_squad=squad, bank=0, free_transfers=5
    )
    for gw in plan.gameweeks:
        assert gw.hits == max(len(gw.transfers_in) - gw.free_transfers, 0)


def test_free_transfers_are_banked_for_a_better_week():
    players, _, squad, _, star = upgrade_setup(gain=0.0)
    second = players.filter(
        (pl.col("position") == "DEF") & ~pl.col("player_code").is_in(squad)
    ).sort("price")["player_code"][0]
    # Both targets are worthless in GW1 (buying early would cost points), great in GW2:
    # the only way to get both without a hit is to bank GW1's free transfer.
    override = {(star, 1): 0.0, (second, 1): 0.0, (star, 2): 6.0, (second, 2): 6.0}
    points = xp(players, gameweeks=(1, 2), override=override)
    plan = solve(
        players,
        points,
        current_squad=squad,
        bank=500,
        free_transfers=1,
        settings=opt.Settings(decay=1.0),
    )
    first, then = plan.gameweeks
    assert first.transfers_in == []
    assert then.free_transfers == 2
    assert {star, second} <= set(then.squad)
    assert then.hits == 0


def test_bank_never_goes_negative():
    players = pool(seed=4)
    squad = cheapest_valid_squad(players)
    plan = solve(
        players, xp(players, gameweeks=(1, 2)), current_squad=squad, bank=30, free_transfers=2
    )
    assert min(plan.bank) >= 0


# --- validator ------------------------------------------------------------------------


def test_validator_rejects_an_invalid_plan():
    players = pool()
    plan = solve(players, xp(players))
    gw = plan.gameweeks[0]
    broken = opt.Plan(
        gameweeks=[opt.Gameweek(**{**gw.__dict__, "lineup": gw.lineup[:10]})],
        objective=plan.objective, status=plan.status, bank=plan.bank,
    )  # fmt: skip
    with pytest.raises(opt.RuleViolationError, match="lineup"):
        opt.validate(
            broken, players, current_squad=None, bank=0, free_transfers=1, selling_prices={}
        )


def test_kept_and_banned_players_are_respected_every_gameweek():
    players = pool(seed=5)
    points = xp(players, gameweeks=(1, 2))
    free = solve(players, points)
    favourite = free.gameweeks[0].squad[0]
    outsider = players.filter(~pl.col("player_code").is_in(free.gameweeks[0].squad))["player_code"][
        0
    ]
    plan = solve(players, points, keep=[outsider], ban=[favourite])
    for gw in plan.gameweeks:
        assert outsider in gw.squad
        assert favourite not in gw.squad
    assert plan.objective <= free.objective


def test_keeping_and_banning_the_same_player_is_rejected():
    players = pool()
    with pytest.raises(opt.InfeasibleError, match="both kept and banned"):
        solve(players, xp(players), keep=[1], ban=[1])


def test_wildcard_week_earns_no_extra_free_transfer():
    players = pool(seed=6)
    plan = solve(players, xp(players, gameweeks=(1, 2)), free_transfers=1)
    assert plan.gameweeks[1].free_transfers == 1


def test_wildcard_chip_rebuilds_an_existing_squad_without_hits():
    players = pool(seed=7)
    squad = cheapest_valid_squad(players)
    points = xp(players, gameweeks=(1, 2))
    normal = solve(players, points, current_squad=squad, bank=250, free_transfers=1)
    chip = solve(players, points, current_squad=squad, bank=250, free_transfers=2, wildcard=True)
    first = chip.gameweeks[0]
    assert len(first.transfers_in) > 2
    assert first.hits == 0
    assert chip.gameweeks[1].free_transfers == 2  # banked transfers kept, none earned
    assert chip.gameweeks[0].expected_points > normal.gameweeks[0].expected_points
