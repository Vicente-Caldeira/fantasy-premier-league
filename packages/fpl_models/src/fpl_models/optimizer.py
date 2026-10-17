"""Squad, lineup, captain and transfer optimisation as a mixed-integer program.

Over a horizon of gameweeks, choose for each gameweek a 15-player squad, a starting
XI and a captain to maximise discounted expected points, subject to FPL's rules:

    squad      2 GKP, 5 DEF, 5 MID, 3 FWD; at most 3 per club; within budget
    lineup     11 starters: 1 GKP, >= 3 DEF, >= 2 MID, >= 1 FWD; captain scores double
    transfers  squad changes between gameweeks; free transfers bank up to 5 (+1 per
               gameweek, none in a wildcard week); each transfer beyond the free ones
               costs 4 points

Free-transfer banking is `ft' = min(5, ft - free_used + 1)`. Having more free transfers
is never worse, so upper bounds alone reproduce the min at the optimum. Hits are exact:
`paid = max(transfers - ft, 0)`, one binary per gameweek, because FPL spends free
transfers first (a solver may otherwise "buy" a hit to keep free transfers banked).

Prices are assumed constant over the horizon. FPL sells at purchase price plus half
of any rise; purchase prices aren't public, so selling prices default to current
prices unless given.

Every solution is checked against the rules by `validate`, independently of the
model's constraints.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import polars as pl
from ortools.linear_solver import pywraplp

SQUAD_SHAPE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
LINEUP_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
LINEUP_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15
LINEUP_SIZE = 11
MAX_PER_CLUB = 3
MAX_FREE_TRANSFERS = 5
HIT_COST = 4
BUDGET = 1000  # tenths of a million
# A transfer can be free *and* pointless (e.g. with 5 banked free transfers, using one
# costs nothing next week). A tiny cost per transfer makes the solver never churn.
TRANSFER_TIEBREAK = 1e-3


@dataclass(frozen=True)
class Settings:
    bench_weight: float = 0.1  # bench players still matter (injuries, auto-subs)
    decay: float = 0.9  # later gameweeks are less certain
    time_limit_seconds: int = 60


@dataclass(frozen=True)
class Gameweek:
    gameweek: int
    squad: list[int]
    lineup: list[int]
    captain: int
    vice_captain: int
    transfers_in: list[int]
    transfers_out: list[int]
    free_transfers: int  # available at this gameweek's deadline
    hits: int
    expected_points: float  # lineup + captain bonus, before hits


@dataclass(frozen=True)
class Plan:
    gameweeks: list[Gameweek]
    objective: float
    status: str
    bank: list[int] = field(default_factory=list)  # tenths, after each gameweek's transfers

    @property
    def total_expected_points(self) -> float:
        return sum(g.expected_points - HIT_COST * g.hits for g in self.gameweeks)


def candidate_pool(
    players: pl.DataFrame,
    expected_points: pl.DataFrame,
    keep: Sequence[int] = (),
    margin: int = 3,
) -> pl.DataFrame:
    """Drop players who are very unlikely to be in an optimal squad.

    A player is dominated by another of his position who costs no more and expects at
    least as many points in *every* gameweek of the horizon (lineup, captain and bench
    choices are per gameweek, so dominance on the total alone would not be safe). With
    at least `margin` x that position's squad slots dominating him, any squad using him
    can almost surely swap him for one without losing points or money; the margin
    covers club limits blocking some swaps. This is a heuristic, not a proof: the
    pruned and full problems are checked to agree in tests and on live data.
    `keep` (e.g. the current squad) is never dropped.
    """
    wide = expected_points.pivot(on="gameweek", index="player_code", values="expected_points")
    weeks = [c for c in wide.columns if c != "player_code"]
    pool = players.join(wide, on="player_code", how="left").with_columns(
        pl.col(weeks).fill_null(0.0)
    )
    kept: list[int] = []
    for position, slots in SQUAD_SHAPE.items():
        group = pool.filter(pl.col("position") == position)
        prices = group["price"].to_numpy()
        points = group.select(weeks).to_numpy()
        codes = group["player_code"].to_list()
        for i, code in enumerate(codes):
            dominators = (prices <= prices[i]) & (points >= points[i]).all(axis=1)
            dominators[i] = False
            if dominators.sum() < margin * slots or code in keep:
                kept.append(code)
    return players.filter(pl.col("player_code").is_in(kept))


class InfeasibleError(RuntimeError):
    """No squad satisfies the rules with the given players, budget and constraints."""


def optimise(  # noqa: PLR0912, PLR0913, PLR0915 - one MIP, kept in one place
    players: pl.DataFrame,
    expected_points: pl.DataFrame,
    *,
    current_squad: Sequence[int] | None = None,
    bank: int = 0,
    free_transfers: int = 1,
    selling_prices: Mapping[int, int] | None = None,
    settings: Settings = Settings(),  # noqa: B008 - frozen dataclass
    prune: bool = True,
    keep: Sequence[int] = (),
    ban: Sequence[int] = (),
) -> Plan:
    """Optimal plan over the gameweeks in `expected_points`.

    players:          player_code, position, team_key, price (tenths)
    expected_points:  player_code, gameweek, expected_points
    current_squad:    15 player codes; None builds a squad from scratch (wildcard),
                      in which case `bank` is ignored and the budget is 100.0m.
    prune:            drop dominated players first (see `candidate_pool`).
    keep / ban:       players that must be in / out of the squad every gameweek.
    """
    if set(keep) & set(ban):
        raise InfeasibleError(f"players both kept and banned: {sorted(set(keep) & set(ban))}")
    if prune:
        protected = [*(current_squad or ()), *keep]
        players = candidate_pool(players, expected_points, keep=protected)
    gameweeks = sorted(expected_points["gameweek"].unique().to_list())
    xp = {
        (r["player_code"], r["gameweek"]): r["expected_points"]
        for r in expected_points.iter_rows(named=True)
    }
    info = {r["player_code"]: r for r in players.iter_rows(named=True)}
    codes = sorted(info)
    if current_squad is not None:
        missing = set(current_squad) - set(codes)
        if missing:
            raise InfeasibleError(f"current squad players not in the pool: {sorted(missing)}")
    sell = {p: (selling_prices or {}).get(p, info[p]["price"]) for p in codes}
    owned = set(current_squad or [])

    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SetTimeLimit(settings.time_limit_seconds * 1000)
    T = range(len(gameweeks))  # noqa: N806 - conventional index name
    squad = {(p, t): solver.BoolVar(f"s_{p}_{t}") for p in codes for t in T}
    lineup = {(p, t): solver.BoolVar(f"l_{p}_{t}") for p in codes for t in T}
    captain = {(p, t): solver.BoolVar(f"c_{p}_{t}") for p in codes for t in T}
    bought = {(p, t): solver.BoolVar(f"in_{p}_{t}") for p in codes for t in T}
    sold = {(p, t): solver.BoolVar(f"out_{p}_{t}") for p in codes for t in T}
    ft = [solver.IntVar(0, MAX_FREE_TRANSFERS, f"ft_{t}") for t in T]
    paid = [solver.IntVar(0, SQUAD_SIZE, f"paid_{t}") for t in T]
    money = [solver.IntVar(0, 10_000, f"bank_{t}") for t in T]
    from_scratch = current_squad is None

    for t in T:
        for p in codes:
            solver.Add(lineup[p, t] <= squad[p, t])
            solver.Add(captain[p, t] <= lineup[p, t])
            before = (1 if p in owned else 0) if t == 0 else squad[p, t - 1]
            solver.Add(squad[p, t] - before == bought[p, t] - sold[p, t])
            solver.Add(bought[p, t] + sold[p, t] <= 1)  # no buying and selling the same player
        for position, count in SQUAD_SHAPE.items():
            members = [p for p in codes if info[p]["position"] == position]
            solver.Add(solver.Sum(squad[p, t] for p in members) == count)
            starters = solver.Sum(lineup[p, t] for p in members)
            solver.Add(starters >= LINEUP_MIN[position])
            solver.Add(starters <= LINEUP_MAX[position])
        for p in keep:
            solver.Add(squad[p, t] == 1)
        for p in ban:
            if p in info:
                solver.Add(squad[p, t] == 0)
        solver.Add(solver.Sum(lineup[p, t] for p in codes) == LINEUP_SIZE)
        solver.Add(solver.Sum(captain[p, t] for p in codes) == 1)
        for club in {info[p]["team_key"] for p in codes}:
            members = [p for p in codes if info[p]["team_key"] == club]
            solver.Add(solver.Sum(squad[p, t] for p in members) <= MAX_PER_CLUB)

        # Money: bank after this gameweek's transfers must stay non-negative.
        spent = solver.Sum(info[p]["price"] * bought[p, t] for p in codes)
        received = solver.Sum(sell[p] * sold[p, t] for p in codes)
        if t == 0 and from_scratch:
            solver.Add(money[t] == BUDGET - spent)
        else:
            previous = bank if t == 0 else money[t - 1]
            solver.Add(money[t] == previous + received - spent)

        # Free transfers: the first gameweek of a from-scratch plan is a wildcard.
        transfers = solver.Sum(bought[p, t] for p in codes)
        if t == 0:
            solver.Add(ft[t] == free_transfers)
        if t == 0 and from_scratch:
            solver.Add(paid[t] == 0)
            free_used = solver.Sum([])  # wildcard: transfers don't use free transfers
        else:
            # FPL uses free transfers first: paid = max(transfers - ft, 0) exactly.
            # Without the upper bounds the solver could pay a hit while holding free
            # transfers, just to keep them banked (the validator caught this).
            free_used = transfers - paid[t]
            takes_hit = solver.BoolVar(f"hit_{t}")
            solver.Add(paid[t] >= transfers - ft[t])
            solver.Add(paid[t] <= transfers - ft[t] + 2 * SQUAD_SIZE * (1 - takes_hit))
            solver.Add(paid[t] <= SQUAD_SIZE * takes_hit)
            solver.Add(free_used >= 0)
        if t + 1 < len(gameweeks):
            # A wildcard keeps banked transfers but earns no +1 that gameweek.
            earned = 0 if t == 0 and from_scratch else 1
            solver.Add(ft[t + 1] <= ft[t] - free_used + earned)
            solver.Add(ft[t + 1] <= MAX_FREE_TRANSFERS)

    objective = (
        solver.Sum(
            settings.decay**t
            * solver.Sum(
                xp.get((p, gameweeks[t]), 0.0)
                * (
                    lineup[p, t]
                    + captain[p, t]
                    + settings.bench_weight * (squad[p, t] - lineup[p, t])
                )
                for p in codes
            )
            for t in T
        )
        - HIT_COST * solver.Sum(paid)
        - TRANSFER_TIEBREAK * solver.Sum(bought.values())
    )
    solver.Maximize(objective)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise InfeasibleError("no feasible squad for the given players and constraints")

    def chosen(var: Mapping[tuple[int, int], pywraplp.Variable], t: int) -> list[int]:
        return [p for p in codes if var[p, t].solution_value() > 0.5]

    plan = []
    # The solver's ft variables are only upper-bounded (enough for optimal decisions),
    # so report free transfers by replaying FPL's rule on the transfers actually made.
    available = free_transfers
    for t, gw in enumerate(gameweeks):
        starters = chosen(lineup, t)
        by_xp = sorted(starters, key=lambda p: xp.get((p, gw), 0.0), reverse=True)
        cap = chosen(captain, t)[0]
        vice = next(p for p in by_xp if p != cap)
        plan.append(
            Gameweek(
                gameweek=gw,
                squad=chosen(squad, t),
                lineup=starters,
                captain=cap,
                vice_captain=vice,
                transfers_in=chosen(bought, t),
                transfers_out=chosen(sold, t),
                free_transfers=available,
                hits=round(paid[t].solution_value()),
                expected_points=sum(xp.get((p, gw), 0.0) for p in starters)
                + xp.get((cap, gw), 0.0),
            )
        )
        wildcard = t == 0 and from_scratch
        used = len(plan[-1].transfers_in) - plan[-1].hits
        available = available if wildcard else min(MAX_FREE_TRANSFERS, available - used + 1)
    result = Plan(
        gameweeks=plan,
        objective=solver.Objective().Value(),
        status="optimal" if status == pywraplp.Solver.OPTIMAL else "feasible",
        bank=[round(m.solution_value()) for m in money],
    )
    validate(result, players, current_squad=current_squad, bank=bank,
             free_transfers=free_transfers, selling_prices=sell, keep=keep, ban=ban)  # fmt: skip
    return result


class RuleViolationError(AssertionError):
    """A plan breaks an FPL rule."""


def validate(  # noqa: PLR0912, PLR0913 - one rule per branch, deliberately explicit
    plan: Plan,
    players: pl.DataFrame,
    *,
    current_squad: Sequence[int] | None,
    bank: int,
    free_transfers: int,
    selling_prices: Mapping[int, int],
    keep: Sequence[int] = (),
    ban: Sequence[int] = (),
) -> None:
    """Check a plan against FPL's rules, independently of how it was produced."""
    info = {r["player_code"]: r for r in players.iter_rows(named=True)}
    squad = list(current_squad) if current_squad is not None else None
    money = bank if current_squad is not None else BUDGET
    ft = free_transfers
    for i, gw in enumerate(plan.gameweeks):
        if len(set(gw.squad)) != SQUAD_SIZE:
            raise RuleViolationError(f"GW{gw.gameweek}: squad has {len(set(gw.squad))} players")
        shape = Counter(info[p]["position"] for p in gw.squad)
        if dict(shape) != SQUAD_SHAPE:
            raise RuleViolationError(f"GW{gw.gameweek}: squad shape {dict(shape)}")
        clubs = Counter(info[p]["team_key"] for p in gw.squad)
        if max(clubs.values()) > MAX_PER_CLUB:
            raise RuleViolationError(f"GW{gw.gameweek}: more than {MAX_PER_CLUB} from one club")
        if len(gw.lineup) != LINEUP_SIZE or not set(gw.lineup) <= set(gw.squad):
            raise RuleViolationError(f"GW{gw.gameweek}: lineup is not 11 squad players")
        starters = Counter(info[p]["position"] for p in gw.lineup)
        for position in SQUAD_SHAPE:
            if not LINEUP_MIN[position] <= starters[position] <= LINEUP_MAX[position]:
                raise RuleViolationError(f"GW{gw.gameweek}: invalid formation {dict(starters)}")
        if not set(keep) <= set(gw.squad) or set(ban) & set(gw.squad):
            raise RuleViolationError(f"GW{gw.gameweek}: kept/banned players not respected")
        if gw.captain not in gw.lineup or gw.vice_captain not in gw.lineup:
            raise RuleViolationError(f"GW{gw.gameweek}: captain or vice not in the lineup")

        if gw.free_transfers != ft:
            raise RuleViolationError(
                f"GW{gw.gameweek}: reports {gw.free_transfers} free, rules say {ft}"
            )
        wildcard = i == 0 and current_squad is None
        if not wildcard:
            if squad is None:  # pragma: no cover - unreachable: set on every gameweek
                raise RuleViolationError("no previous squad")
            if set(gw.transfers_in) != set(gw.squad) - set(squad) or set(gw.transfers_out) != set(
                squad
            ) - set(gw.squad):
                raise RuleViolationError(f"GW{gw.gameweek}: transfers don't match the squad change")
            money += sum(selling_prices.get(p, info[p]["price"]) for p in gw.transfers_out)
            money -= sum(info[p]["price"] for p in gw.transfers_in)
            used = len(gw.transfers_in)
            if gw.hits != max(used - ft, 0):
                raise RuleViolationError(
                    f"GW{gw.gameweek}: {used} transfers with {ft} free -> hits {gw.hits}"
                )
            ft = min(MAX_FREE_TRANSFERS, ft - (used - gw.hits) + 1)
        else:
            money -= sum(info[p]["price"] for p in gw.squad)  # wildcard: ft unchanged
        if money < 0:
            raise RuleViolationError(f"GW{gw.gameweek}: over budget by {-money}")
        squad = gw.squad
