"""Shadow team: the planner manages a real FPL team through the live season.

The honest scoreboard for the whole system: no simulation assumptions, the same
predictions and optimiser a user sees, scored on real points. State lives in the lake:

    gold/shadow/state.json                    squad, purchase prices, bank, free
                                              transfers, the pending plan, history
    gold/shadow/history/gameweek=NN.json      each scored gameweek

Before a deadline, `plan` chooses transfers, XI and captain for the next gameweek from
the latest expected points (re-running before the deadline replaces the plan). Once
that gameweek has finished, `score` applies the transfers and scores it with real
points: auto-subs, captain (vice if he didn't play), hits. `run` does both.

    uv run python -m fpl_models.shadow init --from-path data/experiments/e1/paths/X.json
    uv run python -m fpl_models.shadow run

The state transitions (`plan_next`, `commit`, `from_path`) are pure.
"""

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from fpl_models import optimizer
from fpl_models.data import DEFAULT_WAREHOUSE, next_gameweeks
from fpl_models.plan import latest_expected_points
from fpl_models.season_sim import score
from fpl_models.team_import import selling_price

HORIZON = 3  # as in the season simulator (E7: longer horizons over-trade)


@dataclass(frozen=True)
class Pick:
    """One gameweek's decisions (and, once played, its score)."""

    gameweek: int
    squad: list[int]
    lineup: list[int]
    bench: list[int]  # in substitution order
    captain: int
    vice_captain: int
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    expected_points: float
    points: int | None = None  # after hits, once the gameweek is played


@dataclass(frozen=True)
class State:
    season: str
    squad: list[int]
    paid: dict[int, int]  # purchase price per player, tenths
    bank: int  # tenths
    free_transfers: int  # available for the next gameweek
    pending: Pick | None = None
    history: list[Pick] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(p.points or 0 for p in self.history)


def plan_next(
    state: State,
    players: pl.DataFrame,
    expected_points: pl.DataFrame,
    *,
    settings: optimizer.Settings | None = None,
) -> State:
    """Choose the next gameweek's transfers, XI and captain (replacing any pending plan).

    players:          player_code, position, team_key, price (current, tenths)
    expected_points:  player_code, gameweek, expected_points (next gameweek first)
    """
    price = dict(zip(players["player_code"], players["price"], strict=True))
    plan = optimizer.optimise(
        players,
        expected_points,
        current_squad=state.squad,
        bank=state.bank,
        free_transfers=state.free_transfers,
        selling_prices={p: selling_price(state.paid[p], price[p]) for p in state.squad},
        settings=settings or optimizer.Settings(),
    )
    step = plan.gameweeks[0]
    this_week = {
        p: v for p, g, v in expected_points.select(
            "player_code", "gameweek", "expected_points"
        ).iter_rows() if g == step.gameweek
    }  # fmt: skip
    position = dict(zip(players["player_code"], players["position"], strict=True))
    bench = sorted(
        (p for p in step.squad if p not in step.lineup),
        key=lambda p: (position[p] == "GKP", -this_week.get(p, 0.0)),
    )
    pick = Pick(
        gameweek=step.gameweek,
        squad=sorted(step.squad),
        lineup=sorted(step.lineup),
        bench=bench,
        captain=step.captain,
        vice_captain=step.vice_captain,
        transfers_in=sorted(step.transfers_in),
        transfers_out=sorted(step.transfers_out),
        hits=step.hits,
        expected_points=round(step.expected_points, 2),
    )
    return replace(state, pending=pick)


def commit(
    state: State,
    prices: Mapping[int, int],
    actual: Mapping[int, int],
    minutes: Mapping[int, int],
    position: Mapping[int, str],
) -> State:
    """Apply the pending plan's transfers at that deadline's `prices`, and score it."""
    pick = state.pending
    if pick is None:
        raise ValueError("no pending plan to commit")
    sell = {p: selling_price(state.paid[p], prices[p]) for p in state.squad}
    bank = (
        state.bank
        + sum(sell[p] for p in pick.transfers_out)
        - sum(prices[p] for p in pick.transfers_in)
    )
    if bank < 0:
        raise ValueError(f"GW{pick.gameweek}: transfers overspend the bank ({bank})")
    paid = {p: v for p, v in state.paid.items() if p not in pick.transfers_out}
    paid |= {p: prices[p] for p in pick.transfers_in}
    used = len(pick.transfers_in) - pick.hits
    free = min(optimizer.MAX_FREE_TRANSFERS, state.free_transfers - used + 1)
    gw = optimizer.Gameweek(
        gameweek=pick.gameweek, squad=pick.squad, lineup=pick.lineup, captain=pick.captain,
        vice_captain=pick.vice_captain, transfers_in=pick.transfers_in,
        transfers_out=pick.transfers_out, free_transfers=state.free_transfers,
        hits=pick.hits, expected_points=pick.expected_points,
    )  # fmt: skip
    points = score(gw, pick.bench, dict(actual), dict(minutes), dict(position))
    points -= optimizer.HIT_COST * pick.hits
    return replace(
        state,
        squad=pick.squad,
        paid=paid,
        bank=bank,
        free_transfers=free,
        pending=None,
        history=[*state.history, replace(pick, points=points)],
    )


def from_path(
    season: str, records: Sequence[Mapping[str, object]], prices: Mapping[int, Mapping[int, int]]
) -> State:
    """The state after a simulated path (`e1` path records), replayed at real prices.

    `prices`: gameweek -> player -> price at that deadline. The first record is a
    from-scratch squad on the 100.0m budget; later squads differ by transfers.
    """
    state: State | None = None
    for record in records:
        gw = int(record["gameweek"])  # type: ignore[call-overload]
        squad = sorted(int(p) for p in record["squad"])  # type: ignore[attr-defined]
        hits = int(record["hits"])  # type: ignore[call-overload]
        price = prices[gw]
        if state is None:
            paid = {p: price[p] for p in squad}
            state = State(season, squad, paid, optimizer.BUDGET - sum(paid.values()), 1)
            transfers_in, transfers_out = squad, []
        else:
            transfers_in = sorted(set(squad) - set(state.squad))
            transfers_out = sorted(set(state.squad) - set(squad))
            sell = {p: selling_price(state.paid[p], price[p]) for p in state.squad}
            bank = state.bank + sum(sell[p] for p in transfers_out)
            bank -= sum(price[p] for p in transfers_in)
            paid = {p: v for p, v in state.paid.items() if p not in transfers_out}
            paid |= {p: price[p] for p in transfers_in}
            used = len(transfers_in) - hits
            free = min(optimizer.MAX_FREE_TRANSFERS, state.free_transfers - used + 1)
            state = replace(state, squad=squad, paid=paid, bank=bank, free_transfers=free)
        pick = Pick(
            gameweek=gw, squad=squad, lineup=sorted(record["lineup"]),  # type: ignore[call-overload]
            bench=list(record["bench"]), captain=int(record["captain"]),  # type: ignore[call-overload]
            vice_captain=int(record["vice_captain"]),  # type: ignore[call-overload]
            transfers_in=transfers_in, transfers_out=transfers_out, hits=hits,
            expected_points=0.0, points=int(record["points"]),  # type: ignore[call-overload]
        )  # fmt: skip
        state = replace(state, history=[*state.history, pick])
    if state is None:
        raise ValueError("empty path")
    return state


# --- persistence and I/O -----------------------------------------------------------


def to_json(state: State) -> str:
    return json.dumps(asdict(state), indent=1)


def from_json(text: str) -> State:
    raw = json.loads(text)
    picks = [Pick(**p) for p in raw.pop("history")]
    pending = Pick(**raw["pending"]) if raw.get("pending") else None
    paid = {int(k): v for k, v in raw.pop("paid").items()}
    return State(**{**raw, "paid": paid, "pending": pending, "history": picks})


def _lake() -> Path:
    return Path(os.getenv("FPL_LAKE_URL", "data"))


def _state_file(lake: Path) -> Path:
    return lake / "gold" / "shadow" / "state.json"


def save(state: State, lake: Path) -> None:
    path = _state_file(lake)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(state))
    history = path.parent / "history"
    history.mkdir(exist_ok=True)
    for pick in state.history:
        (history / f"gameweek={pick.gameweek:02d}.json").write_text(json.dumps(asdict(pick)))


def _season_prices(warehouse: Path, season: str) -> dict[int, dict[int, int]]:
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        rows = connection.execute(
            "select gameweek, player_code, max(price) from fct_player_matches"
            " where season = ? group by 1, 2",
            [season],
        ).fetchall()
    out: dict[int, dict[int, int]] = {}
    for gw, code, price in rows:
        out.setdefault(int(gw), {})[int(code)] = int(price)
    return out


def _played(warehouse: Path, season: str, gameweek: int) -> pl.DataFrame:
    """Final points and minutes per player for a finished gameweek (empty if not)."""
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        finished = connection.execute(
            "select finished from dim_gameweeks where season = ? and gameweek = ?",
            [season, gameweek],
        ).fetchone()
        if not finished or not finished[0]:
            return pl.DataFrame()
        return connection.execute(
            """select player_code, position, max(price) as price, sum(total_points) as points,
            sum(minutes) as minutes, bool_or(is_provisional) as provisional
            from fct_player_matches where season = ? and gameweek = ? group by 1, 2""",
            [season, gameweek],
        ).pl()


def run(lake: Path, warehouse: Path) -> State:
    """Score the pending gameweek if it has finished, then plan the next one."""
    state = from_json(_state_file(lake).read_text())
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        players = connection.sql(
            "select player_code, position, team_key, price from current_squads"
        ).pl()
    if state.pending is not None:
        played = _played(warehouse, state.season, state.pending.gameweek)
        if played.height and not played["provisional"].any():
            # Prices at that deadline; players without a fixture (blank) at today's.
            prices = dict(zip(players["player_code"], players["price"], strict=True))
            prices |= dict(zip(played["player_code"], played["price"], strict=True))
            position = dict(zip(players["player_code"], players["position"], strict=True))
            state = commit(
                state,
                prices,
                dict(zip(played["player_code"], played["points"], strict=True)),
                dict(zip(played["player_code"], played["minutes"], strict=True)),
                position | dict(zip(played["player_code"], played["position"], strict=True)),
            )
    gameweeks = next_gameweeks(warehouse, datetime.now(UTC), HORIZON)
    if state.pending is None or state.pending.gameweek == gameweeks[0]:
        xp = latest_expected_points(str(lake), gameweeks).select(
            "player_code", "gameweek", "expected_points"
        )
        state = plan_next(state, players, xp)
    save(state, lake)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "run", "show"])
    parser.add_argument("--from-path", type=Path, help="init: an e1 path file (season paths)")
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    args = parser.parse_args()
    lake = _lake()
    if args.command == "init":
        if args.from_path is None:
            raise SystemExit("init needs --from-path")
        records = json.loads(args.from_path.read_text())
        state = from_path(args.season, records, _season_prices(args.warehouse, args.season))
        # Deadlines passed since the path ended (e.g. a gameweek not final yet): the squad
        # is held, each earning a free transfer; their points are not counted.
        upcoming = next_gameweeks(args.warehouse, datetime.now(UTC), 1)
        skipped = upcoming[0] - state.history[-1].gameweek - 1 if upcoming else 0
        if skipped > 0:
            free = min(optimizer.MAX_FREE_TRANSFERS, state.free_transfers + skipped)
            state = replace(state, free_transfers=free)
            print(f"held the squad through {skipped} unscored gameweek(s)")
        save(state, lake)
    elif args.command == "run":
        state = run(lake, args.warehouse)
    else:
        state = from_json(_state_file(lake).read_text())
    last = state.history[-1].gameweek if state.history else 0
    print(f"shadow {state.season}: {state.total} points through GW{last}, "
          f"bank {state.bank / 10:.1f}m, {state.free_transfers} free transfer(s)")  # fmt: skip
    if state.pending:
        p = state.pending
        print(f"GW{p.gameweek} plan: in {p.transfers_in}, out {p.transfers_out}, "
              f"captain {p.captain}, {p.hits} hit(s), xP {p.expected_points}")  # fmt: skip


if __name__ == "__main__":
    main()
