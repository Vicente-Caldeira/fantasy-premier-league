"""Import a manager's current FPL team from the public API.

Everything the planner needs is derivable without logging in:
  * squad and bank:   the latest gameweek's picks
  * selling prices:   FPL sells at purchase price + half of any rise (rounded down);
                      purchase prices come from the transfer history
                      (`element_in_cost`), or the gameweek-1 price for players held
                      since the start
  * free transfers:   replayed from the gameweek history: 1 at GW2, then
                      min(5, ft - free used + 1); wildcard / free hit keep them
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_FREE_TRANSFERS = 5
KEEPS_FREE_TRANSFERS = frozenset({"wildcard", "freehit"})


@dataclass(frozen=True)
class ImportedTeam:
    entry_id: int
    squad: list[int]  # player codes
    bank: int  # tenths
    free_transfers: int  # available for `next_gameweek`
    selling_prices: dict[int, int]  # player code -> tenths
    next_gameweek: int


def selling_price(purchase: int, current: int) -> int:
    """FPL's rule: keep half of any rise (rounded down); falls are passed on in full."""
    return purchase + (current - purchase) // 2 if current > purchase else current


def free_transfers(
    history: Sequence[Mapping[str, Any]], chips: Sequence[Mapping[str, Any]], next_gameweek: int
) -> int:
    """Free transfers available at `next_gameweek`'s deadline, replayed from history."""
    chip_by_gw = {c["event"]: c["name"] for c in chips}
    ft = 1  # available for gameweek 2; gameweek-1 changes are unlimited
    for week in sorted(history, key=lambda w: w["event"]):
        gw = week["event"]
        if gw == 1 or gw >= next_gameweek:
            continue
        if chip_by_gw.get(gw) in KEEPS_FREE_TRANSFERS:
            ft = min(MAX_FREE_TRANSFERS, ft + 1)
        else:
            ft = min(MAX_FREE_TRANSFERS, max(ft - week["event_transfers"], 0) + 1)
    return ft


def purchase_prices(
    squad_elements: Sequence[int],
    transfers: Sequence[Mapping[str, Any]],
    gameweek_one_prices: Mapping[int, int],
) -> dict[int, int]:
    """What the manager paid for each current player (latest transfer in)."""
    paid: dict[int, int] = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        paid[t["element_in"]] = t["element_in_cost"]
    return {e: paid.get(e, gameweek_one_prices.get(e, 0)) for e in squad_elements}


@dataclass(frozen=True)
class EntryData:
    """The public API responses for one manager."""

    history: Mapping[str, Any]  # /entry/{id}/history/
    transfers: Sequence[Mapping[str, Any]]  # /entry/{id}/transfers/
    picks: Mapping[str, Any]  # /entry/{id}/event/{gw}/picks/ for the latest gameweek


@dataclass(frozen=True)
class PlayerLookup:
    """Maps from this season's FPL element ids."""

    element_to_code: Mapping[int, int]
    current_prices: Mapping[int, int]
    gameweek_one_prices: Mapping[int, int]


def import_team(
    entry_id: int, data: EntryData, lookup: PlayerLookup, next_gameweek: int
) -> ImportedTeam:
    """Combine the API responses into the planner's inputs (keyed by player code)."""
    elements = [p["element"] for p in data.picks["picks"]]
    paid = purchase_prices(elements, data.transfers, lookup.gameweek_one_prices)
    prices = lookup.current_prices
    sell = {
        lookup.element_to_code[e]: selling_price(paid[e] or prices[e], prices[e]) for e in elements
    }
    return ImportedTeam(
        entry_id=entry_id,
        squad=[lookup.element_to_code[e] for e in elements],
        bank=data.picks["entry_history"]["bank"],
        free_transfers=free_transfers(
            data.history["current"], data.history["chips"], next_gameweek
        ),
        selling_prices=sell,
        next_gameweek=next_gameweek,
    )
