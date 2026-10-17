"""Plan transfers, lineups and captains for the coming gameweeks.

    uv run python -m fpl_models.plan                  # best squad from scratch (wildcard)
    uv run python -m fpl_models.plan --entry 1234567  # plan from your current FPL team

Reads the latest `gold/player_points/expected_points` (run the `warehouse_build` job
first) and the current squads from the warehouse.
"""

import argparse
import glob
import os
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from fpl_core.fpl_api import FplApiClient
from fpl_models import optimizer, team_import
from fpl_models.data import DEFAULT_WAREHOUSE, next_gameweeks


def latest_expected_points(lake: str, gameweeks: list[int]) -> pl.DataFrame:
    files = sorted(glob.glob(f"{lake}/gold/player_points/expected_points/*/data.parquet"))
    if not files:
        raise SystemExit("No expected points yet: run the warehouse_build job first.")
    xp = pl.read_parquet(files[-1]).filter(pl.col("gameweek").is_in(gameweeks))
    missing = set(gameweeks) - set(xp["gameweek"].unique())
    if missing:
        raise SystemExit(
            f"Expected points don't cover GW {sorted(missing)}: re-run warehouse_build."
        )
    return xp


def load_team(
    entry_id: int, warehouse: Path, squads: pl.DataFrame, next_gw: int
) -> team_import.ImportedTeam:
    with FplApiClient() as client:
        data = team_import.EntryData(
            history=client.entry_history(entry_id),
            transfers=client.entry_transfers(entry_id),
            picks=client.entry_picks(entry_id, next_gw - 1),
        )
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        gw1 = connection.sql(
            """select element, price from fct_player_matches
            where season = (select max(season) from fct_player_matches) and gameweek = 1"""
        ).pl()
    lookup = team_import.PlayerLookup(
        element_to_code=dict(zip(squads["element"], squads["player_code"], strict=True)),
        current_prices=dict(zip(squads["element"], squads["price"], strict=True)),
        gameweek_one_prices=dict(gw1.iter_rows()),
    )
    return team_import.import_team(entry_id, data, lookup, next_gw)


POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def transfer_pairs(gw: optimizer.Gameweek, position: dict[int, str]) -> list[tuple[int, int]]:
    """Pair each player sold with one bought in the same position (squad shape is fixed)."""
    pairs = []
    for pos in POSITION_ORDER:
        out = sorted(p for p in gw.transfers_out if position[p] == pos)
        new = sorted(p for p in gw.transfers_in if position[p] == pos)
        pairs += list(zip(out, new, strict=True))
    return pairs


def describe(
    plan: optimizer.Plan, squads: pl.DataFrame, team: team_import.ImportedTeam | None
) -> str:
    name = dict(zip(squads["player_code"], squads["web_name"], strict=True))
    position = dict(zip(squads["player_code"], squads["position"], strict=True))
    lines = []
    if team:
        lines.append(
            f"Team {team.entry_id}: bank {team.bank / 10:.1f}m, "
            f"{team.free_transfers} free transfer(s)"
        )
    for i, (gw, bank) in enumerate(zip(plan.gameweeks, plan.bank, strict=True)):
        if i == 0 and team is None:
            moves = "new squad (wildcard)"
        else:
            pairs = transfer_pairs(gw, position)
            moves = ", ".join(f"{name[o]} -> {name[n]}" for o, n in pairs) or "none"
        hits = f", -{gw.hits * optimizer.HIT_COST} hit" if gw.hits else ""
        lines.append(
            f"\nGW{gw.gameweek}  xP {gw.expected_points:.1f}{hits}"
            f"  (FT {gw.free_transfers}, bank {bank / 10:.1f}m)"
        )
        lines.append(f"  transfers: {moves}")
        xi = sorted(gw.lineup, key=lambda p: POSITION_ORDER[position[p]])
        tags = {gw.captain: " (C)", gw.vice_captain: " (V)"}
        lines.append("  XI: " + ", ".join(name[p] + tags.get(p, "") for p in xi))
        lines.append("  bench: " + ", ".join(name[p] for p in gw.squad if p not in gw.lineup))
    lines.append(
        f"\nExpected points over the horizon, after hits: {plan.total_expected_points:.1f}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=int, help="your FPL team id (from the points page URL)")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--free-transfers", type=int, help="override the replayed count")
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    args = parser.parse_args()

    gameweeks = next_gameweeks(args.warehouse, datetime.now(UTC), args.horizon)
    with duckdb.connect(str(args.warehouse), read_only=True) as connection:
        squads = connection.sql("select * from current_squads").pl()
    xp = latest_expected_points(os.getenv("FPL_LAKE_URL", "data"), gameweeks)
    team = load_team(args.entry, args.warehouse, squads, gameweeks[0]) if args.entry else None
    plan = optimizer.optimise(
        squads.select("player_code", "position", "team_key", "price"),
        xp.select("player_code", "gameweek", "expected_points"),
        current_squad=team.squad if team else None,
        bank=team.bank if team else 0,
        free_transfers=args.free_transfers or (team.free_transfers if team else 1),
        selling_prices=team.selling_prices if team else None,
    )
    print(describe(plan, squads, team))


if __name__ == "__main__":
    main()
