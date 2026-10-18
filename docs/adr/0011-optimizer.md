# 11. Squad and transfer planning as a mixed-integer program

- Status: accepted
- Date: 2026-10-17

## Context

Expected points only become decisions through FPL's constraints: a fixed squad shape, a budget, at most three players per club, valid formations, captaincy, and transfers that interact with banked free transfers and −4 hits over several gameweeks.

## Decision

- **Model:** one OR-Tools MIP solved with SCIP, over a horizon of H gameweeks. Per player and gameweek it has squad, lineup, captain, buy and sell binaries, plus per-gameweek integer free-transfer, hit and bank variables.
  - **Objective:** Σ_t 0.9^t · (XI + captain + 0.1·bench expected points) − 4·hits − 0.001·transfers.
  - **Free transfers:** `ft' ≤ ft − free_used + 1` and `ft' ≤ 5`. Upper bounds suffice because more free transfers are never worse.
  - **Hits are exact:** `paid = max(transfers − ft, 0)`, with one binary per gameweek. Free transfers are spent first.
  - **No churn:** buying and selling the same player in one gameweek is forbidden. The 0.001 tiebreak stops costless but pointless transfers, for example when 5 free transfers are banked.
- **Independent validator:** every plan is checked against FPL's rules separately from the solver: squad shape, clubs, formation, captaincy, transfers matching squad changes, the free-transfer and hit arithmetic, and the bank. **It caught two formulation bugs during development:** hits taken while free transfers remained, and churn transfers.
- **Pruning:** a player is dropped when at least 3× his position's slots are cheaper **and** at least as good in every gameweek. It's a heuristic (club limits make an exact proof hard) and is verified against the unpruned problem in tests and on live data (identical objective). It's 2.8× faster: 23 s against 64 s for 5 gameweeks.
- **Team import:** the public API is enough.
  - **Selling prices:** purchase prices from the transfer history (`element_in_cost`), sold at purchase plus half of any rise.
  - **Free transfers:** replayed from the gameweek history and chips (wildcard and free hit keep banked transfers).
- **Horizon:** the next H gameweeks whose **official FPL deadline** is still ahead (`dim_gameweeks`). Previously a gameweek already in progress could be planned.

## Consequences

- `python -m fpl_models.plan [--entry ID]` prints a plan, and the `gold/plans/wildcard` asset publishes the from-scratch plan daily.
- Prices are assumed constant over the horizon. Chips (bench boost, triple captain, free hit) aren't planned yet.
- Plans inherit the player model's biases. On live data the planner sells Haaland for Calvert-Lewin, which reflects the model's under-crediting of stars (ADR 0010), not the optimiser.
