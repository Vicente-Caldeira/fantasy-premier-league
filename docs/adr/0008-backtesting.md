# 8. Walk-forward backtesting with structural leakage prevention

- Status: accepted
- Date: 2026-10-17

## Context

Points models will be iterated on for months. Their offline scores have to predict live performance, and the most common way they don't is **leakage**: information from after a deadline reaching the model, for example post-match `xP`, season totals, or rolling features computed over the whole season.

## Decision

- **Unit:** a player's points for one gameweek, summed over his fixtures (double and triple gameweeks included). Gameweeks with unfinished fixtures are excluded.
- **Deadline:** 90 minutes before the gameweek's first kickoff.
- **Walk-forward:** every gameweek of the evaluation seasons is predicted in deadline order.
  - The model receives `history`: player-gameweeks with a deadline strictly earlier.
  - It also receives `targets`: the gameweek's players with **context columns only**. The outcome columns are removed before the model sees them.
  - So leakage is impossible by construction rather than prevented by convention. Tests assert both properties.
- **Alignment:** predictions are positional. Every lookup join uses `maintain_order="left"`, and a test with a player-specific signal fails if predictions ever drift to the wrong player.
- **Metrics:**
  - MAE, RMSE and bias on all rows, and on **regulars** (players with minutes in any of their previous three gameweeks, decided from pre-deadline data only);
  - **captain points** (the actual points of the top prediction) and **top-11 points**, each against the hindsight oracle;
  - per-gameweek Spearman correlation between predicted and actual ranking.
- **Primary metric:** `regulars_mae` on 2023-24, 2024-25 and 2025-26. A model is promoted only if it beats the current champion there. The decision metrics are reported alongside.
- **Tracking:** every run is logged to MLflow (experiment `backtests`). The leaderboard is a Dagster asset (`gold/backtests/leaderboard`) rebuilt after each warehouse build. Its metadata shows the champion.

## Consequences

- **Targets use each season's own scoring rules,** because history can't be rescored: defensive stats only exist from 2025-26. Component models will predict events and apply the current rules. Evaluating those against historical points carries a small, documented rule mismatch.
- **FPL's own `ep_next` can't be backtested historically.** vaastav's `xP` is scraped after matches. It becomes a baseline once our daily snapshots cover enough gameweeks.
- **Baseline champion at adoption:** `ewm_hl6` (regulars MAE 1.830, captain 6.97 vs oracle 17.79, top-11 51.8 vs 140.0).
