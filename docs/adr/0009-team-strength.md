# 9. Team strength: market odds where priced, Poisson ratings elsewhere

- Status: accepted
- Date: 2026-10-17

## Context

Player points depend on each team's expected goals (λ) in a fixture. Clean-sheet probability is exp(−λ_opponent), and a team's attacking returns are shared out of its λ. FPL decisions also look several gameweeks ahead, where bookmakers haven't priced matches yet.

## Decision

- **Market model:** λ are solved for, under independent Poisson, so that they reproduce the margin-free **opening** odds (1X2 + over/under 2.5). Closing odds are excluded because they're struck after the FPL deadline.
- **Ratings model:** log λ = μ + home + attack − defence, plus learned promoted-team effects. It's fitted by time-weighted Poisson likelihood with an analytic gradient (L-BFGS) and L2 shrinkage, so each gameweek refit takes milliseconds. The target is a 50/50 blend of goals and Understat xG. Unseen teams start at average strength.
- **Tuning** used only 2021-22 and 2022-23, by goals deviance:
  - half-life 180 days, chosen from {90, 180, 365};
  - xG weight 0.5, from {0, 0.5, 1};
  - L2 1.0, bracketed by 0.3 and 5.
- **Production model:** `market_else_ratings` uses market λ where a match has odds and ratings otherwise. In the tuning seasons, blends were dominated by the market alone, so no blend is used.
- **Planning horizons:** the backtest can predict gameweek g from information as of the deadline k gameweeks earlier, with odds withheld.

## Results (held out: 2023-24 → 2025-26, 1,140 matches)

| Model | Horizon | Goals deviance | Clean-sheet log-loss | Outcome log-loss |
|---|---|---|---|---|
| market (= production) | 0 | **1.0514** | **0.5106** | **0.9663** |
| ratings | 0 | 1.0684 | 0.5124 | 0.9742 |
| ratings | 6 | 1.0762 | 0.5150 | 0.9784 |
| league average | 0 | 1.1867 | 0.5425 | 1.0739 |

Ratings six gameweeks ahead are close to the market for next week, and far ahead of the league average. The horizon-1 ratings score fractionally better than horizon 0 (1.0662 vs 1.0684). That's within noise, not an effect to rely on.

## Consequences

- `gold/team_strength/fixture_lambdas` publishes λ, outcome and clean-sheet probabilities for every unplayed fixture, with their `source`. It's the input to the player models.
- Independent Poisson ignores the slight excess of low-scoring draws. A Dixon–Coles correction is a candidate improvement if clean-sheet calibration shows bias.
- The score matrix is truncated at 25 goals. The original 10 dropped up to 4% of the probability mass at the solver's λ bound.
