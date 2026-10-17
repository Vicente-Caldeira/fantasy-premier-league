# 10. Player expected points from component models

- Status: accepted
- Date: 2026-10-17

## Context

FPL points are a sum of scoring events that depend on position: appearance, goals, assists, clean sheets, goals conceded, saves, defensive contributions, bonus and cards. Rules change between seasons (defensive contributions from 2025-26). Modelling events rather than total points lets today's rules be applied to any history, and lets each event use the signal that drives it: team expected goals for clean sheets, a player's share of attacking output for goals.

## Decision

- **Features:** player-fixture rows. Rolling minutes, appearance rates, EWM per-90 rates (goals, xG, assists, xA, bonus, BPS, saves, cards, CBIT/CBIRT) and EWM points are computed over **previous gameweeks only**. Stats are summed per gameweek and shifted by one, so both legs of a double gameweek share one pre-deadline state. Team context is each side's λ implied by opening odds (ADR 0009).
  - **Leakage proof** (tests): rebuilding features after deleting all data from gameweek g onwards leaves g's features unchanged, and changing g's own results doesn't change them either.
- **Components:** scikit-learn HistGradientBoosting.
  - A 3-class minutes model (0 / 1–59 / 60+).
  - Poisson regressions for goals, assists, bonus, saves (GKP only) and yellow cards.
  - A defensive-contribution classifier trained on all CBIT-labelled history (2016-19, 2025+).
  - Feature columns that are empty in the training data are masked identically at fit and predict time, because sklearn crashes on all-NaN columns.
- **Points assembly:** with `fpl_core.scoring` constants, under the evaluated season's rules:
  - appearance: P(1–59) + 2·P(60+);
  - goals and assists: expected count × position points;
  - clean sheet: P(60+)·e^(−λ_opp) × position points;
  - conceded: −P(60+)·E[⌊G/2⌋];
  - saves: E[⌊S/3⌋];
  - defensive contribution: 2·P(hit), from 2025-26 only;
  - bonus, and −yellow cards.
- **Backtest:** refits every 4 gameweeks on the latest 4 seasons before the deadline.
- **Live predictions:** `predict_upcoming` trains on all history and predicts the next 5 gameweeks from each player's latest state, using team-strength fixture expectations. FPL's availability flags have no history, so they can't be learned. Instead, `chance_of_playing_next_round` scales next gameweek's events, and `u` (unavailable) zeroes every gameweek. Output: `gold/player_points/expected_points`.

## Results (held out: 2023-24 → 2025-26)

| Predictor | Regulars MAE | MAE | Bias | Captain | Top-11 | Spearman |
|---|---|---|---|---|---|---|
| **components_gbm** | **1.790** | **0.976** | **−0.001** | **7.05** | **59.2** | **0.708** |
| ewm_hl6 (previous champion) | 1.830 | 1.143 | −0.210 | 6.97 | 51.8 | 0.651 |

Promoted to champion. Re-runs vary by about ±0.001 regulars MAE (1.790 vs 1.791), consistent with multi-threaded histogram building.

## Known limitations and next steps

- **Stars are under-credited.** Haaland's expected goals is ~21% of City's λ, below his usual share. GBM shrinks towards the mean. Candidates: explicit share-of-team-xG features, per-position models, quantile checks on the top decile.
- Minor events (red cards, own goals, missed or saved penalties) aren't modelled.
- No hyperparameter search yet. Any tuning must use 2021-23, never the evaluation seasons.
- Fitted components aren't yet registered in MLflow (Phase 4: serving).
