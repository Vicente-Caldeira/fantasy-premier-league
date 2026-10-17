# Player-model experiments

A running log of what was tried, how it scored, and what was kept. Architecture: [model-architecture.md](model-architecture.md).

## Protocol

1. **Tune on 2021-22 and 2022-23 only** (`fpl_models.experiments.TUNING_SEASONS`). Every run is logged to MLflow under `player_model_tuning`.
2. **Change one thing at a time** against the current champion, then combine the winners.
3. **Score an accepted change once** on the evaluation seasons 2023-24 → 2025-26. Never iterate against them.
4. **Primary metric:** regulars MAE. A change must also not lose captain or top-11 points, since those are the decisions that matter.

```
uv run python -m fpl_models.experiments [--only NAME ...]
```

## Round 1: one change at a time

Tuning seasons 2021-22 + 2022-23 (76 gameweeks). Each row is the change against the baseline (the current champion: regulars MAE 1.8700, captain 7.24, top-11 62.01), as the mean of a paired bootstrap over gameweeks with its 95% interval. For MAE, negative is better.

| Experiment | Δ regulars MAE (95% CI) | Δ captain pts (95% CI) | Δ top-11 pts (95% CI) | Verdict |
|---|---|---|---|---|
| **crowd_current** (this GW's pre-deadline transfers, ownership) | **−0.027 (−0.037, −0.018)** | +0.40 (−0.95, +1.73) | **+3.36 (+0.92, +5.72)** | **Large, real.** Train/serve skew: complete window in training, partial live |
| recency_180d (training-row half-life) | −0.011 (−0.016, −0.005) | −0.63 (−2.15, +0.89) | −1.98 (−4.24, +0.25) | Real MAE gain; decisions unclear |
| l2_1 | −0.008 (−0.011, −0.005) | +1.05 (−0.05, +2.16) | +0.74 (−1.23, +2.73) | Real, small |
| recency_365d | −0.006 (−0.011, −0.001) | +0.22 | −1.25 | Real, small |
| min_leaf_100 | −0.005 (−0.009, −0.002) | +0.37 | +0.01 | Real, small |
| leaves_63 | −0.004 (−0.008, −0.001) | +0.47 | −1.38 | Real, small |
| lr03_iter400 | −0.004 (−0.006, −0.002) | +0.28 | −0.90 | Real, small |
| recency_730d, seasons_6, seasons_2, leaves_15 | CI includes 0 | | leaves_15: +2.15 (+0.15, +4.16) | No MAE effect |
| crowd_lagged (previous GW's transfers) | −0.001 (−0.005, +0.002) | | | **No effect:** only *current* news matters |
| multi_half_life (4 and 16 GW rates too) | −0.000 | | | No effect |

**Takeaways**
- **Team news dominates.** The current gameweek's transfer activity improves MAE 2.5× more than any hyperparameter, and it's the only change that significantly improves a decision metric.
- **Hyperparameters give small but real gains** (≤ 0.6% MAE each). They're combined in round 2 as `tuned`.
- **Captain points are very noisy over two seasons.** No experiment moved them significantly.

## Backlog (ranked by expected value)

| Idea | Why it might help | Notes |
|---|---|---|
| **Crowd signal:** pre-deadline transfers in/out, ownership | Managers react to team news (injuries, rotation) before the deadline. That's the information the minutes model lacks most. | **Check timing first:** confirm the per-gameweek transfer counts are for transfers made *before* that gameweek's deadline |
| **Share-of-team features** (player xG90 ÷ team λ, minutes share) | The model has to infer how team expected goals splits between players; giving the split directly may sharpen attackers | |
| **Per-position models** | Defenders' and forwards' points come from different events | |
| **Recency-weighted training + longer feature half-lives** | Old seasons may describe players who've changed role | Partly covered in round 1 |
| **Opponent-by-position strength** (goals/xG conceded to forwards, etc.) | Team λ is position-agnostic | |
| **Rest days, fixture congestion, European games** | Rotation risk for clubs in Europe | Needs fixture data from other competitions (Elo-Insights has it for 2024+) |
| **Penalty and set-piece takers** | Large share of goals for a few players | FPL exposes takers only for the current state; no history |
| **Hyperparameter search** (Optuna) | Only a handful of settings tried by hand | After the features settle |
| **Distribution, not just the mean** (quantiles / variance) | Captaincy and rank decisions depend on upside and risk | Feeds a risk-aware optimiser objective |
| **Direct-points model stacked with the components** | A second view may correct systematic component errors | |

## Tried and rejected

| Change | Tuning-season result | Decision |
|---|---|---|
| Price-band calibration layer (11.0m+ under-predicted ~5%) | regulars MAE 1.872 vs 1.870; captain 7.17 vs 7.24; top-11 61.5 vs 62.0 | Rejected: no gain |
