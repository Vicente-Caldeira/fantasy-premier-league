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

Results below; baseline = the current champion.

_Pending._

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
