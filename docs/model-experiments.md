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

## Round 2: new inputs and targets

| Experiment | Δ regulars MAE (95% CI) | Δ top-11 pts (95% CI) | Verdict |
|---|---|---|---|
| tuned + crowd_share | −0.034 (−0.040, −0.027) | +1.8 (−0.7, +4.3) | Best accuracy |
| denoise_targets (train on xG/xA) | −0.032 (−0.045, −0.021) | −2.0 (−4.8, +0.5) | Big MAE gain, decisions possibly worse |
| crowd_current | −0.027 (−0.037, −0.018) | +3.4 (+0.9, +5.7) | Real, skew-prone live |
| **crowd_share** (net transfers ÷ all activity) | **−0.022 (−0.030, −0.015)** | **+3.4 (+1.1, +5.8)** | **Keeps almost all the value; window-invariant, safe live** |
| tuned (round 1's significant settings) | −0.012 (−0.017, −0.008) | n.s. | Real, small |
| reconcile (player goals add up to team λ) | −0.009 (−0.012, −0.006) | n.s. | Real, small alone |
| schedule (rest days, 14-day congestion) | −0.005 (−0.009, −0.002) | n.s. | Real, small |

Live check of the crowd counters: a few hours after the GW7 deadline, FPL's `transfers_in_event` / `transfers_out_event` summed to 0.4M vs GW7's final 15.1M (correlation 0.79). They count the **open** window, so the share feature is available live for the next gameweek only. Later gameweeks use a planning model trained without crowd features.

## Round 3: combination and ablation

`all_in` = tuned + crowd_share + schedule + xG targets + reconcile.

| Variant | Δ MAE vs baseline | Δ MAE vs all_in when removed (95% CI) | Δ top-11 vs all_in |
|---|---|---|---|
| all_in | −0.042 (−0.051, −0.034) | | |
| without reconcile | **−0.062 (−0.075, −0.050)** | **−0.020 (−0.030, −0.011): removing helps** | +0.4 |
| without crowd | −0.023 | +0.020 (+0.013, +0.025): essential | **−2.6 (−4.6, −0.5)** |
| without xG targets | −0.035 | +0.008 (+0.002, +0.014): helps MAE | **+2.3 (−0.1, +4.9): costs decisions** |
| without schedule | −0.040 | +0.003 (0.000, +0.006): marginal | −0.8 |
| untuned | −0.040 | +0.002 (n.s.) | −0.4 |

**Takeaways**
- **Reconciliation conflicts with the other changes:** drop it from the point model. The simulator is coherent by construction (team goals = the sum of simulated player goals), so team consistency doesn't depend on it.
- **The crowd signal is the single most important addition**, for accuracy and for decisions.
- **Pure xG targets help MAE but hurt the top-11.** Plausibly they under-reward elite finishers who beat their xG season after season. Round 4 tests a 50/50 blend.

## Round 4: training target blend (no reconciliation)

Same as `all_in` without reconcile; crowd ownership is now the fraction of managers (selected ÷ (all ownership ÷ 15)). That's stable as FPL grows, and identical in effect to the log count: 1.8056 vs 1.8080.

| Target (goals ↔ xG) | Δ MAE vs baseline (95% CI) | Δ captain | Δ top-11 |
|---|---|---|---|
| **xG only (1.0)** | **−0.062 (−0.076, −0.049)** | −1.2 (−2.8, +0.3) | +0.4 (−3.0, +3.7) |
| 50/50 (0.5) | −0.051 (−0.061, −0.042) | −0.4 (−1.7, +1.0) | +1.5 (−1.5, +4.5) |
| goals only (0.0) | −0.039 (−0.046, −0.031) | −1.1 | +1.2 |

Directly, xG-only vs 50/50: MAE −0.011 (−0.016, −0.005), significant; top-11 −1.1 (−2.8, +0.5), not significant. **Both cut MAE by ~3%**, the largest gain so far. Decision metrics can't separate them, so **season simulation is the tie-breaker** (below).

## Season simulation (the acceptance test)

`fpl_models.season_sim` manages a team through a whole past season with decisions made blind:
- at each deadline, only data from before it is used;
- the next gameweek is predicted with the crowd model, later gameweeks with the planning model and ratings-based team strength;
- the optimiser picks transfers, the XI and the captain (3-gameweek horizon);
- scoring is real: captain/vice fallback, formation-valid auto-subs, hits, real price changes and the selling rule.

No chips are used.

| Config | 2021-22 | 2022-23 | Total | Hits |
|---|---|---|---|---|
| baseline | 1925 | 2195 | 4120 | 85 |
| xG targets (1.0) | 2184 | 2228 | 4412 (**+292**) | 60 |
| 50/50 targets (0.5) | 2184 | 2235 | 4419 (**+299**) | 64 |

FPL only has xG from 2022-23, so the two target variants are identical on 2021-22. **+7 points on one season is noise.** Per-gameweek metrics favour pure xG on MAE and the blend on decisions, a wash, so the simpler **pure xG** is kept.

**Optimiser settings** (v2 model):

| Setting | 2021-22 | 2022-23 | Total | Hits |
|---|---|---|---|---|
| **hit cost 4 when deciding, 3-GW horizon (default)** | 2184 | 2228 | **4412** | 60 |
| hit cost 6 | 2056 | 2345 | 4401 | 21 |
| hit cost 8 | 1990 | 2256 | 4246 | 9 |
| 5-GW horizon | 2250 | 2128 | 4378 | 66 |

Defaults kept. Heavy hit aversion costs points. Single-season paths swing by ±120, so differences under ~150 points are noise: one path per season is a weak test. **Next:** many paths per setting (different starting gameweeks or squads) to tighten it.

These totals aren't comparable to real managers' scores: no chips, and no in-season squad from GW1 history.

## Held-out evaluation (run once: 2023-24, 2024-25, 2025-26)

Every choice above was made on 2021-23 only. The final candidate (`player_model.PRODUCTION` = tuned GBM + xG targets + crowd share/ownership + schedule) was then scored a single time on the held-out seasons:

| | regulars MAE | Δ vs v1 (95% CI) | Δ captain pts / GW | Δ top-11 pts / GW |
|---|---|---|---|---|
| v1 (original component model) | 1.7914 | | | |
| **v2 (PRODUCTION)** | **1.7491** | **−0.042 (−0.049, −0.036)** | +0.6 (−0.5, +1.6) | +1.5 (−0.7, +3.6) |

The accuracy gain carries over almost exactly (tuning: −0.062; held-out: −0.042, the usual small shrinkage). The decision metrics point the right way but aren't individually significant. **v2 is promoted:** it is now used by the `expected_points` asset, and both versions appear on the leaderboard.

## Simulator calibration (fit before 2022-23, simulate all of 2022-23)

For player-gameweeks with a simulated mean of at least 2 (about 6,600):

| Simulator | Mean pred / actual | p10–p90 coverage (target 80%) | P(haul ≥10) pred / actual | Brier skill vs base rate |
|---|---|---|---|---|
| Independent bonus, Poisson goals | 3.33 / 3.35 | 85.7% | 0.046 / 0.064 | 0.025 |
| + bonus tied to returns (0.5) | 3.33 / 3.35 | 86.0% | 0.048 / 0.064 | 0.028 |
| **+ gamma-mixed attacking rates (shape 2), now the default** | 3.34 / 3.34 | 86.2% | 0.050 / 0.064 | 0.032 |

- **Means are calibrated.** The intervals are slightly too wide at the bottom: few players score below p10, because points are discrete and many sit at 1–2.
- **Hauls are still under-predicted by about 20%.** Candidates: no penalty-taker or set-piece information, saves and clean-sheet bonus independent of the result, and no in-match game-state effects. That matters for captaincy, so the risk-aware captain work should wait for a fix or apply a calibration factor.

## Are defender captains a bug? (tuning seasons, v2)

Live v2 sometimes ranks an Arsenal defender above Haaland. Checking predictions of 4.5+ by position:

| Position | n | Predicted | Actual | Hauls |
|---|---|---|---|---|
| DEF | 397 | 5.69 | 5.40 (−5%) | 16% |
| GKP | 107 | 6.36 | 5.97 (−6%) | 18% |
| MID | 765 | 5.84 | 5.82 | 19% |
| FWD | 253 | 5.91 | 5.64 (−5%) | 21% |

In the 6 gameweeks where a defender was the model's top pick, he scored 8.8 on average. **There's a mild top-end over-prediction for defenders, goalkeepers and forwards, but no evidence that defender captains are wrong.** Forwards haul more often at the same mean, which matters for rank. That is the case for the risk-aware captaincy item below.

## Backlog (ranked by expected value)

| Idea | Why it might help | Status |
|---|---|---|
| ~~Crowd signal~~ | Team news before the deadline | **Done:** the biggest single gain (window-invariant share) |
| ~~xG/xA training targets~~ | Less label noise | **Done:** large MAE gain; season simulation decides pure vs blend |
| ~~Rest days / congestion (league)~~ | Rotation | Done: small effect |
| ~~Team reconciliation~~ | Coherent team totals | Done: hurts in combination; coherence moved to the simulator |
| **Season simulation as the acceptance test** | Total points is the real objective; per-gameweek metrics are proxies | **In progress** |
| **Risk-aware captaincy and optimiser** (simulator distributions: p90, P(haul)) | Captain picks and rank depend on upside; expected value alone undervalues premiums | Next |
| **European and cup fixtures** (minutes, rest) | Rotation for clubs in Europe | Needs other-competition ingestion (Elo-Insights has it from 2025-26) |
| **Opta features** (touches in box, chances created, xGOT) | Richer attacking signal | Data only from 2024-25: needs a separate tuning split (e.g. tune on 2024-25, hold out 2025-26) |
| **Daily snapshots → availability model** (news, status, chance of playing) | Minutes are the largest error source | Data accrues from 2026-10-17; train once a season of snapshots exists |
| **Prediction timing** (run ~2 h before each deadline) | Crowd share and news are most complete near the deadline | Dagster sensor on `dim_gameweeks` |
| **Per-position models, share-of-team features, hyperparameter search** | Further accuracy | After the above |
| **Chip strategy** (wildcard, free hit, bench boost, triple captain) | Large point swings per season | Simulator + optimiser make it testable |

## Brainstorm: what would make the biggest difference next

1. **Many-path season simulation:** run each setting from many starting gameweeks or squads, so differences under ~150 points become measurable. Every optimiser, chip and captaincy decision then gets a real acceptance test. Cost: CPU only.
2. **Risk-aware captaincy:** choose the captain on the simulated distribution: mean plus a weight on P(haul), or rank-aware value against effective ownership. Tune the weight with (1).
3. **Penalty and set-piece takers:** the bootstrap has `penalties_order` and `corners_and_indirect_freekicks_order`. Daily snapshots start now. Historically, penalties can be recovered from Understat shot data (situation = penalty). This is the most likely explanation for the haul gap.
4. **Chips:** the wildcard, free hit, bench boost and triple captain are worth tens of points a season. The MIP already supports a wildcard week; add the free hit and bench boost, and time all four with (1).
5. **European and cup congestion:** Elo-Insights carries other competitions from 2025-26. Add them to `schedule_features` and test rotation effects on minutes.
6. **Availability model from daily snapshots** (status, news text, chance of playing): minutes are the largest error source. Train once a season of snapshots exists, and keep collecting daily. That's the Dagster schedule; it needs to run unattended (Cloud Run job in the free tier).
7. **Deadline-timed predictions:** a Dagster sensor on `dim_gameweeks` reruns ~2 h before each deadline, when the crowd share and news are most complete.
8. **Model registry and serving:** persist the fitted v2 components to the MLflow registry, so the API serves predictions without retraining (about 50 s per run today).
9. **Top-end bias correction** for DEF, GKP and FWD (−5%), by isotonic regression on the tuning seasons. The earlier price-band calibration failed, but a position × predicted-value one wasn't tried.

## E1: how noisy is season simulation? (20 paths per config × season)

Each path starts from a different reasonable squad (seeded noise on the first deadline's expected points) and then plays normally. Predictions are computed once per config × season and shared by all paths.

| Config | Season | Mean | SD | Min–max | Hits |
|---|---|---|---|---|---|
| v1 | 2021-22 | 1966 | 56 | 1871–2079 | 50 |
| v2 | 2021-22 | 2097 | 80 | 1971–2241 | 49 |
| v1 | 2022-23 | 2132 | 69 | 2029–2275 | 34 |
| v2 | 2022-23 | 2119 | 58 | 2013–2260 | 19 |

- **v2 − v1 = +59 per season-path (95% CI +27, +92).** That's +131 in 2021-22 and −13 in 2022-23. **The earlier single-path +292 was inflated by luck:** the honest figure is about +120 over two seasons.
- **One path per season has an SD of 55–80 points.** Detecting 50 points with 80% power needs **~27 paths per setting**.
- **Pairing paths by seed doesn't help** (SD of differences 107): the same noise yields different starting squads under each model.

## E2: risk-aware captaincy (rejected for total points)

E1's v2 paths were rescored with only the captain and vice re-chosen (1,500 path-gameweeks). As a sanity check, the current rule reproduces 98.5% of recorded scores; the rest are solver near-ties.

| Captain rule | Season gain vs expected points (95% CI) |
|---|---|
| xP + 2·P(haul) | −10 (−30, +0) |
| xP + 5·P(haul) | −11 (−31, −0) |
| xP + 10·P(haul) | −6 (−29, +8) |
| xP + 20·P(haul) | −6 (−30, +9) |
| highest p90 | −6 (−31, +12) |

For **total points**, the highest-expected-points captain is optimal (as theory says), and upside rules only add noise. Upside matters for **rank**, which depends on ownership. That needs effective ownership in the simulation, and becomes a separate experiment (E2b).

## E4a: chip timing (triple captain, bench boost, free hit)

The 2025-26 rules: each chip once per half, one chip per gameweek. A rule plays the chip at the first gameweek its signal reaches T, otherwise in the half's last gameweek. Points are the chip's extra points per season, on the 60 v2 paths (30 per season).

| Chip | Last GW of half (no rule) | Best rule | Gain of rule | Hindsight (upper bound) |
|---|---|---|---|---|
| Triple captain | +12.7 | captain xP ≥ 10 | **+24.3** | +38.0 |
| Bench boost | +9.4 | bench xP ≥ 10–12 | **+19.2** | +42.5 |
| Free hit | +42.0 | best one-week XI ≥ 20 xP above current | +42.5 | +75.8 |
| **All three** (one per GW; triple captain moves on a clash) | | | **+76.8 per season** | |

- **The rules rediscover known FPL wisdom.** A captain xP ≥ 10 and a bench xP ≥ 10 happen essentially only in **double gameweeks**, so "triple captain and bench boost in a double gameweek" is worth about +10 pts each over playing them blindly.
- **The free hit is worth about 42 points wherever it's played.** A fresh optimal one-week squad beats a squad built up over months by about 20 points, so its timing barely matters here. It's limited by the 100.0m budget approximation, which is conservative because real squad values rise.
- **Caveat 1: the intervals are too narrow, so they aren't shown.** The paths share the same gameweeks, so in each season-half there is effectively one decision. Two seasons give 4 halves.
- **Caveat 2: the thresholds are chosen and scored on the same paths, so the gains are optimistic.** They need confirming on the held-out seasons before anyone relies on them.
- **Wildcard:** E4b, which needs new paths.

## E5: position calibration (rejected)

Walk-forward isotonic maps (predicted → actual), refitted every 4 gameweeks on earlier predictions only, with 2020-21 as warm-up. Compared with raw v2 on 2021-23:

| Variant | Δ regulars MAE (95% CI) | Δ captain | Δ top-11 |
|---|---|---|---|
| per position | +0.028 (+0.024, +0.032) | +0.0 (−1.1, +1.2) | −0.8 (−2.3, +0.5) |
| one map | +0.027 (+0.024, +0.031) | +0.7 (−0.0, +1.7) | −0.9 (−2.4, +0.4) |
| half strength, per position | +0.013 (+0.011, +0.015) | −0.3 | −1.2 (−2.1, −0.3) |

- **MAE punishes mean calibration:** points are right-skewed (mean > median), so moving predictions toward the conditional mean raises absolute error. The decision metrics are the relevant test, and they show no gain.
- **The isotonic maps overfit the sparse top end:** high-predicted goalkeepers went 6.28 → 7.89 against 5.86 actual.
- **The bias was smaller than first measured:** with true walk-forward predictions, high-predicted defenders run 5.73 vs 5.58 actual (−3%, not −5%).

## Tried and rejected

| Change | Tuning-season result | Decision |
|---|---|---|
| Price-band calibration layer (11.0m+ under-predicted ~5%) | regulars MAE 1.872 vs 1.870; captain 7.17 vs 7.24; top-11 61.5 vs 62.0 | Rejected: no gain |
