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

**Re-scored after the 2022-23 zero-xG fix** (E8). This is a data correction, so the held-out seasons are scored once more:

| | regulars MAE | Δ vs zero-xG data (95% CI) | Δ captain pts / GW | Δ top-11 pts / GW |
|---|---|---|---|---|
| v1 | 1.7873 | −0.004 (−0.006, −0.002) | −0.0 | +0.1 |
| **v2** | **1.7474** | −0.002 (−0.004, +0.001) | +0.1 (−0.6, +0.9) | −0.1 (−1.4, +1.3) |
| v2 − v1, both fixed | | **−0.040 (−0.045, −0.035)** | +0.7 (−0.3, +1.8) | +1.3 (−1.0, +3.5) |

The fix is neutral on the held-out seasons, though worth +58 points per season in the tuning-season simulation. On the tuning seasons, 2022-23 was predicted *while* its own zero-xG weeks were the most recent training data. In 2023-26 those weeks are only part of one season in a 4-season training window. The fix is kept: it's a correction with no held-out cost.

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
- **Wildcard:** see E4b.

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

## E7: hit cost and planning horizon (rejected)

These are optimiser settings, played on v2's E1 paths. The baseline uses a hit penalty of 4 (the true cost) and a 3-gameweek horizon. Each result is vs v2 with 30 paths per season and an unpaired bootstrap.

| Setting | Δ points per season (95% CI) | Hits per season |
|---|---|---|
| Hit penalty 5 when deciding (scoring still −4) | +9.4 (−20.2, +38.6) | 20.4 |
| Hit penalty 6 | +2.7 (−31.2, +37.2) | 12.1 |
| Horizon 5 gameweeks | **−113.4 (−142.5, −84.7)** | 64.6 |

- **A dearer hit isn't measurably better.** It halves the hits, but the points stay within noise. The true cost (4) stays.
- **Horizon 5 is clearly worse.** Predictions 4–5 weeks out are flat (form features don't extend that far), so the planner chases small far-off gains with hits now. Part of the gap may come from the larger MIP hitting its 30 s time limit under heavy CPU load.
- **Bug found and fixed:** future free transfers were only upper-bounded in the MIP. That was exact at the optimum but not in time-limited solutions, and one horizon-5 path broke the rules (13 transfers, 5 free, 12 hits). The rollover is now exact: `ft[t+1] = min(ft[t] − used + earned, 5)`, modelled with a binary.

## E4b: wildcard timing

The 2025-26 rule is one wildcard per half. In a wildcard week the squad is rebuilt freely: no hits, banked free transfers are kept, and that week earns no +1. Fixed-week rules play the wildcard in the given gameweek of each half. Threshold rules play it when rebuilding beats the normal plan by at least T expected points over the horizon. Each result is vs v2 without a wildcard, with 20 paths per season.

| Rule | Δ points per season (95% CI) | Hits per season |
|---|---|---|
| GW 8 and 24 | +33.0 (+6.4, +59.8) | 26.5 |
| GW 12 and 28 | +55.9 (+22.9, +89.6) | 31.2 |
| GW 16 and 32 | +39.2 (+12.5, +65.5) | 29.3 |
| rebuild gain ≥ 10 xP | +30.4 (+2.7, +57.8) | 30.7 |
| **rebuild gain ≥ 20 xP** | **+71.6 (+39.9, +101.8)** | 27.1 |

- **Every rule beats no wildcard.** Two free rebuilds a season are worth 30–70 points.
- **Waiting for a large gain (≥ 20 xP) works best in-sample.** A low threshold spends the chip on the first modest opportunity. The rules overlap within their intervals and the threshold was picked on these paths. **Held out (2023-26), it's only +10.2 (−12.5, +32.3)**; see the rank section.
- The no-wildcard baseline paths were played before the free-transfer fix (E7). That makes a difference only for time-limited solutions.

## E8: where does v2's error come from?

On the regulars of 2021-23 (26,060 player-gameweeks, MAE 1.806, MSE 7.65), each component's points error splits into a **minutes** part (predicted minus predicted-rescaled-to-actual-minutes) and a **returns** part (the rest). The squared error then decomposes exactly as `MSE = Σ cov(e_part, e)` (`fpl_models.e8`).

| Part | Share of MSE |
|---|---|
| goals (given minutes) | 31.9% |
| bonus | 15.0% |
| clean sheets | 14.4% |
| appearance (minutes class) | 11.8% |
| assists | 11.0% |
| other minutes parts (clean sheet, goals, bonus, assists) | 10.9% |
| conceding, cards, saves | 4.8% |
| other (penalties, own goals, approximations) | 1.8% |
| **minutes overall / returns overall** | **21.0% / 77.2%** |

By segment:

| Segment | Share of rows | Share of abs error | MAE | Bias (pred − actual) |
|---|---|---|---|---|
| new (< 6 GWs of history) | 3.4% | 2.7% | 1.43 | −0.15 |
| premium (≥ 9.0m) | 2.7% | 5.1% | 3.44 | **−0.80** |
| predicted ≥ 5 | 3.8% | 7.9% | 3.75 | +0.28 |
| GKP / DEF / MID / FWD | 6 / 35 / 46 / 13% | 8 / 38 / 41 / 14% | 2.17 / 1.95 / 1.62 / 1.92 | −0.17 / −0.04 / −0.22 / −0.23 |

- **For regulars, most error is in the returns, not the minutes.** Goals, bonus and clean sheets are mostly irreducible noise, but part of it is a fixable bias (below). Minutes dominate over *all* players (rotation and injuries), but regulars are what the optimiser picks.
- **Players new to the league are not a problem.** Their error share is below their row share, so a hierarchical prior for low-history players (planned item 8) isn't justified.
- **Premium players are under-predicted by 0.8 points per gameweek, and 0.9 at ≥ 10.0m.** Of that, 0.50 is goals and 0.15 is assists. Every one of the ten largest season under-predictions is an elite attacker: Haaland 2022-23 had 12.3 expected goals and scored 36; Kane had 10.7 and scored 30.
- **A data bug explains part of it (confirmed below by re-running).** FPL first published expected stats at 2022-23 GW16, and the vaastav history carries **0.0 xG for every player in GW1–15**, not null. With `denoise_weight = 1.0`, v2 learned 15 weeks of "nobody was expected to score", and the xG-per-90 features were deflated the same way. Kane's FPL xG for 2022-23 sums to only 12.0 for that reason. **Fix:** `stg_vaastav__player_matches` nulls the expected stats of any gameweek with no xG published league-wide. The dbt test `assert_expected_stats_missing_not_zero` guards it. The training targets then fall back to actual goals for those weeks. This affects every result trained on 2022-23: the tuning season itself, and the held-out seasons through their 4-season window. The production baseline is re-scored on the fixed data below.
- **Re-recorded on the fixed data, the premium bias halves and the overall goals bias almost vanishes:**

  | | Zero-xG data | Fixed |
  |---|---|---|
  | Premium bias (pts/GW) | −0.80 | **−0.37** |
  | Goals bias, all regulars | −0.089 | **−0.016** |
  | MSE | 7.65 | **7.52** |
  | MAE | 1.806 | 1.826 |

  Squared error falls but absolute error rises. That's the skew effect from E5: points are right-skewed, so better-calibrated, higher predictions cost MAE. The decision metrics and season simulation decide whether the fix helps decisions.
- **Two shrinkage settings are also tested (E9)** against the remaining premium bias: `min_samples_leaf = 100` was chosen on MAE, which barely sees the tail.
- "Back from absence" doesn't occur among regulars (they play by definition), so it isn't shown.

## Fixed-data baseline and E3a: last season's penalty taking (rejected)

After the zero-xG fix (E8), production (`r4_xg_1`) was re-scored on the tuning seasons. The comparison below is fixed vs zero-xG data, per gameweek, with a paired bootstrap:

| | Δ regulars MAE | Δ captain pts/GW | Δ top-11 pts/GW |
|---|---|---|---|
| Fixed vs zero-xG data | +0.020 (+0.010, +0.031) | +0.32 (−0.67, +1.43) | +1.39 (−0.96, +3.92) |

MAE rises, as E8 predicted for less-shrunk, better-calibrated predictions. The decision metrics lean positive but are within noise. A data fix doesn't need to "win" to be adopted, but season simulation checks it doesn't hurt decisions. **It clearly helps: +58.4 points per season (95% CI +33.1, +83.7)**, with 30 paths per season and v2 prepared on fixed vs zero-xG data. That's about as large as the whole v1 → v2 upgrade (+59). Under-predicting the elite attackers was costing real picks and captaincies.

**E3a features** (`fpl_models.penalties`). Understat publishes penalty xG only as season totals (xG − npxG), so a fixture in season S uses season S−1:
- `penalty_xg_prev90`: the player's penalty xG per 90 last season, at any club, with at least 270 minutes;
- `penalty_share_prev`: his share of his club's penalty xG last season, if he's still at the same club (otherwise 0).

Understat has no FPL ID, so players are matched by name within club and season. A token-subset rule handles FPL's legal names ("Bruno Miguel Borges Fernandes"), and ambiguous names are skipped. The match covers 96–100% of penalty xG in every season from 2016-17.

Against production on the fixed data:

| Variant | Δ regulars MAE | Δ captain pts/GW | Δ top-11 pts/GW |
|---|---|---|---|
| both features | −0.0006 (−0.0021, +0.0010) | +0.60 (−0.31, +1.57) | **+1.50 (+0.03, +3.01)** |
| share only | −0.0014 (−0.0028, 0.0000) | +0.66 (−0.36, +1.71) | +1.21 (−0.27, +2.84) |

- **This is the first feature since the crowd signal with a significant decision gain.** Penalties are rare, so MAE barely moves, but they concentrate on the few players the optimiser picks and captains.
- **Season simulation rejects it: −21.4 points per season (95% CI −45.7, +3.2)** vs production on the fixed data, with 30 paths per season. The top-11 gain doesn't survive the squad constraints, transfers and captaincy, so penalties stay out of production. Last season's penalty share is a stale signal (takers change), and the optimiser may over-trust it.
- Mid-season changes of penalty taker aren't seen. Live bootstrap data has `penalties_order`, and the daily snapshots will allow a within-season version later.

## E9: less shrinkage for the elite tail (rejected)

| Variant | Δ regulars MAE | Δ captain pts/GW | Δ top-11 pts/GW |
|---|---|---|---|
| `min_samples_leaf` 30 (from 100) | −0.0004 (−0.0022, +0.0014) | +0.33 (−0.68, +1.25) | +0.69 (−1.24, +2.75) |
| `max_iter` 800 (from 400) | −0.0001 | 0.00 | −0.16 (−0.48, 0.00) |

- **Neither helps.** After the xG fix, the remaining premium bias isn't a leaf-size effect.
- **More iterations do nothing, because training already stops early.** `HistGradientBoosting*` enable early stopping automatically above 10,000 rows (`early_stopping="auto"`: a 10% validation split and 10 rounds of patience). A production fit on 2018-22 stops after 234 iterations for minutes and 134–176 for the count models (goals 146, assists 134), so `max_iter` and part of `learning_rate` were never the binding settings. That's worth knowing for any future hyperparameter search.

## E4a out of sample: chip thresholds on 2018-19 and 2020-21

E4a's thresholds were fixed before this run (`e4.PRE_REGISTERED`) and scored on two seasons never used for tuning: 20 v2 paths each, the same 2025-26 chip rules, and no xG in either season. 2019-20 is skipped because of its COVID gameweek numbering. "Blind" plays the chip in the half's last gameweek.

| Chip | Blind | Pre-registered rule | Rule − blind | Tuning seasons: rule − blind |
|---|---|---|---|---|
| Triple captain (captain xP ≥ 10) | +19.4 | +18.9 | **−0.5** | +11.6 |
| Bench boost (bench xP ≥ 10) | +11.3 | +17.7 | **+6.4** | +9.8 |
| Free hit (≥ 20 xP above current XI) | +27.3 | +27.3 | **0.0** | +0.5 |
| All three | | +51.4 per season | | +76.8 |

- **The bench-boost rule holds up out of sample (+6). The triple-captain rule doesn't (−0.5).** It wasn't diagnosed why. One possibility is that the end-of-half gameweeks in these seasons were themselves strong captain weeks. The free-hit rule never beat blind in either period: the chip is worth a fresh squad wherever it's played.
- **What to use:** play the bench boost in a double gameweek with bench xP ≥ 10. The triple-captain threshold isn't supported; keep it as a weak preference for double gameweeks, not a rule. The free hit can be played whenever convenient (blank gameweeks in practice, which this budget-approximate simulator doesn't model).
- The intervals (path bootstrap) are omitted for the same reason as in E4a: each season-half is effectively one decision.

## How good is it? Gameweek level and estimated overall rank

**Per gameweek** (2021-23, walk-forward v2 on the fixed data). Picks are made from every player with a fixture, with no budget or squad limits:

| Per gameweek | v2 | Form (`points_ewm`) | Hindsight |
|---|---|---|---|
| Correlation, predicted vs actual | **0.58** | 0.51 | 1 |
| Points of the predicted top 11 | **63.4** | 60.9 | 142.9 |
| Points of the predicted top 50 | **231** | 208 | |
| Actual top-20 scorers in the predicted top 20 | **22%** | 18% | 100% |
| Top-1 pick (captain) | 7.4 | 8.3 | 18.2 |

- R² over all player-gameweeks is 0.35: most of a gameweek's points are unpredictable.
- The captain gap to form is noise: −0.95 pts/GW, 95% CI (−2.6, +0.7), over 75 gameweeks.

**Estimated overall rank.** Simulated season totals (after hits, no chips) are mapped to ranks using 700 random managers' past seasons from the public FPL API: final points and rank, with a local log-rank fit. The fit agrees with published cutoffs, e.g. a 2021-22 top-100k cutoff of 2,505.

"+ chips" adds the **out-of-sample** chip values: the wildcard rule's held-out +10 per season (below), plus about +26 for one triple captain, bench boost and free hit (E4a out of sample). 2025-26 allowed two of each, so it gets about +51.

| Season | Data | Paths | Simulated | Rank, no chips | + chips | Rank with chips | Managers |
|---|---|---|---|---|---|---|---|
| 2018-19 | never tuned on | 20 | 2,174 | ~374k (6%) | 2,210 | **~233k (3.7%)** | 6.3M |
| 2020-21 | never tuned on | 20 | 2,207 | ~865k (11%) | 2,243 | **~641k (7.8%)** | 8.2M |
| 2021-22 | tuning | 40 | 2,126 | ~1.7M (20%) | 2,162 | **~1.4M (16.5%)** | 8.8M |
| 2022-23 | tuning | 40 | 2,204 | ~2.7M (24%) | 2,240 | **~2.2M (19.8%)** | 11.1M |
| 2023-24 | held out | 20 | 2,327 | ~811k (7.5%) | 2,363 | **~559k (5.2%)** | 10.8M |
| 2024-25 | held out | 20 | 2,216 | ~2.6M (23%) | 2,252 | **~2.1M (18.5%)** | 11.4M |
| 2025-26 | held out | 20 | 2,028 | ~3.5M (27%) | 2,089 | **~2.5M (18.9%)** | 13.0M |

- **The model is a solid top-20% manager, typically top 5–20%.** It isn't a top-100k (top 1%) manager.
- Ranks are ±20%, and path-to-path luck spans 2–3× in rank (e.g. 2023-24 paths range from ~240k to ~1.5M).
- The field got stronger: the median manager rose from 2,006 (2018-19) to 2,229 (2022-23).
- The gap to the top 100k is 150–250 points. The likely causes are no team news (minutes are 22% of the error), no in-week price timing (E10) and ignoring ownership.

**E4b held out: the wildcard rule is much weaker out of sample.** "Wildcard when the rebuild gains ≥ 20 xP" scores **+10.2 per season (95% CI −12.5, +32.3)** on 2023-26, with 20 paths per season, vs +72 on the tuning seasons where its threshold was chosen. It hurt in 2023-24 (−21) and helped in 2024-25 (+30) and 2025-26 (+21). Two free rebuilds a season are worth less than they looked: a squad that tracks the model's best players week by week rarely needs one.

## E10: price-aware transfers (promising, not adopted)

`fpl_models.prices`: P(rise) and P(fall) by the next deadline, isotonic in the deadline's net transfer share and fitted walk-forward per season. It's well calibrated, e.g. predicted P(rise) 0.54 vs 0.55 actual, and P(fall) 0.51 vs 0.50, in the top 1%.
- Holding a player through the gameweek is worth `0.5·P(rise) − P(fall)` tenths of selling value: FPL keeps half of a rise and all of a fall.
- The optimiser values that at `Settings.value_weight` points per 0.1m. The simulator already moves prices and the bank for real, so any gain is realised, not assumed.

| Value weight | Δ points per season vs v2 (fixed), 95% CI | Hits per season | GW19 squad price |
|---|---|---|---|
| 0 (production) | | 36.6 | £99.0m / £99.3m |
| 0.5 | +20.8 (−4.3, +47.2) | 38.5 | £99.2m / £99.4m |
| 1 | −28.5 (−52.8, −4.5) | 43.3 | £99.6m / £99.3m |
| 2 | +10.2 (−15.1, +36.6) | 49.9 | £99.9m / £100.2m |

Results are with 20 paths per season. The two GW19 squad prices are for 2021-22 / 2022-23.

With 40 paths per season, **weight 0.5 gives +18.4 points per season (95% CI −2.6, +40.2)**, with 38.2 hits.

- **Not adopted (yet):** the gain is suggestive, not significant, and it's the best of three weights (optimistic). Production keeps `value_weight = 0`.
- **The mechanism is real but small at gameweek granularity:** squads end up £0.1–0.9m more valuable by GW19. Real team-value gains (£2–5m a season) come from transferring *within* the week, ahead of an overnight rise or fall, and gameweek-level history can't simulate that. The daily snapshots (from 2026-10-17) will make an in-week version testable, and the price model is ready for it.

## E12: availability, news and set-piece order, point in time

FPL exposes status, news, chance of playing and set-piece order only as current state. **vaastav's repository recommits `players_raw.csv` (a bootstrap-static dump) about once a gameweek, so its history is an archive** (`fpl_models.availability`):
- a blob-less clone gives 263 snapshots for 2019-20 to 2025-26;
- each gameweek takes the latest snapshot *committed* before its deadline, which can't leak;
- set-piece orders exist from 2020-21.

**Coverage, the main limitation:**

| Season | Snapshot ≤ 48 h before the deadline | Median age |
|---|---|---|
| 2019-20 / 2020-21 | 37% / 42% | 78 h / 53 h |
| **2021-22 / 2022-23** (tuning) | **50% / 54%** | **47 h / 43 h** |
| 2023-24 / 2024-25 | 58% / 53% | 44 h / 39 h |
| 2025-26 | 16% | 660 h (few commits) |

vaastav scrapes after a gameweek, not before the next, so the archive misses most late team news (press conferences, final training). The live tool reads the status minutes before planning.

**Results**, on 2021-23 per gameweek with a paired bootstrap. "Multiplier" is what the live tool already does: × chance of playing, and 0 for players who left. It was never backtested before.

| Variant | Against | Δ regulars MAE | Δ captain pts/GW | Δ top-11 pts/GW |
|---|---|---|---|---|
| **production × multiplier** | production | **−0.056 (−0.063, −0.049)** | +0.29 (0.00, +0.87) | +0.05 (−0.20, +0.35) |
| availability as features | production × multiplier | +0.033 (+0.030, +0.036) | −0.06 (−1.23, +1.05) | +0.68 (−0.63, +2.07) |
| availability features × multiplier | production × multiplier | +0.017 (+0.014, +0.021) | +0.08 (−0.95, +1.12) | +0.84 (−0.47, +2.20) |
| penalty order × multiplier | production × multiplier | −0.001 (−0.003, +0.001) | −0.67 (−2.08, +0.71) | +1.42 (−0.31, +3.40) |
| both × multiplier | production × multiplier | +0.017 (+0.013, +0.020) | −0.72 (−1.87, +0.32) | +1.44 (−0.52, +3.45) |

- **The live multiplier is validated, and it's large.** It's the biggest MAE gain measured here, bigger than v2's whole held-out gain over v1, and the captain picks improve. The live tool is right to apply it.
- **Learning from the archive doesn't beat it, so the features are rejected.** With snapshots ~45 h stale, the model learns a blurred version of what the fresh flag gives directly. A model only pays off with deadline-time status, which the daily snapshots (from 2026-10-17) now collect.
- **Within-season penalty order: not significant, not adopted.** Its +1.4 top-11 points per gameweek is the same pattern as E3a, which then failed season simulation. Neither variant passed the per-gameweek gate, so neither got a season-simulation test.

**In season points, the multiplier is worth +15.6 per season (95% CI −9.4, +40.6)** (`e1 --availability`, production on fixed data, 30 paths per tuning season vs 40). It's positive but not significant. With archive snapshots ~45 h stale, this is a lower bound on what the fresh live flag is worth. It also means the earlier rank estimates (which simulate without it) are slightly pessimistic for the live tool.

## Shadow team (live season, 2026-27)

The honest scoreboard. The planner manages its own team through the live season, with the same predictions and optimiser a user gets (`fpl_models.shadow`, Dagster asset `gold/shadow/state`).

**Retro start, GW1–6.** The season simulator ran on 2026-27 with production (v2, fixed xG data), decided blind: 30 seeded paths plus one noise-free path. GW7 was still provisional in the warehouse, so it's not scored. These were compared with 236 random managers active since GW1 (public FPL API, current-season history):

| GW | Sampled median | Shadow (30 paths, mean) |
|---|---|---|
| 1 | 48 | 44.0 |
| 2 | 78 | 64.4 |
| 3 | 50 | 37.9 |
| 4 | 69 | 68.1 |
| 5 | 47 | 47.7 |
| 6 | 57.5 | 40.8 |
| **Total GW1–6** | **353** (p10 293, p90 411) | **302.9** (sd 31, 255–367); noise-free path 300 |

- **The shadow is ~50 points behind the median after six gameweeks**, around rank 8.1M. That's far worse than the simulated full seasons (typically top 5–20%).
- **The data isn't the problem.** 2026-27 features have no nulls and similar means to 2025-26. Per-gameweek prediction correlation is 0.40–0.60, as good as 2025-26's 0.42–0.62.
- **The decisions are.** Captains returned 3.5 base points per gameweek (5.5 in 2025-26 GW1–6), and ~0.7 starters a week didn't play. The early season is where humans' team news and pre-season knowledge (new signings, roles) matter most, and where the model knows least (`gws_in_history` is small for everyone).
- A strong human manager in the same period: 442 through GW6 (rank 178k), 518 through GW7 (rank 113k).

**Live from GW8.** The state is initialised from the noise-free path's GW6 squad, with real purchase prices and bank replayed. GW7 is held: no transfers, +1 free transfer, unscored. The daily `warehouse_build` run then plans each next gameweek and scores it once final:

    uv run python -m fpl_models.shadow init --from-path data/experiments/e1/paths/v2fixn0_2026-27_00.json
    uv run python -m fpl_models.shadow run    # or the Dagster asset gold/shadow/state

**Start-noise bias check.** Season paths start from a squad built on noisy expected points (`start_noise = 0.3`), so the paths differ. Does that noise cost points? With `start_noise = 0.1` on the tuning seasons (30 paths per season): **+15.9 per season (95% CI −6.3, +38.4)** vs 0.3. A small pessimistic bias of up to ~16 points is possible but not established. The rank estimates above may be slightly low, by well under one rank band.

## E13: in-week price timing (measured on 2023-26; no production change)

**Data.** E10 had prices only per gameweek, and FPL changes prices overnight. The Wayback Machine has archived the live player list (`bootstrap-static`) about twice a day since 2023 (`fpl_models.e13`).
- One capture a day was kept, after that night's price changes, from July 2023 to May 2026: 994 captures, of which 966 fall inside a transfer window.
- That gives 515,884 player pairs from consecutive days in the same window.
- The archive refuses connections above ~15 requests a minute, so downloads are sequential with backoff.
- A price change on a given player-day is rare: rises 0.2%, falls 0.5%.

**1. Can the next night's change be predicted from the window so far?** Barely, from public data. The features are net transfers since the deadline, as a share of all activity and per point of ownership. The model is a monotone gradient-boosted classifier, chosen over an unconstrained one by Brier score on 2024-25 (fit on 2023-24). 2025-26 is the test.

| | 2024-25 (validation) | 2025-26 (test) |
|---|---|---|
| P(rise) skill vs base rate | −0.04 | +0.06 |
| P(rise) predicted / actual, top bin | 0.32 / 0.11 | 0.23 / 0.15 |
| P(fall) skill | +0.01 | +0.02 |

The predicted probabilities never reach 0.5. FPL's algorithm works on net transfers since each player's *last price change*, with thresholds and protection flags it doesn't publish, and a cumulative since-deadline count can't track that. Adding the latest day's flow and changes-so-far made the model overconfident out of sample (0.98 predicted vs 0.20 actual). That variant was also chosen after looking at both seasons, so it's reported here and rejected.

**2. What is timing worth?** I took the 3,112 transfers v2's held-out season-simulation paths actually made (20 per season, 2023-26) and compared selling value at the deadline under different execution times. Buying before a rise keeps half of it (FPL's selling rule); selling before a fall saves all of it:

| Execution | £m per season (95% CI) |
|---|---|
| at the deadline (simulator, baseline) | 0 |
| **right after the previous deadline** | **+1.29 (+1.21, +1.37)** |
| when the predictor says P(rise in) or P(fall out) > 0.5 | 0.00 (never triggers) |
| best capture in hindsight (upper bound) | +1.41 (+1.34, +1.48) |

- **v2 already buys risers and sells fallers.** The players it moves are the ones the crowd is moving, so executing the same transfers early captures 91% of the hindsight value.
- **That is about +5 points a season, before costs.** Extra budget measured directly: +£2.0m from GW2 is worth +16.9 points per season (95% CI −3.0, +37.1) on the held-out seasons (`e1 play --extra-bank 20`), about 8 points per £1m. Timing gains build up through the season, averaging about £0.65m, so roughly +5 points.
- **The cost is news.** About 1.6 transfers per season buy a player who was available at the window start but flagged by the deadline. At a few points each, that cancels most of the gain.
- **The decisions are the deadline's.** The deadline plan already saw the whole window's crowd signal, so moving early with less information would change some decisions. This measures the value of execution timing, not of planning earlier.

**Decision.** No automatic early transfers. For the live tool, show a price-watch next to each planned transfer: crowd pressure this window, the E10 gameweek-level P(rise)/P(fall), and whether the player is fit. When the buy is rising and the player is fit, suggest making the transfer early, and leave the news risk to the manager.

## Tried and rejected

| Change | Tuning-season result | Decision |
|---|---|---|
| Price-band calibration layer (11.0m+ under-predicted ~5%) | regulars MAE 1.872 vs 1.870; captain 7.17 vs 7.24; top-11 61.5 vs 62.0 | Rejected: no gain |
