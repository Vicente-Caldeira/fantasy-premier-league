# Model architecture

How expected points are produced, end to end. Exact numbers come from the code: `packages/fpl_models/`.

```
fct_matches ──► team-strength model ──► each fixture's expected goals (λ) per side
                                                │
fct_player_matches ──► player features ◄────────┘
                           │
                           ▼
              seven component models (gradient-boosted trees)
                           │
                           ▼
         expected events ──► FPL scoring rules ──► expected points ─┐
                                                                     ├─ 50/50 ──► optimiser
   player features ──► Kumo Tabular (pretrained, in-context) ───────┘
```

## 1. Team strength (`team_strength.py`)

For every fixture, it predicts each side's expected goals (λ).

- **Priced fixtures:** λ is solved so that an independent-Poisson score model reproduces the bookmakers' margin-free **opening** odds (home/draw/away + over/under 2.5). Closing odds are struck after the FPL deadline, so they're never used.
- **Fixtures not yet priced:** a Poisson attack/defence rating model.
  - log λ_home = μ + home advantage + attack(home) − defence(away) (+ promoted-team effects).
  - It's fitted on a 50/50 blend of goals and xG, over 3 years of matches weighted with a **180-day half-life**, using L2 shrinkage and L-BFGS.
- **Derived:** P(clean sheet) = e^(−λ_opponent).

## 2. Player features (`player_features.py`)

One row per player per fixture. Every rolling input uses **previous gameweeks only**: stats are summed per gameweek and shifted by one, so both legs of a double gameweek see the same pre-deadline state. Tests prove this by deleting later data and checking the inputs are unchanged.

| Group | Inputs |
|---|---|
| Availability | minutes per fixture over the last 1 / 3 / 6 gameweeks, appearance rate, 60+-minute rate, start rate, minutes over the last 10, gameweeks since last appearance |
| Output rates | per-90 goals, xG, assists, xA, bonus, BPS, saves, yellow cards, CBIT, CBIRT, points; exponentially weighted with an **8-gameweek half-life** |
| Form | exponentially weighted points per gameweek |
| Match context | home/away, the team's and the opponent's λ (from team strength), P(clean sheet), fixtures this gameweek |
| Player | position, price |
| Schedule (v2) | the team's and the opponent's rest days since their previous league match, and league matches in the previous 14 days |
| Crowd (v2) | the fraction of managers who own him, and his share of all net transfer activity this gameweek |

**Recency:** there's no season boundary. Older games fade: 8 gameweeks ago counts ½, 16 counts ¼, and a full season ago about 4%. Early in a season last season dominates, and within a couple of months the current one does.

**Crowd signal:** before each deadline, millions of managers react to team news, injuries and press conferences. The model doesn't otherwise see that information. A player's net transfers ÷ everyone's |net transfers| is the same whether the window is complete or half over. That matters because live, FPL's counters hold the window still open. It exists only for the next deadline, so later gameweeks use a **planning model** trained without crowd features.

**Missing data:** xG exists only from 2022-23, and CBIT only for 2016-19 and 2025+. Missing values stay missing (not 0), and the trees handle them natively.

## 3. Component models (`player_model.py`)

scikit-learn **HistGradientBoosting**: gradient-boosted decision trees (the same family as XGBoost and LightGBM), with position as a categorical input. v2 settings (`player_model.PRODUCTION`) were chosen on 2021-23 with paired-bootstrap tests:
- 400 trees, learning rate 0.03;
- at least 100 rows per leaf, L2 regularisation 1.0;
- training rows weighted with a 180-day half-life.

| Model | Target | Loss |
|---|---|---|
| minutes | 0 / 1–59 / 60+ | multinomial log-loss |
| goals, assists, bonus, yellow cards | counts per fixture | Poisson |
| saves | counts (goalkeepers only) | Poisson |
| defensive contribution | reached 10 CBIT (DEF) / 12 CBIRT (MID, FWD) | binary log-loss |

- **Training data:** the latest 4 seasons, with recent rows weighted more (above). The defensive-contribution model uses every season that has CBIT data.
- **Targets (v2):** goals and assists models learn from **xG and xA** where FPL has them (2022-23+). The expected value is the same, with far less noise than whether a 0.3 xG chance went in. This was the single largest MAE gain.

## 3b. Kumo Tabular blend (`kumo.py`, E19)

Live expected points are half the component model's and half **NVIDIA Kumo Tabular**'s: a pretrained tabular foundation model (the small, 28M-parameter regressor) that is never trained on our data.
- **In-context learning:** at each run it reads a *context* of the newest 50,000 labelled fixture rows (about two seasons) with the same 34 features, and predicts each upcoming fixture's total points in one forward pass. The mean of its 999 predicted quantiles is the expected value.
- **Label:** actual points, not the xG-denoised targets, so finishing skill stays in. It rates £10m+ players higher than production on the held-out seasons (4.23 vs 4.06; actual 4.43) and live, but not on the tuning seasons.
- **Same rules as production:** later gameweeks use a context without the crowd features, and FPL's availability flags scale Kumo's points too. The upside distributions (p90, P(haul)) stay the simulator's.
- **Why a blend:** Kumo alone has a worse MAE and correlates 0.97 with production. The blend gains from diversity: +31 points a season in the held-out season simulation (95% CI +14, +48).

## 4. From events to points

The current rules from `fpl_core.scoring`, applied per fixture and summed over the gameweek:

```
points =  P(1–59)·1 + P(60+)·2
        + E[goals]·(10 GKP | 6 DEF | 5 MID | 4 FWD) + E[assists]·3
        + P(60+)·e^(−λ_opp)·(4 GKP/DEF | 1 MID)
        − P(60+)·E[⌊G/2⌋] (GKP/DEF; G ~ Poisson(λ_opp))
        + E[⌊saves/3⌋] (GKP) + 2·P(defensive contribution) (from 2025-26)
        + E[bonus] − E[yellow cards]
```

Live predictions also apply FPL's availability flags, which have no history and so can't be learned: `chance_of_playing_next_round` scales the next gameweek, and `u` (left the club) zeroes all gameweeks.

**Distributions:** `simulator.py` draws 4,000 coherent matches per fixture:
- team goals = the sum of its players' goals + own goals;
- assists rise and fall with the team's goals;
- goals conceded and clean sheets come from the opponent's simulated goals;
- bonus is tied to returns;
- attacking output is overdispersed.

This gives p10/p90 and P(haul ≥10) per player-gameweek. Means are calibrated, but hauls are still under-predicted by about 20% (see the experiment log).

## 5. Training and evaluation

- **Backtest** (`backtest.py`): walk forward through each gameweek of 2023-24 → 2025-26. Models are retrained every 4 gameweeks on data before the deadline only. Models never receive the outcome columns they're asked to predict.
- **Metrics:** MAE on regular players (the primary metric), captain points and top-11 points against hindsight, and rank correlation.
- **Tuning discipline:** settings may only be chosen on 2021-22 and 2022-23. The 2023-26 evaluation seasons are scored once per accepted change.
- **Season simulation** (`season_sim.py`): the end-to-end acceptance test. It manages a team through a whole past season, deciding blind at each deadline with the optimiser and scoring the real gameweek (auto-subs, captain/vice, hits, price changes). v2 scored +292 points over v1 across 2021-23.
- **Live:** each daily run retrains on all history (about 50 s), fits Kumo's two contexts on CPU (about 2.5 min on an M5 Pro), and predicts the next 5 gameweeks. Trained models aren't persisted yet (Phase 4).

## 6. Known weaknesses

See `docs/model-experiments.md` for the work list and results.

- **Only coarse tuning so far:** a one-at-a-time grid, no joint search. The 8-gameweek feature half-life and 4-season window were tested and kept.
- **European and cup fixtures aren't in the schedule features yet.**
- **Premium players are under-predicted.** On the held-out seasons, £10m+ players average 4.06 predicted vs 4.43 actual. The Kumo blend narrows that to 4.15 (Kumo alone: 4.23). A price-band calibration layer didn't help on the tuning seasons and was discarded.
- **The optimiser uses only the mean:** the simulator's distributions (upside, P(haul)) aren't used for captaincy or rank risk yet.
- **Rare events aren't modelled:** red cards, own goals and penalties.
