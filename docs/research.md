# FPL App: Research Notes

_Compiled 2026-10-17. The 2026/27 season is live, with GW7 current._

## 1. Scoring rules

The scoring values below come from the live FPL API (`bootstrap-static` → `game_config.scoring`).
Point values are the same as in 2025/26. The 2026/27 changes only affect how BPS is calculated.

| Event | GKP | DEF | MID | FWD | Notes |
|---|---|---|---|---|---|
| Played 1–59 min | 1 | 1 | 1 | 1 | |
| Played 60+ min | 2 | 2 | 2 | 2 | Replaces the 1-point value above |
| Goal | 10 | 6 | 5 | 4 | |
| Assist | 3 | 3 | 3 | 3 | Assist definition simplified in 25/26, which gave more assists |
| Clean sheet | 4 | 4 | 1 | 0 | Requires 60+ min, with no goal conceded while the player is on the pitch |
| Every 2 goals conceded | −1 | −1 | 0 | 0 | Only goals conceded while the player is on the pitch count |
| Every 3 saves | 1 | – | – | – | |
| Penalty saved | 5 | – | – | – | |
| Penalty missed | −2 | −2 | −2 | −2 | |
| Defensive contribution | – | 2 | 2 | 2 | DEF: 10+ CBIT. MID/FWD: 12+ CBIRT. Capped at 2 per match, with no partial points |
| Bonus | 1–3 | 1–3 | 1–3 | 1–3 | Top 3 BPS in the match get 3/2/1. Ties share the higher value |
| Yellow / Red card | −1 / −3 | | | | |
| Own goal | −2 | | | | |

CBIT = clearances + blocks + interceptions + tackles. CBIRT = CBIT + ball recoveries.

### What changed

**2025/26:**
- Defensive contributions (DefCon) were added.
- Assists were redefined.
- BPS values were re-tuned for GK saves, goal-line clearances, penalty goals and tackles.
- There are two sets of chips (WC, FH, TC, BB), one per half of the season. The first set expires at the GW19 deadline.
- The Assistant Manager chip was removed.
- Free transfers were topped up to 5 for AFCON in GW16.

**2026/27:**
- BPS:
  - Players are no longer docked BPS for being tackled.
  - CBIs now earn 1 BPS per 3, down from 1 per 2.
  - Saves are split into: inside the box = 3, other = 2, big chance saved = +1. Saves from outside the box no longer have their own category.
- Live points and ranks, plus projected bonus after 20 minutes.
- Scores lock at 09:00 UK the day after the last match.
- No AFCON top-up. You can still bank up to 5 free transfers.

### Points components to model, by position

- **All positions:** minutes (P(start), P(60+)), goals, assists, bonus, cards.
- **GKP:** clean sheet, goals conceded, saves, penalty saves. These are mostly team-defence and opponent-attack driven.
- **DEF:** clean sheet, goals conceded, DefCon (10 CBIT), plus attacking returns.
- **MID:** goals, assists, DefCon (12 CBIRT, relevant for defensive midfielders), and 1 point for a clean sheet.
- **FWD:** goals, assists, occasionally DefCon.

## 2. Data sources

| Source | What it gives | Cadence | Notes |
|---|---|---|---|
| **FPL API** (`fantasy.premierleague.com/api/`) | Players, prices, ownership, injury news, fixtures and FDR. Per-match history via `element-summary/{id}`: minutes, goals, assists, xG, xA, xGI, xGC, CBIT, tackles, recoveries, DefCon, saves, BPS, bonus | Live | The ground truth. No auth for public endpoints. Also `fixtures/`, `event/{gw}/live/`, `entry/{id}/` for your team |
| **olbauday/FPL-Elo-Insights** (GitHub) | Opta-like player match stats: touches, touches in the box, chances created, xG, xA, xGOT, shots, big chances, dribbles, passes, tackles, interceptions, recoveries, blocks, clearances, duels, GK xGOT faced and goals prevented. Also ClubElo team ratings | Twice daily | **This is the repo you were thinking of.** Has 2025/26 and 2026/27 data as CSV |
| **vaastav/Fantasy-Premier-League** (GitHub) | Historical per-GW FPL data from 2016/17 | Weekly updates stopped after 24/25. Now 3 updates per season | Best for multi-season training history. The `xP` column leaks post-match info, so don't use it as a feature |
| **Understat** | Shot-level xG and xA, team xG | Per match | Long history (2014+). Scrape with the `understat` or `soccerdata` Python packages |
| **football-data.co.uk** | Results, shots, HxG/AxG, and pre-match + closing odds for 1X2, O/U 2.5 and Asian handicap from several bookmakers | Weekly | Free CSV at `mmz4281/2627/E0.csv`. Upcoming fixtures with odds are in `fixtures.csv` |
| **The Odds API** / OddsPapi | Live odds for upcoming matches (h2h, totals, sometimes BTTS) | Live | Free tiers have request limits |
| **ClubElo** | Team strength ratings | Daily | Already merged into FPL-Elo-Insights |
| ~~FBref~~ | – | – | Lost its Opta advanced stats in January 2026. Don't build on it |

### Clean sheets from odds

Bookmakers rarely offer free clean-sheet odds, so derive them instead:
1. Take the 1X2 and O/U 2.5 odds and remove the bookmaker margin to get probabilities.
2. Fit home and away goal expectations (λ_h, λ_a) under a Poisson or Dixon–Coles model so that they reproduce those probabilities.
3. P(home clean sheet) = P(away scores 0) = e^(−λ_a). λ also gives the expected goals conceded and the expected team goals, which you then share out to players.

## 3. Modelling sketch

```
E[pts] = Σ over the GW's fixtures:
    minutes model:   P(plays), P(60+)            → appearance points
  + team λ_for × player goal share × goal pts
  + team λ_for × player assist share × 3
  + P(CS | 60+) × CS pts − E[floor(conceded / 2)]        (GKP/DEF)
  + P(CBIT ≥ 10 or CBIRT ≥ 12) × 2                       (vs opponent style)
  + E[saves] / 3, penalty saves                           (GKP)
  + E[bonus]  (BPS model, or a regression on the components above)
  − E[cards]
```

1. **Team layer:** fixture odds → λ_for / λ_against. Fall back to a team-strength model (xG for/against plus Elo) when odds are missing or the fixture is far out.
2. **Player layer:** shares of team xG/xA (shrunk toward a positional prior), minutes and start probability (from news, rotation and recent starts), and a DefCon distribution (Poisson or negative binomial on CBIT/CBIRT per 90).
3. **Decision layer:** multi-GW horizon, transfers and hits, a squad optimiser (integer programming with constraints of £100m, 3 per club and the formation), captaincy and chips.
