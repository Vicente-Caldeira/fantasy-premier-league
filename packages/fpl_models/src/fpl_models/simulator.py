"""Monte Carlo match simulator: points distributions, coherent within each fixture.

Expected points need no simulation (a sum's mean is the sum of means). Captaincy, risk
and owning several players from one team depend on the *distribution*, and events in a
fixture are linked. Per draw:

    1. minutes class per player from his minutes model; a 1-59 appearance covers a
       uniform 10-60% of the match
    2. player goals ~ Poisson(rate x time on pitch), with rate set so that the mean is
       the model's (reconciled) expectation. A team's goals are the sum of its
       players' goals plus own goals, so player and team totals agree in every draw
    3. assists scale with the team's goals in that draw (attackers and creators of
       one team rise and fall together)
    4. goals conceded while on the pitch: the opponent's simulated total, thinned by
       time on pitch; clean sheet = 60+ minutes and none conceded
    5. saves, bonus (capped at 3), yellow cards and defensive contributions from each
       player's expectation given that he plays

Bonus and saves are drawn independently of goals; their real correlation with returns
is ignored (a known simplification).
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from fpl_core import scoring
from fpl_models.player_features import POSITIONS

SUB_FRACTION = (0.1, 0.6)
HAUL = 10
GOAL_POINTS = np.array([scoring.GOAL_POINTS[scoring.Position[p]] for p in POSITIONS])
CS_POINTS = np.array([scoring.CLEAN_SHEET_POINTS[scoring.Position[p]] for p in POSITIONS])
GKP, DEF = POSITIONS.index("GKP"), POSITIONS.index("DEF")


@dataclass(frozen=True)
class Simulation:
    points: np.ndarray  # (draws, rows)
    goals: np.ndarray  # (draws, rows)
    own_goals: np.ndarray  # (draws, team-fixtures), conceded by the opponent
    team_goals: np.ndarray  # (draws, team-fixtures), players' goals + own goals
    team_index: np.ndarray  # row -> team-fixture column

    def summary(self) -> pl.DataFrame:
        p = self.points
        return pl.DataFrame({
            "sim_mean": p.mean(axis=0), "sim_sd": p.std(axis=0),
            "sim_p10": np.quantile(p, 0.1, axis=0), "sim_p50": np.quantile(p, 0.5, axis=0),
            "sim_p90": np.quantile(p, 0.9, axis=0), "p_haul": (p >= HAUL).mean(axis=0),
        })  # fmt: skip


@dataclass(frozen=True)
class SimulationSettings:
    draws: int = 4000
    own_goal_rate: float = 0.03  # own goals per unit of team lambda
    defcon_on: bool = True  # defensive contributions score (2025-26 rules onwards)
    seed: int = 0


def simulate(
    rows: pl.DataFrame,
    events: pl.DataFrame,
    settings: SimulationSettings = SimulationSettings(),  # noqa: B008 - frozen dataclass
) -> Simulation:
    """rows: match_key, team_key, opponent_team_key, position, team_lambda.

    events (aligned with rows): p_60, p_1_59, e_goals_scored, e_assists, e_bonus,
    e_saves, e_yellow_cards, p_defcon.
    """
    draws, own_goal_rate = settings.draws, settings.own_goal_rate
    rng = np.random.default_rng(settings.seed)
    n = rows.height
    position = rows["position"].replace_strict({p: i for i, p in enumerate(POSITIONS)}).to_numpy()
    p60, p159 = events["p_60"].to_numpy(), events["p_1_59"].to_numpy()
    p_play = np.clip(p60 + p159, 1e-9, 1.0)

    # Team-fixture index for each row, and for its opponent.
    keys = rows.select(pl.concat_str("match_key", "team_key", separator="|").alias("k"))["k"]
    opp_keys = rows.select(
        pl.concat_str("match_key", "opponent_team_key", separator="|").alias("k")
    )["k"]
    teams = sorted(set(keys.to_list()) | set(opp_keys.to_list()))
    column = {k: i for i, k in enumerate(teams)}
    team = np.array([column[k] for k in keys])
    opponent = np.array([column[k] for k in opp_keys])

    u = rng.random((draws, n))
    plays60 = u < p60
    sub = (u >= p60) & (u < p60 + p159)
    plays = plays60 | sub
    on = plays60 * 1.0 + sub * rng.uniform(*SUB_FRACTION, size=(draws, n))
    expected_on = p60 + p159 * float(np.mean(SUB_FRACTION))

    goals = rng.poisson(events["e_goals_scored"].to_numpy() / np.clip(expected_on, 1e-9, None) * on)
    team_lambda = rows.group_by(pl.Series(keys).alias("k")).agg(rows["team_lambda"].first())
    lam = np.zeros(len(teams))
    for k, value in team_lambda.iter_rows():
        lam[column[k]] = value
    own_goals = rng.poisson(own_goal_rate * np.broadcast_to(lam, (draws, len(teams))))
    team_goals = own_goals.astype(np.float64)
    np.add.at(team_goals.T, team, goals.T)

    # Assists rise and fall with the team's goals in each draw (mean preserved).
    goal_factor = team_goals[:, team] / np.clip(lam[team], 1e-9, None)
    assist_rate = events["e_assists"].to_numpy() / np.clip(expected_on, 1e-9, None)
    assists = rng.poisson(assist_rate * on * goal_factor)

    conceded = rng.binomial(team_goals[:, opponent].astype(np.int64), np.clip(on, 0, 1))

    def given_play(col: str) -> np.ndarray:
        result: np.ndarray = np.where(plays, events[col].to_numpy() / p_play, 0.0)
        return result

    saves = rng.poisson(given_play("e_saves"))
    bonus = np.minimum(rng.poisson(given_play("e_bonus")), 3)
    yellow = rng.random((draws, n)) < np.clip(given_play("e_yellow_cards"), 0, 1)
    p_defcon = np.clip(events["p_defcon"].to_numpy() / np.clip(p60, 1e-9, None), 0, 1)
    defcon = plays60 & (rng.random((draws, n)) < p_defcon)

    is_gk = position == GKP
    concedes = (position == GKP) | (position == DEF)
    points = np.where(
        plays60, scoring.LONG_PLAY_POINTS, np.where(sub, scoring.SHORT_PLAY_POINTS, 0)
    ).astype(np.float64)
    points += goals * GOAL_POINTS[position] + assists * scoring.ASSIST_POINTS
    points += plays60 * (conceded == 0) * CS_POINTS[position]
    points -= concedes * plays * (conceded // scoring.GOALS_CONCEDED_PER_PENALTY)
    points += is_gk * (saves // scoring.SAVES_PER_POINT)
    points += bonus + yellow * scoring.YELLOW_CARD_POINTS
    if settings.defcon_on:
        points += ~is_gk * defcon * scoring.DEFENSIVE_CONTRIBUTION_POINTS
    return Simulation(
        points=points, goals=goals, own_goals=own_goals, team_goals=team_goals, team_index=team
    )


def per_player_gameweek(simulation: Simulation, rows: pl.DataFrame) -> pl.DataFrame:
    """Distribution of a player's gameweek total: fixtures are summed within each draw
    (adding quantiles across a double gameweek would be wrong)."""
    groups = rows.with_row_index("i").group_by("player_code", "gameweek").agg(pl.col("i"))
    out = []
    for code, gameweek, members in groups.iter_rows():
        total = simulation.points[:, members].sum(axis=1)
        out.append({
            "player_code": code, "gameweek": gameweek,
            "sim_mean": float(total.mean()), "sim_sd": float(total.std()),
            "sim_p10": float(np.quantile(total, 0.1)), "sim_p90": float(np.quantile(total, 0.9)),
            "p_haul": float((total >= HAUL).mean()),
        })  # fmt: skip
    return pl.DataFrame(out)
