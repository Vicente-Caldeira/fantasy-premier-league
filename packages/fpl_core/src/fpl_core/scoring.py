"""FPL points scoring rules.

Values mirror `bootstrap-static.game_config.scoring` from the FPL API for 2025/26 and
2026/27 (the two seasons share point values; only BPS internals changed). Bonus points
are taken as an input because they depend on every player's BPS in the match.
"""

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field


class Position(IntEnum):
    """FPL `element_type` ids."""

    GKP = 1
    DEF = 2
    MID = 3
    FWD = 4


GOAL_POINTS = {Position.GKP: 10, Position.DEF: 6, Position.MID: 5, Position.FWD: 4}
CLEAN_SHEET_POINTS = {Position.GKP: 4, Position.DEF: 4, Position.MID: 1, Position.FWD: 0}
CONCEDING_POSITIONS = frozenset({Position.GKP, Position.DEF})

ASSIST_POINTS = 3
SHORT_PLAY_POINTS = 1
LONG_PLAY_POINTS = 2
LONG_PLAY_MINUTES = 60
SAVES_PER_POINT = 3
GOALS_CONCEDED_PER_PENALTY = 2
PENALTY_SAVED_POINTS = 5
PENALTY_MISSED_POINTS = -2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2

DEFENSIVE_CONTRIBUTION_POINTS = 2
DEF_CBIT_THRESHOLD = 10
MID_FWD_CBIRT_THRESHOLD = 12


class MatchStats(BaseModel):
    """One player's raw stats for one match.

    `goals_conceded` counts only goals conceded while the player was on the pitch,
    matching how FPL attributes clean sheets and conceded penalties.
    """

    model_config = ConfigDict(frozen=True)

    minutes: int = Field(ge=0, le=130)
    goals_scored: int = Field(default=0, ge=0)
    assists: int = Field(default=0, ge=0)
    goals_conceded: int = Field(default=0, ge=0)
    saves: int = Field(default=0, ge=0)
    penalties_saved: int = Field(default=0, ge=0)
    penalties_missed: int = Field(default=0, ge=0)
    yellow_cards: int = Field(default=0, ge=0, le=1)
    red_cards: int = Field(default=0, ge=0, le=1)
    own_goals: int = Field(default=0, ge=0)
    clearances_blocks_interceptions: int = Field(default=0, ge=0)
    tackles: int = Field(default=0, ge=0)
    recoveries: int = Field(default=0, ge=0)
    bonus: int = Field(default=0, ge=0, le=3)


def defensive_contribution_points(position: Position, stats: MatchStats) -> int:
    """2 points for reaching the position's DefCon threshold, else 0. Capped per match."""
    cbit = stats.clearances_blocks_interceptions + stats.tackles
    if position is Position.GKP:
        return 0
    if position is Position.DEF:
        reached = cbit >= DEF_CBIT_THRESHOLD
    else:
        reached = cbit + stats.recoveries >= MID_FWD_CBIRT_THRESHOLD
    return DEFENSIVE_CONTRIBUTION_POINTS if reached else 0


def points(position: Position, stats: MatchStats) -> int:
    """Total FPL points for one player in one match."""
    if stats.minutes == 0:
        return 0

    total = LONG_PLAY_POINTS if stats.minutes >= LONG_PLAY_MINUTES else SHORT_PLAY_POINTS
    total += stats.goals_scored * GOAL_POINTS[position]
    total += stats.assists * ASSIST_POINTS

    if stats.minutes >= LONG_PLAY_MINUTES and stats.goals_conceded == 0:
        total += CLEAN_SHEET_POINTS[position]
    if position in CONCEDING_POSITIONS:
        total -= stats.goals_conceded // GOALS_CONCEDED_PER_PENALTY
    if position is Position.GKP:
        total += stats.saves // SAVES_PER_POINT

    total += stats.penalties_saved * PENALTY_SAVED_POINTS
    total += stats.penalties_missed * PENALTY_MISSED_POINTS
    total += stats.yellow_cards * YELLOW_CARD_POINTS
    total += stats.red_cards * RED_CARD_POINTS
    total += stats.own_goals * OWN_GOAL_POINTS
    total += defensive_contribution_points(position, stats)
    return total + stats.bonus
