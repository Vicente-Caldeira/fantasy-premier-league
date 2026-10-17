"""Data contracts for FPL API payloads.

Models declare the fields we depend on and allow (but ignore) extras, so FPL adding a
field never breaks ingestion while removing or retyping one fails loudly. The API sends
many decimals as strings ("0.01"); lax-mode validation coerces them to floats.
"""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    def declared(self) -> dict[str, object]:
        """Only the declared fields, for building typed tables."""
        return self.model_dump(include=set(type(self).model_fields))


class Team(Contract):
    id: int
    code: int
    name: str
    short_name: str
    strength: int | None  # null for every team in 2026-27
    strength_overall_home: int
    strength_overall_away: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int


class Gameweek(Contract):
    id: int
    name: str
    deadline_time: datetime
    finished: bool
    data_checked: bool
    is_previous: bool
    is_current: bool
    is_next: bool
    average_entry_score: int
    highest_score: int | None


class Position(Contract):
    id: int
    singular_name_short: str
    squad_select: int
    squad_min_play: int
    squad_max_play: int


class Player(Contract):
    id: int
    code: int
    opta_code: str
    web_name: str
    first_name: str
    second_name: str
    team: int
    team_code: int
    element_type: int
    status: str
    news: str
    news_added: datetime | None
    chance_of_playing_this_round: int | None
    chance_of_playing_next_round: int | None
    now_cost: int
    cost_change_start: int
    selected_by_percent: float
    form: float
    points_per_game: float
    ep_this: float | None
    ep_next: float | None
    total_points: int
    minutes: int
    starts: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    saves: int
    bonus: int
    bps: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    defensive_contribution: int
    transfers_in_event: int
    transfers_out_event: int
    penalties_order: int | None
    direct_freekicks_order: int | None
    corners_and_indirect_freekicks_order: int | None
    birth_date: date | None


class BootstrapStatic(Contract):
    teams: list[Team]
    events: list[Gameweek]
    element_types: list[Position]
    elements: list[Player]

    @property
    def season(self) -> str:
        """Season label such as "2026-27", from the first gameweek's deadline."""
        start = min(gw.deadline_time for gw in self.events).year
        return f"{start}-{(start + 1) % 100:02d}"


class Fixture(Contract):
    id: int
    code: int
    event: int | None
    kickoff_time: datetime | None
    team_h: int
    team_a: int
    team_h_score: int | None
    team_a_score: int | None
    team_h_difficulty: int
    team_a_difficulty: int
    started: bool | None
    finished: bool
    finished_provisional: bool
    minutes: int


class PlayerMatch(Contract):
    """One player's line in one fixture (`element-summary.history`)."""

    element: int
    fixture: int
    opponent_team: int
    round: int
    was_home: bool
    kickoff_time: datetime
    minutes: int
    starts: int
    total_points: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    saves: int
    bonus: int
    bps: int
    clearances_blocks_interceptions: int
    tackles: int
    recoveries: int
    defensive_contribution: int
    influence: float
    creativity: float
    threat: float
    ict_index: float
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    value: int
    selected: int
    transfers_balance: int
    team_h_score: int | None
    team_a_score: int | None


class ElementSummary(Contract):
    history: list[PlayerMatch]
