"""Response and request bodies. Prices are in millions (FPL shows 7.5, stores 75)."""

from pydantic import BaseModel, Field


class Meta(BaseModel):
    season: str
    current_gameweek: int | None
    next_gameweek: int | None
    next_deadline: str | None
    horizon: list[int]
    predictions_as_of: str


class Opponent(BaseModel):
    team_key: str
    is_home: bool


class SeasonStats(BaseModel):
    points: int
    minutes: int
    appearances: int
    starts: int
    goals: int
    assists: int
    xg: float
    xa: float
    clean_sheets: int
    bonus: int
    saves: int
    defensive_contribution: int
    form: float


class Player(BaseModel):
    code: int
    element: int
    name: str
    position: str
    team_key: str
    price: float
    ownership: float
    status: str
    news: str
    chance_of_playing: int | None
    xp: list[float] = Field(description="expected points per horizon gameweek")
    xp_total: float
    p_haul: float | None
    p10: float | None
    p90: float | None
    p_60: float | None
    p_rise: float | None
    p_fall: float | None
    net_transfers: int | None
    season: SeasonStats


class UpcomingGameweek(BaseModel):
    gameweek: int
    opponents: list[Opponent]
    xp: float
    p_play: float | None
    p_60: float | None
    e_goals: float | None
    e_assists: float | None
    p_clean_sheet: float | None
    availability: float | None
    p10: float | None
    p90: float | None
    p_haul: float | None


class MatchRow(BaseModel):
    season: str
    gameweek: int
    opponent: str
    is_home: bool
    minutes: int
    points: int
    goals: int
    assists: int
    xg: float | None
    xa: float | None
    bonus: int
    bps: int
    clean_sheet: int
    price: float
    selected: int | None


class PastSeason(BaseModel):
    season: str
    points: int
    minutes: int
    goals: int
    assists: int
    clean_sheets: int
    max_price: float


class PlayerDetail(BaseModel):
    player: Player
    upcoming: list[UpcomingGameweek]
    matches: list[MatchRow]
    past_seasons: list[PastSeason]


class Fixture(BaseModel):
    match_key: str
    gameweek: int
    kickoff_utc: str
    home: str
    away: str
    lambda_home: float | None = None
    lambda_away: float | None = None
    home_win: float | None = None
    draw: float | None = None
    away_win: float | None = None
    home_clean_sheet: float | None = None
    away_clean_sheet: float | None = None
    over_2_5: float | None = None
    source: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    home_xg: float | None = None
    away_xg: float | None = None


class ClubFixture(BaseModel):
    gameweek: int
    opponent: str
    is_home: bool
    goals_for: float
    goals_against: float
    clean_sheet: float
    win: float
    draw: float
    loss: float


class ClubRecord(BaseModel):
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    xg_for: float
    xg_against: float


class Club(BaseModel):
    key: str
    name: str
    short_name: str
    fpl_team_code: int
    attack: float
    defence: float
    record: ClubRecord
    fixtures: list[ClubFixture]


class ClubDetail(BaseModel):
    club: Club
    results: list[Fixture]
    players: list[Player]


class GameweekHistory(BaseModel):
    gameweek: int
    points: int
    total_points: int
    overall_rank: int | None
    transfers: int
    hits_cost: int
    bank: float
    value: float


class Entry(BaseModel):
    id: int
    name: str
    manager: str
    overall_rank: int | None
    total_points: int
    squad: list[int]
    lineup: list[int]
    captain: int | None
    vice_captain: int | None
    bank: float
    free_transfers: int
    selling_prices: dict[int, float]
    next_gameweek: int
    chips_used: list[str]
    history: list[GameweekHistory]


class SquadRequest(BaseModel):
    squad: list[int] = Field(min_length=15, max_length=15)
    horizon: int = Field(default=5, ge=1, le=8)


class PlannedGameweek(BaseModel):
    gameweek: int
    squad: list[int]
    lineup: list[int]
    captain: int
    vice_captain: int
    transfers_in: list[int]
    transfers_out: list[int]
    free_transfers: int
    hits: int
    xp: float


class PlanRequest(SquadRequest):
    bank: float = 0.0
    free_transfers: int = Field(default=1, ge=0, le=5)
    selling_prices: dict[int, float] = Field(default_factory=dict)
    keep: list[int] = Field(default_factory=list)
    ban: list[int] = Field(default_factory=list)
    wildcard: bool = False


class Plan(BaseModel):
    status: str
    gameweeks: list[PlannedGameweek]
    total_xp: float = Field(description="after hits")
