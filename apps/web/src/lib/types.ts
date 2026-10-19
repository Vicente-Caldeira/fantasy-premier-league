// Mirrors apps/api/src/fpl_api/schemas.py. Prices are in millions.

export type Position = "GKP" | "DEF" | "MID" | "FWD";

export interface Meta {
  season: string;
  current_gameweek: number | null;
  next_gameweek: number | null;
  next_deadline: string | null;
  horizon: number[];
  predictions_as_of: string;
}

export interface Opponent {
  team_key: string;
  is_home: boolean;
}

export interface SeasonStats {
  points: number;
  minutes: number;
  appearances: number;
  starts: number;
  goals: number;
  assists: number;
  xg: number;
  xa: number;
  clean_sheets: number;
  bonus: number;
  saves: number;
  defensive_contribution: number;
  form: number;
}

export interface Player {
  code: number;
  element: number;
  name: string;
  position: Position;
  team_key: string;
  price: number;
  ownership: number;
  status: string;
  news: string;
  chance_of_playing: number | null;
  xp: number[];
  xp_total: number;
  p_haul: number | null;
  p10: number | null;
  p90: number | null;
  p_60: number | null;
  p_rise: number | null;
  p_fall: number | null;
  net_transfers: number | null;
  season: SeasonStats;
}

export interface UpcomingGameweek {
  gameweek: number;
  opponents: Opponent[];
  xp: number;
  p_play: number | null;
  p_60: number | null;
  e_goals: number | null;
  e_assists: number | null;
  p_clean_sheet: number | null;
  availability: number | null;
  p10: number | null;
  p90: number | null;
  p_haul: number | null;
}

export interface MatchRow {
  season: string;
  gameweek: number;
  opponent: string;
  is_home: boolean;
  minutes: number;
  points: number;
  goals: number;
  assists: number;
  xg: number | null;
  xa: number | null;
  bonus: number;
  bps: number;
  clean_sheet: number;
  price: number;
  selected: number | null;
}

export interface PastSeason {
  season: string;
  points: number;
  minutes: number;
  goals: number;
  assists: number;
  clean_sheets: number;
  max_price: number;
}

export interface PlayerDetail {
  player: Player;
  upcoming: UpcomingGameweek[];
  matches: MatchRow[];
  past_seasons: PastSeason[];
}

export interface Fixture {
  match_key: string;
  gameweek: number;
  kickoff_utc: string;
  home: string;
  away: string;
  lambda_home: number | null;
  lambda_away: number | null;
  home_win: number | null;
  draw: number | null;
  away_win: number | null;
  home_clean_sheet: number | null;
  away_clean_sheet: number | null;
  over_2_5: number | null;
  source: string | null;
  home_goals: number | null;
  away_goals: number | null;
  home_xg: number | null;
  away_xg: number | null;
}

export interface ClubFixture {
  gameweek: number;
  opponent: string;
  is_home: boolean;
  goals_for: number;
  goals_against: number;
  clean_sheet: number;
  win: number;
  draw: number;
  loss: number;
}

export interface ClubRecord {
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goals_for: number;
  goals_against: number;
  xg_for: number;
  xg_against: number;
}

export interface Club {
  key: string;
  name: string;
  short_name: string;
  fpl_team_code: number;
  attack: number;
  defence: number;
  record: ClubRecord;
  fixtures: ClubFixture[];
}

export interface ClubDetail {
  club: Club;
  results: Fixture[];
  players: Player[];
}

export interface GameweekHistory {
  gameweek: number;
  points: number;
  total_points: number;
  overall_rank: number | null;
  transfers: number;
  hits_cost: number;
  bank: number;
  value: number;
}

export interface Entry {
  id: number;
  name: string;
  manager: string;
  overall_rank: number | null;
  total_points: number;
  squad: number[];
  lineup: number[];
  captain: number | null;
  vice_captain: number | null;
  bank: number;
  free_transfers: number;
  selling_prices: Record<string, number>;
  next_gameweek: number;
  chips_used: string[];
  history: GameweekHistory[];
}

export interface PlannedGameweek {
  gameweek: number;
  squad: number[];
  lineup: number[];
  captain: number;
  vice_captain: number;
  transfers_in: number[];
  transfers_out: number[];
  free_transfers: number;
  hits: number;
  xp: number;
}

export interface Plan {
  status: string;
  gameweeks: PlannedGameweek[];
  total_xp: number;
}
