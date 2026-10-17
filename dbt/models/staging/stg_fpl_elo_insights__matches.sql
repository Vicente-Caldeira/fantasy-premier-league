-- Opta-style team stats per match; clubs are already FPL team codes.
select
    season,
    match_id as elo_match_id,
    kickoff_time as kickoff_utc,
    home_team_code,
    away_team_code,
    finished,
    home_score as home_goals,
    away_score as away_goals,
    home_elo,
    away_elo,
    home_xg as home_opta_xg,
    away_xg as away_opta_xg,
    home_non_penalty_xg,
    away_non_penalty_xg,
    home_xgot,
    away_xgot,
    home_possession,
    away_possession,
    home_big_chances,
    away_big_chances
from {{ source('fpl_elo_insights', 'matches') }}
