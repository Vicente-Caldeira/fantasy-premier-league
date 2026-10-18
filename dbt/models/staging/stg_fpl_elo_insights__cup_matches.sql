-- European and domestic cup matches involving Premier League clubs (2025-26 onwards).
-- Team codes are FPL codes; a foreign or lower-league side has none.
select
    season,
    competition,
    match_id as cup_match_id,
    kickoff_time as kickoff_utc,
    finished,
    home_team_code,
    away_team_code
from {{ source('fpl_elo_insights', 'cup_matches') }}
