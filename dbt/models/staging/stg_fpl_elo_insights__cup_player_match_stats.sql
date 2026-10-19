-- Premier League players' stats in cup and European matches (2025-26 onwards).
-- `player_id` is the season's FPL element id, as in the league files.
select
    season,
    competition,
    player_id as element,
    match_id as cup_match_id,
    minutes_played as minutes,
    goals,
    assists,
    xg,
    xa,
    total_shots,
    touches_opposition_box,
    chances_created
from {{ source('fpl_elo_insights', 'cup_player_match_stats') }}
