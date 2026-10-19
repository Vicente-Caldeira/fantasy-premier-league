-- Understat season totals per player. `team_title` lists every club the player
-- appeared for that season, comma-separated (Understat club names).
select
    season,
    player_id as understat_player_id,
    player_name,
    team_title,
    minutes,
    goals - npg as penalty_goals,
    xg - npxg as penalty_xg,
    xg,
    npxg
from {{ source('understat', 'player_seasons') }}
