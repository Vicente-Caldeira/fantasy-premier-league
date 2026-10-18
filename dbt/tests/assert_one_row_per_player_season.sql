-- dim_player_seasons grain: one row per FPL player and season.
select season, player_code, count(*) as rows
from {{ ref('dim_player_seasons') }}
group by season, player_code
having count(*) > 1
