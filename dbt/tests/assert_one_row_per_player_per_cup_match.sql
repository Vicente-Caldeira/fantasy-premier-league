-- fct_cup_player_matches grain: one row per player per cup match.
select cup_match_id, player_code, count(*) as n
from {{ ref('fct_cup_player_matches') }}
group by cup_match_id, player_code
having count(*) > 1
