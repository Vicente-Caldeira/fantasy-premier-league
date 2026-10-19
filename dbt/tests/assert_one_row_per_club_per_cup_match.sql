-- fct_cup_matches grain: one row per club per cup match.
select cup_match_id, team_key, count(*) as copies
from {{ ref('fct_cup_matches') }}
group by cup_match_id, team_key
having count(*) > 1
