-- A gameweek where FPL published no expected stats must have them null, not 0.0
-- (2022-23 GW1-15 shipped zeros), or xG targets read as "no chance of scoring".
select season, gameweek
from {{ ref('fct_player_matches') }}
group by season, gameweek
having count(expected_goals) > 0 and sum(expected_goals) = 0
