-- One Understat row per player per match.
select understat_match_id, understat_player_id, count(*) as rows
from {{ ref('fct_understat_player_matches') }}
group by understat_match_id, understat_player_id
having count(*) > 1
