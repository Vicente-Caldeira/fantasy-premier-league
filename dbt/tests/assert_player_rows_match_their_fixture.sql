-- Each player row belongs to one side of its match, consistent with `is_home`.
select pm.player_match_key, pm.season, pm.team_key, m.home_team_key, m.away_team_key, pm.is_home
from {{ ref('fct_player_matches') }} as pm
inner join {{ ref('fct_matches') }} as m using (match_key)
where pm.team_key != case when pm.is_home then m.home_team_key else m.away_team_key end
   or pm.opponent_team_key != case when pm.is_home then m.away_team_key else m.home_team_key end
