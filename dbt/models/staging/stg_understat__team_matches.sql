select
    team_matches.season,
    team_matches.match_id as understat_match_id,
    teams.team_key,
    team_matches.is_home,
    team_matches.xg,
    team_matches.xga,
    team_matches.npxg,
    team_matches.npxga,
    team_matches.ppda_passes,
    team_matches.ppda_defensive_actions,
    team_matches.ppda_allowed_passes,
    team_matches.ppda_allowed_defensive_actions,
    team_matches.deep,
    team_matches.deep_allowed,
    team_matches.xpts
from {{ source('understat', 'team_matches') }} as team_matches
left join {{ ref('teams') }} as teams on team_matches.team_id = teams.understat_team_id
