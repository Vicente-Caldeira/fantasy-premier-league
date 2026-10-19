-- One row per FPL player and season: club (as of the season's last snapshot) and names.
with players as (
    select season, player_code, team_code, web_name, first_name, second_name
    from {{ ref('stg_vaastav__players') }}
    union all
    select season, player_code, team_code, web_name, first_name, second_name
    from {{ ref('stg_fpl__players') }}
)

select
    players.season,
    players.player_code,
    teams.team_key,
    players.web_name,
    players.first_name,
    players.second_name
from players
left join {{ ref('teams') }} as teams on players.team_code = teams.fpl_team_code
