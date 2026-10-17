-- Every player registered for the current season as of the latest FPL snapshot: club,
-- position, price and availability. The population the live predictions are made for.
with players as (select * from {{ ref('stg_fpl__players') }}),
fpl_teams as (select * from {{ ref('stg_fpl__teams') }}),
teams as (select * from {{ ref('teams') }})

select
    players.season,
    players.player_code,
    players.element,
    players.web_name,
    case players.element_type when 1 then 'GKP' when 2 then 'DEF' when 3 then 'MID' when 4 then 'FWD' end
        as position,
    teams.team_key,
    players.now_cost as price,
    players.selected_by_percent,
    players.status,
    players.news,
    players.chance_of_playing_next_round
from players
inner join fpl_teams on players.fpl_team_id = fpl_teams.fpl_team_id
inner join teams on fpl_teams.fpl_team_code = teams.fpl_team_code
