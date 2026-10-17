-- FPL player-fixture player_rows from history and the live API, linked to fct_matches.
with player_rows as (
    select * from {{ ref('stg_vaastav__player_matches') }}
    union all by name
    select * from {{ ref('stg_fpl__player_matches') }}
),

teams as (select * from {{ ref('teams') }}),
matches as (select * from {{ ref('fct_matches') }})

select
    {{ dbt.hash("player_rows.season || '|' || cast(player_rows.element as varchar) || '|' || cast(player_rows.fpl_fixture_id as varchar)") }}
        as player_match_key,
    matches.match_key,
    matches.elo_match_id,
    team.team_key,
    opponent.team_key as opponent_team_key,
    player_rows.*
from player_rows
left join teams as team on player_rows.team_code = team.fpl_team_code
left join teams as opponent on player_rows.opponent_team_code = opponent.fpl_team_code
left join matches
    on player_rows.season = matches.season and player_rows.fpl_fixture_id = matches.fpl_fixture_id
