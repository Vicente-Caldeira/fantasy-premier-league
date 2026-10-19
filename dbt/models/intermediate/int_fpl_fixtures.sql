-- One FPL-coded fixture list: vaastav history (2016-17 .. 2025-26) + live FPL seasons.
with fixtures as (
    select * from {{ ref('stg_vaastav__fixtures') }}
    union all
    select
        season,
        fpl_fixture_id,
        gameweek,
        kickoff_utc,
        home_team_code,
        away_team_code,
        home_goals,
        away_goals,
        finished,
        finished_provisional
    from {{ ref('stg_fpl__fixtures') }}
),

teams as (select * from {{ ref('teams') }})

select
    fixtures.*,
    home.team_key as home_team_key,
    away.team_key as away_team_key
from fixtures
left join teams as home on fixtures.home_team_code = home.fpl_team_code
left join teams as away on fixtures.away_team_code = away.fpl_team_code
