-- Current-season fixtures from the latest FPL snapshot, keyed by stable team codes.
with latest as (
    select * from {{ source('fpl', 'fixtures') }}
    qualify snapshot_date = max(snapshot_date) over ()
),

teams as (select * from {{ ref('stg_fpl__teams') }})

select
    latest.season,
    latest.id as fpl_fixture_id,
    latest.event as gameweek,
    latest.kickoff_time as kickoff_utc,
    home.fpl_team_code as home_team_code,
    away.fpl_team_code as away_team_code,
    latest.team_h_score as home_goals,
    latest.team_a_score as away_goals,
    latest.team_h_difficulty as home_difficulty,
    latest.team_a_difficulty as away_difficulty,
    latest.finished,
    -- Played, scores in, awaiting FPL's bonus confirmation (hours after full time).
    coalesce(latest.finished_provisional, latest.finished) as finished_provisional
from latest
inner join teams as home on latest.team_h = home.fpl_team_id
inner join teams as away on latest.team_a = away.fpl_team_id
