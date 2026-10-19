-- Fixtures 2016-17 .. 2025-26, reconstructed from home players' rows.
select distinct
    season,
    fixture as fpl_fixture_id,
    round as gameweek,
    kickoff_time as kickoff_utc,
    team_code as home_team_code,
    opponent_team_code as away_team_code,
    team_h_score as home_goals,
    team_a_score as away_goals,
    true as finished,
    true as finished_provisional
from {{ source('vaastav', 'player_matches') }}
where was_home
