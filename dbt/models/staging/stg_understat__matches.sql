-- Understat matches with clubs mapped to team keys. Kickoff times are unreliable
-- (+/-1h around BST), so `match_date` is the join key.
with matches as (select * from {{ source('understat', 'matches') }}),
teams as (select * from {{ ref('teams') }})

select
    matches.season,
    matches.match_id as understat_match_id,
    matches.match_date,
    matches.kickoff_reported,
    home.team_key as home_team_key,
    away.team_key as away_team_key,
    matches.is_result,
    matches.home_goals,
    matches.away_goals,
    matches.home_xg,
    matches.away_xg,
    matches.forecast_home_win,
    matches.forecast_draw,
    matches.forecast_away_win
from matches
left join teams as home on matches.home_team_id = home.understat_team_id
left join teams as away on matches.away_team_id = away.understat_team_id
