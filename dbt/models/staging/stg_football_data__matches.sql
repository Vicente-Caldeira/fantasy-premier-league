-- Results and bookmaker odds, with clubs mapped to team keys.
with matches as (select * from {{ source('football_data', 'matches') }}),
teams as (select * from {{ ref('teams') }})

select
    matches.season,
    matches.match_date,
    home.team_key as home_team_key,
    away.team_key as away_team_key,
    matches.home_goals,
    matches.away_goals,
    matches.home_shots,
    matches.away_shots,
    matches.home_shots_on_target,
    matches.away_shots_on_target,
    matches.referee,
    matches.odds_home,
    matches.odds_draw,
    matches.odds_away,
    matches.odds_over_2_5,
    matches.odds_under_2_5,
    matches.close_odds_home,
    matches.close_odds_draw,
    matches.close_odds_away,
    matches.close_odds_over_2_5,
    matches.close_odds_under_2_5,
    matches.odds_1x2_source,
    matches.close_odds_1x2_source
from matches
left join teams as home on matches.home_team = home.football_data_name
left join teams as away on matches.away_team = away.football_data_name
