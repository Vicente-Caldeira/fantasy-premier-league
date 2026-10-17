{{ config(severity='warn') }}
-- FPL-Elo-Insights scores vs FPL. Warns only: 6 known wrong scores in 2024-26 (e.g.
-- Liverpool 2-2 Arsenal recorded as 2-0), so its goals are never used in fct_matches.
select match_key, season, home_team_key, away_team_key,
    fpl_home_goals, fpl_away_goals, elo_home_goals, elo_away_goals
from {{ ref('int_matches_joined') }}
where fpl_finished and elo_home_goals is not null
  and (elo_home_goals != fpl_home_goals or elo_away_goals != fpl_away_goals)
