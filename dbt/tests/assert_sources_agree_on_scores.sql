-- FPL, football-data and Understat agree on every final score they share.
-- (FPL-Elo-Insights is checked separately: it has known wrong scores.)
with scores as (
    select match_key, 'fpl' as source, fpl_home_goals as home_goals, fpl_away_goals as away_goals
    from {{ ref('int_matches_joined') }} where fpl_finished
    union all
    select match_key, 'football_data', fd_home_goals, fd_away_goals
    from {{ ref('int_matches_joined') }} where fd_home_goals is not null
    union all
    select match_key, 'understat', us_home_goals, us_away_goals
    from {{ ref('int_matches_joined') }} where us_is_result
)

select match_key
from scores
group by match_key
having count(distinct home_goals) > 1 or count(distinct away_goals) > 1
