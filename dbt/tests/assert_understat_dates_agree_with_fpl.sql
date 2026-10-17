{{ config(severity='warn') }}
-- Understat kickoff times are off by -12..+8 hours depending on the season (worst in
-- 2016-17), so its date can be a day out for evening games. Warns only; fct_matches
-- uses Understat's date solely for 2014-15 and 2015-16, where nothing else exists.
select match_key, season, home_team_key, away_team_key,
    cast(fpl_kickoff_utc as date) as fpl_date, us_match_date
from {{ ref('int_matches_joined') }}
where fpl_finished and us_is_result
  and us_match_date != cast(fpl_kickoff_utc as date)
