-- FPL and football-data place every played match on the same UTC date.
select match_key, season, home_team_key, away_team_key,
    cast(fpl_kickoff_utc as date) as fpl_date, fd_match_date
from {{ ref('int_matches_joined') }}
where fpl_finished and fd_match_date is not null
  and fd_match_date != cast(fpl_kickoff_utc as date)
