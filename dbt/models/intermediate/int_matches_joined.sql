-- Every source's view of each match, side by side, before any value is chosen.
-- A Premier League season has each pairing exactly once per home ground, so
-- (season, home, away) identifies a match in every source.
with fpl as (select * from {{ ref('int_fpl_fixtures') }}),
fd as (select * from {{ ref('stg_football_data__matches') }}),
upcoming as (select * from {{ ref('stg_football_data__upcoming_matches') }}),
us as (select * from {{ ref('stg_understat__matches') }}),
elo as (
    select elo.*, home.team_key as home_team_key, away.team_key as away_team_key
    from {{ ref('stg_fpl_elo_insights__matches') }} as elo
    left join {{ ref('teams') }} as home on elo.home_team_code = home.fpl_team_code
    left join {{ ref('teams') }} as away on elo.away_team_code = away.fpl_team_code
),

match_keys as (
    select season, home_team_key, away_team_key from fpl
    union distinct
    select season, home_team_key, away_team_key from fd
    union distinct
    select season, home_team_key, away_team_key from us
    union distinct
    select season, home_team_key, away_team_key from elo
)

select
    {{ dbt.hash("match_keys.season || '|' || match_keys.home_team_key || '|' || match_keys.away_team_key") }} as match_key,
    match_keys.season,
    match_keys.home_team_key,
    match_keys.away_team_key,

    fpl.fpl_fixture_id,
    fpl.gameweek,
    fpl.kickoff_utc as fpl_kickoff_utc,
    fpl.finished as fpl_finished,
    fpl.finished_provisional as fpl_finished_provisional,
    fpl.home_goals as fpl_home_goals,
    fpl.away_goals as fpl_away_goals,

    fd.match_date as fd_match_date,
    fd.home_goals as fd_home_goals,
    fd.away_goals as fd_away_goals,
    fd.home_shots,
    fd.away_shots,
    fd.home_shots_on_target,
    fd.away_shots_on_target,
    fd.referee,
    -- Played matches carry their opening odds; upcoming ones the latest snapshot's.
    coalesce(fd.odds_home, upcoming.odds_home) as odds_home,
    coalesce(fd.odds_draw, upcoming.odds_draw) as odds_draw,
    coalesce(fd.odds_away, upcoming.odds_away) as odds_away,
    coalesce(fd.odds_over_2_5, upcoming.odds_over_2_5) as odds_over_2_5,
    coalesce(fd.odds_under_2_5, upcoming.odds_under_2_5) as odds_under_2_5,
    fd.close_odds_home,
    fd.close_odds_draw,
    fd.close_odds_away,
    fd.close_odds_over_2_5,
    fd.close_odds_under_2_5,
    coalesce(fd.odds_1x2_source, upcoming.odds_1x2_source) as odds_1x2_source,
    fd.close_odds_1x2_source,

    us.understat_match_id,
    us.match_date as us_match_date,
    us.kickoff_reported as us_kickoff_reported,
    us.is_result as us_is_result,
    us.home_goals as us_home_goals,
    us.away_goals as us_away_goals,
    us.home_xg,
    us.away_xg,
    us.forecast_home_win,
    us.forecast_draw,
    us.forecast_away_win,

    elo.elo_match_id,
    elo.kickoff_utc as elo_kickoff_utc,
    elo.home_goals as elo_home_goals,
    elo.away_goals as elo_away_goals,
    elo.home_elo,
    elo.away_elo,
    elo.home_opta_xg,
    elo.away_opta_xg,
    elo.home_xgot,
    elo.away_xgot,
    elo.home_possession,
    elo.away_possession,
    elo.home_big_chances,
    elo.away_big_chances
from match_keys
left join fpl using (season, home_team_key, away_team_key)
left join fd using (season, home_team_key, away_team_key)
left join upcoming using (season, home_team_key, away_team_key)
left join us using (season, home_team_key, away_team_key)
left join elo using (season, home_team_key, away_team_key)
