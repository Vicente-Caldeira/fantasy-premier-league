-- One row per Premier League club per cup match (a Premier League derby in a cup gives
-- two rows). For congestion features: when each club plays outside the league.
with cups as (select * from {{ ref('stg_fpl_elo_insights__cup_matches') }}),

teams as (select * from {{ ref('teams') }}),

sides as (
    select
        cup_match_id,
        season,
        competition,
        kickoff_utc,
        finished,
        home_team_code as team_code,
        true as is_home
    from cups
    where home_team_code is not null
    union all
    select
        cup_match_id,
        season,
        competition,
        kickoff_utc,
        finished,
        away_team_code as team_code,
        false as is_home
    from cups
    where away_team_code is not null
)

select
    sides.cup_match_id,
    sides.season,
    sides.competition,
    sides.kickoff_utc,
    sides.finished,
    teams.team_key,
    sides.is_home
from sides
left join teams on sides.team_code = teams.fpl_team_code
