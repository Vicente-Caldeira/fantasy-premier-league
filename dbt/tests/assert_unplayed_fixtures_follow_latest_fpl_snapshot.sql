-- FPL is the authority for a fixture's gameweek and kickoff: every unplayed match must
-- match the latest FPL fixtures snapshot (postponements and moves included), and every
-- fixture in that snapshot must be in the mart.
with latest as (
    select f.*, home.team_key as home_team_key, away.team_key as away_team_key
    from {{ ref('stg_fpl__fixtures') }} as f
    left join {{ ref('teams') }} as home on f.home_team_code = home.fpl_team_code
    left join {{ ref('teams') }} as away on f.away_team_code = away.fpl_team_code
),

matches as (select * from {{ ref('fct_matches') }})

select latest.fpl_fixture_id, latest.gameweek, matches.gameweek as mart_gameweek
from latest
left join matches
    on latest.season = matches.season
    and latest.home_team_key = matches.home_team_key
    and latest.away_team_key = matches.away_team_key
where
    matches.match_key is null
    or (
        not matches.is_played
        and (
            latest.gameweek is distinct from matches.gameweek
            or latest.kickoff_utc is distinct from matches.kickoff_utc
            or latest.fpl_fixture_id is distinct from matches.fpl_fixture_id
        )
    )
