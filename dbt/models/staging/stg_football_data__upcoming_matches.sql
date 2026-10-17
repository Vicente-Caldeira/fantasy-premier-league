-- Pre-match odds for fixtures not yet played, from the latest daily snapshot.
with latest as (
    select * from {{ source('football_data', 'upcoming_matches') }}
    qualify snapshot_date = max(snapshot_date) over ()
),

teams as (select * from {{ ref('teams') }})

select
    latest.season,
    latest.snapshot_date,
    latest.match_date,
    home.team_key as home_team_key,
    away.team_key as away_team_key,
    latest.odds_home,
    latest.odds_draw,
    latest.odds_away,
    latest.odds_over_2_5,
    latest.odds_under_2_5,
    latest.odds_1x2_source
from latest
left join teams as home on latest.home_team = home.football_data_name
left join teams as away on latest.away_team = away.football_data_name
