-- For every finished match, a side's goals = its players' goals + the opponents'
-- own goals. Proves player rows are complete and attributed to the right team.
with sides as (
    select
        match_key,
        team_key,
        sum(goals_scored) as player_goals,
        sum(own_goals) as own_goals_conceded_to_opponent
    from {{ ref('fct_player_matches') }}
    where not is_provisional
    group by match_key, team_key
),

per_match as (
    select
        m.match_key,
        m.season,
        m.home_team_key,
        m.away_team_key,
        m.home_goals,
        m.away_goals,
        home.player_goals + coalesce(away.own_goals_conceded_to_opponent, 0) as home_from_players,
        away.player_goals + coalesce(home.own_goals_conceded_to_opponent, 0) as away_from_players
    from {{ ref('fct_matches') }} as m
    inner join sides as home on m.match_key = home.match_key and m.home_team_key = home.team_key
    inner join sides as away on m.match_key = away.match_key and m.away_team_key = away.team_key
    where m.is_played
)

select * from per_match
where home_goals != home_from_players or away_goals != away_from_players
