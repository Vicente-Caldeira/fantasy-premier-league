-- Understat per player per match, with the club mapped to its team key. Understat has
-- no FPL identifier: fpl_models matches players by name within club and season.
with player_matches as (select * from {{ source('understat', 'player_matches') }}),
teams as (select * from {{ ref('teams') }})

select
    player_matches.season,
    player_matches.match_id as understat_match_id,
    player_matches.match_date,
    player_matches.understat_player_id,
    player_matches.player_name,
    teams.team_key,
    player_matches.is_home,
    player_matches.position,
    player_matches.minutes,
    player_matches.goals,
    player_matches.own_goals,
    player_matches.shots,
    player_matches.key_passes,
    player_matches.assists,
    player_matches.xg,
    player_matches.npxg,
    player_matches.penalty_xg,
    player_matches.xa,
    player_matches.xg_chain,
    player_matches.xg_buildup,
    player_matches.yellow_cards,
    player_matches.red_cards
from player_matches
left join teams on player_matches.team_id = teams.understat_team_id
