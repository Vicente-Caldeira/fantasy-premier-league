-- One row per Premier League player per cup or European appearance, with the kickoff
-- time and FPL player code: all-competition form and fatigue features. Rows whose match
-- has no kickoff (dropped upstream) or whose element isn't a known player are excluded.
with stats as (select * from {{ ref('stg_fpl_elo_insights__cup_player_match_stats') }}),

cups as (select * from {{ ref('stg_fpl_elo_insights__cup_matches') }}),

elements as (
    select distinct season, element, player_code
    from {{ ref('fct_player_matches') }}
)

select
    stats.season,
    stats.competition,
    stats.cup_match_id,
    cups.kickoff_utc,
    elements.player_code,
    stats.minutes,
    stats.goals,
    stats.assists,
    stats.xg,
    stats.xa,
    stats.total_shots,
    stats.touches_opposition_box,
    stats.chances_created
from stats
inner join cups on stats.cup_match_id = cups.cup_match_id
inner join elements on stats.season = elements.season and stats.element = elements.element
