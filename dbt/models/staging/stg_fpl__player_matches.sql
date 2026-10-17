-- Current-season player-fixture player_rows from the latest FPL snapshot. The API gives no
-- team per row, so it is derived from the fixture (correct across mid-season moves).
with player_rows as (
    select * from {{ source('fpl', 'player_matches') }}
    qualify snapshot_date = max(snapshot_date) over ()
),

fixtures as (select * from {{ ref('stg_fpl__fixtures') }}),
players as (select * from {{ ref('stg_fpl__players') }}),
teams as (select * from {{ ref('stg_fpl__teams') }})

select
    player_rows.season,
    player_rows.element,
    players.player_code,
    players.element_type,
    player_rows.fixture as fpl_fixture_id,
    player_rows.round as gameweek,
    player_rows.kickoff_time as kickoff_utc,
    player_rows.was_home as is_home,
    case when player_rows.was_home then fixtures.home_team_code else fixtures.away_team_code end
        as team_code,
    opponent.fpl_team_code as opponent_team_code,
    not fixtures.finished as is_provisional,
    player_rows.minutes,
    player_rows.starts,
    player_rows.goals_scored,
    player_rows.assists,
    player_rows.clean_sheets,
    player_rows.goals_conceded,
    player_rows.own_goals,
    player_rows.penalties_saved,
    player_rows.penalties_missed,
    player_rows.yellow_cards,
    player_rows.red_cards,
    player_rows.saves,
    player_rows.bonus,
    player_rows.bps,
    player_rows.influence,
    player_rows.creativity,
    player_rows.threat,
    player_rows.ict_index,
    player_rows.expected_goals,
    player_rows.expected_assists,
    player_rows.expected_goal_involvements,
    player_rows.expected_goals_conceded,
    player_rows.tackles,
    player_rows.clearances_blocks_interceptions,
    player_rows.recoveries,
    player_rows.defensive_contribution,
    player_rows.total_points,
    player_rows.value,
    player_rows.selected,
    player_rows.transfers_balance,
    cast(null as bigint) as transfers_in,
    cast(null as bigint) as transfers_out
from player_rows
inner join fixtures on player_rows.fixture = fixtures.fpl_fixture_id
inner join players on player_rows.element = players.element
inner join teams as opponent on player_rows.opponent_team = opponent.fpl_team_id
