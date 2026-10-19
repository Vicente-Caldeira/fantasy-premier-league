-- Historical player-fixture rows, 2016-17 .. 2025-26 (teams already fixture-derived).
-- FPL published expected stats from 2022-23 GW16; earlier 2022-23 rows carry 0.0 for
-- every player, which would read as "no chance of scoring". A gameweek with no
-- expected goals anywhere is treated as missing.
with source as (
    select
        *,
        sum(expected_goals) over (partition by season, round) as gameweek_expected_goals
    from {{ source('vaastav', 'player_matches') }}
)

select
    season,
    element,
    player_code,
    element_type,
    fixture as fpl_fixture_id,
    round as gameweek,
    kickoff_time as kickoff_utc,
    was_home as is_home,
    team_code,
    opponent_team_code,
    false as is_provisional,
    minutes,
    starts,
    goals_scored,
    assists,
    clean_sheets,
    goals_conceded,
    own_goals,
    penalties_saved,
    penalties_missed,
    yellow_cards,
    red_cards,
    saves,
    bonus,
    bps,
    influence,
    creativity,
    threat,
    ict_index,
    case when gameweek_expected_goals > 0 then expected_goals end as expected_goals,
    case when gameweek_expected_goals > 0 then expected_assists end as expected_assists,
    case when gameweek_expected_goals > 0 then expected_goal_involvements end
        as expected_goal_involvements,
    case when gameweek_expected_goals > 0 then expected_goals_conceded end as expected_goals_conceded,
    tackles,
    clearances_blocks_interceptions,
    recoveries,
    defensive_contribution,
    total_points,
    value,
    selected,
    transfers_balance,
    transfers_in,
    transfers_out
from source
