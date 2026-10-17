-- Current-season players from the latest FPL snapshot (point-in-time state lives in
-- the daily snapshots themselves; this is the latest view).
select
    season,
    id as element,
    code as player_code,
    element_type,
    team as fpl_team_id,
    web_name,
    first_name,
    second_name,
    now_cost,
    selected_by_percent,
    status,
    news
from {{ source('fpl', 'players') }}
qualify snapshot_date = max(snapshot_date) over ()
