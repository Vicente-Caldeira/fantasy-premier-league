-- Current-season clubs from the latest FPL snapshot.
with latest as (
    select * from {{ source('fpl', 'teams') }}
    qualify snapshot_date = max(snapshot_date) over ()
)

select
    id as fpl_team_id,
    code as fpl_team_code,
    name,
    short_name,
    season
from latest
