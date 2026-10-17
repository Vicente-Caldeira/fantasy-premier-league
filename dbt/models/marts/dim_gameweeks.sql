-- Current-season gameweeks with FPL's official deadlines, from the latest snapshot.
select
    season,
    id as gameweek,
    deadline_time as deadline_utc,
    finished,
    is_current,
    is_next
from {{ source('fpl', 'gameweeks') }}
qualify snapshot_date = max(snapshot_date) over ()
