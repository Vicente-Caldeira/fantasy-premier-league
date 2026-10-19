-- The live season must carry all 380 fixtures (blank/double gameweeks only move them).
select season, count(*) as fixtures
from {{ ref('fct_matches') }}
where season = (select max(season) from {{ ref('stg_fpl__fixtures') }})
group by season
having count(*) != 380
