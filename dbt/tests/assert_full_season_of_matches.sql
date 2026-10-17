-- Every season has exactly 380 matches. Because fct_matches is a full outer join of all
-- sources, an unmapped or mis-mapped club shows up here as extra rows.
select season, count(*) as matches
from {{ ref('fct_matches') }}
group by season
having count(*) != 380
