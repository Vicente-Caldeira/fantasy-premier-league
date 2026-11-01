-- Understat per player per match (2014-15 onwards), keyed to our matches. Not keyed to
-- FPL players: fpl_models.understat matches names within club and season.
select
    m.match_key,
    pm.*
from {{ ref('stg_understat__player_matches') }} as pm
inner join {{ ref('fct_matches') }} as m on pm.understat_match_id = m.understat_match_id
