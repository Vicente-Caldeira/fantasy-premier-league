-- Understat season totals per player (2014-15 onwards). Not yet keyed to FPL players:
-- Understat has no FPL identifier, so fpl_models.penalties matches names per club.
select * from {{ ref('stg_understat__player_seasons') }}
