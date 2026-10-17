{{ config(severity='warn') }}
-- From 2024-25, every finished player-match with minutes should have Opta detail.
-- Gaps appear when FPL has a result before the next Elo-Insights snapshot.
select player_match_key, season, gameweek, element
from {{ ref('fct_player_matches') }}
where season >= '2024-25' and minutes > 0 and not is_provisional and not has_opta
