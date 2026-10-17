-- One row per player across seasons (FPL's stable `code`), with their latest name.
with seasons as (
    select season, player_code, element, element_type, web_name, first_name, second_name
    from {{ ref('stg_vaastav__players') }}
    union all
    select season, player_code, element, element_type, web_name, first_name, second_name
    from {{ ref('stg_fpl__players') }}
)

select
    player_code,
    web_name,
    first_name,
    second_name,
    case element_type when 1 then 'GKP' when 2 then 'DEF' when 3 then 'MID' when 4 then 'FWD' end
        as latest_position,
    season as latest_season,
    min(season) over (partition by player_code) as first_season,
    count(*) over (partition by player_code) as seasons_played
from seasons
qualify row_number() over (partition by player_code order by season desc) = 1
