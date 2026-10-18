select season, id as element, code as player_code, element_type, web_name, first_name, second_name, team_code
from {{ source('vaastav', 'players') }}
