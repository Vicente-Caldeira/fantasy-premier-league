-- One row per Premier League match since 2014-15: result, xG, odds and kickoff.
-- Bookmaker margins are removed proportionally to give implied probabilities.
with joined as (select * from {{ ref('int_matches_joined') }})

select
    match_key,
    season,
    home_team_key,
    away_team_key,
    fpl_fixture_id,
    understat_match_id,
    elo_match_id,
    gameweek,

    -- FPL and Opta kickoffs are UTC. Understat's date is the last resort (2014-15 and
    -- 2015-16 only) and can be a day out for evening games.
    coalesce(fpl_kickoff_utc, elo_kickoff_utc) as kickoff_utc,
    coalesce(cast(fpl_kickoff_utc as date), fd_match_date, us_match_date) as match_date,
    coalesce(fpl_finished, us_is_result, fd_home_goals is not null) as is_played,

    -- FPL-Elo-Insights goals are excluded: they contain known wrong scores.
    coalesce(fpl_home_goals, fd_home_goals, us_home_goals) as home_goals,
    coalesce(fpl_away_goals, fd_away_goals, us_away_goals) as away_goals,

    home_xg,
    away_xg,
    home_opta_xg,
    away_opta_xg,
    home_xgot,
    away_xgot,
    home_shots,
    away_shots,
    home_shots_on_target,
    away_shots_on_target,
    home_big_chances,
    away_big_chances,
    home_possession,
    away_possession,
    home_elo,
    away_elo,
    referee,

    odds_home,
    odds_draw,
    odds_away,
    odds_over_2_5,
    odds_under_2_5,
    close_odds_home,
    close_odds_draw,
    close_odds_away,
    close_odds_over_2_5,
    close_odds_under_2_5,
    odds_1x2_source,
    close_odds_1x2_source,

    (1 / odds_home) / (1 / odds_home + 1 / odds_draw + 1 / odds_away) as p_home_win,
    (1 / odds_draw) / (1 / odds_home + 1 / odds_draw + 1 / odds_away) as p_draw,
    (1 / odds_away) / (1 / odds_home + 1 / odds_draw + 1 / odds_away) as p_away_win,
    (1 / odds_over_2_5) / (1 / odds_over_2_5 + 1 / odds_under_2_5) as p_over_2_5,
    (1 / close_odds_home) / (1 / close_odds_home + 1 / close_odds_draw + 1 / close_odds_away)
        as close_p_home_win,
    (1 / close_odds_draw) / (1 / close_odds_home + 1 / close_odds_draw + 1 / close_odds_away)
        as close_p_draw,
    (1 / close_odds_away) / (1 / close_odds_home + 1 / close_odds_draw + 1 / close_odds_away)
        as close_p_away_win,
    (1 / close_odds_over_2_5) / (1 / close_odds_over_2_5 + 1 / close_odds_under_2_5)
        as close_p_over_2_5,

    forecast_home_win as understat_forecast_home_win,
    forecast_draw as understat_forecast_draw,
    forecast_away_win as understat_forecast_away_win
from joined
