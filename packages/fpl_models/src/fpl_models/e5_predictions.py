"""Row-level v2 backtest predictions for E5 (calibration). 2020-21 is warm-up only."""

from pathlib import Path

from fpl_models import player_features, player_model, team_strength
from fpl_models.backtest import run_backtest
from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks

OUT = Path("data/experiments/e5/predictions.parquet")


def main() -> None:
    matches = team_strength.load_matches(DEFAULT_WAREHOUSE)
    features = player_features.build_features(
        player_features.load_fixture_rows(DEFAULT_WAREHOUSE),
        player_features.match_lambdas(matches),
        player_features.schedule_features(matches),
    )
    gameweeks = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    model = player_model.ComponentModel(features, config=player_model.PRODUCTION, name="v2")
    predictions = run_backtest(model, gameweeks, ("2020-21", "2021-22", "2022-23")).predictions
    predictions.write_parquet(OUT)
    print(predictions.columns, predictions.height)


if __name__ == "__main__":
    main()
