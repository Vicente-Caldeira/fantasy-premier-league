import polars as pl

from fpl_models import prices


def test_moves_are_monotone_in_net_transfers_and_use_only_earlier_seasons():
    # 2021-22: heavy buying rises, heavy selling falls. 2022-23 is the target season.
    train = [("2021-22", gw, code, share, change)
             for gw in range(1, 30) for code, share, change in
             ((1, 0.02, 1), (2, -0.02, -1), (3, 0.0, 0), (4, 0.001, 0))]  # fmt: skip
    target = [("2022-23", 1, 1, 0.03, None), ("2022-23", 1, 2, -0.03, None),
              ("2022-23", 1, 3, 0.0, None)]  # fmt: skip
    changes = pl.DataFrame(
        train + target,
        schema=["season", "gameweek", "player_code", "crowd_net_transfer_share", "change"],
        orient="row",
    )
    moves = prices.expected_moves(changes, ["2022-23"]).sort("player_code")
    buyer, seller, quiet = moves.iter_rows(named=True)
    assert buyer["p_rise"] == 1.0
    assert buyer["move"] == 0.5  # half of a rise is kept
    assert seller["p_fall"] == 1.0
    assert seller["move"] == -1.0  # all of a fall is lost
    assert quiet["move"] == 0.0
    assert prices.by_gameweek(moves, "2022-23")[1] == {1: 0.5, 2: -1.0, 3: 0.0}


def test_price_changes_look_at_the_next_gameweek_of_the_same_season():
    features = pl.DataFrame({
        "season": ["2021-22", "2021-22", "2022-23"], "gameweek": [1, 2, 1],
        "player_code": [1, 1, 1], "price": [50, 51, 55],
        "crowd_net_transfer_share": [0.01, 0.0, 0.0],
    })  # fmt: skip
    out = prices.price_changes(features).sort("season", "gameweek")
    assert out["change"].to_list() == [1, None, None]
