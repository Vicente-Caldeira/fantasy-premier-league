import pytest

from fpl_models import team_import as ti


@pytest.mark.parametrize(
    ("purchase", "current", "expected"),
    [(50, 50, 50), (50, 51, 50), (50, 52, 51), (50, 55, 52), (60, 57, 57)],
)
def test_selling_price_keeps_half_the_rise_and_all_of_a_fall(purchase, current, expected):
    assert ti.selling_price(purchase, current) == expected


def week(event, transfers):
    return {"event": event, "event_transfers": transfers}


def test_free_transfers_bank_up_to_five():
    history = [week(gw, 0) for gw in range(1, 9)]
    assert ti.free_transfers(history, [], next_gameweek=3) == 2
    assert ti.free_transfers(history, [], next_gameweek=9) == 5


def test_using_transfers_and_taking_hits():
    history = [week(1, 9), week(2, 1), week(3, 3), week(4, 0)]
    # GW2: 1 free, use 1 -> 1 at GW3; GW3: 1 free, use 3 (2 hits) -> 1 at GW4; GW4 bank -> 2.
    assert ti.free_transfers(history, [], next_gameweek=5) == 2


def test_wildcard_and_free_hit_keep_banked_transfers_without_earning_one():
    history = [week(1, 0), week(2, 0), week(3, 15), week(4, 15), week(5, 0)]
    chips = [{"name": "wildcard", "event": 3}, {"name": "freehit", "event": 4}]
    # GW2 bank -> 2; GW3 wildcard keeps 2 (no +1); GW4 free hit keeps 2; GW5 bank -> 3.
    assert ti.free_transfers(history, chips, next_gameweek=6) == 3


def test_replay_matches_a_real_managers_count():
    # A real 2026-27 history: bench boost GW2, wildcard GW3, triple captain GW4.
    # The manager confirmed FPL shows 3 free transfers for GW8.
    history = [week(1, 0), week(2, 1), week(3, 0), week(4, 1), week(5, 0), week(6, 0), week(7, 1)]
    chips = [
        {"name": "bboost", "event": 2},
        {"name": "wildcard", "event": 3},
        {"name": "3xc", "event": 4},
    ]
    assert ti.free_transfers(history, chips, next_gameweek=8) == 3


def test_import_uses_latest_purchase_price_for_selling_prices():
    data = ti.EntryData(
        history={"current": [week(1, 0), week(2, 1)], "chips": []},
        transfers=[
            {"element_in": 7, "element_in_cost": 60, "time": "2026-08-20T10:00:00Z"},
            {"element_in": 7, "element_in_cost": 62, "time": "2026-08-27T10:00:00Z"},
        ],
        picks={"picks": [{"element": 7}, {"element": 8}], "entry_history": {"bank": 15}},
    )
    lookup = ti.PlayerLookup(
        element_to_code={7: 700, 8: 800},
        current_prices={7: 66, 8: 45},
        gameweek_one_prices={8: 50},
    )
    team = ti.import_team(42, data, lookup, next_gameweek=3)
    assert team.squad == [700, 800]
    assert team.bank == 15
    assert team.selling_prices == {700: 64, 800: 45}  # 62 + (66-62)//2 ; fell 50 -> 45
    assert team.free_transfers == 1
