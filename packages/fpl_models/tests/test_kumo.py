"""E19 Kumo plumbing, with a fake `sdm` (the real model needs the `kumo` extra and weights)."""

import sys
import types
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from fpl_models import kumo
from fpl_models import player_features as pf
from fpl_models import player_model as pm

START = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)


class FakeTable:
    def __init__(self, table, stypes):
        self.table, self.stypes = table, stypes

    @classmethod
    def from_arrow(cls, table, stypes, device=None):
        return cls(table, stypes)


class FakeKumo:
    """Predicts the context's mean label plus price / 100, as three quantiles."""

    fits: ClassVar[list] = []

    def __init__(self, task=None, size="large", device=None):
        self.task, self.size, self.mean = task, size, None

    def clear(self):
        self.mean = None

    def fit(self, x, y, num_estimators=None, generator=None):
        FakeKumo.fits.append((x, y, num_estimators, generator))
        self.mean = float(np.mean(y.table.column(0).to_numpy()))

    def predict(self, x):
        assert self.mean is not None, "predict after clear"
        centre = self.mean + x.table.column("price").to_numpy() / 100
        out = pa.table({"q001": centre - 1, "q500": centre, "q999": centre + 1})
        return types.SimpleNamespace(to_arrow=lambda: out)


@pytest.fixture
def fake_sdm(monkeypatch):
    FakeKumo.fits = []
    sdm = types.ModuleType("sdm")
    sdm.TableTensor = FakeTable
    models = types.ModuleType("sdm.models")
    models.KumoTabular = FakeKumo
    sdm.models = models
    monkeypatch.setitem(sys.modules, "sdm", sdm)
    monkeypatch.setitem(sys.modules, "sdm.models", models)
    monkeypatch.setattr(kumo, "_generator", lambda seed: ("generator", seed))
    return FakeKumo


def feature_frame(gameweeks=6, players=4):
    rows = [
        {
            "season": "2024-25", "gameweek": gw, "player_code": p,
            "deadline_utc": START + timedelta(days=7 * gw),
            "total_points": float(gw + p), "price": 50.0 + 10 * p,
        }
        for gw in range(1, gameweeks + 1)
        for p in range(players)
    ]  # fmt: skip
    frame = pl.DataFrame(rows)
    extra = [c for c in kumo.FEATURES if c not in frame.columns]
    return frame.with_columns(
        pl.lit(0.5).alias(c) for c in extra if c not in ("position_code", "is_home")
    ).with_columns(pl.lit(1, dtype=pl.Int8).alias("position_code"), pl.lit(True).alias("is_home"))


def test_feature_sets_match_production_and_planning_drops_the_crowd():
    assert pm.PRODUCTION.features == kumo.FEATURES
    assert set(kumo.FEATURES) - set(kumo.PLANNING_FEATURES) == {
        c for c in kumo.FEATURES if c.startswith("crowd_")
    }
    assert not any(c.startswith("crowd_") for c in kumo.PLANNING_FEATURES)


def test_context_is_the_newest_labelled_rows_strictly_before_the_deadline():
    frame = feature_frame().with_columns(
        pl.when(pl.col("gameweek") == 3).then(None).otherwise(pl.col("total_points"))
        .alias("total_points")
    )  # fmt: skip
    deadline = START + timedelta(days=7 * 5)
    context = kumo.context_before(frame, deadline, rows=6)
    assert context.height == 6
    assert context["deadline_utc"].max() < deadline
    assert context["gameweek"].to_list() == [2, 2, 4, 4, 4, 4]  # GW3 unlabelled
    assert kumo.context_before(frame, None, rows=100).height == 20


def test_regressor_returns_the_mean_of_quantiles_and_casts_booleans(fake_sdm):
    frame = feature_frame()
    regressor = kumo.Regressor(kumo.KumoConfig(seed=7), kumo.FEATURES)
    with pytest.raises(RuntimeError):
        regressor.predict(frame)
    regressor.fit(frame)
    x, y, estimators, generator = fake_sdm.fits[-1]
    assert x.table.schema.field("is_home").type == pa.int8()
    assert x.stypes["position_code"] == "categorical"
    assert x.stypes["price"] == "numerical"
    assert y.table.column_names == ["total_points"]
    assert (estimators, generator) == (1, ("generator", 7))
    mean = frame["total_points"].mean()
    np.testing.assert_allclose(regressor.predict(frame.head(2)), mean + np.array([0.5, 0.6]))
    assert regressor.predict(frame.head(0)).size == 0


def test_backtest_model_refits_on_schedule_and_never_sees_the_deadline(fake_sdm):
    frame = feature_frame(gameweeks=6)
    model = kumo.KumoModel(frame, kumo.KumoConfig(context_rows=1_000), refit_every=2)
    for gw in (3, 4, 5, 6):
        targets = frame.filter(pl.col("gameweek") == gw).select(
            "season", "gameweek", "player_code", "deadline_utc"
        )
        targets = pl.concat(
            [targets, targets.head(1).with_columns(player_code=pl.lit(99, dtype=pl.Int64))]
        )
        predicted = model.predict(frame.head(0), targets)
        assert predicted.len() == targets.height
        assert predicted[-1] == 0.0  # no fixture row: 0, as in ComponentModel
    # Fits before GW3 (context GW1-2) and GW5 (GW1-4): each strictly before its deadline.
    assert [fit[1].table.num_rows for fit in fake_sdm.fits] == [8, 16]


def test_live_blend_weights_kumo_and_scales_it_by_availability(fake_sdm):
    rows = []
    rng = np.random.default_rng(0)
    for gw in range(1, 9):
        kickoff = START + timedelta(days=7 * gw)
        for p in range(6):
            minutes = int(rng.choice([0, 30, 90]))
            rows.append({
                "season": "2024-25", "gameweek": gw, "kickoff_utc": kickoff,
                "match_key": f"m{gw}-{p % 2}", "player_code": p, "position": pf.POSITIONS[p % 4],
                "team_key": f"t{p % 2}", "is_home": p % 2 == 0, "price": 50 + p,
                "is_provisional": False, "minutes": minutes, "starts": int(minutes >= 60),
                "goals_scored": int(rng.random() < 0.2), "assists": int(rng.random() < 0.2),
                "bonus": int(rng.integers(0, 4)), "bps": int(rng.integers(0, 40)),
                "saves": int(rng.integers(0, 5)), "yellow_cards": int(rng.random() < 0.1),
                "total_points": int(rng.integers(0, 12)),
                "expected_goals": float(rng.random()), "expected_assists": float(rng.random()),
                "tackles": int(rng.integers(0, 6)),
                "clearances_blocks_interceptions": int(rng.integers(0, 8)),
                "recoveries": int(rng.integers(0, 8)), "opponent_team_key": f"t{(p + 1) % 2}",
                "touches_opposition_box": int(rng.integers(0, 8)),
                "chances_created": int(rng.integers(0, 4)), "total_shots": int(rng.integers(0, 5)),
                "opta_xgot": float(rng.random()), "big_chances_missed": int(rng.integers(0, 2)),
                "selected": int(rng.integers(1_000, 1_000_000)),
                "transfers_balance": int(rng.integers(-50_000, 50_000)),
            })  # fmt: skip
    played = pf.add_deadlines(pl.DataFrame(rows))
    matches = played.filter(pl.col("is_home")).select(
        "match_key", "kickoff_utc",
        pl.col("team_key").alias("home_team_key"),
        pl.col("opponent_team_key").alias("away_team_key"),
    ).unique()  # fmt: skip
    upcoming = pl.DataFrame({
        "match_key": ["n1", "n2"], "season": ["2024-25"] * 2, "gameweek": [9, 10],
        "kickoff_utc": [START + timedelta(days=70), START + timedelta(days=77)],
        "home_team_key": ["t0", "t1"], "away_team_key": ["t1", "t0"],
        "lambda_home": [1.6, 1.3], "lambda_away": [1.0, 1.2],
    })  # fmt: skip
    schedule = pf.schedule_features(pl.concat([matches, upcoming.select(matches.columns)]))
    keys = played["match_key"].unique()
    lams = pl.DataFrame({"match_key": keys, "lambda_home": 1.5, "lambda_away": 1.1})
    features = pf.build_features(played, lams, schedule)
    live = pl.DataFrame({
        "player_code": list(range(6)), "web_name": list("abcdef"),
        "position": [pf.POSITIONS[p % 4] for p in range(6)],
        "team_key": [f"t{p % 2}" for p in range(6)], "price": [50 + p for p in range(6)],
        "status": ["a"] * 5 + ["u"], "chance_of_playing_next_round": [None, 50, *[None] * 4],
        "selected_by_percent": ["10.0"] * 6, "transfers_in_event": [10, 0, 5, 0, 3, 1],
        "transfers_out_event": [0, 4, 0, 2, 0, 0],
    })  # fmt: skip
    config = replace(pm.PRODUCTION, max_iter=10, min_samples_leaf=2)

    def run(weight):
        return pm.predict_upcoming(
            features, live, upcoming, [9, 10], config=config, schedule=schedule,
            kumo=None if weight is None else kumo.KumoConfig(weight=weight),
        ).sort("player_code", "gameweek")  # fmt: skip

    base, none, half, full = run(None), run(0.0), run(0.5), run(1.0)
    assert none["expected_points"].to_list() == pytest.approx(base["expected_points"].to_list())
    mean = features["total_points"].mean()
    kumo_xp = (mean + full["price"] / 100) * full["availability"]  # one fixture per gameweek
    assert full["expected_points"].to_list() == pytest.approx(kumo_xp.to_list())
    blended = 0.5 * base["expected_points"] + 0.5 * kumo_xp
    assert half["expected_points"].to_list() == pytest.approx(blended.to_list())
    # Flagged 'u' is zeroed, and 50% next gameweek halves Kumo's points too.
    assert full.filter(pl.col("player_code") == 5)["expected_points"].to_list() == [0.0, 0.0]
    p1 = full.filter(pl.col("player_code") == 1)["expected_points"].to_list()
    assert p1[0] == pytest.approx(0.5 * (mean + 0.51))
    assert p1[1] == pytest.approx(mean + 0.51)
    # Next gameweek uses all features, later gameweeks the no-crowd planning features.
    used = [len(fit[0].table.column_names) for fit in fake_sdm.fits[-2:]]
    assert used == [len(kumo.FEATURES), len(kumo.PLANNING_FEATURES)]
