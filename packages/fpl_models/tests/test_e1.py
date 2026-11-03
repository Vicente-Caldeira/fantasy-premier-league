"""Season simulation: the replicated protocol (perturbed predictions, replicate-level CIs)."""

import numpy as np
import polars as pl
import pytest

from fpl_models import e1
from fpl_models.season_sim import Week


def week(gameweek):
    xp = pl.DataFrame({"player_code": [1, 2, 3], "gameweek": [gameweek] * 3,
                       "expected_points": [2.0, 3.0, 4.0]})  # fmt: skip
    return Week(gameweek, pl.DataFrame(), xp, pl.DataFrame(), {}, {})


def test_perturbation_is_one_persistent_shift_per_player_and_reproducible():
    weeks = [week(1), week(2)]
    a, b = e1.perturb(weeks, seed=3), e1.perturb(weeks, seed=3)
    shift = [
        w.xp["expected_points"] - o.xp["expected_points"] for w, o in zip(a, weeks, strict=True)
    ]
    assert np.allclose(shift[0], shift[1])  # the same shift every week
    assert a[0].xp.equals(b[0].xp)
    assert not e1.perturb(weeks, seed=4)[0].xp.equals(a[0].xp)


def test_replicated_comparison_averages_paired_replicates(monkeypatch):
    rows = [
        {"config": e1.replicate_label(name, r), "season": "2025-26", "seed": s,
         "points": 2000 + 25 * r + s + (10 if name == "cand" else 0), "hits": 0}
        for name in ("cand", "base") for r in range(3) for s in range(5)
    ]  # fmt: skip
    monkeypatch.setattr(e1, "totals", lambda: pl.DataFrame(rows))
    mean, low, high = e1.compare_replicated("cand", "base", ["2025-26"], replicates=3, draws=2000)
    assert mean == pytest.approx(10.0)
    assert low <= 10.0 <= high


def test_a_failing_path_is_reported_not_raised(monkeypatch):
    def boom(job):
        raise RuntimeError("no squad")

    monkeypatch.setattr(e1, "_play_path", boom)
    job = e1.Job("v2now", "2025-26", 3, 7, 4.0, "v2nowR0")
    assert e1._play(job).startswith("FAILED v2nowR0 2025-26 seed 7")
