import polars as pl

from fpl_models.experiments import compare


def results(abs_error, captain=5.0, top11=50.0, weeks=30):
    return pl.DataFrame({
        "season": ["s"] * weeks, "gameweek": list(range(weeks)),
        "abs_error": [abs_error + (i % 3) for i in range(weeks)], "regular_rows": [100] * weeks,
        "captain": [captain + (i % 5) for i in range(weeks)], "top11": [top11] * weeks,
    })  # fmt: skip


def test_identical_results_have_zero_difference():
    stats = compare(results(180.0), results(180.0))
    assert stats["regulars_mae_delta"] == 0
    assert stats["regulars_mae_ci_low"] == stats["regulars_mae_ci_high"] == 0


def test_a_clearly_better_model_has_an_interval_below_zero():
    stats = compare(results(170.0), results(180.0))
    assert stats["regulars_mae_ci_high"] < 0
    assert stats["regulars_mae_delta"] == -0.1
