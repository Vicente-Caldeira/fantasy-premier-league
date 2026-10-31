"""E19: NVIDIA Kumo Tabular expected points, blended with production's.

Kumo Tabular (`nvidia/Kumo-Tabular`; code Apache-2.0, weights OpenMDW-1.1) is a
pretrained tabular foundation model. It is never trained on our data. At each refit it
caches a *context* of labelled rows, then predicts new rows in one forward pass
(in-context learning). The regressor returns 999 quantiles, and their mean is the
expected value.

- **Context:** the newest `context_rows` fixture rows whose deadline is strictly before
  the deadline being predicted. That's the same cut as `ComponentModel`, so no
  outcome at or after a predicted deadline is ever seen.
- **Label and features:** total points, with production's features.
- **Planning rows** (later gameweeks) use a second context without the crowd
  features, as production's planning model does.

Production (`expected_points`) blends it 50/50 with the component model: +31 points a
season in the held-out season simulation (docs/model-experiments.md, E19).

Optional dependency of `fpl-models`: the `kumo` extra (structured-data-models with
PyTorch, CPU-only on Linux), which the pipelines install. The weights (revision v1.0.1)
download from Hugging Face on first use, to `HF_HOME`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from fpl_models import player_model
from fpl_models.player_features import CROWD_ALL

FEATURES = player_model.PRODUCTION.features
PLANNING_FEATURES = tuple(c for c in FEATURES if c not in CROWD_ALL)
TARGET = "total_points"
CATEGORICAL = frozenset({"position_code"})


@dataclass(frozen=True)
class KumoConfig:
    weight: float = 0.5  # Kumo's share of the blend with production's expected points
    size: str = "small"  # small (28M parameters), medium or large (215M)
    context_rows: int = 50_000  # newest labelled fixture rows, about two seasons
    estimators: int = 1
    device: str = "cpu"
    seed: int = 0  # sdm's ensembling is random; seeded so every run is reproducible


PRODUCTION = KumoConfig()


def context_before(features: pl.DataFrame, deadline: datetime | None, rows: int) -> pl.DataFrame:
    """The newest `rows` labelled fixture rows with a deadline strictly before
    `deadline` (all of them when it's None, i.e. live: every played fixture)."""
    labelled = features.filter(pl.col(TARGET).is_not_null())
    if deadline is not None:
        labelled = labelled.filter(pl.col("deadline_utc") < deadline)
    return labelled.sort("deadline_utc", maintain_order=True).tail(rows)


def _arrow(frame: pl.DataFrame, columns: Sequence[str]) -> Any:  # noqa: ANN401 - pyarrow.Table
    return frame.select(
        pl.col(c).cast(pl.Int8) if frame.schema[c] == pl.Boolean else pl.col(c) for c in columns
    ).to_arrow()


def _generator(seed: int) -> Any:  # noqa: ANN401 - torch.Generator
    import torch  # noqa: PLC0415 - optional dependency

    return torch.Generator().manual_seed(seed)


class Regressor:
    """Kumo Tabular regressor on polars frames: `fit` caches a context, `predict`
    returns each row's expected value."""

    def __init__(self, config: KumoConfig, columns: Sequence[str]) -> None:
        from sdm.models import KumoTabular  # noqa: PLC0415 - optional dependency

        self.config, self.columns = config, tuple(columns)
        self._model = KumoTabular(task="regression", size=config.size, device=config.device)
        self._fitted = False

    def _tensor(self, frame: pl.DataFrame, columns: Sequence[str]) -> Any:  # noqa: ANN401
        import sdm  # noqa: PLC0415 - optional dependency

        stypes = {c: "categorical" if c in CATEGORICAL else "numerical" for c in columns}
        return sdm.TableTensor.from_arrow(_arrow(frame, columns), stypes, device=self.config.device)

    def fit(self, context: pl.DataFrame) -> None:
        if context.height == 0:
            raise ValueError("Kumo needs a non-empty context")
        self._model.clear()
        self._model.fit(
            self._tensor(context, self.columns),
            self._tensor(context, (TARGET,)),
            num_estimators=self.config.estimators,
            generator=_generator(self.config.seed),
        )
        self._fitted = True

    def predict(self, rows: pl.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit before predict")
        if rows.height == 0:
            return np.zeros(0)
        out = self._model.predict(self._tensor(rows, self.columns)).to_arrow()
        quantiles = [c for c in out.column_names if c.startswith("q")]
        return np.column_stack([out[c].to_numpy() for c in quantiles]).mean(axis=1)


def blend(production: pl.Series, kumo: pl.Series, weight: float) -> pl.Series:
    """(1 - weight) * production + weight * Kumo."""
    return (1 - weight) * production + weight * kumo


@dataclass
class KumoModel:
    """Backtest predictor (`backtest.Predictor`): Kumo's expected points per player,
    summed over his fixtures (0 without one). Refits on `ComponentModel`'s schedule,
    every `refit_every` deadlines counted over every deadline in `features`."""

    features: pl.DataFrame
    config: KumoConfig = PRODUCTION
    refit_every: int = 4
    name: str = "kumo"
    _regressor: Regressor | None = field(default=None, repr=False)
    _fitted_at: int = field(default=-(10**9), repr=False)

    def predict(self, history: pl.DataFrame, targets: pl.DataFrame) -> pl.Series:
        deadline = targets["deadline_utc"][0]
        order = self.features.select("deadline_utc").unique()["deadline_utc"]
        step = int((order < deadline).sum())
        if self._regressor is None:
            self._regressor = Regressor(self.config, FEATURES)
        if step - self._fitted_at >= self.refit_every:
            self._regressor.fit(context_before(self.features, deadline, self.config.context_rows))
            self._fitted_at = step
        season, gameweek = targets["season"][0], targets["gameweek"][0]
        rows = self.features.filter((pl.col("season") == season) & (pl.col("gameweek") == gameweek))
        per_player = (
            rows.select("player_code")
            .with_columns(pl.Series("kumo", self._regressor.predict(rows), dtype=pl.Float64))
            .group_by("player_code")
            .agg(pl.col("kumo").sum())
        )
        return targets.join(per_player, on="player_code", how="left", maintain_order="left")[
            "kumo"
        ].fill_null(0.0)


def fit_pair(
    features: pl.DataFrame,
    deadline: datetime | None,
    config: KumoConfig,
    regressors: tuple[Regressor, Regressor] | None = None,
) -> tuple[Regressor, Regressor]:
    """The next-gameweek regressor (all features) and the planning one (no crowd
    features), fitted on the same context. Pass `regressors` to refit them in place
    (the weights load once)."""
    nxt, planning = regressors or (
        Regressor(config, FEATURES),
        Regressor(config, PLANNING_FEATURES),
    )
    context = context_before(features, deadline, config.context_rows)
    nxt.fit(context)
    planning.fit(context)
    return nxt, planning
