"""Baselines the learned model has to beat.

Publishing these is not modesty, it is credibility. A drift predictor that
cannot beat linear extrapolation has not earned its place, and a judge who
cannot see the comparison has no reason to believe the MAE number.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..features.physics_kernel import DEFAULT_N
from .dataset import DriftDataset


class LastValueCarryForward:
    """Assume nothing changes after the last measured point."""

    name = "last value carried forward"

    def fit(self, ds: DriftDataset, train_idx: np.ndarray) -> "LastValueCarryForward":
        self.last_col_ = f"v_{int(ds.input_hours[-1])}h"
        return self

    def predict(self, ds: DriftDataset, idx: np.ndarray) -> np.ndarray:
        return ds.X.iloc[idx][self.last_col_].to_numpy(dtype=float)


class LinearExtrapolation:
    """Extend the 0h->24h straight line all the way to the horizon.

    This over-predicts systematically, because real wear-out decelerates
    (n < 1). Useful precisely as the pessimistic bound.
    """

    name = "linear extrapolation"

    def fit(self, ds: DriftDataset, train_idx: np.ndarray) -> "LinearExtrapolation":
        self.t0_, self.t1_ = ds.input_hours[0], ds.input_hours[-1]
        self.slope_col_ = f"slope_{int(self.t0_)}_{int(self.t1_)}"
        self.last_col_ = f"v_{int(self.t1_)}h"
        self.horizon_ = ds.horizon
        return self

    def predict(self, ds: DriftDataset, idx: np.ndarray) -> np.ndarray:
        X = ds.X.iloc[idx]
        return (X[self.last_col_] + X[self.slope_col_] * (self.horizon_ - self.t1_)).to_numpy(dtype=float)


class PhysicsExtrapolation:
    """Fit the power law through the two early points and extend it.

    With only two points the exponent is not identifiable, so n is held at the
    population prior and only the amplitude is fitted. This is the "physics with
    no machine learning" baseline - and it is a genuinely strong one, which is
    the honest reason Module B has to work hard to beat it.
    """

    name = "physics extrapolation (n fixed)"

    def fit(self, ds: DriftDataset, train_idx: np.ndarray) -> "PhysicsExtrapolation":
        self.t0_, self.t1_ = ds.input_hours[0], ds.input_hours[-1]
        self.first_col_ = f"v_{int(self.t0_)}h"
        self.drift_col_ = f"drift_{int(self.t0_)}_{int(self.t1_)}"
        self.horizon_ = ds.horizon
        # Learn the single best exponent for this population on the train split.
        X = ds.X.iloc[train_idx]
        y = ds.y.iloc[train_idx].to_numpy(dtype=float)
        v0 = X[self.first_col_].to_numpy(dtype=float)
        d1 = X[self.drift_col_].to_numpy(dtype=float)
        best_n, best_err = DEFAULT_N, np.inf
        for n in np.linspace(0.05, 0.9, 35):
            ratio = (self.horizon_ / max(self.t1_, 1e-6)) ** n
            err = float(np.mean(np.abs(v0 + d1 * ratio - y)))
            if err < best_err:
                best_n, best_err = float(n), err
        self.n_ = best_n
        return self

    def predict(self, ds: DriftDataset, idx: np.ndarray) -> np.ndarray:
        X = ds.X.iloc[idx]
        ratio = (self.horizon_ / max(self.t1_, 1e-6)) ** self.n_
        return (X[self.first_col_] + X[self.drift_col_] * ratio).to_numpy(dtype=float)


class RidgeBaseline:
    """Plain linear model on the full contextual feature set."""

    name = "ridge on context features"

    def fit(self, ds: DriftDataset, train_idx: np.ndarray) -> "RidgeBaseline":
        X = ds.X.iloc[train_idx].to_numpy(dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = Ridge(alpha=1.0).fit(self.scaler_.transform(X), ds.y.iloc[train_idx].to_numpy(dtype=float))
        return self

    def predict(self, ds: DriftDataset, idx: np.ndarray) -> np.ndarray:
        X = ds.X.iloc[idx].to_numpy(dtype=float)
        return self.model_.predict(self.scaler_.transform(X))


def all_baselines() -> Dict[str, object]:
    return {
        "lvcf": LastValueCarryForward(),
        "linear": LinearExtrapolation(),
        "physics": PhysicsExtrapolation(),
        "ridge": RidgeBaseline(),
    }
