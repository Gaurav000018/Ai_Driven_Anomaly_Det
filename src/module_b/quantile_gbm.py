"""Quantile drift predictor.

A point prediction is the wrong output for a safety decision. A mean forecast of
40 uA against a 50 uA limit conceals that the 90th percentile is 58 uA - and the
part that flies is chosen on the tail, not the mean.

So the model predicts P10 / P50 / P90 directly, via pinball loss. LightGBM when
available, scikit-learn's quantile gradient boosting otherwise; both optimise
the same objective, so results are comparable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    _HAVE_LGB = True
except Exception:  # pragma: no cover
    _HAVE_LGB = False

from sklearn.ensemble import GradientBoostingRegressor

from .dataset import DriftDataset

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


class QuantileDriftModel:
    """One gradient-boosted model per quantile."""

    name = "quantile gbm"

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        random_state: int = 0,
        residual_target: bool = True,
    ) -> None:
        self.quantiles = tuple(quantiles)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        # Predict the DRIFT from the last measured point, not the absolute value.
        # v168 is dominated by the part's own level, which a tree can only
        # approximate with axis-aligned splits - so an unanchored model spends
        # its whole capacity relearning "v168 is about v24" and still loses to
        # ridge. Anchoring removes that component analytically and leaves the
        # trees to model only what is actually hard: how much it drifts.
        self.residual_target = residual_target
        self.models_: Dict[float, object] = {}
        self.columns_: List[str] = []
        self.anchor_col_: Optional[str] = None
        self.backend_ = "lightgbm" if _HAVE_LGB else "sklearn"

    def _make(self, q: float):
        if _HAVE_LGB:
            return lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=2**self.max_depth,
                min_child_samples=30,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                random_state=self.random_state,
                verbose=-1,
            )
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=30,
            subsample=0.9,
            random_state=self.random_state,
        )

    def _anchor(self, ds: DriftDataset, idx: np.ndarray) -> np.ndarray:
        if not self.residual_target or self.anchor_col_ is None:
            return np.zeros(len(idx), dtype=float)
        return ds.X.iloc[idx][self.anchor_col_].to_numpy(dtype=float)

    def fit(self, ds: DriftDataset, train_idx: np.ndarray) -> "QuantileDriftModel":
        X = ds.X.iloc[train_idx]
        self.anchor_col_ = f"v_{int(ds.input_hours[-1])}h" if self.residual_target else None
        y = ds.y.iloc[train_idx].to_numpy(dtype=float) - self._anchor(ds, train_idx)
        self.columns_ = list(X.columns)
        # Fit and predict both on DataFrames so the backends see consistent
        # feature names; mixing the two makes LightGBM warn on every call.
        Xv = X[self.columns_].astype(float)
        for q in self.quantiles:
            self.models_[q] = self._make(q).fit(Xv, y)
        return self

    def predict_quantiles(self, ds: DriftDataset, idx: np.ndarray) -> pd.DataFrame:
        X = ds.X.iloc[idx][self.columns_].astype(float)
        anchor = self._anchor(ds, idx)
        out = {f"q{int(q * 100)}": self.models_[q].predict(X) + anchor for q in self.quantiles}
        frame = pd.DataFrame(out, index=ds.X.iloc[idx].index)
        # Quantile crossing is a known artefact of fitting each level
        # independently. Sorting row-wise is the standard, monotonicity-restoring
        # fix and cannot make coverage worse.
        ordered = np.sort(frame.to_numpy(), axis=1)
        return pd.DataFrame(ordered, index=frame.index, columns=sorted(frame.columns, key=lambda c: int(c[1:])))

    def predict(self, ds: DriftDataset, idx: np.ndarray) -> np.ndarray:
        """Median prediction, for MAE against the baselines."""
        q = self.predict_quantiles(ds, idx)
        col = "q50" if "q50" in q.columns else q.columns[len(q.columns) // 2]
        return q[col].to_numpy(dtype=float)

    def feature_importance(self, quantile: float = 0.5) -> pd.Series:
        model = self.models_.get(quantile)
        if model is None:
            return pd.Series(dtype=float)
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            return pd.Series(dtype=float)
        return pd.Series(imp, index=self.columns_).sort_values(ascending=False)
