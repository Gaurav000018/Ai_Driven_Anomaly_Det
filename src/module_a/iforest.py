"""Track A3 - non-parametric outlier detection in signature space.

A1, A2 and A5 all assume something: a direction, a covariance, a power law. A3
assumes nothing about the shape of normality. That is its entire job - to catch
defect geometry nobody anticipated, including mechanisms absent from the library.

Isolation Forest finds globally isolated points; Local Outlier Factor finds
points that are strange relative to their own neighbourhood. They disagree
often enough to be worth running both, and the maximum of the two is taken so a
part flagged by either survives into fusion.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from .base import DetectorTrack, percentile_normalise
from .signature_space import signature_columns, signature_matrix


class IsolationTrack(DetectorTrack):
    name = "isolation"
    track_id = "A3"

    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 300,
        n_neighbors: int = 35,
        use_lof: bool = True,
        random_state: int = 0,
    ) -> None:
        # Contamination is set a little above the true prevalence on purpose.
        # Under-stating it makes the forest treat real defects as normal.
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.use_lof = use_lof
        self.random_state = random_state
        self.columns_: List[str] = []
        self.scaler_: Optional[RobustScaler] = None
        self.iforest_: Optional[IsolationForest] = None
        self.lof_: Optional[LocalOutlierFactor] = None

    def fit(self, features: pd.DataFrame) -> "IsolationTrack":
        self.columns_ = signature_columns(features)
        X, _ = signature_matrix(features, self.columns_)
        # RobustScaler, not StandardScaler - the outliers would otherwise set
        # the scale they are then measured against.
        self.scaler_ = RobustScaler().fit(X)
        Xs = self.scaler_.transform(X)

        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(Xs)

        if self.use_lof:
            # novelty=True so the fitted neighbourhood can score unseen parts.
            self.lof_ = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, max(2, len(Xs) - 1)),
                contamination=self.contamination,
                novelty=True,
            ).fit(Xs)
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        if self.iforest_ is None:
            self.fit(features)
        X, _ = signature_matrix(features, self.columns_)
        Xs = self.scaler_.transform(X)

        # score_samples is higher for normal points, so negate.
        iso = pd.Series(-self.iforest_.score_samples(Xs), index=features.index)
        if self.lof_ is None:
            return iso.rename(self.name)

        lof = pd.Series(-self.lof_.score_samples(Xs), index=features.index)
        # The two live on different scales, so combine by rank before taking
        # the max - otherwise whichever has the wider range simply wins.
        combined = np.maximum(percentile_normalise(iso), percentile_normalise(lof))
        return pd.Series(combined, index=features.index, name=self.name)

    def explain(self, features: pd.DataFrame, part_id) -> dict:
        X, cols = signature_matrix(features, self.columns_)
        Xs = self.scaler_.transform(X)
        i = features.index.get_loc(part_id)
        row = Xs[i]
        order = np.argsort(np.abs(row))[::-1][:3]
        return {
            "track": self.track_id,
            "isolation_score": float(-self.iforest_.score_samples(row.reshape(1, -1))[0]),
            "top_deviations": [{"feature": cols[j], "scaled_value": float(row[j])} for j in order],
            "detail": "isolated from the bulk of the population in signature space",
        }
