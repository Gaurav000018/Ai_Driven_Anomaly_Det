"""Track A2 - correlated multi-parameter drift.

A1 looks at one number at a time and asks whether it is extreme. A2 asks a
harder question: is this *combination* of parameters extreme, given how they
normally move together?

Healthy silicon has correlated parameters. Iddq and leakage rise together;
propagation delay tracks threshold drift. A part whose Iddq climbs while its
delay does not is not extreme on either axis, and so is invisible to A1 - but
it has broken the correlation structure of its population, and that decoupling
is a real defect signature.

Covariance is estimated robustly (MinCovDet), because the outliers we are
hunting would otherwise inflate the very covariance used to judge them - the
same trap that mean-and-sigma falls into, one dimension up.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, MinCovDet

from .base import DetectorTrack
from .signature_space import signature_columns, signature_matrix


class MahalanobisTrack(DetectorTrack):
    name = "mahalanobis"
    track_id = "A2"

    def __init__(self, robust: bool = True, support_fraction: float = 0.85, random_state: int = 0) -> None:
        self.robust = robust
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.columns_: List[str] = []
        self.estimator_ = None

    def fit(self, features: pd.DataFrame) -> "MahalanobisTrack":
        self.columns_ = signature_columns(features)
        X, _ = signature_matrix(features, self.columns_)

        # Drop degenerate columns; a constant feature makes the covariance
        # singular and the distance meaningless.
        keep = np.nanstd(X, axis=0) > 1e-9
        self.columns_ = [c for c, k in zip(self.columns_, keep) if k]
        X = X[:, keep]

        if self.robust:
            try:
                self.estimator_ = MinCovDet(
                    support_fraction=self.support_fraction, random_state=self.random_state
                ).fit(X)
                return self
            except Exception:
                # MinCovDet can fail on near-singular data; Ledoit-Wolf shrinkage
                # always produces an invertible estimate.
                pass
        self.estimator_ = LedoitWolf().fit(X)
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        if self.estimator_ is None:
            self.fit(features)
        X, _ = signature_matrix(features, self.columns_)
        d2 = self.estimator_.mahalanobis(X)
        # Square root puts the score on a sigma-like scale, which keeps it
        # readable in a certificate.
        return pd.Series(np.sqrt(np.clip(d2, 0.0, None)), index=features.index, name=self.name)

    def explain(self, features: pd.DataFrame, part_id) -> dict:
        """Which signature components contributed most to the distance."""
        X, cols = signature_matrix(features, self.columns_)
        row = X[features.index.get_loc(part_id)]
        centre = getattr(self.estimator_, "location_", np.zeros_like(row))
        prec = getattr(self.estimator_, "precision_", np.eye(len(row)))
        diff = row - centre
        contrib = diff * (prec @ diff)
        order = np.argsort(contrib)[::-1][:3]
        return {
            "track": self.track_id,
            "distance": float(np.sqrt(max(float(diff @ prec @ diff), 0.0))),
            "top_contributors": [{"feature": cols[i], "contribution": float(contrib[i])} for i in order],
            "detail": "the combination of parameters departs from how this population normally co-varies",
        }
