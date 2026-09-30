"""Track A1 - dynamic outlier detection against the part's own lot.

This is the track that answers the brief's example directly: in a lot whose
median leakage is 10 uA, a part at 45 uA is a massive anomaly even though the
datasheet maximum is 50 uA. Static screening sees 45 < 50 and passes it. A1
sees roughly 24 MAD above the lot median and rejects it.

The score is the worst lot-relative deviation across every parameter and every
burn-in timepoint. Worst-case rather than average, because a part only has to be
anomalous in one respect to be a reliability risk.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import DetectorTrack


class RobustZTrack(DetectorTrack):
    name = "robust_z"
    track_id = "A1"

    def __init__(self, include_drift: bool = True) -> None:
        self.include_drift = include_drift
        self.columns_: List[str] = []

    def fit(self, features: pd.DataFrame) -> "RobustZTrack":
        # Lot-relative z-scores are computed in the feature fabric, so this
        # track carries no fitted state beyond which columns to read. That is
        # deliberate: it means A1 works on a single lot with no training data,
        # which is what a QA engineer actually has on day one.
        value_z = [c for c in features.columns if "__v_" in c and c.endswith("h_z")]
        drift_z = [c for c in features.columns if c.endswith("__total_drift_frac_z")]
        self.columns_ = value_z + (drift_z if self.include_drift else [])
        if not self.columns_:
            raise ValueError("no lot-relative z columns found; build features first")
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        if not self.columns_:
            self.fit(features)
        z = features[self.columns_].to_numpy(dtype=float)
        worst = np.nanmax(self._positive(z), axis=1)
        return pd.Series(np.nan_to_num(worst), index=features.index, name=self.name)

    def explain(self, features: pd.DataFrame, part_id) -> dict:
        """Which parameter and timepoint drove this part's score."""
        row = features.loc[part_id, self.columns_].astype(float)
        col = row.idxmax()
        param, _, rest = col.partition("__")
        return {
            "track": self.track_id,
            "parameter": param,
            "at": rest.replace("_z", ""),
            "robust_z": float(row.max()),
            "detail": f"{float(row.max()):.1f} MAD above the lot median",
        }
