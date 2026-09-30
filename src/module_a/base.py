"""Common interface for the Module A detector tracks.

Five deliberately heterogeneous detectors. The point is not that any one of them
is strong - it is that their mistakes are uncorrelated, so fusing them gives a
far lower false-negative rate than the best single track. Each returns a score
where higher means more anomalous, on whatever native scale suits it; the
fusion layer calibrates them onto a common footing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd


class DetectorTrack(ABC):
    name: str = "track"
    track_id: str = "A?"

    @abstractmethod
    def fit(self, features: pd.DataFrame) -> "DetectorTrack":
        """Learn whatever reference the track needs from a reference population."""

    @abstractmethod
    def score(self, features: pd.DataFrame) -> pd.Series:
        """Anomaly score, higher = more anomalous. Indexed like `features`."""

    def fit_score(self, features: pd.DataFrame) -> pd.Series:
        return self.fit(features).score(features)

    @staticmethod
    def _cols(features: pd.DataFrame, suffix: str) -> List[str]:
        """Every per-parameter column with the given suffix, e.g. '__n_z'."""
        return [c for c in features.columns if c.endswith(suffix)]

    @staticmethod
    def _positive(x: np.ndarray) -> np.ndarray:
        """One-sided: only drift in the damaging direction counts.

        A part that leaks far *less* than its lot-mates is odd but not a
        reliability risk, and treating it as one wastes the false-positive
        budget that the cost model says is precious.
        """
        return np.clip(x, 0.0, None)


def percentile_normalise(scores: pd.Series) -> pd.Series:
    """Map any score onto 0-1 by rank. Makes heterogeneous tracks comparable."""
    return scores.rank(pct=True, na_option="bottom").astype(float)
