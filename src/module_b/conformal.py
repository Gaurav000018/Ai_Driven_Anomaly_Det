"""Conformalized quantile regression (CQR).

Quantile models give an interval, but nothing guarantees that a nominal 80%
interval actually contains 80% of outcomes - gradient boosting is as capable of
being overconfident as anything else. CQR fixes that with a calibration set and
a distribution-free argument: for exchangeable data the calibrated interval has
marginal coverage of at least 1 - alpha, whatever the underlying model does.

That guarantee is the entire reason this module exists. It converts "the model
thinks 58 uA is the upper bound" into "at most 10% of parts like this exceed the
upper bound", which is a sentence a reliability engineer can actually sign.

Reference: Romano, Patterson & Candes (2019), Conformalized Quantile Regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ConformalInterval:
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    alpha: float
    correction: float

    def as_frame(self, index=None) -> pd.DataFrame:
        return pd.DataFrame(
            {"lower": self.lower, "median": self.median, "upper": self.upper},
            index=index,
        )


class ConformalCalibrator:
    """Split-conformal wrapper around a fitted quantile model.

    Calibration MUST use data the quantile model never saw, and - because lots
    are the unit of correlation here - ideally whole lots it never saw. The
    evaluation harness splits by lot for exactly this reason.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.correction_: float = 0.0
        self.n_calib_: int = 0

    def calibrate(self, lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> "ConformalCalibrator":
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        y = np.asarray(y_true, dtype=float)

        # Conformity score: how far outside the predicted band each point fell.
        # Negative when the point sat comfortably inside.
        scores = np.maximum(lower - y, y - upper)
        scores = scores[np.isfinite(scores)]
        n = scores.size
        if n == 0:
            raise ValueError("no finite conformity scores - calibration set is empty")

        # Finite-sample correction. Using the plain (1-alpha) empirical quantile
        # under-covers on small calibration sets; this rank is what makes the
        # guarantee hold exactly rather than asymptotically.
        rank = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        if rank >= n:
            # Too few calibration points to certify this alpha; fall back to the
            # widest observed score and say so via n_calib_.
            self.correction_ = float(np.max(scores))
        else:
            self.correction_ = float(np.sort(scores)[rank - 1])
        self.n_calib_ = n
        return self

    def apply(self, lower: np.ndarray, median: np.ndarray, upper: np.ndarray) -> ConformalInterval:
        return ConformalInterval(
            lower=np.asarray(lower, dtype=float) - self.correction_,
            median=np.asarray(median, dtype=float),
            upper=np.asarray(upper, dtype=float) + self.correction_,
            alpha=self.alpha,
            correction=self.correction_,
        )

    @property
    def certifiable(self) -> bool:
        """Whether the calibration set was large enough for the stated alpha."""
        return self.n_calib_ >= int(np.ceil(1.0 / self.alpha)) - 1


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def interval_report(y_true: np.ndarray, interval: ConformalInterval) -> dict:
    """Coverage and width together. Neither number means anything alone.

    Perfect coverage with an infinitely wide interval is useless; a tight
    interval that misses its target is dangerous.
    """
    cov = coverage(y_true, interval.lower, interval.upper)
    return {
        "nominal_coverage": 1.0 - interval.alpha,
        "empirical_coverage": cov,
        "coverage_gap": cov - (1.0 - interval.alpha),
        "mean_width": mean_width(interval.lower, interval.upper),
        "conformal_correction": interval.correction,
    }
