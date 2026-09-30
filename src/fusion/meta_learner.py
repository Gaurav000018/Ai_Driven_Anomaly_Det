"""Fuse the five Module A tracks and the Module B margin into one risk score.

Averaging the tracks does not work, and the Phase 1 numbers showed exactly why:
a rank mean of A1 (PR-AUC 0.805) and A5 (0.132) scored 0.456 - worse than A1
alone. An unweighted average lets a weak track dilute a strong one. The fusion
has to *learn* which track to believe, and when.

Two correctness requirements, both easy to get wrong:

  out-of-fold scores  the tracks are themselves fitted models. Training the
                      meta-learner on scores the tracks produced for parts they
                      were fitted on teaches it to trust in-sample confidence.
                      `run_tracks_oof` refits every track per fold.

  group by lot        folds split on lot_id, never on part. Lot statistics are
                      shared within a lot, so a part-level split leaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..eval.harness import lot_folds
from ..module_a.base import DetectorTrack, percentile_normalise


def run_tracks_oof(
    features: pd.DataFrame,
    tracks: Sequence[DetectorTrack],
    lots: pd.Series,
    n_splits: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """Out-of-fold track scores: each part scored by tracks that never saw it."""
    out = pd.DataFrame(index=features.index, columns=[t.name for t in tracks], dtype=float)
    for k, (train_idx, test_idx) in enumerate(lot_folds(lots, n_splits=n_splits)):
        train = features.iloc[train_idx]
        test = features.iloc[test_idx]
        for track in tracks:
            fitted = track.__class__(**_track_kwargs(track))
            fitted.fit(train)
            out.iloc[test_idx, out.columns.get_loc(track.name)] = fitted.score(test).to_numpy()
        if verbose:
            print(f"  fold {k + 1}: scored {len(test_idx):,} parts")
    return out.astype(float)


def _track_kwargs(track: DetectorTrack) -> Dict:
    """Reconstruct a track's constructor arguments so each fold gets a fresh one."""
    skip = {"columns_", "estimator_", "scaler_", "model_", "iforest_", "lof_",
            "params_", "n_spread_", "backend_"}
    return {k: v for k, v in vars(track).items() if not k.endswith("_") and k not in skip}


@dataclass
class FusionReport:
    weights: pd.Series
    n_defect: int
    n: int


class RiskFusion:
    """Calibrated stack over the track scores, emitting Escape Risk 0-100."""

    def __init__(self, C: float = 0.5, random_state: int = 0) -> None:
        self.C = C
        self.random_state = random_state
        self.columns_: List[str] = []
        self.centre_: Optional[pd.Series] = None
        self.spread_: Optional[pd.Series] = None
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[LogisticRegression] = None
        self.calibrator_: Optional[IsotonicRegression] = None
        self.report_: Optional[FusionReport] = None

    def _fit_transform_stats(self, scores: pd.DataFrame) -> None:
        """Learn a per-track robust location and scale.

        Ranking was the first thing tried here and it cost the fusion its best
        track. The tracks' discriminative power lives in their TAILS - A4 flags
        a handful of parts with enormous reconstruction error - and a percentile
        transform compresses "enormous" and "merely large" into 0.9999 against
        0.9995. No linear model can recover a separation that the features no
        longer contain.

        Robust z keeps the magnitude while still putting a Mahalanobis distance
        and a reconstruction error on a comparable footing. The upper clip only
        bounds the influence of a single extreme part on the fitted weights.
        """
        from ..features.robust_stats import mad

        self.centre_ = scores.median()
        spread = scores.apply(lambda s: float(mad(s.to_numpy(dtype=float))))
        # A track that is constant on the reference population carries no
        # information; a floor keeps it from producing infinities.
        self.spread_ = spread.where(spread > 1e-9, 1.0)

    def _prepare(self, scores: pd.DataFrame) -> pd.DataFrame:
        z = (scores - self.centre_) / self.spread_
        return z.clip(lower=-5.0, upper=50.0).fillna(0.0)

    def fit(self, scores: pd.DataFrame, y: np.ndarray, lots: Optional[pd.Series] = None) -> "RiskFusion":
        self._fit_transform_stats(scores)
        X = self._prepare(scores)
        self.columns_ = list(X.columns)
        y = np.asarray(y).astype(int)

        self.scaler_ = StandardScaler().fit(X.to_numpy(dtype=float))
        Xs = self.scaler_.transform(X.to_numpy(dtype=float))

        # class_weight balanced is essential at 1% prevalence; without it the
        # model minimises loss by predicting "good" for everything.
        self.model_ = LogisticRegression(
            C=self.C, class_weight="balanced", max_iter=2000, random_state=self.random_state
        ).fit(Xs, y)

        # Isotonic calibration on out-of-fold decision values, so the emitted
        # risk is monotone in evidence without claiming to be a probability.
        raw = np.zeros(len(y))
        if lots is not None:
            for train_idx, test_idx in lot_folds(lots, n_splits=5):
                inner = LogisticRegression(
                    C=self.C, class_weight="balanced", max_iter=2000, random_state=self.random_state
                ).fit(Xs[train_idx], y[train_idx])
                raw[test_idx] = inner.decision_function(Xs[test_idx])
        else:
            raw = self.model_.decision_function(Xs)
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)

        self.report_ = FusionReport(
            weights=pd.Series(self.model_.coef_[0], index=self.columns_).sort_values(ascending=False),
            n_defect=int(y.sum()),
            n=len(y),
        )
        return self

    def risk(self, scores: pd.DataFrame) -> pd.Series:
        """Escape Risk on a 0-100 scale.

        Isotonic calibration is a step function, so on its own it collapses
        thousands of parts onto a handful of distinct values - 46% of a lot
        landing on exactly one number. The band optimiser then has nothing to
        cut between and the REVIEW band vanishes, not because reviewing is
        wrong but because no threshold separates the parts.

        So ties are broken by the underlying decision value. The nudge is far
        smaller than the gap between isotonic steps, which keeps the score
        monotone in the evidence and the calibration intact to six decimals,
        while restoring the granularity the decision layer needs.
        """
        X = self._prepare(scores)[self.columns_].to_numpy(dtype=float)
        raw = self.model_.decision_function(self.scaler_.transform(X))
        calibrated = self.calibrator_.predict(raw)
        tie_break = 1e-6 * pd.Series(raw, index=scores.index).rank(pct=True).to_numpy()
        risk = np.clip(calibrated + tie_break, 0.0, 1.0)
        return pd.Series(100.0 * risk, index=scores.index, name="escape_risk")

    def contributions(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Per-part, per-track contribution to the decision, for explainability."""
        X = self._prepare(scores)[self.columns_]
        Xs = self.scaler_.transform(X.to_numpy(dtype=float))
        contrib = Xs * self.model_.coef_[0]
        return pd.DataFrame(contrib, index=scores.index, columns=self.columns_)


class UnsupervisedFusion:
    """Fallback when no labels exist at all - a weighted rank mean.

    Deliberately kept as a separate class rather than a mode of RiskFusion. It
    is strictly worse and must never be mistaken for the trained fusion in a
    results table.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or {}

    def risk(self, scores: pd.DataFrame) -> pd.Series:
        ranks = scores.apply(percentile_normalise, axis=0)
        w = np.array([self.weights.get(c, 1.0) for c in ranks.columns], dtype=float)
        w = w / w.sum()
        return pd.Series(100.0 * (ranks.to_numpy() @ w), index=scores.index, name="escape_risk")
