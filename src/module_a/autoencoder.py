"""Track A4 - trajectory autoencoder.

Every other track compresses the trajectory into summary numbers first. A4 does
not: it learns to reconstruct the whole lot-relative drift curve through a
narrow bottleneck, and flags whatever it reconstructs badly.

That matters because the bottleneck can only learn the *shapes the population
actually exhibits*. A curve with a step in the middle, or one that rises and
recovers, has no representation in a latent space trained on smooth monotone
ageing - so it comes back wrong. Electromigration steps and contamination humps
are exactly this, and they are precisely the mechanisms the power law cannot
express.

Built on scikit-learn's MLP rather than a deep framework. The trajectory is
short - a handful of timepoints per parameter - so a small dense bottleneck is
the right capacity, and it keeps the project installable with one pip command.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

from .base import DetectorTrack
from .signature_space import trajectory_columns, trajectory_matrix


class TrajectoryAutoencoderTrack(DetectorTrack):
    name = "autoencoder"
    track_id = "A4"

    def __init__(
        self,
        bottleneck: int = 3,
        hidden: int = 16,
        max_iter: int = 400,
        random_state: int = 0,
        trim_quantile: float = 0.98,
    ) -> None:
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.max_iter = max_iter
        self.random_state = random_state
        # Train on the cleanest part of the population. An autoencoder trained
        # on data containing the anomalies learns to reconstruct them too, and
        # then cannot flag them - the classic self-defeating failure mode.
        self.trim_quantile = trim_quantile
        self.columns_: List[str] = []
        self.scaler_: Optional[RobustScaler] = None
        self.model_: Optional[MLPRegressor] = None

    def fit(self, features: pd.DataFrame) -> "TrajectoryAutoencoderTrack":
        self.columns_ = trajectory_columns(features)
        if not self.columns_:
            raise ValueError("no lot-relative trajectory columns found; build features first")
        X, _ = trajectory_matrix(features, self.columns_)
        self.scaler_ = RobustScaler().fit(X)
        Xs = self.scaler_.transform(X)

        # Trim the most extreme rows by overall magnitude before training.
        magnitude = np.linalg.norm(Xs, axis=1)
        cutoff = np.quantile(magnitude, self.trim_quantile)
        Xtrain = Xs[magnitude <= cutoff]
        if len(Xtrain) < 50:
            Xtrain = Xs

        self.model_ = MLPRegressor(
            hidden_layer_sizes=(self.hidden, self.bottleneck, self.hidden),
            activation="tanh",
            solver="adam",
            learning_rate_init=3e-3,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=self.random_state,
        ).fit(Xtrain, Xtrain)
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            self.fit(features)
        X, _ = trajectory_matrix(features, self.columns_)
        Xs = self.scaler_.transform(X)
        recon = self.model_.predict(Xs)
        err = np.sqrt(np.mean((Xs - recon) ** 2, axis=1))
        return pd.Series(err, index=features.index, name=self.name)

    def explain(self, features: pd.DataFrame, part_id) -> dict:
        X, cols = trajectory_matrix(features, self.columns_)
        Xs = self.scaler_.transform(X)
        i = features.index.get_loc(part_id)
        recon = self.model_.predict(Xs[i].reshape(1, -1))[0]
        resid = Xs[i] - recon
        order = np.argsort(np.abs(resid))[::-1][:3]
        return {
            "track": self.track_id,
            "reconstruction_error": float(np.sqrt(np.mean(resid**2))),
            "worst_points": [{"feature": cols[j], "residual": float(resid[j])} for j in order],
            "detail": "trajectory shape has no counterpart in this population",
        }
