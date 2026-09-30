"""Track A5 - kinetic outliers. The escapee catcher.

A1 asks "is this part's value strange for its lot". A5 asks the question that
actually predicts field failure: "is this part *degrading* by different physics
from its lot".

A part sitting at 18 uA inside a 50 uA limit is unremarkable to every static
rule. But if its fitted exponent is 0.42 while its lot sits at 0.18, it is on a
superlinear runaway - a gate-oxide pinhole - and it will breach the limit in
orbit, not on the test floor. That part is invisible to A1 at 24h and obvious
to A5.

Three sub-signals, combined worst-case per parameter:

  n_shift    the exponent runs hotter than the lot's
  r2_deficit the power law stops explaining the trajectory at all, meaning the
             part is degrading by a mechanism that is not its lot's mechanism
  accel      late drift rate exceeding early drift rate; healthy wear-out
             decelerates because n < 1, so acceleration is intrinsically wrong
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..features.robust_stats import mad
from .base import DetectorTrack

# Below this R2 the power law has stopped describing the part at all.
R2_HEALTHY = 0.90


class PhysicsResidualTrack(DetectorTrack):
    name = "physics_residual"
    track_id = "A5"

    def __init__(self, w_n: float = 1.0, w_r2: float = 0.8, w_accel: float = 0.5) -> None:
        self.w_n = w_n
        self.w_r2 = w_r2
        self.w_accel = w_accel
        self.params_: List[str] = []
        self.n_spread_: Dict[str, float] = {}

    def _parameters(self, features: pd.DataFrame) -> List[str]:
        return sorted({c.split("__")[0] for c in features.columns if c.endswith("__n")})

    def fit(self, features: pd.DataFrame) -> "PhysicsResidualTrack":
        self.params_ = self._parameters(features)
        # Spread of the exponent across the whole reference population, used as
        # the scale when a single lot is too small for its own MAD to be stable.
        for p in self.params_:
            shift = features[f"{p}__n_shift_vs_lot"].to_numpy(dtype=float)
            spread = float(mad(shift))
            self.n_spread_[p] = spread if np.isfinite(spread) and spread > 1e-6 else 0.02
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        if not self.params_:
            self.fit(features)

        per_param = []
        for p in self.params_:
            shift = features[f"{p}__n_shift_vs_lot"].to_numpy(dtype=float)
            r2 = features[f"{p}__r2"].to_numpy(dtype=float)
            accel = features[f"{p}__rate_accel"].to_numpy(dtype=float)

            # A part whose drift is invisible against measurement noise has no
            # kinetics to judge. Gating by identifiability stops A5 inventing
            # evidence from parts that simply are not degrading yet.
            ident = np.nan_to_num(features[f"{p}__identifiable"].to_numpy(dtype=float), nan=0.0)

            n_term = self._positive(shift / self.n_spread_[p])
            r2_term = self._positive(R2_HEALTHY - np.nan_to_num(r2, nan=0.0)) / (1.0 - R2_HEALTHY)
            accel_term = self._positive(np.nan_to_num(accel, nan=0.0) - 1.0)

            combined = self.w_n * n_term + self.w_r2 * r2_term + self.w_accel * accel_term
            # Gate the whole per-parameter score, not just two of its terms.
            # Taking a max across parameters amplifies noise, so any parameter
            # whose kinetics are unmeasurable must contribute nothing at all.
            per_param.append(ident * combined)

        worst = np.nanmax(np.vstack(per_param), axis=0)
        return pd.Series(np.nan_to_num(worst), index=features.index, name=self.name)

    def explain(self, features: pd.DataFrame, part_id) -> dict:
        """Report the driving parameter and the physics behind the score."""
        best, detail = None, None
        for p in self.params_:
            row = features.loc[part_id]
            shift = float(row[f"{p}__n_shift_vs_lot"])
            contribution = shift / self.n_spread_[p]
            if best is None or contribution > best[1]:
                n = float(row[f"{p}__n"])
                lot_n = float(row[f"{p}__lot_n_median"])
                r2 = float(row[f"{p}__r2"])
                best = (p, contribution)
                detail = {
                    "track": self.track_id,
                    "parameter": p,
                    "n": n,
                    "lot_n": lot_n,
                    "n_shift": shift,
                    "r2": r2,
                    "detail": (
                        f"fitted degradation exponent n = {n:.2f} against a lot median "
                        f"of {lot_n:.2f} (R2 = {r2:.2f})"
                    ),
                }
        return detail or {"track": self.track_id, "detail": "no physics fit available"}
