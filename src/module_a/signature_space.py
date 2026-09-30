"""The shared feature view that tracks A2, A3 and A4 operate on.

All three are multivariate detectors, and they must see the same thing for the
fusion layer's diversity argument to mean anything. Two views exist:

  signature  the physics-parameterised description of each part - exponent
             shift, amplitude, drift, noise, jumps, rank mobility - across
             every parameter. This is where correlated multi-parameter drift
             lives, and it is what A2 and A3 read.

  trajectory the normalised drift curve itself, concatenated across
             parameters. No parametric assumption at all, which is what lets
             A4 catch shapes the power law cannot express.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Per-parameter signature components, in lot-relative form wherever available.
SIGNATURE_SUFFIXES = [
    "__n_shift_vs_lot",
    "__A_frac_z",
    "__total_drift_frac_z",
    "__noise_ratio_z",
    "__jump_score_z",
    "__monotonic_violation_z",
    "__rank_mobility",
    "__r2",
    "__identifiable",
]


def signature_columns(features: pd.DataFrame) -> List[str]:
    cols = [c for c in features.columns for s in SIGNATURE_SUFFIXES if c.endswith(s)]
    return sorted(set(cols))


def signature_matrix(features: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    cols = columns if columns is not None else signature_columns(features)
    # Copy: the slice can be a read-only view, and we impute into it below.
    X = np.array(features[cols].to_numpy(dtype=float), copy=True)
    med = np.nanmedian(X, axis=0)
    bad = np.where(np.isnan(X))
    X[bad] = np.take(med, bad[1])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), cols


def trajectory_columns(features: pd.DataFrame) -> List[str]:
    """The v_{h}h_z columns - lot-relative value at every timepoint."""
    return sorted(
        [c for c in features.columns if "__v_" in c and c.endswith("h_z")],
        key=lambda c: (c.split("__")[0], int(c.split("__v_")[1].split("h")[0])),
    )


def trajectory_matrix(features: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    cols = columns if columns is not None else trajectory_columns(features)
    X = features[cols].to_numpy(dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), cols
