"""Assemble the per-part feature matrix.

One row per part. Column names are `{parameter}__{feature}`, so a model never
confuses an Iddq exponent with a t_pd exponent.

Every feature that can be expressed lot-relatively is, and that is the point of
the whole module. `Iddq__n` is a number; `Iddq__n_z` is how strange that number
is inside the population this part actually came from.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..knowledge.library import MechanismLibrary
from .physics_kernel import fit_matrix
from .robust_stats import robust_z, lot_percentile_rank
from .shape import rank_mobility, shape_features
from .trajectory import Panel, build_panel

# Features computed per parameter that are also worth expressing lot-relatively.
LOT_RELATIVE = [
    "n",
    "A_frac",
    "total_drift_frac",
    "noise_ratio",
    "jump_score",
    "rate_accel",
    "monotonic_violation",
]


def build_features(
    df: pd.DataFrame,
    library: MechanismLibrary,
    panel: Optional[Panel] = None,
) -> pd.DataFrame:
    """Return a per-part feature frame indexed by part_id."""
    panel = panel if panel is not None else build_panel(df)
    t_ref = library.time_ref_hours
    hours = panel.hours

    # Parts present for every parameter - a part missing a whole parameter
    # cannot be scored consistently, so it is excluded rather than imputed.
    per_param: Dict[str, pd.DataFrame] = {}
    common: Optional[pd.Index] = None
    for name in panel.param_names():
        sub = panel.param(name)
        idx = sub.index.get_level_values("part_id")
        per_param[name] = sub
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        raise ValueError("no parts have measurements for every parameter")
    common = pd.Index(sorted(set(common)), name="part_id")

    lot_of = panel.meta.loc[common, "lot_id"].astype(str)
    lot_codes = pd.Categorical(lot_of).codes

    blocks: List[pd.DataFrame] = []
    for name, sub in per_param.items():
        spec = library.parameter(name)
        sub = sub.reset_index(level="lot_id", drop=True).reindex(common)
        matrix = sub.to_numpy(dtype=float)

        feats: Dict[str, np.ndarray] = {}
        feats["v0"] = matrix[:, 0]
        feats["v_last"] = matrix[:, -1]

        fit = fit_matrix(hours, matrix, t_ref=t_ref, noise_frac=spec.meas_noise_frac)
        for key in ("A", "A_frac", "n", "n_raw", "r2", "rmse", "snr", "identifiable"):
            feats[key] = fit[key]
        feats["fit_converged"] = fit["converged"].astype(float)

        feats.update(shape_features(hours, matrix, noise_frac=spec.meas_noise_frac))
        feats.update(rank_mobility(matrix, lot_codes))

        # Raw value at every timepoint, plus its lot-relative z-score. The
        # brief's own example - 45 uA in a 10 uA lot - is exactly `v_{h}_z`.
        block = pd.DataFrame(feats, index=common)
        for j, h in enumerate(hours):
            col = f"v_{int(h)}h"
            block[col] = matrix[:, j]
            block[f"{col}_z"] = robust_z(block[col], lot_of).to_numpy()

        for col in LOT_RELATIVE:
            if col in block.columns:
                block[f"{col}_z"] = robust_z(block[col], lot_of).to_numpy()

        # Signed distance of this part's exponent from its lot's exponent. This
        # is the headline kinetic feature: the library's fingerprints are
        # written in exactly these units.
        lot_n = block.groupby(lot_of.to_numpy(), observed=True)["n"].transform("median")
        block["n_shift_vs_lot"] = block["n"] - lot_n
        block["lot_n_median"] = lot_n
        block["v_last_pct"] = lot_percentile_rank(block["v_last"], lot_of).to_numpy()

        # Headroom against the derated limit - how much margin is actually left.
        block["headroom_frac"] = (spec.derated_limit - block["v_last"]) / spec.derated_limit

        block.columns = [f"{name}__{c}" for c in block.columns]
        blocks.append(block)

    features = pd.concat(blocks, axis=1)
    features.insert(0, "lot_id", lot_of)
    for col in ("wafer_id", "x", "y", "is_defect", "mechanism_id", "severity"):
        if col in panel.meta.columns:
            features[col] = panel.meta.loc[common, col]

    return features.replace([np.inf, -np.inf], np.nan)


def feature_columns(features: pd.DataFrame) -> List[str]:
    """Model-visible columns: numeric, and never a label or an identifier."""
    drop = {"lot_id", "wafer_id", "x", "y", "is_defect", "mechanism_id", "severity"}
    return [
        c
        for c in features.columns
        if c not in drop and pd.api.types.is_numeric_dtype(features[c])
    ]


def matrix_for_model(features: pd.DataFrame, columns: Optional[List[str]] = None) -> tuple[np.ndarray, List[str]]:
    cols = columns if columns is not None else feature_columns(features)
    X = features[cols].to_numpy(dtype=float)
    # Median imputation keeps a single ragged part from dropping a whole lot.
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), cols
