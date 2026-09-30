"""Shape descriptors that do not assume the power law holds.

The physics kernel answers "how fast is this degrading". These answer "does it
degrade like its lot-mates at all" - step discontinuities, recovery humps, noise
inflation. They are what catches the mechanisms the power law cannot express.

`rank_mobility` deserves a note: it measures whether a part climbed its lot's
ranking during burn-in. A part that starts at the 40th percentile and ends at
the 98th is being overtaken by nothing and overtaking everything - that is a
latent defect signature even when its absolute value stays unremarkable. Almost
nobody computes it, and it is consistently one of the strongest single features.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / np.where(np.abs(b) < 1e-12, np.nan, b)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def shape_features(hours: np.ndarray, matrix: np.ndarray, noise_frac: float = 0.015) -> Dict[str, np.ndarray]:
    """Compute shape descriptors for a (n_parts, n_hours) matrix.

    `noise_frac` is the expected measurement noise as a fraction of v0, taken
    from the library. It sets the tolerance below which a dip is noise rather
    than a genuine non-monotonicity.
    """
    hours = np.asarray(hours, dtype=float)
    m = np.asarray(matrix, dtype=float)
    v0 = m[:, [0]]

    norm = _safe_div(m - v0, v0)                      # drift as fraction of v0
    dt = np.diff(hours)
    dv = np.diff(norm, axis=1)
    rates = _safe_div(dv, dt[None, :])                # per-hour normalised rate

    tol = 2.0 * noise_frac                            # two sigma of measurement noise

    out: Dict[str, np.ndarray] = {}
    out["total_drift_frac"] = norm[:, -1]
    out["max_drift_frac"] = np.nanmax(norm, axis=1)
    out["final_over_max"] = _safe_div(norm[:, -1], np.nanmax(np.abs(norm), axis=1))

    # Early vs late rate. Healthy power-law ageing decelerates (n < 1), so this
    # ratio sits below 1. A part that accelerates is running away.
    early = rates[:, 0]
    late = rates[:, -1]
    out["rate_early"] = early
    out["rate_late"] = late
    out["rate_accel"] = _safe_div(late, np.abs(early) + 1e-12)

    # Curvature: second difference of the normalised trajectory.
    if norm.shape[1] >= 3:
        out["curvature"] = norm[:, 2:] .mean(axis=1) - 2.0 * norm[:, 1:-1].mean(axis=1) + norm[:, :-2].mean(axis=1)
    else:
        out["curvature"] = np.zeros(m.shape[0])

    # Monotonicity: how far the trajectory moves backwards, beyond noise.
    backsteps = np.clip(-dv, 0.0, None)
    out["monotonic_violation"] = np.nansum(np.clip(backsteps - tol, 0.0, None), axis=1)
    out["n_backsteps"] = np.nansum(backsteps > tol, axis=1).astype(float)

    # Jump detector: the largest single step relative to the typical step.
    abs_dv = np.abs(dv)
    typical = np.nanmedian(abs_dv, axis=1, keepdims=True)
    out["jump_score"] = _safe_div(np.nanmax(abs_dv, axis=1), (typical[:, 0] + tol))
    out["max_step_frac"] = np.nanmax(abs_dv, axis=1)

    # Noise: scatter of the trajectory around its own smooth trend, in units of
    # the expected measurement noise. Bond-wire defects show up here and nowhere else.
    if norm.shape[1] >= 3:
        smooth = 0.5 * (norm[:, :-2] + norm[:, 2:])
        resid = norm[:, 1:-1] - smooth
        out["noise_ratio"] = _safe_div(np.nanstd(resid, axis=1), np.full(m.shape[0], noise_frac))
    else:
        out["noise_ratio"] = np.zeros(m.shape[0])

    return out


def rank_mobility(matrix: np.ndarray, lot_codes: np.ndarray) -> Dict[str, np.ndarray]:
    """Change in within-lot percentile rank between the first and last timepoint.

    Positive means the part climbed - it is degrading faster than its lot-mates.
    """
    m = np.asarray(matrix, dtype=float)
    n_rows, n_cols = m.shape
    ranks = np.zeros((n_rows, n_cols))

    for code in np.unique(lot_codes):
        sel = lot_codes == code
        block = m[sel]
        n = block.shape[0]
        if n < 2:
            ranks[sel] = 0.5
            continue
        order = np.argsort(np.argsort(block, axis=0), axis=0)
        ranks[sel] = order / (n - 1)

    mobility = ranks[:, -1] - ranks[:, 0]
    return {
        "rank_start": ranks[:, 0],
        "rank_end": ranks[:, -1],
        "rank_mobility": mobility,
        "rank_mobility_abs": np.abs(mobility),
        "rank_max": np.nanmax(ranks, axis=1),
    }
