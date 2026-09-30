"""Fit the degradation kinetics of each trajectory.

The model is the classical accelerated wear-out power law

    dV(t) = A * (t / t_ref) ** n

with n ~ 0.15-0.25 for healthy silicon (NBTI/HCI literature). The fitted triple
(A, n, R2) is the part's *Degradation Signature*, and it is the object the whole
system reasons about - not the raw value.

Two parts can sit at identical 24h readings and be completely different risks:
the one whose exponent is 0.42 against a lot of 0.18 is a gate-oxide pinhole
heading for runaway. Low R2 is itself a red flag - it means the part is
degrading by a mechanism that is not its lot's mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # scipy gives a better-conditioned fit, but the log-log fallback is fine
    from scipy.optimize import curve_fit

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

N_BOUNDS = (0.01, 3.0)
DEFAULT_N = 0.18


# The exponent is only identifiable when drift rises clearly above measurement
# noise. Below this signal-to-noise ratio the fit is shrunk toward the prior.
SNR_REF = 3.0


@dataclass
class PhysicsFit:
    A: float           # drift amplitude at t_ref, in the parameter's own units
    A_frac: float      # same, as a fraction of v0 - unit-free, comparable
    n: float           # kinetic exponent, shrunk toward the prior at low SNR
    n_raw: float       # unshrunk fit, for diagnostics
    r2: float          # how well the power law explains this part
    rmse: float        # residual scale, in the parameter's units
    snr: float         # |drift| / measurement noise
    identifiable: float  # 0-1 weight on the fitted exponent
    converged: bool


def _power(t_norm: np.ndarray, A: float, n: float) -> np.ndarray:
    return A * np.power(t_norm, n)


def fit_power_law(
    hours: np.ndarray,
    values: np.ndarray,
    t_ref: float = 168.0,
    noise_sigma: Optional[float] = None,
    n_prior: float = DEFAULT_N,
) -> PhysicsFit:
    """Fit dV = A * (t/t_ref)^n to a single trajectory.

    t = 0 is included in the fit: the model predicts exactly zero drift there,
    so keeping the point costs no parameters and buys a degree of freedom. With
    a 0/24/96/168 schedule that means 4 points against 2 parameters.

    The exponent is then shrunk toward `n_prior` by how identifiable it actually
    is. This matters more than it looks. A healthy part drifts 1-5% of v0 while
    measurement noise is ~1.5%, so its exponent is fitted almost entirely from
    noise and lands anywhere in [0.01, 3]. Left unshrunk, those junk exponents
    inflate the lot's spread until a genuine runaway at n = 0.45 no longer looks
    unusual - the kinetic signal is buried by parts that have no kinetics to
    measure. Weighting by SNR = |A| / noise collapses the healthy population onto
    the prior and lets real outliers stand clear.
    """
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(hours) & np.isfinite(values)
    hours, values = hours[ok], values[ok]

    if hours.size < 3:
        return PhysicsFit(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, False)

    order = np.argsort(hours)
    hours, values = hours[order], values[order]
    v0 = float(values[0])
    delta = values - v0
    t_norm = np.clip(hours, 0.0, None) / t_ref

    scale = max(abs(v0), 1e-9)
    amp0 = float(delta[-1]) if np.isfinite(delta[-1]) else 0.0

    A, n, converged = amp0, DEFAULT_N, False
    if _HAVE_SCIPY:
        try:
            # A is unbounded in sign so recovery/hump trajectories can fit a
            # negative amplitude instead of silently pinning at zero.
            popt, _ = curve_fit(
                _power,
                t_norm,
                delta,
                p0=[amp0 if abs(amp0) > 1e-12 else 1e-6 * scale, DEFAULT_N],
                bounds=([-np.inf, N_BOUNDS[0]], [np.inf, N_BOUNDS[1]]),
                maxfev=5000,
            )
            A, n = float(popt[0]), float(popt[1])
            converged = True
        except Exception:
            converged = False

    if not converged:
        # Log-log least squares on the strictly positive drift points.
        pos = (t_norm > 0) & (delta > 0)
        if pos.sum() >= 2:
            slope, intercept = np.polyfit(np.log(t_norm[pos]), np.log(delta[pos]), 1)
            n = float(np.clip(slope, *N_BOUNDS))
            A = float(np.exp(intercept))
            converged = True
        else:
            n, A = DEFAULT_N, float(amp0)

    n_raw = float(n)
    sigma = float(noise_sigma) if noise_sigma and noise_sigma > 0 else float("nan")
    if np.isfinite(sigma):
        snr = abs(float(A)) / sigma
        # Smooth gate: w -> 0 when drift is invisible, w -> 1 when it dominates.
        w = snr**2 / (snr**2 + SNR_REF**2)
    else:
        snr, w = float("nan"), 1.0
    n = w * n_raw + (1.0 - w) * n_prior

    # R2 measures whether the power-law FAMILY describes this part, so it is
    # computed against the unshrunk best fit. Shrinkage is a statement about
    # confidence in n, not about the model's shape.
    pred = _power(t_norm, A, n_raw)
    resid = delta - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((delta - delta.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-24 else 0.0
    rmse = float(np.sqrt(ss_res / max(len(delta), 1)))

    return PhysicsFit(
        A=float(A),
        A_frac=float(A / scale),
        n=float(n),
        n_raw=n_raw,
        r2=float(np.clip(r2, -1.0, 1.0)),
        rmse=rmse,
        snr=snr,
        identifiable=float(w),
        converged=bool(converged),
    )


def fit_matrix(
    hours: np.ndarray,
    matrix: np.ndarray,
    t_ref: float = 168.0,
    noise_frac: Optional[float] = None,
    n_prior: float = DEFAULT_N,
) -> dict:
    """Fit every row of a (n_parts, n_hours) matrix.

    `noise_frac` is the parameter's measurement noise as a fraction of v0, taken
    from the library; each row's noise scale is noise_frac * v0.
    Returned arrays line up with the rows of `matrix`.
    """
    n_rows = matrix.shape[0]
    keys = ("A", "A_frac", "n", "n_raw", "r2", "rmse", "snr", "identifiable")
    out = {k: np.full(n_rows, np.nan) for k in keys}
    out["converged"] = np.zeros(n_rows, dtype=bool)
    for i in range(n_rows):
        row = matrix[i]
        sigma = None
        if noise_frac:
            v0 = row[0] if np.isfinite(row[0]) else np.nanmedian(row)
            sigma = noise_frac * abs(float(v0))
        fit = fit_power_law(hours, row, t_ref=t_ref, noise_sigma=sigma, n_prior=n_prior)
        for k in keys:
            out[k][i] = getattr(fit, k)
        out["converged"][i] = fit.converged
    return out


def predict(hours: np.ndarray, A: float, n: float, v0: float, t_ref: float = 168.0) -> np.ndarray:
    """Extrapolate a fitted signature forward - used by the drift predictor."""
    t_norm = np.clip(np.asarray(hours, dtype=float), 0.0, None) / t_ref
    return v0 + _power(t_norm, A, n)
