"""Forward consumer of the Defect Mechanism Library: synthesise labelled defects.

Every trajectory is built as

    value(t) = v0  +  nominal_drift(t)  +  subtlety * defect_excess(t)  +  noise

so a defect is literally *an excess on top of ordinary ageing*. Three properties
follow, and each one is load-bearing for evaluation:

  * `subtlety` scales only the defect term, which gives UDE-4 a clean knob for
    measuring Minimum Detectable Drift.
  * The fitted power-law exponent of the combined curve lands between the
    nominal and defect exponents, weighted by amplitude - so detection is not
    trivially "read off n", it has to survive mixing.
  * With `enforce_latent=True` the excess is rescaled until the whole trajectory
    sits below the datasheet limit. Every generated defect therefore PASSES
    static screening, which is the entire premise of the problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..knowledge.library import MechanismLibrary
from ..knowledge.schema import Mechanism, ParameterSpec

# A latent defect must stay below this fraction of the datasheet maximum.
LATENT_CEILING = 0.95


@dataclass
class Trajectory:
    """One part, one parameter, across the burn-in schedule."""

    hours: np.ndarray
    values: np.ndarray
    v0: float
    mechanism_id: str
    is_defect: bool
    severity: str
    subtlety: float
    truncated: bool  # the excess was rescaled to stay under the static limit


def _draw(rng: np.random.Generator, rng_range: Tuple[float, float]) -> float:
    lo, hi = rng_range
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _draw_params(rng: np.random.Generator, mech: Mechanism) -> Dict[str, float]:
    return {k: _draw(rng, v) for k, v in mech.physics.params.items()}


class DefectInjector:
    def __init__(self, library: MechanismLibrary, seed: Optional[int] = None) -> None:
        self.library = library
        self.t_ref = library.time_ref_hours
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- components

    def _nominal_drift(self, hours: np.ndarray, v0: float, n: float, a_frac: float) -> np.ndarray:
        """Ordinary wear-out: dV = A * (t / t_ref) ** n, zero at t = 0."""
        with np.errstate(invalid="ignore"):
            shape = np.power(np.clip(hours, 0.0, None) / self.t_ref, n)
        return a_frac * v0 * np.nan_to_num(shape)

    def _excess(
        self,
        form: str,
        params: Dict[str, float],
        hours: np.ndarray,
        v0: float,
    ) -> Tuple[np.ndarray, float]:
        """The defect-attributable component. Returns (excess, noise_multiplier)."""
        t = np.clip(hours, 0.0, None)
        noise_mult = 1.0

        if form == "power_law":
            shape = np.power(t / self.t_ref, params["n"])
            excess = params["A_frac"] * v0 * np.nan_to_num(shape)

        elif form == "step":
            excess = params["step_frac"] * v0 * (t >= params["step_hour"]).astype(float)

        elif form == "hump":
            peak, width = params["hump_peak_hour"], params["hump_width_hour"]
            g = np.exp(-0.5 * ((t - peak) / width) ** 2)
            g0 = float(np.exp(-0.5 * (peak / width) ** 2))
            # Subtract g(0) so the part still starts at a clean v0; the hump is
            # something burn-in reveals, not something visible at incoming test.
            excess = params["hump_frac"] * v0 * (g - g0)

        elif form == "variance_inflation":
            excess = np.zeros_like(t)
            noise_mult = params["noise_mult"]

        elif form == "early_offset":
            # Damage already present at t=0 - flat, not progressive.
            excess = np.full_like(t, params["offset_frac"] * v0)

        elif form == "exponential":
            tau = params["tau_hour"]
            denom = np.expm1(self.t_ref / tau)
            excess = params["A_frac"] * v0 * np.expm1(t / tau) / denom

        else:  # pragma: no cover - schema validation rejects unknown forms
            raise ValueError(f"unhandled physics form: {form}")

        return np.asarray(excess, dtype=float), float(noise_mult)

    # --------------------------------------------------------------- sampling

    def sample_v0(self, spec: ParameterSpec, lot_median: Optional[float] = None) -> float:
        median = spec.nominal_median if lot_median is None else lot_median
        return float(median * np.exp(self.rng.normal(0.0, spec.nominal_sigma_log)))

    def make_trajectory(
        self,
        hours: np.ndarray,
        spec: ParameterSpec,
        mechanism: Mechanism,
        v0: Optional[float] = None,
        lot_median: Optional[float] = None,
        lot_n_shift: float = 0.0,
        subtlety: float = 1.0,
        enforce_latent: bool = True,
    ) -> Trajectory:
        hours = np.asarray(hours, dtype=float)
        v0 = self.sample_v0(spec, lot_median) if v0 is None else float(v0)

        nominal = self.library.nominal
        nom_params = _draw_params(self.rng, nominal)
        n_nom = max(0.01, nom_params["n"] + lot_n_shift)
        base = v0 + self._nominal_drift(hours, v0, n_nom, nom_params["A_frac"])

        if not mechanism.defect:
            excess = np.zeros_like(hours)
            noise_mult = 1.0
        else:
            excess, noise_mult = self._excess(
                mechanism.physics.form, _draw_params(self.rng, mechanism), hours, v0
            )
        excess = excess * float(subtlety)

        truncated = False
        if enforce_latent and mechanism.defect:
            ceiling = spec.limit_hi * LATENT_CEILING
            peak = float(np.max(base + excess))
            if peak > ceiling:
                headroom = ceiling - float(np.max(base))
                span = float(np.max(excess))
                # If ordinary ageing alone already breaches the ceiling there is
                # no latent defect to make; drop the excess to zero and flag it.
                scale = max(0.0, headroom / span) if span > 0 else 0.0
                excess = excess * scale
                truncated = True

        values = base + excess
        sigma = spec.meas_noise_frac * v0 * noise_mult
        values = values + self.rng.normal(0.0, sigma, size=values.shape)
        values = np.clip(values, 1e-9, None)

        return Trajectory(
            hours=hours,
            values=values,
            v0=v0,
            mechanism_id=mechanism.id,
            is_defect=mechanism.defect,
            severity=mechanism.severity,
            subtlety=float(subtlety),
            truncated=truncated,
        )

    def pick_mechanism(self, parameter: str, allowed: Optional[list[str]] = None) -> Mechanism:
        """Choose a defect mechanism that actually affects this parameter."""
        candidates = [m for m in self.library.affecting(parameter) if allowed is None or m.id in allowed]
        if not candidates:
            raise ValueError(f"no defect mechanism in the library affects parameter '{parameter}'")
        return candidates[int(self.rng.integers(len(candidates)))]
