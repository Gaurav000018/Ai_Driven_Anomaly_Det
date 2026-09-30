"""UDE-2: defects whose physics is deliberately ABSENT from the library.

The sharpest objection to this whole project is "you generated your own
defects, so of course you detect them". UDE-1 answers half of it by holding out
mechanisms. This answers the other half, and it is the harder one: every
mechanism in the library, held out or not, is still drawn from the same small
set of functional forms. A detector could be recognising *the form* rather than
the anomaly, and leave-one-mechanism-out would never reveal it.

So these four forms appear nowhere in mechanisms.yaml and are never used in
training:

  stretched_exponential  exp(-(t/tau)^beta) - dispersive kinetics, common in
                         disordered systems, no power law anywhere in it
  log_time               A * log(1 + t/tau) - saturating logarithmic creep
  sigmoid                latency then rapid transition then plateau
  telegraph              random two-state switching, no trend at all

If recall survives here, the system is detecting anomaly. If it collapses, it
was pattern-matching, and we will say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..ingest.schema import coerce
from ..knowledge.library import MechanismLibrary
from ..knowledge.schema import ParameterSpec
from .defect_injector import LATENT_CEILING, DefectInjector
from .lot_generator import LOT_LEVEL_SCALE, DEFAULT_HOURS

OUT_OF_FAMILY_FORMS = ("stretched_exponential", "log_time", "sigmoid", "telegraph")

# Parameter ranges for the novel forms. Amplitudes are kept in the same band as
# the library's defects so the comparison is about SHAPE, not size - an
# out-of-family defect that is simply larger would prove nothing.
FORM_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "stretched_exponential": {"A_frac": (0.20, 0.70), "tau_hour": (30.0, 120.0), "beta": (0.3, 0.7)},
    "log_time": {"A_frac": (0.20, 0.70), "tau_hour": (8.0, 40.0)},
    "sigmoid": {"A_frac": (0.25, 0.75), "t_mid_hour": (50.0, 130.0), "width_hour": (8.0, 30.0)},
    "telegraph": {"A_frac": (0.15, 0.50), "switch_prob": (0.35, 0.65)},
}


def _draw(rng: np.random.Generator, r: Tuple[float, float]) -> float:
    return float(rng.uniform(*r))


class OutOfFamilyInjector(DefectInjector):
    """Injector that produces shapes the library cannot express."""

    def _excess(self, form: str, params: Dict[str, float], hours: np.ndarray, v0: float):
        t = np.clip(np.asarray(hours, dtype=float), 0.0, None)

        if form == "stretched_exponential":
            tau, beta = params["tau_hour"], params["beta"]
            g = 1.0 - np.exp(-((t / tau) ** beta))
            g_ref = 1.0 - np.exp(-((self.t_ref / tau) ** beta))
            return params["A_frac"] * v0 * g / max(g_ref, 1e-9), 1.0

        if form == "log_time":
            tau = params["tau_hour"]
            g = np.log1p(t / tau)
            return params["A_frac"] * v0 * g / max(float(np.log1p(self.t_ref / tau)), 1e-9), 1.0

        if form == "sigmoid":
            mid, width = params["t_mid_hour"], params["width_hour"]
            g = 1.0 / (1.0 + np.exp(-(t - mid) / width))
            g0 = 1.0 / (1.0 + np.exp(mid / width))
            return params["A_frac"] * v0 * (g - g0), 1.0

        if form == "telegraph":
            # Two-state random switching: no trend, no smoothness, nothing the
            # power law or a monotone prior can represent.
            state = (self.rng.random(t.shape) < params["switch_prob"]).astype(float)
            state[0] = 0.0  # the part starts in the low state at incoming test
            return params["A_frac"] * v0 * state, 1.0

        return super()._excess(form, params, hours, v0)

    def make_out_of_family(
        self,
        hours: np.ndarray,
        spec: ParameterSpec,
        form: str,
        lot_median: Optional[float] = None,
        lot_n_shift: float = 0.0,
        subtlety: float = 1.0,
    ):
        hours = np.asarray(hours, dtype=float)
        v0 = self.sample_v0(spec, lot_median)

        nominal = self.library.nominal
        n_nom = max(0.01, _draw(self.rng, nominal.physics.params["n"]) + lot_n_shift)
        a_nom = _draw(self.rng, nominal.physics.params["A_frac"])
        base = v0 + self._nominal_drift(hours, v0, n_nom, a_nom)

        params = {k: _draw(self.rng, r) for k, r in FORM_PARAMS[form].items()}
        excess, noise_mult = self._excess(form, params, hours, v0)
        excess = excess * float(subtlety)

        # Same latency constraint as the library defects: stay under the
        # datasheet limit, so static screening still passes them.
        ceiling = spec.limit_hi * LATENT_CEILING
        peak = float(np.max(base + excess))
        if peak > ceiling:
            headroom = ceiling - float(np.max(base))
            span = float(np.max(excess))
            excess = excess * (max(0.0, headroom / span) if span > 0 else 0.0)

        values = base + excess
        values = values + self.rng.normal(0.0, spec.meas_noise_frac * v0 * noise_mult, size=values.shape)
        return np.clip(values, 1e-9, None), v0


def generate_out_of_family_lots(
    library: MechanismLibrary,
    n_lots: int = 10,
    parts_per_lot: int = 400,
    prevalence: float = 0.02,
    hours: Sequence[float] = DEFAULT_HOURS,
    subtlety: float = 1.0,
    seed: int = 7,
    lot_prefix: str = "OOF",
) -> pd.DataFrame:
    """Lots of healthy parts with out-of-family defects injected."""
    rng = np.random.default_rng(seed)
    inj = OutOfFamilyInjector(library, seed=seed + 1)
    hours = np.asarray(hours, dtype=float)
    params = library.parameters()
    rows: List[Dict] = []

    for k in range(n_lots):
        lot_id = f"{lot_prefix}{k:03d}"
        level_z = float(rng.normal(0.0, 1.0))
        n_shift = float(rng.normal(0.0, 0.025))
        lot_medians = {
            p.name: p.nominal_median * float(np.exp(level_z * LOT_LEVEL_SCALE * p.nominal_sigma_log))
            for p in params
        }
        n_def = int(round(parts_per_lot * prevalence))
        defect_idx = set(rng.choice(parts_per_lot, size=n_def, replace=False).tolist()) if n_def else set()

        for i in range(parts_per_lot):
            wafer = int(rng.integers(4))
            part_id = f"{lot_id}-W{wafer}-D{i:04d}"
            is_defect = i in defect_idx
            form = OUT_OF_FAMILY_FORMS[int(rng.integers(len(OUT_OF_FAMILY_FORMS)))] if is_defect else ""
            # Express on a random subset, mirroring how library mechanisms
            # affect only some parameters.
            affected = set(rng.choice([p.name for p in params], size=2, replace=False).tolist()) if is_defect else set()

            for p in params:
                if is_defect and p.name in affected:
                    values, _ = inj.make_out_of_family(
                        hours, p, form, lot_median=lot_medians[p.name],
                        lot_n_shift=n_shift, subtlety=subtlety,
                    )
                else:
                    traj = inj.make_trajectory(
                        hours, p, library.nominal, lot_median=lot_medians[p.name], lot_n_shift=n_shift
                    )
                    values = traj.values

                for h, v in zip(hours, values):
                    rows.append({
                        "part_id": part_id, "lot_id": lot_id, "wafer_id": f"{lot_id}-W{wafer}",
                        "x": int(rng.integers(16)), "y": int(rng.integers(16)),
                        "param_name": p.name, "unit": p.unit, "hours": float(h), "value": float(v),
                        "limit_lo": 0.0, "limit_hi": p.limit_hi, "temp_C": 125.0,
                        "is_defect": bool(is_defect),
                        "mechanism_id": f"OOF-{form}" if is_defect else library.nominal.id,
                        "severity": "critical" if is_defect else "none",
                    })

    return coerce(pd.DataFrame(rows))
