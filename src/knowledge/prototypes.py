"""Inverse consumer of the Defect Mechanism Library: name the mechanism.

Each `fingerprint` in the library becomes a prototype point in signature space,
with a per-dimension tolerance taken from the fingerprint's own declared range.
A part is attributed to its nearest prototype, measured in units of those
tolerances - so "distance 1.0" means "one fingerprint-width away", which is a
number a reliability engineer can argue with.

If nothing is within `tau`, the part is labelled UNKNOWN-MECHANISM and the
decision layer floors it into REVIEW. Abstention is the correct behaviour for a
degradation physics nobody has catalogued: "anomalous, mechanism unrecognised"
is honest and actionable, whereas a confident wrong label is neither.

A caveat stated in the architecture and repeated here because it matters: the
same library drives the injector, so on synthetic data attribution is
circular and its accuracy there is not evidence of anything. It is only
meaningful against held-out mechanisms (UDE-1) and real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .library import MechanismLibrary
from .schema import Mechanism

# Signature dimensions used for matching, and the default tolerance for each
# when a fingerprint does not pin it down.
DIMENSIONS = ("n_shift", "r2", "monotonic_violation", "jump_score", "rank_mobility", "noise_ratio")
DEFAULT_TOLERANCE = {
    "n_shift": 0.08,
    "r2": 0.10,
    "monotonic_violation": 0.05,
    "jump_score": 2.0,
    "rank_mobility": 0.25,
    "noise_ratio": 1.5,
}

# A part further than this many fingerprint-widths from every prototype is
# declared unknown. Deliberately generous: wrongly claiming a mechanism is
# worse than admitting ignorance.
DEFAULT_TAU = 3.0

UNKNOWN = "UNKNOWN-MECHANISM"


@dataclass
class Attribution:
    mechanism_id: str
    mechanism_name: str
    distance: float
    is_unknown: bool
    severity: str
    reference: str
    parameter: str
    runner_up: Optional[Tuple[str, float]] = None

    @property
    def confident(self) -> bool:
        """No close second. A tie between two mechanisms is not an attribution."""
        if self.is_unknown or self.runner_up is None:
            return False
        return self.runner_up[1] - self.distance > 0.5


def _prototype(mech: Mechanism) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a fingerprint into a centre vector and a tolerance vector."""
    fp = mech.fingerprint
    lo, hi = fp.n_shift_vs_lot
    centre = {
        "n_shift": 0.5 * (lo + hi),
        "r2": 0.5 * (fp.r2_min + fp.r2_max),
        # A mechanism declared monotonic should show no backward movement.
        "monotonic_violation": 0.0 if fp.monotonic else 0.15,
        # jump_score is a ratio against the typical step, so a declared jump
        # shows up as a large value and no jump as a small one.
        "jump_score": 4.0 if fp.jump_count >= 1 else 1.2,
        "rank_mobility": fp.rank_mobility_value,
        "noise_ratio": 0.5 * (fp.noise_ratio[0] + fp.noise_ratio[1]),
    }
    tol = {
        "n_shift": max(0.5 * (hi - lo), DEFAULT_TOLERANCE["n_shift"]),
        "r2": max(0.5 * (fp.r2_max - fp.r2_min), DEFAULT_TOLERANCE["r2"]),
        "monotonic_violation": DEFAULT_TOLERANCE["monotonic_violation"],
        "jump_score": DEFAULT_TOLERANCE["jump_score"],
        "rank_mobility": DEFAULT_TOLERANCE["rank_mobility"],
        "noise_ratio": max(0.5 * (fp.noise_ratio[1] - fp.noise_ratio[0]), DEFAULT_TOLERANCE["noise_ratio"]),
    }
    return (
        np.array([centre[d] for d in DIMENSIONS], dtype=float),
        np.array([tol[d] for d in DIMENSIONS], dtype=float),
    )


def observed_signature(features: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Extract the matching dimensions for one parameter, for every part."""
    p = parameter
    out = pd.DataFrame(index=features.index)
    out["n_shift"] = features[f"{p}__n_shift_vs_lot"]
    out["r2"] = features[f"{p}__r2"]
    out["monotonic_violation"] = features[f"{p}__monotonic_violation"]
    out["jump_score"] = features[f"{p}__jump_score"]
    out["rank_mobility"] = features[f"{p}__rank_mobility"]
    out["noise_ratio"] = features[f"{p}__noise_ratio"]
    return out[list(DIMENSIONS)].astype(float).fillna(0.0)


class MechanismAttributor:
    def __init__(self, library: MechanismLibrary, tau: float = DEFAULT_TAU) -> None:
        self.library = library
        self.tau = tau
        self.prototypes_: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            m.id: _prototype(m) for m in library.all()
        }
        self.nominal_id = library.nominal.id

    def _distances(self, X: np.ndarray, parameter: str) -> pd.DataFrame:
        """Distance from every part to every candidate prototype for this parameter.

        The nominal baseline is a candidate too, and it has to be. Without it
        every healthy part is forced onto its nearest *defect* prototype, which
        makes the attribution column meaningless and turns peer-clustering into
        noise - 5 healthy lot-mates "sharing a signature" is not a process
        excursion, it is an artefact of having nowhere else to put them.
        """
        cols = {}
        candidates = list(self.library.affecting(parameter)) + [self.library.nominal]
        for mech in candidates:
            centre, tol = self.prototypes_[mech.id]
            d = (X - centre) / tol
            cols[mech.id] = np.sqrt(np.mean(d**2, axis=1))
        return pd.DataFrame(cols)

    def attribute_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Attribute every part, choosing the parameter with the best match.

        A mechanism expresses on specific parameters, so the attribution is run
        per parameter and the closest overall match wins - which also tells the
        inspector *where* to look.
        """
        best = pd.DataFrame(index=features.index)
        best["distance"] = np.inf
        best["mechanism_id"] = UNKNOWN
        best["parameter"] = ""
        best["runner_up_id"] = ""
        best["runner_up_distance"] = np.inf

        for parameter in self.library.parameter_names():
            if f"{parameter}__n_shift_vs_lot" not in features.columns:
                continue
            X = observed_signature(features, parameter).to_numpy(dtype=float)
            dist = self._distances(X, parameter)
            if dist.empty:
                continue
            order = np.argsort(dist.to_numpy(), axis=1)
            names = np.array(dist.columns)
            top = dist.to_numpy()[np.arange(len(dist)), order[:, 0]]
            improved = top < best["distance"].to_numpy()
            best.loc[improved, "distance"] = top[improved]
            best.loc[improved, "mechanism_id"] = names[order[improved, 0]]
            best.loc[improved, "parameter"] = parameter
            if dist.shape[1] > 1:
                second = dist.to_numpy()[np.arange(len(dist)), order[:, 1]]
                best.loc[improved, "runner_up_id"] = names[order[improved, 1]]
                best.loc[improved, "runner_up_distance"] = second[improved]

        # Unknown means "far from EVERY prototype, nominal included". A part
        # close to nominal is not unknown, it is ordinary.
        best["is_unknown"] = best["distance"] > self.tau
        best.loc[best["is_unknown"], "mechanism_id"] = UNKNOWN
        best["matched_nominal"] = best["mechanism_id"] == self.nominal_id
        best["mechanism_name"] = best["mechanism_id"].map(
            lambda m: UNKNOWN if m == UNKNOWN else self.library.get(m).name
        )
        best["severity"] = best["mechanism_id"].map(
            lambda m: "unknown" if m == UNKNOWN else self.library.get(m).severity
        )
        return best

    def attribute(self, features: pd.DataFrame, part_id) -> Attribution:
        row = self.attribute_frame(features.loc[[part_id]]).iloc[0]
        mech_id = str(row["mechanism_id"])
        is_unknown = bool(row["is_unknown"])
        return Attribution(
            mechanism_id=mech_id,
            mechanism_name=str(row["mechanism_name"]),
            distance=float(row["distance"]),
            is_unknown=is_unknown,
            severity=str(row["severity"]),
            reference="" if is_unknown else self.library.get(mech_id).reference,
            parameter=str(row["parameter"]),
            runner_up=(
                (str(row["runner_up_id"]), float(row["runner_up_distance"]))
                if row["runner_up_id"] else None
            ),
        )
