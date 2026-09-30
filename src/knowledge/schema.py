"""Typed view over the Defect Mechanism Library YAML.

The library is read in two directions (see configs/mechanisms/mechanisms.v1.yaml):
`physics` drives generation, `fingerprint` drives diagnosis. These dataclasses
are the single parsing point for both, so the two consumers can never disagree
about what a mechanism means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Range = Tuple[float, float]

IMPLEMENTED_FORMS = {
    "power_law",
    "step",
    "hump",
    "variance_inflation",
    "early_offset",
    "exponential",
}

DETECTABILITY_LEVELS = {"none": 0, "weak": 1, "partial": 2, "strong": 3}


def _as_range(value: Any, name: str) -> Range:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element [lo, hi] range, got {value!r}")
    lo, hi = float(value[0]), float(value[1])
    if hi < lo:
        raise ValueError(f"{name} range is inverted: [{lo}, {hi}]")
    return (lo, hi)


@dataclass(frozen=True)
class ParameterSpec:
    """A measured electrical parameter and its healthy-population statistics."""

    name: str
    unit: str
    limit_hi: float
    derate: float
    nominal_median: float
    nominal_sigma_log: float
    meas_noise_frac: float
    description: str = ""

    @property
    def derated_limit(self) -> float:
        """The limit we actually screen against. Space parts are derated."""
        return self.limit_hi * self.derate

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParameterSpec":
        return cls(
            name=d["name"],
            unit=d["unit"],
            limit_hi=float(d["limit_hi"]),
            derate=float(d.get("derate", 1.0)),
            nominal_median=float(d["nominal_median"]),
            nominal_sigma_log=float(d["nominal_sigma_log"]),
            meas_noise_frac=float(d["meas_noise_frac"]),
            description=d.get("description", ""),
        )


@dataclass(frozen=True)
class Physics:
    """The forward (generative) half of a mechanism entry."""

    form: str
    params: Dict[str, Range]

    @classmethod
    def from_dict(cls, d: Dict[str, Any], mech_id: str) -> "Physics":
        form = d["form"]
        if form not in IMPLEMENTED_FORMS:
            raise ValueError(
                f"{mech_id}: physics.form '{form}' is not implemented. "
                f"Known forms: {sorted(IMPLEMENTED_FORMS)}"
            )
        params = {k: _as_range(v, f"{mech_id}.physics.params.{k}") for k, v in (d.get("params") or {}).items()}
        return cls(form=form, params=params)


@dataclass(frozen=True)
class Fingerprint:
    """The inverse (diagnostic) half of a mechanism entry.

    Used by src/knowledge/prototypes.py to build a prototype vector in signature
    space, and by coverage_audit.py for reporting.
    """

    n_shift_vs_lot: Range
    r2_min: float
    r2_max: float
    monotonic: bool
    jump_count: int
    rank_mobility: str
    noise_ratio: Range

    RANK_MOBILITY_VALUE = {"low": 0.15, "medium": 0.5, "high": 0.85}

    @property
    def rank_mobility_value(self) -> float:
        return self.RANK_MOBILITY_VALUE.get(self.rank_mobility, 0.5)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], mech_id: str) -> "Fingerprint":
        return cls(
            n_shift_vs_lot=_as_range(d["n_shift_vs_lot"], f"{mech_id}.fingerprint.n_shift_vs_lot"),
            r2_min=float(d.get("r2_min", 0.0)),
            r2_max=float(d.get("r2_max", 1.0)),
            monotonic=bool(d.get("monotonic", True)),
            jump_count=int(d.get("jump_count", 0)),
            rank_mobility=str(d.get("rank_mobility", "medium")),
            noise_ratio=_as_range(d.get("noise_ratio", [0.5, 1.5]), f"{mech_id}.fingerprint.noise_ratio"),
        )


@dataclass(frozen=True)
class Mechanism:
    id: str
    name: str
    family: str
    defect: bool
    physics: Physics
    fingerprint: Fingerprint
    affects: List[str]
    severity: str
    reference: str
    detectability: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Mechanism":
        mech_id = d["id"]
        detect = {k: str(v) for k, v in (d.get("detectability") or {}).items()}
        for track, level in detect.items():
            if level not in DETECTABILITY_LEVELS:
                raise ValueError(
                    f"{mech_id}: detectability.{track} = '{level}' is not one of "
                    f"{sorted(DETECTABILITY_LEVELS)}"
                )
        return cls(
            id=mech_id,
            name=d["name"],
            family=d["family"],
            defect=bool(d.get("defect", True)),
            physics=Physics.from_dict(d["physics"], mech_id),
            fingerprint=Fingerprint.from_dict(d["fingerprint"], mech_id),
            affects=list(d["affects"]),
            severity=str(d.get("severity", "major")),
            reference=str(d.get("reference", "")),
            detectability=detect,
        )

    def detectability_score(self, track: str) -> int:
        return DETECTABILITY_LEVELS.get(self.detectability.get(track, "none"), 0)
