"""Loader for the Defect Mechanism Library.

    from src.knowledge.library import MechanismLibrary
    lib = MechanismLibrary.load()
    lib.defects()                      # every mechanism with defect: true
    lib.without("MECH-EM-01")          # the LOMO fold used by UDE-1
    lib.parameter("Iddq").derated_limit
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from .schema import Mechanism, ParameterSpec

DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "configs" / "mechanisms" / "mechanisms.v1.yaml"

NOMINAL_ID = "MECH-NOM-00"


class MechanismLibrary:
    def __init__(
        self,
        version: int,
        time_ref_hours: float,
        parameters: List[ParameterSpec],
        mechanisms: List[Mechanism],
        severity_cost_multiplier: Dict[str, float],
        source_hash: str = "",
        source_path: Optional[Path] = None,
    ) -> None:
        self.version = version
        self.time_ref_hours = time_ref_hours
        self._parameters = {p.name: p for p in parameters}
        self._mechanisms = {m.id: m for m in mechanisms}
        self.severity_cost_multiplier = severity_cost_multiplier
        self.source_hash = source_hash
        self.source_path = source_path
        self._validate()

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "MechanismLibrary":
        path = Path(path) if path is not None else DEFAULT_LIBRARY
        raw_bytes = path.read_bytes()
        doc = yaml.safe_load(raw_bytes.decode("utf-8"))
        return cls(
            version=int(doc.get("version", 1)),
            time_ref_hours=float(doc.get("time_ref_hours", 168.0)),
            parameters=[ParameterSpec.from_dict(p) for p in doc["parameters"]],
            mechanisms=[Mechanism.from_dict(m) for m in doc["mechanisms"]],
            severity_cost_multiplier={k: float(v) for k, v in (doc.get("severity_cost_multiplier") or {}).items()},
            source_hash=hashlib.sha256(raw_bytes).hexdigest(),
            source_path=path,
        )

    def _validate(self) -> None:
        if NOMINAL_ID not in self._mechanisms:
            raise ValueError(f"library must define the nominal baseline entry {NOMINAL_ID}")
        if self._mechanisms[NOMINAL_ID].defect:
            raise ValueError(f"{NOMINAL_ID} must be marked defect: false")
        for mech in self._mechanisms.values():
            unknown = set(mech.affects) - set(self._parameters)
            if unknown:
                raise ValueError(f"{mech.id} affects undeclared parameter(s): {sorted(unknown)}")
        for mech in self.defects():
            if mech.severity not in self.severity_cost_multiplier:
                raise ValueError(f"{mech.id}: severity '{mech.severity}' has no cost multiplier")

    # ------------------------------------------------------------ accessors

    @property
    def nominal(self) -> Mechanism:
        return self._mechanisms[NOMINAL_ID]

    def all(self) -> List[Mechanism]:
        return list(self._mechanisms.values())

    def defects(self) -> List[Mechanism]:
        return [m for m in self._mechanisms.values() if m.defect]

    def defect_ids(self) -> List[str]:
        return [m.id for m in self.defects()]

    def get(self, mech_id: str) -> Mechanism:
        if mech_id not in self._mechanisms:
            raise KeyError(f"unknown mechanism id: {mech_id}")
        return self._mechanisms[mech_id]

    def parameters(self) -> List[ParameterSpec]:
        return list(self._parameters.values())

    def parameter_names(self) -> List[str]:
        return list(self._parameters.keys())

    def parameter(self, name: str) -> ParameterSpec:
        if name not in self._parameters:
            raise KeyError(f"unknown parameter: {name}")
        return self._parameters[name]

    def affecting(self, parameter: str) -> List[Mechanism]:
        return [m for m in self.defects() if parameter in m.affects]

    def cost_multiplier(self, mech_id: Optional[str]) -> float:
        if not mech_id or mech_id == NOMINAL_ID:
            return 0.0
        return self.severity_cost_multiplier.get(self.get(mech_id).severity, 1.0)

    # ------------------------------------------------------------ LOMO folds

    def without(self, *mech_ids: str) -> "MechanismLibrary":
        """Return a copy with the named mechanisms removed.

        This is the leave-one-mechanism-out fold for UDE-1: train and calibrate
        on `lib.without(m)`, then evaluate on defects generated from `m` alone.
        The nominal baseline is never removable.
        """
        drop = set(mech_ids)
        if NOMINAL_ID in drop:
            raise ValueError("the nominal baseline cannot be held out")
        unknown = drop - set(self._mechanisms)
        if unknown:
            raise KeyError(f"cannot hold out unknown mechanism(s): {sorted(unknown)}")
        return MechanismLibrary(
            version=self.version,
            time_ref_hours=self.time_ref_hours,
            parameters=self.parameters(),
            mechanisms=[m for m in self._mechanisms.values() if m.id not in drop],
            severity_cost_multiplier=dict(self.severity_cost_multiplier),
            source_hash=self.source_hash,
            source_path=self.source_path,
        )

    def lomo_folds(self) -> Iterable[tuple[str, "MechanismLibrary"]]:
        """Yield (held_out_id, library_without_it) for every defect mechanism."""
        for mech_id in self.defect_ids():
            yield mech_id, self.without(mech_id)

    def __repr__(self) -> str:
        return (
            f"MechanismLibrary(v{self.version}, "
            f"{len(self._mechanisms)} mechanisms, "
            f"{len(self._parameters)} parameters, "
            f"hash={self.source_hash[:12]})"
        )
