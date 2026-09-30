"""Build whole synthetic lots in the canonical long format.

Lot-level effects are the point. Every lot gets its own median level and its own
kinetic exponent shift, because that is exactly what defeats a global static
limit and forces the detector to reason *relative to the lot*. A part at 18 uA
is unremarkable in one lot and a screaming outlier in another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..ingest.schema import coerce
from ..knowledge.library import MechanismLibrary
from .defect_injector import DefectInjector

DEFAULT_HOURS = (0.0, 24.0, 96.0, 168.0)


# How large a lot-to-lot level shift is, as a fraction of the parameter's own
# part-to-part spread. Lots drift together across parameters (same process) but
# a tight parameter like t_pd shifts far less in absolute terms than Iddq does.
LOT_LEVEL_SCALE = 0.8


@dataclass
class LotSpec:
    lot_id: str
    n_parts: int = 500
    level_z: float = 0.0             # lot-to-lot level shift, in sigma units
    n_shift: float = 0.0             # lot-to-lot kinetic shift
    wafers: int = 4
    die_grid: int = 16               # wafer is die_grid x die_grid


class LotGenerator:
    def __init__(
        self,
        library: MechanismLibrary,
        hours: Sequence[float] = DEFAULT_HOURS,
        seed: Optional[int] = None,
    ) -> None:
        self.library = library
        self.hours = np.asarray(hours, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.injector = DefectInjector(library, seed=None if seed is None else seed + 1)

    # --------------------------------------------------------------- one lot

    def generate_lot(
        self,
        spec: LotSpec,
        prevalence: float = 0.01,
        allowed_mechanisms: Optional[List[str]] = None,
        subtlety: float = 1.0,
        temp_c: float = 125.0,
    ) -> pd.DataFrame:
        n_defective = int(round(spec.n_parts * prevalence))
        defect_idx = set(self.rng.choice(spec.n_parts, size=n_defective, replace=False).tolist()) if n_defective else set()

        params = self.library.parameters()
        lot_medians = {
            p.name: p.nominal_median * float(np.exp(spec.level_z * LOT_LEVEL_SCALE * p.nominal_sigma_log))
            for p in params
        }

        rows: List[Dict] = []
        for i in range(spec.n_parts):
            wafer = int(self.rng.integers(spec.wafers))
            x = int(self.rng.integers(spec.die_grid))
            y = int(self.rng.integers(spec.die_grid))
            part_id = f"{spec.lot_id}-W{wafer}-D{i:04d}"

            if i in defect_idx:
                # Pick one mechanism for the part; it expresses on whichever
                # parameters that mechanism actually affects.
                pool = [m for m in self.library.defects() if allowed_mechanisms is None or m.id in allowed_mechanisms]
                if not pool:
                    raise ValueError("allowed_mechanisms excluded every defect mechanism")
                mech = pool[int(self.rng.integers(len(pool)))]
            else:
                mech = self.library.nominal

            for p in params:
                expressed = mech if (mech.defect and p.name in mech.affects) else self.library.nominal
                traj = self.injector.make_trajectory(
                    hours=self.hours,
                    spec=p,
                    mechanism=expressed,
                    lot_median=lot_medians[p.name],
                    lot_n_shift=spec.n_shift,
                    subtlety=subtlety,
                )
                for h, v in zip(traj.hours, traj.values):
                    rows.append(
                        {
                            "part_id": part_id,
                            "lot_id": spec.lot_id,
                            "wafer_id": f"{spec.lot_id}-W{wafer}",
                            "x": x,
                            "y": y,
                            "param_name": p.name,
                            "unit": p.unit,
                            "hours": float(h),
                            "value": float(v),
                            "limit_lo": 0.0,
                            "limit_hi": p.limit_hi,
                            "temp_C": temp_c,
                            # Ground truth. Labels describe the PART, so a part is
                            # defective even on parameters the mechanism leaves alone.
                            "is_defect": bool(mech.defect),
                            "mechanism_id": mech.id,
                            "severity": mech.severity,
                        }
                    )

        return coerce(pd.DataFrame(rows))

    # -------------------------------------------------------------- many lots

    def generate(
        self,
        n_lots: int = 20,
        parts_per_lot: int = 500,
        prevalence: float = 0.01,
        allowed_mechanisms: Optional[List[str]] = None,
        subtlety: float = 1.0,
        lot_prefix: str = "LOT",
    ) -> pd.DataFrame:
        frames = []
        for k in range(n_lots):
            spec = LotSpec(
                lot_id=f"{lot_prefix}{k:03d}",
                n_parts=parts_per_lot,
                level_z=float(self.rng.normal(0.0, 1.0)),
                n_shift=float(self.rng.normal(0.0, 0.025)),
            )
            frames.append(
                self.generate_lot(
                    spec,
                    prevalence=prevalence,
                    allowed_mechanisms=allowed_mechanisms,
                    subtlety=subtlety,
                )
            )
        return coerce(pd.concat(frames, ignore_index=True))


def part_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long frame to one ground-truth row per part."""
    cols = ["part_id", "lot_id", "is_defect", "mechanism_id", "severity"]
    have = [c for c in cols if c in df.columns]
    return df[have].drop_duplicates(subset=["part_id"]).reset_index(drop=True)


def static_screen(df: pd.DataFrame) -> pd.DataFrame:
    """Classical pass/fail screening - the baseline our system has to beat.

    A part fails if any measurement of any parameter breaches its datasheet
    limit at any timepoint.
    """
    if "limit_hi" not in df.columns:
        raise ValueError("static screening needs a limit_hi column")
    over = df["value"] > df["limit_hi"]
    failed = set(df.loc[over, "part_id"].unique())
    parts = part_labels(df)
    parts["static_fail"] = parts["part_id"].isin(failed)
    return parts
