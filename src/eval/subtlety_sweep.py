"""UDE-4: Minimum Detectable Drift.

Recall at full defect amplitude answers an easy question. The one a reliability
engineer actually needs answered is: *how small a defect can you still see?*

The sweep scales the defect excess down by a factor and re-measures recall at a
fixed false-alarm budget. The output is a single spec number - the MDD, the
smallest subtlety at which recall still clears the target - expressed in units
of the lot's own spread so it transfers across parameters and lots.

This is also where track A5 is expected to earn its keep. At full amplitude a
defect is a level outlier and A1 catches it; as amplitude falls the level signal
disappears before the kinetic one does, because the exponent is a property of
the curve's shape rather than its size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .harness import recall_at_fpr


@dataclass
class SweepPoint:
    subtlety: float
    n_defects: int
    recall_at_fpr: float
    median_defect_z: float      # how far the median defect sits above its lot
    per_track_recall: Dict[str, float]


def median_defect_separation(features: pd.DataFrame) -> float:
    """Median lot-relative departure of the defective parts, in MAD units.

    Converts an abstract subtlety multiplier into something physical: "we
    detect defects down to 1.4 MAD above the lot median".
    """
    z_cols = [c for c in features.columns if "__v_" in c and c.endswith("h_z")]
    if not z_cols:
        return float("nan")
    worst = features[z_cols].max(axis=1)
    defects = worst[features["is_defect"].astype(bool)]
    return float(defects.median()) if len(defects) else float("nan")


def sweep(
    generate: Callable[[float], pd.DataFrame],
    score: Callable[[pd.DataFrame], Dict[str, pd.Series]],
    subtleties: Sequence[float] = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.10),
    fpr_budget: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the sweep.

    `generate(subtlety) -> features` builds a labelled feature frame at that
    defect amplitude. `score(features) -> {name: risk}` returns the fused risk
    plus any per-track scores worth tracking separately.
    """
    rows: List[SweepPoint] = []
    for s in subtleties:
        feats = generate(s)
        y = feats["is_defect"].astype(int).to_numpy()
        scores = score(feats)
        per_track = {name: recall_at_fpr(y, series, fpr_budget) for name, series in scores.items()}
        fused = per_track.get("fusion", float("nan"))
        rows.append(
            SweepPoint(
                subtlety=float(s),
                n_defects=int(y.sum()),
                recall_at_fpr=float(fused),
                median_defect_z=median_defect_separation(feats),
                per_track_recall=per_track,
            )
        )
        if verbose:
            detail = "  ".join(f"{k}={v:.0%}" for k, v in per_track.items())
            print(f"  subtlety {s:<5.2f}  sep {rows[-1].median_defect_z:5.2f} MAD   {detail}")

    flat = []
    for r in rows:
        row = {"subtlety": r.subtlety, "n_defects": r.n_defects,
               "median_defect_z": r.median_defect_z, "recall": r.recall_at_fpr}
        row.update({f"recall_{k}": v for k, v in r.per_track_recall.items()})
        flat.append(row)
    return pd.DataFrame(flat)


def minimum_detectable_drift(curve: pd.DataFrame, target_recall: float = 0.90) -> Dict:
    """The smallest subtlety still clearing the recall target, interpolated.

    Returns NaN when even the largest defects miss the target - which is a
    result, not an error, and must be reported as one.
    """
    ok = curve[curve["recall"] >= target_recall]
    if ok.empty:
        return {"target_recall": target_recall, "mdd_subtlety": float("nan"),
                "mdd_separation_mad": float("nan"), "achieved": False}

    row = ok.loc[ok["subtlety"].idxmin()]
    below = curve[curve["subtlety"] < row["subtlety"]]
    mdd_sub, mdd_sep = float(row["subtlety"]), float(row["median_defect_z"])

    if not below.empty:
        # Linear interpolation between the last passing and first failing point.
        lo = below.loc[below["subtlety"].idxmax()]
        span = row["recall"] - lo["recall"]
        if span > 1e-9:
            frac = (target_recall - lo["recall"]) / span
            mdd_sub = float(lo["subtlety"] + frac * (row["subtlety"] - lo["subtlety"]))
            mdd_sep = float(lo["median_defect_z"] + frac * (row["median_defect_z"] - lo["median_defect_z"]))

    return {
        "target_recall": target_recall,
        "mdd_subtlety": mdd_sub,
        "mdd_separation_mad": mdd_sep,
        "achieved": True,
    }
