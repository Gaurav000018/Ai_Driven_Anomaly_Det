"""Assemble the Module B regression problem.

The brief asks for f(Value_0h, Value_24h) -> Value_168h. We give the model those
two values *plus lot context*, because the same pair means different things in
different populations: 12 uA following 10 uA is unremarkable in a scattered lot
and alarming in a tight one. Contextual regression, not bare extrapolation.

Only early timepoints are allowed as inputs. Anything at or beyond the horizon
would leak the answer, so `make_dataset` enforces that rather than trusting the
caller to remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..features.robust_stats import robust_z
from ..knowledge.library import MechanismLibrary
from ..features.trajectory import Panel, build_panel


@dataclass
class DriftDataset:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series            # lot_id, for GroupKFold
    parameter: str
    horizon: float
    input_hours: List[float]
    limit_hi: float
    derated_limit: float
    meta: pd.DataFrame           # v0, labels, lot_id - never fed to the model


def make_dataset(
    df: pd.DataFrame,
    library: MechanismLibrary,
    parameter: str,
    input_hours: Tuple[float, ...] = (0.0, 24.0),
    horizon: float = 168.0,
    panel: Optional[Panel] = None,
    require_target: bool = True,
) -> DriftDataset:
    bad = [h for h in input_hours if h >= horizon]
    if bad:
        raise ValueError(f"input hours {bad} are at or past the {horizon}h horizon - that leaks the target")

    panel = panel if panel is not None else build_panel(df)
    spec = library.parameter(parameter)
    wide = panel.param(parameter)

    available = set(np.round(panel.hours, 6))
    missing_inputs = [h for h in input_hours if round(h, 6) not in available]
    if missing_inputs:
        raise ValueError(f"input hours {missing_inputs} absent from the data; present: {sorted(available)}")

    # The horizon is required for TRAINING and absent by definition when
    # forecasting a part still in the oven - which is the whole point of the
    # module. Requiring it unconditionally made the predictor unusable for
    # exactly the parts it exists to screen.
    has_target = round(horizon, 6) in available
    if has_target is False and require_target:
        raise ValueError(
            f"horizon {horizon}h absent from the data; present: {sorted(available)}. "
            "Pass require_target=False to build a prediction-only dataset."
        )

    lot = wide.index.get_level_values("lot_id").astype(str)
    lot = pd.Series(lot, index=wide.index.get_level_values("part_id"), name="lot_id")
    idx = lot.index

    feats = pd.DataFrame(index=idx)
    values = {h: pd.Series(wide[h].to_numpy(dtype=float), index=idx) for h in input_hours}

    for h in input_hours:
        feats[f"v_{int(h)}h"] = values[h]
        feats[f"v_{int(h)}h_z"] = robust_z(values[h], lot).to_numpy()

    v_first = values[input_hours[0]]
    for h in input_hours[1:]:
        # Early drift, absolute and relative. The relative form is what makes a
        # model trained on one lot's level transfer to another's.
        feats[f"drift_{int(input_hours[0])}_{int(h)}"] = values[h] - v_first
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (values[h] - v_first) / v_first.replace(0.0, np.nan)
        feats[f"drift_rel_{int(input_hours[0])}_{int(h)}"] = rel.fillna(0.0)
        feats[f"slope_{int(input_hours[0])}_{int(h)}"] = (values[h] - v_first) / (h - input_hours[0])

    # Lot context. These are what turn extrapolation into contextual regression.
    grouped = v_first.groupby(lot.to_numpy(), observed=True)
    feats["lot_median_v0"] = grouped.transform("median")
    feats["lot_iqr_v0"] = grouped.transform(lambda s: float(np.subtract(*np.percentile(s, [75, 25]))))
    last_in = values[input_hours[-1]]
    feats["lot_median_last_in"] = last_in.groupby(lot.to_numpy(), observed=True).transform("median")
    feats["headroom_frac"] = (spec.derated_limit - last_in) / spec.derated_limit

    if has_target:
        y = pd.Series(wide[horizon].to_numpy(dtype=float), index=idx, name=f"v_{int(horizon)}h")
    else:
        y = pd.Series(np.nan, index=idx, name=f"v_{int(horizon)}h", dtype=float)

    meta_cols = [c for c in ("lot_id", "is_defect", "mechanism_id", "severity") if c in panel.meta.columns]
    meta = panel.meta.loc[idx, meta_cols].copy()
    meta["v0"] = v_first

    ok = feats.notna().all(axis=1)
    if has_target:
        ok &= y.notna()
    return DriftDataset(
        X=feats[ok],
        y=y[ok],
        groups=lot[ok],
        parameter=parameter,
        horizon=horizon,
        input_hours=list(input_hours),
        limit_hi=spec.limit_hi,
        derated_limit=spec.derated_limit,
        meta=meta[ok],
    )
