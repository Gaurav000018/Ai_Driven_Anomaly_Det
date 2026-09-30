"""Turn the canonical long frame into per-part, per-parameter trajectories.

The panel is a wide table indexed by (part_id, lot_id, param_name) with one
column per burn-in hour. Everything downstream - physics fits, shape features,
lot statistics - operates on this panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Panel:
    """Wide trajectories plus the hour grid they sit on."""

    values: pd.DataFrame          # index (part_id, lot_id, param_name), columns = hours
    hours: np.ndarray
    meta: pd.DataFrame            # one row per part: lot_id, wafer_id, x, y, labels

    @property
    def n_parts(self) -> int:
        return self.values.index.get_level_values("part_id").nunique()

    def param(self, name: str) -> pd.DataFrame:
        """Trajectories for one parameter, indexed by (part_id, lot_id)."""
        sub = self.values.xs(name, level="param_name")
        return sub

    def param_names(self) -> List[str]:
        return sorted(self.values.index.get_level_values("param_name").unique())


def build_panel(df: pd.DataFrame, min_points: int = 3) -> Panel:
    """Pivot long -> wide, dropping trajectories too short to fit.

    Missing timepoints are interpolated along the hour axis when they sit
    between measured points, and left as NaN at the edges - a trailing NaN is
    an honest "not measured yet", not something to invent.
    """
    wide = df.pivot_table(
        index=["part_id", "lot_id", "param_name"],
        columns="hours",
        values="value",
        aggfunc="mean",
    ).sort_index(axis=1)

    keep = wide.notna().sum(axis=1) >= min_points
    wide = wide[keep]

    hours = wide.columns.to_numpy(dtype=float)
    wide = wide.interpolate(axis=1, method="index", limit_area="inside")

    meta_cols = [c for c in ["lot_id", "wafer_id", "x", "y", "is_defect", "mechanism_id", "severity"] if c in df.columns]
    meta = df[["part_id"] + meta_cols].drop_duplicates(subset=["part_id"]).set_index("part_id")

    return Panel(values=wide, hours=hours, meta=meta)


def deltas(values: np.ndarray) -> np.ndarray:
    """Drift relative to the first measurement: v(t) - v(0)."""
    return values - values[..., [0]]


def normalised(values: np.ndarray) -> np.ndarray:
    """Drift as a fraction of the part's own starting value."""
    v0 = values[..., [0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (values - v0) / np.where(np.abs(v0) < 1e-12, np.nan, v0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
