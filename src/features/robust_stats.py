"""Lot-relative statistics, computed robustly.

The single most important design choice in Module A: never mean and standard
deviation. A lot with one part at 45 uA against a median of 10 uA has its sigma
inflated by that very part, so the outlier hides inside its own contribution to
the spread. Median and MAD are immune - the 45 uA part cannot move a median.

Small lots get shrunk toward the population estimate, because a MAD from 12
parts is mostly noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# 1 / Phi^-1(0.75). Scales MAD so that for Gaussian data it equals sigma.
MAD_TO_SIGMA = 1.4826

# Lots smaller than this are blended toward the population statistics.
SHRINK_K = 30.0


def mad(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Median absolute deviation, scaled to be comparable with a sigma."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    out = MAD_TO_SIGMA * np.nanmedian(np.abs(x - med), axis=axis)
    return out


def robust_z(
    x: pd.Series,
    groups: pd.Series,
    shrink_k: float = SHRINK_K,
    floor_frac: float = 1e-3,
) -> pd.Series:
    """Robust z-score of `x` within each group, with small-group shrinkage.

    Returns (x - centre) / spread where centre and spread blend the group's own
    median/MAD with the population median/MAD, weighted by group size:

        w = n_group / (n_group + shrink_k)

    A 500-part lot is trusted almost entirely (w = 0.94); a 10-part lot leans on
    the population (w = 0.25). This is what keeps a tiny lot from declaring
    half its parts anomalous.
    """
    x = pd.to_numeric(x, errors="coerce")
    frame = pd.DataFrame({"x": x.to_numpy(), "g": groups.to_numpy()}, index=x.index)

    pop_centre = float(np.nanmedian(frame["x"]))
    pop_spread = float(mad(frame["x"].to_numpy()))

    stats = frame.groupby("g", observed=True)["x"].agg(
        n="count",
        centre="median",
        spread=lambda s: float(mad(s.to_numpy())),
    )
    w = stats["n"] / (stats["n"] + shrink_k)
    stats["centre_eff"] = w * stats["centre"] + (1.0 - w) * pop_centre
    stats["spread_eff"] = w * stats["spread"] + (1.0 - w) * pop_spread

    # A spread of exactly zero happens with quantised or saturated readings.
    floor = max(floor_frac * abs(pop_centre), 1e-12)
    stats["spread_eff"] = stats["spread_eff"].clip(lower=floor)

    centre = frame["g"].map(stats["centre_eff"])
    spread = frame["g"].map(stats["spread_eff"])
    return ((frame["x"] - centre) / spread).astype(float)


def lot_percentile_rank(x: pd.Series, groups: pd.Series) -> pd.Series:
    """Where each part sits within its lot, as a 0-1 percentile."""
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce").to_numpy(), "g": groups.to_numpy()}, index=x.index)
    return frame.groupby("g", observed=True)["x"].rank(pct=True).astype(float)


@dataclass
class LotReference:
    """The reference population a part is judged against.

    Persisted with the model so a rejection can be replayed months later and
    produce byte-identical numbers - which is what the audit trail promises.
    """

    centre: pd.Series
    spread: pd.Series
    n: pd.Series
    pop_centre: float
    pop_spread: float

    def z(self, x: pd.Series, groups: pd.Series) -> pd.Series:
        centre = groups.map(self.centre).fillna(self.pop_centre)
        spread = groups.map(self.spread).fillna(self.pop_spread)
        spread = spread.clip(lower=max(1e-3 * abs(self.pop_centre), 1e-12))
        return ((pd.to_numeric(x, errors="coerce") - centre) / spread).astype(float)

    @classmethod
    def fit(cls, x: pd.Series, groups: pd.Series, shrink_k: float = SHRINK_K) -> "LotReference":
        frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce").to_numpy(), "g": groups.to_numpy()})
        pop_centre = float(np.nanmedian(frame["x"]))
        pop_spread = float(mad(frame["x"].to_numpy()))
        stats = frame.groupby("g", observed=True)["x"].agg(
            n="count", centre="median", spread=lambda s: float(mad(s.to_numpy()))
        )
        w = stats["n"] / (stats["n"] + shrink_k)
        return cls(
            centre=w * stats["centre"] + (1.0 - w) * pop_centre,
            spread=(w * stats["spread"] + (1.0 - w) * pop_spread).clip(lower=max(1e-3 * abs(pop_centre), 1e-12)),
            n=stats["n"],
            pop_centre=pop_centre,
            pop_spread=pop_spread,
        )
