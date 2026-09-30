"""Turn a forecast into a rejection decision.

Two independent rules, either of which is enough to reject:

  limit rule   the conformal UPPER bound at the horizon breaches the derated
               datasheet limit. Rejecting on the upper bound rather than the
               median is the whole point - the median lets the tail fly.

  slope rule   the projected drift rate exceeds the lot's own safe slope,
               taken as a high percentile of the lot's drift distribution. This
               catches the part that will not breach by 168h but is clearly on a
               trajectory its lot-mates are not on.

Both are expressed as a signed margin, so the fusion layer gets a continuous
risk signal rather than a bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .conformal import ConformalInterval

# The lot's safe slope is this percentile of its own observed drift rates.
SAFE_SLOPE_PCT = 99.0


@dataclass
class SafetyVerdict:
    margin_limit: pd.Series     # >0 means the upper bound breaches the limit
    margin_slope: pd.Series     # >0 means the projected slope is unsafe
    reject: pd.Series           # either rule fired
    risk: pd.Series             # continuous, 0 = safe, higher = worse

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "margin_limit": self.margin_limit,
                "margin_slope": self.margin_slope,
                "reject": self.reject,
                "risk": self.risk,
            }
        )


class SafetySlope:
    def __init__(self, safe_slope_pct: float = SAFE_SLOPE_PCT) -> None:
        self.safe_slope_pct = safe_slope_pct
        self.lot_safe_slope_: Optional[pd.Series] = None
        self.global_safe_slope_: float = np.inf

    def fit(self, drift_rate: pd.Series, lots: pd.Series) -> "SafetySlope":
        """Learn each lot's safe slope from that lot's own healthy behaviour.

        Fitted on the reference population, which is overwhelmingly healthy at
        1% prevalence, so a high percentile is a sound proxy for "as fast as a
        good part in this lot ever drifts".
        """
        rate = pd.Series(np.asarray(drift_rate, dtype=float)).reset_index(drop=True)
        lot_key = pd.Series(np.asarray(lots)).reset_index(drop=True)
        self.global_safe_slope_ = float(np.nanpercentile(rate.dropna(), self.safe_slope_pct))
        self.lot_safe_slope_ = rate.groupby(lot_key, observed=True).apply(
            lambda s: float(np.nanpercentile(s.dropna(), self.safe_slope_pct)) if s.notna().any() else np.nan
        )
        return self

    def evaluate(
        self,
        interval: ConformalInterval,
        v0: pd.Series,
        lots: pd.Series,
        derated_limit: float,
        horizon: float,
        index=None,
    ) -> SafetyVerdict:
        index = index if index is not None else v0.index
        upper = pd.Series(interval.upper, index=index)

        # Limit rule, normalised so the margin is comparable across parameters.
        margin_limit = (upper - derated_limit) / derated_limit

        # Slope rule, using the upper bound so the projection is conservative.
        projected = (upper - pd.Series(v0.to_numpy(dtype=float), index=index)) / horizon
        safe = pd.Series(lots.to_numpy(), index=index).map(self.lot_safe_slope_)
        safe = safe.fillna(self.global_safe_slope_).replace(0.0, np.nan)
        margin_slope = (projected - safe) / safe.abs()
        margin_slope = margin_slope.fillna(0.0)

        reject = (margin_limit > 0) | (margin_slope > 0)
        risk = np.maximum(margin_limit.clip(lower=0), margin_slope.clip(lower=0))

        return SafetyVerdict(
            margin_limit=margin_limit,
            margin_slope=margin_slope,
            reject=reject,
            risk=pd.Series(risk, index=index, name="module_b_risk"),
        )
