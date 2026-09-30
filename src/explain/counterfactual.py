"""Counterfactual pass conditions.

The useful question for a QA inspector is not "which features mattered" but
"what would this part have had to look like to pass". That is a checkable
statement, and it converts a model output into a specification.

The counterfactual here is envelope-based rather than gradient-based, and that
is a deliberate choice. A gradient counterfactual gives a point in feature space
that may be physically impossible - a trajectory with a negative exponent and a
positive jump, say. The envelope of parts the system actually accepted is, by
construction, achievable: every point in it is a real part that really passed.

So the statement produced is of the form:

    pass requires  Iddq at 24h <= 13.5 uA  and  exponent n <= 0.26

where each bound is the most permissive value seen among accepted parts in the
same lot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Features an inspector can actually reason about, in priority order.
# Derived scores (reconstruction error, Mahalanobis distance) are excluded on
# purpose: "reduce your reconstruction error" is not a pass condition.
ACTIONABLE_SUFFIXES = [
    ("__v_24h", "{param} at 24h", "{value:.3g} {unit}"),
    ("__v_96h", "{param} at 96h", "{value:.3g} {unit}"),
    ("__v_168h", "{param} at 168h", "{value:.3g} {unit}"),
    ("__n", "{param} degradation exponent n", "{value:.3f}"),
    ("__total_drift_frac", "{param} total drift", "{value:.1%} of initial"),
    ("__jump_score", "{param} largest step", "{value:.2g}x typical"),
    ("__noise_ratio", "{param} measurement scatter", "{value:.2g}x expected"),
]


@dataclass
class Condition:
    feature: str
    label: str
    actual: float
    bound: float
    unit: str
    formatted_actual: str
    formatted_bound: str
    exceedance: float          # how far over the bound, in bound units

    def sentence(self) -> str:
        return f"{self.label} <= {self.formatted_bound} (this part: {self.formatted_actual})"


def _fmt(template: str, value: float, unit: str) -> str:
    return template.format(value=value, unit=unit).strip()


class CounterfactualExplainer:
    def __init__(self, quantile: float = 1.0) -> None:
        # 1.0 takes the true maximum among accepted parts - the most permissive
        # honest bound. Lowering it gives a stricter, more conservative
        # statement if a fab wants margin.
        self.quantile = quantile
        self.bounds_: Dict[str, pd.Series] = {}
        self.global_bounds_: Dict[str, float] = {}
        self.units_: Dict[str, str] = {}

    def fit(
        self,
        features: pd.DataFrame,
        accepted: pd.Series,
        units: Optional[Dict[str, str]] = None,
    ) -> "CounterfactualExplainer":
        """Learn the accepted envelope, per lot and globally."""
        self.units_ = units or {}
        ok = features[accepted.reindex(features.index).fillna(False).astype(bool)]
        if ok.empty:
            raise ValueError("no accepted parts to build an envelope from")
        lots = ok["lot_id"].astype(str)

        for col in self._candidate_columns(features):
            series = pd.to_numeric(ok[col], errors="coerce")
            self.global_bounds_[col] = float(series.quantile(self.quantile))
            self.bounds_[col] = series.groupby(lots.to_numpy(), observed=True).quantile(self.quantile)
        return self

    @staticmethod
    def _candidate_columns(features: pd.DataFrame) -> List[str]:
        cols = []
        for suffix, _, _ in ACTIONABLE_SUFFIXES:
            cols.extend([c for c in features.columns if c.endswith(suffix)])
        return cols

    def explain(self, features: pd.DataFrame, part_id, top_k: int = 3) -> List[Condition]:
        """The conditions this part violates, worst first."""
        row = features.loc[part_id]
        lot = str(row["lot_id"])
        conditions: List[Condition] = []

        for suffix, label_tpl, value_tpl in ACTIONABLE_SUFFIXES:
            for col in [c for c in features.columns if c.endswith(suffix)]:
                if col not in self.bounds_:
                    continue
                actual = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                if not np.isfinite(actual):
                    continue
                bound = self.bounds_[col].get(lot, np.nan)
                if not np.isfinite(bound):
                    bound = self.global_bounds_.get(col, np.nan)
                if not np.isfinite(bound) or abs(bound) < 1e-12:
                    continue
                if actual <= bound:
                    continue  # this one already passes

                param = col.split("__")[0]
                unit = self.units_.get(param, "")
                conditions.append(
                    Condition(
                        feature=col,
                        label=label_tpl.format(param=param),
                        actual=float(actual),
                        bound=float(bound),
                        unit=unit,
                        formatted_actual=_fmt(value_tpl, float(actual), unit),
                        formatted_bound=_fmt(value_tpl, float(bound), unit),
                        exceedance=float((actual - bound) / abs(bound)),
                    )
                )

        conditions.sort(key=lambda c: c.exceedance, reverse=True)
        return conditions[:top_k]

    def sentence(self, features: pd.DataFrame, part_id, top_k: int = 2) -> str:
        conds = self.explain(features, part_id, top_k=top_k)
        if not conds:
            return "this part sits inside the accepted envelope on every actionable measure"
        return "pass requires " + " and ".join(c.sentence() for c in conds)
