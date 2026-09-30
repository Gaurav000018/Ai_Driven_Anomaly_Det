"""Digital twin: what-if analysis and adaptive burn-in.

Burn-in is expensive. A 168-hour soak at 125 C occupies oven capacity, and
every part sits there for the full duration because the schedule is fixed in
advance - even though most parts have declared themselves clean long before the
end.

Adaptive burn-in asks a different question at each checkpoint: *can this part be
resolved now?* A part whose risk is confidently below the accept edge on the
evidence available at 96 h does not need another 72 hours in the oven. A part
still ambiguous gets extended rather than waved through.

The safety argument has to run the other way round from the savings argument.
The escape rate is fixed first - early release is only permitted where it does
not increase escapes against the full-duration baseline - and whatever oven time
that leaves over is the saving. Framing it as "save 38% of oven time and see
what it costs" is how a screening programme gets quietly degraded.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CheckpointResult:
    hours: float
    n_resolved: int             # parts confidently accepted and released early
    n_flagged: int              # parts sent to reject/review at this checkpoint
    n_continuing: int           # still ambiguous, stay in the oven
    escapes_here: int           # defects released early - must stay at zero
    oven_hours_saved: float
    cumulative_released: int

    def as_row(self) -> Dict:
        return asdict(self)


@dataclass
class AdaptivePlan:
    checkpoints: List[CheckpointResult]
    total_oven_hours_baseline: float
    total_oven_hours_adaptive: float
    escapes_baseline: int
    escapes_adaptive: int
    n_parts: int

    @property
    def oven_time_saved(self) -> float:
        if self.total_oven_hours_baseline <= 0:
            return 0.0
        return 1.0 - self.total_oven_hours_adaptive / self.total_oven_hours_baseline

    @property
    def safe(self) -> bool:
        """Early release must not cost a single extra escape."""
        return self.escapes_adaptive <= self.escapes_baseline

    def summary(self) -> str:
        lines = [
            f"parts                     : {self.n_parts:,}",
            f"baseline oven time        : {self.total_oven_hours_baseline:,.0f} part-hours",
            f"adaptive oven time        : {self.total_oven_hours_adaptive:,.0f} part-hours",
            f"oven time saved           : {self.oven_time_saved:.1%}",
            f"escapes, full duration    : {self.escapes_baseline}",
            f"escapes, adaptive         : {self.escapes_adaptive}",
            f"safety                    : {'PRESERVED' if self.safe else 'DEGRADED - do not deploy'}",
            "",
            "  checkpoint   released   flagged   continuing   escapes   hours saved",
        ]
        for c in self.checkpoints:
            lines.append(
                f"  {c.hours:>7.0f} h {c.n_resolved:>10,} {c.n_flagged:>9,} "
                f"{c.n_continuing:>12,} {c.escapes_here:>9} {c.oven_hours_saved:>13,.0f}"
            )
        return "\n".join(lines)


def plan_adaptive_burnin(
    score_at: Callable[[float], pd.Series],
    labels: pd.Series,
    checkpoints: Sequence[float],
    full_duration: float,
    accept_below: float,
    confidence_margin: float = 0.5,
    baseline_escapes: int = 0,
) -> AdaptivePlan:
    """Simulate early release against a fixed escape budget.

    `score_at(hours) -> risk` must return the risk each part would have been
    assigned using ONLY measurements up to `hours`. Anything else is leakage
    from the future and makes the whole analysis meaningless.

    `confidence_margin` shrinks the accept edge at early checkpoints. Evidence
    from two timepoints is weaker than from four, and releasing on a threshold
    calibrated for the full trajectory would be trusting a partial trajectory as
    though it were complete.
    """
    y = labels.astype(int)
    released: Dict[str, float] = {}
    results: List[CheckpointResult] = []
    escapes = 0

    for hours in sorted(checkpoints):
        if hours >= full_duration:
            continue
        risk = score_at(hours)
        pending = risk.index.difference(pd.Index(list(released)))
        if len(pending) == 0:
            continue

        r = risk.loc[pending]
        early_edge = accept_below * confidence_margin
        resolve = r[r < early_edge].index

        escapes_here = int(y.loc[resolve].sum()) if len(resolve) else 0
        escapes += escapes_here
        for part in resolve:
            released[part] = hours

        saved = float(len(resolve) * (full_duration - hours))
        results.append(
            CheckpointResult(
                hours=float(hours),
                n_resolved=int(len(resolve)),
                n_flagged=0,
                n_continuing=int(len(pending) - len(resolve)),
                escapes_here=escapes_here,
                oven_hours_saved=saved,
                cumulative_released=len(released),
            )
        )

    n = len(labels)
    baseline_hours = float(n * full_duration)
    adaptive_hours = baseline_hours - sum(c.oven_hours_saved for c in results)

    return AdaptivePlan(
        checkpoints=results,
        total_oven_hours_baseline=baseline_hours,
        total_oven_hours_adaptive=adaptive_hours,
        escapes_baseline=baseline_escapes,
        escapes_adaptive=baseline_escapes + escapes,
        n_parts=n,
    )


def extend_burnin_whatif(
    risk: pd.Series,
    labels: pd.Series,
    accept_below: float,
    extra_hours: float,
    cost_per_part_hour: float = 0.01,
    sensitivity_gain: float = 0.15,
) -> Dict:
    """What would extending the soak buy?

    Parts still near the decision boundary are the ones an extended soak could
    resolve, because more hours means more drift and a clearer signal. The gain
    is modelled, not measured - it assumes a longer soak separates borderline
    parts by `sensitivity_gain` - so this is a planning tool for choosing what
    to run next, not evidence about what would happen.
    """
    borderline = risk[(risk >= accept_below * 0.5) & (risk < accept_below * 2.0)]
    would_resolve = int(round(len(borderline) * sensitivity_gain))
    defect_rate = float(labels.loc[borderline.index].mean()) if len(borderline) else 0.0
    extra_catches = int(round(would_resolve * defect_rate))

    return {
        "extra_hours": extra_hours,
        "borderline_parts": int(len(borderline)),
        "expected_newly_resolved": would_resolve,
        "expected_extra_defects_caught": extra_catches,
        "added_cost": float(len(risk) * extra_hours * cost_per_part_hour),
        "modelled": True,
        "caveat": "gain is modelled from a sensitivity assumption, not measured",
    }
