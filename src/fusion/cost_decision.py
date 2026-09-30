"""Turn Escape Risk into ACCEPT / REVIEW / REJECT under asymmetric cost.

The threshold is chosen by minimising expected cost, not by maximising F1.
Optimising F1 implicitly prices an escaped latent defect at one scrapped
component; the real ratio is nearer 1000:1, and at that ratio the optimal
operating point is nowhere near the F1 optimum.

Three bands rather than two, because that is how a QA floor actually runs. The
REVIEW band is where false-negative risk goes to die: an ambiguous part gets a
human instead of a coin flip. Its width is itself a cost decision - review time
is priced in `costs.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

DEFAULT_COSTS = Path(__file__).resolve().parents[2] / "configs" / "costs.yaml"

ACCEPT, REVIEW, REJECT = "ACCEPT", "REVIEW", "REJECT"


@dataclass
class CostModel:
    cost_false_negative: float = 1000.0
    cost_false_positive: float = 1.0
    cost_review: float = 12.0
    accept_below: float = 30.0
    reject_above: float = 70.0

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CostModel":
        path = path or DEFAULT_COSTS
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        bands = doc.get("bands", {})
        return cls(
            cost_false_negative=float(doc["cost_false_negative"]),
            cost_false_positive=float(doc["cost_false_positive"]),
            cost_review=float(doc.get("cost_review", 12.0)),
            accept_below=float(bands.get("accept_below", 30.0)),
            reject_above=float(bands.get("reject_above", 70.0)),
        )


@dataclass
class Operating:
    """The chosen operating point and what it costs."""

    threshold: float
    expected_cost: float
    recall: float
    fpr: float
    escapes: int
    false_alarms: int
    reviewed: int

    def as_dict(self) -> Dict:
        return asdict(self)


class CostDecision:
    def __init__(
        self,
        costs: Optional[CostModel] = None,
        data_driven_bands: bool = True,
        reject_precision: float = 0.5,
    ) -> None:
        self.costs = costs or CostModel.load()
        # Fixed band edges on a calibrated-probability scale do not work. At 1%
        # prevalence the calibrated risk piles up near zero, so a literal
        # "accept below 30" accepts nearly everything - including most defects.
        # The bands have to come from the risk distribution actually observed.
        self.data_driven_bands = data_driven_bands
        self.reject_precision = reject_precision
        self.operating_: Optional[Operating] = None
        self.accept_below_: Optional[float] = None
        self.reject_above_: Optional[float] = None

    # ------------------------------------------------------------- calibration

    def fit(
        self,
        risk: pd.Series,
        y_true: np.ndarray,
        severity_weight: Optional[np.ndarray] = None,
    ) -> "CostDecision":
        """Find the risk threshold with the lowest expected cost."""
        s = risk.to_numpy(dtype=float)
        y = np.asarray(y_true).astype(int)
        w = np.ones_like(y, dtype=float) if severity_weight is None else np.asarray(severity_weight, dtype=float)

        grid = np.unique(np.concatenate([np.linspace(0, 100, 501), s]))
        best: Optional[Operating] = None
        for t in grid:
            pred = s >= t
            missed = (~pred) & (y == 1)
            false_alarm = pred & (y == 0)
            cost = (
                self.costs.cost_false_negative * float(w[missed].sum())
                + self.costs.cost_false_positive * int(false_alarm.sum())
            )
            if best is None or cost < best.expected_cost:
                best = Operating(
                    threshold=float(t),
                    expected_cost=float(cost),
                    recall=float((pred & (y == 1)).sum() / max(int((y == 1).sum()), 1)),
                    fpr=float(int(false_alarm.sum()) / max(int((y == 0).sum()), 1)),
                    escapes=int(missed.sum()),
                    false_alarms=int(false_alarm.sum()),
                    reviewed=0,
                )
        self.operating_ = best

        if self.data_driven_bands and best is not None:
            # ACCEPT below the cost-optimal threshold: that is precisely the
            # point where the expected escape cost stops justifying more
            # screening, so auto-accepting below it is the defensible choice.
            self.accept_below_ = best.threshold
            # Auto-REJECT only where the flag is precise enough to scrap without
            # a human. Everything in between is REVIEW - which is the band that
            # exists to absorb false-negative risk.
            self.reject_above_ = self._precision_threshold(s, y, best.threshold)
        else:
            self.accept_below_ = self.costs.accept_below
            self.reject_above_ = self.costs.reject_above
        return self

    def _precision_threshold(self, s: np.ndarray, y: np.ndarray, floor: float) -> float:
        """Lowest threshold at or above `floor` whose precision clears the target.

        If no threshold is precise enough, returns above the maximum score so
        nothing is auto-rejected and every flagged part goes to a human. Failing
        safe here is the whole reason the REVIEW band exists.
        """
        grid = np.unique(s[s >= floor])
        for t in grid:
            pred = s >= t
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            if tp + fp > 0 and tp / (tp + fp) >= self.reject_precision:
                return float(t)
        return float(np.max(s)) + 1e-9

    # ----------------------------------------------------------------- decide

    def decide(
        self,
        risk: pd.Series,
        unknown_mechanism: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Assign each part a band.

        `unknown_mechanism` is the open-set flag from mechanism attribution. A
        part whose signature matches no catalogued mechanism is never accepted,
        whatever its score - an unrecognised degradation physics is the last
        thing that should be waved through on a low number.
        """
        s = risk.astype(float)
        accept_below = self.accept_below_ if self.accept_below_ is not None else self.costs.accept_below
        reject_above = self.reject_above_ if self.reject_above_ is not None else self.costs.reject_above

        band = pd.Series(REVIEW, index=s.index, name="decision")
        band[s < accept_below] = ACCEPT
        band[s >= reject_above] = REJECT

        floored = pd.Series(False, index=s.index)
        if unknown_mechanism is not None:
            unk = unknown_mechanism.reindex(s.index).fillna(False).astype(bool)
            floored = unk & (band == ACCEPT)
            band[floored] = REVIEW

        return pd.DataFrame(
            {
                "escape_risk": s,
                "decision": band,
                "floored_unknown": floored,
            }
        )

    def realised_cost(self, decisions: pd.DataFrame, y_true: np.ndarray) -> Dict:
        """What the three-band policy actually costs, review time included."""
        y = np.asarray(y_true).astype(int)
        band = decisions["decision"].to_numpy()
        accepted_defect = int(((band == ACCEPT) & (y == 1)).sum())
        rejected_good = int(((band == REJECT) & (y == 0)).sum())
        reviewed = int((band == REVIEW).sum())
        reviewed_defect = int(((band == REVIEW) & (y == 1)).sum())
        return {
            "escapes": accepted_defect,
            "scrapped_good": rejected_good,
            "reviewed": reviewed,
            "defects_sent_to_review": reviewed_defect,
            "recall_if_review_catches_all": float(
                (int((band == REJECT).sum() and ((band == REJECT) & (y == 1)).sum()) + reviewed_defect)
                / max(int((y == 1).sum()), 1)
            ),
            "total_cost": float(
                self.costs.cost_false_negative * accepted_defect
                + self.costs.cost_false_positive * rejected_good
                + self.costs.cost_review * reviewed
            ),
        }
