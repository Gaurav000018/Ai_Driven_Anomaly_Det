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
    cost_review: float = 0.25
    accept_below: float = 30.0
    reject_above: float = 70.0
    max_review_fraction: float = 0.05

    def __post_init__(self) -> None:
        # Reviewing a part is only rational if it is cheaper than scrapping it.
        # Violate this and the band optimiser correctly - and uselessly -
        # collapses REVIEW to nothing, which is a confusing way to discover a
        # typo in a config file.
        if self.cost_review >= self.cost_false_positive:
            raise ValueError(
                f"cost_review ({self.cost_review}) must be below cost_false_positive "
                f"({self.cost_false_positive}); otherwise scrapping always beats "
                "reviewing and the REVIEW band cannot exist"
            )
        if self.cost_false_negative <= self.cost_false_positive:
            raise ValueError("cost_false_negative must exceed cost_false_positive for screening to make sense")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CostModel":
        path = path or DEFAULT_COSTS
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        bands = doc.get("bands", {})
        return cls(
            cost_false_negative=float(doc["cost_false_negative"]),
            cost_false_positive=float(doc["cost_false_positive"]),
            cost_review=float(doc.get("cost_review", 0.25)),
            max_review_fraction=float(doc.get("max_review_fraction", 0.05)),
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
            self.accept_below_, self.reject_above_ = self._optimise_bands(s, y, w)
        else:
            self.accept_below_ = self.costs.accept_below
            self.reject_above_ = self.costs.reject_above
        return self

    def _optimise_bands(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple:
        """Choose both band edges together, with review priced in.

        Setting ACCEPT at the binary cost-optimal threshold sends everything
        above it to REVIEW, which at a 1000:1 cost ratio meant 45% of the lot
        going to a human - technically cheap, operationally nonsense, because
        review time was in costs.yaml and never entered the objective.

        The real objective has three terms:

            C_FN * escapes + C_FP * scrapped_good + C_review * reviewed

        with reviewed defects counted as caught, since that is what the band is
        for. Optimising both edges against it yields a REVIEW band sized to what
        the inspection budget can actually absorb.
        """
        grid = np.unique(np.quantile(s, np.linspace(0.0, 1.0, 201)))
        review_cap = int(np.floor(self.costs.max_review_fraction * len(s)))
        best_cost, best_pair = np.inf, (float(grid[0]), float(grid[-1]))

        for accept in grid:
            accepted = s < accept
            # Escapes can only happen below the ACCEPT edge; above it a part is
            # either scrapped or seen by a human, and both catch the defect.
            escapes = float(w[accepted & (y == 1)].sum())
            base = self.costs.cost_false_negative * escapes
            if base >= best_cost:
                continue  # no reject edge can rescue this accept edge
            for reject in grid[grid >= accept]:
                rejected = s >= reject
                reviewed = int((~accepted & ~rejected).sum())
                if reviewed > review_cap:
                    continue  # beyond what the line can actually inspect
                scrapped_good = int((rejected & (y == 0)).sum())
                cost = (
                    base
                    + self.costs.cost_false_positive * scrapped_good
                    + self.costs.cost_review * reviewed
                )
                if cost < best_cost:
                    best_cost, best_pair = cost, (float(accept), float(reject))

        self.band_cost_ = float(best_cost)
        return best_pair

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
