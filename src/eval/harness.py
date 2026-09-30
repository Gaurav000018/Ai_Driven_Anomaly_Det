"""Evaluation built around the fact that a false negative is catastrophic.

Three choices here are deliberate and worth defending:

  * PR-AUC, not ROC-AUC. At 1% prevalence ROC-AUC flatters everything - a
    detector can score 0.95 while its top-ranked parts are almost all good ones.
  * F-beta with beta = 5, which weights recall 25x over precision.
  * Expected cost as the primary threshold criterion, at C_FN : C_FP = 1000 : 1.
    Optimising F1 would implicitly price an escaped defect at one scrapped part.

Cross-validation groups by lot. Splitting a lot across folds leaks its median
and MAD into training and inflates every number reported.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

DEFAULT_C_FN = 1000.0
DEFAULT_C_FP = 1.0
DEFAULT_BETA = 5.0


@dataclass
class ScreeningResult:
    name: str
    n: int
    n_defect: int
    pr_auc: float
    roc_auc: float
    recall_at_fpr: float
    fpr_budget: float
    best_fbeta: float
    best_fbeta_threshold: float
    recall_at_best_fbeta: float
    precision_at_best_fbeta: float
    min_expected_cost: float
    cost_threshold: float
    recall_at_cost_opt: float
    fpr_at_cost_opt: float
    escapes_at_cost_opt: int

    def as_row(self) -> Dict:
        return asdict(self)


def _as_arrays(y_true, scores) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(pd.Series(y_true).astype(int))
    s = np.asarray(pd.Series(scores).astype(float))
    s = np.nan_to_num(s, nan=np.nanmin(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0)
    return y, s


def recall_at_fpr(y_true, scores, max_fpr: float = 0.05) -> float:
    """Best recall achievable while holding the false-alarm rate at or below budget."""
    y, s = _as_arrays(y_true, scores)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def fbeta_curve(y_true, scores, beta: float = DEFAULT_BETA) -> pd.DataFrame:
    y, s = _as_arrays(y_true, scores)
    thresholds = np.unique(s)
    if len(thresholds) > 2000:
        thresholds = np.quantile(s, np.linspace(0, 1, 2000))
    rows = []
    b2 = beta**2
    for t in thresholds:
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denom = b2 * precision + recall
        fbeta = (1 + b2) * precision * recall / denom if denom > 0 else 0.0
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "fbeta": fbeta,
                     "tp": tp, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def cost_curve(
    y_true,
    scores,
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
    severity_weight: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Expected screening cost as a function of the rejection threshold.

    `severity_weight` lets a critical mechanism's escape cost more than a minor
    one, taken from the library's severity_cost_multiplier.
    """
    y, s = _as_arrays(y_true, scores)
    w = np.ones_like(y, dtype=float) if severity_weight is None else np.asarray(severity_weight, dtype=float)

    thresholds = np.unique(s)
    if len(thresholds) > 2000:
        thresholds = np.quantile(s, np.linspace(0, 1, 2000))

    rows = []
    for t in thresholds:
        pred = s >= t
        missed = (~pred) & (y == 1)
        false_alarm = pred & (y == 0)
        cost = c_fn * float(w[missed].sum()) + c_fp * int(false_alarm.sum())
        tp = int((pred & (y == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "cost": cost,
                "escapes": int(missed.sum()),
                "false_alarms": int(false_alarm.sum()),
                "recall": tp / max(int((y == 1).sum()), 1),
                "fpr": int(false_alarm.sum()) / max(int((y == 0).sum()), 1),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    y_true,
    scores,
    name: str = "detector",
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
    beta: float = DEFAULT_BETA,
    fpr_budget: float = 0.05,
    severity_weight: Optional[np.ndarray] = None,
) -> ScreeningResult:
    y, s = _as_arrays(y_true, scores)
    n_defect = int(y.sum())

    fb = fbeta_curve(y, s, beta=beta)
    best = fb.loc[fb["fbeta"].idxmax()]

    cc = cost_curve(y, s, c_fn=c_fn, c_fp=c_fp, severity_weight=severity_weight)
    cheapest = cc.loc[cc["cost"].idxmin()]

    return ScreeningResult(
        name=name,
        n=len(y),
        n_defect=n_defect,
        pr_auc=float(average_precision_score(y, s)) if 0 < n_defect < len(y) else float("nan"),
        roc_auc=float(roc_auc_score(y, s)) if 0 < n_defect < len(y) else float("nan"),
        recall_at_fpr=recall_at_fpr(y, s, fpr_budget),
        fpr_budget=fpr_budget,
        best_fbeta=float(best["fbeta"]),
        best_fbeta_threshold=float(best["threshold"]),
        recall_at_best_fbeta=float(best["recall"]),
        precision_at_best_fbeta=float(best["precision"]),
        min_expected_cost=float(cheapest["cost"]),
        cost_threshold=float(cheapest["threshold"]),
        recall_at_cost_opt=float(cheapest["recall"]),
        fpr_at_cost_opt=float(cheapest["fpr"]),
        escapes_at_cost_opt=int(cheapest["escapes"]),
    )


def compare(results: Iterable[ScreeningResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.as_row() for r in results])
    cols = ["name", "pr_auc", "recall_at_fpr", "best_fbeta", "recall_at_cost_opt",
            "fpr_at_cost_opt", "escapes_at_cost_opt", "min_expected_cost"]
    return df[cols].sort_values("min_expected_cost").reset_index(drop=True)


def by_mechanism(y_true, scores, mechanism_ids, threshold: float) -> pd.DataFrame:
    """Per-mechanism recall at a fixed threshold.

    The headline number is the MINIMUM of this table, not the mean. A system
    averaging 94% while missing electromigration entirely is a system that
    loses satellites to electromigration.
    """
    y, s = _as_arrays(y_true, scores)
    mech = pd.Series(mechanism_ids).astype(str).to_numpy()
    caught = s >= threshold
    rows = []
    for m in sorted(set(mech[y == 1])):
        sel = (mech == m) & (y == 1)
        rows.append(
            {
                "mechanism_id": m,
                "n": int(sel.sum()),
                "caught": int((sel & caught).sum()),
                "recall": float((sel & caught).sum() / max(int(sel.sum()), 1)),
            }
        )
    return pd.DataFrame(rows).sort_values("recall").reset_index(drop=True)


def lot_folds(lot_ids: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by lot. Never split a lot across folds."""
    lots = pd.Series(lot_ids).astype(str).to_numpy()
    n_splits = min(n_splits, len(np.unique(lots)))
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(lots), 1))
    return list(gkf.split(dummy, groups=lots))
