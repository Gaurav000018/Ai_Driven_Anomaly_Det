"""Unknown Defect Evaluation - the credibility firewall.

UDE-1  leave-one-mechanism-out. Train on every mechanism except m, evaluate on
       m alone. Answers: does this generalise to physics it has never seen?
UDE-2  out-of-family. Evaluate on functional forms absent from the library.
       Answers: is it detecting anomaly, or recognising its own parametric form?
UDE-3  open-set. Can attribution tell a known mechanism from an unknown one,
       and does it abstain rather than guess?

Two reporting rules, both non-negotiable:

  Report the WORST-CASE mechanism, not the mean. A system averaging 94% while
  missing electromigration entirely is a system that loses satellites to
  electromigration.

  UDE-1 recall is the honest headline. Closed-set recall is reported alongside
  and labelled optimistic, because that is what it is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..fusion.meta_learner import RiskFusion, run_tracks_oof
from ..module_a.base import DetectorTrack
from .harness import lot_folds, recall_at_fpr


@dataclass
class LomoResult:
    held_out: str
    mechanism_name: str
    severity: str
    n_test_defects: int
    recall_at_fpr: float
    recall_at_threshold: float
    closed_set_recall: float
    threshold: float

    def as_row(self) -> Dict:
        return asdict(self)


def _fresh(track: DetectorTrack) -> DetectorTrack:
    from ..fusion.meta_learner import _track_kwargs

    return track.__class__(**_track_kwargs(track))


def _fit_score(
    tracks: Sequence[DetectorTrack],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every track on `train`, score both frames."""
    train_scores, test_scores = {}, {}
    for track in tracks:
        fitted = _fresh(track).fit(train)
        train_scores[track.name] = fitted.score(train)
        test_scores[track.name] = fitted.score(test)
    return pd.DataFrame(train_scores), pd.DataFrame(test_scores)


def lomo(
    features: pd.DataFrame,
    tracks: Sequence[DetectorTrack],
    library,
    fpr_budget: float = 0.05,
    n_splits: int = 3,
    verbose: bool = True,
    closed_set: bool = True,
) -> pd.DataFrame:
    """UDE-1. Returns one row per held-out mechanism.

    Lots are split first, then mechanism m is removed from the TRAINING side
    only. Splitting by lot before removing the mechanism is what keeps the
    healthy population from appearing on both sides - remove the mechanism
    first and every healthy part leaks.
    """
    mech = features["mechanism_id"].astype(str)
    y_all = features["is_defect"].astype(int).to_numpy()
    lots = features["lot_id"].astype(str)
    rows: List[LomoResult] = []

    for held in library.defect_ids():
        recalls, thr_recalls, closed, thresholds, n_def = [], [], [], [], 0

        for train_lots_idx, test_idx in lot_folds(lots, n_splits=n_splits):
            train_mask = np.zeros(len(features), dtype=bool)
            train_mask[train_lots_idx] = True
            # The held-out mechanism is invisible during training.
            train_mask &= (mech != held).to_numpy()

            test_mask = np.zeros(len(features), dtype=bool)
            test_mask[test_idx] = True
            # Test on healthy parts plus the held-out mechanism only.
            eval_mask = test_mask & ((mech == held) | (y_all == 0))

            train = features[train_mask]
            test = features[eval_mask]
            if test["is_defect"].sum() == 0 or len(train) < 100:
                continue

            tr_scores, te_scores = _fit_score(tracks, train, test)
            fusion = RiskFusion().fit(tr_scores, features.loc[train.index, "is_defect"].astype(int).to_numpy(),
                                      lots=lots.loc[train.index])
            risk_test = fusion.risk(te_scores)
            y_test = test["is_defect"].astype(int).to_numpy()

            recalls.append(recall_at_fpr(y_test, risk_test, fpr_budget))

            # Threshold chosen on the training side only - never on the test
            # fold, which would be exactly the leak this protocol exists to stop.
            risk_train = fusion.risk(tr_scores)
            y_train = features.loc[train.index, "is_defect"].astype(int).to_numpy()
            thr = _threshold_at_fpr(y_train, risk_train, fpr_budget)
            thresholds.append(thr)
            thr_recalls.append(float(((risk_test >= thr) & (y_test == 1)).sum() / max(int(y_test.sum()), 1)))

            # Closed-set comparison: identical fold, but m stays in training.
            # The gap between this and the held-out number is the honest
            # measure of how much the system is leaning on having seen it.
            # It doubles the cost of the whole protocol, hence the switch.
            if closed_set:
                closed_mask = np.zeros(len(features), dtype=bool)
                closed_mask[train_lots_idx] = True
                closed_train = features[closed_mask]
                tr2, te2 = _fit_score(tracks, closed_train, test)
                f2 = RiskFusion().fit(
                    tr2, closed_train["is_defect"].astype(int).to_numpy(), lots=lots.loc[closed_train.index]
                )
                closed.append(recall_at_fpr(y_test, f2.risk(te2), fpr_budget))

            n_def += int(y_test.sum())

        if not recalls:
            continue
        m = library.get(held)
        rows.append(
            LomoResult(
                held_out=held,
                mechanism_name=m.name,
                severity=m.severity,
                n_test_defects=n_def,
                recall_at_fpr=float(np.mean(recalls)),
                recall_at_threshold=float(np.mean(thr_recalls)),
                closed_set_recall=float(np.mean(closed)) if closed else float('nan'),
                threshold=float(np.mean(thresholds)),
            )
        )
        if verbose:
            r = rows[-1]
            cs = f"(closed-set {r.closed_set_recall:6.1%})" if closed_set else ""
            print(f"  {held:<16} held-out recall@{fpr_budget:.0%}FPR = {r.recall_at_fpr:6.1%}  "
                  f"{cs}  n={r.n_test_defects}", flush=True)

    return pd.DataFrame([r.as_row() for r in rows])


def _threshold_at_fpr(y: np.ndarray, scores: pd.Series, max_fpr: float) -> float:
    """Highest-recall threshold whose false-alarm rate stays within budget."""
    s = np.asarray(scores, dtype=float)
    good = s[np.asarray(y) == 0]
    if good.size == 0:
        return float(np.min(s))
    return float(np.quantile(good, 1.0 - max_fpr))


def out_of_family_eval(
    train_features: pd.DataFrame,
    oof_features: pd.DataFrame,
    tracks: Sequence[DetectorTrack],
    fpr_budget: float = 0.05,
) -> Dict:
    """UDE-2. Train on library physics, test on forms the library cannot express."""
    tr_scores, te_scores = _fit_score(tracks, train_features, oof_features)
    y_train = train_features["is_defect"].astype(int).to_numpy()
    fusion = RiskFusion().fit(tr_scores, y_train, lots=train_features["lot_id"].astype(str))

    risk = fusion.risk(te_scores)
    y = oof_features["is_defect"].astype(int).to_numpy()
    thr = _threshold_at_fpr(y_train, fusion.risk(tr_scores), fpr_budget)

    per_form = {}
    forms = oof_features["mechanism_id"].astype(str)
    for form in sorted(set(forms[y == 1])):
        sel = (forms == form).to_numpy() & (y == 1)
        per_form[form] = float(((risk >= thr) & sel).sum() / max(int(sel.sum()), 1))

    return {
        "recall_at_fpr": recall_at_fpr(y, risk, fpr_budget),
        "recall_at_train_threshold": float(((risk >= thr) & (y == 1)).sum() / max(int(y.sum()), 1)),
        "roc_auc": float(roc_auc_score(y, risk)) if 0 < y.sum() < len(y) else float("nan"),
        "threshold": thr,
        "n_defects": int(y.sum()),
        "per_form_recall": per_form,
        "risk": risk,
    }


def open_set_eval(
    known_attr: pd.DataFrame,
    known_labels: np.ndarray,
    unknown_attr: pd.DataFrame,
    unknown_labels: np.ndarray,
    tau: float,
) -> Dict:
    """UDE-3. Does attribution distance separate known physics from unknown?

    The discriminator is the distance to the nearest prototype. If it works,
    a real defect from an uncatalogued mechanism lands far from everything and
    is abstained on rather than mislabelled.
    """
    d_known = known_attr.loc[known_labels == 1, "distance"].to_numpy(dtype=float)
    d_unknown = unknown_attr.loc[unknown_labels == 1, "distance"].to_numpy(dtype=float)
    if d_known.size == 0 or d_unknown.size == 0:
        return {"auroc": float("nan"), "n_known": int(d_known.size), "n_unknown": int(d_unknown.size)}

    y = np.concatenate([np.zeros(d_known.size), np.ones(d_unknown.size)])
    s = np.concatenate([d_known, d_unknown])
    abstain_unknown = float(np.mean(d_unknown > tau))
    abstain_known = float(np.mean(d_known > tau))
    flagged = d_unknown.size * abstain_unknown + d_known.size * abstain_known
    return {
        "auroc": float(roc_auc_score(y, s)),
        "tau": tau,
        "abstention_rate_on_unknown": abstain_unknown,
        "abstention_rate_on_known": abstain_known,
        "abstention_precision": float(d_unknown.size * abstain_unknown / flagged) if flagged > 0 else float("nan"),
        "median_distance_known": float(np.median(d_known)),
        "median_distance_unknown": float(np.median(d_unknown)),
        "n_known": int(d_known.size),
        "n_unknown": int(d_unknown.size),
    }


def suggest_tau(known_attr: pd.DataFrame, known_labels: np.ndarray, target_keep: float = 0.95) -> float:
    """Pick tau so that `target_keep` of KNOWN defects are still attributed.

    Setting tau from the known side alone is deliberate: the threshold must be
    choosable without ever having seen an unknown mechanism, which is the whole
    situation it is meant to handle.
    """
    d = known_attr.loc[known_labels == 1, "distance"].to_numpy(dtype=float)
    if d.size == 0:
        return 3.0
    return float(np.quantile(d, target_keep))
