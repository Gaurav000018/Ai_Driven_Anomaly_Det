"""Is the explanation actually true?

An attribution nobody checked is decoration. A certificate that names
`robust_z 51% | autoencoder 35%` is making a falsifiable claim about what drove
the decision, and these are the tests that falsify it.

Two properties, and they are independent:

  faithfulness  if the named drivers really carry the decision, removing them
                should collapse the score. Removing features the explanation
                called irrelevant should not. Measured by deletion: replace the
                top-k attributed tracks with their population median and see how
                far the risk falls, against the same deletion applied to
                randomly chosen tracks.

  stability     the same part explained twice should give the same reasons. A
                model refitted on a bootstrap resample should rank the drivers
                the same way. Rank correlation below ~0.85 means the
                explanation is an artefact of the particular fit, and telling a
                QA inspector "it was the autoencoder" when a resample would
                have said "it was robust-Z" is worse than saying nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..fusion.meta_learner import RiskFusion


@dataclass
class FaithfulnessResult:
    top_k: int
    n_parts: int
    drop_top_k: float          # mean fractional fall in risk, top-k deleted
    drop_random_k: float       # same, k randomly chosen tracks
    drop_bottom_k: float       # same, k least-attributed tracks
    advantage: float           # drop_top_k - drop_random_k
    passes: bool

    def as_row(self) -> Dict:
        return asdict(self)


@dataclass
class StabilityResult:
    n_bootstrap: int
    mean_rank_correlation: float
    min_rank_correlation: float
    frac_above_threshold: float
    threshold: float
    passes: bool

    def as_row(self) -> Dict:
        return asdict(self)


def _delete(scores: pd.DataFrame, columns: Sequence[str], baseline: pd.Series) -> pd.DataFrame:
    """Replace the given tracks with their population median.

    Median substitution rather than zeroing: zero is an extreme value on a
    robust-z scale and would itself move the score, confounding the test.
    """
    out = scores.copy()
    for col in columns:
        out[col] = baseline[col]
    return out


def faithfulness(
    fusion: RiskFusion,
    scores: pd.DataFrame,
    subset: Optional[pd.Index] = None,
    top_k: int = 2,
    n_random: int = 5,
    seed: int = 0,
    min_advantage: float = 0.10,
) -> FaithfulnessResult:
    """Deletion test on the parts the system actually flagged.

    Run on flagged parts only. Deleting the drivers of an already-low risk
    score has nowhere to fall, so averaging over the whole population would
    dilute the measurement into noise.
    """
    idx = subset if subset is not None else scores.index
    sub = scores.loc[idx]
    rng = np.random.default_rng(seed)

    baseline_medians = scores.median()
    risk0 = fusion.risk(sub)
    contrib = fusion.contributions(sub)
    cols = list(scores.columns)
    k = min(top_k, len(cols) - 1)

    def frac_drop(modified: pd.DataFrame) -> np.ndarray:
        r = fusion.risk(modified)
        denom = risk0.to_numpy()
        denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
        return np.clip((risk0.to_numpy() - r.to_numpy()) / denom, -5.0, 1.0)

    # Per-part top-k and bottom-k, since different parts have different drivers.
    top_mod, bot_mod = sub.copy(), sub.copy()
    order = np.argsort(-contrib.to_numpy(), axis=1)
    for i, part in enumerate(sub.index):
        top_cols = [cols[j] for j in order[i, :k]]
        bot_cols = [cols[j] for j in order[i, -k:]]
        for c in top_cols:
            top_mod.loc[part, c] = baseline_medians[c]
        for c in bot_cols:
            bot_mod.loc[part, c] = baseline_medians[c]

    drop_top = float(np.nanmean(frac_drop(top_mod)))
    drop_bot = float(np.nanmean(frac_drop(bot_mod)))

    rand_drops = []
    for _ in range(n_random):
        pick = rng.choice(cols, size=k, replace=False)
        rand_drops.append(float(np.nanmean(frac_drop(_delete(sub, pick, baseline_medians)))))
    drop_rand = float(np.mean(rand_drops))

    advantage = drop_top - drop_rand
    return FaithfulnessResult(
        top_k=k,
        n_parts=int(len(sub)),
        drop_top_k=drop_top,
        drop_random_k=drop_rand,
        drop_bottom_k=drop_bot,
        advantage=advantage,
        passes=bool(advantage >= min_advantage and drop_top > drop_bot),
    )


def stability(
    scores: pd.DataFrame,
    y: np.ndarray,
    lots: pd.Series,
    subset: Optional[pd.Index] = None,
    n_bootstrap: int = 12,
    seed: int = 0,
    threshold: float = 0.85,
) -> StabilityResult:
    """Refit the fusion on bootstrap resamples and compare driver rankings.

    Resampling is by LOT, not by part. Parts within a lot share their reference
    statistics, so a part-level bootstrap would leave almost every lot intact
    and report a stability that the model does not have.
    """
    idx = subset if subset is not None else scores.index
    lot_series = lots.astype(str)
    unique_lots = lot_series.unique()
    rng = np.random.default_rng(seed)

    reference = RiskFusion().fit(scores, y, lots=lot_series)
    ref_contrib = reference.contributions(scores.loc[idx])

    correlations: List[float] = []
    for _ in range(n_bootstrap):
        picked = rng.choice(unique_lots, size=len(unique_lots), replace=True)
        # A bootstrap draws lots with replacement, so build the row index by
        # concatenating each drawn lot's parts.
        rows = np.concatenate([np.flatnonzero((lot_series == l).to_numpy()) for l in picked])
        boot_scores = scores.iloc[rows]
        boot_y = np.asarray(y)[rows]
        if boot_y.sum() < 2 or boot_y.sum() == len(boot_y):
            continue

        boot = RiskFusion().fit(boot_scores, boot_y, lots=lot_series.iloc[rows])
        boot_contrib = boot.contributions(scores.loc[idx])

        for part in idx:
            a = ref_contrib.loc[part].to_numpy(dtype=float)
            b = boot_contrib.loc[part].to_numpy(dtype=float)
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            rho = spearmanr(a, b).statistic
            if np.isfinite(rho):
                correlations.append(float(rho))

    if not correlations:
        return StabilityResult(0, float("nan"), float("nan"), float("nan"), threshold, False)

    arr = np.asarray(correlations)
    mean_rho = float(arr.mean())
    return StabilityResult(
        n_bootstrap=n_bootstrap,
        mean_rank_correlation=mean_rho,
        min_rank_correlation=float(arr.min()),
        frac_above_threshold=float((arr >= threshold).mean()),
        threshold=threshold,
        passes=bool(mean_rho >= threshold),
    )


def counterfactual_validity(
    features: pd.DataFrame,
    explainer,
    part_ids: Sequence,
    accepted: pd.Series,
) -> Dict:
    """Are the stated pass conditions actually satisfiable?

    The envelope counterfactual promises every bound is achievable because some
    real accepted part achieved it. This checks that promise literally: for each
    named condition, at least one accepted part in the same lot must satisfy it.
    """
    total, satisfiable, no_conditions = 0, 0, 0
    for pid in part_ids:
        conds = explainer.explain(features, pid, top_k=3)
        if not conds:
            no_conditions += 1
            continue
        lot = str(features.loc[pid, "lot_id"])
        peers = features[(features["lot_id"].astype(str) == lot)
                         & accepted.reindex(features.index).fillna(False).to_numpy()]
        for c in conds:
            total += 1
            if len(peers) and (pd.to_numeric(peers[c.feature], errors="coerce") <= c.bound).any():
                satisfiable += 1

    return {
        "conditions_checked": total,
        "satisfiable": satisfiable,
        "satisfiable_fraction": float(satisfiable / total) if total else float("nan"),
        "parts_with_no_conditions": no_conditions,
    }
