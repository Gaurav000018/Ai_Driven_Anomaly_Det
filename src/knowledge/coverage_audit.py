"""Coverage audit: what the library CLAIMS each track catches, against measurement.

Every mechanism entry declares a `detectability` map - which tracks should catch
it and how strongly. That is a falsifiable claim, and this module falsifies it.

The output is the most useful planning artefact in the project: the cells where
the library promised a strong detector and the measurement disagrees are the
next sprint's backlog, and the mechanisms with no strong detector at all are the
gaps to admit openly rather than wait for a judge to find.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .library import MechanismLibrary
from .schema import DETECTABILITY_LEVELS

# Map a track's class name onto the key used in the library's detectability map.
TRACK_KEYS = {
    "robust_z": "robust_z",
    "mahalanobis": "mahalanobis",
    "isolation": "iforest",
    "autoencoder": "autoencoder",
    "physics_residual": "physics_residual",
}

# Measured recall at or above this is treated as "strong" for comparison.
STRONG_RECALL = 0.80
PARTIAL_RECALL = 0.40


def claimed_matrix(library: MechanismLibrary) -> pd.DataFrame:
    """The library's own claims, as an ordinal matrix."""
    rows = {}
    for mech in library.defects():
        rows[mech.id] = {
            track: DETECTABILITY_LEVELS.get(mech.detectability.get(key, "none"), 0)
            for track, key in TRACK_KEYS.items()
        }
    return pd.DataFrame(rows).T


def measured_matrix(
    features: pd.DataFrame,
    track_scores: pd.DataFrame,
    fpr_budget: float = 0.05,
) -> pd.DataFrame:
    """Per-mechanism recall for each track, at a shared false-alarm budget."""
    y = features["is_defect"].astype(int).to_numpy()
    mech = features["mechanism_id"].astype(str).to_numpy()
    out: Dict[str, Dict[str, float]] = {}

    for track in track_scores.columns:
        s = track_scores[track].to_numpy(dtype=float)
        good = s[y == 0]
        thr = float(np.quantile(good, 1.0 - fpr_budget)) if good.size else float(np.min(s))
        for m in sorted(set(mech[y == 1])):
            sel = (mech == m) & (y == 1)
            recall = float(((s >= thr) & sel).sum() / max(int(sel.sum()), 1))
            out.setdefault(m, {})[track] = recall

    return pd.DataFrame(out).T


def _ordinal(recall: float) -> int:
    if recall >= STRONG_RECALL:
        return 3
    if recall >= PARTIAL_RECALL:
        return 2
    if recall > 0.05:
        return 1
    return 0


def audit(
    library: MechanismLibrary,
    features: pd.DataFrame,
    track_scores: pd.DataFrame,
    fpr_budget: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Compare claims against measurement and surface the disagreements."""
    claimed = claimed_matrix(library)
    measured = measured_matrix(features, track_scores, fpr_budget=fpr_budget)

    common_tracks = [t for t in claimed.columns if t in measured.columns]
    common_mech = [m for m in claimed.index if m in measured.index]
    claimed = claimed.loc[common_mech, common_tracks]
    measured = measured.loc[common_mech, common_tracks]
    measured_ord = measured.map(_ordinal)

    gap = measured_ord - claimed

    # Mechanisms with no strong detector anywhere - the real coverage holes.
    uncovered = measured.max(axis=1)
    holes = pd.DataFrame(
        {
            "best_track": measured.idxmax(axis=1),
            "best_recall": uncovered,
            "severity": [library.get(m).severity for m in measured.index],
        }
    ).sort_values("best_recall")

    overclaimed = []
    for m in common_mech:
        for t in common_tracks:
            if claimed.loc[m, t] >= 3 and measured_ord.loc[m, t] < 3:
                overclaimed.append(
                    {"mechanism_id": m, "track": t,
                     "claimed": "strong", "measured_recall": float(measured.loc[m, t])}
                )

    return {
        "claimed": claimed,
        "measured": measured,
        "gap": gap,
        "holes": holes,
        "overclaimed": pd.DataFrame(overclaimed),
    }


def render(result: Dict[str, pd.DataFrame], library: MechanismLibrary) -> str:
    lines = ["measured per-mechanism recall by track (at the shared FPR budget):", ""]
    measured = result["measured"]
    header = "  " + f"{'mechanism':<16}" + "".join(f"{t[:12]:>14}" for t in measured.columns)
    lines.append(header)
    for m in measured.index:
        row = "  " + f"{m:<16}" + "".join(f"{measured.loc[m, t]:>13.0%} " for t in measured.columns)
        lines.append(row)

    lines.append("")
    lines.append("coverage holes - mechanisms with no strong detector:")
    holes = result["holes"]
    none_strong = holes[holes["best_recall"] < STRONG_RECALL]
    if none_strong.empty:
        lines.append("  none; every mechanism has at least one track above "
                     f"{STRONG_RECALL:.0%} recall")
    else:
        for m, r in none_strong.iterrows():
            lines.append(f"  {m:<16} best {r['best_track']} at {r['best_recall']:.0%} "
                         f"(severity {r['severity']})")

    over = result["overclaimed"]
    lines.append("")
    lines.append("library claims not supported by measurement:")
    if over.empty:
        lines.append("  none; every 'strong' claim is borne out")
    else:
        for _, r in over.iterrows():
            lines.append(f"  {r['mechanism_id']:<16} claims {r['track']} strong, "
                         f"measured {r['measured_recall']:.0%}")
    return "\n".join(lines)
