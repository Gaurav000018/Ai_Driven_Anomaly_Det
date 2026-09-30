"""Measure adaptive burn-in: can parts be released early without losing safety?

    python scripts/eval_adaptive_burnin.py

The architecture claims oven-time savings from early release. That claim was
written before anything measured it, so this script exists to either support it
or retract it.

The only honest way to do this is to score each checkpoint using ONLY the
measurements available at that checkpoint - a model that has seen 168 h data
cannot be asked whether 96 h would have been enough. So features, tracks,
fusion and threshold are all refitted per truncation, out-of-fold by lot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.harness import recall_at_fpr  # noqa: E402
from src.features.build import build_features  # noqa: E402
from src.fusion.adaptive_burnin import plan_adaptive_burnin  # noqa: E402
from src.fusion.cost_decision import CostDecision, CostModel  # noqa: E402
from src.fusion.meta_learner import RiskFusion, run_tracks_oof  # noqa: E402
from src.ingest.csv_reader import read_csv  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402
from src.module_a.autoencoder import TrajectoryAutoencoderTrack  # noqa: E402
from src.module_a.iforest import IsolationTrack  # noqa: E402
from src.module_a.mahalanobis import MahalanobisTrack  # noqa: E402
from src.module_a.physics_residual import PhysicsResidualTrack  # noqa: E402
from src.module_a.robust_z import RobustZTrack  # noqa: E402


def tracks():
    return [RobustZTrack(), MahalanobisTrack(), IsolationTrack(),
            TrajectoryAutoencoderTrack(), PhysicsResidualTrack()]


MIN_TIMEPOINTS = 3


def usable(df: pd.DataFrame, hours: float) -> bool:
    """Can a decision be made from the evidence available at this checkpoint?

    The power-law kernel fits two parameters, so it needs at least three
    timepoints. On a 0/24/96/168 schedule that makes 96 h the earliest
    checkpoint with any kinetic information at all - at 24 h there are two
    readings and a straight line through them says nothing about curvature.
    This is a real constraint on how early a part can be released, not a
    limitation worth engineering around.
    """
    return df.loc[df["hours"] <= hours, "hours"].nunique() >= MIN_TIMEPOINTS


def risk_at(df: pd.DataFrame, lib: MechanismLibrary, hours: float, n_splits: int = 3):
    """Out-of-fold risk using only measurements up to `hours`."""
    truncated = df[df["hours"] <= hours]
    feats = build_features(truncated, lib)
    y = feats["is_defect"].astype(int).to_numpy()
    lots = feats["lot_id"].astype(str)
    scores = run_tracks_oof(feats, tracks(), lots, n_splits=n_splits).fillna(0.0)
    fusion = RiskFusion().fit(scores, y, lots=lots)
    return fusion.risk(scores), y, feats, scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic" / "burnin.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "adaptive_burnin.json")
    ap.add_argument("--checkpoints", type=float, nargs="+", default=[24.0, 96.0])  # 24h is skipped: too few points
    ap.add_argument("--full-duration", type=float, default=168.0)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()

    lib = MechanismLibrary.load()
    df = read_csv(args.data, library=lib)
    costs = CostModel.load()

    print("=" * 76)
    print("  ADAPTIVE BURN-IN - can parts be released early, safely?")
    print("=" * 76)

    # Full-duration baseline: the escape count early release must not worsen.
    print(f"\n  baseline at {args.full_duration:.0f} h (all measurements) ...", flush=True)
    risk_full, y, feats_full, _ = risk_at(df, lib, args.full_duration, n_splits=args.splits)
    decider = CostDecision(costs).fit(risk_full, y)
    baseline_escapes = int(((risk_full < decider.accept_below_) & (y == 1)).sum())
    print(f"    recall@5%FPR {recall_at_fpr(y, risk_full, 0.05):.1%}, "
          f"escapes {baseline_escapes} of {int(y.sum())}")

    # Risk at each checkpoint, from truncated evidence only.
    per_checkpoint = {}
    for h in args.checkpoints:
        if not usable(df, h):
            n = int(df.loc[df["hours"] <= h, "hours"].nunique())
            print(f"\n  checkpoint {h:.0f} h SKIPPED - only {n} timepoint(s) available, "
                  f"the power-law kernel needs {MIN_TIMEPOINTS}")
            continue
        print(f"\n  checkpoint {h:.0f} h (evidence up to {h:.0f} h only) ...", flush=True)
        r, yc, _, _ = risk_at(df, lib, h, n_splits=args.splits)
        print(f"    recall@5%FPR {recall_at_fpr(yc, r, 0.05):.1%}")
        per_checkpoint[h] = r

    if not per_checkpoint:
        print("\n  no checkpoint has enough timepoints to decide on. Adaptive release "
              "needs a schedule with at least 3 readings before the horizon.")
        return 1

    labels = pd.Series(y, index=feats_full.index)

    plan = plan_adaptive_burnin(
        score_at=lambda h: per_checkpoint[h].reindex(labels.index).fillna(1e9),
        labels=labels,
        checkpoints=sorted(per_checkpoint),
        full_duration=args.full_duration,
        accept_below=float(decider.accept_below_),
        confidence_margin=args.margin,
        baseline_escapes=baseline_escapes,
    )

    print("\n" + "-" * 76)
    print(plan.summary())
    print("-" * 76)

    if plan.safe:
        print(f"\n  VERDICT: safe. {plan.oven_time_saved:.1%} of oven time released early "
              f"with no additional escapes.")
    else:
        print(f"\n  VERDICT: NOT safe at margin {args.margin}. Early release costs "
              f"{plan.escapes_adaptive - plan.escapes_baseline} extra escape(s). "
              "Tighten the margin or drop the earliest checkpoint.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "baseline_escapes": baseline_escapes,
        "oven_time_saved": plan.oven_time_saved,
        "escapes_adaptive": plan.escapes_adaptive,
        "safe": plan.safe,
        "confidence_margin": args.margin,
        "checkpoints": [c.as_row() for c in plan.checkpoints],
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
