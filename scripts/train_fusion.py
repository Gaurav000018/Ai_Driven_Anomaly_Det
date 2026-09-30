"""Phase 3 checkpoint: does learned fusion beat the best single track?

    python scripts/train_fusion.py

Runs all five Module A tracks out-of-fold, adds the Module B margin, fits the
calibrated meta-learner, and prints the ablation table showing each track's
marginal contribution.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.harness import by_mechanism, compare, evaluate  # noqa: E402
from src.features.build import build_features  # noqa: E402
from src.features.trajectory import build_panel  # noqa: E402
from src.fusion.cost_decision import CostDecision, CostModel  # noqa: E402
from src.fusion.meta_learner import RiskFusion, UnsupervisedFusion, run_tracks_oof  # noqa: E402
from src.ingest.csv_reader import read_csv  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402
from src.module_a.autoencoder import TrajectoryAutoencoderTrack  # noqa: E402
from src.module_a.iforest import IsolationTrack  # noqa: E402
from src.module_a.mahalanobis import MahalanobisTrack  # noqa: E402
from src.module_a.physics_residual import PhysicsResidualTrack  # noqa: E402
from src.module_a.robust_z import RobustZTrack  # noqa: E402
from src.module_b.pipeline import oof_risk  # noqa: E402


def all_tracks():
    return [
        RobustZTrack(),
        MahalanobisTrack(),
        IsolationTrack(),
        TrajectoryAutoencoderTrack(),
        PhysicsResidualTrack(),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic" / "burnin.csv")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "processed" / "features.parquet")
    ap.add_argument("--scores", type=Path, default=ROOT / "data" / "processed" / "oof_scores.parquet")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-module-b", action="store_true")
    args = ap.parse_args()

    lib = MechanismLibrary.load()
    df = read_csv(args.data)

    if args.cache.exists() and not args.rebuild:
        feats = pd.read_parquet(args.cache)
    else:
        print("features : building ...")
        feats = build_features(df, lib)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            feats.to_parquet(args.cache)
        except Exception:
            pass
    print(f"features : {feats.shape[0]:,} parts x {feats.shape[1]} columns")

    y = feats["is_defect"].astype(int).to_numpy()
    mech = feats["mechanism_id"].astype(str).to_numpy()
    lots = feats["lot_id"].astype(str)
    sev = np.array([lib.cost_multiplier(m) if d else 1.0 for m, d in zip(mech, y)])
    sev = np.where(sev > 0, sev, 1.0)

    if args.scores.exists() and not args.rebuild:
        scores = pd.read_parquet(args.scores)
        print(f"scores   : loaded from cache ({', '.join(scores.columns)})")
    else:
        print("scores   : running 5 Module A tracks out-of-fold ...")
        scores = run_tracks_oof(feats, all_tracks(), lots, n_splits=args.splits, verbose=True)
        if not args.no_module_b:
            print("scores   : running Module B out-of-fold ...")
            panel = build_panel(df)
            b = oof_risk(df, lib, panel=panel, n_splits=args.splits, verbose=True)
            scores = scores.join(b[["B_risk_max"]].reindex(scores.index))
        scores = scores.fillna(0.0)
        try:
            scores.to_parquet(args.scores)
        except Exception:
            pass

    results = [evaluate(y, scores[c], name=c, severity_weight=sev) for c in scores.columns]

    unsup = UnsupervisedFusion().risk(scores)
    results.append(evaluate(y, unsup, name="unsupervised rank mean", severity_weight=sev))

    fusion = RiskFusion().fit(scores, y, lots=lots)
    risk = fusion.risk(scores)
    results.append(evaluate(y, risk, name="RISK FUSION (learned)", severity_weight=sev))

    print("\n" + "=" * 100)
    print("  MODULE A + B FUSION - PHASE 3")
    print("=" * 100)
    print(compare(results).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nlearned track weights:")
    for name, w in fusion.report_.weights.items():
        bar = "#" * int(round(abs(w) * 12))
        print(f"  {name:<20} {w:>+7.3f}  {bar}")

    print("\nablation - drop one track, refit, measure the loss:")
    full = evaluate(y, risk, name="full", severity_weight=sev)
    rows = []
    for col in scores.columns:
        reduced = scores.drop(columns=[col])
        r = evaluate(y, RiskFusion().fit(reduced, y, lots=lots).risk(reduced), name=f"without {col}",
                     severity_weight=sev)
        rows.append({
            "dropped": col,
            "pr_auc": r.pr_auc,
            "d_pr_auc": r.pr_auc - full.pr_auc,
            "recall@fpr": r.recall_at_fpr,
            "d_recall": r.recall_at_fpr - full.recall_at_fpr,
        })
    abl = pd.DataFrame(rows).sort_values("d_pr_auc")
    print(abl.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\n  full model: pr_auc {full.pr_auc:.3f}, recall@5%FPR {full.recall_at_fpr:.3f}")

    costs = CostModel.load()
    decider = CostDecision(costs).fit(risk, y, severity_weight=sev)
    op = decider.operating_
    print("\ncost-optimal operating point "
          f"(C_FN:C_FP = {costs.cost_false_negative:.0f}:{costs.cost_false_positive:.0f}):")
    print(f"  threshold        : {op.threshold:.1f} / 100")
    print(f"  recall           : {op.recall:.1%}")
    print(f"  false alarm rate : {op.fpr:.2%}")
    print(f"  escapes          : {op.escapes} of {int(y.sum())}")
    print(f"  expected cost    : {op.expected_cost:,.0f}")

    decisions = decider.decide(risk)
    realised = decider.realised_cost(decisions, y)
    counts = decisions["decision"].value_counts()
    print(f"\nthree-band policy, data-driven edges "
          f"(accept <{decider.accept_below_:.3f}, reject >={decider.reject_above_:.3f}):")
    for band in ("ACCEPT", "REVIEW", "REJECT"):
        print(f"  {band:<8} {int(counts.get(band, 0)):>6,} parts")
    print(f"  escapes (defect accepted)   : {realised['escapes']}")
    print(f"  defects routed to review    : {realised['defects_sent_to_review']}")
    print(f"  good parts scrapped         : {realised['scrapped_good']}")

    per_mech = by_mechanism(y, risk, mech, op.threshold)
    print("\nper-mechanism recall at the cost-optimal threshold:")
    for _, r in per_mech.iterrows():
        bar = "#" * int(round(r["recall"] * 24))
        print(f"  {r['mechanism_id']:<16} {r['caught']:>3}/{r['n']:<3} {r['recall']:>6.1%}  {bar}")
    print(f"\n  WORST-CASE mechanism recall : {per_mech['recall'].min():.1%}")

    out = ROOT / "models" / "fusion.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump({"fusion": fusion, "costs": costs, "operating": op.as_dict(),
                     "library_hash": lib.source_hash}, fh)
    print(f"\nsaved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
