"""Phase 4.5: Unknown Defect Evaluation.

    python scripts/run_ude.py                 # all four tiers
    python scripts/run_ude.py --only lomo

This is the protocol that answers "you generated your own defects, so of course
you detect them" with numbers instead of argument.
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
from src.eval.subtlety_sweep import minimum_detectable_drift, sweep  # noqa: E402
from src.eval.unknown_defect import (  # noqa: E402
    lomo, open_set_eval, out_of_family_eval, suggest_tau, _fit_score, _threshold_at_fpr,
)
from src.features.build import build_features  # noqa: E402
from src.fusion.meta_learner import RiskFusion  # noqa: E402
from src.ingest.csv_reader import read_csv  # noqa: E402
from src.knowledge.coverage_audit import audit, render  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402
from src.knowledge.prototypes import MechanismAttributor, NoveltyDetector  # noqa: E402
from src.module_a.autoencoder import TrajectoryAutoencoderTrack  # noqa: E402
from src.module_a.iforest import IsolationTrack  # noqa: E402
from src.module_a.mahalanobis import MahalanobisTrack  # noqa: E402
from src.module_a.physics_residual import PhysicsResidualTrack  # noqa: E402
from src.module_a.robust_z import RobustZTrack  # noqa: E402
from src.simulate.lot_generator import LotGenerator  # noqa: E402
from src.simulate.out_of_family import generate_out_of_family_lots  # noqa: E402

FPR_BUDGET = 0.05


def tracks():
    return [RobustZTrack(), MahalanobisTrack(), IsolationTrack(),
            TrajectoryAutoencoderTrack(), PhysicsResidualTrack()]


def banner(title: str) -> None:
    print("\n" + "=" * 84)
    print(f"  {title}")
    print("=" * 84)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic" / "burnin.csv")
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "features.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "ude.json")
    ap.add_argument("--only", type=str, default=None,
                    choices=["lomo", "oof", "openset", "sweep", "coverage"])
    ap.add_argument("--no-closed-set", action="store_true",
                    help="skip the closed-set comparison; halves UDE-1 runtime")
    ap.add_argument("--lomo-splits", type=int, default=3)
    ap.add_argument("--sweep-lots", type=int, default=8)
    ap.add_argument("--sweep-parts", type=int, default=300)
    args = ap.parse_args()

    lib = MechanismLibrary.load()
    feats = pd.read_parquet(args.features)
    y = feats["is_defect"].astype(int).to_numpy()
    print(f"library  : {lib}")
    print(f"features : {feats.shape[0]:,} parts, {int(y.sum())} defects")

    report = {}
    run = lambda name: args.only is None or args.only == name  # noqa: E731

    # ---------------------------------------------------------------- UDE-1
    if run("lomo"):
        banner("UDE-1  leave-one-mechanism-out")
        print("  train on every mechanism except the held-out one, test on it alone\n")
        table = lomo(feats, tracks(), lib, fpr_budget=FPR_BUDGET,
                     n_splits=args.lomo_splits, closed_set=not args.no_closed_set)
        if not table.empty:
            table = table.sort_values("recall_at_fpr")
            print("\n  " + f"{'held out':<16}{'held-out':>11}{'closed-set':>13}{'gap':>9}  severity")
            for _, r in table.iterrows():
                gap = r["recall_at_fpr"] - r["closed_set_recall"]
                print(f"  {r['held_out']:<16}{r['recall_at_fpr']:>10.1%}"
                      f"{r['closed_set_recall']:>13.1%}{gap:>+9.1%}  {r['severity']}")
            worst = table.iloc[0]
            print(f"\n  WORST-CASE held-out recall : {worst['recall_at_fpr']:.1%} "
                  f"({worst['held_out']}, {worst['mechanism_name']})")
            print(f"  mean held-out recall       : {table['recall_at_fpr'].mean():.1%}")
            print(f"  mean closed-set recall     : {table['closed_set_recall'].mean():.1%}  <- optimistic")
            report["ude1"] = table.to_dict(orient="records")

    # ---------------------------------------------------------------- UDE-2
    oof_feats = None
    if run("oof") or run("openset"):
        banner("UDE-2  out-of-family physics")
        print("  stretched exponential, log-time, sigmoid, telegraph - forms absent")
        print("  from the library and never seen in training\n")
        oof_df = generate_out_of_family_lots(lib, n_lots=10, parts_per_lot=400, prevalence=0.02)
        oof_feats = build_features(oof_df, lib)
        res = out_of_family_eval(feats, oof_feats, tracks(), fpr_budget=FPR_BUDGET)
        print(f"  defects injected            : {res['n_defects']}")
        print(f"  recall @ {FPR_BUDGET:.0%} FPR            : {res['recall_at_fpr']:.1%}")
        print(f"  recall at train threshold   : {res['recall_at_train_threshold']:.1%}")
        print(f"  ROC-AUC                     : {res['roc_auc']:.3f}")
        print("\n  per novel form:")
        for form, r in sorted(res["per_form_recall"].items(), key=lambda kv: kv[1]):
            print(f"    {form:<28} {r:>6.1%}")
        report["ude2"] = {k: v for k, v in res.items() if k != "risk"}

    # ---------------------------------------------------------------- UDE-3
    if run("openset") and oof_feats is not None:
        banner("UDE-3  open-set detection and abstention")
        attributor = MechanismAttributor(lib)
        known_attr = attributor.attribute_frame(feats)
        unknown_attr = attributor.attribute_frame(oof_feats)
        y_oof = oof_feats["is_defect"].astype(int).to_numpy()

        tau = suggest_tau(known_attr, y, target_keep=0.95)
        print(f"  tau chosen from KNOWN defects only (keep 95%) : {tau:.2f}\n")

        print("  (a) prototype-centroid distance - the original design")
        res_proto = open_set_eval(known_attr, y, unknown_attr, y_oof, tau=tau)
        print(f"      AUROC {res_proto['auroc']:.3f}   median known "
              f"{res_proto['median_distance_known']:.2f} vs unknown "
              f"{res_proto['median_distance_unknown']:.2f}")
        report["ude3_prototype"] = res_proto

        print("\n  (b) k-NN to actual known-defect examples")
        nov = NoveltyDetector(lib, k=1).fit(feats, y, keep=0.95)
        kn = pd.DataFrame({"distance": nov.distance(feats, exclude_self=True)})
        un = pd.DataFrame({"distance": nov.distance(oof_feats)})
        res = open_set_eval(kn, y, un, y_oof, tau=nov.threshold_)
        print()
        print(f"  known-vs-unknown AUROC        : {res['auroc']:.3f}")
        print(f"  median distance, known        : {res['median_distance_known']:.2f}")
        print(f"  median distance, unknown      : {res['median_distance_unknown']:.2f}")
        print(f"  abstained on unknown defects  : {res['abstention_rate_on_unknown']:.1%}")
        print(f"  abstained on known defects    : {res['abstention_rate_on_known']:.1%}  (false abstention)")
        print(f"  abstention precision          : {res['abstention_precision']:.1%}")
        report["ude3"] = res

    # ---------------------------------------------------------------- UDE-4
    if run("sweep"):
        banner("UDE-4  subtlety sweep and Minimum Detectable Drift")
        print(f"  {args.sweep_lots} lots x {args.sweep_parts} parts per point\n")

        train_scores_cache = {}

        def generate(subtlety: float) -> pd.DataFrame:
            gen = LotGenerator(lib, seed=int(1000 + subtlety * 997))
            df = gen.generate(n_lots=args.sweep_lots, parts_per_lot=args.sweep_parts,
                              prevalence=0.03, subtlety=subtlety, lot_prefix="SW")
            return build_features(df, lib)

        def score(f: pd.DataFrame):
            tr, te = _fit_score(tracks(), feats, f)
            fusion = RiskFusion().fit(tr, y, lots=feats["lot_id"].astype(str))
            out = {"fusion": fusion.risk(te)}
            for col in ("robust_z", "physics_residual", "autoencoder"):
                if col in te.columns:
                    out[col] = te[col]
            return out

        curve = sweep(generate, score, fpr_budget=FPR_BUDGET)
        mdd = minimum_detectable_drift(curve, target_recall=0.90)
        print()
        if mdd["achieved"]:
            print(f"  MINIMUM DETECTABLE DRIFT (90% recall) : subtlety {mdd['mdd_subtlety']:.2f}, "
                  f"{mdd['mdd_separation_mad']:.2f} MAD above the lot median")
        else:
            print("  MINIMUM DETECTABLE DRIFT (90% recall) : not achieved at any tested amplitude")
        report["ude4"] = {"curve": curve.to_dict(orient="records"), "mdd": mdd}

    # ------------------------------------------------------------- coverage
    if run("coverage"):
        banner("Coverage audit  -  library claims vs measurement")
        tr, _ = _fit_score(tracks(), feats, feats.iloc[:1])
        result = audit(lib, feats, tr, fpr_budget=FPR_BUDGET)
        print(render(result, lib))
        report["coverage"] = {
            "measured": result["measured"].to_dict(),
            "overclaimed": result["overclaimed"].to_dict(orient="records"),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
