"""Does training across defect amplitudes fix the fusion's subtle-regime gap?

UDE-4 found the fusion matching A4 at full amplitude (99% vs 99%) but trailing
it once defects get subtle (51% vs 60% at subtlety 0.15). The diagnosis was that
its weights were fitted on full-amplitude defects only, where A1's level signal
dominates, and so do not transfer to a regime where the kinetic signal matters
more.

This tests that diagnosis by A/B rather than assuming it:

  fusion_full   trained on full-amplitude defects only
  fusion_mixed  trained on the same number of parts, spread across amplitudes

Both are then evaluated on the same freshly generated sweep. If the diagnosis
is right, fusion_mixed should close the gap at low subtlety; if it is wrong,
this says so.

    python scripts/train_fusion_mixed.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.harness import recall_at_fpr  # noqa: E402
from src.eval.unknown_defect import _fit_score  # noqa: E402
from src.features.build import build_features  # noqa: E402
from src.fusion.meta_learner import RiskFusion, run_tracks_oof  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402
from src.module_a.autoencoder import TrajectoryAutoencoderTrack  # noqa: E402
from src.module_a.iforest import IsolationTrack  # noqa: E402
from src.module_a.mahalanobis import MahalanobisTrack  # noqa: E402
from src.module_a.physics_residual import PhysicsResidualTrack  # noqa: E402
from src.module_a.robust_z import RobustZTrack  # noqa: E402
from src.simulate.lot_generator import LotGenerator  # noqa: E402

TRAIN_SUBTLETIES = (1.0, 0.5, 0.25, 0.15)
EVAL_SUBTLETIES = (1.0, 0.5, 0.25, 0.15, 0.10)
FPR_BUDGET = 0.05


def tracks():
    return [RobustZTrack(), MahalanobisTrack(), IsolationTrack(),
            TrajectoryAutoencoderTrack(), PhysicsResidualTrack()]


def make_features(lib, lots, parts, subtlety, seed, prefix):
    df = LotGenerator(lib, seed=seed).generate(
        n_lots=lots, parts_per_lot=parts, prevalence=0.03,
        subtlety=subtlety, lot_prefix=prefix,
    )
    return build_features(df, lib)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lots-per-level", type=int, default=5)
    ap.add_argument("--parts", type=int, default=300)
    ap.add_argument("--eval-lots", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "fusion_mixed.json")
    args = ap.parse_args()

    lib = MechanismLibrary.load()
    n_levels = len(TRAIN_SUBTLETIES)

    print("=" * 84)
    print("  FUSION TRAINING REGIME - full amplitude only vs mixed amplitudes")
    print("=" * 84)

    # Mixed: lots-per-level at each subtlety.
    print(f"\n  building mixed training set ({n_levels} levels x {args.lots_per_level} lots) ...", flush=True)
    mixed = pd.concat(
        [make_features(lib, args.lots_per_level, args.parts, s, 500 + i, f"MX{i}")
         for i, s in enumerate(TRAIN_SUBTLETIES)]
    )

    # Full: the same TOTAL number of lots, all at full amplitude. Equal budget
    # is the only way the comparison means anything.
    print(f"  building full-amplitude training set ({n_levels * args.lots_per_level} lots) ...", flush=True)
    full = make_features(lib, n_levels * args.lots_per_level, args.parts, 1.0, 900, "FL")

    fusions = {}
    for name, feats in (("full", full), ("mixed", mixed)):
        y = feats["is_defect"].astype(int).to_numpy()
        lots = feats["lot_id"].astype(str)
        print(f"\n  fitting fusion_{name} on {len(feats):,} parts, {int(y.sum())} defects ...", flush=True)
        scores = run_tracks_oof(feats, tracks(), lots, n_splits=3).fillna(0.0)
        fusions[name] = (RiskFusion().fit(scores, y, lots=lots), feats)

    print("\n  evaluating both on freshly generated sweep data ...\n", flush=True)
    rows = []
    for s in EVAL_SUBTLETIES:
        test = make_features(lib, args.eval_lots, args.parts, s, 7000 + int(s * 100), "EV")
        y_test = test["is_defect"].astype(int).to_numpy()

        row = {"subtlety": s, "n_defects": int(y_test.sum())}
        for name, (fusion, train_feats) in fusions.items():
            tr, te = _fit_score(tracks(), train_feats, test)
            # Refit the stack on this training population's track scores so the
            # only thing differing between the arms is the training regime.
            f = RiskFusion().fit(tr, train_feats["is_defect"].astype(int).to_numpy(),
                                 lots=train_feats["lot_id"].astype(str))
            row[name] = recall_at_fpr(y_test, f.risk(te), FPR_BUDGET)
        row["a4_autoencoder"] = recall_at_fpr(y_test, te["autoencoder"], FPR_BUDGET)
        rows.append(row)
        print(f"    subtlety {s:<5.2f}  full={row['full']:.0%}  mixed={row['mixed']:.0%}  "
              f"A4={row['a4_autoencoder']:.0%}   delta={row['mixed'] - row['full']:+.0%}")

    table = pd.DataFrame(rows)
    subtle = table[table["subtlety"] <= 0.25]
    gain = float((subtle["mixed"] - subtle["full"]).mean())
    beats_a4 = int((subtle["mixed"] >= subtle["a4_autoencoder"]).sum())

    print("\n" + "-" * 84)
    print(f"  mean recall gain at subtlety <= 0.25 : {gain:+.1%}")
    print(f"  mixed fusion matches or beats A4     : {beats_a4} of {len(subtle)} subtle levels")
    if gain > 0.02:
        print("\n  VERDICT: the diagnosis holds. Training across amplitudes closes the gap,")
        print("           so the production fusion should be fitted this way.")
    else:
        print("\n  VERDICT: the diagnosis does NOT hold. Mixed-amplitude training does not")
        print("           recover the gap, so the cause lies elsewhere - most likely in the")
        print("           track scores themselves rather than the stack's weights.")
    print("-" * 84)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"table": rows, "mean_gain_subtle": gain, "mixed_beats_a4": beats_a4,
         "train_subtleties": list(TRAIN_SUBTLETIES)}, indent=2), encoding="utf-8")

    best = "mixed" if gain > 0.02 else "full"
    out_model = ROOT / "models" / "fusion_mixed.pkl"
    with out_model.open("wb") as fh:
        pickle.dump({"fusion": fusions[best][0], "regime": best,
                     "library_hash": lib.source_hash}, fh)
    print(f"\nwrote {args.out.relative_to(ROOT)} and {out_model.relative_to(ROOT)} (regime: {best})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
