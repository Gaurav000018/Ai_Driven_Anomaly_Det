"""Phase 1 checkpoint: do the lot-relative and kinetic tracks beat static limits?

    python scripts/eval_module_a.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.harness import by_mechanism, compare, evaluate  # noqa: E402
from src.features.build import build_features  # noqa: E402
from src.ingest.csv_reader import read_csv  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402
from src.module_a.physics_residual import PhysicsResidualTrack  # noqa: E402
from src.module_a.robust_z import RobustZTrack  # noqa: E402
from src.simulate.lot_generator import static_screen  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic" / "burnin.csv")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "processed" / "features.parquet")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    lib = MechanismLibrary.load()
    df = read_csv(args.data)

    if args.cache.exists() and not args.rebuild:
        feats = pd.read_parquet(args.cache)
        print(f"features : loaded from cache {args.cache.name}")
    else:
        print("features : building ...")
        feats = build_features(df, lib)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            feats.to_parquet(args.cache)
        except Exception as exc:  # pyarrow may be absent; the cache is optional
            print(f"           (cache skipped: {exc})")
    print(f"           {feats.shape[0]:,} parts x {feats.shape[1]} columns\n")

    y = feats["is_defect"].astype(int).to_numpy()
    mech = feats["mechanism_id"].astype(str).to_numpy()
    sev_mult = np.array([lib.cost_multiplier(m if d else None) for m, d in zip(mech, y)])
    sev_mult = np.where(sev_mult > 0, sev_mult, 1.0)

    results = []

    screen = static_screen(df).set_index("part_id").reindex(feats.index)
    results.append(
        evaluate(y, screen["static_fail"].astype(float).to_numpy(), name="static limits",
                 severity_weight=sev_mult)
    )

    a1 = RobustZTrack().fit(feats)
    s1 = a1.score(feats)
    results.append(evaluate(y, s1, name="A1 robust-Z / MAD", severity_weight=sev_mult))

    a5 = PhysicsResidualTrack().fit(feats)
    s5 = a5.score(feats)
    results.append(evaluate(y, s5, name="A5 physics residual", severity_weight=sev_mult))

    # Simple rank-average of the two, to show the fusion thesis already holds.
    fused = s1.rank(pct=True) + s5.rank(pct=True)
    results.append(evaluate(y, fused, name="A1 + A5 (rank mean)", severity_weight=sev_mult))

    print("=" * 96)
    print("  MODULE A - PHASE 1")
    print("=" * 96)
    table = compare(results)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    best = results[-1]
    print(f"per-mechanism recall for '{best.name}' at its cost-optimal threshold:")
    per_mech = by_mechanism(y, fused, mech, best.cost_threshold)
    for _, r in per_mech.iterrows():
        bar = "#" * int(round(r["recall"] * 24))
        print(f"  {r['mechanism_id']:<16} {r['caught']:>3}/{r['n']:<3} {r['recall']:>6.1%}  {bar}")
    print(f"\n  WORST-CASE mechanism recall : {per_mech['recall'].min():.1%}  <- the number that matters")
    print(f"  mean mechanism recall       : {per_mech['recall'].mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
