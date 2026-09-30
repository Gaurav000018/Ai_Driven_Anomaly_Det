"""Measure the third scored metric: is the explanation actually true?

    python scripts/eval_explainability.py

Faithfulness by deletion, stability by lot-level bootstrap, and a literal check
that every counterfactual pass condition is achievable.
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

from src.eval.explain_metrics import counterfactual_validity, faithfulness, stability  # noqa: E402
from src.explain.counterfactual import CounterfactualExplainer  # noqa: E402
from src.fusion.cost_decision import CostDecision, CostModel  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "features.parquet")
    ap.add_argument("--scores", type=Path, default=ROOT / "data" / "processed" / "oof_scores.parquet")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "fusion.pkl")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "explainability.json")
    ap.add_argument("--bootstrap", type=int, default=12)
    ap.add_argument("--sample", type=int, default=150)
    args = ap.parse_args()

    lib = MechanismLibrary.load()
    feats = pd.read_parquet(args.features)
    scores = pd.read_parquet(args.scores)
    with args.model.open("rb") as fh:
        fusion = pickle.load(fh)["fusion"]

    y = feats["is_defect"].astype(int).to_numpy()
    lots = feats["lot_id"].astype(str)

    risk = fusion.risk(scores)
    decider = CostDecision(CostModel.load()).fit(risk, y)
    decisions = decider.decide(risk)
    flagged = decisions.index[decisions["decision"] != "ACCEPT"]
    accepted = decisions["decision"] == "ACCEPT"

    rng = np.random.default_rng(0)
    sample = pd.Index(rng.choice(flagged, size=min(args.sample, len(flagged)), replace=False))

    print("=" * 76)
    print("  EXPLAINABILITY - is the explanation actually true?")
    print("=" * 76)
    print(f"  flagged parts {len(flagged):,}, sampled {len(sample):,}\n")

    print("  FAITHFULNESS (deletion test)")
    print("    delete the named drivers and the score should collapse;")
    print("    delete unnamed tracks and it should not\n")
    fa = faithfulness(fusion, scores, subset=sample, top_k=2)
    print(f"    risk drop, top-{fa.top_k} drivers deleted   : {fa.drop_top_k:>7.1%}")
    print(f"    risk drop, {fa.top_k} random tracks deleted   : {fa.drop_random_k:>7.1%}")
    print(f"    risk drop, bottom-{fa.top_k} deleted         : {fa.drop_bottom_k:>7.1%}")
    print(f"    advantage over random                : {fa.advantage:>+7.1%}")
    print(f"    verdict                              : {'PASS' if fa.passes else 'FAIL'}\n")

    print("  STABILITY (bootstrap by lot)")
    print("    the same part explained twice should give the same reasons\n")
    st = stability(scores, y, lots, subset=sample[:60], n_bootstrap=args.bootstrap)
    print(f"    mean rank correlation                : {st.mean_rank_correlation:>7.3f}")
    print(f"    minimum                              : {st.min_rank_correlation:>7.3f}")
    print(f"    fraction above {st.threshold:.2f}                  : {st.frac_above_threshold:>7.1%}")
    print(f"    verdict (target >= {st.threshold:.2f})            : {'PASS' if st.passes else 'FAIL'}\n")

    print("  COUNTERFACTUAL VALIDITY")
    print("    every stated pass condition must be achievable by a real part\n")
    units = {p.name: p.unit for p in lib.parameters()}
    cf = CounterfactualExplainer().fit(feats, accepted, units=units)
    cv = counterfactual_validity(feats, cf, sample[:100], accepted)
    print(f"    conditions checked                   : {cv['conditions_checked']:,}")
    print(f"    satisfiable by an accepted lot-mate  : {cv['satisfiable_fraction']:>7.1%}")
    print(f"    parts with no violated condition     : {cv['parts_with_no_conditions']}")

    report = {"faithfulness": fa.as_row(), "stability": st.as_row(), "counterfactual": cv}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
