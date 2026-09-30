"""Phase 4 checkpoint: produce QA certificates.

    python scripts/explain_parts.py --n 3
    python scripts/explain_parts.py --part LOT007-W3-D0412

Writes every rejected/reviewed part's certificate to reports/certificates.jsonl
and prints a few in full.
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

from src.explain.audit import AuditTrail, hash_file, hash_frame  # noqa: E402
from src.explain.certificate import CertificateBuilder  # noqa: E402
from src.explain.counterfactual import CounterfactualExplainer  # noqa: E402
from src.fusion.cost_decision import CostDecision, CostModel  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402
from src.knowledge.prototypes import UNKNOWN, MechanismAttributor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "features.parquet")
    ap.add_argument("--scores", type=Path, default=ROOT / "data" / "processed" / "oof_scores.parquet")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "fusion.pkl")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "certificates.jsonl")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--part", type=str, default=None)
    ap.add_argument("--tau", type=float, default=3.0)
    args = ap.parse_args()

    lib = MechanismLibrary.load()
    feats = pd.read_parquet(args.features)
    scores = pd.read_parquet(args.scores)
    with args.model.open("rb") as fh:
        bundle = pickle.load(fh)
    fusion = bundle["fusion"]

    y = feats["is_defect"].astype(int).to_numpy()
    sev = np.array([lib.cost_multiplier(m) if d else 1.0
                    for m, d in zip(feats["mechanism_id"].astype(str), y)])
    sev = np.where(sev > 0, sev, 1.0)

    risk = fusion.risk(scores)
    decider = CostDecision(CostModel.load()).fit(risk, y, severity_weight=sev)

    attributor = MechanismAttributor(lib, tau=args.tau)
    attributions = attributor.attribute_frame(feats)
    decisions = decider.decide(risk, unknown_mechanism=attributions["is_unknown"])

    accepted = decisions["decision"] == "ACCEPT"
    units = {p.name: p.unit for p in lib.parameters()}
    cf = CounterfactualExplainer().fit(feats, accepted, units=units)

    audit = AuditTrail(
        model_hash=hash_file(args.model),
        library_hash=lib.source_hash,
        library_version=lib.version,
        data_hash=hash_frame(feats[[c for c in feats.columns if feats[c].dtype.kind in "fi"]]),
        threshold=decider.accept_below_,
        code_version=AuditTrail.git_version(ROOT),
    )

    builder = CertificateBuilder(lib, fusion, attributor, cf, audit)
    builder.attributions_ = attributions

    flagged = decisions.index[decisions["decision"] != "ACCEPT"]
    print(f"parts flagged : {len(flagged):,} of {len(decisions):,}")
    print(f"  REJECT      : {int((decisions['decision'] == 'REJECT').sum()):,}")
    print(f"  REVIEW      : {int((decisions['decision'] == 'REVIEW').sum()):,}")
    print(f"  floored to REVIEW by UNKNOWN-MECHANISM : {int(decisions['floored_unknown'].sum()):,}")
    print(f"  attributed UNKNOWN overall            : {int(attributions['is_unknown'].sum()):,}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for pid in flagged:
            cert = builder.build(feats, scores, decisions, pid)
            fh.write(json.dumps(cert.to_dict(), default=str) + "\n")
    print(f"wrote {len(flagged):,} certificates -> {args.out.relative_to(ROOT)}\n")

    if args.part:
        show = [args.part]
    else:
        # Prefer genuine defects that were caught, so the sample is informative.
        caught = [p for p in flagged if bool(feats.loc[p, "is_defect"])]
        show = caught[: args.n] if caught else list(flagged[: args.n])

    for pid in show:
        print("\n" + "-" * 78)
        print(builder.build(feats, scores, decisions, pid).render())
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
