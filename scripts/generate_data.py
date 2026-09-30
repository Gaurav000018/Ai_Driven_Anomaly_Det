"""Phase 0 entry point: build the synthetic burn-in dataset.

    python scripts/generate_data.py --lots 20 --parts 500 --prevalence 0.01

Writes data/synthetic/burnin.csv plus a labels file, and prints the headline
fact the whole project rests on: how many injected latent defects classical
static screening lets through.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.validate import assert_valid  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402
from src.simulate.lot_generator import LotGenerator, part_labels, static_screen  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic burn-in data")
    ap.add_argument("--lots", type=int, default=20)
    ap.add_argument("--parts", type=int, default=500)
    ap.add_argument("--prevalence", type=float, default=0.01)
    ap.add_argument("--subtlety", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic" / "burnin.csv")
    args = ap.parse_args()

    lib = MechanismLibrary.load()
    print(f"library : {lib}")
    print(f"params  : {lib.parameter_names()}")
    print(f"defects : {len(lib.defects())} mechanisms\n")

    gen = LotGenerator(lib, seed=args.seed)
    df = gen.generate(
        n_lots=args.lots,
        parts_per_lot=args.parts,
        prevalence=args.prevalence,
        subtlety=args.subtlety,
    )

    print("validating ...")
    assert_valid(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    labels = part_labels(df)
    labels.to_csv(args.out.with_name(args.out.stem + "_labels.csv"), index=False)

    screen = static_screen(df)
    n_parts = len(screen)
    n_defect = int(screen["is_defect"].sum())
    caught = int((screen["is_defect"] & screen["static_fail"]).sum())
    false_alarm = int((~screen["is_defect"] & screen["static_fail"]).sum())

    print(f"\nwrote {len(df):,} measurements  ->  {args.out}")
    print(f"      {n_parts:,} parts across {df['lot_id'].nunique()} lots")
    print(f"      {n_defect} injected latent defects ({n_defect / n_parts:.2%})\n")

    print("=" * 62)
    print("  STATIC SCREENING BASELINE (datasheet pass/fail limits)")
    print("=" * 62)
    print(f"  injected latent defects      : {n_defect}")
    print(f"  caught by static limits      : {caught}")
    print(f"  ESCAPED into the payload     : {n_defect - caught}")
    print(f"  recall                       : {caught / n_defect:.1%}" if n_defect else "  recall : n/a")
    print(f"  false alarms on good parts   : {false_alarm}")
    print("=" * 62)
    by_mech = screen[screen["is_defect"]].groupby("mechanism_id").size().sort_values(ascending=False)
    print("\ninjected by mechanism:")
    for mech_id, count in by_mech.items():
        print(f"  {mech_id:<16} {count:>4}   {lib.get(mech_id).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
