"""Phase 2 checkpoint: does the drift predictor beat its baselines, and is its
uncertainty honest?

    python scripts/eval_module_b.py

Reports MAE against four baselines and, separately, MAE on the defective subset
- accuracy where it actually matters - plus conformal coverage and width.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.harness import lot_folds  # noqa: E402
from src.features.trajectory import build_panel  # noqa: E402
from src.ingest.csv_reader import read_csv  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402
from src.module_b.baselines import all_baselines  # noqa: E402
from src.module_b.conformal import ConformalCalibrator, interval_report  # noqa: E402
from src.module_b.dataset import make_dataset  # noqa: E402
from src.module_b.quantile_gbm import QuantileDriftModel  # noqa: E402
from src.module_b.safety_slope import SafetySlope  # noqa: E402


def split_train_calib(ds, train_pool: np.ndarray, calib_frac: float = 0.3, seed: int = 0):
    """Split the training pool by LOT, never by part.

    Conformal calibration assumes exchangeability. Parts within a lot share a
    median and a MAD, so putting some of a lot in train and the rest in
    calibration leaks and quietly breaks the coverage guarantee.
    """
    lots = ds.groups.iloc[train_pool].to_numpy()
    uniq = np.unique(lots)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_calib = max(1, int(round(len(uniq) * calib_frac)))
    calib_lots = set(uniq[:n_calib].tolist())
    is_calib = np.array([l in calib_lots for l in lots])
    return train_pool[~is_calib], train_pool[is_calib]


def run_parameter(df, lib, parameter: str, panel, alpha: float = 0.1, n_splits: int = 5) -> dict:
    ds = make_dataset(df, lib, parameter=parameter, panel=panel)
    folds = lot_folds(ds.groups, n_splits=n_splits)

    names = list(all_baselines().keys()) + ["quantile gbm"]
    abs_err = {k: [] for k in names}
    abs_err_defect = {k: [] for k in names}
    cov_rows, safety_rows = [], []

    for train_pool, test_idx in folds:
        train_idx, calib_idx = split_train_calib(ds, train_pool)
        y_test = ds.y.iloc[test_idx].to_numpy(dtype=float)
        is_def = ds.meta.iloc[test_idx]["is_defect"].to_numpy(dtype=bool) if "is_defect" in ds.meta else np.zeros(len(test_idx), bool)

        for key, model in all_baselines().items():
            model.fit(ds, train_idx)
            pred = model.predict(ds, test_idx)
            abs_err[key].append(np.abs(pred - y_test))
            abs_err_defect[key].append(np.abs(pred - y_test)[is_def])

        qm = QuantileDriftModel(quantiles=(alpha / 2, 0.5, 1 - alpha / 2)).fit(ds, train_idx)
        q_test = qm.predict_quantiles(ds, test_idx)
        lo_col, hi_col = q_test.columns[0], q_test.columns[-1]
        pred = q_test["q50"].to_numpy() if "q50" in q_test else q_test.iloc[:, 1].to_numpy()
        abs_err["quantile gbm"].append(np.abs(pred - y_test))
        abs_err_defect["quantile gbm"].append(np.abs(pred - y_test)[is_def])

        q_cal = qm.predict_quantiles(ds, calib_idx)
        cal = ConformalCalibrator(alpha=alpha).calibrate(
            q_cal[lo_col].to_numpy(), q_cal[hi_col].to_numpy(), ds.y.iloc[calib_idx].to_numpy(dtype=float)
        )

        raw = interval_report(y_test, ConformalCalibrator(alpha=alpha).apply(
            q_test[lo_col].to_numpy(), pred, q_test[hi_col].to_numpy()))
        conf = interval_report(y_test, cal.apply(q_test[lo_col].to_numpy(), pred, q_test[hi_col].to_numpy()))
        cov_rows.append({"raw_coverage": raw["empirical_coverage"], "raw_width": raw["mean_width"],
                         "conformal_coverage": conf["empirical_coverage"], "conformal_width": conf["mean_width"]})

        # Rejection behaviour on the conformal upper bound vs the median.
        interval = cal.apply(q_test[lo_col].to_numpy(), pred, q_test[hi_col].to_numpy())
        v0 = ds.meta.iloc[test_idx]["v0"]
        slope = SafetySlope().fit(
            (ds.y.iloc[train_idx].to_numpy() - ds.meta.iloc[train_idx]["v0"].to_numpy()) / ds.horizon,
            ds.groups.iloc[train_idx],
        )
        verdict = slope.evaluate(interval, v0, ds.groups.iloc[test_idx], ds.derated_limit, ds.horizon,
                                 index=ds.X.iloc[test_idx].index)
        safety_rows.append({
            "reject_on_upper": int(verdict.reject.sum()),
            "reject_on_median": int((pred > ds.derated_limit).sum()),
            "defect_caught_upper": int((verdict.reject.to_numpy() & is_def).sum()),
            "defect_caught_median": int(((pred > ds.derated_limit) & is_def).sum()),
            "n_defect": int(is_def.sum()),
        })

    mae = {k: float(np.mean(np.concatenate(v))) for k, v in abs_err.items()}
    mae_def = {k: float(np.mean(np.concatenate(v))) if np.concatenate(v).size else float("nan")
               for k, v in abs_err_defect.items()}
    return {
        "parameter": parameter,
        "unit": lib.parameter(parameter).unit,
        "mae": mae,
        "mae_defect": mae_def,
        "coverage": pd.DataFrame(cov_rows).mean().to_dict(),
        "safety": pd.DataFrame(safety_rows).sum().to_dict(),
        "n": len(ds.y),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic" / "burnin.csv")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    lib = MechanismLibrary.load()
    df = read_csv(args.data, library=lib)
    panel = build_panel(df)

    for parameter in lib.parameter_names():
        r = run_parameter(df, lib, parameter, panel, alpha=args.alpha, n_splits=args.splits)
        unit = r["unit"]
        print("=" * 78)
        print(f"  MODULE B - {parameter}  (predict Value_168h from Value_0h + Value_24h)")
        print("=" * 78)
        print(f"  {'model':<32} {'MAE':>12} {'MAE on defects':>18}")
        for k in sorted(r["mae"], key=r["mae"].get):
            star = "  <-" if k == "quantile gbm" else ""
            print(f"  {k:<32} {r['mae'][k]:>9.4f} {unit}  {r['mae_defect'][k]:>12.4f} {unit}{star}")

        c = r["coverage"]
        print(f"\n  nominal coverage               : {1 - args.alpha:.0%}")
        print(f"  raw quantile coverage          : {c['raw_coverage']:.1%}  (width {c['raw_width']:.3f} {unit})")
        print(f"  conformal coverage             : {c['conformal_coverage']:.1%}  (width {c['conformal_width']:.3f} {unit})")

        s = r["safety"]
        print(f"\n  rejecting on the MEDIAN        : {int(s['reject_on_median']):>5} parts, "
              f"{int(s['defect_caught_median'])}/{int(s['n_defect'])} defects caught")
        print(f"  rejecting on the UPPER bound   : {int(s['reject_on_upper']):>5} parts, "
              f"{int(s['defect_caught_upper'])}/{int(s['n_defect'])} defects caught")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
