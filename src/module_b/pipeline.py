"""Module B end to end: fit, calibrate, forecast, and emit a fusion-ready risk.

`oof_risk` produces the Module B signal for every part without any part ever
being scored by a model that saw it. That is what lets the fusion layer treat
the Module B margin as just another track score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..eval.harness import lot_folds
from ..features.trajectory import Panel, build_panel
from ..knowledge.library import MechanismLibrary
from .conformal import ConformalCalibrator, ConformalInterval
from .dataset import DriftDataset, make_dataset
from .quantile_gbm import QuantileDriftModel
from .safety_slope import SafetySlope


def split_train_calib(
    ds: DriftDataset, train_pool: np.ndarray, calib_frac: float = 0.3, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the training pool by lot, never by part - conformal needs it."""
    lots = ds.groups.iloc[train_pool].to_numpy()
    uniq = np.unique(lots)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_calib = max(1, int(round(len(uniq) * calib_frac)))
    calib_lots = set(uniq[:n_calib].tolist())
    is_calib = np.array([l in calib_lots for l in lots])
    return train_pool[~is_calib], train_pool[is_calib]


@dataclass
class DriftForecast:
    parameter: str
    lower: pd.Series
    median: pd.Series
    upper: pd.Series
    margin_limit: pd.Series
    margin_slope: pd.Series
    risk: pd.Series
    derated_limit: float
    unit: str


class DriftPipeline:
    """Fit-once, predict-many wrapper used by the API and the dashboard."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.models_: Dict[str, QuantileDriftModel] = {}
        self.calibrators_: Dict[str, ConformalCalibrator] = {}
        self.slopes_: Dict[str, SafetySlope] = {}
        self.datasets_: Dict[str, DriftDataset] = {}

    def fit(self, df: pd.DataFrame, library: MechanismLibrary, panel: Optional[Panel] = None) -> "DriftPipeline":
        panel = panel if panel is not None else build_panel(df)
        for parameter in library.parameter_names():
            ds = make_dataset(df, library, parameter=parameter, panel=panel)
            idx = np.arange(len(ds.y))
            train_idx, calib_idx = split_train_calib(ds, idx)

            model = QuantileDriftModel(quantiles=(self.alpha / 2, 0.5, 1 - self.alpha / 2)).fit(ds, train_idx)
            q_cal = model.predict_quantiles(ds, calib_idx)
            cal = ConformalCalibrator(alpha=self.alpha).calibrate(
                q_cal.iloc[:, 0].to_numpy(), q_cal.iloc[:, -1].to_numpy(),
                ds.y.iloc[calib_idx].to_numpy(dtype=float),
            )
            slope = SafetySlope().fit(
                (ds.y.iloc[train_idx].to_numpy() - ds.meta.iloc[train_idx]["v0"].to_numpy()) / ds.horizon,
                ds.groups.iloc[train_idx],
            )

            self.datasets_[parameter] = ds
            self.models_[parameter] = model
            self.calibrators_[parameter] = cal
            self.slopes_[parameter] = slope
        return self

    def forecast(self, parameter: str, ds: Optional[DriftDataset] = None, idx: Optional[np.ndarray] = None) -> DriftForecast:
        ds = ds if ds is not None else self.datasets_[parameter]
        idx = idx if idx is not None else np.arange(len(ds.y))
        model, cal, slope = self.models_[parameter], self.calibrators_[parameter], self.slopes_[parameter]

        q = model.predict_quantiles(ds, idx)
        median = q["q50"].to_numpy() if "q50" in q.columns else q.iloc[:, 1].to_numpy()
        interval = cal.apply(q.iloc[:, 0].to_numpy(), median, q.iloc[:, -1].to_numpy())
        index = ds.X.iloc[idx].index

        verdict = slope.evaluate(
            interval, ds.meta.iloc[idx]["v0"], ds.groups.iloc[idx],
            ds.derated_limit, ds.horizon, index=index,
        )
        return DriftForecast(
            parameter=parameter,
            lower=pd.Series(interval.lower, index=index),
            median=pd.Series(interval.median, index=index),
            upper=pd.Series(interval.upper, index=index),
            margin_limit=verdict.margin_limit,
            margin_slope=verdict.margin_slope,
            risk=verdict.risk,
            derated_limit=ds.derated_limit,
            unit="",
        )


def oof_risk(
    df: pd.DataFrame,
    library: MechanismLibrary,
    panel: Optional[Panel] = None,
    alpha: float = 0.1,
    n_splits: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """Out-of-fold Module B risk, one column per parameter plus a max column."""
    panel = panel if panel is not None else build_panel(df)
    per_param: Dict[str, pd.Series] = {}

    for parameter in library.parameter_names():
        ds = make_dataset(df, library, parameter=parameter, panel=panel)
        risk = pd.Series(np.nan, index=ds.X.index, dtype=float)
        margin = pd.Series(np.nan, index=ds.X.index, dtype=float)

        for train_pool, test_idx in lot_folds(ds.groups, n_splits=n_splits):
            train_idx, calib_idx = split_train_calib(ds, train_pool)
            model = QuantileDriftModel(quantiles=(alpha / 2, 0.5, 1 - alpha / 2)).fit(ds, train_idx)
            q_cal = model.predict_quantiles(ds, calib_idx)
            cal = ConformalCalibrator(alpha=alpha).calibrate(
                q_cal.iloc[:, 0].to_numpy(), q_cal.iloc[:, -1].to_numpy(),
                ds.y.iloc[calib_idx].to_numpy(dtype=float),
            )
            slope = SafetySlope().fit(
                (ds.y.iloc[train_idx].to_numpy() - ds.meta.iloc[train_idx]["v0"].to_numpy()) / ds.horizon,
                ds.groups.iloc[train_idx],
            )
            q = model.predict_quantiles(ds, test_idx)
            median = q["q50"].to_numpy() if "q50" in q.columns else q.iloc[:, 1].to_numpy()
            interval = cal.apply(q.iloc[:, 0].to_numpy(), median, q.iloc[:, -1].to_numpy())
            index = ds.X.iloc[test_idx].index
            verdict = slope.evaluate(
                interval, ds.meta.iloc[test_idx]["v0"], ds.groups.iloc[test_idx],
                ds.derated_limit, ds.horizon, index=index,
            )
            risk.loc[index] = verdict.risk.to_numpy()
            margin.loc[index] = verdict.margin_limit.to_numpy()

        per_param[f"B_risk_{parameter}"] = risk
        per_param[f"B_margin_{parameter}"] = margin
        if verbose:
            print(f"  module B out-of-fold: {parameter} done")

    out = pd.DataFrame(per_param)
    risk_cols = [c for c in out.columns if c.startswith("B_risk_")]
    out["B_risk_max"] = out[risk_cols].max(axis=1)
    return out
