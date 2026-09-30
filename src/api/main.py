"""SENTINEL-BI REST API.

    uvicorn src.api.main:app --reload

Endpoints mirror how a screening floor actually works: score a whole lot, pull
a certificate for any part a QA engineer questions, and forecast drift for
parts still in the oven.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402

from src.api.schemas import (  # noqa: E402
    CertificateResponse, DriftForecastItem, DriftRequest, DriftResponse,
    HealthResponse, PartDecision, ScoreRequest, ScoreResponse,
)
from src.api.service import LotScore, ScoringService  # noqa: E402
from src.explain.audit import hash_file  # noqa: E402
from src.ingest.schema import coerce  # noqa: E402
from src.module_b.conformal import ConformalCalibrator  # noqa: E402
from src.module_b.dataset import make_dataset  # noqa: E402
from src.module_b.pipeline import split_train_calib  # noqa: E402
from src.module_b.quantile_gbm import QuantileDriftModel  # noqa: E402

app = FastAPI(
    title="SENTINEL-BI",
    description="AI-driven anomaly detection in component burn-in and screening",
    version="1.0.0",
)

_service: ScoringService | None = None
# Lots stay in memory between scoring and certificate requests. A certificate
# has to be built against the same lot statistics that produced the decision,
# so re-deriving it from a different population would silently change the
# number being explained.
_lots: Dict[str, LotScore] = {}


def service() -> ScoringService:
    global _service
    if _service is None:
        try:
            _service = ScoringService()
        except FileNotFoundError as exc:
            raise HTTPException(503, f"model not available; run scripts/train_fusion.py first ({exc})")
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
    return _service


def _frame(measurements) -> pd.DataFrame:
    return coerce(pd.DataFrame([m.model_dump() for m in measurements]))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = service()
    return HealthResponse(
        status="ok",
        library_version=svc.library.version,
        library_hash=svc.library.source_hash,
        model_hash=hash_file(svc.model_path),
        tracks=[t.name for t in svc.tracks],
        reference_parts=int(len(svc.reference_)) if svc.reference_ is not None else 0,
    )


@app.post("/score/lot", response_model=ScoreResponse)
def score_lot(req: ScoreRequest) -> ScoreResponse:
    svc = service()
    try:
        lot = svc.score_lot(_frame(req.measurements))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    for lot_id in lot.features["lot_id"].astype(str).unique():
        _lots[lot_id] = lot

    attr = lot.attributions
    decisions = [
        PartDecision(
            part_id=str(pid),
            escape_risk=float(row["escape_risk"]),
            decision=str(row["decision"]),
            mechanism_id=str(attr.loc[pid, "mechanism_id"]),
            mechanism_name=str(attr.loc[pid, "mechanism_name"]),
            mechanism_distance=float(attr.loc[pid, "distance"]),
            is_unknown_mechanism=bool(attr.loc[pid, "is_unknown"]),
            severity=str(attr.loc[pid, "severity"]),
            floored_unknown=bool(row.get("floored_unknown", False)),
        )
        for pid, row in lot.decisions.iterrows()
    ]

    return ScoreResponse(
        summary=lot.summary(),
        decisions=decisions,
        accept_below=float(svc.decider_.accept_below_ or 0.0),
        reject_above=float(svc.decider_.reject_above_ or 100.0),
        library_version=svc.library.version,
        library_hash=svc.library.source_hash,
    )


@app.get("/explain/{part_id}", response_model=CertificateResponse)
def explain(part_id: str) -> CertificateResponse:
    svc = service()
    for lot in _lots.values():
        if part_id in lot.decisions.index:
            cert = svc.certificate(lot, part_id)
            return CertificateResponse(
                part_id=part_id,
                text=cert["text"],
                decision=cert["decision"],
                escape_risk=cert["escape_risk"],
                mechanism_id=cert["mechanism_id"],
                mechanism_name=cert["mechanism_name"],
                counterfactual=cert["counterfactual"],
                drivers=cert["drivers"],
                audit=cert["audit"],
                physics=cert["physics"],
                peers=cert["peers"],
            )
    raise HTTPException(404, f"part {part_id} not found; score its lot first via POST /score/lot")


@app.post("/predict/drift", response_model=DriftResponse)
def predict_drift(req: DriftRequest) -> DriftResponse:
    svc = service()
    df = _frame(req.measurements)
    try:
        spec = svc.library.parameter(req.parameter)
    except KeyError:
        raise HTTPException(400, f"unknown parameter '{req.parameter}'; "
                                 f"known: {svc.library.parameter_names()}")

    try:
        ds = make_dataset(df, svc.library, parameter=req.parameter, horizon=req.horizon)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    idx = np.arange(len(ds.y))
    train_idx, calib_idx = split_train_calib(ds, idx)
    model = QuantileDriftModel(quantiles=(req.alpha / 2, 0.5, 1 - req.alpha / 2)).fit(ds, train_idx)
    q_cal = model.predict_quantiles(ds, calib_idx)
    cal = ConformalCalibrator(alpha=req.alpha).calibrate(
        q_cal.iloc[:, 0].to_numpy(), q_cal.iloc[:, -1].to_numpy(),
        ds.y.iloc[calib_idx].to_numpy(dtype=float),
    )
    q = model.predict_quantiles(ds, idx)
    median = q["q50"].to_numpy() if "q50" in q.columns else q.iloc[:, 1].to_numpy()
    interval = cal.apply(q.iloc[:, 0].to_numpy(), median, q.iloc[:, -1].to_numpy())

    items: List[DriftForecastItem] = []
    for i, pid in enumerate(ds.X.index):
        items.append(
            DriftForecastItem(
                part_id=str(pid),
                lower=float(interval.lower[i]),
                median=float(interval.median[i]),
                upper=float(interval.upper[i]),
                derated_limit=spec.derated_limit,
                breaches_on_upper=bool(interval.upper[i] > spec.derated_limit),
                breaches_on_median=bool(interval.median[i] > spec.derated_limit),
            )
        )

    return DriftResponse(
        parameter=req.parameter,
        horizon=req.horizon,
        coverage=1.0 - req.alpha,
        unit=spec.unit,
        forecasts=items,
        n_breaching_upper=sum(i.breaches_on_upper for i in items),
        n_breaching_median=sum(i.breaches_on_median for i in items),
    )


@app.get("/mechanisms")
def mechanisms() -> Dict:
    """The Defect Mechanism Library, as served to any client that asks."""
    svc = service()
    return {
        "version": svc.library.version,
        "hash": svc.library.source_hash,
        "parameters": [
            {"name": p.name, "unit": p.unit, "limit_hi": p.limit_hi, "derated_limit": p.derated_limit}
            for p in svc.library.parameters()
        ],
        "mechanisms": [
            {"id": m.id, "name": m.name, "family": m.family, "severity": m.severity,
             "affects": m.affects, "reference": m.reference, "detectability": m.detectability}
            for m in svc.library.defects()
        ],
    }
