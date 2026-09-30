"""Request and response models for the REST API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Measurement(BaseModel):
    part_id: str
    lot_id: str
    param_name: str
    unit: str
    hours: float
    value: float
    wafer_id: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    limit_hi: Optional[float] = None
    temp_C: Optional[float] = None


class ScoreRequest(BaseModel):
    measurements: List[Measurement] = Field(
        ...,
        description=(
            "Every measurement for the whole lot. Module A is lot-relative, so a "
            "partial lot cannot be scored - the median and MAD would be computed "
            "from an unrepresentative sample."
        ),
    )


class PartDecision(BaseModel):
    part_id: str
    escape_risk: float
    decision: str
    mechanism_id: str
    mechanism_name: str
    mechanism_distance: float
    is_unknown_mechanism: bool
    severity: str
    floored_unknown: bool = Field(
        False, description="Accepted on score but floored into REVIEW by an unrecognised mechanism"
    )


class ScoreResponse(BaseModel):
    summary: Dict
    decisions: List[PartDecision]
    accept_below: float
    reject_above: float
    library_version: int
    library_hash: str


class CertificateResponse(BaseModel):
    part_id: str
    text: str
    decision: str
    escape_risk: float
    mechanism_id: str
    mechanism_name: str
    counterfactual: str
    drivers: str
    audit: Dict
    physics: Dict
    peers: List[str] = []


class DriftRequest(BaseModel):
    measurements: List[Measurement]
    parameter: str
    horizon: float = 168.0
    alpha: float = 0.1


class DriftForecastItem(BaseModel):
    part_id: str
    lower: float
    median: float
    upper: float
    derated_limit: float
    breaches_on_upper: bool
    breaches_on_median: bool


class DriftResponse(BaseModel):
    parameter: str
    horizon: float
    coverage: float
    unit: str
    forecasts: List[DriftForecastItem]
    n_breaching_upper: int
    n_breaching_median: int


class HealthResponse(BaseModel):
    status: str
    library_version: int
    library_hash: str
    model_hash: str
    tracks: List[str]
    reference_parts: int
