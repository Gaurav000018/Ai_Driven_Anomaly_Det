"""The canonical long-format schema every reader must produce.

One row = one measurement of one parameter on one part at one burn-in hour.
Long format (rather than a wide 0h/24h/96h/168h table) is deliberate: real ATE
logs have ragged timepoints, and the physics kernel fits arbitrary t anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

# Required columns, in canonical order.
REQUIRED = [
    "part_id",
    "lot_id",
    "param_name",
    "unit",
    "hours",
    "value",
]

# Optional but used when present.
OPTIONAL = [
    "wafer_id",
    "x",
    "y",
    "limit_lo",
    "limit_hi",
    "temp_C",
]

# Columns present only in synthetic data (ground truth for evaluation).
LABEL_COLUMNS = [
    "is_defect",
    "mechanism_id",
    "severity",
]

DTYPES = {
    "part_id": "string",
    "lot_id": "string",
    "wafer_id": "string",
    "param_name": "string",
    "unit": "string",
    "hours": "float64",
    "value": "float64",
    "x": "Int64",
    "y": "Int64",
    "limit_lo": "float64",
    "limit_hi": "float64",
    "temp_C": "float64",
}


@dataclass
class ValidationIssue:
    level: str          # "error" | "warning"
    code: str
    message: str
    n_rows: int = 0

    def __str__(self) -> str:
        suffix = f" ({self.n_rows} rows)" if self.n_rows else ""
        return f"[{self.level.upper()}] {self.code}: {self.message}{suffix}"


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical dtypes to whichever known columns are present."""
    out = df.copy()
    for col, dtype in DTYPES.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    ordered = [c for c in REQUIRED + OPTIONAL + LABEL_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest]


def missing_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED if c not in df.columns]
