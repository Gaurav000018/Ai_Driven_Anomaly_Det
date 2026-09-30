"""Readers that produce the canonical long-format frame.

`read_csv` handles the already-canonical case. `read_wide` handles the shape the
problem statement describes literally - one row per part with Value_0h,
Value_24h, Value_96h, Value_168h columns - which is how lot data usually arrives
from a spreadsheet.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .schema import coerce

# Matches Value_0h, Iddq_24h, v168, 96h, ...
_HOUR_PATTERN = re.compile(r"(?:^|_)(\d+(?:\.\d+)?)\s*h(?:rs?|ours?)?$", re.IGNORECASE)

# Unit aliases seen in real lot sheets.
UNIT_ALIASES = {
    "ua": "uA", "µa": "uA", "μa": "uA", "microamp": "uA", "microamps": "uA",
    "na": "nA", "nanoamp": "nA",
    "ma": "mA",
    "ns": "ns", "nsec": "ns",
    "ps": "ps",
    "v": "V", "mv": "mV",
}

# Multiplicative factors into each parameter's canonical unit.
UNIT_SCALE = {
    ("nA", "uA"): 1e-3,
    ("mA", "uA"): 1e3,
    ("uA", "nA"): 1e3,
    ("ps", "ns"): 1e-3,
    ("mV", "V"): 1e-3,
}


def normalise_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None or (isinstance(unit, float) and pd.isna(unit)):
        return None
    key = str(unit).strip().lower()
    return UNIT_ALIASES.get(key, str(unit).strip())


def harmonise_units(df: pd.DataFrame, target: Dict[str, str]) -> pd.DataFrame:
    """Convert every measurement into its parameter's canonical unit.

    `target` maps param_name -> canonical unit, normally taken from the
    Defect Mechanism Library so there is one source of truth for units.
    """
    out = df.copy()
    out["unit"] = out["unit"].map(normalise_unit)
    for param, want in target.items():
        mask = out["param_name"] == param
        if not mask.any():
            continue
        for have in out.loc[mask, "unit"].dropna().unique():
            if have == want:
                continue
            factor = UNIT_SCALE.get((have, want))
            if factor is None:
                raise ValueError(f"no conversion from '{have}' to '{want}' for parameter '{param}'")
            sel = mask & (out["unit"] == have)
            out.loc[sel, "value"] = out.loc[sel, "value"] * factor
            for col in ("limit_lo", "limit_hi"):
                if col in out.columns:
                    out.loc[sel, col] = out.loc[sel, col] * factor
            out.loc[sel, "unit"] = want
    return out


def read_csv(path: Path | str) -> pd.DataFrame:
    """Read an already-canonical long-format CSV."""
    return coerce(pd.read_csv(path))


def read_wide(
    path: Path | str,
    param_name: str,
    unit: str,
    id_col: str = "part_id",
    lot_col: str = "lot_id",
    limit_hi: Optional[float] = None,
) -> pd.DataFrame:
    """Read a wide lot sheet (one row per part, one column per timepoint).

    Any column whose name ends in an hour token is treated as a measurement:
    `Value_0h`, `Value_24h`, `Iddq_96h`, `168h` all work.
    """
    raw = pd.read_csv(path)
    hour_cols = {c: float(m.group(1)) for c in raw.columns if (m := _HOUR_PATTERN.search(str(c)))}
    if not hour_cols:
        raise ValueError(
            f"no timepoint columns found in {path}. Expected names ending in an hour "
            f"token such as 'Value_0h' or '168h'; got {list(raw.columns)}"
        )

    id_vars = [c for c in raw.columns if c not in hour_cols]
    long = raw.melt(id_vars=id_vars, value_vars=list(hour_cols), var_name="_col", value_name="value")
    long["hours"] = long["_col"].map(hour_cols)
    long = long.drop(columns="_col")

    long = long.rename(columns={id_col: "part_id", lot_col: "lot_id"})
    if "lot_id" not in long.columns:
        long["lot_id"] = "LOT-UNKNOWN"
    long["param_name"] = param_name
    long["unit"] = unit
    if limit_hi is not None and "limit_hi" not in long.columns:
        long["limit_hi"] = limit_hi

    return coerce(long.dropna(subset=["value"]))
