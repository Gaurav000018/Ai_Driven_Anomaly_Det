"""Data contract checks. Run before anything is fitted.

Catches the failure modes that silently corrupt lot statistics: duplicate
measurements, ragged timepoints, unit drift within a parameter, and lots too
small for a MAD estimate to mean anything.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .schema import ValidationIssue, missing_columns

# Below this many parts a lot's median/MAD is not a usable reference population.
MIN_LOT_SIZE = 30


def validate(df: pd.DataFrame, min_lot_size: int = MIN_LOT_SIZE) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    missing = missing_columns(df)
    if missing:
        issues.append(ValidationIssue("error", "missing_columns", f"required columns absent: {missing}"))
        return issues  # nothing else is meaningful without these

    n_null = int(df[["part_id", "lot_id", "param_name", "hours", "value"]].isna().any(axis=1).sum())
    if n_null:
        issues.append(ValidationIssue("error", "null_key_fields", "null in a key field or value", n_null))

    dup = df.duplicated(subset=["part_id", "param_name", "hours"], keep=False)
    if dup.any():
        issues.append(
            ValidationIssue("error", "duplicate_measurement", "same part/param/hour measured more than once", int(dup.sum()))
        )

    if (df["hours"] < 0).any():
        issues.append(ValidationIssue("error", "negative_hours", "burn-in hours must be >= 0", int((df["hours"] < 0).sum())))

    # A parameter must carry exactly one unit everywhere. Mixed uA/nA silently
    # destroys every lot statistic downstream.
    for param, grp in df.groupby("param_name", observed=True):
        units = set(grp["unit"].dropna().unique())
        if len(units) > 1:
            issues.append(
                ValidationIssue("error", "mixed_units", f"parameter '{param}' carries multiple units: {sorted(units)}")
            )

    # Ragged timepoints: warn, do not fail. The physics kernel handles arbitrary t.
    grids = df.groupby(["part_id", "param_name"], observed=True)["hours"].apply(lambda s: tuple(sorted(s.unique())))
    if grids.nunique() > 1:
        common = grids.value_counts()
        issues.append(
            ValidationIssue(
                "warning",
                "ragged_timepoints",
                f"{grids.nunique()} distinct time grids; most common is {list(common.index[0])}",
                int(len(grids) - common.iloc[0]),
            )
        )

    # Parts with fewer than 3 points cannot support a power-law fit (2 free params).
    counts = df.groupby(["part_id", "param_name"], observed=True)["hours"].size()
    thin = counts[counts < 3]
    if len(thin):
        issues.append(
            ValidationIssue("warning", "insufficient_timepoints", "part/param with <3 timepoints cannot be fitted", int(len(thin)))
        )

    lot_sizes = df.groupby("lot_id", observed=True)["part_id"].nunique()
    small = lot_sizes[lot_sizes < min_lot_size]
    if len(small):
        issues.append(
            ValidationIssue(
                "warning",
                "small_lot",
                f"{len(small)} lot(s) below {min_lot_size} parts; robust statistics will be shrunk toward the population prior",
                int(small.sum()),
            )
        )

    if "limit_hi" in df.columns:
        over = df["value"] > df["limit_hi"]
        if over.any():
            issues.append(
                ValidationIssue(
                    "warning",
                    "static_limit_exceeded",
                    "measurements above the datasheet maximum (these are ordinary failures, not latent defects)",
                    int(over.sum()),
                )
            )

    return issues


def assert_valid(df: pd.DataFrame, min_lot_size: int = MIN_LOT_SIZE, verbose: bool = True) -> pd.DataFrame:
    """Raise on any error-level issue; print warnings and continue."""
    issues = validate(df, min_lot_size=min_lot_size)
    errors = [i for i in issues if i.level == "error"]
    if verbose:
        for issue in issues:
            print(f"  {issue}")
    if errors:
        raise ValueError("data contract violated:\n" + "\n".join(str(e) for e in errors))
    return df
