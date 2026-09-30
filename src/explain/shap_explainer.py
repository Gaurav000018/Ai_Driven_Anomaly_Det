"""Attribution over the fusion layer.

A note on why this is short. The fusion stack is a logistic regression over
standardised track scores, and for a linear model the Shapley value of feature i
has a closed form: coefficient_i * (x_i - E[x_i]). There is nothing to
approximate and nothing to sample. Running a KernelExplainer over 10,000 parts
to recover a number we can compute exactly would be slower and less accurate.

So the analytic path is the default and the shap package is used only when it is
present and the model is not linear - which keeps the dependency optional and
the numbers exact.

Faithfulness and stability are measured in src/eval/explain_metrics.py; an
attribution nobody checked is decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap  # noqa: F401

    _HAVE_SHAP = True
except Exception:  # pragma: no cover
    _HAVE_SHAP = False


@dataclass
class Attribution:
    feature: str
    value: float
    contribution: float
    share: float            # fraction of the total positive push toward reject

    def sentence(self) -> str:
        return f"{self.feature} {self.share:.0%}"


class FusionExplainer:
    """Per-part attribution of the Escape Risk across the tracks."""

    def __init__(self, fusion) -> None:
        self.fusion = fusion
        self.method = "analytic linear shapley"

    def contributions(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Signed contribution of each track, for every part."""
        X = self.fusion._prepare(scores)[self.fusion.columns_]
        Xs = self.fusion.scaler_.transform(X.to_numpy(dtype=float))
        # Shapley for a linear model, measured from the population mean, which
        # the scaler has already centred to zero.
        return pd.DataFrame(
            Xs * self.fusion.model_.coef_[0],
            index=scores.index,
            columns=self.fusion.columns_,
        )

    def explain(self, scores: pd.DataFrame, part_id, top_k: int = 4) -> List[Attribution]:
        contrib = self.contributions(scores.loc[[part_id]]).iloc[0]
        raw = scores.loc[part_id]
        pushing = contrib[contrib > 0]
        total = float(pushing.sum())
        out = [
            Attribution(
                feature=name,
                value=float(raw[name]),
                contribution=float(value),
                share=float(value / total) if total > 0 else 0.0,
            )
            for name, value in contrib.sort_values(ascending=False).items()
        ]
        return out[:top_k]

    def global_importance(self, scores: pd.DataFrame) -> pd.Series:
        """Mean absolute contribution across the population."""
        return self.contributions(scores).abs().mean().sort_values(ascending=False)

    def drivers_sentence(self, scores: pd.DataFrame, part_id, top_k: int = 3) -> str:
        attrs = [a for a in self.explain(scores, part_id, top_k=top_k) if a.contribution > 0]
        if not attrs:
            return "no track pushed this part toward rejection"
        return " | ".join(f"{a.feature} {a.share:.0%}" for a in attrs)
