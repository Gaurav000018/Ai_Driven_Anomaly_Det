"""Wire the explainability pieces into one certificate per part."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..knowledge.library import MechanismLibrary
from ..knowledge.prototypes import MechanismAttributor
from .audit import AuditTrail
from .counterfactual import CounterfactualExplainer
from .narrative import Certificate, find_peers
from .shap_explainer import FusionExplainer


class CertificateBuilder:
    def __init__(
        self,
        library: MechanismLibrary,
        fusion,
        attributor: MechanismAttributor,
        counterfactual: CounterfactualExplainer,
        audit: AuditTrail,
    ) -> None:
        self.library = library
        self.fusion = fusion
        self.attributor = attributor
        self.counterfactual = counterfactual
        self.audit = audit
        self.explainer = FusionExplainer(fusion)
        self.attributions_: Optional[pd.DataFrame] = None

    def prepare(self, features: pd.DataFrame) -> "CertificateBuilder":
        self.attributions_ = self.attributor.attribute_frame(features)
        return self

    def driving_parameter(self, features: pd.DataFrame, part_id) -> str:
        """The parameter carrying the actual evidence against this part.

        The mechanism's own `parameter` is where the signature *matched*, which
        is not always where the part looks worst - reporting Iddq while the
        evidence sits in t_pd produces a certificate that contradicts its own
        counterfactual. Pick the parameter with the strongest lot-relative
        departure on either level or drift.
        """
        row = features.loc[part_id]
        best, best_score = self.library.parameter_names()[0], -np.inf
        for p in self.library.parameter_names():
            z_cols = [c for c in features.columns if c.startswith(f"{p}__v_") and c.endswith("h_z")]
            candidates = [float(row[c]) for c in z_cols if np.isfinite(float(row[c]))]
            drift_col = f"{p}__total_drift_frac_z"
            if drift_col in features.columns and np.isfinite(float(row[drift_col])):
                candidates.append(float(row[drift_col]))
            score = max(candidates) if candidates else -np.inf
            if score > best_score:
                best, best_score = p, score
        return best

    def _physics(self, features: pd.DataFrame, part_id, parameter: str) -> Dict[str, float]:
        row = features.loc[part_id]
        if not parameter or f"{parameter}__n" not in features.columns:
            return {}

        physics: Dict[str, float] = {
            "parameter": parameter,
            "unit": self.library.parameter(parameter).unit,
            "n": float(row[f"{parameter}__n"]),
            "lot_n": float(row[f"{parameter}__lot_n_median"]),
            "r2": float(row[f"{parameter}__r2"]),
            "identifiable": float(row.get(f"{parameter}__identifiable", np.nan)),
        }

        # Report the timepoint where the part looked worst against its lot.
        z_cols = [c for c in features.columns if c.startswith(f"{parameter}__v_") and c.endswith("h_z")]
        if z_cols:
            worst = max(z_cols, key=lambda c: float(row[c]))
            hours = int(worst.split("__v_")[1].split("h")[0])
            value_col = f"{parameter}__v_{hours}h"
            in_lot = features["lot_id"].astype(str) == str(row["lot_id"])
            physics.update(
                {
                    "hours_at": float(hours),
                    "value_at": float(row[value_col]),
                    "robust_z": float(row[worst]),
                    "lot_median": float(features.loc[in_lot, value_col].median()),
                }
            )
        return physics

    def build(
        self,
        features: pd.DataFrame,
        scores: pd.DataFrame,
        decisions: pd.DataFrame,
        part_id,
        forecasts: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Certificate:
        if self.attributions_ is None:
            self.prepare(features)

        mech = self.attributor.attribute(features, part_id)
        row = decisions.loc[part_id]
        parameter = self.driving_parameter(features, part_id)

        forecast = None
        if forecasts and parameter in forecasts:
            forecast = dict(forecasts[parameter])
            forecast.setdefault("derated_limit", self.library.parameter(parameter).derated_limit)

        return Certificate(
            part_id=str(part_id),
            lot_id=str(features.loc[part_id, "lot_id"]),
            decision=str(row["decision"]),
            escape_risk=float(row["escape_risk"]),
            mechanism=mech,
            drivers=self.explainer.drivers_sentence(scores, part_id),
            conditions=self.counterfactual.explain(features, part_id, top_k=2),
            physics=self._physics(features, part_id, parameter),
            forecast=forecast,
            audit=self.audit.record(part_id, str(row["decision"]), float(row["escape_risk"])),
            # Only flagged lot-mates count as peers. A cluster of accepted parts
            # sharing an attribution is not a process excursion.
            peers=find_peers(
                features,
                self.attributions_,
                part_id,
                eligible=decisions.index[decisions["decision"] != "ACCEPT"],
            ),
        )
