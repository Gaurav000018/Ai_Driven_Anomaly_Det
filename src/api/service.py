"""The scoring service: one path from raw measurements to decisions.

Shared by the REST API and the dashboard so both cannot drift apart. Everything
stateful - the library, the fitted tracks, the fusion stack, the cost model -
is loaded once and held.

One property matters more than the rest: a lot is scored *as a lot*. Lot-
relative statistics are the whole basis of Module A, so a single part cannot be
scored in isolation. The service rejects that rather than silently borrowing
another lot's reference population, which would be an invisible and serious
error.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..eval.harness import lot_folds  # noqa: F401  (re-exported for callers)
from ..explain.audit import AuditTrail, hash_file, hash_frame
from ..explain.certificate import CertificateBuilder
from ..explain.counterfactual import CounterfactualExplainer
from ..features.build import build_features
from ..features.trajectory import build_panel
from ..fusion.cost_decision import CostDecision, CostModel
from ..fusion.meta_learner import RiskFusion
from ..ingest.validate import assert_valid
from ..knowledge.library import MechanismLibrary
from ..knowledge.prototypes import MechanismAttributor
from ..module_a.autoencoder import TrajectoryAutoencoderTrack
from ..module_a.iforest import IsolationTrack
from ..module_a.mahalanobis import MahalanobisTrack
from ..module_a.physics_residual import PhysicsResidualTrack
from ..module_a.robust_z import RobustZTrack

ROOT = Path(__file__).resolve().parents[2]

# Below this a lot's median and MAD are not a usable reference population.
MIN_LOT_FOR_SCORING = 30


def default_tracks():
    return [RobustZTrack(), MahalanobisTrack(), IsolationTrack(),
            TrajectoryAutoencoderTrack(), PhysicsResidualTrack()]


@dataclass
class LotScore:
    decisions: pd.DataFrame        # escape_risk, decision, floored_unknown
    attributions: pd.DataFrame
    features: pd.DataFrame
    scores: pd.DataFrame

    def summary(self) -> Dict:
        counts = self.decisions["decision"].value_counts().to_dict()
        return {
            "n_parts": int(len(self.decisions)),
            "accept": int(counts.get("ACCEPT", 0)),
            "review": int(counts.get("REVIEW", 0)),
            "reject": int(counts.get("REJECT", 0)),
            "unknown_mechanism": int(self.attributions["is_unknown"].sum()),
            "mean_escape_risk": float(self.decisions["escape_risk"].mean()),
        }


class ScoringService:
    def __init__(
        self,
        model_path: Path = ROOT / "models" / "fusion.pkl",
        reference_features: Optional[Path] = ROOT / "data" / "processed" / "features.parquet",
        tau: float = 3.0,
    ) -> None:
        self.library = MechanismLibrary.load()
        self.costs = CostModel.load()
        self.model_path = Path(model_path)
        self.tau = tau

        with self.model_path.open("rb") as fh:
            bundle = pickle.load(fh)
        self.fusion: RiskFusion = bundle["fusion"]
        self.operating = bundle.get("operating", {})

        if bundle.get("library_hash") and bundle["library_hash"] != self.library.source_hash:
            # Refusing here is the point of hashing the library in the first
            # place: a model fitted against different physics is not this model.
            raise RuntimeError(
                "mechanism library has changed since this model was fitted "
                f"(model {bundle['library_hash'][:12]}, current {self.library.source_hash[:12]}). "
                "Retrain before scoring."
            )

        self.attributor = MechanismAttributor(self.library, tau=tau)
        self.tracks = default_tracks()
        self.reference_: Optional[pd.DataFrame] = None
        self.counterfactual_: Optional[CounterfactualExplainer] = None
        self.decider_ = CostDecision(self.costs)

        if reference_features and Path(reference_features).exists():
            self.fit_reference(pd.read_parquet(reference_features))

    # ------------------------------------------------------------- reference

    def fit_reference(self, features: pd.DataFrame) -> "ScoringService":
        """Fit the tracks and the accepted envelope on a reference population."""
        self.reference_ = features
        for track in self.tracks:
            track.fit(features)

        scores = pd.DataFrame({t.name: t.score(features) for t in self.tracks})
        if "B_risk_max" in self.fusion.columns_:
            scores["B_risk_max"] = 0.0
        scores = scores[self.fusion.columns_]

        risk = self.fusion.risk(scores)
        if "is_defect" in features.columns:
            y = features["is_defect"].astype(int).to_numpy()
            self.decider_.fit(risk, y)
            decisions = self.decider_.decide(risk)
            accepted = decisions["decision"] == "ACCEPT"
        else:
            self.decider_.accept_below_ = float(np.quantile(risk, 0.80))
            self.decider_.reject_above_ = float(np.quantile(risk, 0.98))
            accepted = risk < self.decider_.accept_below_

        units = {p.name: p.unit for p in self.library.parameters()}
        self.counterfactual_ = CounterfactualExplainer().fit(features, accepted, units=units)
        return self

    # ---------------------------------------------------------------- score

    def score_lot(self, df: pd.DataFrame, validate: bool = True) -> LotScore:
        if self.reference_ is None:
            raise RuntimeError("no reference population fitted; call fit_reference first")
        if validate:
            assert_valid(df, verbose=False)

        sizes = df.groupby("lot_id", observed=True)["part_id"].nunique()
        too_small = sizes[sizes < MIN_LOT_FOR_SCORING]
        if len(too_small):
            raise ValueError(
                f"lot(s) {list(too_small.index)} have fewer than {MIN_LOT_FOR_SCORING} parts. "
                "Module A is lot-relative: a median and MAD from a handful of parts is noise, "
                "and borrowing another lot's reference would be silently wrong."
            )

        features = build_features(df, self.library, panel=build_panel(df))
        scores = pd.DataFrame({t.name: t.score(features) for t in self.tracks})
        if "B_risk_max" in self.fusion.columns_ and "B_risk_max" not in scores:
            scores["B_risk_max"] = 0.0
        scores = scores[self.fusion.columns_]

        risk = self.fusion.risk(scores)
        attributions = self.attributor.attribute_frame(features)
        decisions = self.decider_.decide(risk, unknown_mechanism=attributions["is_unknown"])
        return LotScore(decisions=decisions, attributions=attributions, features=features, scores=scores)

    # -------------------------------------------------------------- explain

    def certificate(self, lot: LotScore, part_id: str) -> Dict:
        audit = AuditTrail(
            model_hash=hash_file(self.model_path),
            library_hash=self.library.source_hash,
            library_version=self.library.version,
            data_hash=hash_frame(lot.features.select_dtypes(include=[np.number])),
            threshold=float(self.decider_.accept_below_ or 0.0),
            code_version=AuditTrail.git_version(ROOT),
        )
        builder = CertificateBuilder(
            self.library, self.fusion, self.attributor, self.counterfactual_, audit
        )
        builder.attributions_ = lot.attributions
        cert = builder.build(lot.features, lot.scores, lot.decisions, part_id)
        return {"text": cert.render(), **cert.to_dict()}
