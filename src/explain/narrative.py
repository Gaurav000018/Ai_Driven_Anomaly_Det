"""Assemble the QA certificate.

The target reader is a quality engineer, not a data scientist. So the certificate
leads with the physics and the decision, puts the attribution underneath, and
ends with an audit line - rather than opening with a bar chart of SHAP values.

Every number in the text is traceable to a computed feature; nothing here is
generated prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..knowledge.prototypes import UNKNOWN, Attribution as MechAttribution
from .audit import AuditRecord
from .counterfactual import Condition


@dataclass
class Certificate:
    part_id: str
    lot_id: str
    decision: str
    escape_risk: float
    mechanism: MechAttribution
    drivers: str
    conditions: List[Condition]
    physics: Dict[str, float]
    forecast: Optional[Dict[str, float]]
    audit: AuditRecord
    peers: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ text

    def render(self) -> str:
        # ASCII only. These certificates are read in terminals, pasted into
        # ticketing systems and printed into travellers, and an em dash that
        # renders as a replacement character undermines the whole point of
        # writing an audit document.
        lines: List[str] = []
        head = f"Part {self.part_id} - {self.decision} - Escape Risk {self.escape_risk:.0f}/100"
        lines.append(head)
        lines.append("=" * len(head))
        lines.append(f"Lot {self.lot_id}")
        lines.append("")

        lines.append(self._physics_paragraph())
        if self.forecast:
            lines.append("")
            lines.append(self._forecast_paragraph())

        if self.peers:
            lines.append("")
            lines.append(
                f"{len(self.peers)} lot-mate(s) share this signature "
                f"({', '.join(self.peers[:3])}{'...' if len(self.peers) > 3 else ''}), "
                "suggesting a localised process excursion rather than an isolated part."
            )

        lines.append("")
        lines.append(f"Decision drivers : {self.drivers}")
        lines.append(f"Counterfactual   : {self._counterfactual_text()}")
        lines.append(f"Audit            : {self.audit.short()}")
        return "\n".join(lines)

    def _physics_paragraph(self) -> str:
        p = self.physics
        # The parameter the evidence is in, which is not always the one whose
        # fingerprint matched. Labelling a t_pd reading "Iddq" is worse than
        # saying nothing.
        param = p.get("parameter") or self.mechanism.parameter or "the driving parameter"

        if self.mechanism.is_unknown:
            verdict = (
                "Its degradation signature matches no mechanism in the library "
                f"(nearest prototype {self.mechanism.distance:.1f} fingerprint-widths away). "
                "Flagged UNKNOWN-MECHANISM and routed for human review: an unrecognised "
                "degradation physics is the last thing that should be accepted on a low score."
            )
        else:
            confidence = "consistent with" if self.mechanism.confident else "most closely matching"
            verdict = (
                f"The signature is {confidence} **{self.mechanism.mechanism_name}** "
                f"({self.mechanism.mechanism_id}, {self.mechanism.distance:.1f} fingerprint-widths, "
                f"severity {self.mechanism.severity}; {self.mechanism.reference})."
            )

        measured = ""
        if "value_at" in p and "hours_at" in p:
            z = float(p.get("robust_z", float("nan")))
            # "-0.3 MAD above the median" is not a sentence. Say what is true.
            if np.isfinite(z):
                direction = "above" if z >= 0 else "below"
                sits = f"sitting {abs(z):.1f} MAD {direction} its lot median"
            else:
                sits = "with no usable lot reference"
            unit = f" {p['unit']}" if p.get("unit") else ""
            measured = (
                f"{param} measured {p['value_at']:.3g}{unit} at {p['hours_at']:.0f} h, "
                f"{sits} of {p.get('lot_median', float('nan')):.3g}{unit}, "
                "though inside the datasheet maximum. "
            )

        kinetics = ""
        if np.isfinite(p.get("n", np.nan)):
            ident = float(p.get("identifiable", float("nan")))
            caveat = ""
            if np.isfinite(ident) and ident < 0.3:
                # Say so rather than quoting a number fitted from noise.
                caveat = (
                    " That exponent is only weakly identifiable here "
                    f"(drift-to-noise weight {ident:.2f}), so it is reported for context, not as evidence."
                )
            kinetics = (
                f"Its fitted degradation exponent is n = {p['n']:.2f} against a lot median of "
                f"{p.get('lot_n', float('nan')):.2f} (R2 = {p.get('r2', float('nan')):.2f}).{caveat} "
            )

        return measured + kinetics + verdict

    def _forecast_paragraph(self) -> str:
        f = self.forecast or {}
        breach = "exceeds" if f.get("upper", 0) > f.get("derated_limit", np.inf) else "stays within"
        return (
            f"Conformal forecast at {f.get('horizon', 168):.0f} h: "
            f"median {f.get('median', float('nan')):.3g}, "
            f"upper bound {f.get('upper', float('nan')):.3g} "
            f"at {f.get('coverage', 0.9):.0%} coverage - the upper bound {breach} "
            f"the derated limit of {f.get('derated_limit', float('nan')):.3g}."
        )

    def _counterfactual_text(self) -> str:
        if not self.conditions:
            return "inside the accepted envelope on every actionable measure"
        return " and ".join(c.sentence() for c in self.conditions)

    def to_dict(self) -> Dict:
        return {
            "part_id": self.part_id,
            "lot_id": self.lot_id,
            "decision": self.decision,
            "escape_risk": self.escape_risk,
            "mechanism_id": self.mechanism.mechanism_id,
            "mechanism_name": self.mechanism.mechanism_name,
            "mechanism_distance": self.mechanism.distance,
            "is_unknown_mechanism": self.mechanism.is_unknown,
            "severity": self.mechanism.severity,
            "reference": self.mechanism.reference,
            "drivers": self.drivers,
            "counterfactual": self._counterfactual_text(),
            "physics": self.physics,
            "forecast": self.forecast,
            "peers": self.peers,
            "audit": self.audit.to_dict(),
        }


def find_peers(
    features: pd.DataFrame,
    attributions: pd.DataFrame,
    part_id,
    max_peers: int = 5,
    eligible: Optional[pd.Index] = None,
) -> List[str]:
    """Lot-mates attributed to the same mechanism.

    A cluster of identical signatures in one lot is a process excursion, and
    saying so changes what the fab does next - it is a different action from
    scrapping one part.
    """
    if part_id not in attributions.index:
        return []
    mech = attributions.loc[part_id, "mechanism_id"]
    # A nominal or unknown attribution says nothing about a shared root cause.
    if mech in (UNKNOWN, "") or bool(attributions.loc[part_id].get("matched_nominal", False)):
        return []

    lot = str(features.loc[part_id, "lot_id"])
    same_lot = features.index[features["lot_id"].astype(str) == lot]
    if eligible is not None:
        same_lot = same_lot.intersection(eligible)
    candidates = attributions.reindex(same_lot)
    peers = candidates.index[(candidates["mechanism_id"] == mech) & (candidates.index != part_id)]
    return [str(p) for p in peers[:max_peers]]
