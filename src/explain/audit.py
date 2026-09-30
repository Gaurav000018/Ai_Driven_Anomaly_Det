"""Immutable audit trail.

This is what makes the system usable under ECSS / MIL-STD-883 quality regimes.
A rejection has to be replayable: months later, someone must be able to take the
same part, the same model and the same library, and reproduce the number
exactly. Hashing all three and stamping them onto every certificate is how that
promise is kept - and it is cheap.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_frame(df: pd.DataFrame) -> str:
    """Stable hash of a frame's contents, independent of row order."""
    idx = np.argsort(df.index.astype(str).to_numpy())
    ordered = df.iloc[idx]
    return hash_bytes(pd.util.hash_pandas_object(ordered, index=True).to_numpy().tobytes())


@dataclass
class AuditRecord:
    part_id: str
    decision: str
    escape_risk: float
    timestamp: str
    model_hash: str
    library_hash: str
    library_version: int
    data_hash: str
    threshold: float
    code_version: str = ""
    environment: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def short(self) -> str:
        return (
            f"model sha256:{self.model_hash[:12]} | "
            f"library v{self.library_version} sha256:{self.library_hash[:12]} | "
            f"data sha256:{self.data_hash[:12]} | {self.timestamp}"
        )


class AuditTrail:
    def __init__(
        self,
        model_hash: str,
        library_hash: str,
        library_version: int,
        data_hash: str,
        threshold: float,
        code_version: str = "",
    ) -> None:
        self.model_hash = model_hash
        self.library_hash = library_hash
        self.library_version = library_version
        self.data_hash = data_hash
        self.threshold = threshold
        self.code_version = code_version
        self.environment = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }

    def record(self, part_id: str, decision: str, escape_risk: float) -> AuditRecord:
        return AuditRecord(
            part_id=str(part_id),
            decision=decision,
            escape_risk=float(escape_risk),
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            model_hash=self.model_hash,
            library_hash=self.library_hash,
            library_version=self.library_version,
            data_hash=self.data_hash,
            threshold=float(self.threshold),
            code_version=self.code_version,
            environment=dict(self.environment),
        )

    @staticmethod
    def git_version(root: Optional[Path] = None) -> str:
        """Best-effort commit id, so a certificate names the code that produced it."""
        import subprocess

        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(root or Path(__file__).resolve().parents[2]),
                capture_output=True,
                text=True,
                timeout=5,
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""
