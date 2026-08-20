from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(
    value: Any,
) -> str:

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def signature(
    value: Any,
) -> str:

    return hashlib.sha256(
        canonical_json(
            value
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class StageManifest:
    stage: str
    algorithm: str
    signature: str
    status: str
    payload: dict[str, Any]

    def write(
        self,
        path: str | Path,
    ) -> None:

        p = Path(path)

        p.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        obj = {
            "format":
                "pyPSDS-GAMMA-stage-manifest-v1",
            "stage":
                self.stage,
            "algorithm":
                self.algorithm,
            "signature":
                self.signature,
            "status":
                self.status,
            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
            "payload":
                self.payload,
        }

        p.write_text(
            json.dumps(
                obj,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


def build_stage_signature(
    *,
    algorithm: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
) -> str:

    return signature({
        "algorithm":
            algorithm,
        "parameters":
            parameters,
        "inputs":
            inputs,
    })
