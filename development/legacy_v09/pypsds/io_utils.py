from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .ds.shp import ShpResult
from .ds.covariance import CovarianceResult
from .ds.phase_linking import PhaseLinkResult


def save_shp(path: Path, result: ShpResult, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packed_bits=result.packed_bits,
        counts=result.counts,
        offsets=result.offsets,
        method=np.asarray(result.method),
        alpha=np.asarray(result.alpha),
        half_window=np.asarray(result.half_window, dtype=np.int16),
        window_level=(result.window_level if result.window_level is not None else np.asarray([], dtype=np.uint8)),
        window_candidates=(result.window_candidates if result.window_candidates is not None else np.asarray([result.half_window], dtype=np.int16)),
        target_samples=np.asarray(result.target_samples, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata or {})),
    )


def load_shp(path: Path) -> tuple[ShpResult, dict]:
    z = np.load(path, allow_pickle=False)
    r = ShpResult(
        packed_bits=z["packed_bits"],
        counts=z["counts"],
        offsets=z["offsets"],
        method=str(z["method"].item()),
        alpha=float(z["alpha"].item()),
        half_window=tuple(int(x) for x in z["half_window"].tolist()),
        window_level=(z["window_level"] if "window_level" in z.files and z["window_level"].size else None),
        window_candidates=(z["window_candidates"] if "window_candidates" in z.files else None),
        target_samples=(int(z["target_samples"].item()) if "target_samples" in z.files else 0),
    )
    meta = json.loads(str(z["metadata"].item()))
    return r, meta


def save_covariance(path: Path, result: CovarianceResult, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        centers=result.centers,
        shp_counts=result.shp_counts,
        covariance=result.covariance,
        coherence=result.coherence,
        box_coherence=result.box_coherence,
        metadata=np.asarray(json.dumps(metadata or {})),
    )


def save_phase_link(path: Path, result: PhaseLinkResult, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        phase_rad=result.phase_rad,
        cpx_phase=result.cpx_phase,
        temporal_coherence=result.temporal_coherence,
        eigenvalue=result.eigenvalue,
        estimator=np.asarray(result.estimator),
        estimator_code=result.estimator_code,
        metadata=np.asarray(json.dumps(metadata or {})),
    )
