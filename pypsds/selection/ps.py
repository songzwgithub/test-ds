from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class PsCandidateResult:
    mask: np.ndarray
    amplitude_mean: np.ndarray
    amplitude_dispersion: np.ndarray
    valid_fraction: np.ndarray
    threshold: float


def select_ps_candidates(
    stack: np.ndarray,
    *,
    amplitude_dispersion_threshold: float = 0.25,
    min_valid_fraction: float = 0.9,
) -> PsCandidateResult:
    """Initial PS candidate selection from amplitude dispersion.

    This is a candidate stage, not final StaMPS PS probability estimation.
    """
    if stack.ndim != 3 or not np.iscomplexobj(stack):
        raise ValueError("stack must be complex with shape (dates, rows, cols)")
    amp = np.abs(stack).astype(np.float32, copy=False)
    finite = np.isfinite(amp)
    valid_fraction = finite.mean(axis=0).astype(np.float32)
    mean = np.nanmean(amp, axis=0, dtype=np.float64).astype(np.float32)
    std = np.nanstd(amp, axis=0, dtype=np.float64).astype(np.float32)
    eps = np.finfo(np.float32).tiny
    disp = (std / np.maximum(mean, eps)).astype(np.float32)
    mask = (
        np.isfinite(disp)
        & (valid_fraction >= float(min_valid_fraction))
        & (disp <= float(amplitude_dispersion_threshold))
        & (mean > 0)
    )
    return PsCandidateResult(mask, mean, disp, valid_fraction, float(amplitude_dispersion_threshold))
