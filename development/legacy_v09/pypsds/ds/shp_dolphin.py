from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True, slots=True)
class GlrtConfig:
    alpha: float = 0.005
    half_row: int = 5
    half_col: int = 11


def rayleigh_scale_squared(mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Dolphin GLRT Rayleigh scale^2 = (variance + mean^2) / 2."""
    return (np.asarray(var, np.float64) + np.asarray(mean, np.float64) ** 2) / 2.0


def glrt_threshold(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return float(chi2.ppf(1.0 - alpha, df=1))


def glrt_statistic(scale1, scale2, *, nslc: int):
    """Exact Dolphin two-Rayleigh GLRT statistic.

    test = N * [2 log((s1+s2)/2) - log(s1) - log(s2)]
    A neighbor is accepted as SHP when test < chi2.ppf(1-alpha, 1).
    """
    s1 = np.asarray(scale1, dtype=np.float64)
    s2 = np.asarray(scale2, dtype=np.float64)
    pooled = 0.5 * (s1 + s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(nslc) * (2.0 * np.log(pooled) - np.log(s1) - np.log(s2))


def compute_amplitude_statistics(slc: np.ndarray, *, strict_zero_invalid: bool = True):
    """Return strict-valid mask, amplitude mean/variance and scale^2.

    `slc` shape is [Ndate,H,W]. For GAMMA SCOMPLEX, a complex 0+0j sample is
    treated as invalid by default. A center is strict-valid only when all dates
    are finite and non-zero.
    """
    slc = np.asarray(slc)
    if slc.ndim != 3:
        raise ValueError("slc must have shape [Ndate,H,W]")
    finite = np.isfinite(slc.real) & np.isfinite(slc.imag)
    if strict_zero_invalid:
        finite &= ~((slc.real == 0) & (slc.imag == 0))
    valid = np.all(finite, axis=0)
    amp = np.abs(slc).astype(np.float32)
    mean = np.full(valid.shape, np.nan, np.float32)
    var = np.full(valid.shape, np.nan, np.float32)
    if np.any(valid):
        x = amp[:, valid].astype(np.float64)
        mean[valid] = np.mean(x, axis=0)
        var[valid] = np.var(x, axis=0, ddof=0)
    scale2 = rayleigh_scale_squared(mean, var)
    return valid, mean, var, scale2


__all__ = [
    "GlrtConfig",
    "compute_amplitude_statistics",
    "glrt_statistic",
    "glrt_threshold",
    "rayleigh_scale_squared",
]
