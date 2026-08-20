from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import chi2, norm, ks_2samp


@dataclass(frozen=True, slots=True)
class ShpResult:
    packed_bits: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    method: str
    alpha: float
    half_window: tuple[int, int]
    window_level: np.ndarray | None = None
    window_candidates: np.ndarray | None = None
    target_samples: int = 0


def window_offsets(half_window: tuple[int, int]) -> np.ndarray:
    hy, hx = map(int, half_window)
    return np.asarray(
        [(dy, dx) for dy in range(-hy, hy + 1) for dx in range(-hx, hx + 1)],
        dtype=np.int16,
    )


def _shift(arr: np.ndarray, dy: int, dx: int, fill: float = np.nan) -> np.ndarray:
    out = np.full(arr.shape, fill, dtype=arr.dtype)
    r_src0 = max(0, -dy); r_src1 = min(arr.shape[-2], arr.shape[-2] - dy)
    c_src0 = max(0, -dx); c_src1 = min(arr.shape[-1], arr.shape[-1] - dx)
    r_dst0 = r_src0 + dy; r_dst1 = r_src1 + dy
    c_dst0 = c_src0 + dx; c_dst1 = c_src1 + dx
    if r_src1 > r_src0 and c_src1 > c_src0:
        if arr.ndim == 2:
            out[r_dst0:r_dst1, c_dst0:c_dst1] = arr[r_src0:r_src1, c_src0:c_src1]
        elif arr.ndim == 3:
            out[:, r_dst0:r_dst1, c_dst0:c_dst1] = arr[:, r_src0:r_src1, c_src0:c_src1]
        else:
            raise ValueError("_shift supports 2D/3D arrays")
    return out


def _glrt_accept(mean_intensity: np.ndarray, neighbor: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Rayleigh-scale GLRT, equivalent to the Dolphin amplitude-statistic form."""
    eps = np.finfo(np.float32).tiny
    a = np.maximum(mean_intensity.astype(np.float64), eps)
    b = np.maximum(neighbor.astype(np.float64), eps)
    pooled = 0.5 * (a + b)
    stat = n * (2.0 * np.log(pooled) - np.log(a) - np.log(b))
    threshold = float(chi2.ppf(1.0 - alpha, df=1))
    return np.isfinite(neighbor) & np.isfinite(mean_intensity) & (stat <= threshold)


def _fashps_ci_accept(mean_amp: np.ndarray, neighbor: np.ndarray, n: int, alpha: float) -> np.ndarray:
    z = float(norm.ppf(1.0 - alpha / 2.0))
    rayleigh_cv = math.sqrt((4.0 - math.pi) / math.pi)
    a = mean_amp.astype(np.float64); b = neighbor.astype(np.float64)
    se_a = rayleigh_cv * np.maximum(a, 0.0) / math.sqrt(max(n, 1))
    se_b = rayleigh_cv * np.maximum(b, 0.0) / math.sqrt(max(n, 1))
    lo_a, hi_a = a - z * se_a, a + z * se_a
    lo_b, hi_b = b - z * se_b, b + z * se_b
    mutual = (b >= lo_a) & (b <= hi_a) & (a >= lo_b) & (a <= hi_b)
    return np.isfinite(a) & np.isfinite(b) & mutual


def _ks_accept(amp: np.ndarray, dy: int, dx: int, alpha: float, legacy_reverse: bool) -> np.ndarray:
    """Two-sample KS option for validation/benchmarking.

    `ks` uses standard hypothesis-test semantics: p >= alpha means we do not
    reject equal distributions. `ks_reference` deliberately reproduces the
    p < alpha convention seen in the supplied Moraine-derived reference code;
    it is retained only for controlled comparison, not as the v0.5 default.
    """
    neighbor = _shift(amp, dy, dx)
    try:
        res = ks_2samp(
            amp,
            neighbor,
            axis=0,
            method="asymp",
            nan_policy="omit",
        )
        p = np.asarray(res.pvalue)
    except TypeError:
        # Compatibility fallback for older SciPy: intentionally slow and only
        # used when the optional KS mode is requested.
        rows, cols = amp.shape[1:]
        p = np.full((rows, cols), np.nan, dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                a = amp[:, r, c]; b = neighbor[:, r, c]
                a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
                if len(a) >= 3 and len(b) >= 3:
                    p[r, c] = ks_2samp(a, b, method="asymp").pvalue
    valid = np.isfinite(p)
    return valid & ((p < alpha) if legacy_reverse else (p >= alpha))


def _normalize_candidates(half_window, window_candidates):
    if window_candidates is None:
        return np.asarray([half_window], dtype=np.int16)
    c = np.asarray(window_candidates, dtype=np.int16).reshape(-1, 2)
    if np.any(c <= 0):
        raise ValueError("window candidates must have positive half-window sizes")
    c = np.unique(c, axis=0)
    order = np.argsort((2*c[:, 0] + 1) * (2*c[:, 1] + 1))
    return c[order]


def select_shp(
    stack: np.ndarray,
    *,
    half_window: tuple[int, int] = (5, 11),
    method: str = "glrt",
    alpha: float = 0.001,
    window_candidates=None,
    min_samples_absolute: int = 20,
    min_samples_per_date: float = 0.0,
    offsets: np.ndarray | None = None,
    progress=None,
    progress_every: int | None = None,
) -> ShpResult:
    """Identify statistically homogeneous neighbours in a fixed search support.

    v0.5 production uses a physical elliptical offset set supplied by
    `ShpWindowSpec.offsets()`. Legacy nested rectangles remain available for
    research scripts.
    """
    if stack.ndim != 3 or not np.iscomplexobj(stack):
        raise ValueError("stack must be complex with shape (dates, rows, cols)")
    n_dates, rows, cols = stack.shape
    if n_dates < 3:
        raise ValueError("At least 3 dates are required for SHP statistics")
    method = method.lower()
    if method not in {"glrt", "fashps", "ks", "ks_reference"}:
        raise ValueError("method must be glrt, fashps, ks, or ks_reference")

    custom_offsets = offsets is not None
    if custom_offsets:
        offsets = np.asarray(offsets, dtype=np.int16).reshape(-1, 2)
        if offsets.size == 0:
            raise ValueError("offsets is empty")
        max_hy = int(np.max(np.abs(offsets[:, 0])))
        max_hx = int(np.max(np.abs(offsets[:, 1])))
        candidates = np.asarray([[max_hy, max_hx]], dtype=np.int16)
    else:
        candidates = _normalize_candidates(half_window, window_candidates)
        max_hy = int(candidates[:, 0].max()); max_hx = int(candidates[:, 1].max())
        offsets = window_offsets((max_hy, max_hx))

    n_offsets = len(offsets); nbytes = (n_offsets + 7) // 8
    raw_packed = np.zeros((rows, cols, nbytes), dtype=np.uint8)
    level_counts = np.zeros((rows, cols, len(candidates)), dtype=np.uint16)

    amp = np.abs(stack).astype(np.float32, copy=False)
    mean_amp = np.nanmean(amp, axis=0, dtype=np.float64).astype(np.float32)
    mean_intensity = np.nanmean(amp * amp, axis=0, dtype=np.float64).astype(np.float32)

    report_every = int(progress_every or max(1, n_offsets // 20))
    for k, (dy0, dx0) in enumerate(offsets.tolist()):
        dy, dx = int(dy0), int(dx0)
        if dy == 0 and dx == 0:
            accept = np.zeros_like(mean_amp, dtype=bool)
        elif method == "glrt":
            accept = _glrt_accept(mean_intensity, _shift(mean_intensity, dy, dx), n_dates, alpha)
        elif method == "fashps":
            accept = _fashps_ci_accept(mean_amp, _shift(mean_amp, dy, dx), n_dates, alpha)
        else:
            accept = _ks_accept(amp, dy, dx, alpha, legacy_reverse=(method == "ks_reference"))

        byte, bit = k // 8, k % 8
        raw_packed[..., byte] |= (accept.astype(np.uint8) << bit)
        for lev, (hy, hx) in enumerate(candidates.tolist()):
            if abs(dy) <= int(hy) and abs(dx) <= int(hx):
                level_counts[..., lev] += accept.astype(np.uint16)
        if progress is not None and ((k + 1) % report_every == 0 or k + 1 == n_offsets):
            progress(f"SHP offsets {k+1}/{n_offsets} ({100*(k+1)/n_offsets:.1f}%)")

    target = max(int(min_samples_absolute), int(math.ceil(float(min_samples_per_date) * n_dates)))
    if custom_offsets or len(candidates) == 1 or target <= 0:
        chosen_level = np.full((rows, cols), len(candidates)-1, dtype=np.uint8)
    else:
        meets = level_counts >= target
        any_meets = np.any(meets, axis=2)
        first = np.argmax(meets, axis=2).astype(np.uint8)
        chosen_level = np.where(any_meets, first, len(candidates)-1).astype(np.uint8)

    if custom_offsets or len(candidates) == 1:
        packed = raw_packed
        counts = level_counts[..., -1]
    else:
        packed = np.zeros_like(raw_packed)
        for k, (dy0, dx0) in enumerate(offsets.tolist()):
            dy, dx = int(dy0), int(dx0)
            raw_accept = ((raw_packed[..., k // 8] >> (k % 8)) & 1).astype(bool)
            allowed = np.zeros((rows, cols), dtype=bool)
            for lev, (hy, hx) in enumerate(candidates.tolist()):
                allowed |= (chosen_level == lev) & (abs(dy) <= int(hy)) & (abs(dx) <= int(hx))
            final_accept = raw_accept & allowed
            packed[..., k // 8] |= (final_accept.astype(np.uint8) << (k % 8))
        counts = np.take_along_axis(level_counts, chosen_level[..., None], axis=2)[..., 0]

    return ShpResult(
        packed_bits=packed,
        counts=counts.astype(np.uint16),
        offsets=offsets,
        method=method,
        alpha=float(alpha),
        half_window=(max_hy, max_hx),
        window_level=chosen_level,
        window_candidates=candidates,
        target_samples=int(target),
    )


def unpack_pixel_support(packed_pixel: np.ndarray, n_offsets: int) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed_pixel, dtype=np.uint8), bitorder="little")
    return bits[:n_offsets].astype(bool)
