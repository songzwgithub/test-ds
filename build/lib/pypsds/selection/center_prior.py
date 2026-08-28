from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def moraine_ks_d_sorted(ref, sec):
    """Literal Moraine two-sample KS distance for two sorted histories."""
    n = ref.size
    j1 = 0
    j2 = 0
    dmax = 0.0
    while j1 < n and j2 < n:
        f1 = ref[j1]
        f2 = sec[j2]
        if f1 <= f2:
            j1 += 1
        if f1 >= f2:
            j2 += 1
        d = abs((j2 - j1) / n)
        if d > dmax:
            dmax = d
    return dmax


@njit(cache=True)
def moraine_ks_p(x):
    """Literal Moraine asymptotic KS survival-series approximation."""
    x2 = -2.0 * x * x
    p = 0.0
    p2 = 0.0
    sign = 1.0
    for i in range(1, 101):
        p += sign * 2.0 * math.exp(x2 * i * i)
        if p == p2:
            return p
        sign = -sign
        p2 = p
    return p


@njit(cache=True, parallel=True, nogil=True)
def _selected_for_offset(sorted_amp, valid, dy, dx, p_max):
    n, H, W = sorted_amp.shape
    selected = np.zeros((H, W), np.uint8)
    available = np.zeros((H, W), np.uint8)

    if dy >= 0:
        r0, r1 = 0, H - dy
    else:
        r0, r1 = -dy, H
    if dx >= 0:
        c0, c1 = 0, W - dx
    else:
        c0, c1 = -dx, W

    en = math.sqrt(n / 2.0)
    corr = en + 0.12 + 0.11 / en

    for r in prange(r0, r1):
        rr = r + dy
        for c in range(c0, c1):
            cc = c + dx
            if not (valid[r, c] and valid[rr, cc]):
                continue
            d = moraine_ks_d_sorted(sorted_amp[:, r, c], sorted_amp[:, rr, cc])
            p = moraine_ks_p(corr * d)
            available[r, c] = 1
            if p < p_max:
                selected[r, c] = 1
    return selected, available


def strict_valid_mask(rslc: np.ndarray) -> np.ndarray:
    finite = np.isfinite(rslc.real) & np.isfinite(rslc.imag)
    nonzero = ~((rslc.real == 0) & (rslc.imag == 0))
    return np.all(finite & nonzero, axis=0)


def exact_moraine_shp_count(
    rslc: np.ndarray,
    *,
    half_row: int = 5,
    half_col: int = 5,
    p_max: float = 0.05,
    valid: np.ndarray | None = None,
):
    """Compute the reproduced Moraine `p < p_max` neighborhood count.

    Input uses amplitude = abs(RSLC), exactly matching Moraine's rslc2amp path.
    The center offset is included in the sweep but never selected because p=1,
    so an 11x11 window has a practical maximum selected count of 120.
    """
    if rslc.ndim != 3:
        raise ValueError("rslc must have shape [Ndate,H,W]")
    if valid is None:
        valid = strict_valid_mask(rslc)
    amp = np.abs(rslc).astype(np.float32)
    amp[:, ~valid] = np.nan
    sorted_amp = np.sort(amp, axis=0)
    H, W = valid.shape
    count = np.zeros((H, W), np.int16)
    available = np.zeros((H, W), np.int16)

    for dy in range(-half_row, half_row + 1):
        for dx in range(-half_col, half_col + 1):
            sel, ava = _selected_for_offset(sorted_amp, valid, dy, dx, float(p_max))
            count += sel.astype(np.int16)
            available += ava.astype(np.int16)
    return count, available, valid


__all__ = [
    "exact_moraine_shp_count",
    "moraine_ks_d_sorted",
    "moraine_ks_p",
    "strict_valid_mask",
]
