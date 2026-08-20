from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shp import ShpResult, unpack_pixel_support


@dataclass(frozen=True, slots=True)
class CovarianceResult:
    centers: np.ndarray       # (npix, 2) local row,col
    shp_counts: np.ndarray    # (npix,)
    covariance: np.ndarray    # (npix, ndate, ndate), complex64
    coherence: np.ndarray     # same
    box_coherence: np.ndarray # same, rectangular-window benchmark


def _normalize_covariance(cov: np.ndarray) -> np.ndarray:
    d = np.real(np.diag(cov)).astype(np.float64)
    denom = np.sqrt(np.maximum(d[:, None] * d[None, :], np.finfo(np.float64).tiny))
    coh = cov / denom
    np.fill_diagonal(coh, 1.0 + 0.0j)
    return coh.astype(np.complex64)


def _cov_from_samples(samples: np.ndarray) -> np.ndarray:
    # samples shape (dates, looks)
    if samples.shape[1] < 1:
        raise ValueError("No samples for covariance")
    valid = np.all(np.isfinite(samples.real) & np.isfinite(samples.imag), axis=0)
    x = samples[:, valid]
    if x.shape[1] < 1:
        raise ValueError("No finite samples for covariance")
    return (x @ x.conj().T / x.shape[1]).astype(np.complex64)


def choose_centers(
    counts: np.ndarray,
    *,
    min_shp: int,
    stride: int = 1,
    max_pixels: int | None = None,
    margin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    hy, hx = margin
    mask = counts >= int(min_shp)
    if hy > 0:
        mask[:hy, :] = False
        mask[-hy:, :] = False
    if hx > 0:
        mask[:, :hx] = False
        mask[:, -hx:] = False
    if stride > 1:
        grid = np.zeros_like(mask)
        grid[::stride, ::stride] = True
        mask &= grid
    centers = np.argwhere(mask).astype(np.int32)
    if max_pixels is not None and len(centers) > max_pixels:
        # deterministic even spacing, better for debug than random sampling
        idx = np.linspace(0, len(centers) - 1, int(max_pixels), dtype=np.int64)
        centers = centers[idx]
    return centers


def estimate_selected_covariances(
    stack: np.ndarray,
    shp: ShpResult,
    *,
    min_shp: int = 20,
    center_stride: int = 1,
    max_pixels: int | None = 2000,
    box_half_window: tuple[int, int] | None = None,
    progress=None,
    progress_every: int = 100,
) -> CovarianceResult:
    if stack.ndim != 3:
        raise ValueError("stack must be (dates, rows, cols)")
    ndate, rows, cols = stack.shape
    hy, hx = shp.half_window
    box_hy, box_hx = box_half_window if box_half_window is not None else shp.half_window
    centers = choose_centers(
        shp.counts,
        min_shp=min_shp,
        stride=center_stride,
        max_pixels=max_pixels,
        margin=(hy, hx),
    )
    if len(centers) == 0:
        raise ValueError("No covariance centers passed min_shp/stride/margin filters")

    offsets = shp.offsets.astype(np.int32)
    covs = np.empty((len(centers), ndate, ndate), dtype=np.complex64)
    cohs = np.empty_like(covs)
    box_cohs = np.empty_like(covs)
    counts = np.empty(len(centers), dtype=np.uint16)

    for p, (r, c) in enumerate(centers.tolist()):
        support = unpack_pixel_support(shp.packed_bits[r, c], len(offsets))
        selected_offsets = offsets[support]
        rr = r + selected_offsets[:, 0]
        cc = c + selected_offsets[:, 1]
        samples = stack[:, rr, cc]
        cov = _cov_from_samples(samples)
        covs[p] = cov
        cohs[p] = _normalize_covariance(cov)
        counts[p] = len(selected_offsets)

        box = stack[:, r-box_hy:r+box_hy+1, c-box_hx:c+box_hx+1].reshape(ndate, -1)
        box_cov = _cov_from_samples(box)
        box_cohs[p] = _normalize_covariance(box_cov)
        if progress is not None and ((p + 1) % max(1,int(progress_every)) == 0 or p + 1 == len(centers)):
            progress(f"Covariance centers: {p+1}/{len(centers)} ({100*(p+1)/len(centers):.1f}%)")

    return CovarianceResult(centers, counts, covs, cohs, box_cohs)
