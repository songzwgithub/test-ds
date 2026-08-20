from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import os

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except Exception:  # optional runtime helper
    threadpool_limits = None


ESTIMATOR_EVD = np.uint8(0)
ESTIMATOR_EMI = np.uint8(1)
ESTIMATOR_INVALID = np.uint8(255)


def image_pairs(n: int) -> np.ndarray:
    i, j = np.triu_indices(n, k=1)
    return np.column_stack([i, j]).astype(np.int32)


def uncompress_coherence(
    coh: np.ndarray,
    n_images: int,
    pairs: np.ndarray,
) -> np.ndarray:
    coh = np.asarray(coh, dtype=np.complex64)
    b = coh.shape[0]
    C = np.zeros((b, n_images, n_images), dtype=np.complex64)
    ii = pairs[:, 0]
    jj = pairs[:, 1]
    C[:, ii, jj] = coh
    C[:, jj, ii] = np.conj(coh)
    k = np.arange(n_images)
    C[:, k, k] = 1.0 + 0.0j
    return C


def _reference_unit_phase(vec: np.ndarray, reference_idx: int) -> np.ndarray:
    phase = np.exp(1j * np.angle(vec))
    phase *= np.exp(-1j * np.angle(phase[:, reference_idx]))[:, None]
    return phase.astype(np.complex64, copy=False)


def _take_eigvec(eigvecs: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.take_along_axis(
        eigvecs,
        idx[:, None, None],
        axis=2,
    )[:, :, 0]


def robust_emi_batch(
    coh: np.ndarray,
    *,
    n_images: int,
    pairs: np.ndarray,
    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,
    reference_idx: int = 0,
    min_gamma_eig: float = 1e-7,
):
    """Robust EMI with lazy EVD fallback.

    Production implementation:
      * EVD fallback is NOT computed for every point.
      * It is evaluated only for points where EMI is numerically rejected.

    Mathematical definition remains the validated v0.6 definition:
      Gamma = abs(C)
      Gamma_reg = (1-beta) Gamma + beta I + jitter I
      A = inv(Gamma_reg) * C            [Hadamard]
      EMI eigenpair = eigenvalue closest to mu
      fallback = largest eigenpair of C * abs(C)
    """
    C = uncompress_coherence(coh, n_images, pairs).astype(
        np.complex128, copy=False
    )
    b = C.shape[0]
    eye = np.eye(n_images, dtype=np.float64)

    Gamma = np.abs(C).real
    if beta > 0:
        Gamma = (1.0 - beta) * Gamma + beta * eye[None, :, :]
    Gamma = Gamma + gamma_jitter * eye[None, :, :]
    Gamma = 0.5 * (Gamma + np.swapaxes(Gamma, -1, -2))

    # Validated v0.6 inversion route retained for numerical parity.
    gw, gv = np.linalg.eigh(Gamma)
    gamma_min_eig = gw[:, 0].real
    gamma_ok = (
        np.all(np.isfinite(gw), axis=1)
        & (gamma_min_eig > min_gamma_eig)
    )

    safe_w = np.where(gw > min_gamma_eig, gw, 1.0)
    Gamma_inv = np.einsum(
        "bik,bk,bjk->bij",
        gv,
        1.0 / safe_w,
        gv,
        optimize=True,
    )

    A = Gamma_inv * C
    A = 0.5 * (A + np.swapaxes(A.conj(), -1, -2))
    ew, ev = np.linalg.eigh(A)
    emi_idx = np.argmin(np.abs(ew.real - emi_mu), axis=1)
    emi_vec = _take_eigvec(ev, emi_idx)
    emi_val_all = ew[np.arange(b), emi_idx].real

    emi_ok = (
        gamma_ok
        & np.isfinite(emi_val_all)
        & np.all(np.isfinite(emi_vec.real) & np.isfinite(emi_vec.imag), axis=1)
    )

    phase = np.full((b, n_images), np.nan + 1j * np.nan, dtype=np.complex64)
    estimator = np.full(b, ESTIMATOR_INVALID, dtype=np.uint8)
    emi_eigenvalue = np.full(b, np.nan, dtype=np.float32)
    evd_eigenvalue = np.full(b, np.nan, dtype=np.float32)

    if np.any(emi_ok):
        # Dolphin EMI normalization to sqrt(N), phase is unchanged but keep parity.
        v = emi_vec[emi_ok]
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm = np.where(norm > 0, norm, 1.0)
        v = np.sqrt(n_images) * v / norm
        phase[emi_ok] = _reference_unit_phase(v, reference_idx)
        estimator[emi_ok] = ESTIMATOR_EMI
        emi_eigenvalue[emi_ok] = emi_val_all[emi_ok].astype(np.float32)

    # LAZY fallback: only rejected EMI points pay for EVD.
    bad = ~emi_ok
    if np.any(bad):
        Cb = C[bad]
        B = Cb * np.abs(Cb)
        B = 0.5 * (B + np.swapaxes(B.conj(), -1, -2))
        bw, bv = np.linalg.eigh(B)
        evd_idx = np.argmax(bw.real, axis=1)
        evd_vec = _take_eigvec(bv, evd_idx)
        evd_val = bw[np.arange(bw.shape[0]), evd_idx].real
        evd_ok_local = (
            np.isfinite(evd_val)
            & np.all(
                np.isfinite(evd_vec.real) & np.isfinite(evd_vec.imag),
                axis=1,
            )
        )
        bad_ids = np.flatnonzero(bad)
        good_bad_ids = bad_ids[evd_ok_local]
        if good_bad_ids.size:
            phase[good_bad_ids] = _reference_unit_phase(
                evd_vec[evd_ok_local], reference_idx
            )
            estimator[good_bad_ids] = ESTIMATOR_EVD
            evd_eigenvalue[good_bad_ids] = evd_val[evd_ok_local].astype(np.float32)

    return (
        phase,
        estimator,
        emi_eigenvalue,
        evd_eigenvalue,
        gamma_min_eig.astype(np.float32),
    )


def robust_emi_threaded(
    coh: np.ndarray,
    *,
    n_images: int,
    pairs: np.ndarray,
    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,
    reference_idx: int = 0,
    min_gamma_eig: float = 1e-7,
    workers: int = 16,
    chunk_size: int = 512,
):
    """Parallelize small 38x38 LAPACK jobs across independent chunks.

    NumPy batch eigh is otherwise effectively serial on this workload when
    BLAS/LAPACK threads are limited to one. Threaded chunks let independent
    LAPACK calls execute concurrently because NumPy releases the GIL.
    """
    coh = np.asarray(coh, dtype=np.complex64)
    b = coh.shape[0]
    if b == 0:
        return (
            np.empty((0, n_images), np.complex64),
            np.empty(0, np.uint8),
            np.empty(0, np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.float32),
        )

    workers = max(1, int(workers))
    chunk_size = max(1, int(chunk_size))
    if workers == 1 or b <= chunk_size:
        return robust_emi_batch(
            coh,
            n_images=n_images,
            pairs=pairs,
            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,
            reference_idx=reference_idx,
            min_gamma_eig=min_gamma_eig,
        )

    phase = np.empty((b, n_images), dtype=np.complex64)
    estimator = np.empty(b, dtype=np.uint8)
    emi_eig = np.empty(b, dtype=np.float32)
    evd_eig = np.empty(b, dtype=np.float32)
    gamma_min = np.empty(b, dtype=np.float32)

    ranges = [(s, min(b, s + chunk_size)) for s in range(0, b, chunk_size)]

    def work(s: int, e: int):
        return s, e, robust_emi_batch(
            coh[s:e],
            n_images=n_images,
            pairs=pairs,
            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,
            reference_idx=reference_idx,
            min_gamma_eig=min_gamma_eig,
        )

    ctx = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
    with ctx:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pypsds-pl") as ex:
            futures = [ex.submit(work, s, e) for s, e in ranges]
            for fut in as_completed(futures):
                s, e, result = fut.result()
                ph, est, ee, ve, gm = result
                phase[s:e] = ph
                estimator[s:e] = est
                emi_eig[s:e] = ee
                evd_eig[s:e] = ve
                gamma_min[s:e] = gm

    return phase, estimator, emi_eig, evd_eig, gamma_min


def temporal_coherence(
    coh: np.ndarray,
    phase: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    ii = pairs[:, 0]
    jj = pairs[:, 1]
    predicted = phase[:, ii] * np.conj(phase[:, jj])
    mag = np.abs(coh)
    valid = (
        np.isfinite(coh.real)
        & np.isfinite(coh.imag)
        & (mag > 0)
        & np.isfinite(predicted.real)
        & np.isfinite(predicted.imag)
    )
    observed = np.zeros_like(coh, dtype=np.complex64)
    observed[valid] = coh[valid] / mag[valid]
    residual = observed * np.conj(predicted)
    total = np.sum(np.where(valid, residual, 0.0 + 0.0j), axis=1)
    count = np.sum(valid, axis=1)
    out = np.full(coh.shape[0], np.nan, dtype=np.float32)
    good = count > 0
    out[good] = np.abs(total[good] / count[good]).astype(np.float32)
    return out


def median_pair_coherence(coh: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.abs(coh), axis=1).astype(np.float32)
