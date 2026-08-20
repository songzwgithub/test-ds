from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ESTIMATOR_EVD = np.int8(0)
ESTIMATOR_EMI = np.int8(1)

PL_STATUS_OK = np.int8(0)
PL_STATUS_EVD_REQUESTED = np.int8(1)
PL_STATUS_EMI_CHOLESKY_FALLBACK = np.int8(10)
PL_STATUS_EMI_NONFINITE_FALLBACK = np.int8(11)
PL_STATUS_EMI_EIGH_FALLBACK = np.int8(12)
PL_STATUS_EMI_VECTOR_FALLBACK = np.int8(13)


@dataclass(frozen=True, slots=True)
class PhaseLinkResult:
    phase_rad: np.ndarray
    cpx_phase: np.ndarray
    temporal_coherence: np.ndarray
    eigenvalue: np.ndarray
    estimator: str
    estimator_code: np.ndarray
    status_code: np.ndarray


def _reference(v: np.ndarray, reference_idx: int) -> np.ndarray:
    return v * np.exp(-1j * np.angle(v[reference_idx]))


def evd_link_one(C: np.ndarray, reference_idx: int = 0) -> tuple[np.ndarray, float]:
    A = np.asarray(C, dtype=np.complex128) * np.abs(C)
    A = 0.5 * (A + A.conj().T)
    vals, vecs = np.linalg.eigh(A)
    idx = int(np.argmax(vals.real))
    v = _reference(vecs[:, idx], reference_idx)
    v = np.exp(1j * np.angle(v))
    return v.astype(np.complex64), float(vals[idx].real)


def _fallback(C, reference_idx, status):
    v, e = evd_link_one(C, reference_idx)
    return v, e, ESTIMATOR_EVD, np.int8(status)


def _emi_link_one_diagnostic(
    C: np.ndarray,
    reference_idx: int = 0,
    beta: float = 0.05,
) -> tuple[np.ndarray, float, np.int8, np.int8]:
    """CPU robust EMI with Dolphin-style regularization and EVD fallback."""
    C64 = np.asarray(C, dtype=np.complex128)
    n = C64.shape[0]
    I = np.eye(n, dtype=np.float64)
    Gamma = np.abs(C64).astype(np.float64, copy=False)
    Gamma = 0.5 * (Gamma + Gamma.T)
    beta = float(np.clip(beta, 0.0, 0.999))
    if beta > 0.0:
        Gamma = (1.0 - beta) * Gamma + beta * I
    Gamma_j = Gamma + 1.0e-6 * I

    try:
        L = np.linalg.cholesky(Gamma_j)
        Y = np.linalg.solve(L, I)
        Gamma_inv = np.linalg.solve(L.T, Y)
    except np.linalg.LinAlgError:
        return _fallback(C, reference_idx, PL_STATUS_EMI_CHOLESKY_FALLBACK)

    if not np.all(np.isfinite(Gamma_inv)):
        return _fallback(C, reference_idx, PL_STATUS_EMI_NONFINITE_FALLBACK)

    A = Gamma_inv * C64  # Hadamard product, not matrix product.
    A = 0.5 * (A + A.conj().T)
    if not np.all(np.isfinite(A)):
        return _fallback(C, reference_idx, PL_STATUS_EMI_NONFINITE_FALLBACK)

    try:
        vals, vecs = np.linalg.eigh(A)
    except np.linalg.LinAlgError:
        return _fallback(C, reference_idx, PL_STATUS_EMI_EIGH_FALLBACK)

    idx = int(np.argmin(np.abs(vals.real - 1.0)))
    v = vecs[:, idx]
    norm = np.linalg.norm(v)
    if not np.all(np.isfinite(v)) or not np.isfinite(norm) or norm <= 0:
        return _fallback(C, reference_idx, PL_STATUS_EMI_VECTOR_FALLBACK)

    v = np.sqrt(n) * v / norm
    v = _reference(v, reference_idx)
    v = np.exp(1j * np.angle(v))
    return v.astype(np.complex64), float(vals[idx].real), ESTIMATOR_EMI, PL_STATUS_OK


def emi_link_one(C: np.ndarray, reference_idx: int = 0, beta: float = 0.05):
    v, e, _, _ = _emi_link_one_diagnostic(C, reference_idx, beta)
    return v, e


def temporal_coherence_one(v: np.ndarray, C: np.ndarray, weighted: bool = False) -> float:
    predicted = v[:, None] @ v[None, :].conj()
    observed = np.exp(1j * np.angle(C))
    differences = observed * np.exp(-1j * np.angle(predicted))
    W = np.abs(C) if weighted else np.ones(C.shape, dtype=np.float32)
    rows, cols = np.triu_indices(C.shape[0], k=1)
    z = differences[rows, cols]; w = W[rows, cols]
    valid = np.isfinite(z.real) & np.isfinite(z.imag) & np.isfinite(w) & (np.abs(C[rows, cols]) > 0)
    if not np.any(valid) or np.sum(w[valid]) <= 0:
        return 0.0
    return float(np.abs(np.sum(z[valid] * w[valid]) / np.sum(w[valid])))


def median_pair_coherence(C: np.ndarray) -> float:
    rows, cols = np.triu_indices(C.shape[0], k=1)
    x = np.abs(C[rows, cols]); x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def link_one_diagnostic(
    C: np.ndarray,
    *,
    method: str = "emi",
    reference_idx: int = 0,
    beta: float = 0.05,
    weighted_temp_coh: bool = False,
):
    method = method.lower()
    if method == "evd":
        v, eig = evd_link_one(C, reference_idx)
        code, status = ESTIMATOR_EVD, PL_STATUS_EVD_REQUESTED
    elif method in {"emi", "robust_emi"}:
        v, eig, code, status = _emi_link_one_diagnostic(C, reference_idx, beta)
    else:
        raise ValueError("method must be emi/robust_emi or evd")
    tc = temporal_coherence_one(v, C, weighted=weighted_temp_coh)
    return v, eig, tc, code, status


def link_one(
    C: np.ndarray,
    *,
    method: str = "emi",
    reference_idx: int = 0,
    beta: float = 0.05,
    weighted_temp_coh: bool = False,
):
    v, eig, tc, code, _ = link_one_diagnostic(
        C,
        method=method,
        reference_idx=reference_idx,
        beta=beta,
        weighted_temp_coh=weighted_temp_coh,
    )
    return v, eig, tc, code


def link_stack(
    coherence: np.ndarray,
    *,
    method: str = "emi",
    reference_idx: int = 0,
    beta: float = 0.05,
    weighted_temp_coh: bool = False,
    progress=None,
    progress_every: int = 100,
) -> PhaseLinkResult:
    if coherence.ndim != 3 or coherence.shape[1] != coherence.shape[2]:
        raise ValueError("coherence must be (pixels, dates, dates)")
    npix, ndate, _ = coherence.shape
    cpx = np.empty((npix, ndate), dtype=np.complex64)
    eig = np.empty(npix, dtype=np.float32); tc = np.empty(npix, dtype=np.float32)
    estimator_code = np.empty(npix, dtype=np.int8); status = np.empty(npix, dtype=np.int8)
    for i in range(npix):
        v, e, t, code, st = link_one_diagnostic(
            coherence[i], method=method, reference_idx=reference_idx,
            beta=beta, weighted_temp_coh=weighted_temp_coh,
        )
        cpx[i] = v; eig[i] = e; tc[i] = t; estimator_code[i] = code; status[i] = st
        if progress is not None and ((i + 1) % max(1, int(progress_every)) == 0 or i + 1 == npix):
            progress(f"Phase linking pixels: {i+1}/{npix} ({100*(i+1)/npix:.1f}%)")
    return PhaseLinkResult(
        phase_rad=np.angle(cpx).astype(np.float32), cpx_phase=cpx,
        temporal_coherence=tc, eigenvalue=eig, estimator=method,
        estimator_code=estimator_code, status_code=status,
    )
