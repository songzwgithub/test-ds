from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, nogil=True)
def compressed_coherence(
    rslc_yxt,
    rows,
    cols,
    support,
    pair_i,
    pair_j,
):
    """Candidate-only coherence for C-contiguous [H,W,Ndate] data.

    Production implementation:
      * phase-corrected cache is consumed in Y-X-Time layout;
      * each 38-date sample history is loaded once into a local contiguous vector;
      * pair products reuse the local vector instead of re-reading global memory.
    """
    H, W, N = rslc_yxt.shape
    B = rows.size
    wh = support.shape[1]
    ww = support.shape[2]
    hr = wh // 2
    hc = ww // 2
    npair = pair_i.size
    out = np.empty((B, npair), dtype=np.complex64)

    for p in prange(B):
        numer = np.zeros(npair, dtype=np.complex64)
        power = np.zeros(N, dtype=np.float64)
        zvec = np.empty(N, dtype=np.complex64)
        K = 0
        cr = rows[p]
        cc = cols[p]

        for ky in range(wh):
            rr = cr - hr + ky
            if rr < 0 or rr >= H:
                continue
            for kx in range(ww):
                if not support[p, ky, kx]:
                    continue
                rc = cc - hc + kx
                if rc < 0 or rc >= W:
                    continue

                K += 1
                for m in range(N):
                    z = rslc_yxt[rr, rc, m]
                    zvec[m] = z
                    power[m] += z.real * z.real + z.imag * z.imag

                for q in range(npair):
                    i = pair_i[q]
                    j = pair_j[q]
                    numer[q] += zvec[i] * np.conj(zvec[j])

        if K <= 0:
            for q in range(npair):
                out[p, q] = np.nan + 1j * np.nan
            continue

        for q in range(npair):
            i = pair_i[q]
            j = pair_j[q]
            den = math.sqrt(power[i] * power[j])
            if den > 0:
                out[p, q] = numer[q] / den
            else:
                out[p, q] = np.nan + 1j * np.nan

    return out
