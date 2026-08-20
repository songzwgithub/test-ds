from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(
    cache=True,
    parallel=True,
    nogil=True,
)
def temporal_quality_streaming(
    coh,
    phase,
    pair_i,
    pair_j,
):
    """
    Memory-bounded temporal coherence + median pair coherence.

    Scientific definition matches current pyPSDS-GAMMA:

        predicted_ij = ph_i * conj(ph_j)

        residual_ij =
            (coh_ij / abs(coh_ij))
            * conj(predicted_ij)

        TC = abs(mean(residual_ij))

    No B x Npair predicted/observed/residual arrays are created.

    Parameters
    ----------
    coh
        complex64 [B, Npair]

    phase
        complex64 [B, Ndate]

    pair_i, pair_j
        int32 [Npair]

    Returns
    -------
    tc
        float32 [B]

    median_pair_coherence
        float32 [B]
    """

    B = coh.shape[0]
    Q = coh.shape[1]

    tc = np.empty(
        B,
        dtype=np.float32,
    )

    pair_median = np.empty(
        B,
        dtype=np.float32,
    )

    for p in prange(B):

        sum_r = 0.0
        sum_i = 0.0
        count = 0

        # Only thread-local O(Npair) storage.
        mags = np.empty(
            Q,
            dtype=np.float32,
        )

        nmag = 0

        for q in range(Q):

            z = coh[p, q]

            zr = z.real
            zi = z.imag

            if not (
                math.isfinite(zr)
                and
                math.isfinite(zi)
            ):
                continue

            mag = math.sqrt(
                zr * zr
                +
                zi * zi
            )

            mags[nmag] = np.float32(
                mag
            )

            nmag += 1

            if not (
                mag > 0.0
            ):
                continue

            i = pair_i[q]
            j = pair_j[q]

            a = phase[p, i]
            b = phase[p, j]

            if not (
                math.isfinite(a.real)
                and
                math.isfinite(a.imag)
                and
                math.isfinite(b.real)
                and
                math.isfinite(b.imag)
            ):
                continue

            # predicted = a * conj(b)
            pr = (
                a.real * b.real
                +
                a.imag * b.imag
            )

            pi = (
                a.imag * b.real
                -
                a.real * b.imag
            )

            # observed = z / |z|
            orr = zr / mag
            oii = zi / mag

            # residual =
            #     observed * conj(predicted)
            rr = (
                orr * pr
                +
                oii * pi
            )

            ri = (
                oii * pr
                -
                orr * pi
            )

            sum_r += rr
            sum_i += ri
            count += 1

        if count > 0:

            tc[p] = np.float32(
                math.sqrt(
                    sum_r * sum_r
                    +
                    sum_i * sum_i
                )
                /
                count
            )

        else:

            tc[p] = np.nan

        if nmag == 0:

            pair_median[p] = np.nan

        else:

            x = np.sort(
                mags[:nmag]
            )

            m = nmag // 2

            if nmag % 2:

                pair_median[p] = x[m]

            else:

                pair_median[p] = np.float32(
                    0.5
                    *
                    (
                        float(x[m - 1])
                        +
                        float(x[m])
                    )
                )

    return (
        tc,
        pair_median,
    )


__all__ = [
    "temporal_quality_streaming",
]
