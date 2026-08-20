from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(
    cache=True,
    parallel=True,
    nogil=True,
)
def glrt_support_bitset(
    scale2,
    valid,
    ps,
    rows,
    cols,
    *,
    half_row,
    half_col,
    threshold,
    nslc,
):
    """
    Exact pyPSDS/Dolphin GLRT SHP definition, stored as bitsets.

    Current scientific test:

        stat =
            N * [
                2 log((s1+s2)/2)
                - log(s1)
                - log(s2)
            ]

        SHP iff stat < threshold

    Center pixel is excluded exactly as in the frozen v1.0
    implementation.

    Returns
    -------
    support_bits
        uint64 [B, n_words]

    shp_count
        int16 [B]
    """

    B = rows.size

    wh = (
        2 * half_row
        +
        1
    )

    ww = (
        2 * half_col
        +
        1
    )

    nwin = (
        wh
        *
        ww
    )

    nwords = (
        nwin
        +
        63
    ) // 64

    support_bits = np.zeros(
        (
            B,
            nwords,
        ),
        dtype=np.uint64,
    )

    shp_count = np.zeros(
        B,
        dtype=np.int16,
    )

    H, W = valid.shape

    for p in prange(B):

        cr = rows[p]
        cc = cols[p]

        s0 = float(
            scale2[
                cr,
                cc,
            ]
        )

        if (
            not math.isfinite(s0)
            or
            s0 <= 0.0
        ):
            continue

        count = 0

        for ky in range(wh):

            rr = (
                cr
                -
                half_row
                +
                ky
            )

            if (
                rr < 0
                or
                rr >= H
            ):
                continue

            for kx in range(ww):

                if (
                    ky == half_row
                    and
                    kx == half_col
                ):
                    continue

                rc = (
                    cc
                    -
                    half_col
                    +
                    kx
                )

                if (
                    rc < 0
                    or
                    rc >= W
                ):
                    continue

                if (
                    not valid[
                        rr,
                        rc,
                    ]
                    or
                    ps[
                        rr,
                        rc,
                    ]
                ):
                    continue

                s1 = float(
                    scale2[
                        rr,
                        rc,
                    ]
                )

                if (
                    not math.isfinite(s1)
                    or
                    s1 <= 0.0
                ):
                    continue

                pooled = (
                    0.5
                    *
                    (
                        s0
                        +
                        s1
                    )
                )

                stat = (
                    float(nslc)
                    *
                    (
                        2.0
                        *
                        math.log(
                            pooled
                        )
                        -
                        math.log(
                            s0
                        )
                        -
                        math.log(
                            s1
                        )
                    )
                )

                if (
                    math.isfinite(stat)
                    and
                    stat < threshold
                ):

                    flat = (
                        ky
                        *
                        ww
                        +
                        kx
                    )

                    word = (
                        flat
                        >>
                        6
                    )

                    bit = (
                        flat
                        &
                        63
                    )

                    support_bits[
                        p,
                        word,
                    ] |= (
                        np.uint64(1)
                        <<
                        np.uint64(bit)
                    )

                    count += 1

        shp_count[p] = np.int16(
            count
        )

    return (
        support_bits,
        shp_count,
    )


@njit(
    cache=True,
    parallel=True,
    nogil=True,
)
def compressed_coherence_bitset(
    rslc_yxt,
    rows,
    cols,
    support_bits,
    pair_i,
    pair_j,
    *,
    half_row,
    half_col,
):
    """
    Candidate-only coherence directly from bit-packed SHP support.

    Mathematical accumulation order matches the existing
    compressed_coherence() implementation.
    """

    H, W, N = (
        rslc_yxt.shape
    )

    B = rows.size

    wh = (
        2 * half_row
        +
        1
    )

    ww = (
        2 * half_col
        +
        1
    )

    npair = (
        pair_i.size
    )

    out = np.empty(
        (
            B,
            npair,
        ),
        dtype=np.complex64,
    )

    for p in prange(B):

        numer = np.zeros(
            npair,
            dtype=np.complex64,
        )

        power = np.zeros(
            N,
            dtype=np.float64,
        )

        zvec = np.empty(
            N,
            dtype=np.complex64,
        )

        cr = rows[p]
        cc = cols[p]

        K = 0

        for ky in range(wh):

            rr = (
                cr
                -
                half_row
                +
                ky
            )

            if (
                rr < 0
                or
                rr >= H
            ):
                continue

            for kx in range(ww):

                rc = (
                    cc
                    -
                    half_col
                    +
                    kx
                )

                if (
                    rc < 0
                    or
                    rc >= W
                ):
                    continue

                flat = (
                    ky
                    *
                    ww
                    +
                    kx
                )

                word = (
                    flat
                    >>
                    6
                )

                bit = (
                    flat
                    &
                    63
                )

                flag = (
                    support_bits[
                        p,
                        word,
                    ]
                    >>
                    np.uint64(
                        bit
                    )
                ) & np.uint64(1)

                if flag == 0:
                    continue

                K += 1

                for m in range(N):

                    z = rslc_yxt[
                        rr,
                        rc,
                        m,
                    ]

                    zvec[m] = z

                    power[m] += (
                        z.real
                        *
                        z.real
                        +
                        z.imag
                        *
                        z.imag
                    )

                for q in range(
                    npair
                ):

                    i = pair_i[q]
                    j = pair_j[q]

                    numer[q] += (
                        zvec[i]
                        *
                        np.conj(
                            zvec[j]
                        )
                    )

        if K <= 0:

            for q in range(
                npair
            ):
                out[
                    p,
                    q,
                ] = (
                    np.nan
                    +
                    1j
                    *
                    np.nan
                )

            continue

        for q in range(
            npair
        ):

            i = pair_i[q]
            j = pair_j[q]

            den = math.sqrt(
                power[i]
                *
                power[j]
            )

            if den > 0.0:

                out[
                    p,
                    q,
                ] = (
                    numer[q]
                    /
                    den
                )

            else:

                out[
                    p,
                    q,
                ] = (
                    np.nan
                    +
                    1j
                    *
                    np.nan
                )

    return out


__all__ = [
    "glrt_support_bitset",
    "compressed_coherence_bitset",
]
