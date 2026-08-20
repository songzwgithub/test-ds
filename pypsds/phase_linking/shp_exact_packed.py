from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import numpy as np
from numba import njit, prange

from pypsds.selection.shp import (
    glrt_statistic,
    glrt_threshold,
)


def _exact_worker(
    scale2,
    valid,
    ps,
    rows,
    cols,
    *,
    half_row,
    half_col,
    alpha,
    nslc,
):
    """
    Frozen exact GLRT implementation, but persist support as uint64 bits.

    Important:
      GLRT arithmetic deliberately remains in NumPy and calls the existing
      glrt_statistic(). This is intended to preserve v1.0 scientific parity.
    """

    rows = np.asarray(
        rows,
        dtype=np.int32,
    )

    cols = np.asarray(
        cols,
        dtype=np.int32,
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

    bits = np.zeros(
        (
            B,
            nwords,
        ),
        dtype=np.uint64,
    )

    count = np.zeros(
        B,
        dtype=np.int16,
    )

    center_scale = np.asarray(
        scale2[
            rows,
            cols,
        ],
        dtype=np.float64,
    )

    threshold = glrt_threshold(
        alpha
    )

    H, W = valid.shape

    for ky, dy in enumerate(
        range(
            -half_row,
            half_row + 1,
        )
    ):

        for kx, dx in enumerate(
            range(
                -half_col,
                half_col + 1,
            )
        ):

            # Frozen implementation excludes center.
            if (
                dy == 0
                and
                dx == 0
            ):
                continue

            rr = (
                rows
                +
                dy
            )

            cc = (
                cols
                +
                dx
            )

            inside = (
                (rr >= 0)
                &
                (rr < H)
                &
                (cc >= 0)
                &
                (cc < W)
            )

            if not np.any(
                inside
            ):
                continue

            ids = np.flatnonzero(
                inside
            )

            r2 = rr[
                ids
            ]

            c2 = cc[
                ids
            ]

            ngood = (
                valid[
                    r2,
                    c2,
                ]
                &
                ~ps[
                    r2,
                    c2,
                ]
            )

            if not np.any(
                ngood
            ):
                continue

            ids2 = ids[
                ngood
            ]

            r3 = rr[
                ids2
            ]

            c3 = cc[
                ids2
            ]

            # --------------------------------------------------
            # EXACT existing NumPy path.
            # --------------------------------------------------

            stat = glrt_statistic(
                center_scale[
                    ids2
                ],
                scale2[
                    r3,
                    c3,
                ],
                nslc=nslc,
            )

            accepted = (
                np.isfinite(
                    stat
                )
                &
                (
                    stat
                    <
                    threshold
                )
            )

            if not np.any(
                accepted
            ):
                continue

            good_ids = ids2[
                accepted
            ]

            flat = (
                ky
                *
                ww
                +
                kx
            )

            word = (
                flat
                //
                64
            )

            bit = (
                flat
                %
                64
            )

            mask = (
                np.uint64(1)
                <<
                np.uint64(
                    bit
                )
            )

            bits[
                good_ids,
                word,
            ] |= mask

            count[
                good_ids
            ] += np.int16(1)

    return (
        bits,
        count,
    )


def glrt_support_exact_packed(
    scale2,
    valid,
    ps,
    rows,
    cols,
    *,
    half_row,
    half_col,
    alpha,
    nslc,
    workers=1,
    chunk_size=2048,
):
    """
    Thread-parallel exact NumPy GLRT.

    Parallel decomposition is across center pixels, so every worker writes
    an independent output slice. No shared-bit races occur.
    """

    rows = np.asarray(
        rows,
        dtype=np.int32,
    )

    cols = np.asarray(
        cols,
        dtype=np.int32,
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

    nwords = (
        (
            wh
            *
            ww
        )
        +
        63
    ) // 64

    bits = np.zeros(
        (
            B,
            nwords,
        ),
        dtype=np.uint64,
    )

    count = np.zeros(
        B,
        dtype=np.int16,
    )

    workers = max(
        1,
        int(
            workers
        ),
    )

    chunk_size = max(
        1,
        int(
            chunk_size
        ),
    )

    if (
        workers == 1
        or
        B <= chunk_size
    ):

        return _exact_worker(
            scale2,
            valid,
            ps,
            rows,
            cols,
            half_row=half_row,
            half_col=half_col,
            alpha=alpha,
            nslc=nslc,
        )

    ranges = [
        (
            s,
            min(
                B,
                s
                +
                chunk_size,
            ),
        )
        for s in range(
            0,
            B,
            chunk_size,
        )
    ]

    def work(
        s,
        e,
    ):
        x = _exact_worker(
            scale2,
            valid,
            ps,
            rows[
                s:e
            ],
            cols[
                s:e
            ],
            half_row=half_row,
            half_col=half_col,
            alpha=alpha,
            nslc=nslc,
        )

        return (
            s,
            e,
            x,
        )

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="pypsds-shp",
    ) as ex:

        futures = [
            ex.submit(
                work,
                s,
                e,
            )
            for s, e
            in ranges
        ]

        for fut in as_completed(
            futures
        ):

            s, e, result = (
                fut.result()
            )

            b, k = result

            bits[
                s:e
            ] = b

            count[
                s:e
            ] = k

    return (
        bits,
        count,
    )


@njit(
    cache=True,
    parallel=True,
    nogil=True,
)
def unpack_support_bitset(
    bits,
    wh,
    ww,
):
    """
    Convert compact support only when a downstream kernel requires bool.
    """

    B = bits.shape[0]

    out = np.zeros(
        (
            B,
            wh,
            ww,
        ),
        dtype=np.bool_,
    )

    for p in prange(B):

        for ky in range(
            wh
        ):

            for kx in range(
                ww
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

                out[
                    p,
                    ky,
                    kx,
                ] = (
                    (
                        bits[
                            p,
                            word,
                        ]
                        >>
                        np.uint64(
                            bit
                        )
                    )
                    &
                    np.uint64(1)
                ) != 0

    return out


__all__ = [
    "glrt_support_exact_packed",
    "unpack_support_bitset",
]
