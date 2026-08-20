from __future__ import annotations

import time

import numpy as np

from ..progress import ProgressReporter

from .shp_vectorized_exact import (
    glrt_support_vectorized_exact,
)


def make_windows(
    mask,
    *,
    half_row,
    half_col,
):
    pad = (
        (half_row, half_row),
        (half_col, half_col),
    )

    x = np.pad(
        mask,
        pad,
        mode="constant",
        constant_values=False,
    )

    return (
        np.lib.stride_tricks
        .sliding_window_view(
            x,
            (
                2 * half_row + 1,
                2 * half_col + 1,
            ),
        )
    )

def compute_original_K(
    *,
    ctx,
    mask,
    alpha,
    ndate,
    batch,
    support_block,
    support_cache=None,
):
    H, W = mask.shape

    Kmap = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    rr, cc = np.where(
        mask
    )

    rr = rr.astype(
        np.int32,
        copy=False,
    )

    cc = cc.astype(
        np.int32,
        copy=False,
    )

    total = rr.size


    if support_cache is not None:

        cached = np.asarray(
            support_cache.static_k[
                rr,
                cc,
            ],
            dtype=np.int16,
        )

        if np.any(
            cached < 0
        ):

            bad = int(
                np.flatnonzero(
                    cached < 0
                )[0]
            )

            raise RuntimeError(
                "static support cache missing original K at "
                f"({int(rr[bad])},{int(cc[bad])})"
            )

        Kmap[
            rr,
            cc,
        ] = cached

        print(
            "original K source : exact static cache"
        )

        return Kmap

    t0 = time.perf_counter()

    progress = ProgressReporter(
        label="original-K",
        total=total,
        unit="center",
        min_interval=10.0,
    )

    for start in range(
        0,
        total,
        batch,
    ):

        stop = min(
            total,
            start + batch,
        )

        br = rr[
            start:stop
        ]

        bc = cc[
            start:stop
        ]

        support, K = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=alpha,
                nslc=ndate,
                block_size=support_block,
            )
        )

        Kmap[
            br,
            bc,
        ] = K

        del support

        progress.update(
            stop
        )

        if (
            stop == total
            or
            stop % (
                batch * 10
            ) == 0
        ):

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            rate = (
                stop / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"original K "
                f"{stop:,}/{total:,} "
                f"({100*stop/total:6.2f}%) "
                f"rate={rate:,.0f} center/s"
            )

    progress.finish(
        total
    )

    return Kmap

def effective_counts(
    *,
    ctx,
    center_mask,
    state_mask,
    alpha,
    ndate,
    batch,
    support_block,
    half_row,
    half_col,
    support_cache=None,
):

    rr, cc = np.where(
        center_mask
    )

    rr = rr.astype(
        np.int32,
        copy=False,
    )

    cc = cc.astype(
        np.int32,
        copy=False,
    )

    total = rr.size

    counts = np.full(
        total,
        -1,
        dtype=np.int16,
    )

    state_windows = make_windows(
        state_mask,
        half_row=half_row,
        half_col=half_col,
    )

    progress = ProgressReporter(
        label="effective-K",
        total=total,
        unit="center",
        min_interval=10.0,
    )

    for start in range(
        0,
        total,
        batch,
    ):

        stop = min(
            total,
            start + batch,
        )

        br = rr[
            start:stop
        ]

        bc = cc[
            start:stop
        ]

        if support_cache is None:

            support, _ = (
                glrt_support_vectorized_exact(
                    ctx,
                    br,
                    bc,
                    alpha=alpha,
                    nslc=ndate,
                    block_size=support_block,
                )
            )

        else:

            support = (
                support_cache.support(
                    br,
                    bc,
                )
            )

        state_local = np.asarray(
            state_windows[
                br,
                bc,
            ],
            dtype=np.bool_,
        )

        counts[
            start:stop
        ] = np.sum(
            support
            &
            state_local,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

        del support
        del state_local

        progress.update(
            stop
        )

    progress.finish(
        total
    )

    return (
        rr,
        cc,
        counts,
    )

def fixed_point_core(
    *,
    ctx,
    valid_nonps,
    original_K,
    threshold,
    alpha,
    ndate,
    batch,
    support_block,
    half_row,
    half_col,
    support_cache=None,
):

    # --------------------------------------------------------
    # Initial candidate state.
    #
    # No state with original K < threshold can ever survive.
    # --------------------------------------------------------

    state = (
        valid_nonps
        &
        (
            original_K
            >=
            threshold
        )
    )

    history = []

    iteration = 0

    while True:

        iteration += 1

        n_before = int(
            np.count_nonzero(
                state
            )
        )

        state_windows = make_windows(
            state,
            half_row=half_row,
            half_col=half_col,
        )

        rr, cc = np.where(
            state
        )

        rr = rr.astype(
            np.int32,
            copy=False,
        )

        cc = cc.astype(
            np.int32,
            copy=False,
        )

        remove = np.zeros(
            state.shape,
            dtype=np.bool_,
        )

        min_eff = None

        t0 = time.perf_counter()

        iter_progress = ProgressReporter(
            label=(
                f"Kstate-{threshold}-"
                f"iter-{iteration}"
            ),
            total=rr.size,
            unit="center",
            min_interval=10.0,
        )

        for start in range(
            0,
            rr.size,
            batch,
        ):

            stop = min(
                rr.size,
                start + batch,
            )

            br = rr[
                start:stop
            ]

            bc = cc[
                start:stop
            ]

            if support_cache is None:

                support, _ = (
                    glrt_support_vectorized_exact(
                        ctx,
                        br,
                        bc,
                        alpha=alpha,
                        nslc=ndate,
                        block_size=support_block,
                    )
                )

            else:

                support = (
                    support_cache.support(
                        br,
                        bc,
                    )
                )

            state_local = np.asarray(
                state_windows[
                    br,
                    bc,
                ],
                dtype=np.bool_,
            )

            eff = np.sum(
                support
                &
                state_local,
                axis=(1, 2),
                dtype=np.int32,
            )

            cur_min = int(
                np.min(
                    eff
                )
            ) if eff.size else 0

            if (
                min_eff is None
                or
                cur_min < min_eff
            ):
                min_eff = cur_min

            bad = (
                eff
                <
                threshold
            )

            if np.any(
                bad
            ):
                remove[
                    br[bad],
                    bc[bad],
                ] = True

            del support
            del state_local

            iter_progress.update(
                stop
            )

        iter_progress.finish(
            rr.size
        )

        n_remove = int(
            np.count_nonzero(
                remove
            )
        )

        state[
            remove
        ] = False

        n_after = int(
            np.count_nonzero(
                state
            )
        )

        elapsed = (
            time.perf_counter()
            -
            t0
        )

        history.append(
            {
                "iteration":
                    iteration,

                "before":
                    n_before,

                "removed":
                    n_remove,

                "after":
                    n_after,

                "minimum_effective_K":
                    min_eff,

                "seconds":
                    elapsed,
            }
        )

        print(
            f"Kstate={threshold:2d} "
            f"iter={iteration:2d}: "
            f"before={n_before:,} "
            f"remove={n_remove:,} "
            f"after={n_after:,} "
            f"wall={elapsed:.2f}s"
        )

        if n_remove == 0:
            break

        if iteration > 100:
            raise RuntimeError(
                "fixed-point did not converge "
                "within 100 iterations"
            )

    return (
        state,
        history,
    )

# Production semantic name.
#
# state_min_shp (K24 in the validated production configuration)
# controls compressed-state continuity only.  It does NOT replace
# the formal DS eligibility threshold selection.shp.min_count=48.
build_fixed_point_state_core = fixed_point_core


__all__ = [
    "make_windows",
    "compute_original_K",
    "effective_counts",
    "fixed_point_core",
    "build_fixed_point_state_core",
]
