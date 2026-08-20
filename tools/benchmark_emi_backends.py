#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from pypsds.context import open_from_config
from pypsds.phase_linking.coherence import compressed_coherence
from pypsds.phase_linking.emi import (
    ESTIMATOR_EVD,
    ESTIMATOR_EMI,
    ESTIMATOR_INVALID,
    image_pairs,
    robust_emi_threaded,
    uncompress_coherence,
)
from pypsds.selection.shp import glrt_statistic, glrt_threshold


def reference_unit_phase(vec, reference_idx=0):
    phase = np.exp(
        1j * np.angle(vec)
    )

    phase *= np.exp(
        -1j
        * np.angle(
            phase[:, reference_idx]
        )
    )[:, None]

    return phase.astype(
        np.complex64,
        copy=False,
    )


def take_eigvec(eigvecs, idx):
    return np.take_along_axis(
        eigvecs,
        idx[:, None, None],
        axis=2,
    )[:, :, 0]


def make_support_batch(
    scale2,
    valid,
    ps,
    rows,
    cols,
    *,
    half_row,
    half_col,
    alpha,
    ndate,
):
    B = rows.size
    wh = 2 * half_row + 1
    ww = 2 * half_col + 1

    out = np.zeros(
        (B, wh, ww),
        dtype=np.bool_,
    )

    center_scale = np.asarray(
        scale2[rows, cols],
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
            if dy == 0 and dx == 0:
                continue

            rr = rows + dy
            cc = cols + dx

            inside = (
                (rr >= 0)
                &
                (rr < H)
                &
                (cc >= 0)
                &
                (cc < W)
            )

            if not np.any(inside):
                continue

            ids = np.flatnonzero(
                inside
            )

            r2 = rr[ids]
            c2 = cc[ids]

            ngood = (
                valid[r2, c2]
                &
                ~ps[r2, c2]
            )

            if not np.any(ngood):
                continue

            ids2 = ids[ngood]

            r3 = rr[ids2]
            c3 = cc[ids2]

            stat = glrt_statistic(
                center_scale[ids2],
                scale2[r3, c3],
                nslc=ndate,
            )

            out[
                ids2,
                ky,
                kx,
            ] = (
                np.isfinite(stat)
                &
                (stat < threshold)
            )

    return out


def recursive_inverse(
    Gamma,
):
    """
    Batch inverse with recursive isolation if one matrix
    makes np.linalg.inv fail.
    """

    B = Gamma.shape[0]

    out = np.zeros_like(
        Gamma,
        dtype=np.float64,
    )

    ok = np.zeros(
        B,
        dtype=bool,
    )

    def solve(s, e):
        if s >= e:
            return

        try:
            inv = np.linalg.inv(
                Gamma[s:e]
            )

            finite = np.all(
                np.isfinite(inv),
                axis=(1, 2),
            )

            ids = np.flatnonzero(
                finite
            )

            if ids.size:
                out[s + ids] = inv[ids]
                ok[s + ids] = True

        except np.linalg.LinAlgError:
            if e - s == 1:
                return

            m = (s + e) // 2

            solve(s, m)
            solve(m, e)

    solve(
        0,
        B,
    )

    return out, ok


def recursive_cholesky_inverse(
    Gamma,
):
    """
    Cholesky inverse with recursive isolation of
    non-positive-definite matrices.
    """

    B, N, _ = Gamma.shape

    out = np.zeros_like(
        Gamma,
        dtype=np.float64,
    )

    ok = np.zeros(
        B,
        dtype=bool,
    )

    eye = np.eye(
        N,
        dtype=np.float64,
    )

    def solve(s, e):
        if s >= e:
            return

        G = Gamma[s:e]

        try:
            L = np.linalg.cholesky(
                G
            )

            rhs = np.broadcast_to(
                eye,
                (
                    e - s,
                    N,
                    N,
                ),
            )

            y = np.linalg.solve(
                L,
                rhs,
            )

            inv = np.linalg.solve(
                np.swapaxes(
                    L,
                    -1,
                    -2,
                ),
                y,
            )

            finite = np.all(
                np.isfinite(inv),
                axis=(1, 2),
            )

            ids = np.flatnonzero(
                finite
            )

            if ids.size:
                out[s + ids] = inv[ids]
                ok[s + ids] = True

        except np.linalg.LinAlgError:
            if e - s == 1:
                return

            m = (s + e) // 2

            solve(s, m)
            solve(m, e)

    solve(
        0,
        B,
    )

    return out, ok



def threshold_cholesky_inverse(
    Gamma,
    *,
    min_gamma_eig=1e-7,
):
    """
    Preserve the production EMI acceptance condition

        lambda_min(Gamma) > min_gamma_eig

    without computing the full Gamma eigensystem.

    For Hermitian Gamma:

        Gamma - tau I is positive definite

    iff

        lambda_min(Gamma) > tau.

    A successful Cholesky of Gamma - tau*I therefore
    reproduces the mathematical production gate.

    After the gate is passed, invert the ORIGINAL Gamma
    using Cholesky solves.
    """

    Gamma = np.asarray(
        Gamma,
        dtype=np.float64,
    )

    B, N, _ = Gamma.shape

    inv = np.zeros_like(
        Gamma,
        dtype=np.float64,
    )

    ok = np.zeros(
        B,
        dtype=np.bool_,
    )

    eye = np.eye(
        N,
        dtype=np.float64,
    )

    shifted = (
        Gamma
        -
        float(min_gamma_eig)
        *
        eye[
            None,
            :,
            :,
        ]
    )

    # --------------------------------------------------------
    # Recursive isolation keeps one non-PD matrix from
    # rejecting the whole NumPy batch.
    # --------------------------------------------------------

    def solve_range(
        start,
        stop,
    ):

        if start >= stop:
            return

        Gs = shifted[
            start:stop
        ]

        try:
            # Exact threshold gate.
            np.linalg.cholesky(
                Gs
            )

        except np.linalg.LinAlgError:

            if stop - start == 1:
                return

            middle = (
                start + stop
            ) // 2

            solve_range(
                start,
                middle,
            )

            solve_range(
                middle,
                stop,
            )

            return

        # ----------------------------------------------------
        # Every matrix in this range passed the threshold gate.
        # Invert ORIGINAL Gamma, not shifted Gamma.
        # ----------------------------------------------------

        G = Gamma[
            start:stop
        ]

        try:

            L = np.linalg.cholesky(
                G
            )

            rhs = np.broadcast_to(
                eye,
                (
                    stop - start,
                    N,
                    N,
                ),
            )

            y = np.linalg.solve(
                L,
                rhs,
            )

            cur_inv = np.linalg.solve(
                np.swapaxes(
                    L,
                    -1,
                    -2,
                ),
                y,
            )

            finite = np.all(
                np.isfinite(
                    cur_inv
                ),
                axis=(1, 2),
            )

            ids = np.flatnonzero(
                finite
            )

            if ids.size:

                inv[
                    start + ids
                ] = cur_inv[
                    ids
                ]

                ok[
                    start + ids
                ] = True

        except np.linalg.LinAlgError:

            # Extremely rare numerical inconsistency.
            if stop - start == 1:
                return

            middle = (
                start + stop
            ) // 2

            solve_range(
                start,
                middle,
            )

            solve_range(
                middle,
                stop,
            )

    solve_range(
        0,
        B,
    )

    return (
        inv,
        ok,
    )


def guarded_cholesky_inverse(
    Gamma,
    gamma_ok,
):
    """
    Exact current gamma-min acceptance mask followed by
    Cholesky inverse.
    """

    B, N, _ = Gamma.shape

    out = np.zeros_like(
        Gamma,
        dtype=np.float64,
    )

    ok = np.zeros(
        B,
        dtype=bool,
    )

    ids = np.flatnonzero(
        gamma_ok
    )

    if ids.size == 0:
        return out, ok

    G = Gamma[ids]

    eye = np.eye(
        N,
        dtype=np.float64,
    )

    rhs = np.broadcast_to(
        eye,
        (
            ids.size,
            N,
            N,
        ),
    )

    try:
        L = np.linalg.cholesky(
            G
        )

        y = np.linalg.solve(
            L,
            rhs,
        )

        inv = np.linalg.solve(
            np.swapaxes(
                L,
                -1,
                -2,
            ),
            y,
        )

        finite = np.all(
            np.isfinite(inv),
            axis=(1, 2),
        )

        good_ids = ids[
            finite
        ]

        if good_ids.size:
            out[good_ids] = inv[
                finite
            ]

            ok[good_ids] = True

    except np.linalg.LinAlgError:
        # Rare path: preserve robustness.
        sub, sub_ok = (
            recursive_cholesky_inverse(
                G
            )
        )

        good_ids = ids[
            sub_ok
        ]

        if good_ids.size:
            out[
                good_ids
            ] = sub[
                sub_ok
            ]

            ok[
                good_ids
            ] = True

    return out, ok


def emi_batch_backend(
    coh,
    *,
    n_images,
    pairs,
    backend,
    beta=0.0,
    gamma_jitter=1e-6,
    emi_mu=0.99,
    reference_idx=0,
    min_gamma_eig=1e-7,
):
    """
    Alternative Gamma-inverse routes.

    Everything after Gamma^-1 is deliberately identical
    in mathematical definition.
    """

    C = uncompress_coherence(
        coh,
        n_images,
        pairs,
    ).astype(
        np.complex128,
        copy=False,
    )

    B = C.shape[0]

    eye = np.eye(
        n_images,
        dtype=np.float64,
    )

    Gamma = np.abs(
        C
    ).real

    if beta > 0:
        Gamma = (
            (1.0 - beta)
            *
            Gamma
            +
            beta
            *
            eye[None, :, :]
        )

    Gamma = (
        Gamma
        +
        gamma_jitter
        *
        eye[None, :, :]
    )

    Gamma = 0.5 * (
        Gamma
        +
        np.swapaxes(
            Gamma,
            -1,
            -2,
        )
    )

    gamma_min = np.full(
        B,
        np.nan,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Backend-specific Gamma inverse.
    # ---------------------------------------------------------

    if backend in {
        "eigvalsh_inv",
        "eigvalsh_cholesky",
    }:
        gw = np.linalg.eigvalsh(
            Gamma
        )

        gamma_min = gw[:, 0]

        gamma_ok = (
            np.all(
                np.isfinite(gw),
                axis=1,
            )
            &
            (
                gamma_min
                >
                min_gamma_eig
            )
        )

        if backend == "eigvalsh_inv":
            Gamma_inv = np.zeros_like(
                Gamma
            )

            inv_ok = np.zeros(
                B,
                dtype=bool,
            )

            ids = np.flatnonzero(
                gamma_ok
            )

            if ids.size:
                try:
                    inv = np.linalg.inv(
                        Gamma[ids]
                    )

                    finite = np.all(
                        np.isfinite(inv),
                        axis=(1, 2),
                    )

                    good_ids = ids[
                        finite
                    ]

                    Gamma_inv[
                        good_ids
                    ] = inv[
                        finite
                    ]

                    inv_ok[
                        good_ids
                    ] = True

                except np.linalg.LinAlgError:
                    sub, sub_ok = (
                        recursive_inverse(
                            Gamma[ids]
                        )
                    )

                    good_ids = ids[
                        sub_ok
                    ]

                    Gamma_inv[
                        good_ids
                    ] = sub[
                        sub_ok
                    ]

                    inv_ok[
                        good_ids
                    ] = True

        else:
            (
                Gamma_inv,
                inv_ok,
            ) = guarded_cholesky_inverse(
                Gamma,
                gamma_ok,
            )

    elif backend == "threshold_cholesky":

        (
            Gamma_inv,
            inv_ok,
        ) = threshold_cholesky_inverse(
            Gamma,
            min_gamma_eig=min_gamma_eig,
        )

    elif backend == "fast_cholesky":

        (
            Gamma_inv,
            inv_ok,
        ) = recursive_cholesky_inverse(
            Gamma
        )

    elif backend == "fast_inverse":

        (
            Gamma_inv,
            inv_ok,
        ) = recursive_inverse(
            Gamma
        )

    else:
        raise ValueError(
            backend
        )

    phase = np.full(
        (
            B,
            n_images,
        ),
        np.nan
        +
        1j * np.nan,
        dtype=np.complex64,
    )

    estimator = np.full(
        B,
        ESTIMATOR_INVALID,
        dtype=np.uint8,
    )

    # ---------------------------------------------------------
    # EMI only for inverse-valid points.
    # ---------------------------------------------------------

    ids = np.flatnonzero(
        inv_ok
    )

    emi_ok = np.zeros(
        B,
        dtype=bool,
    )

    if ids.size:

        A = (
            Gamma_inv[ids]
            *
            C[ids]
        )

        A = 0.5 * (
            A
            +
            np.swapaxes(
                A.conj(),
                -1,
                -2,
            )
        )

        ew, ev = np.linalg.eigh(
            A
        )

        idx = np.argmin(
            np.abs(
                ew.real
                -
                emi_mu
            ),
            axis=1,
        )

        vec = take_eigvec(
            ev,
            idx,
        )

        val = ew[
            np.arange(
                ids.size
            ),
            idx,
        ].real

        finite = (
            np.isfinite(val)
            &
            np.all(
                np.isfinite(
                    vec.real
                )
                &
                np.isfinite(
                    vec.imag
                ),
                axis=1,
            )
        )

        good_ids = ids[
            finite
        ]

        if good_ids.size:

            v = vec[
                finite
            ]

            norm = np.linalg.norm(
                v,
                axis=1,
                keepdims=True,
            )

            norm = np.where(
                norm > 0,
                norm,
                1.0,
            )

            v = (
                np.sqrt(
                    n_images
                )
                *
                v
                /
                norm
            )

            phase[
                good_ids
            ] = reference_unit_phase(
                v,
                reference_idx,
            )

            estimator[
                good_ids
            ] = ESTIMATOR_EMI

            emi_ok[
                good_ids
            ] = True

    # ---------------------------------------------------------
    # Same lazy EVD fallback definition.
    # ---------------------------------------------------------

    bad = ~emi_ok

    if np.any(bad):

        Cb = C[
            bad
        ]

        E = (
            Cb
            *
            np.abs(
                Cb
            )
        )

        E = 0.5 * (
            E
            +
            np.swapaxes(
                E.conj(),
                -1,
                -2,
            )
        )

        bw, bv = np.linalg.eigh(
            E
        )

        idx = np.argmax(
            bw.real,
            axis=1,
        )

        vec = take_eigvec(
            bv,
            idx,
        )

        val = bw[
            np.arange(
                bw.shape[0]
            ),
            idx,
        ].real

        finite = (
            np.isfinite(val)
            &
            np.all(
                np.isfinite(
                    vec.real
                )
                &
                np.isfinite(
                    vec.imag
                ),
                axis=1,
            )
        )

        bad_ids = np.flatnonzero(
            bad
        )

        good_ids = bad_ids[
            finite
        ]

        if good_ids.size:

            phase[
                good_ids
            ] = reference_unit_phase(
                vec[
                    finite
                ],
                reference_idx,
            )

            estimator[
                good_ids
            ] = ESTIMATOR_EVD

    return (
        phase,
        estimator,
    )


def threaded_custom(
    coh,
    *,
    n_images,
    pairs,
    backend,
    workers,
    chunk_size,
):
    B = coh.shape[0]

    phase = np.empty(
        (
            B,
            n_images,
        ),
        dtype=np.complex64,
    )

    estimator = np.empty(
        B,
        dtype=np.uint8,
    )

    ranges = [
        (
            s,
            min(
                B,
                s + chunk_size,
            ),
        )
        for s in range(
            0,
            B,
            chunk_size,
        )
    ]

    def work(s, e):

        result = emi_batch_backend(
            coh[
                s:e
            ],
            n_images=n_images,
            pairs=pairs,
            backend=backend,
        )

        return (
            s,
            e,
            result,
        )

    ctx = (
        threadpool_limits(
            limits=1
        )
        if threadpool_limits
        is not None
        else nullcontext()
    )

    with ctx:

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=(
                "pypsds-emi-bench"
            ),
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

                (
                    s,
                    e,
                    result,
                ) = fut.result()

                ph, est = result

                phase[
                    s:e
                ] = ph

                estimator[
                    s:e
                ] = est

    return (
        phase,
        estimator,
    )


def compare(
    ref_phase,
    ref_est,
    phase,
    est,
):
    estimator_bad = int(
        np.count_nonzero(
            ref_est
            !=
            est
        )
    )

    same_finite = (
        np.isfinite(
            ref_phase.real
        )
        &
        np.isfinite(
            ref_phase.imag
        )
        &
        np.isfinite(
            phase.real
        )
        &
        np.isfinite(
            phase.imag
        )
    )

    if np.any(
        same_finite
    ):
        phase_max = float(
            np.max(
                np.abs(
                    ref_phase[
                        same_finite
                    ]
                    -
                    phase[
                        same_finite
                    ]
                )
            )
        )
    else:
        phase_max = 0.0

    return {
        "estimator_bad":
            estimator_bad,

        "max_phase_difference":
            phase_max,

        "parity":
            (
                estimator_bad == 0
                and
                phase_max <= 5e-6
            ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--sample",
        type=int,
        default=16000,
    )

    ap.add_argument(
        "--repeat",
        type=int,
        default=2,
    )


    ap.add_argument(
        "--solver-size",
        type=int,
        default=19,
        help=(
            "PL solver dimension. "
            "Use 19 for stage-0 and 20 for stage-1. "
            "GLRT still uses the full acquisition count."
        ),
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (_, _, H, W),
    ) = open_from_config(
        args.config
    )

    ndate = len(
        stack.dates
    )


    solver_n = int(
        args.solver_size
    )

    if not (
        2
        <=
        solver_n
        <=
        ndate
    ):
        raise ValueError(
            f"solver-size={solver_n} outside [2,{ndate}]"
        )

    processing = (
        Path(
            paths.output_dir
        )
        /
        "processing"
    )

    stats = (
        processing
        /
        "ds_statistics"
    )

    # ---------------------------------------------------------
    # Read winner from U2.1.
    # ---------------------------------------------------------

    tune_path = (
        processing
        /
        "ds_tiled"
        /
        "pl_cpu_autotune.json"
    )

    tune = json.loads(
        tune_path.read_text(
            encoding="utf-8"
        )
    )

    workers = int(
        tune[
            "winner"
        ][
            "workers"
        ]
    )

    chunk = int(
        tune[
            "winner"
        ][
            "chunk"
        ]
    )

    # ---------------------------------------------------------
    # Build same real eligible sample.
    # ---------------------------------------------------------

    scale2 = np.load(
        stats
        /
        "rayleigh_scale2.npy",
        mmap_mode="r",
    )

    raw_valid = np.load(
        stats
        /
        "raw_valid.npy",
        mmap_mode="r",
    )

    ps_raw = np.load(
        stats
        /
        "ps_mask.npy",
        mmap_mode="r",
    )

    geom = np.load(
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    yxt = np.load(
        processing
        /
        "cache"
        /
        "phase_corrected_yxt.npy",
        mmap_mode="r",
    )

    valid = (
        np.asarray(
            raw_valid
        )
        &
        np.asarray(
            geom
        )
    )

    ps = (
        np.asarray(
            ps_raw
        )
        &
        valid
    )

    # Current production scientific center domain:
    # every geometry-valid non-PS pixel.
    center = (
        valid
        &
        ~ps
    )

    rr, cc = np.where(
        center
    )

    pairs = image_pairs(
        solver_n
    )

    pi = pairs[:, 0]
    pj = pairs[:, 1]

    selected_r = []
    selected_c = []
    selected_s = []

    found = 0
    pos = 0
    scan = 20000

    while (
        pos < rr.size
        and
        found < args.sample
    ):

        e = min(
            rr.size,
            pos + scan,
        )

        br = rr[
            pos:e
        ].astype(
            np.int32,
            copy=False,
        )

        bc = cc[
            pos:e
        ].astype(
            np.int32,
            copy=False,
        )

        support = make_support_batch(
            scale2,
            valid,
            ps,
            br,
            bc,
            half_row=5,
            half_col=11,
            alpha=0.005,
            ndate=ndate,
        )

        K = np.sum(
            support,
            axis=(1, 2),
        )

        good = K >= 48

        if np.any(good):
            selected_r.append(
                br[good]
            )

            selected_c.append(
                bc[good]
            )

            selected_s.append(
                support[good]
            )

            found += int(
                np.count_nonzero(
                    good
                )
            )

        pos = e

    gr = np.concatenate(
        selected_r
    )[:args.sample]

    gc = np.concatenate(
        selected_c
    )[:args.sample]

    gs = np.concatenate(
        selected_s,
        axis=0,
    )[:args.sample]

    print("=" * 92)
    print("pyPSDS-GAMMA U2.2 EMI backend benchmark")
    print("=" * 92)

    print(
        "dates            :",
        ndate,
    )


    print(
        "solver size      :",
        solver_n,
    )

    print(
        "pairs            :",
        pairs.shape[0],
    )

    print(
        "sample           :",
        gr.size,
    )

    print(
        "workers          :",
        workers,
    )

    print(
        "chunk            :",
        chunk,
    )

    print()
    print(
        "Building real coherence sample..."
    )

    ts = time.perf_counter()

    solver_yxt = (
        yxt[
            :,
            :,
            :solver_n,
        ]
    )

    coh = compressed_coherence(
        solver_yxt,
        gr,
        gc,
        gs,
        pi,
        pj,
    )

    print(
        "coherence        :",
        f"{time.perf_counter()-ts:.3f} s",
    )

    # ---------------------------------------------------------
    # Numerical reference.
    # ---------------------------------------------------------

    reference = robust_emi_threaded(
        coh,
        n_images=solver_n,
        pairs=pairs,
        beta=0.0,
        gamma_jitter=1e-6,
        emi_mu=0.99,
        reference_idx=0,
        workers=workers,
        chunk_size=chunk,
    )

    ref_phase = reference[0]
    ref_est = reference[1]

    backends = [
        "current_eigh",
        "eigvalsh_inv",
        "eigvalsh_cholesky",
        "threshold_cholesky",
        "fast_cholesky",
        "fast_inverse",
    ]

    results = []

    print()
    print(
        f"{'backend':22s}"
        f"{'best_s':>10s}"
        f"{'pts/s':>12s}"
        f"{'est_bad':>10s}"
        f"{'phase_diff':>14s}"
        f"{'parity':>10s}"
    )

    print(
        "-" * 78
    )

    for backend in backends:

        timings = []
        last = None

        # Warm one call before measurement.
        if backend == "current_eigh":

            _ = robust_emi_threaded(
                coh,
                n_images=solver_n,
                pairs=pairs,
                beta=0.0,
                gamma_jitter=1e-6,
                emi_mu=0.99,
                reference_idx=0,
                workers=workers,
                chunk_size=chunk,
            )

        else:

            _ = threaded_custom(
                coh,
                n_images=solver_n,
                pairs=pairs,
                backend=backend,
                workers=workers,
                chunk_size=chunk,
            )

        for _ in range(
            max(
                1,
                args.repeat,
            )
        ):

            ts = time.perf_counter()

            if backend == "current_eigh":

                x = robust_emi_threaded(
                    coh,
                    n_images=solver_n,
                    pairs=pairs,
                    beta=0.0,
                    gamma_jitter=1e-6,
                    emi_mu=0.99,
                    reference_idx=0,
                    workers=workers,
                    chunk_size=chunk,
                )

                last = (
                    x[0],
                    x[1],
                )

            else:

                last = threaded_custom(
                    coh,
                    n_images=solver_n,
                    pairs=pairs,
                    backend=backend,
                    workers=workers,
                    chunk_size=chunk,
                )

            timings.append(
                time.perf_counter()
                -
                ts
            )

        best = min(
            timings
        )

        phase, est = last

        q = compare(
            ref_phase,
            ref_est,
            phase,
            est,
        )

        rate = (
            coh.shape[0]
            /
            best
        )

        result = {
            "backend":
                backend,

            "seconds":
                best,

            "points_per_second":
                rate,

            **q,
        }

        results.append(
            result
        )

        print(
            f"{backend:22s}"
            f"{best:10.3f}"
            f"{rate:12.1f}"
            f"{q['estimator_bad']:10d}"
            f"{q['max_phase_difference']:14.3e}"
            f"{str(q['parity']):>10s}"
        )

    valid_results = [
        x
        for x in results
        if x[
            "parity"
        ]
    ]

    winner = min(
        valid_results,
        key=lambda x:
            x[
                "seconds"
            ],
    )

    out = (
        processing
        /
        "ds_tiled"
        /
        "emi_backend_benchmark.json"
    )

    out.write_text(
        json.dumps(
            {
                "format":
                    "pyPSDS-GAMMA-emi-backend-benchmark-v1",

                "ndate":
                    ndate,


                "solver_size":
                    solver_n,

                "npair":
                    int(
                        pairs.shape[0]
                    ),

                "sample_points":
                    int(
                        coh.shape[0]
                    ),

                "workers":
                    workers,

                "chunk":
                    chunk,

                "results":
                    results,

                "winner":
                    winner,
            },
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print("U2.2 WINNER")
    print("=" * 92)

    for k, v in winner.items():
        print(
            f"{k:24s}:",
            v,
        )

    print(
        "saved                   :",
        out,
    )

    print()
    print(
        "U2.2 EMI BACKEND BENCHMARK: PASS"
    )


if __name__ == "__main__":
    main()
