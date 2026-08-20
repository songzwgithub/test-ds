#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.context import open_from_config
from pypsds.phase_linking.coherence import compressed_coherence
from pypsds.phase_linking.emi import (
    ESTIMATOR_INVALID,
    image_pairs,
    robust_emi_threaded,
    temporal_coherence,
)
from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


def bool_windows(x, hr, hc):
    x = np.asarray(x, dtype=np.bool_)

    p = np.pad(
        x,
        ((hr, hr), (hc, hc)),
        mode="constant",
        constant_values=False,
    )

    return np.lib.stride_tricks.sliding_window_view(
        p,
        (2 * hr + 1, 2 * hc + 1),
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--pl-workers",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--pl-chunk",
        type=int,
        default=512,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (row0, col0, H, W),
    ) = open_from_config(
        args.config
    )

    processing = (
        Path(paths.output_dir)
        / "processing"
    )

    seqdir = (
        processing
        / "sequential"
    )

    # ------------------------------------------------------------
    # Fixed M16 scientific evaluation population.
    #
    # Do not use generic u34a_rows, since those files may have
    # subsequently been overwritten by another ministack test.
    # ------------------------------------------------------------

    z = np.load(
        seqdir
        / "u34c_M16_phase_metrics.npz"
    )

    rr = z["rows"].astype(
        np.int32
    )

    cc = z["cols"].astype(
        np.int32
    )

    if (
        rr.ndim != 1
        or
        cc.ndim != 1
        or
        rr.size != cc.size
    ):
        raise RuntimeError(
            "Invalid M16 audit coordinates"
        )

    N = rr.size

    print("=" * 110)
    print("U3.4m-1 BETA0 FULL38 MATCHED-SUPPORT REFERENCE")
    print("=" * 110)

    print(
        "centers       :",
        f"{N:,}",
    )

    print(
        "scene         :",
        f"{H} x {W}",
    )

    print(
        "solver        :",
        "robust EMI, beta=0, mu=0.99",
    )

    # ------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------

    yxt = np.load(
        processing
        / "cache"
        / "phase_corrected_yxt.npy",
        mmap_mode="r",
    )

    scale2 = np.load(
        processing
        / "ds_statistics"
        / "rayleigh_scale2.npy",
        mmap_mode="r",
    )

    raw_valid = np.load(
        processing
        / "ds_statistics"
        / "raw_valid.npy",
        mmap_mode="r",
    )

    geom = np.load(
        processing
        / "cache"
        / "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    ps = np.load(
        processing
        / "ps_mask.npy",
        mmap_mode="r",
    )

    core = np.load(
        seqdir
        / "compression_state_core_K24.npy",
        mmap_mode="r",
    )

    valid = (
        np.asarray(
            raw_valid,
            dtype=np.bool_,
        )
        &
        np.asarray(
            geom,
            dtype=np.bool_,
        )
    )

    ctx = prepare_glrt_window_context(
        scale2,
        valid,
        np.asarray(
            ps,
            dtype=np.bool_,
        ),
        half_row=5,
        half_col=11,
    )

    core_windows = bool_windows(
        np.asarray(
            core,
            dtype=np.bool_,
        ),
        5,
        11,
    )

    pairs = image_pairs(38)

    pi = pairs[:, 0]
    pj = pairs[:, 1]

    # ------------------------------------------------------------
    # Dense-shaped audit references.
    #
    # Only the M16 evaluation centers are populated. This allows
    # the already-validated U3.4a indexing code to be reused
    # exactly without rebuilding its reconstruction logic.
    # ------------------------------------------------------------

    phase_path = (
        processing
        / "u34m_beta0_full_phase.npy"
    )

    tc_path = (
        processing
        / "u34m_beta0_temporal_coherence.npy"
    )

    phase = np.lib.format.open_memmap(
        phase_path,
        mode="w+",
        dtype=np.complex64,
        shape=(38, H, W),
    )

    phase[...] = (
        np.nan
        +
        1j * np.nan
    )

    tc_map = np.lib.format.open_memmap(
        tc_path,
        mode="w+",
        dtype=np.float32,
        shape=(H, W),
    )

    tc_map[...] = np.nan

    K_all = np.empty(
        N,
        dtype=np.int16,
    )

    estimator_all = np.empty(
        N,
        dtype=np.uint8,
    )

    tc_all = np.empty(
        N,
        dtype=np.float32,
    )

    t0 = perf_counter()

    invalid_total = 0

    for b0 in range(
        0,
        N,
        args.batch,
    ):

        b1 = min(
            N,
            b0 + args.batch,
        )

        br = rr[b0:b1]
        bc = cc[b0:b1]

        support, _ = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=0.005,
                nslc=38,
                block_size=args.support_block,
            )
        )

        support &= np.asarray(
            core_windows[
                br,
                bc,
            ],
            dtype=np.bool_,
        )

        K = np.sum(
            support,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

        K_all[b0:b1] = K

        if np.any(
            K < 48
        ):
            bad = np.flatnonzero(
                K < 48
            )

            raise RuntimeError(
                "M16 evaluation population contains "
                f"{bad.size} center(s) with K<48; "
                "the saved M16 population no longer matches "
                "the current K24 support."
            )

        coh = compressed_coherence(
            yxt,
            br,
            bc,
            support,
            pi,
            pj,
        )

        (
            ph,
            estimator,
            emi_eig,
            evd_eig,
            gamma_min,
        ) = robust_emi_threaded(
            coh,
            n_images=38,
            pairs=pairs,
            beta=0.0,
            gamma_jitter=1e-6,
            emi_mu=0.99,
            reference_idx=0,
            workers=args.pl_workers,
            chunk_size=args.pl_chunk,
        )

        bad = (
            estimator
            ==
            ESTIMATOR_INVALID
        )

        invalid_total += int(
            np.count_nonzero(
                bad
            )
        )

        if np.any(
            bad
        ):
            raise RuntimeError(
                f"beta0 full38 produced "
                f"{np.count_nonzero(bad)} invalid "
                f"solution(s) in batch {b0}:{b1}"
            )

        tc = temporal_coherence(
            coh,
            ph,
            pairs,
        )

        if not np.all(
            np.isfinite(tc)
        ):
            raise RuntimeError(
                "Non-finite beta0 full38 temporal coherence"
            )

        # npy phase convention is [date, row, col].
        phase[
            :,
            br,
            bc,
        ] = ph.T

        tc_map[
            br,
            bc,
        ] = tc

        estimator_all[
            b0:b1
        ] = estimator

        tc_all[
            b0:b1
        ] = tc

        elapsed = (
            perf_counter()
            -
            t0
        )

        print(
            f"{b1:7,d}/{N:7,d} "
            f"({100*b1/N:6.2f}%) "
            f"rate={b1/elapsed:,.0f} center/s "
            f"Kmin={int(K.min())} "
            f"Kmed={float(np.median(K)):.1f}"
        )

    phase.flush()
    tc_map.flush()

    meta_path = (
        seqdir
        / "u34m_beta0_full_reference_metrics.npz"
    )

    np.savez_compressed(
        meta_path,
        rows=rr,
        cols=cc,
        effective_K=K_all,
        estimator=estimator_all,
        temporal_coherence=tc_all,
    )

    print()
    print("=" * 110)
    print("U3.4m-1 RESULT")
    print("=" * 110)

    print(
        "centers       :",
        f"{N:,}",
    )

    print(
        "K min/median :",
        f"{int(K_all.min())} / "
        f"{float(np.median(K_all)):.1f}",
    )

    print(
        "invalid       :",
        f"{invalid_total:,}",
    )

    print(
        "TC median     :",
        f"{np.median(tc_all):.6f}",
    )

    print(
        "phase         :",
        phase_path,
    )

    print(
        "TC            :",
        tc_path,
    )

    print(
        "metrics       :",
        meta_path,
    )

    print()
    print(
        "U3.4m-1 BETA0 FULL38 REFERENCE: PASS"
    )


if __name__ == "__main__":
    main()
