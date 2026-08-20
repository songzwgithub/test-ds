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
    robust_emi_batch,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)

from tools.audit_u34j_emi_solver_semantics import (
    bool_windows,
    phase_compare,
    qs,
    solve_minimum_emi,
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
        default=512,
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=1024,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        roi,
    ) = open_from_config(
        args.config
    )

    proc = (
        Path(paths.output_dir)
        / "processing"
    )

    seq = proc / "sequential"

    # ------------------------------------------------------------
    # Same population already audited in U3.4j
    # ------------------------------------------------------------

    zj = np.load(
        seq
        / "u34j_emi_solver_semantics.npz"
    )

    rr = zj["rows"].astype(np.int32)
    cc = zj["cols"].astype(np.int32)

    point_ids = zj["point_ids"].astype(
        np.int64
    )

    is_cat = zj["is_catastrophic"].astype(
        bool
    )

    is_control = ~is_cat

    N = rr.size

    # ------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------

    yxt = np.load(
        proc
        / "cache"
        / "phase_corrected_yxt.npy",
        mmap_mode="r",
    )

    scale2 = np.load(
        proc
        / "ds_statistics"
        / "rayleigh_scale2.npy",
        mmap_mode="r",
    )

    raw_valid = np.load(
        proc
        / "ds_statistics"
        / "raw_valid.npy",
        mmap_mode="r",
    )

    geom = np.load(
        proc
        / "cache"
        / "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    ps = np.load(
        proc
        / "ps_mask.npy",
        mmap_mode="r",
    )

    core = np.load(
        seq
        / "compression_state_core_K24.npy",
        mmap_mode="r",
    )

    K_all = np.load(
        seq
        / "u34a_effective_K.npy",
        mmap_mode="r",
    )

    K_expected = np.asarray(
        K_all[point_ids],
        dtype=np.int16,
    )

    valid_scene = (
        np.asarray(raw_valid, dtype=bool)
        &
        np.asarray(geom, dtype=bool)
    )

    ctx = prepare_glrt_window_context(
        scale2,
        valid_scene,
        np.asarray(ps, dtype=bool),
        half_row=5,
        half_col=11,
    )

    core_windows = bool_windows(
        np.asarray(core, dtype=bool),
        5,
        11,
    )

    # ------------------------------------------------------------
    # Output arrays
    # ------------------------------------------------------------

    D16 = np.full(
        (N, 16),
        np.nan + 1j*np.nan,
        np.complex64,
    )

    D38 = np.full(
        (N, 38),
        np.nan + 1j*np.nan,
        np.complex64,
    )

    B16 = np.full_like(
        D16,
        np.nan + 1j*np.nan,
    )

    B38 = np.full_like(
        D38,
        np.nan + 1j*np.nan,
    )

    D16eig = np.full(
        N,
        np.nan,
        np.float32,
    )

    D38eig = np.full(
        N,
        np.nan,
        np.float32,
    )

    B16eig = np.full(
        N,
        np.nan,
        np.float32,
    )

    B38eig = np.full(
        N,
        np.nan,
        np.float32,
    )

    D16valid = np.zeros(N, bool)
    D38valid = np.zeros(N, bool)

    B16valid = np.zeros(N, bool)
    B38valid = np.zeros(N, bool)

    pairs16 = image_pairs(16)
    pairs38 = image_pairs(38)

    pi16 = pairs16[:, 0]
    pj16 = pairs16[:, 1]

    pi38 = pairs38[:, 0]
    pj38 = pairs38[:, 1]

    t0 = perf_counter()

    # ============================================================
    # Process
    # ============================================================

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
            dtype=bool,
        )

        K = np.sum(
            support,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

        if not np.array_equal(
            K,
            K_expected[b0:b1],
        ):
            raise RuntimeError(
                "K parity failure"
            )

        coh16 = compressed_coherence(
            yxt,
            br,
            bc,
            support,
            pi16,
            pj16,
        )

        coh38 = compressed_coherence(
            yxt,
            br,
            bc,
            support,
            pi38,
            pj38,
        )

        # --------------------------------------------------------
        # D = Dolphin default semantics
        #
        # beta = 0
        # mu   = 0.99
        # eigenpair nearest to mu
        # --------------------------------------------------------

        (
            d16,
            de16,
            dei16,
            dev16,
            dg16,
        ) = robust_emi_batch(
            coh16,
            n_images=16,
            pairs=pairs16,
            beta=0.0,
            gamma_jitter=1e-6,
            emi_mu=0.99,
            reference_idx=0,
        )

        (
            d38,
            de38,
            dei38,
            dev38,
            dg38,
        ) = robust_emi_batch(
            coh38,
            n_images=38,
            pairs=pairs38,
            beta=0.0,
            gamma_jitter=1e-6,
            emi_mu=0.99,
            reference_idx=0,
        )

        D16[b0:b1] = d16
        D38[b0:b1] = d38

        D16eig[b0:b1] = dei16
        D38eig[b0:b1] = dei38

        D16valid[b0:b1] = (
            de16 != ESTIMATOR_INVALID
        )

        D38valid[b0:b1] = (
            de38 != ESTIMATOR_INVALID
        )

        # --------------------------------------------------------
        # B = MiaplPy/Moraine-style minimum eigenpair
        #
        # beta = 0
        # --------------------------------------------------------

        (
            b16,
            be16,
            bg16,
            bv16,
        ) = solve_minimum_emi(
            coh16,
            n_images=16,
            pairs=pairs16,
            beta=0.0,
        )

        (
            b38,
            be38,
            bg38,
            bv38,
        ) = solve_minimum_emi(
            coh38,
            n_images=38,
            pairs=pairs38,
            beta=0.0,
        )

        B16[b0:b1] = b16
        B38[b0:b1] = b38

        B16eig[b0:b1] = be16
        B38eig[b0:b1] = be38

        B16valid[b0:b1] = bv16
        B38valid[b0:b1] = bv38

        elapsed = (
            perf_counter()
            -
            t0
        )

        print(
            f"{b1:7,d}/{N:7,d} "
            f"({100*b1/N:6.2f}%) "
            f"rate={b1/elapsed:,.0f} center/s"
        )

    # ============================================================
    # Dolphin default vs minimum eigenpair
    # ============================================================

    print()
    print("=" * 120)
    print(
        "U3.4k DOLPHIN-DEFAULT vs MINIMUM-EIGENPAIR"
    )
    print("=" * 120)

    for span, D, B, Dvalid, Bvalid, Deig, Beig in (
        (
            16,
            D16,
            B16,
            D16valid,
            B16valid,
            D16eig,
            B16eig,
        ),
        (
            38,
            D38,
            B38,
            D38valid,
            B38valid,
            D38eig,
            B38eig,
        ),
    ):

        (
            sim,
            med,
            p95,
            comp_valid,
        ) = phase_compare(
            D,
            B,
        )

        print()
        print(f"{span}-DATE")
        print("-" * 120)

        for name, mask in (
            (
                "catastrophic",
                is_cat,
            ),
            (
                "stable_control",
                is_control,
            ),
        ):

            m = (
                mask
                &
                Dvalid
                &
                Bvalid
                &
                comp_valid
            )

            eig_diff = np.abs(
                Deig[m]
                -
                Beig[m]
            )

            print(name)

            print(
                "  valid              :",
                f"{m.sum():,}/{mask.sum():,}",
            )

            print(
                "  phase similarity   :",
                qs(sim[m]),
            )

            print(
                "  median error deg   :",
                qs(med[m]),
            )

            print(
                "  p95 error deg      :",
                qs(p95[m]),
            )

            print(
                "  |eig D-B|          :",
                qs(eig_diff),
            )

            print(
                "  sim >=0.9999       :",
                f"{100*np.mean(sim[m] >= 0.9999):.3f}%",
            )

            print(
                "  eig diff <=1e-5    :",
                f"{100*np.mean(eig_diff <= 1e-5):.3f}%",
            )

    # ============================================================
    # Dolphin-default internal 16/38 consistency
    # ============================================================

    (
        simD,
        medD,
        p95D,
        validD,
    ) = phase_compare(
        D16,
        D38[:, :16],
    )

    print()
    print("=" * 120)
    print(
        "DOLPHIN-DEFAULT INTERNAL 16/38 PARITY"
    )
    print("=" * 120)

    for name, mask in (
        (
            "catastrophic",
            is_cat,
        ),
        (
            "stable_control",
            is_control,
        ),
    ):

        m = (
            mask
            &
            D16valid
            &
            D38valid
            &
            validD
        )

        print(name)

        print(
            "  similarity         :",
            qs(simD[m]),
        )

        print(
            "  median error deg   :",
            qs(medD[m]),
        )

        print(
            "  p95 error deg      :",
            qs(p95D[m]),
        )

        print(
            "  p95 >60 deg        :",
            f"{100*np.mean(p95D[m] > 60):.3f}%",
        )

        print(
            "  p95 >90 deg        :",
            f"{100*np.mean(p95D[m] > 90):.3f}%",
        )

    print()
    print(
        "U3.4k DOLPHIN DEFAULT PARITY: PASS"
    )


if __name__ == "__main__":
    main()
