#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.context import open_from_config

from pypsds.phase_linking.coherence import compressed_coherence

from pypsds.phase_linking.emi import (
    ESTIMATOR_EMI,
    ESTIMATOR_EVD,
    ESTIMATOR_INVALID,
    image_pairs,
    robust_emi_batch,
    temporal_coherence,
    uncompress_coherence,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


# =====================================================================
# Utilities
# =====================================================================

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


def qs(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    p = np.percentile(
        x,
        [5, 25, 50, 75, 95, 99],
    )

    return {
        "p05": float(p[0]),
        "p25": float(p[1]),
        "median": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
        "p99": float(p[5]),
    }


def take_eigvec(ev, idx):
    return np.take_along_axis(
        ev,
        idx[:, None, None],
        axis=2,
    )[:, :, 0]


def reference_unit(vec, reference_idx=0):
    ph = np.exp(
        1j * np.angle(vec)
    ).astype(np.complex64)

    ph *= np.conj(
        ph[:, reference_idx:reference_idx + 1]
    )

    return ph


def phase_compare(a, b):
    """
    Compare same-date phase vectors after removing
    their arbitrary common phase.
    """
    a = np.asarray(a, dtype=np.complex64)
    b = np.asarray(b, dtype=np.complex64)

    valid = (
        np.all(np.isfinite(a.real), axis=1)
        & np.all(np.isfinite(a.imag), axis=1)
        & np.all(np.isfinite(b.real), axis=1)
        & np.all(np.isfinite(b.imag), axis=1)
    )

    sim = np.full(a.shape[0], np.nan, np.float32)
    med = np.full(a.shape[0], np.nan, np.float32)
    p95 = np.full(a.shape[0], np.nan, np.float32)

    if not np.any(valid):
        return sim, med, p95, valid

    aa = a[valid].copy()
    bb = b[valid].copy()

    aa *= np.conj(aa[:, :1])
    bb *= np.conj(bb[:, :1])

    d = aa * np.conj(bb)

    sim[valid] = np.abs(
        np.mean(
            d,
            axis=1,
        )
    ).astype(np.float32)

    err = (
        np.abs(
            np.angle(d)
        )
        * 180.0
        / np.pi
    )

    med[valid] = np.median(
        err,
        axis=1,
    ).astype(np.float32)

    p95[valid] = np.percentile(
        err,
        95,
        axis=1,
    ).astype(np.float32)

    return sim, med, p95, valid


# =====================================================================
# Standard/minimum-eigenpair EMI
# =====================================================================

def solve_minimum_emi(
    coh,
    *,
    n_images,
    pairs,
    beta,
    gamma_jitter=1e-6,
    min_gamma_eig=1e-7,
):
    """
    Minimum-eigenpair EMI.

    beta=0:
        Dolphin/MiaplPy/Moraine-style EMI semantics,
        retaining only tiny numerical jitter.

    beta=0.05:
        control experiment isolating the effect of
        nearest-0.99 versus minimum-eigenpair selection.
    """

    C = uncompress_coherence(
        coh,
        n_images,
        pairs,
    ).astype(
        np.complex128,
        copy=False,
    )

    b = C.shape[0]

    eye = np.eye(
        n_images,
        dtype=np.float64,
    )

    Gamma = np.abs(C).real

    if beta > 0:
        Gamma = (
            (1.0 - beta) * Gamma
            + beta * eye[None, :, :]
        )

    # Numerical safeguard only.
    Gamma = (
        Gamma
        + gamma_jitter * eye[None, :, :]
    )

    Gamma = 0.5 * (
        Gamma
        + np.swapaxes(
            Gamma,
            -1,
            -2,
        )
    )

    gw, gv = np.linalg.eigh(
        Gamma
    )

    gamma_min = gw[:, 0].real

    gamma_ok = (
        np.all(np.isfinite(gw), axis=1)
        & (gamma_min > min_gamma_eig)
    )

    safe_w = np.where(
        gw > min_gamma_eig,
        gw,
        1.0,
    )

    Gamma_inv = np.einsum(
        "bik,bk,bjk->bij",
        gv,
        1.0 / safe_w,
        gv,
        optimize=True,
    )

    A = Gamma_inv * C

    A = 0.5 * (
        A
        + np.swapaxes(
            A.conj(),
            -1,
            -2,
        )
    )

    ew, ev = np.linalg.eigh(
        A
    )

    # ---------------------------------------------------------
    # Mature EMI semantics:
    # minimum eigenvalue, NOT nearest-to-0.99.
    # ---------------------------------------------------------
    idx = np.zeros(
        b,
        dtype=np.int64,
    )

    selected_val = ew[:, 0].real

    vec = take_eigvec(
        ev,
        idx,
    )

    good = (
        gamma_ok
        & np.isfinite(selected_val)
        & np.all(
            np.isfinite(vec.real)
            & np.isfinite(vec.imag),
            axis=1,
        )
    )

    phase = np.full(
        (b, n_images),
        np.nan + 1j * np.nan,
        dtype=np.complex64,
    )

    phase[good] = reference_unit(
        vec[good],
        0,
    )

    return (
        phase,
        selected_val.astype(np.float32),
        gamma_min.astype(np.float32),
        good,
    )


# =====================================================================
# Reporting
# =====================================================================

def group_summary(
    mask,
    *,
    valid,
    sim,
    med,
    p95,
    tc16,
    tc38,
    eig16,
    eig38,
):
    m = (
        np.asarray(mask, dtype=bool)
        & np.asarray(valid, dtype=bool)
    )

    if not np.any(m):
        return {
            "n_total": int(np.count_nonzero(mask)),
            "n_valid": 0,
        }

    return {
        "n_total":
            int(np.count_nonzero(mask)),

        "n_valid":
            int(np.count_nonzero(m)),

        "valid_fraction":
            float(
                np.count_nonzero(m)
                /
                np.count_nonzero(mask)
            ),

        "similarity_16_vs_38":
            qs(sim[m]),

        "median_error_deg_16_vs_38":
            qs(med[m]),

        "p95_error_deg_16_vs_38":
            qs(p95[m]),

        "tc16":
            qs(tc16[m]),

        "tc38":
            qs(tc38[m]),

        "selected_eigenvalue_16":
            qs(eig16[m]),

        "selected_eigenvalue_38":
            qs(eig38[m]),

        "similarity_ge_0p99_fraction":
            float(
                np.mean(
                    sim[m] >= 0.99
                )
            ),

        "median_error_le_10_fraction":
            float(
                np.mean(
                    med[m] <= 10.0
                )
            ),

        "p95_error_gt60_fraction":
            float(
                np.mean(
                    p95[m] > 60.0
                )
            ),

        "p95_error_gt90_fraction":
            float(
                np.mean(
                    p95[m] > 90.0
                )
            ),
    }


def print_group(name, x):
    print()
    print(name)
    print("-" * 115)

    print(
        "valid                 :",
        f"{x['n_valid']:,}/{x['n_total']:,}",
    )

    if x["n_valid"] == 0:
        return

    print(
        "similarity 16/38      :",
        x["similarity_16_vs_38"],
    )

    print(
        "median error deg      :",
        x["median_error_deg_16_vs_38"],
    )

    print(
        "p95 error deg         :",
        x["p95_error_deg_16_vs_38"],
    )

    print(
        "TC16                  :",
        x["tc16"],
    )

    print(
        "TC38                  :",
        x["tc38"],
    )

    print(
        "eigenvalue16          :",
        x["selected_eigenvalue_16"],
    )

    print(
        "eigenvalue38          :",
        x["selected_eigenvalue_38"],
    )

    print(
        "similarity >=0.99     :",
        f"{100*x['similarity_ge_0p99_fraction']:.3f}%",
    )

    print(
        "median error <=10 deg :",
        f"{100*x['median_error_le_10_fraction']:.3f}%",
    )

    print(
        "p95 >60 deg           :",
        f"{100*x['p95_error_gt60_fraction']:.3f}%",
    )

    print(
        "p95 >90 deg           :",
        f"{100*x['p95_error_gt90_fraction']:.3f}%",
    )


# =====================================================================
# Main
# =====================================================================

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

    # -----------------------------------------------------------------
    # U3.4i population
    # -----------------------------------------------------------------

    zi = np.load(
        seqdir
        / "u34i_stage0_eigen_branch.npz"
    )

    point_ids = zi[
        "point_ids"
    ].astype(
        np.int64
    )

    is_cat = zi[
        "is_catastrophic"
    ].astype(bool)

    if (
        point_ids.size != 24458
        or
        is_cat.size != point_ids.size
    ):
        raise RuntimeError(
            "Unexpected U3.4i population"
        )

    # -----------------------------------------------------------------
    # Recover coordinates from M16 parity population.
    # point_ids are indices into this population.
    # -----------------------------------------------------------------

    zm = np.load(
        seqdir
        / "u34c_M16_phase_metrics.npz"
    )

    rr_all = zm["rows"]
    cc_all = zm["cols"]

    if np.any(
        point_ids < 0
    ) or np.any(
        point_ids >= rr_all.size
    ):
        raise RuntimeError(
            "U3.4i point_ids outside U3.4c population"
        )

    rr = rr_all[
        point_ids
    ].astype(
        np.int32
    )

    cc = cc_all[
        point_ids
    ].astype(
        np.int32
    )

    is_control = ~is_cat

    print("=" * 125)
    print("U3.4j EMI SOLVER SEMANTICS PARITY")
    print("=" * 125)

    print(
        "population       :",
        f"{point_ids.size:,}",
    )

    print(
        "catastrophic     :",
        f"{is_cat.sum():,}",
    )

    print(
        "stable controls  :",
        f"{is_control.sum():,}",
    )

    print()

    print("Variants:")
    print(
        " A_CURRENT      beta=.05, nearest eigenvalue to 0.99"
    )
    print(
        " B_STANDARD     beta=0, minimum eigenpair"
    )
    print(
        " C_MIN_BETA05   beta=.05, minimum eigenpair"
    )

    # -----------------------------------------------------------------
    # Existing stage0 phase for exact A-current parity check.
    # -----------------------------------------------------------------

    seq_phase = np.load(
        seqdir
        / "u34a_sequential_phase_points.npy",
        mmap_mode="r",
    )

    actual_stage0 = np.asarray(
        seq_phase[
            point_ids,
            :16,
        ],
        dtype=np.complex64,
    )

    # -----------------------------------------------------------------
    # Input data / exact frozen support.
    # -----------------------------------------------------------------

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

    K_expected_all = np.load(
        seqdir
        / "u34a_effective_K.npy",
        mmap_mode="r",
    )

    K_expected = np.asarray(
        K_expected_all[
            point_ids
        ],
        dtype=np.int16,
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

    pairs16 = image_pairs(16)
    pairs38 = image_pairs(38)

    pi16 = pairs16[:, 0]
    pj16 = pairs16[:, 1]

    pi38 = pairs38[:, 0]
    pj38 = pairs38[:, 1]

    N = point_ids.size

    variant_names = (
        "A_CURRENT",
        "B_STANDARD",
        "C_MIN_BETA05",
    )

    result = {}

    for name in variant_names:

        result[name] = {
            "phase16":
                np.full(
                    (N, 16),
                    np.nan + 1j*np.nan,
                    np.complex64,
                ),

            "phase38":
                np.full(
                    (N, 38),
                    np.nan + 1j*np.nan,
                    np.complex64,
                ),

            "tc16":
                np.full(
                    N,
                    np.nan,
                    np.float32,
                ),

            "tc38":
                np.full(
                    N,
                    np.nan,
                    np.float32,
                ),

            "eig16":
                np.full(
                    N,
                    np.nan,
                    np.float32,
                ),

            "eig38":
                np.full(
                    N,
                    np.nan,
                    np.float32,
                ),

            "valid16":
                np.zeros(
                    N,
                    dtype=bool,
                ),

            "valid38":
                np.zeros(
                    N,
                    dtype=bool,
                ),
        }

    t0 = perf_counter()

    # =================================================================
    # Batch processing
    # =================================================================

    for b0 in range(
        0,
        N,
        args.batch,
    ):

        b1 = min(
            N,
            b0 + args.batch,
        )

        br = rr[
            b0:b1
        ]

        bc = cc[
            b0:b1
        ]

        # -------------------------------------------------------------
        # Exact same 38-date GLRT + K24 support used in U3.4i/g.
        # -------------------------------------------------------------

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

        if not np.array_equal(
            K,
            K_expected[
                b0:b1
            ],
        ):
            bad = int(
                np.flatnonzero(
                    K
                    !=
                    K_expected[b0:b1]
                )[0]
            )

            raise RuntimeError(
                "K parity failure at "
                f"({int(br[bad])},{int(bc[bad])}) "
                f"got={int(K[bad])} "
                f"expected={int(K_expected[b0+bad])}"
            )

        # -------------------------------------------------------------
        # SAME spatial support.
        #
        # One covariance uses dates 0:16.
        # One uses all dates 0:38.
        # -------------------------------------------------------------

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

        # =============================================================
        # A_CURRENT
        # =============================================================

        (
            ph16_A,
            est16_A,
            emi16_A,
            evd16_A,
            gm16_A,
        ) = robust_emi_batch(
            coh16,
            n_images=16,
            pairs=pairs16,
            beta=0.05,
            gamma_jitter=1e-6,
            emi_mu=0.99,
            reference_idx=0,
        )

        (
            ph38_A,
            est38_A,
            emi38_A,
            evd38_A,
            gm38_A,
        ) = robust_emi_batch(
            coh38,
            n_images=38,
            pairs=pairs38,
            beta=0.05,
            gamma_jitter=1e-6,
            emi_mu=0.99,
            reference_idx=0,
        )

        valid16_A = (
            est16_A
            !=
            ESTIMATOR_INVALID
        )

        valid38_A = (
            est38_A
            !=
            ESTIMATOR_INVALID
        )

        eig16_A = np.where(
            est16_A == ESTIMATOR_EMI,
            emi16_A,
            evd16_A,
        )

        eig38_A = np.where(
            est38_A == ESTIMATOR_EMI,
            emi38_A,
            evd38_A,
        )

        # =============================================================
        # B_STANDARD
        # beta=0 + minimum eigenpair
        # =============================================================

        (
            ph16_B,
            eig16_B,
            gm16_B,
            valid16_B,
        ) = solve_minimum_emi(
            coh16,
            n_images=16,
            pairs=pairs16,
            beta=0.0,
        )

        (
            ph38_B,
            eig38_B,
            gm38_B,
            valid38_B,
        ) = solve_minimum_emi(
            coh38,
            n_images=38,
            pairs=pairs38,
            beta=0.0,
        )

        # =============================================================
        # C_MIN_BETA05
        # beta=.05 + minimum eigenpair
        # =============================================================

        (
            ph16_C,
            eig16_C,
            gm16_C,
            valid16_C,
        ) = solve_minimum_emi(
            coh16,
            n_images=16,
            pairs=pairs16,
            beta=0.05,
        )

        (
            ph38_C,
            eig38_C,
            gm38_C,
            valid38_C,
        ) = solve_minimum_emi(
            coh38,
            n_images=38,
            pairs=pairs38,
            beta=0.05,
        )

        bundle = {
            "A_CURRENT": (
                ph16_A,
                ph38_A,
                eig16_A,
                eig38_A,
                valid16_A,
                valid38_A,
            ),

            "B_STANDARD": (
                ph16_B,
                ph38_B,
                eig16_B,
                eig38_B,
                valid16_B,
                valid38_B,
            ),

            "C_MIN_BETA05": (
                ph16_C,
                ph38_C,
                eig16_C,
                eig38_C,
                valid16_C,
                valid38_C,
            ),
        }

        for name, (
            ph16,
            ph38,
            eig16,
            eig38,
            v16,
            v38,
        ) in bundle.items():

            r = result[name]

            r["phase16"][
                b0:b1
            ] = ph16

            r["phase38"][
                b0:b1
            ] = ph38

            r["eig16"][
                b0:b1
            ] = eig16

            r["eig38"][
                b0:b1
            ] = eig38

            r["valid16"][
                b0:b1
            ] = v16

            r["valid38"][
                b0:b1
            ] = v38

            r["tc16"][
                b0:b1
            ] = temporal_coherence(
                coh16,
                ph16,
                pairs16,
            )

            r["tc38"][
                b0:b1
            ] = temporal_coherence(
                coh38,
                ph38,
                pairs38,
            )

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

    # =================================================================
    # Mandatory A-current parity with actual M16 stage0
    # =================================================================

    (
        parity_sim,
        parity_med,
        parity_p95,
        parity_valid,
    ) = phase_compare(
        result["A_CURRENT"]["phase16"],
        actual_stage0,
    )

    print()
    print("=" * 125)
    print("CURRENT SOLVER STAGE0 RECONSTRUCTION PARITY")
    print("=" * 125)

    print(
        "similarity:",
        qs(
            parity_sim[
                parity_valid
            ]
        ),
    )

    print(
        "median err:",
        qs(
            parity_med[
                parity_valid
            ]
        ),
    )

    print(
        "p95 err   :",
        qs(
            parity_p95[
                parity_valid
            ]
        ),
    )

    if (
        np.nanmin(
            parity_sim[
                parity_valid
            ]
        )
        <
        0.9999
    ):
        raise RuntimeError(
            "A_CURRENT stage0 parity failed"
        )

    # =================================================================
    # Variant-internal 16 versus 38 parity
    # =================================================================

    report = {
        "format":
            "pyPSDS-GAMMA-U3.4j-EMI-solver-semantics-v1",

        "population":
            int(N),

        "catastrophic":
            int(is_cat.sum()),

        "stable_control":
            int(is_control.sum()),

        "variants": {},
    }

    print()
    print("=" * 125)
    print("U3.4j MATCHED 16-DATE vs 38-DATE EMI PARITY")
    print("=" * 125)

    print(
        "variant           group          "
        "sim50   sim05   medErr50 medErr95 "
        "p95Err50 p95Err95   TC16_50 TC38_50"
    )

    print("-" * 125)

    for name in variant_names:

        r = result[name]

        (
            sim,
            med,
            p95,
            compare_valid,
        ) = phase_compare(
            r["phase16"],
            r["phase38"][:, :16],
        )

        valid = (
            compare_valid
            &
            r["valid16"]
            &
            r["valid38"]
        )

        r["sim"] = sim
        r["med"] = med
        r["p95"] = p95
        r["compare_valid"] = valid

        variant_report = {}

        for group_name, mask in (
            ("catastrophic", is_cat),
            ("stable_control", is_control),
        ):

            s = group_summary(
                mask,
                valid=valid,
                sim=sim,
                med=med,
                p95=p95,
                tc16=r["tc16"],
                tc38=r["tc38"],
                eig16=r["eig16"],
                eig38=r["eig38"],
            )

            variant_report[
                group_name
            ] = s

            m = (
                mask
                &
                valid
            )

            if np.any(m):
                print(
                    f"{name:<17s} "
                    f"{group_name:<14s} "
                    f"{np.median(sim[m]):7.4f} "
                    f"{np.percentile(sim[m],5):7.4f} "
                    f"{np.median(med[m]):9.3f} "
                    f"{np.percentile(med[m],95):9.3f} "
                    f"{np.median(p95[m]):9.3f} "
                    f"{np.percentile(p95[m],95):9.3f} "
                    f"{np.median(r['tc16'][m]):8.4f} "
                    f"{np.median(r['tc38'][m]):8.4f}"
                )

        report[
            "variants"
        ][
            name
        ] = variant_report

    # =================================================================
    # Cross-variant diagnostics
    #
    # This tells us WHICH semantic change moves the solution.
    # =================================================================

    print()
    print("=" * 125)
    print("CROSS-VARIANT CHANGE RELATIVE TO CURRENT")
    print("=" * 125)

    cross_report = {}

    for name in (
        "B_STANDARD",
        "C_MIN_BETA05",
    ):

        cross_report[name] = {}

        for span, key in (
            ("16", "phase16"),
            ("38", "phase38"),
        ):

            (
                sim,
                med,
                p95,
                valid_cross,
            ) = phase_compare(
                result[name][key],
                result["A_CURRENT"][key],
            )

            cross_report[
                name
            ][
                span
            ] = {}

            print()
            print(
                f"{name} vs A_CURRENT, {span}-date"
            )

            for group_name, mask in (
                ("catastrophic", is_cat),
                ("stable_control", is_control),
            ):

                m = (
                    mask
                    &
                    valid_cross
                )

                x = {
                    "n":
                        int(
                            np.count_nonzero(m)
                        ),

                    "similarity":
                        qs(
                            sim[m]
                        ),

                    "median_error_deg":
                        qs(
                            med[m]
                        ),

                    "p95_error_deg":
                        qs(
                            p95[m]
                        ),
                }

                cross_report[
                    name
                ][
                    span
                ][
                    group_name
                ] = x

                print(
                    f"  {group_name:<15s} "
                    f"sim50={np.median(sim[m]):.4f} "
                    f"med50={np.median(med[m]):.3f} "
                    f"p9550={np.median(p95[m]):.3f}"
                )

    report[
        "cross_variant"
    ] = cross_report

    # =================================================================
    # Detailed group output
    # =================================================================

    for name in variant_names:

        print()
        print("#" * 125)
        print(name)
        print("#" * 125)

        print_group(
            "CATASTROPHIC",
            report[
                "variants"
            ][
                name
            ][
                "catastrophic"
            ],
        )

        print_group(
            "STABLE CONTROL",
            report[
                "variants"
            ][
                name
            ][
                "stable_control"
            ],
        )

    # =================================================================
    # Save compact metrics -- no need to save six phase matrices.
    # =================================================================

    out_npz = (
        seqdir
        /
        "u34j_emi_solver_semantics.npz"
    )

    out_json = (
        seqdir
        /
        "u34j_emi_solver_semantics.json"
    )

    save = {
        "point_ids":
            point_ids,

        "rows":
            rr,

        "cols":
            cc,

        "is_catastrophic":
            is_cat,
    }

    for name in variant_names:

        r = result[name]

        save[
            f"{name}_similarity_16_vs_38"
        ] = r["sim"]

        save[
            f"{name}_median_error_deg_16_vs_38"
        ] = r["med"]

        save[
            f"{name}_p95_error_deg_16_vs_38"
        ] = r["p95"]

        save[
            f"{name}_tc16"
        ] = r["tc16"]

        save[
            f"{name}_tc38"
        ] = r["tc38"]

        save[
            f"{name}_eig16"
        ] = r["eig16"]

        save[
            f"{name}_eig38"
        ] = r["eig38"]

        save[
            f"{name}_valid16"
        ] = r["valid16"]

        save[
            f"{name}_valid38"
        ] = r["valid38"]

    np.savez_compressed(
        out_npz,
        **save,
    )

    report[
        "A_current_stage0_parity"
    ] = {
        "similarity":
            qs(
                parity_sim[
                    parity_valid
                ]
            ),

        "median_error_deg":
            qs(
                parity_med[
                    parity_valid
                ]
            ),

        "p95_error_deg":
            qs(
                parity_p95[
                    parity_valid
                ]
            ),
    }

    report[
        "decision"
    ] = "pending_observed_solver_semantics"

    out_json.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 125)
    print("OUTPUT")
    print("=" * 125)

    print(
        "json:",
        out_json,
    )

    print(
        "npz :",
        out_npz,
    )

    print()
    print(
        "U3.4j EMI SOLVER SEMANTICS AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
