#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config

from pypsds.phase_linking.coherence import compressed_coherence
from pypsds.phase_linking.emi import (
    ESTIMATOR_INVALID,
    image_pairs,
    robust_emi_threaded,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)

from pypsds.phase_linking.temporal_plan import (
    TemporalStrategy,
    build_temporal_plan,
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


def q(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    v = np.percentile(
        x,
        [0, 5, 25, 50, 75, 95, 99, 100],
    )

    names = (
        "min",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
        "max",
    )

    return {
        k: float(vv)
        for k, vv in zip(names, v)
    }


def summarize(mask, sim, med_deg, p95_deg):
    n = int(np.count_nonzero(mask))

    if n == 0:
        return {"n": 0}

    return {
        "n": n,
        "similarity": q(sim[mask]),
        "median_abs_error_deg": q(med_deg[mask]),
        "p95_abs_error_deg": q(p95_deg[mask]),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--ministack-size",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--max-num-compressed",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=4096,
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
    ) = open_from_config(args.config)

    processing = (
        Path(paths.output_dir)
        / "processing"
    )

    seqdir = (
        processing
        / "sequential"
    )

    yxt_path = (
        processing
        / "cache"
        / "phase_corrected_yxt.npy"
    )

    geom_path = (
        processing
        / "cache"
        / "phase_geometry_valid.npy"
    )

    scale_path = (
        processing
        / "ds_statistics"
        / "rayleigh_scale2.npy"
    )

    raw_valid_path = (
        processing
        / "ds_statistics"
        / "raw_valid.npy"
    )

    ps_path = (
        processing
        / "ps_mask.npy"
    )

    prior_path = (
        processing
        / "center_prior.npy"
    )

    pl_valid_path = (
        processing
        / "pl_valid.npy"
    )

    full_phase_path = (
        processing
        / "u34m_beta0_full_phase.npy"
    )

    full_tc_path = (
        processing
        / "u34m_beta0_temporal_coherence.npy"
    )

    core_path = (
        seqdir
        / "compression_state_core_K24.npy"
    )

    required = (
        yxt_path,
        geom_path,
        scale_path,
        raw_valid_path,
        ps_path,
        prior_path,
        pl_valid_path,
        full_phase_path,
        full_tc_path,
        core_path,
    )

    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)

    yxt = np.load(
        yxt_path,
        mmap_mode="r",
    )

    scale2 = np.load(
        scale_path,
        mmap_mode="r",
    )

    raw_valid = np.load(
        raw_valid_path,
        mmap_mode="r",
    )

    geom = np.load(
        geom_path,
        mmap_mode="r",
    )

    ps = np.load(
        ps_path,
        mmap_mode="r",
    )

    prior = np.load(
        prior_path,
        mmap_mode="r",
    )

    pl_valid = np.load(
        pl_valid_path,
        mmap_mode="r",
    )

    full_phase = np.load(
        full_phase_path,
        mmap_mode="r",
    )

    full_tc = np.load(
        full_tc_path,
        mmap_mode="r",
    )

    state_core = np.load(
        core_path,
        mmap_mode="r",
    )

    ndate = int(yxt.shape[2])

    if yxt.shape != (H, W, ndate):
        raise RuntimeError(
            f"YXT shape mismatch: {yxt.shape}"
        )

    if full_phase.shape != (
        ndate,
        H,
        W,
    ):
        raise RuntimeError(
            f"full linked phase shape mismatch: "
            f"{full_phase.shape}"
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

    ps_bool = (
        np.asarray(
            ps,
            dtype=np.bool_,
        )
        &
        valid
    )

    prior_bool = np.asarray(
        prior,
        dtype=np.bool_,
    )

    full_ds = (
        np.asarray(
            pl_valid,
            dtype=np.bool_,
        )
        &
        prior_bool
        &
        ~ps_bool
    )

    rr_all, cc_all = np.where(
        full_ds
    )

    rr_all = rr_all.astype(
        np.int32,
        copy=False,
    )

    cc_all = cc_all.astype(
        np.int32,
        copy=False,
    )

    n_full_ds = rr_all.size

    print("=" * 104)
    print(
        "U3.4m beta0 sequential vs 38-date full-SCM phase parity"
    )
    print("=" * 104)

    print(
        "scene                  :",
        f"{H} x {W}",
    )

    print(
        "dates                  :",
        ndate,
    )

    print(
        "full-SCM DS centers    :",
        f"{n_full_ds:,}",
    )

    print(
        "sequential state core  :",
        f"{np.count_nonzero(state_core):,}",
    )

    print()

    # --------------------------------------------------------
    # Temporal plan
    # --------------------------------------------------------

    dates = tuple(
        str(x)
        for x in stack.dates
    )

    plan = build_temporal_plan(
        dates,
        strategy=TemporalStrategy.SEQUENTIAL,
        ministack_size=args.ministack_size,
        max_num_compressed=args.max_num_compressed,
        reference_index=0,
    )

    if (
        plan.effective_strategy
        != TemporalStrategy.SEQUENTIAL.value
    ):
        raise RuntimeError(
            "true sequential plan required"
        )

    # --------------------------------------------------------
    # Exact static SHP context.
    # --------------------------------------------------------

    ctx = prepare_glrt_window_context(
        scale2,
        valid,
        ps_bool,
        half_row=5,
        half_col=11,
    )

    core_windows = bool_windows(
        state_core,
        5,
        11,
    )

    # --------------------------------------------------------
    # First pass:
    #
    # Determine which formal full-SCM DS still satisfy
    # effective K >= 48 after K24 state restriction.
    # --------------------------------------------------------

    K_eff = np.full(
        n_full_ds,
        -1,
        dtype=np.int16,
    )

    t0 = time.perf_counter()

    for b0 in range(
        0,
        n_full_ds,
        args.batch,
    ):

        b1 = min(
            n_full_ds,
            b0 + args.batch,
        )

        br = rr_all[b0:b1]
        bc = cc_all[b0:b1]

        support, _ = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=0.005,
                nslc=ndate,
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

        K_eff[b0:b1] = np.sum(
            support,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

    eligible = (
        K_eff >= 48
    )

    fallback_24_47 = (
        (K_eff >= 24)
        &
        (K_eff < 48)
    )

    fallback_lt24 = (
        K_eff < 24
    )

    rr = rr_all[eligible]
    cc = cc_all[eligible]
    K = K_eff[eligible]

    n = rr.size

    print(
        "routing audit elapsed  :",
        f"{time.perf_counter()-t0:.3f} s",
    )

    print(
        "sequential K_eff>=48   :",
        f"{n:,}",
        f"({100*n/n_full_ds:.6f}%)",
    )

    print(
        "fallback 24<=K<48      :",
        f"{np.count_nonzero(fallback_24_47):,}",
    )

    print(
        "fallback K<24          :",
        f"{np.count_nonzero(fallback_lt24):,}",
    )

    print()

    # --------------------------------------------------------
    # Output point-domain phase stack.
    #
    # This is B x 38 only, not an H x W x N scene cube.
    # --------------------------------------------------------

    seq_phase_path = (
        seqdir
        / "u34m_beta0_sequential_phase_points.npy"
    )

    seq_phase = np.lib.format.open_memmap(
        seq_phase_path,
        mode="w+",
        dtype=np.complex64,
        shape=(n, ndate),
    )

    seq_phase[...] = (
        np.nan
        +
        1j
        *
        np.nan
    )

    np.save(
        seqdir
        / "u34m_beta0_rows.npy",
        rr,
    )

    np.save(
        seqdir
        / "u34m_beta0_cols.npy",
        cc,
    )

    np.save(
        seqdir
        / "u34m_beta0_effective_K.npy",
        K,
    )

    # --------------------------------------------------------
    # Reconstruct sequential real-acquisition phase outputs.
    #
    # Dolphin-compatible semantics:
    # only current real SLC phase outputs are stitched.
    # --------------------------------------------------------

    total_stage_seconds = 0.0
    stage_reports = []

    for stage in plan.stages:

        s0 = time.perf_counter()

        ncomp = stage.compressed_count
        first_real_idx = ncomp

        reference_idx = (
            0
            if ncomp == 0
            else ncomp - 1
        )

        stage_n = stage.solver_size

        # --------------------------------------------
        # Build audit-only stage stack in RAM.
        #
        # Maximum here is only:
        # 600 x 2000 x 14 complex64 ~= 128 MiB.
        #
        # Production executor remains tile-fused.
        # --------------------------------------------

        stage_stack = np.empty(
            (
                H,
                W,
                stage_n,
            ),
            dtype=np.complex64,
        )

        for j, cref in enumerate(
            stage.compressed_inputs
        ):
            p = (
                seqdir
                /
                f"u33b_stage"
                f"{cref.source_stage:04d}"
                f"_compressed.npy"
            )

            if not p.is_file():
                raise FileNotFoundError(p)

            comp = np.load(
                p,
                mmap_mode="r",
            )

            stage_stack[
                :,
                :,
                j,
            ] = comp

        real_indices = tuple(
            stage.real_indices
        )

        rs = real_indices[0]
        re = real_indices[-1] + 1

        if real_indices != tuple(
            range(rs, re)
        ):
            raise RuntimeError(
                "non-contiguous real stage"
            )

        stage_stack[
            :,
            :,
            first_real_idx:
        ] = yxt[
            :,
            :,
            rs:re,
        ]

        stage_valid = np.all(
            np.isfinite(stage_stack.real)
            &
            np.isfinite(stage_stack.imag),
            axis=2,
        )

        stage_valid_windows = bool_windows(
            stage_valid,
            5,
            11,
        )

        pairs = image_pairs(
            stage_n
        )

        pi = np.asarray(
            pairs[:, 0],
            dtype=np.int32,
        )

        pj = np.asarray(
            pairs[:, 1],
            dtype=np.int32,
        )

        stage_bad_K = 0
        stage_bad_PL = 0

        for b0 in range(
            0,
            n,
            args.batch,
        ):

            b1 = min(
                n,
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
                    nslc=ndate,
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

            support &= np.asarray(
                stage_valid_windows[
                    br,
                    bc,
                ],
                dtype=np.bool_,
            )

            K_stage = np.sum(
                support,
                axis=(1, 2),
                dtype=np.int32,
            ).astype(
                np.int16
            )

            mismatch = (
                K_stage
                !=
                K[b0:b1]
            )

            if np.any(mismatch):
                stage_bad_K += int(
                    np.count_nonzero(
                        mismatch
                    )
                )

                bad = int(
                    np.flatnonzero(
                        mismatch
                    )[0]
                )

                raise RuntimeError(
                    f"stage {stage.stage_index} "
                    f"K mismatch at "
                    f"({int(br[bad])},"
                    f"{int(bc[bad])}): "
                    f"{int(K_stage[bad])} != "
                    f"{int(K[b0+bad])}"
                )

            coh = compressed_coherence(
                stage_stack,
                br,
                bc,
                support,
                pi,
                pj,
            )

            (
                ph,
                est,
                _,
                _,
                _,
            ) = robust_emi_threaded(
                coh,
                n_images=stage_n,
                pairs=pairs,
                beta=0.0,
                gamma_jitter=1.0e-6,
                emi_mu=0.99,
                reference_idx=reference_idx,
                workers=args.pl_workers,
                chunk_size=args.pl_chunk,
            )

            ok = (
                (est != ESTIMATOR_INVALID)
                &
                np.all(
                    np.isfinite(ph.real)
                    &
                    np.isfinite(ph.imag),
                    axis=1,
                )
            )

            if not np.all(ok):
                stage_bad_PL += int(
                    np.count_nonzero(
                        ~ok
                    )
                )

                raise RuntimeError(
                    f"stage {stage.stage_index}: "
                    f"{stage_bad_PL} invalid "
                    f"sequential phase estimates"
                )

            # Only real acquisitions become final output phases.
            seq_phase[
                b0:b1,
                rs:re,
            ] = ph[
                :,
                first_real_idx:
            ]

        seq_phase.flush()

        elapsed = (
            time.perf_counter()
            -
            s0
        )

        total_stage_seconds += elapsed

        stage_reports.append(
            {
                "stage_index":
                    stage.stage_index,

                "solver_size":
                    stage_n,

                "reference_idx":
                    reference_idx,

                "first_real_idx":
                    first_real_idx,

                "real_indices":
                    list(real_indices),

                "K_mismatch":
                    stage_bad_K,

                "PL_invalid":
                    stage_bad_PL,

                "seconds":
                    elapsed,
            }
        )

        print(
            f"stage {stage.stage_index}: "
            f"solver={stage_n}, "
            f"ref={reference_idx}, "
            f"real={rs}:{re}, "
            f"Kbad={stage_bad_K}, "
            f"PLbad={stage_bad_PL}, "
            f"wall={elapsed:.2f}s"
        )

        del stage_stack

    # --------------------------------------------------------
    # Computational integrity.
    # --------------------------------------------------------

    if np.any(
        ~np.isfinite(
            seq_phase.real
        )
        |
        ~np.isfinite(
            seq_phase.imag
        )
    ):
        raise RuntimeError(
            "reconstructed sequential phase contains NaN/Inf"
        )

    # --------------------------------------------------------
    # Compare against existing full-SCM phase.
    # --------------------------------------------------------

    center_similarity = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    center_median_deg = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    center_p95_deg = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    center_max_deg = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    compare_t0 = time.perf_counter()

    for b0 in range(
        0,
        n,
        args.batch,
    ):

        b1 = min(
            n,
            b0 + args.batch,
        )

        br = rr[b0:b1]
        bc = cc[b0:b1]

        f = (
            full_phase[
                :,
                br,
                bc,
            ]
            .T
            .astype(
                np.complex64,
                copy=False,
            )
        )

        s = np.asarray(
            seq_phase[
                b0:b1,
                :
            ]
        )

        if np.any(
            ~np.isfinite(f.real)
            |
            ~np.isfinite(f.imag)
        ):
            raise RuntimeError(
                "full-SCM reference phase contains NaN/Inf "
                "on selected parity centers"
            )

        # Defensive unit normalization.
        f = np.exp(
            1j
            *
            np.angle(f)
        ).astype(
            np.complex64
        )

        s = np.exp(
            1j
            *
            np.angle(s)
        ).astype(
            np.complex64
        )

        # Both should already use date 0.
        f *= np.conj(
            f[:, 0]
        )[:, None]

        s *= np.conj(
            s[:, 0]
        )[:, None]

        delta = (
            s[:, 1:]
            *
            np.conj(
                f[:, 1:]
            )
        )

        angle_deg = (
            np.abs(
                np.angle(
                    delta
                )
            )
            *
            180.0
            /
            np.pi
        ).astype(
            np.float32
        )

        center_similarity[
            b0:b1
        ] = np.abs(
            np.mean(
                delta,
                axis=1,
            )
        ).astype(
            np.float32
        )

        center_median_deg[
            b0:b1
        ] = np.median(
            angle_deg,
            axis=1,
        )

        center_p95_deg[
            b0:b1
        ] = np.percentile(
            angle_deg,
            95,
            axis=1,
        ).astype(
            np.float32
        )

        center_max_deg[
            b0:b1
        ] = np.max(
            angle_deg,
            axis=1,
        )

    # --------------------------------------------------------
    # Per-date exact error distributions.
    # --------------------------------------------------------

    per_date = []

    for d in range(
        ndate
    ):

        f = full_phase[
            d,
            rr,
            cc,
        ]

        s = seq_phase[
            :,
            d,
        ]

        delta = (
            np.exp(
                1j
                *
                np.angle(s)
            )
            *
            np.conj(
                np.exp(
                    1j
                    *
                    np.angle(f)
                )
            )
        )

        deg = (
            np.abs(
                np.angle(
                    delta
                )
            )
            *
            180.0
            /
            np.pi
        )

        per_date.append(
            {
                "index": d,
                "date": dates[d],
                "error_deg": q(deg),
            }
        )

    # --------------------------------------------------------
    # Stage-wise distributions.
    # --------------------------------------------------------

    per_stage = []

    for stage in plan.stages:

        inds = np.asarray(
            stage.real_indices,
            dtype=np.int32,
        )

        # Reference date gives zero by construction.
        if 0 in inds and inds.size > 1:
            inds_eval = inds[
                inds != 0
            ]
        else:
            inds_eval = inds

        # Extract all selected dates for all selected pixels.
        #
        # Do NOT use:
        #     full_phase[inds_eval, rr, cc]
        #
        # because inds_eval has shape (Ndate_stage,) while
        # rr/cc have shape (Ncenter,), and NumPy advanced
        # indexing tries to broadcast all three arrays.
        #
        # Build the desired matrix explicitly:
        #     (Ncenter, Ndate_stage)
        f = np.column_stack(
            [
                np.asarray(
                    full_phase[
                        int(d),
                        rr,
                        cc,
                    ]
                )
                for d in inds_eval
            ]
        )

        s = seq_phase[
            :,
            inds_eval,
        ]

        delta = (
            np.exp(
                1j
                *
                np.angle(s)
            )
            *
            np.conj(
                np.exp(
                    1j
                    *
                    np.angle(f)
                )
            )
        )

        deg = (
            np.abs(
                np.angle(delta)
            )
            *
            180.0
            /
            np.pi
        )

        sim = np.abs(
            np.mean(
                delta,
                axis=1,
            )
        )

        per_stage.append(
            {
                "stage_index":
                    stage.stage_index,

                "real_indices":
                    list(
                        stage.real_indices
                    ),

                "point_epoch_error_deg":
                    q(
                        deg.ravel()
                    ),

                "phase_vector_similarity":
                    q(sim),
            }
        )

    # --------------------------------------------------------
    # K stratification.
    # --------------------------------------------------------

    k_groups = {}

    for name, mask in {
        "K48_63":
            (K >= 48)
            &
            (K <= 63),

        "K64_95":
            (K >= 64)
            &
            (K <= 95),

        "K96_127":
            (K >= 96)
            &
            (K <= 127),

        "K128_plus":
            K >= 128,
    }.items():

        k_groups[name] = summarize(
            mask,
            center_similarity,
            center_median_deg,
            center_p95_deg,
        )

    # --------------------------------------------------------
    # Full-SCM TC stratification.
    # This is grouping only, not claiming the stage TC is
    # numerically equivalent to 38-date full-SCM TC.
    # --------------------------------------------------------

    tc = np.asarray(
        full_tc[
            rr,
            cc,
        ],
        dtype=np.float32,
    )

    tc_groups = {}

    for name, mask in {
        "TC_lt_0p8":
            tc < 0.8,

        "TC_0p8_0p9":
            (tc >= 0.8)
            &
            (tc < 0.9),

        "TC_ge_0p9":
            tc >= 0.9,
    }.items():

        tc_groups[name] = summarize(
            mask,
            center_similarity,
            center_median_deg,
            center_p95_deg,
        )

    # --------------------------------------------------------
    # Save compact point metrics.
    # --------------------------------------------------------

    metrics_path = (
        seqdir
        /
        "u34m_beta0_phase_parity_metrics.npz"
    )

    np.savez_compressed(
        metrics_path,

        rows=rr,
        cols=cc,

        effective_K=K,

        full_temporal_coherence=tc,

        phase_similarity=center_similarity,

        median_abs_error_deg=center_median_deg,

        p95_abs_error_deg=center_p95_deg,

        max_abs_error_deg=center_max_deg,
    )

    report = {
        "format":
            "pyPSDS-GAMMA-U3.4m beta0-sequential-full-phase-parity-v1",

        "shape":
            [H, W],

        "ndate":
            ndate,

        "ministack_size":
            args.ministack_size,

        "max_num_compressed":
            args.max_num_compressed,

        "formal_full_scm_DS":
            int(
                n_full_ds
            ),

        "sequential_Keff_ge48":
            int(n),

        "fallback_K24_47":
            int(
                np.count_nonzero(
                    fallback_24_47
                )
            ),

        "fallback_Klt24":
            int(
                np.count_nonzero(
                    fallback_lt24
                )
            ),

        "global":
            {
                "phase_similarity":
                    q(
                        center_similarity
                    ),

                "median_abs_error_deg":
                    q(
                        center_median_deg
                    ),

                "p95_abs_error_deg":
                    q(
                        center_p95_deg
                    ),

                "max_abs_error_deg":
                    q(
                        center_max_deg
                    ),
            },

        "K_groups":
            k_groups,

        "full_TC_groups":
            tc_groups,

        "per_stage":
            per_stage,

        "per_date":
            per_date,

        "stage_reconstruction":
            stage_reports,

        "sequential_reconstruction_seconds":
            total_stage_seconds,

        "comparison_seconds":
            (
                time.perf_counter()
                -
                compare_t0
            ),

        "scientific_decision":
            "pending_observed_error_distribution",
    }

    json_path = (
        seqdir
        /
        "u34m_beta0_phase_parity_report.json"
    )

    json_path.write_text(
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
    print("=" * 112)
    print(
        "U3.4m beta0 GLOBAL PHASE PARITY"
    )
    print("=" * 112)

    print(
        "centers               :",
        f"{n:,}",
    )

    print(
        "phase similarity      :",
        q(
            center_similarity
        ),
    )

    print(
        "center median err deg :",
        q(
            center_median_deg
        ),
    )

    print(
        "center p95 err deg    :",
        q(
            center_p95_deg
        ),
    )

    print()

    print("=" * 112)
    print(
        "K STRATIFICATION"
    )
    print("=" * 112)

    print(
        "group       n        "
        "sim50      medErr50(deg)   p95Err50(deg)"
    )

    print("-" * 112)

    for name, x in k_groups.items():

        if not x.get(
            "n",
            0
        ):
            continue

        print(
            f"{name:<11s} "
            f"{x['n']:8,d} "
            f"{x['similarity']['median']:10.4f} "
            f"{x['median_abs_error_deg']['median']:15.3f} "
            f"{x['p95_abs_error_deg']['median']:15.3f}"
        )

    print()

    print("=" * 112)
    print(
        "FULL-SCM TC STRATIFICATION"
    )
    print("=" * 112)

    print(
        "group       n        "
        "sim50      medErr50(deg)   p95Err50(deg)"
    )

    print("-" * 112)

    for name, x in tc_groups.items():

        if not x.get(
            "n",
            0
        ):
            continue

        print(
            f"{name:<11s} "
            f"{x['n']:8,d} "
            f"{x['similarity']['median']:10.4f} "
            f"{x['median_abs_error_deg']['median']:15.3f} "
            f"{x['p95_abs_error_deg']['median']:15.3f}"
        )

    print()

    print("=" * 112)
    print(
        "STAGE PARITY"
    )
    print("=" * 112)

    print(
        "stage | dates | "
        "point-epoch med | point-epoch p95 | "
        "stage similarity median"
    )

    print("-" * 112)

    for x in per_stage:

        e = x[
            "point_epoch_error_deg"
        ]

        s = x[
            "phase_vector_similarity"
        ]

        print(
            f"{x['stage_index']:5d} | "
            f"{len(x['real_indices']):5d} | "
            f"{e['median']:15.3f} | "
            f"{e['p95']:15.3f} | "
            f"{s['median']:23.4f}"
        )

    print()

    print(
        "seq phase points :",
        seq_phase_path,
    )

    print(
        "metrics          :",
        metrics_path,
    )

    print(
        "report           :",
        json_path,
    )

    print()

    print(
        "U3.4m beta0 COMPUTATIONAL INTEGRITY: PASS"
    )

    print(
        "U3.4m beta0 SCIENTIFIC DECISION: "
        "PENDING OBSERVED METRICS"
    )


if __name__ == "__main__":
    main()
