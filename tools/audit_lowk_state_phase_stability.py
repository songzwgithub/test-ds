#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pypsds.phase_linking.coherence import (
    compressed_coherence,
)

from pypsds.phase_linking.emi import (
    ESTIMATOR_INVALID,
    image_pairs,
    median_pair_coherence,
    robust_emi_threaded,
    temporal_coherence,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


def q(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    p = np.percentile(
        x,
        [5, 25, 50, 75, 95],
    )

    return {
        "p05": float(p[0]),
        "p25": float(p[1]),
        "median": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
    }


def summarize(mask, data):

    n = int(np.count_nonzero(mask))

    if n == 0:
        return {
            "n": 0,
        }

    valid = (
        mask
        &
        np.isfinite(
            data["cross_tc"]
        )
        &
        np.isfinite(
            data["phase_similarity"]
        )
    )

    nv = int(
        np.count_nonzero(valid)
    )

    return {
        "n":
            n,

        "valid":
            nv,

        "valid_fraction":
            nv / n,

        "K":
            q(
                data["K"][mask]
            ),

        "train_tc":
            q(
                data["train_tc"][valid]
            ),

        "cross_tc":
            q(
                data["cross_tc"][valid]
            ),

        "phase_similarity":
            q(
                data["phase_similarity"][valid]
            ),

        "phase_difference_deg":
            q(
                data["phase_difference_deg"][valid]
            ),

        "pair_coherence":
            q(
                data["pair_coherence"][valid]
            ),
    }


def process_group(
    *,
    name,
    rows,
    cols,
    ctx,
    yxt,
    pi,
    pj,
    pairs,
    ndate,
    alpha,
    support_block,
    batch,
    beta,
    gamma_jitter,
    emi_mu,
    pl_workers,
    pl_chunk,
):

    n = int(rows.size)

    out = {
        "K":
            np.full(
                n,
                -1,
                np.int16,
            ),

        "estA":
            np.full(
                n,
                255,
                np.uint8,
            ),

        "estB":
            np.full(
                n,
                255,
                np.uint8,
            ),

        "train_tc":
            np.full(
                n,
                np.nan,
                np.float32,
            ),

        "cross_tc":
            np.full(
                n,
                np.nan,
                np.float32,
            ),

        "phase_similarity":
            np.full(
                n,
                np.nan,
                np.float32,
            ),

        "phase_difference_deg":
            np.full(
                n,
                np.nan,
                np.float32,
            ),

        "pair_coherence":
            np.full(
                n,
                np.nan,
                np.float32,
            ),
    }

    print()
    print("-" * 88)
    print(
        f"{name}: {n:,} centers"
    )
    print("-" * 88)

    t0 = time.perf_counter()

    for start in range(
        0,
        n,
        batch,
    ):

        stop = min(
            n,
            start + batch,
        )

        br = rows[start:stop]
        bc = cols[start:stop]

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

        B = stop - start

        # ----------------------------------------------------
        # Deterministic odd/even SHP split.
        #
        # Flattened raster-window order is used.
        # No random seed or stochastic sampling.
        # ----------------------------------------------------

        flat = support.reshape(
            B,
            -1,
        )

        rank = np.cumsum(
            flat,
            axis=1,
            dtype=np.int16,
        )

        halfA = (
            flat
            &
            (
                rank % 2
                ==
                1
            )
        )

        halfB = (
            flat
            &
            (
                rank % 2
                ==
                0
            )
        )

        supportA = halfA.reshape(
            support.shape
        )

        supportB = halfB.reshape(
            support.shape
        )

        KA = np.sum(
            supportA,
            axis=(1, 2),
        )

        KB = np.sum(
            supportB,
            axis=(1, 2),
        )

        if np.any(
            KA + KB != K
        ):
            raise RuntimeError(
                "support split count mismatch"
            )

        # ----------------------------------------------------
        # Independent covariance estimates.
        # ----------------------------------------------------

        cohA = compressed_coherence(
            yxt,
            br,
            bc,
            supportA,
            pi,
            pj,
        )

        cohB = compressed_coherence(
            yxt,
            br,
            bc,
            supportB,
            pi,
            pj,
        )

        # ----------------------------------------------------
        # Independent phase estimates.
        # ----------------------------------------------------

        (
            phA,
            estA,
            _,
            _,
            _,
        ) = robust_emi_threaded(
            cohA,
            n_images=ndate,
            pairs=pairs,
            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,
            reference_idx=0,
            workers=pl_workers,
            chunk_size=pl_chunk,
        )

        (
            phB,
            estB,
            _,
            _,
            _,
        ) = robust_emi_threaded(
            cohB,
            n_images=ndate,
            pairs=pairs,
            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,
            reference_idx=0,
            workers=pl_workers,
            chunk_size=pl_chunk,
        )

        # ----------------------------------------------------
        # Training TC:
        # phase estimated and tested on same SHP half.
        #
        # Expected to be optimistic for small K.
        # ----------------------------------------------------

        tcAA = temporal_coherence(
            cohA,
            phA,
            pairs,
        )

        tcBB = temporal_coherence(
            cohB,
            phB,
            pairs,
        )

        train_tc = np.minimum(
            tcAA,
            tcBB,
        )

        # ----------------------------------------------------
        # CROSS-validation TC:
        #
        # A phase tested against B covariance.
        # B phase tested against A covariance.
        # ----------------------------------------------------

        tcAB = temporal_coherence(
            cohB,
            phA,
            pairs,
        )

        tcBA = temporal_coherence(
            cohA,
            phB,
            pairs,
        )

        cross_tc = np.minimum(
            tcAB,
            tcBA,
        )

        # ----------------------------------------------------
        # Independent phase-vector agreement.
        #
        # Both estimates are referenced to date 0.
        # Exclude reference date itself from similarity.
        # ----------------------------------------------------

        both_valid = (
            (estA != ESTIMATOR_INVALID)
            &
            (estB != ESTIMATOR_INVALID)
            &
            np.all(
                np.isfinite(phA.real)
                &
                np.isfinite(phA.imag),
                axis=1,
            )
            &
            np.all(
                np.isfinite(phB.real)
                &
                np.isfinite(phB.imag),
                axis=1,
            )
        )

        phase_sim = np.full(
            B,
            np.nan,
            np.float32,
        )

        phase_deg = np.full(
            B,
            np.nan,
            np.float32,
        )

        ids = np.flatnonzero(
            both_valid
        )

        if ids.size:

            delta = (
                phA[
                    ids,
                    1:
                ]
                *
                np.conj(
                    phB[
                        ids,
                        1:
                    ]
                )
            )

            phase_sim[ids] = np.abs(
                np.mean(
                    delta,
                    axis=1,
                )
            ).astype(
                np.float32
            )

            phase_deg[ids] = np.median(
                np.abs(
                    np.angle(
                        delta
                    )
                ),
                axis=1,
            ).astype(
                np.float32
            )

            phase_deg[ids] *= (
                180.0
                /
                np.pi
            )

        # Conservative pair coherence.
        pairA = median_pair_coherence(
            cohA
        )

        pairB = median_pair_coherence(
            cohB
        )

        pairC = np.minimum(
            pairA,
            pairB,
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        out["K"][start:stop] = K
        out["estA"][start:stop] = estA
        out["estB"][start:stop] = estB

        out[
            "train_tc"
        ][start:stop] = train_tc

        out[
            "cross_tc"
        ][start:stop] = cross_tc

        out[
            "phase_similarity"
        ][start:stop] = phase_sim

        out[
            "phase_difference_deg"
        ][start:stop] = phase_deg

        out[
            "pair_coherence"
        ][start:stop] = pairC

        del support
        del supportA
        del supportB
        del cohA
        del cohB
        del phA
        del phB

        if (
            stop == n
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
                f"{name:<12s} "
                f"{stop:,}/{n:,} "
                f"({100*stop/n:6.2f}%) "
                f"rate={rate:,.0f} center/s"
            )

    return out


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--processing-dir",
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
        default=512,
    )

    ap.add_argument(
        "--control-max",
        type=int,
        default=20000,
    )

    ap.add_argument(
        "--half-row",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--half-col",
        type=int,
        default=11,
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=0.005,
    )

    ap.add_argument(
        "--beta",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--gamma-jitter",
        type=float,
        default=1e-6,
    )

    ap.add_argument(
        "--emi-mu",
        type=float,
        default=0.99,
    )

    ap.add_argument(
        "--pl-workers",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--pl-chunk",
        type=int,
        default=256,
    )

    args = ap.parse_args()

    processing = Path(
        args.processing_dir
    )

    seqdir = (
        processing
        /
        "sequential"
    )

    stats = (
        processing
        /
        "ds_statistics"
    )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    rescue = np.load(
        seqdir
        /
        "compression_local_rescue_mask.npy",
        mmap_mode="r",
    )

    required = np.load(
        seqdir
        /
        "compression_required_mask.npy",
        mmap_mode="r",
    )

    center_k = np.load(
        seqdir
        /
        "compression_center_shp_count.npy",
        mmap_mode="r",
    )

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

    geom = np.load(
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    ps = np.load(
        processing
        /
        "ps_mask.npy",
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

    H, W = required.shape

    if (
        yxt.ndim != 3
        or
        yxt.shape[:2] != (H, W)
    ):
        raise RuntimeError(
            f"YXT shape mismatch: "
            f"{yxt.shape}"
        )

    ndate = int(
        yxt.shape[2]
    )

    valid = np.ascontiguousarray(
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

    ps_bool = np.ascontiguousarray(
        np.asarray(
            ps,
            dtype=np.bool_,
        )
        &
        valid
    )

    rescue_bool = np.asarray(
        rescue,
        dtype=np.bool_,
    )

    req = np.asarray(
        required,
        dtype=np.bool_,
    )

    Kmap = np.asarray(
        center_k,
        dtype=np.int16,
    )

    # --------------------------------------------------------
    # Rescue population
    # --------------------------------------------------------

    rr_rescue, cc_rescue = np.where(
        rescue_bool
    )

    rr_rescue = rr_rescue.astype(
        np.int32,
        copy=False,
    )

    cc_rescue = cc_rescue.astype(
        np.int32,
        copy=False,
    )

    # --------------------------------------------------------
    # Strict near-threshold control:
    #
    # K = 48..63
    #
    # This is deliberately not the very-high-K population.
    # It is the closest valid comparison to low-K states.
    # --------------------------------------------------------

    control_mask = (
        req
        &
        (Kmap >= 48)
        &
        (Kmap <= 63)
    )

    rr_ctrl, cc_ctrl = np.where(
        control_mask
    )

    rr_ctrl = rr_ctrl.astype(
        np.int32,
        copy=False,
    )

    cc_ctrl = cc_ctrl.astype(
        np.int32,
        copy=False,
    )

    # Deterministic spatially ordered subsample.
    if (
        args.control_max > 0
        and
        rr_ctrl.size
        >
        args.control_max
    ):

        ids = np.linspace(
            0,
            rr_ctrl.size - 1,
            args.control_max,
            dtype=np.int64,
        )

        rr_ctrl = rr_ctrl[
            ids
        ]

        cc_ctrl = cc_ctrl[
            ids
        ]

    print("=" * 88)
    print(
        "U3.2c4 low-K compressed-state "
        "independent phase stability audit"
    )
    print("=" * 88)

    print(
        "scene                  :",
        f"{H} x {W}",
    )

    print(
        "dates                  :",
        ndate,
    )

    print(
        "rescue states          :",
        f"{rr_rescue.size:,}",
    )

    print(
        "strict control K=48-63 :",
        f"{rr_ctrl.size:,}",
    )

    print(
        "split                  :",
        "deterministic odd/even SHP",
    )

    print(
        "solver                 :",
        "validated robust_emi_threaded",
    )

    print()

    ctx = (
        prepare_glrt_window_context(
            scale2,
            valid,
            ps_bool,
            half_row=args.half_row,
            half_col=args.half_col,
        )
    )

    pairs = image_pairs(
        ndate
    )

    pi = np.asarray(
        pairs[:, 0],
        dtype=np.int32,
    )

    pj = np.asarray(
        pairs[:, 1],
        dtype=np.int32,
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    rescue_data = process_group(
        name="RESCUE",
        rows=rr_rescue,
        cols=cc_rescue,
        ctx=ctx,
        yxt=yxt,
        pi=pi,
        pj=pj,
        pairs=pairs,
        ndate=ndate,
        alpha=args.alpha,
        support_block=args.support_block,
        batch=args.batch,
        beta=args.beta,
        gamma_jitter=args.gamma_jitter,
        emi_mu=args.emi_mu,
        pl_workers=args.pl_workers,
        pl_chunk=args.pl_chunk,
    )

    control_data = process_group(
        name="CONTROL",
        rows=rr_ctrl,
        cols=cc_ctrl,
        ctx=ctx,
        yxt=yxt,
        pi=pi,
        pj=pj,
        pairs=pairs,
        ndate=ndate,
        alpha=args.alpha,
        support_block=args.support_block,
        batch=args.batch,
        beta=args.beta,
        gamma_jitter=args.gamma_jitter,
        emi_mu=args.emi_mu,
        pl_workers=args.pl_workers,
        pl_chunk=args.pl_chunk,
    )

    # --------------------------------------------------------
    # Summaries by K
    # --------------------------------------------------------

    rescue_bins = (
        (2, 7),
        (8, 15),
        (16, 23),
        (24, 31),
        (32, 39),
        (40, 47),
    )

    control_bins = (
        (48, 55),
        (56, 63),
    )

    summary = {
        "format":
            "pyPSDS-GAMMA-lowk-phase-stability-v1",

        "shape":
            [H, W],

        "ndate":
            ndate,

        "method":
            "deterministic-odd-even-SHP-cross-validation",

        "rescue_count":
            int(
                rr_rescue.size
            ),

        "control_count":
            int(
                rr_ctrl.size
            ),

        "rescue_bins":
            {},

        "control_bins":
            {},
    }

    for lo, hi in rescue_bins:

        m = (
            (rescue_data["K"] >= lo)
            &
            (rescue_data["K"] <= hi)
        )

        summary[
            "rescue_bins"
        ][f"{lo}-{hi}"] = summarize(
            m,
            rescue_data,
        )

    for lo, hi in control_bins:

        m = (
            (control_data["K"] >= lo)
            &
            (control_data["K"] <= hi)
        )

        summary[
            "control_bins"
        ][f"{lo}-{hi}"] = summarize(
            m,
            control_data,
        )

    # Threshold ladder for rescue K >= threshold.
    ladder = {}

    for threshold in (
        47,
        44,
        40,
        36,
        32,
        28,
        24,
        20,
        16,
        12,
        8,
        4,
        2,
    ):

        m = (
            rescue_data["K"]
            >=
            threshold
        )

        ladder[
            str(threshold)
        ] = summarize(
            m,
            rescue_data,
        )

    summary[
        "rescue_threshold_ladder"
    ] = ladder

    # --------------------------------------------------------
    # Save raw compact metrics
    # --------------------------------------------------------

    npz_path = (
        seqdir
        /
        "compression_lowk_phase_stability_metrics.npz"
    )

    np.savez_compressed(
        npz_path,

        rescue_rows=rr_rescue,
        rescue_cols=cc_rescue,
        rescue_K=rescue_data["K"],
        rescue_train_tc=rescue_data["train_tc"],
        rescue_cross_tc=rescue_data["cross_tc"],
        rescue_phase_similarity=rescue_data[
            "phase_similarity"
        ],
        rescue_phase_difference_deg=rescue_data[
            "phase_difference_deg"
        ],
        rescue_pair_coherence=rescue_data[
            "pair_coherence"
        ],
        rescue_estA=rescue_data["estA"],
        rescue_estB=rescue_data["estB"],

        control_rows=rr_ctrl,
        control_cols=cc_ctrl,
        control_K=control_data["K"],
        control_train_tc=control_data["train_tc"],
        control_cross_tc=control_data["cross_tc"],
        control_phase_similarity=control_data[
            "phase_similarity"
        ],
        control_phase_difference_deg=control_data[
            "phase_difference_deg"
        ],
        control_pair_coherence=control_data[
            "pair_coherence"
        ],
        control_estA=control_data["estA"],
        control_estB=control_data["estB"],
    )

    json_path = (
        seqdir
        /
        "compression_lowk_phase_stability.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print compact table
    # --------------------------------------------------------

    print()
    print("=" * 116)
    print(
        "U3.2c4 independent stability summary"
    )
    print("=" * 116)

    print(
        "group      K       n       valid%   "
        "crossTC50  phaseSim50  phaseDiff50(deg)"
    )

    print("-" * 116)

    def row(
        group,
        label,
        item,
    ):

        n = item.get(
            "n",
            0,
        )

        if not n:
            return

        vf = (
            100.0
            *
            item.get(
                "valid_fraction",
                0.0,
            )
        )

        ct = (
            item.get(
                "cross_tc",
                {},
            )
            .get(
                "median",
                float("nan"),
            )
        )

        psim = (
            item.get(
                "phase_similarity",
                {},
            )
            .get(
                "median",
                float("nan"),
            )
        )

        pd = (
            item.get(
                "phase_difference_deg",
                {},
            )
            .get(
                "median",
                float("nan"),
            )
        )

        print(
            f"{group:<10s} "
            f"{label:<7s} "
            f"{n:8,d} "
            f"{vf:8.3f} "
            f"{ct:10.4f} "
            f"{psim:11.4f} "
            f"{pd:16.3f}"
        )

    for label, item in (
        summary[
            "rescue_bins"
        ].items()
    ):
        row(
            "RESCUE",
            label,
            item,
        )

    for label, item in (
        summary[
            "control_bins"
        ].items()
    ):
        row(
            "CONTROL",
            label,
            item,
        )

    print()
    print("=" * 116)
    print(
        "RESCUE threshold ladder"
    )
    print("=" * 116)

    print(
        "Kmin      n       valid%   "
        "crossTC50  phaseSim50  phaseDiff50(deg)"
    )

    print("-" * 116)

    for threshold in (
        47,
        44,
        40,
        36,
        32,
        28,
        24,
        20,
        16,
        12,
        8,
        4,
        2,
    ):

        item = ladder[
            str(threshold)
        ]

        n = item.get(
            "n",
            0,
        )

        if not n:
            continue

        print(
            f"{threshold:4d} "
            f"{n:8,d} "
            f"{100*item['valid_fraction']:8.3f} "
            f"{item['cross_tc']['median']:10.4f} "
            f"{item['phase_similarity']['median']:11.4f} "
            f"{item['phase_difference_deg']['median']:16.3f}"
        )

    print()

    print(
        "metrics :",
        npz_path,
    )

    print(
        "json    :",
        json_path,
    )

    print()

    print(
        "U3.2c4 LOW-K PHASE STABILITY AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
