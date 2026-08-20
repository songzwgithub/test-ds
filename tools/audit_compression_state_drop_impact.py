#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def quantiles(x):
    x = np.asarray(x)

    if x.size == 0:
        return {}

    q = np.percentile(
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
        k: float(v)
        for k, v in zip(names, q)
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--processing-dir",
        required=True,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=16000,
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=1024,
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
        "--min-shp",
        type=int,
        default=48,
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

    required_path = (
        seqdir
        /
        "compression_required_mask.npy"
    )

    fallback_path = (
        seqdir
        /
        "compression_missing_ineligible_mask.npy"
    )

    c1_k_path = (
        seqdir
        /
        "compression_center_shp_count.npy"
    )

    scale_path = (
        stats
        /
        "rayleigh_scale2.npy"
    )

    raw_valid_path = (
        stats
        /
        "raw_valid.npy"
    )

    geom_path = (
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy"
    )

    ps_path = (
        processing
        /
        "ps_mask.npy"
    )

    prior_path = (
        processing
        /
        "center_prior.npy"
    )

    pl_path = (
        processing
        /
        "pl_valid.npy"
    )

    linked_path = (
        processing
        /
        "linked_phase.npy"
    )

    for p in (
        required_path,
        fallback_path,
        c1_k_path,
        scale_path,
        raw_valid_path,
        geom_path,
        ps_path,
        prior_path,
        pl_path,
        linked_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    required = np.load(
        required_path,
        mmap_mode="r",
    )

    fallback = np.load(
        fallback_path,
        mmap_mode="r",
    )

    c1_k = np.load(
        c1_k_path,
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
        pl_path,
        mmap_mode="r",
    )

    linked = np.load(
        linked_path,
        mmap_mode="r",
    )

    H, W = required.shape

    # --------------------------------------------------------
    # ndate
    # --------------------------------------------------------

    if (
        linked.ndim == 3
        and
        linked.shape[1:] == (H, W)
    ):
        ndate = int(
            linked.shape[0]
        )

    elif (
        linked.ndim == 3
        and
        linked.shape[:2] == (H, W)
    ):
        ndate = int(
            linked.shape[2]
        )

    else:
        raise RuntimeError(
            f"unexpected linked phase shape: "
            f"{linked.shape}"
        )

    # --------------------------------------------------------
    # Frozen spatial masks
    # --------------------------------------------------------

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

    req = np.asarray(
        required,
        dtype=np.bool_,
    )

    fb = np.asarray(
        fallback,
        dtype=np.bool_,
    )

    prior_bool = np.asarray(
        prior,
        dtype=np.bool_,
    )

    pl_bool = np.asarray(
        pl_valid,
        dtype=np.bool_,
    )

    # Original DS output-center population.
    target = (
        prior_bool
        &
        valid
        &
        ~ps_bool
    )

    # We need to audit:
    #
    #   1. every dense compression center
    #   2. every original target center
    #
    # target is included separately because a target center does
    # not mathematically have to appear in another center's SHP union.
    analysis_mask = (
        req
        |
        target
    )

    if np.any(
        fb
        &
        ~req
    ):
        raise RuntimeError(
            "fallback mask is not subset of "
            "compression-required mask"
        )

    # --------------------------------------------------------
    # Exact GLRT context
    # --------------------------------------------------------

    ctx = (
        prepare_glrt_window_context(
            scale2,
            valid,
            ps_bool,
            half_row=args.half_row,
            half_col=args.half_col,
        )
    )

    # --------------------------------------------------------
    # Sliding view of fallback mask.
    #
    # This answers:
    #
    #   among a center's original SHP samples,
    #   how many belong to the K<48 compressed-state mask?
    # --------------------------------------------------------

    pad = (
        (args.half_row, args.half_row),
        (args.half_col, args.half_col),
    )

    fb_pad = np.pad(
        fb,
        pad,
        mode="constant",
        constant_values=False,
    )

    fb_windows = (
        np.lib.stride_tricks
        .sliding_window_view(
            fb_pad,
            (
                2 * args.half_row + 1,
                2 * args.half_col + 1,
            ),
        )
    )

    if (
        fb_windows.shape[:2]
        !=
        (H, W)
    ):
        raise RuntimeError(
            "fallback window shape mismatch"
        )

    # --------------------------------------------------------
    # Centers
    # --------------------------------------------------------

    rr, cc = np.where(
        analysis_mask
    )

    rr = rr.astype(
        np.int32,
        copy=False,
    )

    cc = cc.astype(
        np.int32,
        copy=False,
    )

    ncenter = int(
        rr.size
    )

    # --------------------------------------------------------
    # Audit maps
    # --------------------------------------------------------

    K_before = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    fallback_loss = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    K_after = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    print("=" * 88)
    print(
        "U3.2c2 compressed-state low-K drop impact audit"
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
        "analysis centers       :",
        f"{ncenter:,}",
    )

    print(
        "compression required   :",
        f"{np.count_nonzero(req):,}",
    )

    print(
        "low-K state pixels     :",
        f"{np.count_nonzero(fb):,}",
    )

    print(
        "min SHP                :",
        args.min_shp,
    )

    print()

    t0 = time.perf_counter()

    # --------------------------------------------------------
    # Regenerate exact SHP support and remove low-K state pixels.
    # --------------------------------------------------------

    for start in range(
        0,
        ncenter,
        args.batch,
    ):

        stop = min(
            ncenter,
            start + args.batch,
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
                alpha=args.alpha,
                nslc=ndate,
                block_size=args.support_block,
            )
        )

        # Exact U3.2c1 parity check for required pixels.
        is_req = req[
            br,
            bc,
        ]

        if np.any(
            is_req
        ):

            expected = c1_k[
                br[is_req],
                bc[is_req],
            ]

            got = K[
                is_req
            ]

            mismatch = (
                expected
                !=
                got
            )

            if np.any(
                mismatch
            ):
                bad = int(
                    np.count_nonzero(
                        mismatch
                    )
                )

                raise RuntimeError(
                    "U3.2c1 K parity failure: "
                    f"{bad} centers differ"
                )

        fb_local = np.asarray(
            fb_windows[
                br,
                bc,
            ],
            dtype=np.bool_,
        )

        # Number of original SHP samples lost if all K<48
        # compressed-state pixels are treated as unavailable.
        loss = np.sum(
            support
            &
            fb_local,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

        remaining = (
            K.astype(
                np.int32
            )
            -
            loss.astype(
                np.int32
            )
        ).astype(
            np.int16
        )

        K_before[
            br,
            bc,
        ] = K

        fallback_loss[
            br,
            bc,
        ] = loss

        K_after[
            br,
            bc,
        ] = remaining

        del support
        del fb_local

        if (
            stop == ncenter
            or
            stop % (
                args.batch
                *
                10
            ) == 0
        ):

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            rate = (
                stop
                /
                elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"centers "
                f"{stop:,}/"
                f"{ncenter:,} "
                f"({100*stop/ncenter:6.2f}%) "
                f"rate={rate:,.0f} center/s"
            )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    # --------------------------------------------------------
    # Populations
    # --------------------------------------------------------

    dense_eligible = (
        req
        &
        (
            K_before
            >=
            args.min_shp
        )
    )

    target_eligible = (
        target
        &
        (
            K_before
            >=
            args.min_shp
        )
    )

    current_pl = (
        pl_bool
        &
        analysis_mask
    )

    # Any low-K state pixel appears in this center's SHP.
    dense_touched = (
        dense_eligible
        &
        (
            fallback_loss
            >
            0
        )
    )

    target_touched = (
        target_eligible
        &
        (
            fallback_loss
            >
            0
        )
    )

    # Would dropping all low-K state samples make the center
    # violate the existing K>=48 rule?
    dense_fail = (
        dense_eligible
        &
        (
            K_after
            <
            args.min_shp
        )
    )

    target_fail = (
        target_eligible
        &
        (
            K_after
            <
            args.min_shp
        )
    )

    current_pl_fail = (
        current_pl
        &
        (
            K_after
            <
            args.min_shp
        )
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    n_dense = int(
        np.count_nonzero(
            dense_eligible
        )
    )

    n_dense_touched = int(
        np.count_nonzero(
            dense_touched
        )
    )

    n_dense_fail = int(
        np.count_nonzero(
            dense_fail
        )
    )

    n_target = int(
        np.count_nonzero(
            target_eligible
        )
    )

    n_target_touched = int(
        np.count_nonzero(
            target_touched
        )
    )

    n_target_fail = int(
        np.count_nonzero(
            target_fail
        )
    )

    n_pl = int(
        np.count_nonzero(
            current_pl
        )
    )

    n_pl_fail = int(
        np.count_nonzero(
            current_pl_fail
        )
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if (
        n_dense_fail == 0
        and
        n_target_fail == 0
        and
        n_pl_fail == 0
    ):

        decision = (
            "drop_low_k_state_structurally_safe"
        )

        recommendation = (
            "The K<min_shp compression-state pixels can be "
            "left invalid without causing any currently "
            "eligible dense-compression or DS-output center "
            "to fall below min_shp. No fallback phase "
            "estimator is structurally required. The next "
            "step is a fused sequential PL/compression "
            "prototype with an explicit compressed-state "
            "validity mask."
        )

    else:

        decision = (
            "local_state_rescue_required"
        )

        recommendation = (
            "Dropping every K<min_shp compression-state pixel "
            "would make at least one currently eligible center "
            "fall below min_shp. Do not globally lower min_shp. "
            "The next step must design a local rescue policy "
            "only for the state pixels needed by those affected "
            "centers."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    k_before_path = (
        seqdir
        /
        "compression_drop_K_before.npy"
    )

    loss_path = (
        seqdir
        /
        "compression_drop_support_loss.npy"
    )

    k_after_path = (
        seqdir
        /
        "compression_drop_K_after.npy"
    )

    dense_fail_path = (
        seqdir
        /
        "compression_drop_dense_fail_mask.npy"
    )

    target_fail_path = (
        seqdir
        /
        "compression_drop_target_fail_mask.npy"
    )

    np.save(
        k_before_path,
        K_before,
    )

    np.save(
        loss_path,
        fallback_loss,
    )

    np.save(
        k_after_path,
        K_after,
    )

    np.save(
        dense_fail_path,
        dense_fail,
    )

    np.save(
        target_fail_path,
        target_fail,
    )

    report = {
        "format":
            "pyPSDS-GAMMA-compression-state-drop-impact-v1",

        "shape":
            [H, W],

        "ndate":
            ndate,

        "min_shp":
            args.min_shp,

        "low_k_state_pixels":
            int(
                np.count_nonzero(
                    fb
                )
            ),

        "dense_eligible_centers":
            n_dense,

        "dense_centers_touched":
            n_dense_touched,

        "dense_centers_touched_fraction":
            (
                n_dense_touched
                /
                n_dense
                if n_dense
                else 0.0
            ),

        "dense_centers_fail_after_drop":
            n_dense_fail,

        "dense_fail_fraction":
            (
                n_dense_fail
                /
                n_dense
                if n_dense
                else 0.0
            ),

        "target_eligible_centers":
            n_target,

        "target_centers_touched":
            n_target_touched,

        "target_centers_fail_after_drop":
            n_target_fail,

        "current_pl_centers":
            n_pl,

        "current_pl_fail_after_drop":
            n_pl_fail,

        "dense_support_loss_quantiles":
            quantiles(
                fallback_loss[
                    dense_touched
                ]
            ),

        "dense_K_before_quantiles":
            quantiles(
                K_before[
                    dense_eligible
                ]
            ),

        "dense_K_after_quantiles":
            quantiles(
                K_after[
                    dense_eligible
                ]
            ),

        "failed_dense_K_before_quantiles":
            quantiles(
                K_before[
                    dense_fail
                ]
            ),

        "failed_dense_support_loss_quantiles":
            quantiles(
                fallback_loss[
                    dense_fail
                ]
            ),

        "failed_dense_K_after_quantiles":
            quantiles(
                K_after[
                    dense_fail
                ]
            ),

        "elapsed_seconds":
            elapsed,

        "decision":
            decision,

        "recommendation":
            recommendation,
    }

    json_path = (
        seqdir
        /
        "compression_state_drop_impact.json"
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

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print(
        "U3.2c2 result"
    )
    print("=" * 88)

    print(
        "low-K state pixels          :",
        f"{np.count_nonzero(fb):,}",
    )

    print()

    print(
        "dense eligible centers      :",
        f"{n_dense:,}",
    )

    print(
        "dense touched by low-K      :",
        f"{n_dense_touched:,}",
        f"({pct(n_dense_touched, n_dense):.3f}%)",
    )

    print(
        "dense K<48 after drop       :",
        f"{n_dense_fail:,}",
        f"({pct(n_dense_fail, n_dense):.6f}%)",
    )

    print()

    print(
        "target eligible centers     :",
        f"{n_target:,}",
    )

    print(
        "target touched by low-K     :",
        f"{n_target_touched:,}",
        f"({pct(n_target_touched, n_target):.3f}%)",
    )

    print(
        "target K<48 after drop      :",
        f"{n_target_fail:,}",
        f"({pct(n_target_fail, n_target):.6f}%)",
    )

    print()

    print(
        "current PL centers          :",
        f"{n_pl:,}",
    )

    print(
        "current PL K<48 after drop  :",
        f"{n_pl_fail:,}",
    )

    print()

    print(
        "support-loss quantiles      :",
        quantiles(
            fallback_loss[
                dense_touched
            ]
        ),
    )

    print(
        "dense K before              :",
        quantiles(
            K_before[
                dense_eligible
            ]
        ),
    )

    print(
        "dense K after               :",
        quantiles(
            K_after[
                dense_eligible
            ]
        ),
    )

    if n_dense_fail:

        print()

        print(
            "failed K before            :",
            quantiles(
                K_before[
                    dense_fail
                ]
            ),
        )

        print(
            "failed support loss        :",
            quantiles(
                fallback_loss[
                    dense_fail
                ]
            ),
        )

        print(
            "failed K after             :",
            quantiles(
                K_after[
                    dense_fail
                ]
            ),
        )

    print()

    print(
        "elapsed                     :",
        f"{elapsed:.3f} s",
    )

    print()

    print(
        "K before :",
        k_before_path,
    )

    print(
        "loss map :",
        loss_path,
    )

    print(
        "K after  :",
        k_after_path,
    )

    print(
        "dense fail:",
        dense_fail_path,
    )

    print(
        "target fail:",
        target_fail_path,
    )

    print(
        "json     :",
        json_path,
    )

    print()

    print(
        "decision :",
        decision,
    )

    print()

    print(
        recommendation
    )

    print()

    print(
        "U3.2c2 LOW-K STATE DROP IMPACT AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
