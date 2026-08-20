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
        default=4096,
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

    lowk_path = (
        seqdir
        /
        "compression_missing_ineligible_mask.npy"
    )

    center_k_path = (
        seqdir
        /
        "compression_center_shp_count.npy"
    )

    k_before_path = (
        seqdir
        /
        "compression_drop_K_before.npy"
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

    linked_path = (
        processing
        /
        "linked_phase.npy"
    )

    for p in (
        required_path,
        lowk_path,
        center_k_path,
        k_before_path,
        k_after_path,
        dense_fail_path,
        target_fail_path,
        scale_path,
        raw_valid_path,
        geom_path,
        ps_path,
        linked_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    required = np.load(
        required_path,
        mmap_mode="r",
    )

    lowk = np.load(
        lowk_path,
        mmap_mode="r",
    )

    center_k = np.load(
        center_k_path,
        mmap_mode="r",
    )

    K_before_map = np.load(
        k_before_path,
        mmap_mode="r",
    )

    K_after_map = np.load(
        k_after_path,
        mmap_mode="r",
    )

    dense_fail = np.load(
        dense_fail_path,
        mmap_mode="r",
    )

    target_fail = np.load(
        target_fail_path,
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

    linked = np.load(
        linked_path,
        mmap_mode="r",
    )

    H, W = required.shape

    # --------------------------------------------------------
    # Temporal size
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
            f"Unexpected linked phase shape: "
            f"{linked.shape}"
        )

    # --------------------------------------------------------
    # Masks
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

    low = np.asarray(
        lowk,
        dtype=np.bool_,
    )

    dense_fail_bool = np.asarray(
        dense_fail,
        dtype=np.bool_,
    )

    target_fail_bool = np.asarray(
        target_fail,
        dtype=np.bool_,
    )

    # Protect every center that failed U3.2c2.
    fail_union = (
        dense_fail_bool
        |
        target_fail_bool
    )

    rr, cc = np.where(
        fail_union
    )

    rr = rr.astype(
        np.int32,
        copy=False,
    )

    cc = cc.astype(
        np.int32,
        copy=False,
    )

    nfail = int(
        rr.size
    )

    n_dense_fail = int(
        np.count_nonzero(
            dense_fail_bool
        )
    )

    n_target_fail = int(
        np.count_nonzero(
            target_fail_bool
        )
    )

    n_low = int(
        np.count_nonzero(
            low
        )
    )

    if nfail == 0:
        raise RuntimeError(
            "No failed centers found; "
            "U3.2c3 is unnecessary."
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

    pad = (
        (args.half_row, args.half_row),
        (args.half_col, args.half_col),
    )

    # Is a neighbor one of the low-K state candidates?
    low_pad = np.pad(
        low,
        pad,
        mode="constant",
        constant_values=False,
    )

    low_windows = (
        np.lib.stride_tricks
        .sliding_window_view(
            low_pad,
            (
                2 * args.half_row + 1,
                2 * args.half_col + 1,
            ),
        )
    )

    # Self-K of every potential rescued state pixel.
    k_pad = np.pad(
        np.asarray(
            center_k,
            dtype=np.int16,
        ),
        pad,
        mode="constant",
        constant_values=-1,
    )

    k_windows = (
        np.lib.stride_tricks
        .sliding_window_view(
            k_pad,
            (
                2 * args.half_row + 1,
                2 * args.half_col + 1,
            ),
        )
    )

    # --------------------------------------------------------
    # Rescue counters.
    #
    # rescue_counts[:, t] =
    #
    # number of lost low-K SHP samples which would return
    # if compression-only states with self-K >= t are allowed.
    #
    # t = 1 ... 47.
    # --------------------------------------------------------

    max_fallback_k = (
        args.min_shp - 1
    )

    rescue_counts = np.zeros(
        (
            nfail,
            args.min_shp,
        ),
        dtype=np.int16,
    )

    # Base K after ALL low-K state was dropped.
    base_k = np.asarray(
        K_after_map[
            rr,
            cc,
        ],
        dtype=np.int16,
    )

    original_k = np.asarray(
        K_before_map[
            rr,
            cc,
        ],
        dtype=np.int16,
    )

    # Sanity.
    if np.any(
        base_k
        >=
        args.min_shp
    ):
        raise RuntimeError(
            "fail_union contains center with "
            "K_after >= min_shp"
        )

    print("=" * 88)
    print(
        "U3.2c3 local compressed-state rescue threshold audit"
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
        "low-K state candidates :",
        f"{n_low:,}",
    )

    print(
        "dense failed centers   :",
        f"{n_dense_fail:,}",
    )

    print(
        "target failed centers  :",
        f"{n_target_fail:,}",
    )

    print(
        "fail union             :",
        f"{nfail:,}",
    )

    print(
        "fallback thresholds    :",
        f"{max_fallback_k} -> 1",
    )

    print()

    t0 = time.perf_counter()

    # --------------------------------------------------------
    # Analyze only affected centers.
    # --------------------------------------------------------

    for start in range(
        0,
        nfail,
        args.batch,
    ):

        stop = min(
            nfail,
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

        # Exact parity with U3.2c2.
        expected = np.asarray(
            K_before_map[
                br,
                bc,
            ],
            dtype=np.int16,
        )

        mismatch = (
            K
            !=
            expected
        )

        if np.any(
            mismatch
        ):
            raise RuntimeError(
                "U3.2c2 K parity failure: "
                f"{np.count_nonzero(mismatch)} "
                "centers differ"
            )

        low_local = np.asarray(
            low_windows[
                br,
                bc,
            ],
            dtype=np.bool_,
        )

        kval_local = np.asarray(
            k_windows[
                br,
                bc,
            ],
            dtype=np.int16,
        )

        lost_support = (
            support
            &
            low_local
        )

        # ----------------------------------------------------
        # Threshold ladder.
        #
        # Only candidate state quality changes.
        # The frozen center SHP definition does not.
        # ----------------------------------------------------

        for threshold in range(
            1,
            args.min_shp,
        ):

            rescue_counts[
                start:stop,
                threshold,
            ] = np.sum(
                lost_support
                &
                (
                    kval_local
                    >=
                    threshold
                ),
                axis=(1, 2),
                dtype=np.int32,
            ).astype(
                np.int16
            )

        del support
        del low_local
        del kval_local
        del lost_support

        if (
            stop == nfail
            or
            stop % (
                args.batch
                *
                2
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
                f"failed centers "
                f"{stop:,}/"
                f"{nfail:,} "
                f"({100*stop/nfail:6.2f}%) "
                f"rate={rate:,.0f} center/s"
            )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    # --------------------------------------------------------
    # Threshold results
    # --------------------------------------------------------

    dense_flags = np.asarray(
        dense_fail_bool[
            rr,
            cc,
        ],
        dtype=np.bool_,
    )

    target_flags = np.asarray(
        target_fail_bool[
            rr,
            cc,
        ],
        dtype=np.bool_,
    )

    results = []

    chosen_threshold = None

    # Search HIGH -> LOW:
    #
    # choose the highest possible fallback K.
    for threshold in range(
        max_fallback_k,
        0,
        -1,
    ):

        effective = (
            base_k.astype(
                np.int32
            )
            +
            rescue_counts[
                :,
                threshold,
            ].astype(
                np.int32
            )
        )

        remain = (
            effective
            <
            args.min_shp
        )

        dense_remain = int(
            np.count_nonzero(
                remain
                &
                dense_flags
            )
        )

        target_remain = int(
            np.count_nonzero(
                remain
                &
                target_flags
            )
        )

        union_remain = int(
            np.count_nonzero(
                remain
            )
        )

        candidate_count = int(
            np.count_nonzero(
                low
                &
                (
                    np.asarray(
                        center_k
                    )
                    >=
                    threshold
                )
            )
        )

        results.append(
            {
                "threshold":
                    threshold,

                "candidate_state_count":
                    candidate_count,

                "candidate_fraction_low_k":
                    (
                        candidate_count
                        /
                        n_low
                        if n_low
                        else 0.0
                    ),

                "dense_fail_remaining":
                    dense_remain,

                "target_fail_remaining":
                    target_remain,

                "union_fail_remaining":
                    union_remain,

                "rescued_union":
                    (
                        nfail
                        -
                        union_remain
                    ),
            }
        )

        if (
            chosen_threshold is None
            and
            union_remain == 0
        ):
            chosen_threshold = threshold

    if chosen_threshold is None:
        raise RuntimeError(
            "Even K>=1 rescue failed to recover "
            "all affected centers; audit logic inconsistent."
        )

    # --------------------------------------------------------
    # Build LOCAL rescue mask.
    #
    # Do not enable every low-K state satisfying the threshold.
    #
    # Only enable those actually appearing as SHP samples
    # of one of the U3.2c2 failed centers.
    # --------------------------------------------------------

    rescue_mask = np.zeros(
        (H, W),
        dtype=np.bool_,
    )

    wh = (
        2 * args.half_row + 1
    )

    ww = (
        2 * args.half_col + 1
    )

    for start in range(
        0,
        nfail,
        args.batch,
    ):

        stop = min(
            nfail,
            start + args.batch,
        )

        br = rr[
            start:stop
        ]

        bc = cc[
            start:stop
        ]

        support, _ = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=args.alpha,
                nslc=ndate,
                block_size=args.support_block,
            )
        )

        # 253 spatial offsets only.
        for ky in range(
            wh
        ):

            dr = (
                ky
                -
                args.half_row
            )

            r2 = (
                br
                +
                dr
            )

            valid_r = (
                (r2 >= 0)
                &
                (r2 < H)
            )

            if not np.any(
                valid_r
            ):
                continue

            for kx in range(
                ww
            ):

                dc = (
                    kx
                    -
                    args.half_col
                )

                c2 = (
                    bc
                    +
                    dc
                )

                inside = (
                    valid_r
                    &
                    (c2 >= 0)
                    &
                    (c2 < W)
                    &
                    support[
                        :,
                        ky,
                        kx,
                    ]
                )

                if not np.any(
                    inside
                ):
                    continue

                ids = np.flatnonzero(
                    inside
                )

                r3 = r2[
                    ids
                ]

                c3 = c2[
                    ids
                ]

                good = (
                    low[
                        r3,
                        c3,
                    ]
                    &
                    (
                        center_k[
                            r3,
                            c3,
                        ]
                        >=
                        chosen_threshold
                    )
                )

                if np.any(
                    good
                ):

                    rescue_mask[
                        r3[good],
                        c3[good],
                    ] = True

        del support

    n_local_rescue = int(
        np.count_nonzero(
            rescue_mask
        )
    )

    # --------------------------------------------------------
    # Verify LOCAL mask itself is sufficient.
    # --------------------------------------------------------

    rescue_pad = np.pad(
        rescue_mask,
        pad,
        mode="constant",
        constant_values=False,
    )

    rescue_windows = (
        np.lib.stride_tricks
        .sliding_window_view(
            rescue_pad,
            (
                wh,
                ww,
            ),
        )
    )

    local_remaining = np.zeros(
        nfail,
        dtype=np.bool_,
    )

    local_effective_k = np.zeros(
        nfail,
        dtype=np.int16,
    )

    for start in range(
        0,
        nfail,
        args.batch,
    ):

        stop = min(
            nfail,
            start + args.batch,
        )

        br = rr[
            start:stop
        ]

        bc = cc[
            start:stop
        ]

        support, _ = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=args.alpha,
                nslc=ndate,
                block_size=args.support_block,
            )
        )

        rescue_local = np.asarray(
            rescue_windows[
                br,
                bc,
            ],
            dtype=np.bool_,
        )

        nreturn = np.sum(
            support
            &
            rescue_local,
            axis=(1, 2),
            dtype=np.int32,
        )

        effective = (
            base_k[
                start:stop
            ].astype(
                np.int32
            )
            +
            nreturn
        )

        local_effective_k[
            start:stop
        ] = effective.astype(
            np.int16
        )

        local_remaining[
            start:stop
        ] = (
            effective
            <
            args.min_shp
        )

        del support
        del rescue_local

    n_local_remaining = int(
        np.count_nonzero(
            local_remaining
        )
    )

    if n_local_remaining:
        raise RuntimeError(
            "Local rescue-mask verification failed: "
            f"{n_local_remaining} centers still below "
            f"{args.min_shp}"
        )

    # --------------------------------------------------------
    # Quality distribution of rescued states
    # --------------------------------------------------------

    rescue_k = np.asarray(
        center_k[
            rescue_mask
        ],
        dtype=np.int16,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    rescue_path = (
        seqdir
        /
        "compression_local_rescue_mask.npy"
    )

    state_valid_path = (
        seqdir
        /
        "compression_state_valid_candidate_mask.npy"
    )

    # Candidate production state:
    #
    # original direct K>=48 state
    # +
    # only locally-required fallback state.
    #
    direct_state = (
        req
        &
        (
            np.asarray(
                center_k
            )
            >=
            args.min_shp
        )
    )

    state_valid_candidate = (
        direct_state
        |
        rescue_mask
    )

    np.save(
        rescue_path,
        rescue_mask,
    )

    np.save(
        state_valid_path,
        state_valid_candidate,
    )

    report = {
        "format":
            "pyPSDS-GAMMA-local-state-rescue-threshold-v1",

        "shape":
            [H, W],

        "ndate":
            ndate,

        "output_min_shp":
            args.min_shp,

        "low_k_state_candidates":
            n_low,

        "dense_fail_input":
            n_dense_fail,

        "target_fail_input":
            n_target_fail,

        "fail_union_input":
            nfail,

        "chosen_fallback_min_shp":
            chosen_threshold,

        "local_rescue_state_count":
            n_local_rescue,

        "local_rescue_fraction_low_k":
            (
                n_local_rescue
                /
                n_low
                if n_low
                else 0.0
            ),

        "local_rescue_K_quantiles":
            quantiles(
                rescue_k
            ),

        "local_verification_fail_remaining":
            n_local_remaining,

        "effective_K_after_local_rescue_quantiles":
            quantiles(
                local_effective_k
            ),

        "threshold_results":
            results,

        "elapsed_seconds":
            elapsed,

        "decision":
            "local_quality_ranked_rescue_candidate",

        "recommendation":
            (
                "Use the selected fallback K threshold only "
                "for compression-state pixels required to "
                "protect centers that failed the strict K>=48 "
                "state-drop audit. Do not change the DS-output "
                "min_shp=48 criterion. Before enabling the "
                "sequential executor, validate phase-linking "
                "quality and stage-to-stage state validity for "
                "this rescue population."
            ),
    }

    json_path = (
        seqdir
        /
        "compression_local_rescue_threshold.json"
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
        "U3.2c3 threshold ladder"
    )
    print("=" * 88)

    print(
        " Kmin | candidates | dense remain | "
        "target remain | union remain"
    )

    print("-" * 88)

    for item in results:

        t = item[
            "threshold"
        ]

        # Print all important thresholds plus neighborhood
        # around selected threshold.
        if (
            t in (
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
                1,
            )
            or
            abs(
                t
                -
                chosen_threshold
            )
            <=
            2
        ):

            print(
                f"{t:5d} | "
                f"{item['candidate_state_count']:10,d} | "
                f"{item['dense_fail_remaining']:12,d} | "
                f"{item['target_fail_remaining']:13,d} | "
                f"{item['union_fail_remaining']:11,d}"
            )

    print()
    print("=" * 88)
    print(
        "U3.2c3 result"
    )
    print("=" * 88)

    print(
        "chosen fallback Kmin        :",
        chosen_threshold,
    )

    print(
        "low-K candidates total      :",
        f"{n_low:,}",
    )

    print(
        "local rescue states         :",
        f"{n_local_rescue:,}",
        f"({pct(n_local_rescue, n_low):.3f}%)",
    )

    print(
        "rescue-state K quantiles    :",
        quantiles(
            rescue_k
        ),
    )

    print(
        "failed centers after rescue :",
        n_local_remaining,
    )

    print(
        "effective K quantiles       :",
        quantiles(
            local_effective_k
        ),
    )

    print()

    print(
        "rescue mask :",
        rescue_path,
    )

    print(
        "state mask  :",
        state_valid_path,
    )

    print(
        "json        :",
        json_path,
    )

    print()

    print(
        "U3.2c3 LOCAL STATE RESCUE THRESHOLD AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
