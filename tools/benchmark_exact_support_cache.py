#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.context import (
    open_from_config,
)

from pypsds.phase_linking.coherence import (
    compressed_coherence,
)

from pypsds.phase_linking.emi import (
    image_pairs,
)

from pypsds.phase_linking.shp_coherence_bitset import (
    compressed_coherence_bitset,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)

from pypsds.phase_linking.support_cache import (
    bool_windows,
    pack_support_bool,
    popcount_support_bits,
    unpack_support_bits,
)


def best_time(
    fn,
    repeat,
):

    times = []
    result = None

    for _ in range(
        max(
            1,
            repeat,
        )
    ):

        t0 = perf_counter()

        result = fn()

        times.append(
            perf_counter()
            -
            t0
        )

    return (
        min(
            times
        ),
        result,
    )


def coherence_diff(
    a,
    b,
):

    fa = (
        np.isfinite(
            a.real
        )
        &
        np.isfinite(
            a.imag
        )
    )

    fb = (
        np.isfinite(
            b.real
        )
        &
        np.isfinite(
            b.imag
        )
    )

    finite_mismatch = int(
        np.count_nonzero(
            fa
            !=
            fb
        )
    )

    good = (
        fa
        &
        fb
    )

    if np.any(
        good
    ):

        d = float(
            np.max(
                np.abs(
                    np.asarray(
                        a[good],
                        dtype=np.complex128,
                    )
                    -
                    np.asarray(
                        b[good],
                        dtype=np.complex128,
                    )
                )
            )
        )

    else:

        d = 0.0

    return (
        finite_mismatch,
        d,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--sample",
        type=int,
        default=64000,
    )

    ap.add_argument(
        "--fullspan-sample",
        type=int,
        default=24000,
    )

    ap.add_argument(
        "--repeat",
        type=int,
        default=2,
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
        (_, _, H, W),
    ) = open_from_config(
        args.config
    )


    ndate = len(
        stack.dates
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

    cache_dir = (
        processing
        /
        "exact_support_cache"
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

    static_bits_map = np.load(
        cache_dir
        /
        "static_support_bits.npy",
        mmap_mode="r",
    )

    static_k_map = np.load(
        cache_dir
        /
        "static_shp_count.npy",
        mmap_mode="r",
    )

    state_core = np.load(
        processing
        /
        "sequential"
        /
        "compression_state_core_K24.npy",
        mmap_mode="r",
    )

    comp0 = np.load(
        processing
        /
        "sequential"
        /
        "sequential_stage0000_compressed.npy",
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


    ctx = (
        prepare_glrt_window_context(
            scale2,
            valid,
            ps,
            half_row=(
                args.half_row
            ),
            half_col=(
                args.half_col
            ),
        )
    )


    core_windows = bool_windows(
        state_core,
        half_row=args.half_row,
        half_col=args.half_col,
    )


    all_r, all_c = np.where(
        state_core
    )


    sample_n = min(
        args.sample,
        all_r.size,
    )


    ids = np.linspace(
        0,
        all_r.size - 1,
        sample_n,
        dtype=np.int64,
    )


    rr = all_r[
        ids
    ].astype(
        np.int32,
        copy=False,
    )

    cc = all_c[
        ids
    ].astype(
        np.int32,
        copy=False,
    )


    print("=" * 100)
    print(
        "pyPSDS-GAMMA P8D1 exact support-cache benchmark"
    )
    print("=" * 100)

    print(
        "scene          :",
        f"{H} x {W}",
    )

    print(
        "dates          :",
        ndate,
    )

    print(
        "sample         :",
        sample_n,
    )

    print(
        "fullspan sample:",
        min(
            args.fullspan_sample,
            sample_n,
        ),
    )

    print(
        "cache dtype    :",
        static_bits_map.dtype,
    )

    print(
        "cache shape    :",
        static_bits_map.shape,
    )

    print()


    results = []


    def run_case(
        *,
        label,
        zstack,
        stage_valid,
        rows,
        cols,
    ):

        solver_n = (
            zstack.shape[2]
        )

        pairs = image_pairs(
            solver_n
        )

        pi = pairs[
            :,
            0
        ].astype(
            np.int32,
            copy=False,
        )

        pj = pairs[
            :,
            1
        ].astype(
            np.int32,
            copy=False,
        )


        # --------------------------------------------------------------
        # Dynamic support mask.
        # --------------------------------------------------------------

        dynamic = np.asarray(
            core_windows[
                rows,
                cols,
            ],
            dtype=np.bool_,
        ).copy()


        if stage_valid is not None:

            stage_windows = (
                bool_windows(
                    stage_valid,
                    half_row=(
                        args.half_row
                    ),
                    half_col=(
                        args.half_col
                    ),
                )
            )

            dynamic &= np.asarray(
                stage_windows[
                    rows,
                    cols,
                ],
                dtype=np.bool_,
            )


        # --------------------------------------------------------------
        # Current exact path
        # --------------------------------------------------------------

        def current_support():

            support, _ = (
                glrt_support_vectorized_exact(
                    ctx,
                    rows,
                    cols,
                    alpha=args.alpha,
                    nslc=ndate,
                    block_size=(
                        args.support_block
                    ),
                )
            )

            support &= dynamic

            K = np.sum(
                support,
                axis=(1, 2),
                dtype=np.int32,
            ).astype(
                np.int16
            )

            return (
                support,
                K,
            )


        cur_support_s, (
            cur_support,
            cur_k,
        ) = best_time(
            current_support,
            args.repeat,
        )


        def current_coh():

            return compressed_coherence(
                zstack,
                rows,
                cols,
                cur_support,
                pi,
                pj,
            )


        cur_coh_s, cur_coh = (
            best_time(
                current_coh,
                args.repeat,
            )
        )


        # --------------------------------------------------------------
        # Cached exact bits -> bool -> SAME coherence kernel
        # --------------------------------------------------------------

        static_bits = np.asarray(
            static_bits_map[
                rows,
                cols,
                :,
            ],
            dtype=np.uint64,
        )


        def cache_bool_support():

            support = (
                unpack_support_bits(
                    static_bits,
                    half_row=(
                        args.half_row
                    ),
                    half_col=(
                        args.half_col
                    ),
                )
            )

            support &= dynamic

            K = np.sum(
                support,
                axis=(1, 2),
                dtype=np.int32,
            ).astype(
                np.int16
            )

            return (
                support,
                K,
            )


        cb_support_s, (
            cb_support,
            cb_k,
        ) = best_time(
            cache_bool_support,
            args.repeat,
        )


        np.testing.assert_array_equal(
            cur_support,
            cb_support,
        )

        np.testing.assert_array_equal(
            cur_k,
            cb_k,
        )


        def cache_bool_coh():

            return compressed_coherence(
                zstack,
                rows,
                cols,
                cb_support,
                pi,
                pj,
            )


        cb_coh_s, cb_coh = (
            best_time(
                cache_bool_coh,
                args.repeat,
            )
        )


        fb, db = coherence_diff(
            cur_coh,
            cb_coh,
        )

        if (
            fb != 0
            or
            db != 0.0
        ):
            raise RuntimeError(
                f"{label}: cache_bool "
                "coherence is not exact"
            )


        # --------------------------------------------------------------
        # Cached exact bits -> packed dynamic mask -> bitset coherence
        # --------------------------------------------------------------

        def cache_bitset_support():

            dynamic_bits = (
                pack_support_bool(
                    dynamic
                )
            )

            bits = (
                static_bits
                &
                dynamic_bits
            )

            K = (
                popcount_support_bits(
                    bits
                )
            )

            return (
                bits,
                K,
            )


        bs_support_s, (
            stage_bits,
            bs_k,
        ) = best_time(
            cache_bitset_support,
            args.repeat,
        )


        np.testing.assert_array_equal(
            cur_k,
            bs_k,
        )


        # Warm Numba compilation outside timing.
        warm_n = min(
            256,
            rows.size,
        )

        _ = compressed_coherence_bitset(
            zstack,
            rows[
                :warm_n
            ],
            cols[
                :warm_n
            ],
            stage_bits[
                :warm_n
            ],
            pi,
            pj,
            half_row=(
                args.half_row
            ),
            half_col=(
                args.half_col
            ),
        )


        def cache_bitset_coh():

            return (
                compressed_coherence_bitset(
                    zstack,
                    rows,
                    cols,
                    stage_bits,
                    pi,
                    pj,
                    half_row=(
                        args.half_row
                    ),
                    half_col=(
                        args.half_col
                    ),
                )
            )


        bs_coh_s, bs_coh = (
            best_time(
                cache_bitset_coh,
                args.repeat,
            )
        )


        bit_finite_bad, bit_diff = (
            coherence_diff(
                cur_coh,
                bs_coh,
            )
        )


        # --------------------------------------------------------------
        # Existing static K also has to agree BEFORE dynamic filtering.
        # --------------------------------------------------------------

        cached_static_k = (
            popcount_support_bits(
                static_bits
            )
        )

        expected_static_k = np.asarray(
            static_k_map[
                rows,
                cols,
            ],
            dtype=np.int16,
        )

        np.testing.assert_array_equal(
            cached_static_k,
            expected_static_k,
        )


        cur_total = (
            cur_support_s
            +
            cur_coh_s
        )

        cb_total = (
            cb_support_s
            +
            cb_coh_s
        )

        bs_total = (
            bs_support_s
            +
            bs_coh_s
        )


        out = {
            "label":
                label,

            "solver_n":
                int(
                    solver_n
                ),

            "points":
                int(
                    rows.size
                ),

            "current_support_s":
                cur_support_s,

            "current_coherence_s":
                cur_coh_s,

            "current_total_s":
                cur_total,

            "cache_bool_support_s":
                cb_support_s,

            "cache_bool_coherence_s":
                cb_coh_s,

            "cache_bool_total_s":
                cb_total,

            "cache_bool_speedup":
                (
                    cur_total
                    /
                    cb_total
                ),

            "bitset_support_s":
                bs_support_s,

            "bitset_coherence_s":
                bs_coh_s,

            "bitset_total_s":
                bs_total,

            "bitset_speedup":
                (
                    cur_total
                    /
                    bs_total
                ),

            "bitset_finite_mismatch":
                bit_finite_bad,

            "bitset_coherence_max_diff":
                bit_diff,

            "bitset_parity":
                (
                    bit_finite_bad
                    ==
                    0
                    and
                    bit_diff
                    <=
                    1e-6
                ),
        }


        print()
        print("-" * 100)
        print(label)
        print("-" * 100)

        print(
            "solver size             :",
            solver_n,
        )

        print(
            "points                  :",
            f"{rows.size:,}",
        )

        print()
        print(
            "CURRENT exact support   :",
            f"{cur_support_s:.3f} s",
        )

        print(
            "CURRENT coherence       :",
            f"{cur_coh_s:.3f} s",
        )

        print(
            "CURRENT support+coh     :",
            f"{cur_total:.3f} s",
        )

        print()
        print(
            "CACHE bool support      :",
            f"{cb_support_s:.3f} s",
        )

        print(
            "CACHE bool coherence    :",
            f"{cb_coh_s:.3f} s",
        )

        print(
            "CACHE bool total        :",
            f"{cb_total:.3f} s",
        )

        print(
            "CACHE bool speedup      :",
            f"{out['cache_bool_speedup']:.3f}x",
        )

        print(
            "CACHE bool parity       : EXACT",
        )

        print()
        print(
            "BITSET dynamic support  :",
            f"{bs_support_s:.3f} s",
        )

        print(
            "BITSET coherence        :",
            f"{bs_coh_s:.3f} s",
        )

        print(
            "BITSET total            :",
            f"{bs_total:.3f} s",
        )

        print(
            "BITSET speedup          :",
            f"{out['bitset_speedup']:.3f}x",
        )

        print(
            "BITSET finite mismatch  :",
            bit_finite_bad,
        )

        print(
            "BITSET coherence diff   :",
            f"{bit_diff:.3e}",
        )

        print(
            "BITSET parity           :",
            out[
                "bitset_parity"
            ],
        )


        del (
            dynamic,
            cur_support,
            cur_coh,
            cb_support,
            cb_coh,
            static_bits,
            stage_bits,
            bs_coh,
        )

        gc.collect()

        return out


    # ==================================================================
    # Stage 0: 19 real SLCs
    # ==================================================================

    stage0 = np.ascontiguousarray(
        yxt[
            :,
            :,
            0:19,
        ],
        dtype=np.complex64,
    )

    stage0_valid = np.all(
        np.isfinite(
            stage0.real
        )
        &
        np.isfinite(
            stage0.imag
        ),
        axis=2,
    )

    results.append(
        run_case(
            label="stage0_M19",
            zstack=stage0,
            stage_valid=stage0_valid,
            rows=rr,
            cols=cc,
        )
    )

    del stage0
    del stage0_valid

    gc.collect()


    # ==================================================================
    # Stage 1:
    #   c0000 + real acquisitions 19:38
    # ==================================================================

    stage1 = np.empty(
        (
            H,
            W,
            20,
        ),
        dtype=np.complex64,
    )

    stage1[
        :,
        :,
        0,
    ] = comp0

    stage1[
        :,
        :,
        1:,
    ] = yxt[
        :,
        :,
        19:38,
    ]

    stage1_valid = np.all(
        np.isfinite(
            stage1.real
        )
        &
        np.isfinite(
            stage1.imag
        ),
        axis=2,
    )

    results.append(
        run_case(
            label="stage1_M20",
            zstack=stage1,
            stage_valid=stage1_valid,
            rows=rr,
            cols=cc,
        )
    )

    del stage1
    del stage1_valid

    gc.collect()


    # ==================================================================
    # Full-span quality:
    # static exact support & K24 core only, N=38.
    # ==================================================================

    nf = min(
        args.fullspan_sample,
        rr.size,
    )

    fi = np.linspace(
        0,
        rr.size - 1,
        nf,
        dtype=np.int64,
    )

    fr = rr[
        fi
    ]

    fc = cc[
        fi
    ]

    results.append(
        run_case(
            label="fullspan_N38",
            zstack=yxt,
            stage_valid=None,
            rows=fr,
            cols=fc,
        )
    )


    # ==================================================================
    # Decision
    # ==================================================================

    safe_bool = all(
        x[
            "cache_bool_speedup"
        ]
        >
        1.0
        for x in results
    )

    bitset_parity = all(
        x[
            "bitset_parity"
        ]
        for x in results
    )

    bool_min = min(
        x[
            "cache_bool_speedup"
        ]
        for x in results
    )

    bitset_min = min(
        x[
            "bitset_speedup"
        ]
        for x in results
    )


    bitset_faster_than_bool = all(
        x[
            "bitset_total_s"
        ]
        <
        x[
            "cache_bool_total_s"
        ]
        for x in results
    )


    report = {
        "format":
            "pyPSDS-GAMMA-P8D1-support-cache-benchmark-v1",

        "ndate":
            ndate,

        "sample":
            sample_n,

        "results":
            results,

        "safe_bool_candidate":
            safe_bool,

        "safe_bool_min_speedup":
            bool_min,

        "bitset_parity":
            bitset_parity,

        "bitset_min_speedup":
            bitset_min,

        "bitset_faster_than_bool":
            bitset_faster_than_bool,
    }


    out = (
        cache_dir
        /
        "p8d1_benchmark.json"
    )

    out.write_text(
        json.dumps(
            report,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )


    print()
    print("=" * 100)
    print("P8D1 DECISION")
    print("=" * 100)

    print(
        "cache_bool minimum speedup :",
        f"{bool_min:.3f}x",
    )

    print(
        "cache_bool exact parity    :",
        True,
    )

    print(
        "bitset minimum speedup     :",
        f"{bitset_min:.3f}x",
    )

    print(
        "bitset parity              :",
        bitset_parity,
    )

    print(
        "bitset faster than bool    :",
        bitset_faster_than_bool,
    )

    print()


    if (
        safe_bool
        and
        bool_min >= 1.05
    ):

        print(
            "SAFE CACHE PROMOTION      : YES"
        )

    else:

        print(
            "SAFE CACHE PROMOTION      : NO"
        )


    if (
        bitset_parity
        and
        bitset_faster_than_bool
        and
        bitset_min >= 1.05
    ):

        print(
            "BITSET COHERENCE CANDIDATE: YES"
        )

    else:

        print(
            "BITSET COHERENCE CANDIDATE: NO"
        )


    print()
    print(
        "saved                     :",
        out,
    )

    print()
    print(
        "P8D1 EXACT SUPPORT CACHE AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
