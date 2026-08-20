#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config

from pypsds.gamma.phase_correction import (
    GammaPointPhaseCorrectionProvider,
)


def corrected_yxt(
    provider,
    stack,
    *,
    global_row0,
    global_col0,
    rows,
    cols,
    label,
):
    t0 = time.perf_counter()

    raw = stack.read_window(
        row0=global_row0,
        col0=global_col0,
        rows=rows,
        cols=cols,
    ).astype(
        np.complex64,
        copy=False,
    )

    read_s = (
        time.perf_counter()
        -
        t0
    )

    (
        corrected,
        valid,
        stats,
    ) = provider.correct_block(
        raw,
        global_row0=global_row0,
        global_col0=global_col0,
        tile_label=label,
    )

    yxt = np.ascontiguousarray(
        np.moveaxis(
            corrected,
            0,
            -1,
        ),
        dtype=np.complex64,
    )

    del (
        corrected,
        raw,
    )

    return (
        yxt,
        np.ascontiguousarray(
            valid,
            dtype=np.bool_,
        ),
        read_s,
        stats,
    )


def compare(
    name,
    a,
    a_valid,
    b,
    b_valid,
):
    valid_bad = int(
        np.count_nonzero(
            a_valid
            !=
            b_valid
        )
    )

    both = (
        a_valid
        &
        b_valid
    )

    both3 = (
        both[
            :,
            :,
            None,
        ]
        &
        np.isfinite(
            a.real
        )
        &
        np.isfinite(
            a.imag
        )
        &
        np.isfinite(
            b.real
        )
        &
        np.isfinite(
            b.imag
        )
    )

    if np.any(
        both3
    ):

        av = a[
            both3
        ]

        bv = b[
            both3
        ]

        exact = bool(
            np.array_equal(
                av,
                bv,
            )
        )

        max_complex = float(
            np.max(
                np.abs(
                    av
                    -
                    bv
                )
            )
        )

        nonzero = (
            (np.abs(av) > 0)
            &
            (np.abs(bv) > 0)
        )

        if np.any(
            nonzero
        ):

            phase_diff = np.angle(
                av[
                    nonzero
                ]
                *
                np.conj(
                    bv[
                        nonzero
                    ]
                )
            )

            max_phase = float(
                np.max(
                    np.abs(
                        phase_diff
                    )
                )
            )

        else:

            max_phase = 0.0

    else:

        exact = True
        max_complex = 0.0
        max_phase = 0.0

    result = {
        "name":
            name,

        "geometry_mismatch":
            valid_bad,

        "complex_exact":
            exact,

        "max_abs_complex_difference":
            max_complex,

        "max_abs_phase_difference_rad":
            max_phase,
    }

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    # Large enough to cross several canonical cache tiles,
    # but small enough for rapid diagnosis.
    ap.add_argument(
        "--row0",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--col0",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--rows",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--cols",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--canonical-rows",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--canonical-cols",
        type=int,
        default=256,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (
            base_row0,
            base_col0,
            H,
            W,
        ),
    ) = open_from_config(
        args.config
    )

    ndate = len(
        stack.dates
    )

    r0 = args.row0
    c0 = args.col0

    r1 = r0 + args.rows
    c1 = c0 + args.cols

    if (
        r0 < 0
        or
        c0 < 0
        or
        r1 > H
        or
        c1 > W
    ):
        raise RuntimeError(
            "audit region outside ROI"
        )

    processing = (
        Path(
            paths.output_dir
        )
        /
        "processing"
    )

    cache_path = (
        processing
        /
        "cache"
        /
        "phase_corrected_yxt.npy"
    )

    geom_path = (
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy"
    )

    if not cache_path.is_file():
        raise FileNotFoundError(
            cache_path
        )

    if not geom_path.is_file():
        raise FileNotFoundError(
            geom_path
        )

    cache = np.load(
        cache_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    cache_geom = np.load(
        geom_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    reference = np.ascontiguousarray(
        cache[
            r0:r1,
            c0:c1,
            :
        ],
        dtype=np.complex64,
    )

    reference_geom = (
        np.ascontiguousarray(
            cache_geom[
                r0:r1,
                c0:c1,
            ],
            dtype=np.bool_,
        )
    )

    provider = (
        GammaPointPhaseCorrectionProvider(
            cfg,
            paths,
            stack,
        )
    )

    provider.prepare()

    workers = (
        provider.phase_sim_worker_count(
            max(
                0,
                ndate - 1,
            )
        )
    )

    print(
        "=" * 92
    )

    print(
        "pyPSDS-GAMMA phase-correction tiling parity audit"
    )

    print(
        "=" * 92
    )

    print(
        "region            :",
        f"{r0}:{r1}, {c0}:{c1}",
    )

    print(
        "shape             :",
        f"{args.rows} x {args.cols}",
    )

    print(
        "dates             :",
        ndate,
    )

    print(
        "phase_sim workers :",
        workers,
    )

    print(
        "canonical tile    :",
        f"{args.canonical_rows} x "
        f"{args.canonical_cols}",
    )

    print()

    # -------------------------------------------------------
    # A. Entire region as one phase-correction point set.
    # -------------------------------------------------------

    print(
        "A. one-shot correction"
    )

    (
        one_yxt,
        one_geom,
        one_read,
        one_stats,
    ) = corrected_yxt(
        provider,
        stack,
        global_row0=(
            base_row0
            +
            r0
        ),
        global_col0=(
            base_col0
            +
            c0
        ),
        rows=args.rows,
        cols=args.cols,
        label=(
            f"audit_oneshot_"
            f"r{r0}_{r1}_"
            f"c{c0}_{c1}"
        ),
    )

    print(
        "   read           :",
        f"{one_read:.3f} s",
    )

    print(
        "   phase_sim      :",
        f"{one_stats.simulation_seconds:.3f} s",
    )

    # -------------------------------------------------------
    # B. Same region assembled from the cache builder's
    #    canonical 128x256 correction tiles.
    # -------------------------------------------------------

    print()
    print(
        "B. canonical-tile mosaic"
    )

    canonical_yxt = np.empty(
        (
            args.rows,
            args.cols,
            ndate,
        ),
        dtype=np.complex64,
    )

    canonical_geom = np.zeros(
        (
            args.rows,
            args.cols,
        ),
        dtype=np.bool_,
    )

    canonical_seconds = 0.0

    tile_count = 0

    for lr0 in range(
        0,
        args.rows,
        args.canonical_rows,
    ):

        lr1 = min(
            args.rows,
            lr0
            +
            args.canonical_rows,
        )

        for lc0 in range(
            0,
            args.cols,
            args.canonical_cols,
        ):

            lc1 = min(
                args.cols,
                lc0
                +
                args.canonical_cols,
            )

            (
                cur_yxt,
                cur_geom,
                read_s,
                stats,
            ) = corrected_yxt(
                provider,
                stack,
                global_row0=(
                    base_row0
                    +
                    r0
                    +
                    lr0
                ),
                global_col0=(
                    base_col0
                    +
                    c0
                    +
                    lc0
                ),
                rows=(
                    lr1
                    -
                    lr0
                ),
                cols=(
                    lc1
                    -
                    lc0
                ),
                label=(
                    f"audit_canonical_"
                    f"r{r0+lr0}_{r0+lr1}_"
                    f"c{c0+lc0}_{c0+lc1}"
                ),
            )

            canonical_yxt[
                lr0:lr1,
                lc0:lc1,
                :
            ] = cur_yxt

            canonical_geom[
                lr0:lr1,
                lc0:lc1,
            ] = cur_geom

            canonical_seconds += (
                read_s
                +
                stats.total_seconds
            )

            tile_count += 1

    print(
        "   tiles          :",
        tile_count,
    )

    print(
        "   elapsed approx :",
        f"{canonical_seconds:.3f} s",
    )

    # -------------------------------------------------------
    # Comparisons
    # -------------------------------------------------------

    results = [
        compare(
            "one_shot_vs_cache",
            one_yxt,
            one_geom,
            reference,
            reference_geom,
        ),

        compare(
            "canonical_vs_cache",
            canonical_yxt,
            canonical_geom,
            reference,
            reference_geom,
        ),

        compare(
            "one_shot_vs_canonical",
            one_yxt,
            one_geom,
            canonical_yxt,
            canonical_geom,
        ),
    ]

    print()

    print(
        "=" * 92
    )

    print(
        "PARITY RESULTS"
    )

    print(
        "=" * 92
    )

    for x in results:

        print()

        print(
            x[
                "name"
            ]
        )

        print(
            "  geometry mismatch :",
            x[
                "geometry_mismatch"
            ],
        )

        print(
            "  complex exact     :",
            x[
                "complex_exact"
            ],
        )

        print(
            "  max complex diff  :",
            f"{x['max_abs_complex_difference']:.9e}",
        )

        print(
            "  max phase diff    :",
            f"{x['max_abs_phase_difference_rad']:.9e}",
            "rad",
        )

    report = {
        "format":
            "pyPSDS-GAMMA-phase-correction-tiling-audit-v1",

        "region":
            [
                r0,
                r1,
                c0,
                c1,
            ],

        "canonical_tile":
            [
                args.canonical_rows,
                args.canonical_cols,
            ],

        "ndate":
            ndate,

        "phase_sim_workers":
            workers,

        "results":
            results,
    }

    out = (
        processing
        /
        "phase_correction_tiling_audit.json"
    )

    out.write_text(
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
    print(
        "saved:",
        out,
    )


if __name__ == "__main__":
    main()
