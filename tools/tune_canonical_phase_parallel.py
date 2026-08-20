#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config
from pypsds.gamma.phase_correction import (
    GammaPointPhaseCorrectionProvider,
)


def max_phase_diff(a, b, valid):
    m = (
        valid[:, :, None]
        &
        np.isfinite(a.real)
        &
        np.isfinite(a.imag)
        &
        np.isfinite(b.real)
        &
        np.isfinite(b.imag)
        &
        (np.abs(a) > 0)
        &
        (np.abs(b) > 0)
    )

    if not np.any(m):
        return 0.0

    d = np.angle(
        a[m]
        *
        np.conj(
            b[m]
        )
    )

    return float(
        np.max(
            np.abs(d)
        )
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

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
        "--tile-rows",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--tile-cols",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--grid-rows",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--grid-cols",
        type=int,
        default=4,
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

    R = (
        args.grid_rows
        *
        args.tile_rows
    )

    C = (
        args.grid_cols
        *
        args.tile_cols
    )

    r0 = args.row0
    c0 = args.col0

    r1 = r0 + R
    c1 = c0 + C

    if (
        r1 > H
        or
        c1 > W
    ):
        raise RuntimeError(
            "benchmark region outside scene"
        )

    processing = (
        Path(paths.output_dir)
        /
        "processing"
    )

    cache = np.load(
        processing
        /
        "cache"
        /
        "phase_corrected_yxt.npy",
        mmap_mode="r",
    )

    geom_cache = np.load(
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    reference = np.ascontiguousarray(
        cache[
            r0:r1,
            c0:c1,
            :
        ],
        dtype=np.complex64,
    )

    reference_geom = np.ascontiguousarray(
        geom_cache[
            r0:r1,
            c0:c1
        ],
        dtype=np.bool_,
    )

    # --------------------------------------------------------
    # Canonical tasks.
    # --------------------------------------------------------

    tasks = []

    tid = 0

    for ir in range(
        args.grid_rows
    ):
        for ic in range(
            args.grid_cols
        ):

            lr0 = (
                ir
                *
                args.tile_rows
            )

            lc0 = (
                ic
                *
                args.tile_cols
            )

            lr1 = (
                lr0
                +
                args.tile_rows
            )

            lc1 = (
                lc0
                +
                args.tile_cols
            )

            tasks.append(
                (
                    tid,
                    lr0,
                    lr1,
                    lc0,
                    lc1,
                )
            )

            tid += 1

    # --------------------------------------------------------
    # Pre-read all RSLC tiles.
    #
    # This autotune is deliberately measuring GAMMA correction
    # scheduling, not filesystem-cache randomness.
    # --------------------------------------------------------

    raw_tiles = {}

    print(
        "=" * 92
    )

    print(
        "pyPSDS-GAMMA canonical phase parallel autotune"
    )

    print(
        "=" * 92
    )

    print(
        "dates             :",
        ndate,
    )

    print(
        "canonical tile    :",
        f"{args.tile_rows} x "
        f"{args.tile_cols}",
    )

    print(
        "test grid         :",
        f"{args.grid_rows} x "
        f"{args.grid_cols}",
    )

    print(
        "canonical tiles   :",
        len(tasks),
    )

    print(
        "test region       :",
        f"{R} x {C}",
    )

    print()

    print(
        "Pre-reading RSLC canonical tiles..."
    )

    tr = time.perf_counter()

    for (
        tid,
        lr0,
        lr1,
        lc0,
        lc1,
    ) in tasks:

        raw_tiles[tid] = (
            stack.read_window(
                row0=(
                    base_row0
                    +
                    r0
                    +
                    lr0
                ),
                col0=(
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
            )
            .astype(
                np.complex64,
                copy=False,
            )
        )

    print(
        "RSLC pre-read      :",
        f"{time.perf_counter()-tr:.3f} s",
    )

    # --------------------------------------------------------
    # spatial_workers, pair_workers
    # --------------------------------------------------------

    layouts = [
        (1, 16),
        (2, 8),
        (4, 4),
        (8, 2),

        # Use all / most 32 CPUs.
        (2, 16),
        (4, 8),
        (8, 4),
    ]

    results = []

    print()

    print(
        f"{'spatial':>8s}"
        f"{'pair':>8s}"
        f"{'maxproc':>10s}"
        f"{'seconds':>12s}"
        f"{'tiles/s':>12s}"
        f"{'exact':>10s}"
        f"{'phase':>14s}"
    )

    print(
        "-" * 82
    )

    for (
        spatial_workers,
        pair_workers,
    ) in layouts:

        provider = (
            GammaPointPhaseCorrectionProvider(
                cfg,
                paths,
                stack,
            )
        )

        provider.prepare()

        # Explicit benchmark override.
        provider.phase_sim_worker_count = (
            lambda n_pairs, pw=pair_workers:
                min(
                    pw,
                    n_pairs,
                )
        )

        result_yxt = np.empty(
            (
                R,
                C,
                ndate,
            ),
            dtype=np.complex64,
        )

        result_geom = np.zeros(
            (
                R,
                C,
            ),
            dtype=np.bool_,
        )

        def worker(task):

            (
                tid,
                lr0,
                lr1,
                lc0,
                lc1,
            ) = task

            raw = raw_tiles[
                tid
            ]

            (
                corrected,
                valid,
                stats,
            ) = provider.correct_block(
                raw,
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
                tile_label=(
                    f"canonical_tune_"
                    f"s{spatial_workers}_"
                    f"p{pair_workers}_"
                    f"tile{tid:03d}"
                ),
            )

            yxt = np.ascontiguousarray(
                np.moveaxis(
                    corrected,
                    0,
                    -1,
                ),
                dtype=np.complex64,
            )

            return (
                tid,
                lr0,
                lr1,
                lc0,
                lc1,
                yxt,
                np.ascontiguousarray(
                    valid,
                    dtype=np.bool_,
                ),
            )

        t0 = time.perf_counter()

        with ThreadPoolExecutor(
            max_workers=spatial_workers,
            thread_name_prefix=(
                "pypsds-canonical"
            ),
        ) as ex:

            futures = [
                ex.submit(
                    worker,
                    task,
                )
                for task in tasks
            ]

            for fut in as_completed(
                futures
            ):

                (
                    tid,
                    lr0,
                    lr1,
                    lc0,
                    lc1,
                    yxt,
                    valid,
                ) = fut.result()

                result_yxt[
                    lr0:lr1,
                    lc0:lc1,
                    :
                ] = yxt

                result_geom[
                    lr0:lr1,
                    lc0:lc1,
                ] = valid

        seconds = (
            time.perf_counter()
            -
            t0
        )

        geom_bad = int(
            np.count_nonzero(
                result_geom
                !=
                reference_geom
            )
        )

        finite = (
            reference_geom[
                :,
                :,
                None,
            ]
            &
            np.isfinite(
                reference.real
            )
            &
            np.isfinite(
                reference.imag
            )
            &
            np.isfinite(
                result_yxt.real
            )
            &
            np.isfinite(
                result_yxt.imag
            )
        )

        if np.any(
            finite
        ):

            complex_exact = bool(
                np.array_equal(
                    reference[
                        finite
                    ],
                    result_yxt[
                        finite
                    ],
                )
            )

            max_complex = float(
                np.max(
                    np.abs(
                        reference[
                            finite
                        ]
                        -
                        result_yxt[
                            finite
                        ]
                    )
                )
            )

        else:

            complex_exact = True
            max_complex = 0.0

        phase_diff = max_phase_diff(
            reference,
            result_yxt,
            (
                reference_geom
                &
                result_geom
            ),
        )

        exact = (
            geom_bad == 0
            and
            complex_exact
        )

        maxproc = (
            spatial_workers
            *
            pair_workers
        )

        rate = (
            len(tasks)
            /
            seconds
        )

        row = {
            "spatial_workers":
                spatial_workers,

            "pair_workers":
                pair_workers,

            "max_phase_processes":
                maxproc,

            "seconds":
                seconds,

            "tiles_per_second":
                rate,

            "geometry_mismatch":
                geom_bad,

            "complex_exact":
                complex_exact,

            "max_abs_complex_difference":
                max_complex,

            "max_abs_phase_difference":
                phase_diff,

            "parity":
                exact,
        }

        results.append(
            row
        )

        print(
            f"{spatial_workers:8d}"
            f"{pair_workers:8d}"
            f"{maxproc:10d}"
            f"{seconds:12.3f}"
            f"{rate:12.3f}"
            f"{str(exact):>10s}"
            f"{phase_diff:14.3e}"
        )

    valid_results = [
        x
        for x in results
        if x[
            "parity"
        ]
    ]

    if not valid_results:

        raise RuntimeError(
            "No canonical parallel layout passed exact parity."
        )

    winner = min(
        valid_results,
        key=lambda x:
            x[
                "seconds"
            ],
    )

    report = {
        "format":
            "pyPSDS-GAMMA-canonical-phase-parallel-v1",

        "canonical_tile":
            [
                args.tile_rows,
                args.tile_cols,
            ],

        "test_grid":
            [
                args.grid_rows,
                args.grid_cols,
            ],

        "ndate":
            ndate,

        "results":
            results,

        "winner":
            winner,
    }

    out = (
        Path(paths.output_dir)
        /
        "processing"
        /
        "canonical_phase_parallel_autotune.json"
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
        "=" * 92
    )

    print(
        "WINNER"
    )

    print(
        "=" * 92
    )

    for k, v in winner.items():

        print(
            f"{k:30s}:",
            v,
        )

    print()

    print(
        "saved:",
        out,
    )

    print()

    print(
        "CANONICAL PHASE AUTOTUNE: PASS"
    )


if __name__ == "__main__":
    main()
