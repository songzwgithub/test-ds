#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
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


def run_command(cmd):
    t0 = time.perf_counter()

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    if p.returncode != 0:
        raise RuntimeError(
            "\n".join(
                [
                    "Command failed:",
                    " ".join(cmd),
                    p.stdout[-4000:],
                ]
            )
        )

    return elapsed


def write_chunk_itab(
    path,
    *,
    ref_one_based,
    secondary_one_based,
):
    """
    Important:
    local output record numbers are rewritten as 1..Nchunk.

    This avoids depending on sparse/global record numbering
    inside phase_sim_orb_pt output files.
    """

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for local_rec, sec in enumerate(
            secondary_one_based,
            1,
        ):
            f.write(
                f"{ref_one_based} "
                f"{sec} "
                f"{local_rec} "
                f"1\n"
            )


def split_indices(
    n,
    workers,
):
    workers = min(
        max(1, workers),
        n,
    )

    parts = np.array_split(
        np.arange(
            n,
            dtype=np.int32,
        ),
        workers,
    )

    return [
        x
        for x in parts
        if x.size
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    # Representative but much smaller than the 581x1035
    # production tile so that the benchmark finishes quickly.
    ap.add_argument(
        "--row0",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--col0",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--rows",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--cols",
        type=int,
        default=512,
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

    provider = (
        GammaPointPhaseCorrectionProvider(
            cfg,
            paths,
            stack,
        )
    )

    assets = provider.prepare()

    commands = provider._commands

    row0 = args.row0
    col0 = args.col0
    rows = args.rows
    cols = args.cols

    if (
        row0 < 0
        or
        col0 < 0
        or
        row0 + rows > H
        or
        col0 + cols > W
    ):
        raise RuntimeError(
            "benchmark tile outside ROI"
        )

    npoints = (
        rows
        *
        cols
    )

    work_root = (
        Path(paths.output_dir)
        /
        "processing"
        /
        "phase_sim_parallel_benchmark"
    )

    if work_root.exists():
        shutil.rmtree(
            work_root
        )

    work_root.mkdir(
        parents=True
    )

    # -------------------------------------------------------
    # Same plist geometry as production provider.
    # -------------------------------------------------------

    global_row0 = (
        base_row0
        +
        row0
    )

    global_col0 = (
        base_col0
        +
        col0
    )

    rr, cc = np.meshgrid(
        np.arange(
            global_row0,
            global_row0 + rows,
            dtype=np.int32,
        ),
        np.arange(
            global_col0,
            global_col0 + cols,
            dtype=np.int32,
        ),
        indexing="ij",
    )

    plist_arr = np.column_stack(
        (
            cc.ravel(),
            rr.ravel(),
        )
    ).astype(
        ">i4",
        copy=False,
    )

    plist = (
        work_root
        /
        "plist"
    )

    plist_arr.tofile(
        plist
    )

    phgt = (
        work_root
        /
        "phgt"
    )

    pmask = (
        work_root
        /
        "pmask"
    )

    # -------------------------------------------------------
    # Height only once.
    # -------------------------------------------------------

    print(
        "=" * 88
    )

    print(
        "phase_sim_orb_pt pair-parallel benchmark"
    )

    print(
        "=" * 88
    )

    print(
        "dates      :",
        len(stack.dates),
    )

    print(
        "pairs      :",
        len(
            assets.pair_secondary_indices
        ),
    )

    print(
        "tile       :",
        f"{rows} x {cols}",
    )

    print(
        "points     :",
        npoints,
    )

    print()

    t_height = run_command(
        [
            commands["data2pt"],
            str(
                assets.height_path
            ),
            str(
                assets.height_geometry_par
            ),
            str(plist),
            str(
                assets.reference_par
            ),
            str(phgt),
            "1",
            "2",
        ]
    )

    h = np.fromfile(
        phgt,
        dtype=">f4",
    )

    if h.size != npoints:
        raise RuntimeError(
            "data2pt size mismatch"
        )

    h_native = h.astype(
        np.float32
    )

    finite = np.isfinite(
        h_native
    )

    if provider.zero_height_is_valid:

        valid = finite

        zero = (
            finite
            &
            (
                h_native == 0.0
            )
        )

        h_native[
            zero
        ] = (
            provider.zero_height_epsilon_m
        )

        h_native.astype(
            ">f4"
        ).tofile(
            phgt
        )

    else:

        valid = (
            finite
            &
            (
                h_native != 0.0
            )
        )

    valid.astype(
        np.uint8
    ).tofile(
        pmask
    )

    print(
        "data2pt    :",
        f"{t_height:.3f} s",
    )

    # -------------------------------------------------------
    # Pair metadata.
    # -------------------------------------------------------

    secondaries = list(
        assets.pair_secondary_indices
    )

    # GAMMA uses 1-based acquisition indices.
    ref_one_based = (
        assets.reference_index
        +
        1
    )

    secondary_one_based = [
        x + 1
        for x in secondaries
    ]

    npair = len(
        secondaries
    )

    # -------------------------------------------------------
    # SERIAL reference
    # -------------------------------------------------------

    serial_dir = (
        work_root
        /
        "serial"
    )

    serial_dir.mkdir()

    serial_itab = (
        serial_dir
        /
        "itab"
    )

    write_chunk_itab(
        serial_itab,
        ref_one_based=ref_one_based,
        secondary_one_based=(
            secondary_one_based
        ),
    )

    serial_out = (
        serial_dir
        /
        "psim"
    )

    print()
    print(
        "Building serial reference..."
    )

    serial_seconds = run_command(
        [
            commands[
                "phase_sim_orb_pt"
            ],
            str(plist),
            str(pmask),
            str(
                assets.pslc_par
            ),
            "-",
            str(serial_itab),
            "-",
            str(phgt),
            str(serial_out),
            str(
                assets.reference_par
            ),
            "-",
            "0",
        ]
    )

    serial_raw = np.fromfile(
        serial_out,
        dtype=">f4",
    )

    expected = (
        npair
        *
        npoints
    )

    if serial_raw.size != expected:
        raise RuntimeError(
            f"serial output size "
            f"{serial_raw.size} != "
            f"{expected}"
        )

    serial = (
        serial_raw
        .astype(
            np.float32
        )
        .reshape(
            npair,
            npoints,
        )
    )

    print(
        "serial     :",
        f"{serial_seconds:.3f} s",
    )

    # -------------------------------------------------------
    # Parallel layouts
    # -------------------------------------------------------

    layouts = [
        2,
        4,
        8,
        16,
    ]

    results = []

    print()

    print(
        f"{'workers':>8s}"
        f"{'seconds':>12s}"
        f"{'speedup':>12s}"
        f"{'maxdiff':>16s}"
        f"{'exact':>10s}"
    )

    print(
        "-" * 62
    )

    for workers in layouts:

        chunks = split_indices(
            npair,
            workers,
        )

        layout_dir = (
            work_root
            /
            f"w{workers:02d}"
        )

        layout_dir.mkdir()

        def worker(
            chunk_id,
            ids,
        ):
            d = (
                layout_dir
                /
                f"chunk_{chunk_id:03d}"
            )

            d.mkdir()

            itab = (
                d
                /
                "itab"
            )

            output = (
                d
                /
                "psim"
            )

            sec = [
                secondary_one_based[
                    int(i)
                ]
                for i in ids
            ]

            write_chunk_itab(
                itab,
                ref_one_based=(
                    ref_one_based
                ),
                secondary_one_based=sec,
            )

            elapsed = run_command(
                [
                    commands[
                        "phase_sim_orb_pt"
                    ],
                    str(plist),
                    str(pmask),
                    str(
                        assets.pslc_par
                    ),
                    "-",
                    str(itab),
                    "-",
                    str(phgt),
                    str(output),
                    str(
                        assets.reference_par
                    ),
                    "-",
                    "0",
                ]
            )

            raw = np.fromfile(
                output,
                dtype=">f4",
            )

            expected_chunk = (
                len(ids)
                *
                npoints
            )

            if raw.size != expected_chunk:
                raise RuntimeError(
                    f"worker {chunk_id}: "
                    f"{raw.size} != "
                    f"{expected_chunk}"
                )

            arr = (
                raw
                .astype(
                    np.float32
                )
                .reshape(
                    len(ids),
                    npoints,
                )
            )

            return (
                ids,
                arr,
                elapsed,
            )

        t0 = time.perf_counter()

        parallel = np.empty_like(
            serial
        )

        worker_times = []

        with ThreadPoolExecutor(
            max_workers=len(
                chunks
            )
        ) as ex:

            futures = [
                ex.submit(
                    worker,
                    chunk_id,
                    ids,
                )
                for chunk_id, ids
                in enumerate(
                    chunks
                )
            ]

            for fut in as_completed(
                futures
            ):

                (
                    ids,
                    arr,
                    elapsed,
                ) = fut.result()

                parallel[
                    ids
                ] = arr

                worker_times.append(
                    elapsed
                )

        seconds = (
            time.perf_counter()
            -
            t0
        )

        exact = np.array_equal(
            serial,
            parallel,
            equal_nan=True,
        )

        finite = (
            np.isfinite(
                serial
            )
            &
            np.isfinite(
                parallel
            )
        )

        if np.any(
            finite
        ):

            maxdiff = float(
                np.max(
                    np.abs(
                        serial[
                            finite
                        ]
                        -
                        parallel[
                            finite
                        ]
                    )
                )
            )

        else:

            maxdiff = 0.0

        speedup = (
            serial_seconds
            /
            seconds
        )

        result = {
            "workers":
                workers,

            "seconds":
                seconds,

            "speedup":
                speedup,

            "exact":
                bool(
                    exact
                ),

            "max_abs_difference":
                maxdiff,

            "max_worker_seconds":
                max(
                    worker_times
                ),
        }

        results.append(
            result
        )

        print(
            f"{workers:8d}"
            f"{seconds:12.3f}"
            f"{speedup:12.2f}"
            f"{maxdiff:16.9e}"
            f"{str(exact):>10s}"
        )

    valid_results = [
        x
        for x in results
        if x[
            "exact"
        ]
    ]

    if valid_results:

        winner = min(
            valid_results,
            key=lambda x:
                x[
                    "seconds"
                ],
        )

    else:

        winner = None

    report = {
        "format":
            "pyPSDS-GAMMA-phase-sim-pair-parallel-v1",

        "tile":
            [
                row0,
                col0,
                rows,
                cols,
            ],

        "points":
            npoints,

        "ndate":
            len(
                stack.dates
            ),

        "npair":
            npair,

        "serial_seconds":
            serial_seconds,

        "results":
            results,

        "winner":
            winner,
    }

    report_path = (
        work_root
        /
        "benchmark.json"
    )

    report_path.write_text(
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
        "=" * 88
    )

    print(
        "WINNER"
    )

    print(
        "=" * 88
    )

    if winner is None:

        print(
            "NO EXACT PARALLEL "
            "LAYOUT PASSED"
        )

        raise SystemExit(1)

    for key, value in (
        winner.items()
    ):
        print(
            f"{key:24s}:",
            value,
        )

    print()

    print(
        "saved:",
        report_path,
    )


if __name__ == "__main__":
    main()
