#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config
from pypsds.runtime import build_runtime_plan


MiB = 1024 ** 2
GiB = 1024 ** 3


def open_or_create_npy(
    path: Path,
    *,
    shape,
    dtype,
    fill,
    resume: bool,
):
    dtype = np.dtype(dtype)

    if resume and path.is_file():
        arr = np.load(
            path,
            mmap_mode="r+",
            allow_pickle=False,
        )

        if arr.shape != tuple(shape):
            raise RuntimeError(
                f"shape mismatch for {path}: "
                f"{arr.shape} != {tuple(shape)}"
            )

        if arr.dtype != dtype:
            raise RuntimeError(
                f"dtype mismatch for {path}: "
                f"{arr.dtype} != {dtype}"
            )

        return arr

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arr = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )

    arr[...] = fill
    arr.flush()

    return arr


def compute_tile_statistics(
    raw: np.ndarray,
    *,
    adi_max: float,
):
    """
    Exact scientific definition currently used by 04_run_psds.py,
    but applied to one tile only.

    raw
        [Ndate, tile_rows, tile_cols], complex64

    Returns
    -------
    valid       bool
    scale2      float32
    ps          bool
    adi         float32

    Important
    ---------
    mean/variance/std are evaluated in float64 after amplitude
    calculation in float32, matching the validated v1.0 route.

    They are NOT persisted because downstream GLRT only needs
    Rayleigh scale^2.
    """

    if raw.ndim != 3:
        raise ValueError(
            "raw must have shape [Ndate,H,W]"
        )

    raw = np.asarray(
        raw,
        dtype=np.complex64,
    )

    zero = (
        (raw.real == 0)
        &
        (raw.imag == 0)
    )

    valid = ~(
        np.any(
            zero,
            axis=0,
        )
        |
        np.any(
            ~np.isfinite(
                raw.real
            ),
            axis=0,
        )
        |
        np.any(
            ~np.isfinite(
                raw.imag
            ),
            axis=0,
        )
    )

    shape = valid.shape

    mean_amp = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )

    var_amp = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )

    std_amp = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )

    amp = np.abs(
        raw
    ).astype(
        np.float32,
        copy=False,
    )

    if np.any(valid):

        x = amp[
            :,
            valid,
        ].astype(
            np.float64,
            copy=False,
        )

        mean_amp[
            valid
        ] = np.mean(
            x,
            axis=0,
        )

        var_amp[
            valid
        ] = np.var(
            x,
            axis=0,
            ddof=0,
        )

        std_amp[
            valid
        ] = np.std(
            x,
            axis=0,
            ddof=0,
        )

        del x

    # ----------------------------------------------------------
    # Match existing production arithmetic.
    #
    # mean_amp and var_amp are float32 at this point.
    #
    # Storing scale2 as float32 does NOT discard meaningful
    # precision relative to the current implementation because
    # current scale2 is formed from these float32 quantities.
    #
    # GLRT later promotes scale2 to float64.
    # ----------------------------------------------------------

    scale2 = (
        (
            var_amp
            +
            mean_amp
            *
            mean_amp
        )
        *
        np.float32(0.5)
    ).astype(
        np.float32,
        copy=False,
    )

    scale2[
        ~valid
    ] = np.nan

    adi = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )

    good_mean = (
        valid
        &
        np.isfinite(
            mean_amp
        )
        &
        (mean_amp > 0)
    )

    adi[
        good_mean
    ] = (
        std_amp[
            good_mean
        ]
        /
        mean_amp[
            good_mean
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    ps = (
        valid
        &
        np.isfinite(
            adi
        )
        &
        (
            adi
            <=
            np.float32(
                adi_max
            )
        )
    )

    return (
        valid,
        scale2,
        ps,
        adi,
    )


def choose_tile_shape(
    *,
    H: int,
    W: int,
    ndate: int,
    usable_memory: int,
    tile_rows: int,
    tile_cols: int,
):
    """
    Choose a bounded-memory tile for the exact amplitude statistics
    implementation.

    Worst active arrays per pixel are approximately:

        raw       N * complex64 = 8N
        amplitude N * float32   = 4N
        x         N * float64   = 8N
        2-D work arrays

    => approximately 20*N bytes/pixel plus safety margin.

    Only a conservative fraction of available memory is allocated to
    this stage. Linux page cache and other pipeline processes retain
    substantial headroom.
    """

    if tile_rows > 0 and tile_cols > 0:
        return (
            min(
                H,
                int(tile_rows),
            ),
            min(
                W,
                int(tile_cols),
            ),
        )

    # Use at most ~1 GiB for one statistics tile.
    #
    # Also avoid consuming more than 20% of RuntimePlan usable RAM.
    tile_budget = min(
        1 * GiB,
        max(
            256 * MiB,
            usable_memory // 5
            if usable_memory > 0
            else 512 * MiB,
        ),
    )

    bytes_per_pixel = (
        20
        *
        ndate
        +
        128
    )

    max_pixels = max(
        4096,
        tile_budget
        //
        max(
            1,
            bytes_per_pixel,
        ),
    )

    # Wide tiles give efficient contiguous range reads.
    tc = min(
        W,
        2048,
    )

    tr = max(
        32,
        max_pixels
        //
        max(
            1,
            tc,
        ),
    )

    tr = min(
        H,
        tr,
        2048,
    )

    # Align rows for predictable I/O/chunk behavior.
    if tr >= 32:
        tr = max(
            32,
            (tr // 32)
            *
            32,
        )

    return (
        int(tr),
        int(tc),
    )


def count_true_tiled(
    arr,
    *,
    block_rows=2048,
):
    H = arr.shape[0]

    total = 0

    for r0 in range(
        0,
        H,
        block_rows,
    ):
        r1 = min(
            H,
            r0
            +
            block_rows,
        )

        total += int(
            np.count_nonzero(
                arr[
                    r0:r1
                ]
            )
        )

    return total


def compare_bool_tiled(
    a,
    b,
    *,
    block_rows=2048,
):
    if a.shape != b.shape:
        return (
            False,
            -1,
        )

    H = a.shape[0]

    mismatch = 0

    for r0 in range(
        0,
        H,
        block_rows,
    ):
        r1 = min(
            H,
            r0
            +
            block_rows,
        )

        mismatch += int(
            np.count_nonzero(
                np.asarray(
                    a[
                        r0:r1
                    ],
                    dtype=bool,
                )
                !=
                np.asarray(
                    b[
                        r0:r1
                    ],
                    dtype=bool,
                )
            )
        )

    return (
        mismatch == 0,
        mismatch,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--adi-max",
        type=float,
        default=0.25,
    )

    ap.add_argument(
        "--tile-rows",
        type=int,
        default=0,
        help=(
            "0 = automatic bounded-memory tile size"
        ),
    )

    ap.add_argument(
        "--tile-cols",
        type=int,
        default=0,
        help=(
            "0 = automatic bounded-memory tile size"
        ),
    )

    ap.add_argument(
        "--resume",
        action="store_true",
    )

    ap.add_argument(
        "--save-adi",
        action="store_true",
        help=(
            "Persist ADI raster. Disabled by default "
            "for minimum disk usage."
        ),
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (
            row0,
            col0,
            H,
            W,
        ),
    ) = open_from_config(
        args.config
    )

    ndate = len(
        stack.dates
    )

    plan = build_runtime_plan(
        ndate=ndate,
    )

    # GammaStack already parallelizes independent acquisition reads.
    stack.io_workers = (
        plan.io_workers
    )

    (
        tile_rows,
        tile_cols,
    ) = choose_tile_shape(
        H=H,
        W=W,
        ndate=ndate,
        usable_memory=(
            plan.usable_memory_bytes
        ),
        tile_rows=(
            args.tile_rows
        ),
        tile_cols=(
            args.tile_cols
        ),
    )

    nr = math.ceil(
        H
        /
        tile_rows
    )

    nc = math.ceil(
        W
        /
        tile_cols
    )

    ntile = (
        nr
        *
        nc
    )

    outdir = (
        Path(
            paths.output_dir
        )
        /
        "processing"
        /
        "ds_statistics"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_map = open_or_create_npy(
        outdir
        /
        "raw_valid.npy",
        shape=(H, W),
        dtype=np.bool_,
        fill=False,
        resume=args.resume,
    )

    scale2_map = open_or_create_npy(
        outdir
        /
        "rayleigh_scale2.npy",
        shape=(H, W),
        dtype=np.float32,
        fill=np.nan,
        resume=args.resume,
    )

    ps_map = open_or_create_npy(
        outdir
        /
        "ps_mask.npy",
        shape=(H, W),
        dtype=np.bool_,
        fill=False,
        resume=args.resume,
    )

    if args.save_adi:
        adi_map = open_or_create_npy(
            outdir
            /
            "amplitude_dispersion_index.npy",
            shape=(H, W),
            dtype=np.float32,
            fill=np.nan,
            resume=args.resume,
        )
    else:
        adi_map = None

    tile_done = open_or_create_npy(
        outdir
        /
        "tile_done.npy",
        shape=(nr, nc),
        dtype=np.bool_,
        fill=False,
        resume=args.resume,
    )

    print(
        "=" * 88
    )

    print(
        "pyPSDS-GAMMA "
        "large-stack DS statistics"
    )

    print(
        "=" * 88
    )

    print(
        "config          :",
        config_path,
    )

    print(
        "scene           :",
        f"{H} x {W}",
    )

    print(
        "acquisitions    :",
        ndate,
    )

    print(
        "tile            :",
        f"{tile_rows} x {tile_cols}",
    )

    print(
        "tiles           :",
        f"{nr} x {nc} = {ntile}",
    )

    print(
        "I/O workers     :",
        stack.io_workers,
    )

    print(
        "available RAM   :",
        f"{plan.available_memory_bytes / GiB:.2f} GiB",
    )

    print(
        "usable RAM      :",
        f"{plan.usable_memory_bytes / GiB:.2f} GiB",
    )

    print(
        "scale2 storage  : float32"
    )

    print(
        "ADI storage     :",
        (
            "enabled"
            if args.save_adi
            else "disabled"
        ),
    )

    print(
        "resume          :",
        args.resume,
    )

    t0 = time.perf_counter()

    processed_pixels = 0
    processed_tiles = 0

    for ir in range(
        nr
    ):
        r0 = (
            ir
            *
            tile_rows
        )

        r1 = min(
            H,
            r0
            +
            tile_rows,
        )

        for ic in range(
            nc
        ):
            c0 = (
                ic
                *
                tile_cols
            )

            c1 = min(
                W,
                c0
                +
                tile_cols,
            )

            npix = (
                (r1 - r0)
                *
                (c1 - c0)
            )

            if tile_done[
                ir,
                ic,
            ]:
                processed_pixels += (
                    npix
                )

                processed_tiles += 1

                print(
                    f"tile "
                    f"{processed_tiles:5d}/"
                    f"{ntile:5d} "
                    f"r={r0}:{r1} "
                    f"c={c0}:{c1} "
                    f"RESUME"
                )

                continue

            ts = time.perf_counter()

            raw = stack.read_window(
                row0=(
                    row0
                    +
                    r0
                ),
                col0=(
                    col0
                    +
                    c0
                ),
                rows=(
                    r1
                    -
                    r0
                ),
                cols=(
                    c1
                    -
                    c0
                ),
            )

            (
                valid,
                scale2,
                ps,
                adi,
            ) = compute_tile_statistics(
                raw,
                adi_max=(
                    args.adi_max
                ),
            )

            del raw

            valid_map[
                r0:r1,
                c0:c1,
            ] = valid

            scale2_map[
                r0:r1,
                c0:c1,
            ] = scale2

            ps_map[
                r0:r1,
                c0:c1,
            ] = ps

            if adi_map is not None:
                adi_map[
                    r0:r1,
                    c0:c1,
                ] = adi

            del (
                valid,
                scale2,
                ps,
                adi,
            )

            tile_done[
                ir,
                ic,
            ] = True

            # Flush only persistent tile products.
            valid_map.flush()
            scale2_map.flush()
            ps_map.flush()

            if adi_map is not None:
                adi_map.flush()

            tile_done.flush()

            processed_pixels += (
                npix
            )

            processed_tiles += 1

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            rate = (
                processed_pixels
                /
                elapsed
                if elapsed > 0
                else 0.0
            )

            tile_s = (
                time.perf_counter()
                -
                ts
            )

            print(
                f"tile "
                f"{processed_tiles:5d}/"
                f"{ntile:5d} "
                f"r={r0}:{r1} "
                f"c={c0}:{c1} "
                f"{tile_s:7.2f}s "
                f"{rate / 1e6:7.3f} Mpix/s"
            )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    raw_valid_count = (
        count_true_tiled(
            valid_map
        )
    )

    ps_count = (
        count_true_tiled(
            ps_map
        )
    )

    # ----------------------------------------------------------
    # Geometry-valid counts without materializing another H x W
    # boolean raster.
    # ----------------------------------------------------------

    geom_path = (
        Path(
            paths.output_dir
        )
        /
        "processing"
        /
        "cache"
        /
        "phase_geometry_valid.npy"
    )

    geom_valid_count = None
    final_valid_count = None
    final_ps_count = None

    if geom_path.is_file():

        geom = np.load(
            geom_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if geom.shape != (
            H,
            W,
        ):
            raise RuntimeError(
                f"geometry valid shape "
                f"{geom.shape} != {(H,W)}"
            )

        geom_valid_count = 0
        final_valid_count = 0
        final_ps_count = 0

        for br0 in range(
            0,
            H,
            2048,
        ):
            br1 = min(
                H,
                br0
                +
                2048,
            )

            g = np.asarray(
                geom[
                    br0:br1
                ],
                dtype=bool,
            )

            v = np.asarray(
                valid_map[
                    br0:br1
                ],
                dtype=bool,
            )

            p = np.asarray(
                ps_map[
                    br0:br1
                ],
                dtype=bool,
            )

            geom_valid_count += int(
                np.count_nonzero(
                    g
                )
            )

            final_valid_count += int(
                np.count_nonzero(
                    g
                    &
                    v
                )
            )

            final_ps_count += int(
                np.count_nonzero(
                    g
                    &
                    p
                )
            )

    # ----------------------------------------------------------
    # Existing v1.0 PS parity, if the frozen production output is
    # still present.
    # ----------------------------------------------------------

    old_ps_path = (
        Path(
            paths.output_dir
        )
        /
        "processing"
        /
        "ps_mask.npy"
    )

    ps_exact = None
    ps_mismatch = None

    if old_ps_path.is_file():

        old_ps = np.load(
            old_ps_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        (
            ps_exact,
            ps_mismatch,
        ) = compare_bool_tiled(
            old_ps,
            ps_map,
        )

    old_final_ps_path = (
        Path(
            paths.output_dir
        )
        /
        "processing"
        /
        "final_ps_mask.npy"
    )

    final_ps_exact = None
    final_ps_mismatch = None

    if (
        old_final_ps_path.is_file()
        and
        geom_path.is_file()
    ):
        old_final_ps = np.load(
            old_final_ps_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        mismatch = 0

        for br0 in range(
            0,
            H,
            2048,
        ):
            br1 = min(
                H,
                br0
                +
                2048,
            )

            candidate = (
                np.asarray(
                    ps_map[
                        br0:br1
                    ],
                    dtype=bool,
                )
                &
                np.asarray(
                    geom[
                        br0:br1
                    ],
                    dtype=bool,
                )
            )

            baseline = np.asarray(
                old_final_ps[
                    br0:br1
                ],
                dtype=bool,
            )

            mismatch += int(
                np.count_nonzero(
                    candidate
                    !=
                    baseline
                )
            )

        final_ps_mismatch = (
            mismatch
        )

        final_ps_exact = (
            mismatch
            ==
            0
        )

    maxrss_kb = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    manifest = {
        "format":
            "pyPSDS-GAMMA-ds-statistics-v1",

        "config":
            str(
                config_path
            ),

        "shape":
            [
                H,
                W,
            ],

        "ndate":
            ndate,

        "tile_rows":
            tile_rows,

        "tile_cols":
            tile_cols,

        "tile_count":
            ntile,

        "io_workers":
            stack.io_workers,

        "adi_max":
            float(
                args.adi_max
            ),

        "raw_valid_count":
            raw_valid_count,

        "ps_count":
            ps_count,

        "geometry_valid_count":
            geom_valid_count,

        "final_valid_count":
            final_valid_count,

        "final_ps_count":
            final_ps_count,

        "elapsed_seconds":
            elapsed,

        "throughput_mpix_s":
            (
                H
                *
                W
                /
                max(
                    elapsed,
                    1e-12,
                )
                /
                1e6
            ),

        "maxrss_gib":
            (
                maxrss_kb
                *
                1024
                /
                GiB
            ),

        "scale2_dtype":
            "float32",

        "ps_parity_exact":
            ps_exact,

        "ps_parity_mismatch":
            ps_mismatch,

        "final_ps_parity_exact":
            final_ps_exact,

        "final_ps_parity_mismatch":
            final_ps_mismatch,
    }

    manifest_path = (
        outdir
        /
        "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
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
        "DS STATISTICS SUMMARY"
    )

    print(
        "=" * 88
    )

    print(
        "raw valid       :",
        raw_valid_count,
    )

    print(
        "PS              :",
        ps_count,
    )

    if geom_valid_count is not None:
        print(
            "geometry valid  :",
            geom_valid_count,
        )

        print(
            "final valid     :",
            final_valid_count,
        )

        print(
            "final PS        :",
            final_ps_count,
        )

    print(
        "elapsed         :",
        f"{elapsed:.3f} s",
    )

    print(
        "throughput      :",
        f"{H * W / max(elapsed,1e-12) / 1e6:.3f} Mpix/s",
    )

    print(
        "peak RSS        :",
        f"{maxrss_kb * 1024 / GiB:.3f} GiB",
    )

    if ps_exact is not None:
        print(
            "raw PS parity   :",
            (
                "EXACT PASS"
                if ps_exact
                else
                f"FAIL ({ps_mismatch} pixels)"
            ),
        )

    if final_ps_exact is not None:
        print(
            "final PS parity :",
            (
                "EXACT PASS"
                if final_ps_exact
                else
                f"FAIL ({final_ps_mismatch} pixels)"
            ),
        )

    print(
        "manifest        :",
        manifest_path,
    )

    if ps_exact is False:
        raise SystemExit(
            "Raw PS parity failed."
        )

    if final_ps_exact is False:
        raise SystemExit(
            "Final PS parity failed."
        )

    print()
    print(
        "LARGE-STACK DS STATISTICS: PASS"
    )


if __name__ == "__main__":
    main()
