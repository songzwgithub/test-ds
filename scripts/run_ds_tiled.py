#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np

from numba import set_num_threads

from pypsds.context import open_from_config
from pypsds.runtime import build_runtime_plan

from pypsds.selection.shp import (
    glrt_statistic,
    glrt_threshold,
)

from pypsds.phase_linking.coherence import (
    compressed_coherence,
)

from pypsds.phase_linking.shp_coherence_bitset import (
    glrt_support_bitset,
    compressed_coherence_bitset,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    prepare_glrt_window_context,
    glrt_support_vectorized_exact,
)

from pypsds.phase_linking.emi import (
    ESTIMATOR_EVD,
    ESTIMATOR_EMI,
    image_pairs,
    robust_emi_threaded,
)

from pypsds.phase_linking.streaming_quality import (
    temporal_quality_streaming,
)

from pypsds.phase_linking.phase_source import (
    CachedPhaseSource,
    GammaStreamingPhaseSource,
)

from pypsds.phase_linking.emi_fast import (
    robust_emi_cholesky_threaded,
)


MiB = 1024 ** 2
GiB = 1024 ** 3


def make_support_batch(
    scale2,
    valid,
    ps,
    rows,
    cols,
    *,
    half_row,
    half_col,
    alpha,
    ndate,
):
    """
    Exact current GLRT support definition.

    rows/cols are LOCAL coordinates inside the current
    tile+halo arrays.
    """

    B = rows.size

    wh = (
        2 * half_row + 1
    )

    ww = (
        2 * half_col + 1
    )

    support = np.zeros(
        (
            B,
            wh,
            ww,
        ),
        dtype=np.bool_,
    )

    center_scale = (
        scale2[
            rows,
            cols,
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    threshold = glrt_threshold(
        alpha
    )

    H, W = valid.shape

    for ky, dy in enumerate(
        range(
            -half_row,
            half_row + 1,
        )
    ):

        for kx, dx in enumerate(
            range(
                -half_col,
                half_col + 1,
            )
        ):

            if (
                dy == 0
                and
                dx == 0
            ):
                continue

            rr = rows + dy
            cc = cols + dx

            inside = (
                (rr >= 0)
                &
                (rr < H)
                &
                (cc >= 0)
                &
                (cc < W)
            )

            if not np.any(
                inside
            ):
                continue

            ids = np.flatnonzero(
                inside
            )

            r2 = rr[ids]
            c2 = cc[ids]

            ngood = (
                valid[
                    r2,
                    c2,
                ]
                &
                ~ps[
                    r2,
                    c2,
                ]
            )

            if not np.any(
                ngood
            ):
                continue

            ids2 = ids[
                ngood
            ]

            r3 = rr[
                ids2
            ]

            c3 = cc[
                ids2
            ]

            stat = glrt_statistic(
                center_scale[
                    ids2
                ],
                scale2[
                    r3,
                    c3,
                ],
                nslc=ndate,
            )

            support[
                ids2,
                ky,
                kx,
            ] = (
                np.isfinite(
                    stat
                )
                &
                (
                    stat
                    <
                    threshold
                )
            )

    return support


def choose_core_shape(
    *,
    H,
    W,
    ndate,
    usable_memory,
    requested_rows,
    requested_cols,
):
    """
    Choose a bounded tile based primarily on the corrected
    complex64 tile+halo working set.

    Spatial scene size NEVER enters the RAM complexity.
    """

    if (
        requested_rows > 0
        and
        requested_cols > 0
    ):
        return (
            min(
                H,
                requested_rows,
            ),
            min(
                W,
                requested_cols,
            ),
        )

    # Target <= 512 MiB corrected-YXT tile and no more
    # than ~1 million core pixels.
    budget = min(
        512 * MiB,
        max(
            128 * MiB,
            usable_memory // 16
            if usable_memory > 0
            else 256 * MiB,
        ),
    )

    bytes_per_pixel = max(
        8 * ndate,
        8,
    )

    max_pixels = min(
        1_000_000,
        max(
            4096,
            budget
            //
            bytes_per_pixel,
        ),
    )

    # Prefer long contiguous range reads, but reduce width
    # automatically for extremely long time series.
    tc = min(
        W,
        1024,
    )

    while (
        tc > 64
        and
        max_pixels // tc < 32
    ):
        tc //= 2

    tr = max(
        16,
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
        1024,
    )

    if tr >= 32:
        tr = max(
            32,
            (tr // 32)
            *
            32,
        )

    if tc >= 64:
        tc = max(
            64,
            (tc // 64)
            *
            64,
        )

    return (
        int(tr),
        int(tc),
    )


def save_tile(
    *,
    tile_dir,
    rows,
    cols,
    phase,
    tc,
    pair_coh,
    shp_count,
    estimator,
    metadata,
):
    """
    Crash-safe tile output.

    manifest.json is written into a temporary directory,
    then the directory is atomically renamed.
    """

    tmp = tile_dir.with_name(
        tile_dir.name
        +
        ".tmp"
    )

    if tmp.exists():
        shutil.rmtree(
            tmp
        )

    tmp.mkdir(
        parents=True,
        exist_ok=False,
    )

    np.save(
        tmp / "rows.npy",
        np.asarray(
            rows,
            dtype=np.int32,
        ),
    )

    np.save(
        tmp / "cols.npy",
        np.asarray(
            cols,
            dtype=np.int32,
        ),
    )

    np.save(
        tmp / "phase.npy",
        np.asarray(
            phase,
            dtype=np.complex64,
        ),
    )

    np.save(
        tmp
        /
        "temporal_coherence.npy",
        np.asarray(
            tc,
            dtype=np.float32,
        ),
    )

    np.save(
        tmp
        /
        "median_pair_coherence.npy",
        np.asarray(
            pair_coh,
            dtype=np.float32,
        ),
    )

    np.save(
        tmp
        /
        "shp_count.npy",
        np.asarray(
            shp_count,
            dtype=np.int16,
        ),
    )

    np.save(
        tmp
        /
        "estimator.npy",
        np.asarray(
            estimator,
            dtype=np.uint8,
        ),
    )

    (
        tmp
        /
        "manifest.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    if tile_dir.exists():
        shutil.rmtree(
            tile_dir
        )

    os.replace(
        tmp,
        tile_dir,
    )


def concatenate_or_empty(
    arrays,
    *,
    shape_tail,
    dtype,
):
    if arrays:
        return np.concatenate(
            arrays,
            axis=0,
        ).astype(
            dtype,
            copy=False,
        )

    return np.empty(
        (
            0,
            *shape_tail,
        ),
        dtype=dtype,
    )


def load_tile_manifest(
    path,
):
    return json.loads(
        (
            path
            /
            "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )


def count_bool_tiled(
    arr,
):
    H = arr.shape[0]

    total = 0

    for r0 in range(
        0,
        H,
        2048,
    ):
        r1 = min(
            H,
            r0 + 2048,
        )

        total += int(
            np.count_nonzero(
                arr[
                    r0:r1
                ]
            )
        )

    return total


def parity_check(
    *,
    processing_dir,
    tiles_root,
    W,
    ndate,
    tc_check,
):
    """
    Compare sparse tiled results against the frozen full-scene
    implementation.

    This is only a migration test. It is not required by future
    ultra-large production scenes.
    """

    required = {
        "pl":
            processing_dir
            /
            "pl_valid.npy",

        "tc":
            processing_dir
            /
            "temporal_coherence.npy",

        "pc":
            processing_dir
            /
            "median_pair_coherence.npy",

        "est":
            processing_dir
            /
            "estimator_code.npy",

        "K":
            processing_dir
            /
            "shp_count.npy",

        "phase":
            processing_dir
            /
            "linked_phase.npy",
    }

    if not all(
        p.is_file()
        for p in required.values()
    ):

        print()
        print(
            "PARITY: skipped "
            "(frozen full-scene products not all present)"
        )

        return None

    base_pl = np.load(
        required["pl"],
        mmap_mode="r",
    )

    base_tc = np.load(
        required["tc"],
        mmap_mode="r",
    )

    base_pc = np.load(
        required["pc"],
        mmap_mode="r",
    )

    base_est = np.load(
        required["est"],
        mmap_mode="r",
    )

    base_K = np.load(
        required["K"],
        mmap_mode="r",
    )

    base_phase = np.load(
        required["phase"],
        mmap_mode="r",
    )

    base_count = count_bool_tiled(
        base_pl
    )

    new_count = 0
    subset_bad = 0
    estimator_bad = 0
    shp_bad = 0
    selected_bad = 0

    max_tc = 0.0
    max_pc = 0.0
    max_phase = 0.0

    new_selected = 0

    tile_dirs = sorted(
        p
        for p in tiles_root.iterdir()
        if (
            p.is_dir()
            and
            (
                p
                /
                "manifest.json"
            ).is_file()
        )
    )

    for td in tile_dirs:

        rr = np.load(
            td
            /
            "rows.npy",
        )

        cc = np.load(
            td
            /
            "cols.npy",
        )

        ph = np.load(
            td
            /
            "phase.npy",
            mmap_mode="r",
        )

        tc = np.load(
            td
            /
            "temporal_coherence.npy",
        )

        pc = np.load(
            td
            /
            "median_pair_coherence.npy",
        )

        est = np.load(
            td
            /
            "estimator.npy",
        )

        K = np.load(
            td
            /
            "shp_count.npy",
        )

        n = rr.size
        new_count += int(
            n
        )

        if n == 0:
            continue

        bp = np.asarray(
            base_pl[
                rr,
                cc,
            ],
            dtype=bool,
        )

        subset_bad += int(
            np.count_nonzero(
                ~bp
            )
        )

        btc = np.asarray(
            base_tc[
                rr,
                cc,
            ],
            dtype=np.float32,
        )

        bpc = np.asarray(
            base_pc[
                rr,
                cc,
            ],
            dtype=np.float32,
        )

        best = np.asarray(
            base_est[
                rr,
                cc,
            ],
            dtype=np.uint8,
        )

        bK = np.asarray(
            base_K[
                rr,
                cc,
            ],
            dtype=np.int16,
        )

        estimator_bad += int(
            np.count_nonzero(
                best
                !=
                est
            )
        )

        shp_bad += int(
            np.count_nonzero(
                bK
                !=
                K
            )
        )

        good_tc = (
            np.isfinite(
                btc
            )
            &
            np.isfinite(
                tc
            )
        )

        if np.any(
            good_tc
        ):
            max_tc = max(
                max_tc,
                float(
                    np.max(
                        np.abs(
                            btc[
                                good_tc
                            ]
                            -
                            tc[
                                good_tc
                            ]
                        )
                    )
                ),
            )

        good_pc = (
            np.isfinite(
                bpc
            )
            &
            np.isfinite(
                pc
            )
        )

        if np.any(
            good_pc
        ):
            max_pc = max(
                max_pc,
                float(
                    np.max(
                        np.abs(
                            bpc[
                                good_pc
                            ]
                            -
                            pc[
                                good_pc
                            ]
                        )
                    )
                ),
            )

        # Read only sparse phase histories.
        bph = np.asarray(
            base_phase[
                :,
                rr,
                cc,
            ].T,
            dtype=np.complex64,
        )

        if bph.shape != (
            n,
            ndate,
        ):
            raise RuntimeError(
                f"baseline phase shape "
                f"{bph.shape} != {(n,ndate)}"
            )

        finite = (
            np.isfinite(
                bph.real
            )
            &
            np.isfinite(
                bph.imag
            )
            &
            np.isfinite(
                ph.real
            )
            &
            np.isfinite(
                ph.imag
            )
        )

        if np.any(
            finite
        ):
            max_phase = max(
                max_phase,
                float(
                    np.max(
                        np.abs(
                            bph[
                                finite
                            ]
                            -
                            ph[
                                finite
                            ]
                        )
                    )
                ),
            )

        new_sel = (
            np.isfinite(
                tc
            )
            &
            np.isfinite(
                pc
            )
            &
            (
                tc
                >=
                tc_check
            )
        )

        base_sel = (
            np.isfinite(
                btc
            )
            &
            np.isfinite(
                bpc
            )
            &
            (
                btc
                >=
                tc_check
            )
        )

        selected_bad += int(
            np.count_nonzero(
                new_sel
                !=
                base_sel
            )
        )

        new_selected += int(
            np.count_nonzero(
                new_sel
            )
        )

    same_set = (
        new_count
        ==
        base_count
        and
        subset_bad
        ==
        0
    )

    # These are numerical migration tolerances, not
    # scientific thresholds.
    passed = (
        same_set
        and
        estimator_bad == 0
        and
        shp_bad == 0
        and
        selected_bad == 0
        and
        max_tc <= 1e-6
        and
        max_pc <= 1e-6
        and
        max_phase <= 5e-6
    )

    result = {
        "baseline_pl_valid":
            base_count,

        "tiled_pl_valid":
            new_count,

        "same_point_set":
            same_set,

        "subset_bad":
            subset_bad,

        "estimator_bad":
            estimator_bad,

        "shp_count_bad":
            shp_bad,

        "selected_mask_bad":
            selected_bad,

        "tc_check":
            tc_check,

        "tiled_tc_selected":
            new_selected,

        "max_abs_tc_difference":
            max_tc,

        "max_abs_pair_coherence_difference":
            max_pc,

        "max_abs_phase_difference":
            max_phase,

        "pass":
            passed,
    }

    print()
    print(
        "=" * 92
    )

    print(
        "TILED DS PARITY"
    )

    print(
        "=" * 92
    )

    for k, v in result.items():
        print(
            f"{k:36s}:",
            v,
        )

    print()

    print(
        "TILED DS PARITY:",
        (
            "PASS"
            if passed
            else
            "FAIL"
        ),
    )

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--phase-source",
        choices=(
            "cache",
            "stream",
        ),
        default="cache",
        help=(
            "cache = existing full corrected YXT cache; "
            "stream = read current GAMMA RSLC tile+halo and "
            "apply geometry correction on demand."
        ),
    )

    ap.add_argument(
        "--center-mode",
        choices=(
            "current",
            "all",
        ),
        default="all",
        help=(
            "current = frozen Moraine center prior "
            "for exact migration testing; "
            "all = every final-valid non-PS center."
        ),
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

    ap.add_argument(
        "--beta",
        type=float,
        default=0.0,
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
        "--support-backend",
        choices=(
            "legacy",
            "vectorized_exact",
            "bitset",
        ),
        default="vectorized_exact",
        help=(
            "SHP/coherence backend. "
            "legacy is the validated exact production route; "
            "bitset is experimental and must not be used for "
            "production unless exact parity is demonstrated."
        ),
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=1024,
        help=(
            "Center block size for vectorized_exact GLRT. "
            "1024 is the validated fastest value on the "
            "current 38-date benchmark while keeping memory bounded."
        ),
    )

    ap.add_argument(
        "--emi-backend",
        choices=(
            "current_eigh",
            "fast_cholesky",
        ),
        default="fast_cholesky",
        help=(
            "EMI numerical backend. "
            "fast_cholesky is the optimized backend; "
            "current_eigh preserves the frozen v1.0 route."
        ),
    )

    ap.add_argument(
        "--core-rows",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--core-cols",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--pl-workers",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--pl-chunk",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--tc-check",
        type=float,
        default=0.8,
    )

    ap.add_argument(
        "--resume",
        action="store_true",
    )

    ap.add_argument(
        "--no-parity",
        action="store_true",
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

    set_num_threads(
        plan.numba_threads
    )

    processing = (
        Path(
            paths.output_dir
        )
        /
        "processing"
    )

    stats_dir = (
        processing
        /
        "ds_statistics"
    )

    raw_valid_path = (
        stats_dir
        /
        "raw_valid.npy"
    )

    scale2_path = (
        stats_dir
        /
        "rayleigh_scale2.npy"
    )

    ps_path = (
        stats_dir
        /
        "ps_mask.npy"
    )

    for p in (
        raw_valid_path,
        scale2_path,
        ps_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(
                p
            )

    raw_valid = np.load(
        raw_valid_path,
        mmap_mode="r",
    )

    scale2 = np.load(
        scale2_path,
        mmap_mode="r",
    )

    ps_raw = np.load(
        ps_path,
        mmap_mode="r",
    )

    if args.phase_source == "cache":

        phase_source = CachedPhaseSource(
            processing_dir=processing,
            H=H,
            W=W,
            ndate=ndate,
        )

    else:

        phase_source = GammaStreamingPhaseSource(
            cfg=cfg,
            paths=paths,
            stack=stack,
            base_row0=row0,
            base_col0=col0,
            io_workers=plan.io_workers,
        )

    # --------------------------------------------------------
    # Existing center prior only for migration parity.
    # Future large-data default will be direct GLRT.
    # --------------------------------------------------------

    center_prior = None

    if args.center_mode == "current":

        prior_path = (
            processing
            /
            "moraine_center_prior.npz"
        )

        if not prior_path.is_file():
            raise FileNotFoundError(
                prior_path
            )

        with np.load(
            prior_path
        ) as z:

            center_prior = (
                z[
                    "candidate_mask"
                ].astype(
                    bool,
                    copy=False,
                )
            )

        if center_prior.shape != (
            H,
            W,
        ):
            raise RuntimeError(
                "center prior shape mismatch"
            )

    pairs = image_pairs(
        ndate
    )

    pair_i = pairs[
        :,
        0,
    ]

    pair_j = pairs[
        :,
        1,
    ]

    (
        core_rows,
        core_cols,
    ) = choose_core_shape(
        H=H,
        W=W,
        ndate=ndate,
        usable_memory=(
            plan.usable_memory_bytes
        ),
        requested_rows=(
            args.core_rows
        ),
        requested_cols=(
            args.core_cols
        ),
    )

    batch_size = (
        args.batch_size
        if args.batch_size > 0
        else
        min(
            16000,
            plan.phase_link_batch_size,
        )
    )

    auto_workers = (
        plan.phase_link_workers
    )

    auto_chunk = min(
        512,
        plan.phase_link_chunk_size,
    )

    autotune_path = (
        processing
        /
        "ds_tiled"
        /
        "pl_cpu_autotune.json"
    )

    if autotune_path.is_file():

        try:
            autotune = json.loads(
                autotune_path.read_text(
                    encoding="utf-8"
                )
            )

            winner = autotune.get(
                "winner",
                {},
            )

            tuned_workers = int(
                winner.get(
                    "workers",
                    0,
                )
            )

            tuned_chunk = int(
                winner.get(
                    "chunk",
                    0,
                )
            )

            if tuned_workers > 0:
                auto_workers = min(
                    plan.cpu_count,
                    tuned_workers,
                )

            if tuned_chunk > 0:
                auto_chunk = (
                    tuned_chunk
                )

        except Exception as exc:

            print(
                "WARNING: ignoring invalid "
                "PL autotune file:",
                exc,
            )

    pl_workers = (
        args.pl_workers
        if args.pl_workers > 0
        else
        auto_workers
    )

    pl_chunk = (
        args.pl_chunk
        if args.pl_chunk > 0
        else
        auto_chunk
    )

    nr = math.ceil(
        H
        /
        core_rows
    )

    nc = math.ceil(
        W
        /
        core_cols
    )

    ntile = (
        nr
        *
        nc
    )

    outdir = (
        processing
        /
        "ds_tiled"
    )

    tiles_root = (
        outdir
        /
        "tiles"
    )

    if (
        not args.resume
        and
        tiles_root.exists()
    ):
        shutil.rmtree(
            tiles_root
        )

    tiles_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 92
    )

    print(
        "pyPSDS-GAMMA "
        "U2 tiled sparse DS phase linking"
    )

    print(
        "=" * 92
    )

    print(
        "scene            :",
        f"{H} x {W}",
    )

    print(
        "dates            :",
        ndate,
    )

    print(
        "pairs            :",
        pairs.shape[0],
    )

    print(
        "phase source     :",
        args.phase_source,
    )

    print(
        "center mode      :",
        args.center_mode,
    )

    print(
        "core tile        :",
        f"{core_rows} x {core_cols}",
    )

    print(
        "halo             :",
        f"{args.half_row} x {args.half_col}",
    )

    print(
        "tiles            :",
        f"{nr} x {nc} = {ntile}",
    )

    print(
        "batch            :",
        batch_size,
    )

    print(
        "support backend  :",
        args.support_backend,
    )

    if (
        args.support_backend
        ==
        "vectorized_exact"
    ):
        print(
            "support block    :",
            args.support_block,
        )

    print(
        "EMI backend      :",
        args.emi_backend,
    )

    print(
        "PL workers       :",
        pl_workers,
    )

    print(
        "PL chunk         :",
        pl_chunk,
    )

    print(
        "Numba threads    :",
        plan.numba_threads,
    )

    print(
        "available RAM    :",
        f"{plan.available_memory_bytes/GiB:.2f} GiB",
    )

    print(
        "usable RAM       :",
        f"{plan.usable_memory_bytes/GiB:.2f} GiB",
    )

    print(
        "resume           :",
        args.resume,
    )

    # --------------------------------------------------------
    # Warm Numba kernels outside measured tile timings.
    # --------------------------------------------------------

    dummy_coh = np.ones(
        (
            1,
            pairs.shape[0],
        ),
        dtype=np.complex64,
    )

    dummy_phase = np.ones(
        (
            1,
            ndate,
        ),
        dtype=np.complex64,
    )

    _ = temporal_quality_streaming(
        dummy_coh,
        dummy_phase,
        pair_i,
        pair_j,
    )

    del (
        dummy_coh,
        dummy_phase,
    )

    t_all = time.perf_counter()

    tile_index = 0

    for ir in range(
        nr
    ):

        r0 = (
            ir
            *
            core_rows
        )

        r1 = min(
            H,
            r0
            +
            core_rows,
        )

        for ic in range(
            nc
        ):

            c0 = (
                ic
                *
                core_cols
            )

            c1 = min(
                W,
                c0
                +
                core_cols,
            )

            tile_dir = (
                tiles_root
                /
                f"tile_{tile_index:06d}"
            )

            if (
                args.resume
                and
                (
                    tile_dir
                    /
                    "manifest.json"
                ).is_file()
            ):

                m = load_tile_manifest(
                    tile_dir
                )

                print(
                    f"tile "
                    f"{tile_index+1:5d}/"
                    f"{ntile:5d} "
                    f"RESUME "
                    f"PL={m['pl_valid_count']}"
                )

                tile_index += 1
                continue

            ts_tile = time.perf_counter()

            # -----------------------------------------------
            # Halo bounds in ROI-local coordinates.
            # -----------------------------------------------

            hr0 = max(
                0,
                r0
                -
                args.half_row,
            )

            hr1 = min(
                H,
                r1
                +
                args.half_row,
            )

            hc0 = max(
                0,
                c0
                -
                args.half_col,
            )

            hc1 = min(
                W,
                c1
                +
                args.half_col,
            )

            # -----------------------------------------------
            # Small tile-local statistical arrays.
            # -----------------------------------------------

            phase_tile = phase_source.read_tile(
                local_row0=hr0,
                local_row1=hr1,
                local_col0=hc0,
                local_col1=hc1,
            )

            local_yxt = phase_tile.yxt

            local_valid = np.ascontiguousarray(
                np.asarray(
                    raw_valid[
                        hr0:hr1,
                        hc0:hc1,
                    ],
                    dtype=bool,
                )
                &
                phase_tile.geometry_valid
            )

            local_ps = np.ascontiguousarray(
                np.asarray(
                    ps_raw[
                        hr0:hr1,
                        hc0:hc1,
                    ],
                    dtype=bool,
                )
                &
                local_valid
            )

            local_scale2 = np.ascontiguousarray(
                scale2[
                    hr0:hr1,
                    hc0:hc1,
                ],
                dtype=np.float32,
            )

            cr0 = (
                r0
                -
                hr0
            )

            cr1 = (
                cr0
                +
                (
                    r1
                    -
                    r0
                )
            )

            cc0 = (
                c0
                -
                hc0
            )

            cc1 = (
                cc0
                +
                (
                    c1
                    -
                    c0
                )
            )

            center_mask = (
                local_valid[
                    cr0:cr1,
                    cc0:cc1,
                ]
                &
                ~local_ps[
                    cr0:cr1,
                    cc0:cc1,
                ]
            )

            if center_prior is not None:

                center_mask &= center_prior[
                    r0:r1,
                    c0:c1,
                ]

            rr_core, cc_core = np.where(
                center_mask
            )

            center_count = int(
                rr_core.size
            )

            # Convert core-local to halo-local.
            rr = (
                rr_core
                +
                cr0
            ).astype(
                np.int32,
                copy=False,
            )

            cc = (
                cc_core
                +
                cc0
            ).astype(
                np.int32,
                copy=False,
            )

            del (
                rr_core,
                cc_core,
                center_mask,
            )

            rows_out = []
            cols_out = []
            phase_out = []
            tc_out = []
            pc_out = []
            K_out = []
            est_out = []

            eligible_count = 0
            pl_count = 0
            emi_count = 0
            evd_count = 0
            tc_check_count = 0

            support_seconds = 0.0
            coherence_seconds = 0.0
            emi_seconds = 0.0
            quality_seconds = 0.0

            support_ctx = None

            if (
                args.support_backend
                ==
                "vectorized_exact"
            ):

                ts = time.perf_counter()

                support_ctx = (
                    prepare_glrt_window_context(
                        local_scale2,
                        local_valid,
                        local_ps,
                        half_row=(
                            args.half_row
                        ),
                        half_col=(
                            args.half_col
                        ),
                    )
                )

                support_seconds += (
                    time.perf_counter()
                    -
                    ts
                )

            # Corrected Y-X-Time data were loaded/generated
            # by phase_source for this tile+halo.

            for b0 in range(
                0,
                center_count,
                batch_size,
            ):

                b1 = min(
                    center_count,
                    b0
                    +
                    batch_size,
                )

                br = rr[
                    b0:b1
                ]

                bc = cc[
                    b0:b1
                ]

                ts = time.perf_counter()

                if (
                    args.support_backend
                    ==
                    "bitset"
                ):

                    (
                        support_bits,
                        K,
                    ) = glrt_support_bitset(
                        local_scale2,
                        local_valid,
                        local_ps,
                        br,
                        bc,
                        half_row=(
                            args.half_row
                        ),
                        half_col=(
                            args.half_col
                        ),
                        threshold=glrt_threshold(
                            args.alpha
                        ),
                        nslc=ndate,
                    )

                elif (
                    args.support_backend
                    ==
                    "vectorized_exact"
                ):

                    (
                        support,
                        K,
                    ) = (
                        glrt_support_vectorized_exact(
                            support_ctx,
                            br,
                            bc,
                            alpha=(
                                args.alpha
                            ),
                            nslc=ndate,
                            block_size=(
                                args.support_block
                            ),
                        )
                    )

                else:

                    support = make_support_batch(
                        local_scale2,
                        local_valid,
                        local_ps,
                        br,
                        bc,
                        half_row=(
                            args.half_row
                        ),
                        half_col=(
                            args.half_col
                        ),
                        alpha=(
                            args.alpha
                        ),
                        ndate=ndate,
                    )

                    K = np.sum(
                        support,
                        axis=(
                            1,
                            2,
                        ),
                    ).astype(
                        np.int16
                    )

                support_seconds += (
                    time.perf_counter()
                    -
                    ts
                )

                good = (
                    K
                    >=
                    args.min_shp
                )

                ngood = int(
                    np.count_nonzero(
                        good
                    )
                )

                eligible_count += ngood

                if ngood == 0:

                    if (
                        args.support_backend
                        ==
                        "bitset"
                    ):
                        del support_bits

                    else:
                        del support

                    del (
                        K,
                        good,
                    )

                    continue

                gr = br[
                    good
                ]

                gc = bc[
                    good
                ]

                gK = K[
                    good
                ]

                if (
                    args.support_backend
                    ==
                    "bitset"
                ):

                    gsupport_bits = (
                        support_bits[
                            good
                        ]
                    )

                    del support_bits

                else:

                    gsupport = support[
                        good
                    ]

                    del support

                del (
                    K,
                    good,
                )

                ts = time.perf_counter()

                if (
                    args.support_backend
                    ==
                    "bitset"
                ):

                    coh = (
                        compressed_coherence_bitset(
                            local_yxt,
                            gr,
                            gc,
                            gsupport_bits,
                            pair_i,
                            pair_j,
                            half_row=(
                                args.half_row
                            ),
                            half_col=(
                                args.half_col
                            ),
                        )
                    )

                    del gsupport_bits

                else:

                    coh = compressed_coherence(
                        local_yxt,
                        gr,
                        gc,
                        gsupport,
                        pair_i,
                        pair_j,
                    )

                    del gsupport

                coherence_seconds += (
                    time.perf_counter()
                    -
                    ts
                )

                ts = time.perf_counter()

                if (
                    args.emi_backend
                    ==
                    "fast_cholesky"
                ):

                    (
                        ph,
                        est,
                    ) = (
                        robust_emi_cholesky_threaded(
                            coh,
                            n_images=ndate,
                            pairs=pairs,
                            beta=args.beta,
                            gamma_jitter=(
                                args.gamma_jitter
                            ),
                            emi_mu=(
                                args.emi_mu
                            ),
                            reference_idx=0,
                            workers=pl_workers,
                            chunk_size=pl_chunk,
                        )
                    )

                else:

                    (
                        ph,
                        est,
                        emi_eig,
                        evd_eig,
                        gamma_min,
                    ) = robust_emi_threaded(
                        coh,
                        n_images=ndate,
                        pairs=pairs,
                        beta=args.beta,
                        gamma_jitter=(
                            args.gamma_jitter
                        ),
                        emi_mu=(
                            args.emi_mu
                        ),
                        reference_idx=0,
                        workers=pl_workers,
                        chunk_size=pl_chunk,
                    )

                    # Diagnostics not needed by sparse output.
                    del (
                        emi_eig,
                        evd_eig,
                        gamma_min,
                    )

                emi_seconds += (
                    time.perf_counter()
                    -
                    ts
                )

                ts = time.perf_counter()

                (
                    tc,
                    pc,
                ) = temporal_quality_streaming(
                    coh,
                    ph,
                    pair_i,
                    pair_j,
                )

                quality_seconds += (
                    time.perf_counter()
                    -
                    ts
                )

                del coh

                ok = (
                    (est != 255)
                    &
                    np.isfinite(
                        tc
                    )
                    &
                    np.isfinite(
                        pc
                    )
                )

                if np.any(
                    ok
                ):

                    gr2 = gr[
                        ok
                    ]

                    gc2 = gc[
                        ok
                    ]

                    rows_global = (
                        hr0
                        +
                        gr2
                    ).astype(
                        np.int32,
                        copy=False,
                    )

                    cols_global = (
                        hc0
                        +
                        gc2
                    ).astype(
                        np.int32,
                        copy=False,
                    )

                    rows_out.append(
                        rows_global
                    )

                    cols_out.append(
                        cols_global
                    )

                    phase_out.append(
                        ph[
                            ok
                        ].astype(
                            np.complex64,
                            copy=False,
                        )
                    )

                    tc_out.append(
                        tc[
                            ok
                        ]
                    )

                    pc_out.append(
                        pc[
                            ok
                        ]
                    )

                    K_out.append(
                        gK[
                            ok
                        ]
                    )

                    est_out.append(
                        est[
                            ok
                        ]
                    )

                    n_ok = int(
                        np.count_nonzero(
                            ok
                        )
                    )

                    pl_count += n_ok

                    emi_count += int(
                        np.count_nonzero(
                            est[
                                ok
                            ]
                            ==
                            ESTIMATOR_EMI
                        )
                    )

                    evd_count += int(
                        np.count_nonzero(
                            est[
                                ok
                            ]
                            ==
                            ESTIMATOR_EVD
                        )
                    )

                    tc_check_count += int(
                        np.count_nonzero(
                            tc[
                                ok
                            ]
                            >=
                            args.tc_check
                        )
                    )

                del (
                    gr,
                    gc,
                    gK,
                    ph,
                    est,
                    tc,
                    pc,
                    ok,
                )

            del local_yxt

            rows_arr = concatenate_or_empty(
                rows_out,
                shape_tail=(),
                dtype=np.int32,
            )

            cols_arr = concatenate_or_empty(
                cols_out,
                shape_tail=(),
                dtype=np.int32,
            )

            phase_arr = concatenate_or_empty(
                phase_out,
                shape_tail=(ndate,),
                dtype=np.complex64,
            )

            tc_arr = concatenate_or_empty(
                tc_out,
                shape_tail=(),
                dtype=np.float32,
            )

            pc_arr = concatenate_or_empty(
                pc_out,
                shape_tail=(),
                dtype=np.float32,
            )

            K_arr = concatenate_or_empty(
                K_out,
                shape_tail=(),
                dtype=np.int16,
            )

            est_arr = concatenate_or_empty(
                est_out,
                shape_tail=(),
                dtype=np.uint8,
            )

            tile_seconds = (
                time.perf_counter()
                -
                ts_tile
            )

            metadata = {
                "tile_index":
                    tile_index,

                "core":
                    [
                        r0,
                        r1,
                        c0,
                        c1,
                    ],

                "halo":
                    [
                        hr0,
                        hr1,
                        hc0,
                        hc1,
                    ],

                "center_count":
                    center_count,

                "eligible_count":
                    eligible_count,

                "pl_valid_count":
                    pl_count,

                "emi_count":
                    emi_count,

                "evd_count":
                    evd_count,

                "tc_check":
                    args.tc_check,

                "tc_check_count":
                    tc_check_count,

                "phase_read_seconds":
                    float(
                        phase_tile.read_seconds
                    ),

                "phase_correction_seconds":
                    float(
                        phase_tile.correction_seconds
                    ),

                "support_seconds":
                    support_seconds,

                "coherence_seconds":
                    coherence_seconds,

                "emi_seconds":
                    emi_seconds,

                "quality_seconds":
                    quality_seconds,

                "tile_seconds":
                    tile_seconds,
            }

            save_tile(
                tile_dir=tile_dir,
                rows=rows_arr,
                cols=cols_arr,
                phase=phase_arr,
                tc=tc_arr,
                pair_coh=pc_arr,
                shp_count=K_arr,
                estimator=est_arr,
                metadata=metadata,
            )

            print(
                f"tile "
                f"{tile_index+1:5d}/"
                f"{ntile:5d} "
                f"centers={center_count:8d} "
                f"K={eligible_count:8d} "
                f"PL={pl_count:8d} "
                f"TC>={args.tc_check:g}:"
                f"{tc_check_count:8d} "
                f"{tile_seconds:8.2f}s"
            )

            tile_index += 1

    elapsed = (
        time.perf_counter()
        -
        t_all
    )

    # --------------------------------------------------------
    # Aggregate only tiny per-tile manifests.
    # --------------------------------------------------------

    totals = {
        "center_count": 0,
        "eligible_count": 0,
        "pl_valid_count": 0,
        "emi_count": 0,
        "evd_count": 0,
        "tc_check_count": 0,
        "phase_read_seconds": 0.0,
        "phase_correction_seconds": 0.0,
        "support_seconds": 0.0,
        "coherence_seconds": 0.0,
        "emi_seconds": 0.0,
        "quality_seconds": 0.0,
    }

    for td in sorted(
        p
        for p in tiles_root.iterdir()
        if (
            p.is_dir()
            and
            (
                p
                /
                "manifest.json"
            ).is_file()
        )
    ):

        m = load_tile_manifest(
            td
        )

        for key in totals:

            totals[
                key
            ] += m[
                key
            ]

    maxrss_kb = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    parity = None

    if not args.no_parity:

        parity = parity_check(
            processing_dir=processing,
            tiles_root=tiles_root,
            W=W,
            ndate=ndate,
            tc_check=(
                args.tc_check
            ),
        )

    root_manifest = {
        "format":
            "pyPSDS-GAMMA-tiled-sparse-pl-v1",

        "shape":
            [
                H,
                W,
            ],

        "ndate":
            ndate,

        "npair":
            int(
                pairs.shape[0]
            ),

        "phase_source":
            args.phase_source,

        "center_mode":
            args.center_mode,

        "core_rows":
            core_rows,

        "core_cols":
            core_cols,

        "half_row":
            args.half_row,

        "half_col":
            args.half_col,

        "tile_count":
            ntile,

        "batch_size":
            batch_size,

        "support_backend":
            args.support_backend,

        "support_block":
            (
                args.support_block
                if
                args.support_backend
                ==
                "vectorized_exact"
                else
                None
            ),

        "emi_backend":
            args.emi_backend,

        "pl_workers":
            pl_workers,

        "pl_chunk":
            pl_chunk,

        "numba_threads":
            plan.numba_threads,

        "tc_check":
            args.tc_check,

        "elapsed_seconds":
            elapsed,

        "maxrss_gib":
            (
                maxrss_kb
                *
                1024
                /
                GiB
            ),

        "totals":
            totals,

        "parity":
            parity,
    }

    (
        outdir
        /
        "manifest.json"
    ).write_text(
        json.dumps(
            root_manifest,
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
        "U2 TILED SPARSE DS SUMMARY"
    )

    print(
        "=" * 92
    )

    for k, v in totals.items():

        if k.endswith(
            "_seconds"
        ):

            print(
                f"{k:26s}:",
                f"{v:.3f}",
            )

        else:

            print(
                f"{k:26s}:",
                v,
            )

    print(
        f"{'elapsed_seconds':26s}:",
        f"{elapsed:.3f}",
    )

    print(
        f"{'peak_RSS_GiB':26s}:",
        f"{maxrss_kb*1024/GiB:.3f}",
    )

    print(
        "manifest                  :",
        outdir
        /
        "manifest.json",
    )

    if (
        parity is not None
        and
        parity["pass"] is False
    ):
        raise SystemExit(
            "U2 migration parity failed."
        )

    print()
    print(
        "U2 TILED SPARSE DS: PASS"
    )


if __name__ == "__main__":
    main()
