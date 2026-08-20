#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
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
from pypsds.phase_linking.coherence import compressed_coherence
from pypsds.phase_linking.emi import (
    image_pairs,
    robust_emi_threaded,
)


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
    B = rows.size
    wh = 2 * half_row + 1
    ww = 2 * half_col + 1

    out = np.zeros(
        (B, wh, ww),
        dtype=np.bool_,
    )

    center_scale = (
        scale2[rows, cols]
        .astype(np.float64, copy=False)
    )

    threshold = glrt_threshold(alpha)

    H, W = valid.shape

    for ky, dy in enumerate(
        range(-half_row, half_row + 1)
    ):
        for kx, dx in enumerate(
            range(-half_col, half_col + 1)
        ):
            if dy == 0 and dx == 0:
                continue

            rr = rows + dy
            cc = cols + dx

            inside = (
                (rr >= 0)
                & (rr < H)
                & (cc >= 0)
                & (cc < W)
            )

            if not np.any(inside):
                continue

            ids = np.flatnonzero(inside)

            r2 = rr[ids]
            c2 = cc[ids]

            good = (
                valid[r2, c2]
                &
                ~ps[r2, c2]
            )

            if not np.any(good):
                continue

            ids2 = ids[good]

            r3 = rr[ids2]
            c3 = cc[ids2]

            stat = glrt_statistic(
                center_scale[ids2],
                scale2[r3, c3],
                nslc=ndate,
            )

            out[
                ids2,
                ky,
                kx,
            ] = (
                np.isfinite(stat)
                &
                (stat < threshold)
            )

    return out


def compare_result(ref, cur):
    ref_ph, ref_est, ref_emi, ref_evd, ref_gm = ref
    ph, est, emi, evd, gm = cur

    if not np.array_equal(ref_est, est):
        return False, None

    finite = (
        np.isfinite(ref_ph.real)
        & np.isfinite(ref_ph.imag)
        & np.isfinite(ph.real)
        & np.isfinite(ph.imag)
    )

    if np.any(finite):
        phase_max = float(
            np.max(
                np.abs(
                    ref_ph[finite]
                    -
                    ph[finite]
                )
            )
        )
    else:
        phase_max = 0.0

    return (
        phase_max <= 5e-6,
        phase_max,
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
        default=16000,
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
        "--repeat",
        type=int,
        default=2,
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

    ndate = len(stack.dates)

    solver_n = min(
        ndate,
        int(
            cfg.get(
                "phase_linking",
                {},
            )
            .get(
                "temporal",
                {},
            )
            .get(
                "ministack_size",
                ndate,
            )
        ),
    )

    plan = build_runtime_plan(
        ndate=ndate
    )

    set_num_threads(
        plan.numba_threads
    )

    processing = (
        Path(paths.output_dir)
        /
        "processing"
    )

    stats = (
        processing
        /
        "ds_statistics"
    )

    scale2 = np.load(
        stats / "rayleigh_scale2.npy",
        mmap_mode="r",
    )

    raw_valid = np.load(
        stats / "raw_valid.npy",
        mmap_mode="r",
    )

    ps_raw = np.load(
        stats / "ps_mask.npy",
        mmap_mode="r",
    )

    geom_valid = np.load(
        processing
        / "cache"
        / "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    yxt = np.load(
        processing
        / "cache"
        / "phase_corrected_yxt.npy",
        mmap_mode="r",
    )

    valid = (
        np.asarray(raw_valid)
        &
        np.asarray(geom_valid)
    )

    ps = (
        np.asarray(ps_raw)
        &
        valid
    )

    # Production scientific center domain.
    center = (
        valid
        &
        ~ps
    )

    rr, cc = np.where(center)

    # ---------------------------------------------------------
    # Do not generate the full support cube.
    # Find a real eligible sample progressively.
    # ---------------------------------------------------------

    pairs = image_pairs(solver_n)
    pi = pairs[:, 0]
    pj = pairs[:, 1]

    want = int(args.sample)

    selected_r = []
    selected_c = []
    selected_s = []

    scan = 20000

    print("=" * 88)
    print("pyPSDS-GAMMA EMI CPU autotune")
    print("=" * 88)

    print("CPU              :", plan.cpu_count)
    print("dates            :", ndate)
    print("solver size      :", solver_n)
    print("pairs            :", pairs.shape[0])
    print("target sample    :", want)
    print("candidate centers:", rr.size)

    pos = 0

    while (
        pos < rr.size
        and
        sum(x.size for x in selected_r) < want
    ):
        end = min(
            rr.size,
            pos + scan,
        )

        br = rr[pos:end].astype(
            np.int32,
            copy=False,
        )

        bc = cc[pos:end].astype(
            np.int32,
            copy=False,
        )

        support = make_support_batch(
            scale2,
            valid,
            ps,
            br,
            bc,
            half_row=args.half_row,
            half_col=args.half_col,
            alpha=args.alpha,
            ndate=ndate,
        )

        K = np.sum(
            support,
            axis=(1, 2),
        )

        good = K >= args.min_shp

        if np.any(good):
            selected_r.append(
                br[good]
            )
            selected_c.append(
                bc[good]
            )
            selected_s.append(
                support[good]
            )

        pos = end

    if not selected_r:
        raise RuntimeError(
            "No eligible DS sample found."
        )

    gr = np.concatenate(
        selected_r
    )[:want]

    gc = np.concatenate(
        selected_c
    )[:want]

    gs = np.concatenate(
        selected_s,
        axis=0,
    )[:want]

    del (
        selected_r,
        selected_c,
        selected_s,
        rr,
        cc,
        center,
    )

    print("actual sample    :", gr.size)

    print()
    print("Building real coherence sample...")

    t0 = time.perf_counter()

    coh = compressed_coherence(
        yxt,
        gr,
        gc,
        gs,
        pi,
        pj,
    )

    print(
        "coherence build :",
        f"{time.perf_counter()-t0:.3f} s",
    )

    del gs

    # ---------------------------------------------------------
    # Candidate layouts.
    #
    # Intentionally includes fewer workers + larger chunks.
    # These usually reduce allocator/scheduler contention for
    # small 38x38 LAPACK problems.
    # ---------------------------------------------------------

    cpu = plan.cpu_count

    candidates = [
        (4, 2048),
        (8, 2048),
        (8, 1024),
        (12, 1024),
        (16, 1024),
        (16, 512),
        (24, 512),
        (32, 512),
        (24, 256),
        (32, 256),
    ]

    candidates = [
        (w, c)
        for w, c in candidates
        if w <= cpu
    ]

    # Remove layouts with substantially more workers than
    # actual chunks: those cannot improve parallelism.
    filtered = []

    for workers, chunk in candidates:
        nchunk = (
            coh.shape[0]
            +
            chunk
            -
            1
        ) // chunk

        if workers <= max(
            1,
            2 * nchunk,
        ):
            filtered.append(
                (workers, chunk)
            )

    candidates = filtered

    print()
    print(
        "candidate layouts:",
        candidates,
    )

    # ---------------------------------------------------------
    # Reference result.
    # ---------------------------------------------------------

    ref_workers, ref_chunk = candidates[0]

    print()
    print(
        "Building numerical reference..."
    )

    reference = robust_emi_threaded(
        coh,
        n_images=solver_n,
        pairs=pairs,
        beta=0.0,
        gamma_jitter=1e-6,
        emi_mu=0.99,
        reference_idx=0,
        workers=ref_workers,
        chunk_size=ref_chunk,
    )

    # ---------------------------------------------------------
    # Benchmark.
    # ---------------------------------------------------------

    rows = []

    print()
    print(
        f"{'workers':>8s} "
        f"{'chunk':>8s} "
        f"{'chunks':>8s} "
        f"{'best_s':>10s} "
        f"{'pts/s':>12s} "
        f"{'phase_diff':>12s} "
        f"{'parity':>8s}"
    )

    print("-" * 78)

    for workers, chunk in candidates:

        nchunk = (
            coh.shape[0]
            +
            chunk
            -
            1
        ) // chunk

        timings = []
        last = None

        for _ in range(
            max(
                1,
                args.repeat,
            )
        ):
            ts = time.perf_counter()

            last = robust_emi_threaded(
                coh,
                n_images=solver_n,
                pairs=pairs,
                beta=0.0,
                gamma_jitter=1e-6,
                emi_mu=0.99,
                reference_idx=0,
                workers=workers,
                chunk_size=chunk,
            )

            timings.append(
                time.perf_counter()
                -
                ts
            )

        best_s = min(timings)

        ok, phase_diff = compare_result(
            reference,
            last,
        )

        rate = (
            coh.shape[0]
            /
            best_s
        )

        row = {
            "workers":
                workers,

            "chunk":
                chunk,

            "chunks":
                nchunk,

            "seconds":
                best_s,

            "points_per_second":
                rate,

            "parity":
                bool(ok),

            "max_phase_difference":
                phase_diff,
        }

        rows.append(
            row
        )

        print(
            f"{workers:8d} "
            f"{chunk:8d} "
            f"{nchunk:8d} "
            f"{best_s:10.3f} "
            f"{rate:12.1f} "
            f"{phase_diff:12.3e} "
            f"{str(ok):>8s}"
        )

    valid_rows = [
        x
        for x in rows
        if x["parity"]
    ]

    if not valid_rows:
        raise RuntimeError(
            "No numerically valid EMI layout."
        )

    winner = min(
        valid_rows,
        key=lambda x: x[
            "seconds"
        ],
    )

    # ---------------------------------------------------------
    # Persist machine-specific plan.
    # ---------------------------------------------------------

    out = (
        processing
        /
        "ds_tiled"
        /
        "pl_cpu_autotune.json"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "format":
            "pyPSDS-GAMMA-pl-cpu-autotune-v1",

        "machine":
            platform.node(),

        "platform":
            platform.platform(),

        "python":
            platform.python_version(),

        "numpy":
            np.__version__,

        "cpu_count":
            cpu,

        "ndate":
            ndate,

        "solver_size":
            solver_n,

        "npair":
            int(
                pairs.shape[0]
            ),

        "sample_points":
            int(
                coh.shape[0]
            ),

        "results":
            rows,

        "winner":
            winner,
    }

    out.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("WINNER")
    print("=" * 88)

    print(
        "workers      :",
        winner["workers"],
    )

    print(
        "chunk        :",
        winner["chunk"],
    )

    print(
        "seconds      :",
        winner["seconds"],
    )

    print(
        "points/s     :",
        winner["points_per_second"],
    )

    print(
        "parity       :",
        winner["parity"],
    )

    print(
        "saved        :",
        out,
    )

    print()
    print(
        "EMI CPU AUTOTUNE: PASS"
    )


if __name__ == "__main__":
    main()
