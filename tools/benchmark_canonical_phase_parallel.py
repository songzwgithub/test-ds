#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config
from pypsds.phase_linking.phase_source import (
    GammaStreamingPhaseSource,
    canonical_autotune_runtime_identity,
)
from pypsds.runtime import logical_cpu_count


FORMAT = "pyPSDS-GAMMA-canonical-phase-parallel-benchmark-v2"


def parse_candidates(text: str):
    out = []
    for token in str(text).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(
                f"invalid candidate {token!r}; expected e.g. 4x6"
            )
        a, b = token.split("x", 1)
        s = int(a)
        p = int(b)
        if s < 1 or p < 1:
            raise ValueError("workers must be >= 1")
        out.append((s, p))
    if not out:
        raise ValueError("no candidates supplied")
    return out


def set_nested_phase_scratch(cfg: dict, scratch: Path):
    pc = cfg.setdefault("phase_correction", {})
    pc["scratch_dir"] = str(scratch.resolve())


def build_source(
    *,
    cfg,
    paths,
    stack,
    base_row0,
    base_col0,
    io_workers,
    spatial_workers,
    pair_workers,
):
    source = GammaStreamingPhaseSource(
        cfg=cfg,
        paths=paths,
        stack=stack,
        base_row0=base_row0,
        base_col0=base_col0,
        io_workers=io_workers,
    )

    source.spatial_workers = int(spatial_workers)
    source.pair_workers = int(pair_workers)
    source.provider._phase_sim_workers_override = int(pair_workers)

    # Keep enough canonical cells for several waves.
    source.cache_max_cells = max(
        int(source.cache_max_cells),
        4 * int(spatial_workers),
    )
    source._cache.clear()

    return source


def compare(reference, candidate):
    rg = reference.geometry_valid
    cg = candidate.geometry_valid

    geometry_mismatch = int(np.count_nonzero(rg != cg))

    a = reference.yxt
    b = candidate.yxt

    if a.shape != b.shape:
        raise RuntimeError(
            f"phase shape mismatch: {a.shape} != {b.shape}"
        )

    finite_a = np.isfinite(a.real) & np.isfinite(a.imag)
    finite_b = np.isfinite(b.real) & np.isfinite(b.imag)
    finite_pattern_equal = bool(
        np.array_equal(finite_a, finite_b)
    )

    both = finite_a & finite_b

    if not np.any(both):
        return {
            "geometry_mismatch": geometry_mismatch,
            "finite_pattern_equal": finite_pattern_equal,
            "complex_exact": True,
            "max_abs_complex_difference": 0.0,
            "max_abs_phase_difference_rad": 0.0,
        }

    av = a[both]
    bv = b[both]

    complex_exact = bool(np.array_equal(av, bv))
    max_complex = float(np.max(np.abs(av - bv)))

    nz = (np.abs(av) > 0.0) & (np.abs(bv) > 0.0)
    if np.any(nz):
        phase_diff = np.angle(av[nz] * np.conj(bv[nz]))
        max_phase = float(np.max(np.abs(phase_diff)))
    else:
        max_phase = 0.0

    return {
        "geometry_mismatch": geometry_mismatch,
        "finite_pattern_equal": finite_pattern_equal,
        "complex_exact": complex_exact,
        "max_abs_complex_difference": max_complex,
        "max_abs_phase_difference_rad": max_phase,
    }


def parity_pass(cmp):
    # Same canonical 128x256 point-list grouping should normally
    # reproduce bit-exact output.  Keep a tiny fallback tolerance
    # for platform/library effects.
    return bool(
        cmp["geometry_mismatch"] == 0
        and cmp["finite_pattern_equal"]
        and (
            cmp["complex_exact"]
            or (
                cmp["max_abs_complex_difference"] <= 5.0e-6
                and cmp["max_abs_phase_difference_rad"] <= 5.0e-6
            )
        )
    )


def run_once(
    *,
    cfg,
    paths,
    stack,
    base_row0,
    base_col0,
    io_workers,
    spatial_workers,
    pair_workers,
    row0,
    col0,
    rows,
    cols,
    date_indices,
):
    source = build_source(
        cfg=cfg,
        paths=paths,
        stack=stack,
        base_row0=base_row0,
        base_col0=base_col0,
        io_workers=io_workers,
        spatial_workers=spatial_workers,
        pair_workers=pair_workers,
    )

    t0 = time.perf_counter()

    tile = source.read_tile(
        local_row0=row0,
        local_row1=row0 + rows,
        local_col0=col0,
        local_col1=col0 + cols,
        date_indices=date_indices,
    )

    elapsed = time.perf_counter() - t0

    return tile, elapsed


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark canonical GAMMA phase streaming parallelism "
            "without changing the validated 128x256 numerical grid."
        )
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--row0", type=int, default=6144)
    ap.add_argument("--col0", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--date-start", type=int, default=0)
    ap.add_argument("--ndates", type=int, default=19)
    ap.add_argument("--io-workers", type=int, default=4)
    ap.add_argument(
        "--candidates",
        default="1x16,2x12,3x8,4x6,6x4",
    )
    ap.add_argument(
        "--mode",
        choices=("cold", "warm", "both"),
        default="both",
        help=(
            "cold: each candidate builds its own geometry cache; "
            "warm: prebuild one shared geometry cache and benchmark "
            "phase streaming with geometry reuse; both: run both."
        ),
    )
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument(
        "--scratch-root",
        default=None,
        help=(
            "benchmark scratch root; default is "
            "<output>/benchmark/canonical_phase_parallel"
        ),
    )
    ap.add_argument(
        "--install-winner",
        action="store_true",
        help=(
            "write the best parity-passing schedule to "
            "<output>/processing/canonical_phase_parallel_autotune.json"
        ),
    )
    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (base_row0, base_col0, H, W),
    ) = open_from_config(args.config)

    runtime_identity = canonical_autotune_runtime_identity(
        cfg,
        stack,
    )

    candidates = parse_candidates(args.candidates)

    if args.row0 < 0 or args.col0 < 0:
        raise ValueError("row0/col0 must be >= 0")
    if args.row0 + args.rows > H:
        raise ValueError("benchmark rows outside processing ROI")
    if args.col0 + args.cols > W:
        raise ValueError("benchmark cols outside processing ROI")

    d0 = int(args.date_start)
    d1 = d0 + int(args.ndates)
    if d0 < 0 or d1 > len(stack.dates):
        raise ValueError(
            f"date subset {d0}:{d1} outside 0:{len(stack.dates)}"
        )
    date_indices = tuple(range(d0, d1))

    if args.scratch_root is None:
        scratch_root = (
            Path(paths.output_dir)
            / "benchmark"
            / "canonical_phase_parallel"
        )
    else:
        scratch_root = Path(args.scratch_root).expanduser().resolve()

    scratch_root.mkdir(parents=True, exist_ok=True)

    canonical_rows = 128
    canonical_cols = 256
    ncell = (
        math.ceil(args.rows / canonical_rows)
        * math.ceil(args.cols / canonical_cols)
    )
    npix = int(args.rows) * int(args.cols)

    print("=" * 108)
    print("pyPSDS-GAMMA canonical GAMMA phase parallel benchmark")
    print("=" * 108)
    print("config           :", config_path)
    print("scene            :", f"{H} x {W}")
    print(
        "benchmark ROI    :",
        f"r{args.row0}:{args.row0+args.rows} "
        f"c{args.col0}:{args.col0+args.cols}",
    )
    print("ROI pixels       :", f"{npix:,}")
    print("canonical cells  :", ncell)
    print(
        "dates            :",
        f"{d0}:{d1-1} ({len(date_indices)})",
    )
    print("logical CPUs     :", logical_cpu_count())
    print("effective CPUs   :", runtime_identity["effective_cpu_count"])
    print("io workers       :", args.io_workers)
    print("candidates       :", candidates)
    print("mode             :", args.mode)
    print("scratch          :", scratch_root)
    print()

    all_results = []
    reference_by_mode = {}

    modes = (
        ("cold", "warm")
        if args.mode == "both"
        else (args.mode,)
    )

    for mode in modes:
        print("=" * 108)
        print("MODE:", mode.upper())
        print("=" * 108)

        shared_scratch = scratch_root / "warm_shared"

        if mode == "warm":
            # Ensure all candidates start with the same persistent
            # geometry cache.  Use a discarded 1x16 warmup first.
            shutil.rmtree(shared_scratch, ignore_errors=True)

            warm_cfg = copy.deepcopy(cfg)
            set_nested_phase_scratch(warm_cfg, shared_scratch)

            print(
                "Warm-up: building shared geometry cache "
                "with baseline 1x16 ..."
            )
            _tile, warm_s = run_once(
                cfg=warm_cfg,
                paths=paths,
                stack=stack,
                base_row0=base_row0,
                base_col0=base_col0,
                io_workers=args.io_workers,
                spatial_workers=1,
                pair_workers=min(16, len(date_indices)),
                row0=args.row0,
                col0=args.col0,
                rows=args.rows,
                cols=args.cols,
                date_indices=date_indices,
            )
            del _tile
            print(f"Warm-up elapsed : {warm_s:.3f} s")
            print()

        for spatial_workers, pair_workers in candidates:
            timings = []
            last_tile = None

            for rep in range(max(1, int(args.repeats))):
                if mode == "cold":
                    candidate_scratch = (
                        scratch_root
                        / (
                            f"cold_s{spatial_workers}_"
                            f"p{pair_workers}_r{rep}"
                        )
                    )
                    shutil.rmtree(
                        candidate_scratch,
                        ignore_errors=True,
                    )
                else:
                    candidate_scratch = shared_scratch

                candidate_cfg = copy.deepcopy(cfg)
                set_nested_phase_scratch(
                    candidate_cfg,
                    candidate_scratch,
                )

                tile, elapsed = run_once(
                    cfg=candidate_cfg,
                    paths=paths,
                    stack=stack,
                    base_row0=base_row0,
                    base_col0=base_col0,
                    io_workers=args.io_workers,
                    spatial_workers=spatial_workers,
                    pair_workers=pair_workers,
                    row0=args.row0,
                    col0=args.col0,
                    rows=args.rows,
                    cols=args.cols,
                    date_indices=date_indices,
                )

                timings.append(float(elapsed))
                last_tile = tile

            median_s = float(
                np.median(
                    np.asarray(timings, dtype=np.float64)
                )
            )

            throughput_mpix_s = (
                float(npix) / median_s / 1.0e6
            )
            cells_per_hour = (
                float(ncell) / median_s * 3600.0
            )

            if mode not in reference_by_mode:
                reference_by_mode[mode] = last_tile
                cmp = {
                    "geometry_mismatch": 0,
                    "finite_pattern_equal": True,
                    "complex_exact": True,
                    "max_abs_complex_difference": 0.0,
                    "max_abs_phase_difference_rad": 0.0,
                }
                parity = True
            else:
                cmp = compare(
                    reference_by_mode[mode],
                    last_tile,
                )
                parity = parity_pass(cmp)

            row = {
                "mode": mode,
                "spatial_workers": int(spatial_workers),
                "pair_workers": int(pair_workers),
                "max_gamma_processes": int(
                    spatial_workers * pair_workers
                ),
                "median_seconds": median_s,
                "timings_seconds": timings,
                "throughput_mpix_per_second": (
                    throughput_mpix_s
                ),
                "canonical_cells_per_hour": cells_per_hour,
                "parity": bool(parity),
                **cmp,
            }
            all_results.append(row)

            print(
                f"{mode:5s} "
                f"s={spatial_workers:2d} "
                f"p={pair_workers:2d} "
                f"gamma={spatial_workers*pair_workers:2d} "
                f"median={median_s:8.3f}s "
                f"cells/h={cells_per_hour:9.1f} "
                f"Mpix/s={throughput_mpix_s:8.4f} "
                f"parity={'PASS' if parity else 'FAIL'} "
                f"dphi={cmp['max_abs_phase_difference_rad']:.3e}"
            )

        print()

    valid = [x for x in all_results if x["parity"]]

    # Prefer warm mode for production after geometry-cache build;
    # if only cold mode was requested, use cold.
    preferred_mode = (
        "warm"
        if any(x["mode"] == "warm" for x in valid)
        else "cold"
    )
    pool = [
        x for x in valid
        if x["mode"] == preferred_mode
    ]

    if not pool:
        raise RuntimeError(
            "no parity-passing candidate schedule"
        )

    winner = min(
        pool,
        key=lambda x: (
            x["median_seconds"],
            x["max_gamma_processes"],
        ),
    )

    baseline = next(
        (
            x
            for x in pool
            if x["spatial_workers"] == 1
            and x["pair_workers"] == 16
        ),
        pool[0],
    )

    speedup = (
        baseline["median_seconds"]
        / winner["median_seconds"]
    )

    payload = {
        "format": FORMAT,
        "config": str(config_path),
        "canonical_tile": [128, 256],
        "region": [
            int(args.row0),
            int(args.row0 + args.rows),
            int(args.col0),
            int(args.col0 + args.cols),
        ],
        "date_indices": list(date_indices),
        "logical_cpu_count": int(logical_cpu_count()),
        "runtime_identity": runtime_identity,
        "io_workers": int(args.io_workers),
        "preferred_mode": preferred_mode,
        "baseline": baseline,
        "winner": {
            **winner,
            # Key consumed by GammaStreamingPhaseSource.
            "parity": True,
        },
        "speedup_vs_1x16": float(speedup),
        "results": all_results,
    }

    report_path = scratch_root / "benchmark.json"
    report_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("WINNER")
    print("=" * 108)
    print("mode             :", preferred_mode)
    print(
        "schedule         :",
        f"{winner['spatial_workers']}x"
        f"{winner['pair_workers']}",
    )
    print(
        "median seconds   :",
        f"{winner['median_seconds']:.3f}",
    )
    print(
        "cells/hour       :",
        f"{winner['canonical_cells_per_hour']:.1f}",
    )
    print(
        "speedup vs 1x16  :",
        f"{speedup:.3f}x",
    )
    print("numerical parity : PASS")
    print("report           :", report_path)

    if args.install_winner:
        tune_path = (
            Path(paths.output_dir)
            / "processing"
            / "canonical_phase_parallel_autotune.json"
        )

        install_payload = {
            "format": FORMAT,
            "canonical_tile": [128, 256],
            "runtime_identity": runtime_identity,
            "winner": {
                "parity": True,
                "spatial_workers": int(
                    winner["spatial_workers"]
                ),
                "pair_workers": int(
                    winner["pair_workers"]
                ),
                "median_seconds": float(
                    winner["median_seconds"]
                ),
                "canonical_cells_per_hour": float(
                    winner["canonical_cells_per_hour"]
                ),
                "speedup_vs_1x16": float(speedup),
                "mode": preferred_mode,
            },
            "benchmark_report": str(report_path),
        }

        tune_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        tmp = tune_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                install_payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(tune_path)

        print("installed tune   :", tune_path)


if __name__ == "__main__":
    main()
