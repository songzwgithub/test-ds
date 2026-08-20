#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config
from pypsds.phase_linking.sequential_multistage import run_sequential_stage
from pypsds.phase_linking.temporal_plan import (
    TemporalStrategy,
    build_temporal_plan,
)


def max_finite_diff(a, b):
    good = (
        np.isfinite(a.real)
        & np.isfinite(a.imag)
        & np.isfinite(b.real)
        & np.isfinite(b.imag)
    )
    if not np.any(good):
        return float("nan")
    return float(np.max(np.abs(a[good] - b[good])))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--ministack-size", type=int, default=12)
    ap.add_argument("--max-num-compressed", type=int, default=5)
    ap.add_argument("--state-min-shp", type=int, default=24)

    ap.add_argument("--tile-rows", type=int, default=256)
    ap.add_argument("--tile-cols", type=int, default=512)
    ap.add_argument("--center-batch", type=int, default=16000)
    ap.add_argument("--support-block", type=int, default=1024)

    ap.add_argument("--pl-workers", type=int, default=16)
    ap.add_argument("--pl-chunk", type=int, default=512)

    ap.add_argument("--beta", type=float, default=0.05, help="EMI Gamma regularization beta")

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (row0, col0, H, W),
    ) = open_from_config(args.config)

    processing = Path(paths.output_dir) / "processing"
    seqdir = processing / "sequential"
    seqdir.mkdir(parents=True, exist_ok=True)

    yxt_path = processing / "cache" / "phase_corrected_yxt.npy"
    geom_path = processing / "cache" / "phase_geometry_valid.npy"
    scale_path = processing / "ds_statistics" / "rayleigh_scale2.npy"
    raw_valid_path = processing / "ds_statistics" / "raw_valid.npy"
    ps_path = processing / "ps_mask.npy"

    core_path = seqdir / "compression_state_core_K24.npy"
    effective_k_path = (
        seqdir
        / "compression_state_core_K24_effective_shp_count.npy"
    )

    required = (
        yxt_path,
        geom_path,
        scale_path,
        raw_valid_path,
        ps_path,
        core_path,
        effective_k_path,
    )

    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)

    yxt = np.load(yxt_path, mmap_mode="r")
    geom = np.load(geom_path, mmap_mode="r")
    scale2 = np.load(scale_path, mmap_mode="r")
    raw_valid = np.load(raw_valid_path, mmap_mode="r")
    ps = np.load(ps_path, mmap_mode="r")
    state_core = np.load(core_path, mmap_mode="r")
    expected_k = np.load(effective_k_path, mmap_mode="r")

    ndate = int(yxt.shape[2])

    if yxt.shape != (H, W, ndate):
        raise RuntimeError(
            f"YXT shape={yxt.shape}, expected H/W={H}/{W}"
        )

    valid = (
        np.asarray(raw_valid, dtype=np.bool_)
        & np.asarray(geom, dtype=np.bool_)
    )

    dates = tuple(str(x) for x in stack.dates)

    plan = build_temporal_plan(
        dates,
        strategy=TemporalStrategy.SEQUENTIAL,
        ministack_size=args.ministack_size,
        max_num_compressed=args.max_num_compressed,
        reference_index=0,
    )

    if plan.effective_strategy != TemporalStrategy.SEQUENTIAL.value:
        raise RuntimeError(
            "U3.3b requires true ministack_size < ndate sequential plan"
        )

    if len(plan.stages) < 2:
        raise RuntimeError("U3.3b requires multiple ministacks")

    print("=" * 100)
    print("U3.3b multi-ministack sequential executor")
    print("=" * 100)
    print("config          :", config_path)
    print("scene           :", f"{H} x {W}")
    print("dates           :", ndate)
    print("ministack       :", args.ministack_size)
    print("max compressed  :", args.max_num_compressed)
    print("stage count     :", len(plan.stages))
    print("K24 state core  :", f"{np.count_nonzero(state_core):,}")
    print()

    for stage in plan.stages:
        ncomp = stage.compressed_count
        ref_idx = 0 if ncomp == 0 else ncomp - 1

        print(
            f"stage {stage.stage_index}: "
            f"compressed={ncomp}, "
            f"real={list(stage.real_indices)}, "
            f"solver={stage.solver_size}, "
            f"first_real={ncomp}, "
            f"reference={ref_idx}"
        )

    print()

    compressed_registry = {}
    valid_registry = {}

    results = []

    stage0_u33a_diff = None
    stage0_u33a_valid_equal = None

    total_seconds = 0.0

    for stage in plan.stages:

        input_ids = tuple(
            x.ref_id
            for x in stage.compressed_inputs
        )

        compressed_inputs = tuple(
            compressed_registry[x]
            for x in input_ids
        )

        compressed_valids = tuple(
            valid_registry[x]
            for x in input_ids
        )

        if compressed_valids:
            inputs_complete = all(
                bool(np.all(v[state_core]))
                for v in compressed_valids
            )
        else:
            inputs_complete = True

        result = run_sequential_stage(
            stage_index=stage.stage_index,

            compressed_input_ids=input_ids,
            compressed_inputs=compressed_inputs,

            yxt=yxt,
            real_indices=stage.real_indices,

            scale2=scale2,
            valid=valid,
            ps=ps,

            state_core=state_core,
            expected_effective_k=expected_k,

            output_dir=seqdir,

            full_glrt_nslc=ndate,
            state_min_shp=args.state_min_shp,

            inputs_complete=inputs_complete,

            half_row=5,
            half_col=11,
            alpha=0.005,

            beta=args.beta,
            gamma_jitter=1.0e-6,
            emi_mu=0.99,

            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,

            center_batch=args.center_batch,
            support_block=args.support_block,

            pl_workers=args.pl_workers,
            pl_chunk_size=args.pl_chunk,

            formula_audit_points=5000,
        )

        if stage.compressed_output is None:
            raise RuntimeError(
                f"stage {stage.stage_index} has no compressed output"
            )

        ref_id = stage.compressed_output.ref_id

        compressed_registry[ref_id] = np.load(
            result.compressed_path,
            mmap_mode="r",
        )

        valid_registry[ref_id] = np.load(
            result.valid_path,
            mmap_mode="r",
        )

        total_seconds += result.elapsed_seconds

        valid_fraction = (
            result.state_valid
            / result.state_pixels
            if result.state_pixels
            else 0.0
        )

        # ----------------------------------------------------
        # Stage-0 parity against already validated U3.3a.
        # ----------------------------------------------------

        if (
            stage.stage_index == 0
            and args.ministack_size == 12
            and abs(args.beta - 0.05) <= 1e-12
        ):

            old_comp_path = (
                seqdir
                / "u33a_stage0000_compressed.npy"
            )

            old_valid_path = (
                seqdir
                / "u33a_stage0000_state_valid.npy"
            )

            if (
                old_comp_path.is_file()
                and old_valid_path.is_file()
            ):
                old_comp = np.load(
                    old_comp_path,
                    mmap_mode="r",
                )

                old_valid = np.load(
                    old_valid_path,
                    mmap_mode="r",
                )

                stage0_u33a_diff = max_finite_diff(
                    compressed_registry[ref_id],
                    old_comp,
                )

                stage0_u33a_valid_equal = bool(
                    np.array_equal(
                        valid_registry[ref_id],
                        old_valid,
                    )
                )

                if not stage0_u33a_valid_equal:
                    raise RuntimeError(
                        "stage-0 valid mask differs from U3.3a"
                    )

                if (
                    not np.isfinite(stage0_u33a_diff)
                    or stage0_u33a_diff > 1.0e-5
                ):
                    raise RuntimeError(
                        "stage-0 compressed SLC differs from U3.3a: "
                        f"{stage0_u33a_diff}"
                    )

        item = {
            "stage_index": result.stage_index,
            "compressed_input_ids": list(
                result.compressed_input_ids
            ),
            "real_indices": list(result.real_indices),

            "solver_size": result.solver_size,
            "first_real_idx": result.first_real_idx,
            "reference_idx": result.reference_idx,

            "inputs_complete": inputs_complete,

            "state_pixels": result.state_pixels,
            "state_valid": result.state_valid,
            "state_valid_fraction": valid_fraction,

            "low_k": result.low_k,
            "pl_invalid": result.pl_invalid,
            "compression_invalid": result.compression_invalid,
            "center_input_invalid": result.center_input_invalid,

            "static_k_excess": result.static_k_excess,
            "static_k_mismatch": result.static_k_mismatch,

            "compression_formula_max_abs_diff":
                result.compression_formula_max_abs_diff,

            "support_seconds": result.support_seconds,
            "covariance_seconds": result.covariance_seconds,
            "phase_linking_seconds":
                result.phase_linking_seconds,
            "compression_seconds":
                result.compression_seconds,
            "elapsed_seconds": result.elapsed_seconds,

            "compressed_ref_id": ref_id,
            "compressed_path": str(
                result.compressed_path
            ),
            "valid_path": str(
                result.valid_path
            ),
        }

        results.append(item)

        print()
        print("-" * 100)
        print(f"STAGE {stage.stage_index} RESULT")
        print("-" * 100)

        print(
            "state valid          :",
            f"{result.state_valid:,}/"
            f"{result.state_pixels:,}",
            f"({100.0*valid_fraction:.6f}%)",
        )

        print("K < 24              :", result.low_k)
        print("PL invalid          :", result.pl_invalid)
        print(
            "compression invalid :",
            result.compression_invalid,
        )
        print(
            "center invalid      :",
            result.center_input_invalid,
        )
        print(
            "K excess            :",
            result.static_k_excess,
        )
        print(
            "K parity mismatch   :",
            result.static_k_mismatch,
        )
        print(
            "compression max diff:",
            result.compression_formula_max_abs_diff,
        )
        print(
            "elapsed             :",
            f"{result.elapsed_seconds:.3f} s",
        )

    # --------------------------------------------------------
    # Overall gates.
    # --------------------------------------------------------

    formula_ok = all(
        x["compression_formula_max_abs_diff"] <= 1.0e-5
        for x in results
    )

    no_k_excess = all(
        x["static_k_excess"] == 0
        for x in results
    )

    dense_parity = all(
        (not x["inputs_complete"])
        or x["static_k_mismatch"] == 0
        for x in results
    )

    dense_complete = all(
        x["state_valid"] == x["state_pixels"]
        for x in results
    )

    if not formula_ok:
        decision = "FAIL_compression_formula"

    elif not no_k_excess:
        decision = "FAIL_dynamic_support_invariant"

    elif not dense_parity:
        decision = "FAIL_dense_input_K_parity"

    elif dense_complete:
        decision = "multistage_dense_state_complete"

    else:
        decision = (
            "multistage_dynamic_state_validity_required"
        )

    report = {
        "format":
            "pyPSDS-GAMMA-U3.3b-multistage-v1",

        "config": str(config_path),
        "shape": [H, W],
        "ndate": ndate,

        "ministack_size":
            args.ministack_size,

        "max_num_compressed":
            args.max_num_compressed,

        "state_min_shp":
            args.state_min_shp,

        "stage_count":
            len(plan.stages),

        "state_core_pixels":
            int(np.count_nonzero(state_core)),

        "stage0_u33a_max_abs_diff":
            stage0_u33a_diff,

        "stage0_u33a_valid_equal":
            stage0_u33a_valid_equal,

        "total_elapsed_seconds":
            total_seconds,

        "stages":
            results,

        "decision":
            decision,
    }

    report_path = (
        seqdir
        / "u33b_multistage_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 132)
    print("U3.3b MULTISTAGE SUMMARY")
    print("=" * 132)

    print(
        "stage | solver | ref | "
        "state valid              | K<24 | "
        "PLbad | COMPbad | K mismatch | "
        "formula diff | seconds"
    )

    print("-" * 132)

    for x in results:
        print(
            f"{x['stage_index']:5d} | "
            f"{x['solver_size']:6d} | "
            f"{x['reference_idx']:3d} | "
            f"{x['state_valid']:9,d}/"
            f"{x['state_pixels']:9,d} | "
            f"{x['low_k']:5,d} | "
            f"{x['pl_invalid']:5,d} | "
            f"{x['compression_invalid']:7,d} | "
            f"{x['static_k_mismatch']:10,d} | "
            f"{x['compression_formula_max_abs_diff']:12.4g} | "
            f"{x['elapsed_seconds']:7.2f}"
        )

    print()

    print(
        "stage0 vs U3.3a max diff :",
        stage0_u33a_diff,
    )

    print(
        "stage0 valid equal       :",
        stage0_u33a_valid_equal,
    )

    print(
        "total stage time         :",
        f"{total_seconds:.3f} s",
    )

    print(
        "decision                 :",
        decision,
    )

    print(
        "report                   :",
        report_path,
    )

    print()

    if decision.startswith("FAIL"):
        raise RuntimeError(
            f"U3.3b failed: {decision}"
        )

    print(
        "U3.3b MULTISTAGE SEQUENTIAL EXECUTOR: PASS"
    )


if __name__ == "__main__":
    main()
