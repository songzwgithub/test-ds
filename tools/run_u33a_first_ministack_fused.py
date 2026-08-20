#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config

from _historical_u33_sequential_executor import (
    run_first_ministack_fused,
)

from pypsds.phase_linking.temporal_plan import (
    TemporalStrategy,
    build_temporal_plan,
)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--ministack-size",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--max-num-compressed",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--state-min-shp",
        type=int,
        default=24,
    )

    ap.add_argument(
        "--tile-rows",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--tile-cols",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--center-batch",
        type=int,
        default=16000,
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--pl-workers",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--pl-chunk",
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
            row0,
            col0,
            H,
            W,
        ),
    ) = open_from_config(
        args.config
    )

    processing = (
        Path(
            paths.output_dir
        )
        /
        "processing"
    )

    seqdir = (
        processing
        /
        "sequential"
    )

    seqdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    yxt_path = (
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

    scale_path = (
        processing
        /
        "ds_statistics"
        /
        "rayleigh_scale2.npy"
    )

    raw_valid_path = (
        processing
        /
        "ds_statistics"
        /
        "raw_valid.npy"
    )

    ps_path = (
        processing
        /
        "ps_mask.npy"
    )

    core_path = (
        seqdir
        /
        "compression_state_core_K24.npy"
    )

    effective_k_path = (
        seqdir
        /
        "compression_state_core_K24_effective_shp_count.npy"
    )

    for p in (
        yxt_path,
        geom_path,
        scale_path,
        raw_valid_path,
        ps_path,
        core_path,
        effective_k_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(
                p
            )

    yxt = np.load(
        yxt_path,
        mmap_mode="r",
    )

    geom = np.load(
        geom_path,
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

    ps = np.load(
        ps_path,
        mmap_mode="r",
    )

    state_core = np.load(
        core_path,
        mmap_mode="r",
    )

    expected_k = np.load(
        effective_k_path,
        mmap_mode="r",
    )

    ndate = int(
        yxt.shape[2]
    )

    if yxt.shape != (
        H,
        W,
        ndate,
    ):
        raise RuntimeError(
            f"YXT shape={yxt.shape}, "
            f"ROI expected H/W={H}/{W}"
        )

    valid = (
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

    dates = tuple(
        str(x)
        for x
        in stack.dates
    )

    plan = build_temporal_plan(
        dates,
        strategy=(
            TemporalStrategy.SEQUENTIAL
        ),
        ministack_size=(
            args.ministack_size
        ),
        max_num_compressed=(
            args.max_num_compressed
        ),
        reference_index=0,
    )

    if (
        plan.effective_strategy
        !=
        TemporalStrategy.SEQUENTIAL.value
    ):
        raise RuntimeError(
            "U3.3a requires a true "
            "M < N sequential plan"
        )

    if len(
        plan.stages
    ) < 2:
        raise RuntimeError(
            "Need at least two stages "
            "for true sequential prototype"
        )

    stage = plan.stages[
        0
    ]

    if stage.compressed_count != 0:
        raise RuntimeError(
            "stage 0 unexpectedly has "
            "compressed inputs"
        )

    if (
        stage.output_reference
        !=
        "real:0"
    ):
        raise RuntimeError(
            "stage 0 output reference "
            "is not real:0"
        )

    print(
        "=" * 96
    )

    print(
        "U3.3a first-ministack fused "
        "sequential executor"
    )

    print(
        "=" * 96
    )

    print(
        "config                 :",
        config_path,
    )

    print(
        "scene                  :",
        f"{H} x {W}",
    )

    print(
        "full dates             :",
        ndate,
    )

    print(
        "ministack size         :",
        args.ministack_size,
    )

    print(
        "stage 0 real indices   :",
        list(
            stage.real_indices
        ),
    )

    print(
        "stage 0 real dates     :",
        list(
            stage.real_dates
        ),
    )

    print(
        "state min SHP          :",
        args.state_min_shp,
    )

    print(
        "state core pixels      :",
        f"{np.count_nonzero(state_core):,}",
    )

    print(
        "tile                   :",
        f"{args.tile_rows} x "
        f"{args.tile_cols}",
    )

    print(
        "center batch           :",
        args.center_batch,
    )

    print(
        "PL workers             :",
        args.pl_workers,
    )

    print(
        "PL chunk               :",
        args.pl_chunk,
    )

    print()

    result = (
        run_first_ministack_fused(
            yxt=yxt,
            scale2=scale2,
            valid=valid,
            ps=ps,
            state_core=state_core,
            expected_effective_k=(
                expected_k
            ),
            real_indices=(
                stage.real_indices
            ),
            output_dir=seqdir,
            full_glrt_nslc=ndate,
            state_min_shp=(
                args.state_min_shp
            ),
            half_row=5,
            half_col=11,
            alpha=0.005,
            beta=0.05,
            gamma_jitter=1e-6,
            emi_mu=0.99,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            center_batch=(
                args.center_batch
            ),
            support_block=(
                args.support_block
            ),
            pl_workers=(
                args.pl_workers
            ),
            pl_chunk_size=(
                args.pl_chunk
            ),
            formula_audit_points=5000,
        )
    )

    valid_fraction = (
        result.state_valid
        /
        result.state_pixels
        if result.state_pixels
        else 0.0
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if (
        result.k_parity_mismatch
        !=
        0
    ):
        decision = (
            "FAIL_tile_halo_K_mismatch"
        )

    elif (
        result.compression_formula_max_abs_diff
        >
        1e-5
    ):
        decision = (
            "FAIL_compression_formula_mismatch"
        )

    elif (
        result.state_valid
        ==
        result.state_pixels
    ):
        decision = (
            "first_ministack_dense_state_complete"
        )

    else:
        decision = (
            "first_ministack_dynamic_state_validity_required"
        )

    report = {
        "format":
            "pyPSDS-GAMMA-U3.3a-first-ministack-fused-v1",

        "config":
            str(
                config_path
            ),

        "shape":
            [
                H,
                W,
            ],

        "full_ndate":
            ndate,

        "ministack_size":
            args.ministack_size,

        "stage_index":
            0,

        "stage_real_indices":
            list(
                stage.real_indices
            ),

        "stage_real_dates":
            list(
                stage.real_dates
            ),

        "state_min_shp":
            args.state_min_shp,

        "state_pixels":
            result.state_pixels,

        "state_valid":
            result.state_valid,

        "state_valid_fraction":
            valid_fraction,

        "low_k":
            result.low_k,

        "pl_invalid":
            result.pl_invalid,

        "compression_invalid":
            result.compression_invalid,

        "center_input_invalid":
            result.center_input_invalid,

        "k_parity_mismatch":
            result.k_parity_mismatch,

        "compression_formula_max_abs_diff":
            result.compression_formula_max_abs_diff,

        "elapsed_seconds":
            result.elapsed_seconds,

        "support_seconds":
            result.support_seconds,

        "covariance_seconds":
            result.covariance_seconds,

        "phase_linking_seconds":
            result.phase_linking_seconds,

        "compression_seconds":
            result.compression_seconds,

        "compressed_path":
            str(
                result.compressed_path
            ),

        "valid_path":
            str(
                result.valid_path
            ),

        "state_code_path":
            str(
                result.state_code_path
            ),

        "shp_count_path":
            str(
                result.shp_count_path
            ),

        "temporal_coherence_path":
            str(
                result.temporal_coherence_path
            ),

        "estimator_path":
            str(
                result.estimator_path
            ),

        "decision":
            decision,
    }

    json_path = (
        seqdir
        /
        "u33a_stage0000_report.json"
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

    print()
    print(
        "=" * 96
    )

    print(
        "U3.3a result"
    )

    print(
        "=" * 96
    )

    print(
        "state pixels            :",
        f"{result.state_pixels:,}",
    )

    print(
        "state valid             :",
        f"{result.state_valid:,}",
        f"({100*valid_fraction:.6f}%)",
    )

    print(
        "K < 24                 :",
        f"{result.low_k:,}",
    )

    print(
        "PL invalid             :",
        f"{result.pl_invalid:,}",
    )

    print(
        "compression invalid    :",
        f"{result.compression_invalid:,}",
    )

    print(
        "center input invalid   :",
        f"{result.center_input_invalid:,}",
    )

    print()

    print(
        "K parity mismatch      :",
        result.k_parity_mismatch,
    )

    print(
        "compression max |diff| :",
        f"{result.compression_formula_max_abs_diff:.9g}",
    )

    print()

    print(
        "support                :",
        f"{result.support_seconds:.3f} s",
    )

    print(
        "covariance             :",
        f"{result.covariance_seconds:.3f} s",
    )

    print(
        "phase linking          :",
        f"{result.phase_linking_seconds:.3f} s",
    )

    print(
        "compression            :",
        f"{result.compression_seconds:.3f} s",
    )

    print(
        "total                  :",
        f"{result.elapsed_seconds:.3f} s",
    )

    if result.elapsed_seconds > 0:

        print(
            "effective throughput   :",
            f"{result.state_pixels/result.elapsed_seconds:,.0f} "
            "state center/s",
        )

    print()

    print(
        "compressed SLC         :",
        result.compressed_path,
    )

    print(
        "state valid mask       :",
        result.valid_path,
    )

    print(
        "state code             :",
        result.state_code_path,
    )

    print(
        "report                 :",
        json_path,
    )

    print()

    print(
        "decision               :",
        decision,
    )

    print()

    print(
        "U3.3a FIRST MINISTACK FUSED EXECUTOR: PASS"
    )


if __name__ == "__main__":
    main()
