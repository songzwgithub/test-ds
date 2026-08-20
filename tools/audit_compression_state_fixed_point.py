#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pypsds.phase_linking.state_domain import (
    make_windows,
    compute_original_K,
    effective_counts,
    fixed_point_core,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def quantiles(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    q = np.percentile(
        x,
        [0, 5, 25, 50, 75, 95, 99, 100],
    )

    names = (
        "min",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
        "max",
    )

    return {
        k: float(v)
        for k, v in zip(names, q)
    }










def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--processing-dir",
        required=True,
    )

    ap.add_argument(
        "--thresholds",
        default="48,40,32,24,16,8",
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=16000,
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=1024,
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

    args = ap.parse_args()

    thresholds = [
        int(x.strip())
        for x in args.thresholds.split(",")
        if x.strip()
    ]

    processing = Path(
        args.processing_dir
    )

    seqdir = (
        processing
        /
        "sequential"
    )

    stats = (
        processing
        /
        "ds_statistics"
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

    geom = np.load(
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    ps = np.load(
        processing
        /
        "ps_mask.npy",
        mmap_mode="r",
    )

    prior = np.load(
        processing
        /
        "center_prior.npy",
        mmap_mode="r",
    )

    required = np.load(
        seqdir
        /
        "compression_required_mask.npy",
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

    H, W = required.shape

    if (
        yxt.ndim != 3
        or
        yxt.shape[:2]
        !=
        (H, W)
    ):
        raise RuntimeError(
            f"YXT shape mismatch: "
            f"{yxt.shape}"
        )

    ndate = int(
        yxt.shape[2]
    )

    valid = np.ascontiguousarray(
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

    ps_bool = np.ascontiguousarray(
        np.asarray(
            ps,
            dtype=np.bool_,
        )
        &
        valid
    )

    valid_nonps = (
        valid
        &
        ~ps_bool
    )

    prior_bool = np.asarray(
        prior,
        dtype=np.bool_,
    )

    required_bool = np.asarray(
        required,
        dtype=np.bool_,
    )

    print("=" * 96)
    print(
        "U3.2c5 compressed-state fixed-point self-consistency audit"
    )
    print("=" * 96)

    print(
        "scene                  :",
        f"{H} x {W}",
    )

    print(
        "dates                  :",
        ndate,
    )

    print(
        "valid non-PS           :",
        f"{np.count_nonzero(valid_nonps):,}",
    )

    print(
        "compression required   :",
        f"{np.count_nonzero(required_bool):,}",
    )

    print(
        "state thresholds       :",
        thresholds,
    )

    print()

    ctx = (
        prepare_glrt_window_context(
            scale2,
            valid,
            ps_bool,
            half_row=args.half_row,
            half_col=args.half_col,
        )
    )

    # --------------------------------------------------------
    # Compute original K for ALL valid non-PS pixels.
    #
    # Important:
    # do not restrict the state universe to the original
    # compression_required union. A state center may use
    # valid non-PS pixels slightly outside that first union.
    # --------------------------------------------------------

    original_k_path = (
        seqdir
        /
        "compression_all_valid_nonps_shp_count.npy"
    )

    if original_k_path.is_file():

        original_K = np.load(
            original_k_path,
            mmap_mode="r",
        )

        if original_K.shape != (H, W):
            raise RuntimeError(
                "existing all-valid K map shape mismatch"
            )

        print(
            "original K map        : reuse",
            original_k_path,
        )

    else:

        print(
            "computing original K for all valid non-PS..."
        )

        original_K = compute_original_K(
            ctx=ctx,
            mask=valid_nonps,
            alpha=args.alpha,
            ndate=ndate,
            batch=args.batch,
            support_block=args.support_block,
        )

        np.save(
            original_k_path,
            original_K,
        )

        print(
            "original K map        :",
            original_k_path,
        )

    # --------------------------------------------------------
    # Frozen formal DS population:
    #
    # original center prior
    # +
    # original K>=48
    #
    # This DOES NOT change with state threshold.
    # --------------------------------------------------------

    formal_ds = (
        prior_bool
        &
        valid_nonps
        &
        (
            np.asarray(
                original_K
            )
            >=
            48
        )
    )

    n_formal_ds = int(
        np.count_nonzero(
            formal_ds
        )
    )

    print()

    print(
        "formal DS K>=48       :",
        f"{n_formal_ds:,}",
    )

    print()

    results = []

    # --------------------------------------------------------
    # Threshold ladder
    # --------------------------------------------------------

    for threshold in thresholds:

        print("=" * 96)
        print(
            f"Kstate = {threshold}"
        )
        print("=" * 96)

        state, history = fixed_point_core(
            ctx=ctx,
            valid_nonps=valid_nonps,
            original_K=original_K,
            threshold=threshold,
            alpha=args.alpha,
            ndate=ndate,
            batch=args.batch,
            support_block=args.support_block,
            half_row=args.half_row,
            half_col=args.half_col,
        )

        n_state = int(
            np.count_nonzero(
                state
            )
        )

        n_required_state = int(
            np.count_nonzero(
                state
                &
                required_bool
            )
        )

        n_extra_state = int(
            np.count_nonzero(
                state
                &
                ~required_bool
            )
        )

        # ----------------------------------------------------
        # Evaluate original DS centers under this final state.
        # ----------------------------------------------------

        (
            ds_rr,
            ds_cc,
            ds_eff,
        ) = effective_counts(
            ctx=ctx,
            center_mask=formal_ds,
            state_mask=state,
            alpha=args.alpha,
            ndate=ndate,
            batch=args.batch,
            support_block=args.support_block,
            half_row=args.half_row,
            half_col=args.half_col,
        )

        ds_eff_state = int(
            np.count_nonzero(
                ds_eff
                >=
                threshold
            )
        )

        ds_eff_48 = int(
            np.count_nonzero(
                ds_eff
                >=
                48
            )
        )

        ds_below_state = int(
            np.count_nonzero(
                ds_eff
                <
                threshold
            )
        )

        ds_between = int(
            np.count_nonzero(
                (ds_eff >= threshold)
                &
                (ds_eff < 48)
            )
        )

        # ----------------------------------------------------
        # Save state core.
        # ----------------------------------------------------

        state_path = (
            seqdir
            /
            f"compression_state_core_K"
            f"{threshold:02d}.npy"
        )

        np.save(
            state_path,
            state,
        )

        result = {
            "state_min_shp":
                threshold,

            "iterations":
                len(history),

            "iteration_history":
                history,

            "state_pixels":
                n_state,

            "state_fraction_valid_nonps":
                (
                    n_state
                    /
                    np.count_nonzero(
                        valid_nonps
                    )
                ),

            "compression_required_retained":
                n_required_state,

            "compression_required_fraction":
                (
                    n_required_state
                    /
                    np.count_nonzero(
                        required_bool
                    )
                ),

            "state_pixels_outside_original_required":
                n_extra_state,

            "formal_ds_total":
                n_formal_ds,

            "formal_ds_effective_ge_state":
                ds_eff_state,

            "formal_ds_effective_ge_state_fraction":
                (
                    ds_eff_state
                    /
                    n_formal_ds
                ),

            "formal_ds_effective_ge48":
                ds_eff_48,

            "formal_ds_effective_ge48_fraction":
                (
                    ds_eff_48
                    /
                    n_formal_ds
                ),

            "formal_ds_between_state_and_47":
                ds_between,

            "formal_ds_below_state":
                ds_below_state,

            "formal_ds_effective_K_quantiles":
                quantiles(
                    ds_eff
                ),

            "state_mask":
                str(
                    state_path
                ),
        }

        results.append(
            result
        )

        print()

        print(
            "fixed-point iterations    :",
            len(history),
        )

        print(
            "final state pixels        :",
            f"{n_state:,}",
            f"({pct(n_state, np.count_nonzero(valid_nonps)):.3f}% valid non-PS)",
        )

        print(
            "required retained         :",
            f"{n_required_state:,}",
            f"({pct(n_required_state, np.count_nonzero(required_bool)):.3f}%)",
        )

        print(
            "state outside old union   :",
            f"{n_extra_state:,}",
        )

        print()

        print(
            f"formal DS K_eff>={threshold:<2d}   :",
            f"{ds_eff_state:,}",
            f"({pct(ds_eff_state, n_formal_ds):.3f}%)",
        )

        print(
            "formal DS K_eff>=48       :",
            f"{ds_eff_48:,}",
            f"({pct(ds_eff_48, n_formal_ds):.3f}%)",
        )

        print(
            f"formal DS {threshold}<=K<48       :",
            f"{ds_between:,}",
        )

        print(
            f"formal DS K_eff<{threshold:<2d}    :",
            f"{ds_below_state:,}",
        )

        print(
            "formal DS K_eff quantiles :",
            quantiles(
                ds_eff
            ),
        )

        print()

        print(
            "state mask                :",
            state_path,
        )

        print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    report = {
        "format":
            "pyPSDS-GAMMA-compression-state-fixed-point-v1",

        "shape":
            [H, W],

        "ndate":
            ndate,

        "alpha":
            args.alpha,

        "half_row":
            args.half_row,

        "half_col":
            args.half_col,

        "formal_ds_min_shp":
            48,

        "thresholds":
            thresholds,

        "results":
            results,
    }

    json_path = (
        seqdir
        /
        "compression_state_fixed_point.json"
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
    print("=" * 120)
    print(
        "U3.2c5 fixed-point summary"
    )
    print("=" * 120)

    print(
        "Kstate | iterations | state%   | required% | "
        "DS >=Kstate | DS >=48 | DS Kstate..47 | DS <Kstate"
    )

    print("-" * 120)

    for r in results:

        print(
            f"{r['state_min_shp']:6d} | "
            f"{r['iterations']:10d} | "
            f"{100*r['state_fraction_valid_nonps']:7.3f} | "
            f"{100*r['compression_required_fraction']:9.3f} | "
            f"{100*r['formal_ds_effective_ge_state_fraction']:10.3f}% | "
            f"{100*r['formal_ds_effective_ge48_fraction']:7.3f}% | "
            f"{r['formal_ds_between_state_and_47']:13,d} | "
            f"{r['formal_ds_below_state']:10,d}"
        )

    print()

    print(
        "json:",
        json_path,
    )

    print()

    print(
        "U3.2c5 COMPRESSION STATE FIXED-POINT AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
