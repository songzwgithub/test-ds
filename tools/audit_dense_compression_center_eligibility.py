#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def qdict(x):
    x = np.asarray(x)

    if x.size == 0:
        return {}

    q = np.percentile(
        x,
        [0, 5, 25, 50, 75, 95, 100],
    )

    names = (
        "min",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
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

    ap.add_argument(
        "--min-shp",
        type=int,
        default=48,
    )

    args = ap.parse_args()

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

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    required_path = (
        seqdir
        /
        "compression_required_mask.npy"
    )

    missing_path = (
        seqdir
        /
        "compression_phase_missing_mask.npy"
    )

    scale_path = (
        stats
        /
        "rayleigh_scale2.npy"
    )

    raw_valid_path = (
        stats
        /
        "raw_valid.npy"
    )

    geom_path = (
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy"
    )

    ps_path = (
        processing
        /
        "ps_mask.npy"
    )

    prior_path = (
        processing
        /
        "center_prior.npy"
    )

    pl_path = (
        processing
        /
        "pl_valid.npy"
    )

    linked_path = (
        processing
        /
        "linked_phase.npy"
    )

    for p in (
        required_path,
        missing_path,
        scale_path,
        raw_valid_path,
        geom_path,
        ps_path,
        prior_path,
        pl_path,
        linked_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    required = np.load(
        required_path,
        mmap_mode="r",
    )

    missing = np.load(
        missing_path,
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

    geom = np.load(
        geom_path,
        mmap_mode="r",
    )

    ps = np.load(
        ps_path,
        mmap_mode="r",
    )

    prior = np.load(
        prior_path,
        mmap_mode="r",
    )

    pl_valid = np.load(
        pl_path,
        mmap_mode="r",
    )

    linked = np.load(
        linked_path,
        mmap_mode="r",
    )

    H, W = required.shape

    for name, arr in (
        ("missing", missing),
        ("scale2", scale2),
        ("raw_valid", raw_valid),
        ("geometry_valid", geom),
        ("ps", ps),
        ("center_prior", prior),
        ("pl_valid", pl_valid),
    ):
        if arr.shape != (H, W):
            raise RuntimeError(
                f"{name} shape={arr.shape}, "
                f"expected={(H, W)}"
            )

    # Current production layout is T,H,W.
    if (
        linked.ndim == 3
        and
        linked.shape[1:] == (H, W)
    ):
        ndate = int(
            linked.shape[0]
        )

    elif (
        linked.ndim == 3
        and
        linked.shape[:2] == (H, W)
    ):
        ndate = int(
            linked.shape[2]
        )

    else:
        raise RuntimeError(
            f"Unexpected linked phase shape: "
            f"{linked.shape}"
        )

    # --------------------------------------------------------
    # Frozen validity definition
    # --------------------------------------------------------

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

    req = np.asarray(
        required,
        dtype=np.bool_,
    )

    miss = np.asarray(
        missing,
        dtype=np.bool_,
    )

    prior_bool = np.asarray(
        prior,
        dtype=np.bool_,
    )

    pl_bool = np.asarray(
        pl_valid,
        dtype=np.bool_,
    )

    # Sanity checks.
    if np.any(
        req
        &
        ~valid
    ):
        raise RuntimeError(
            "compression_required contains invalid pixels"
        )

    if np.any(
        req
        &
        ps_bool
    ):
        raise RuntimeError(
            "compression_required unexpectedly contains PS"
        )

    if np.any(
        miss
        &
        ~req
    ):
        raise RuntimeError(
            "missing mask is not a subset of required mask"
        )

    # --------------------------------------------------------
    # Every required pixel becomes a test center.
    # --------------------------------------------------------

    rr, cc = np.where(
        req
    )

    rr = rr.astype(
        np.int32,
        copy=False,
    )

    cc = cc.astype(
        np.int32,
        copy=False,
    )

    n_required = int(
        rr.size
    )

    # Store K only as a 2-D audit product.
    K_map = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    eligible = np.zeros(
        (H, W),
        dtype=np.bool_,
    )

    ctx = (
        prepare_glrt_window_context(
            scale2,
            valid,
            ps_bool,
            half_row=args.half_row,
            half_col=args.half_col,
        )
    )

    print("=" * 88)
    print(
        "U3.2c1 dense compression-center eligibility audit"
    )
    print("=" * 88)

    print(
        "scene                  :",
        f"{H} x {W}",
    )

    print(
        "dates                  :",
        ndate,
    )

    print(
        "required centers       :",
        f"{n_required:,}",
    )

    print(
        "GLRT window            :",
        f"{2*args.half_row+1} x "
        f"{2*args.half_col+1}",
    )

    print(
        "alpha                   :",
        args.alpha,
    )

    print(
        "min SHP                 :",
        args.min_shp,
    )

    print()

    t0 = time.perf_counter()

    for start in range(
        0,
        n_required,
        args.batch,
    ):

        stop = min(
            n_required,
            start + args.batch,
        )

        br = rr[
            start:stop
        ]

        bc = cc[
            start:stop
        ]

        # We need exact K.
        # support is released after each batch.
        support, K = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=args.alpha,
                nslc=ndate,
                block_size=args.support_block,
            )
        )

        del support

        K_map[
            br,
            bc,
        ] = K

        good = (
            K
            >=
            args.min_shp
        )

        eligible[
            br,
            bc,
        ] = good

        if (
            stop == n_required
            or
            stop % (
                args.batch
                *
                10
            ) == 0
        ):
            elapsed = (
                time.perf_counter()
                -
                t0
            )

            rate = (
                stop / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"centers "
                f"{stop:,}/"
                f"{n_required:,} "
                f"({100*stop/n_required:6.2f}%) "
                f"rate={rate:,.0f} center/s"
            )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    req_eligible = (
        req
        &
        eligible
    )

    miss_eligible = (
        miss
        &
        eligible
    )

    miss_ineligible = (
        miss
        &
        ~eligible
    )

    # Missing because the output-center prior never selected it.
    missing_not_prior = (
        miss
        &
        ~prior_bool
    )

    missing_prior = (
        miss
        &
        prior_bool
    )

    missing_not_prior_eligible = (
        missing_not_prior
        &
        eligible
    )

    # Useful consistency QA.
    current_pl_ineligible = (
        req
        &
        pl_bool
        &
        ~eligible
    )

    n_missing = int(
        np.count_nonzero(
            miss
        )
    )

    n_req_eligible = int(
        np.count_nonzero(
            req_eligible
        )
    )

    n_miss_eligible = int(
        np.count_nonzero(
            miss_eligible
        )
    )

    n_miss_ineligible = int(
        np.count_nonzero(
            miss_ineligible
        )
    )

    n_missing_not_prior = int(
        np.count_nonzero(
            missing_not_prior
        )
    )

    n_missing_prior = int(
        np.count_nonzero(
            missing_prior
        )
    )

    n_missing_not_prior_eligible = int(
        np.count_nonzero(
            missing_not_prior_eligible
        )
    )

    n_current_pl_ineligible = int(
        np.count_nonzero(
            current_pl_ineligible
        )
    )

    kval_req = (
        K_map[
            req
        ]
    )

    kval_miss = (
        K_map[
            miss
        ]
    )

    # Additional threshold diagnostics.
    thresholds = (
        16,
        24,
        32,
        40,
        48,
    )

    missing_threshold_counts = {}

    for threshold in thresholds:

        n = int(
            np.count_nonzero(
                miss
                &
                (
                    K_map
                    >=
                    threshold
                )
            )
        )

        missing_threshold_counts[
            str(threshold)
        ] = {
            "count": n,
            "fraction": (
                n / n_missing
                if n_missing
                else 0.0
            ),
        }

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if n_miss_ineligible == 0:

        decision = (
            "direct_dense_pl_is_sufficient"
        )

        recommendation = (
            "Every currently missing compression-state pixel "
            "is independently eligible for the frozen GLRT "
            "phase-linking rule. U3.2c2 can implement the "
            "fused dense PL -> immediate compression path "
            "without a separate fallback phase policy."
        )

    else:

        decision = (
            "compression_phase_fallback_policy_required"
        )

        recommendation = (
            "At least one compression-required pixel with no "
            "current phase source is not independently eligible "
            "for K>=min_shp. Do not simply run EMI on every "
            "required pixel. U3.2c2 must define and validate a "
            "fallback/state-validity policy before the fused "
            "sequential executor is enabled."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    k_path = (
        seqdir
        /
        "compression_center_shp_count.npy"
    )

    eligible_path = (
        seqdir
        /
        "compression_center_eligible_mask.npy"
    )

    ineligible_path = (
        seqdir
        /
        "compression_missing_ineligible_mask.npy"
    )

    np.save(
        k_path,
        K_map,
    )

    np.save(
        eligible_path,
        eligible,
    )

    np.save(
        ineligible_path,
        miss_ineligible,
    )

    report = {
        "format":
            "pyPSDS-GAMMA-dense-compression-center-eligibility-v1",

        "shape":
            [H, W],

        "ndate":
            ndate,

        "half_row":
            args.half_row,

        "half_col":
            args.half_col,

        "alpha":
            args.alpha,

        "min_shp":
            args.min_shp,

        "compression_required":
            n_required,

        "compression_missing":
            n_missing,

        "required_direct_eligible":
            n_req_eligible,

        "required_direct_eligible_fraction":
            (
                n_req_eligible
                /
                n_required
                if n_required
                else 0.0
            ),

        "missing_direct_eligible":
            n_miss_eligible,

        "missing_direct_eligible_fraction":
            (
                n_miss_eligible
                /
                n_missing
                if n_missing
                else 0.0
            ),

        "missing_direct_ineligible":
            n_miss_ineligible,

        "missing_direct_ineligible_fraction":
            (
                n_miss_ineligible
                /
                n_missing
                if n_missing
                else 0.0
            ),

        "missing_not_center_prior":
            n_missing_not_prior,

        "missing_center_prior":
            n_missing_prior,

        "missing_not_prior_but_direct_eligible":
            n_missing_not_prior_eligible,

        "current_pl_but_direct_ineligible":
            n_current_pl_ineligible,

        "required_K_quantiles":
            qdict(
                kval_req
            ),

        "missing_K_quantiles":
            qdict(
                kval_miss
            ),

        "missing_threshold_counts":
            missing_threshold_counts,

        "elapsed_seconds":
            elapsed,

        "decision":
            decision,

        "recommendation":
            recommendation,
    }

    json_path = (
        seqdir
        /
        "compression_center_eligibility.json"
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

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print(
        "U3.2c1 result"
    )
    print("=" * 88)

    print(
        "compression required        :",
        f"{n_required:,}",
    )

    print(
        "missing current phase       :",
        f"{n_missing:,}",
    )

    print()

    print(
        "required K>=48              :",
        f"{n_req_eligible:,}",
        f"({pct(n_req_eligible, n_required):.3f}%)",
    )

    print(
        "missing K>=48               :",
        f"{n_miss_eligible:,}",
        f"({pct(n_miss_eligible, n_missing):.3f}%)",
    )

    print(
        "missing K<48                :",
        f"{n_miss_ineligible:,}",
        f"({pct(n_miss_ineligible, n_missing):.3f}%)",
    )

    print()

    print(
        "missing not center_prior    :",
        f"{n_missing_not_prior:,}",
        f"({pct(n_missing_not_prior, n_missing):.3f}%)",
    )

    print(
        "missing but center_prior    :",
        f"{n_missing_prior:,}",
        f"({pct(n_missing_prior, n_missing):.3f}%)",
    )

    print(
        "not-prior yet K>=48         :",
        f"{n_missing_not_prior_eligible:,}",
        f"({pct(n_missing_not_prior_eligible, n_missing_not_prior):.3f}%)",
    )

    print(
        "current PL but K<48         :",
        f"{n_current_pl_ineligible:,}",
    )

    print()

    print(
        "required K quantiles        :",
        qdict(kval_req),
    )

    print(
        "missing K quantiles         :",
        qdict(kval_miss),
    )

    print()

    for threshold in thresholds:

        item = (
            missing_threshold_counts[
                str(threshold)
            ]
        )

        print(
            f"missing K>={threshold:<2d}               :",
            f"{item['count']:,}",
            f"({100*item['fraction']:.3f}%)",
        )

    print()

    print(
        "elapsed                     :",
        f"{elapsed:.3f} s",
    )

    print()

    print(
        "K map       :",
        k_path,
    )

    print(
        "eligible mask:",
        eligible_path,
    )

    print(
        "fallback mask:",
        ineligible_path,
    )

    print(
        "json        :",
        json_path,
    )

    print()

    print(
        "decision    :",
        decision,
    )

    print()
    print(
        recommendation
    )

    print()
    print(
        "U3.2c1 DENSE CENTER ELIGIBILITY AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
