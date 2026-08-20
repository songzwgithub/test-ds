#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fullspan-min", type=float, default=0.80)

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        roi,
    ) = open_from_config(args.config)

    processing = Path(paths.output_dir) / "processing"
    seqdir = processing / "sequential"

    # --------------------------------------------------------
    # Current M16 stage TC maps
    # --------------------------------------------------------

    report = json.loads(
        (seqdir / "u33b_multistage_report.json").read_text()
    )

    if report["ministack_size"] != 16:
        raise RuntimeError(
            f"Current U3.3b is M={report['ministack_size']}, expected M16"
        )

    tc0 = np.load(
        seqdir / "u33b_stage0000_temporal_coherence.npy",
        mmap_mode="r",
    )

    tc1 = np.load(
        seqdir / "u33b_stage0001_temporal_coherence.npy",
        mmap_mode="r",
    )

    tc2 = np.load(
        seqdir / "u33b_stage0002_temporal_coherence.npy",
        mmap_mode="r",
    )

    # --------------------------------------------------------
    # U3.4e full-span sequential TC
    # --------------------------------------------------------

    ztc = np.load(
        seqdir / "u34e_fullspan_sequential_tc.npz"
    )

    rr = ztc["rows"]
    cc = ztc["cols"]

    full_tc = ztc["full_scm_tc"]
    seq_full_tc = ztc["sequential_fullspan_tc"]

    # --------------------------------------------------------
    # M16 phase parity metrics
    # --------------------------------------------------------

    zm = np.load(
        seqdir / "u34c_M16_phase_metrics.npz"
    )

    if not (
        np.array_equal(rr, zm["rows"])
        and
        np.array_equal(cc, zm["cols"])
    ):
        raise RuntimeError(
            "M16 metric point order mismatch"
        )

    med_err = zm["median_abs_error_deg"]
    p95_err = zm["p95_abs_error_deg"]
    similarity = zm["phase_similarity"]

    # --------------------------------------------------------
    # Sequential-only internal metrics
    # --------------------------------------------------------

    s0 = np.asarray(tc0[rr, cc], dtype=np.float32)
    s1 = np.asarray(tc1[rr, cc], dtype=np.float32)
    s2 = np.asarray(tc2[rr, cc], dtype=np.float32)

    stage_min = np.minimum(
        np.minimum(s0, s1),
        s2,
    )

    stage_max = np.maximum(
        np.maximum(s0, s1),
        s2,
    )

    stage_spread = stage_max - stage_min

    weighted = (
        16.0 * s0
        + 16.0 * s1
        + 6.0 * s2
    ) / 38.0

    ref_accept = (
        full_tc >= 0.80
    )

    base = (
        seq_full_tc >= args.fullspan_min
    )

    n = rr.size
    nref = int(ref_accept.sum())

    print("=" * 160)
    print("U3.4f M16 HYBRID SEQUENTIAL/FULL-SCM QUALITY ROUTER")
    print("=" * 160)

    print("sequential candidates      :", f"{n:,}")
    print("full-SCM accepted reference:", f"{nref:,}")
    print("fullspan sequential >=0.80 :", f"{base.sum():,}")
    print()

    print(
        "stageMin  directSeq   fallback    "
        "keepRef%   falseAccept  dangerous>60  dangerous>90  "
        "directP95Err50 directP95Err95"
    )

    print("-" * 160)

    rows = []

    for threshold in np.arange(
        0.80,
        0.951,
        0.01,
    ):

        direct = (
            base
            &
            (stage_min >= threshold)
        )

        fallback = ~direct

        direct_n = int(direct.sum())
        fallback_n = int(fallback.sum())

        kept_reference = int(
            np.count_nonzero(
                direct & ref_accept
            )
        )

        false_accept = (
            direct
            &
            ~ref_accept
        )

        dangerous60 = (
            false_accept
            &
            (p95_err > 60.0)
        )

        dangerous90 = (
            false_accept
            &
            (p95_err > 90.0)
        )

        # These are NOT lost:
        # fallback points will be rerun with full-SCM.
        keep_ref_frac = (
            kept_reference / nref
            if nref
            else np.nan
        )

        fa_n = int(
            false_accept.sum()
        )

        d60_n = int(
            dangerous60.sum()
        )

        d90_n = int(
            dangerous90.sum()
        )

        if direct_n:

            p50 = float(
                np.median(
                    p95_err[direct]
                )
            )

            p95 = float(
                np.percentile(
                    p95_err[direct],
                    95,
                )
            )

        else:

            p50 = np.nan
            p95 = np.nan

        item = {
            "stage_min_threshold":
                float(threshold),

            "direct_sequential":
                direct_n,

            "fallback":
                fallback_n,

            "direct_fraction":
                float(direct_n / n),

            "reference_accept_direct":
                kept_reference,

            "reference_accept_direct_fraction":
                float(keep_ref_frac),

            "false_accept":
                fa_n,

            "false_accept_fraction_of_direct":
                float(
                    fa_n / direct_n
                )
                if direct_n
                else None,

            "dangerous_false_accept_gt60":
                d60_n,

            "dangerous_false_accept_gt90":
                d90_n,

            "dangerous_gt60_fraction_of_direct":
                float(
                    d60_n / direct_n
                )
                if direct_n
                else None,

            "dangerous_gt90_fraction_of_direct":
                float(
                    d90_n / direct_n
                )
                if direct_n
                else None,

            "direct_p95_error_median_deg":
                p50,

            "direct_p95_error_p95_deg":
                p95,
        }

        rows.append(item)

        print(
            f"{threshold:8.2f} "
            f"{direct_n:10,d} "
            f"{fallback_n:10,d} "
            f"{100*keep_ref_frac:9.3f}% "
            f"{fa_n:11,d} "
            f"{d60_n:12,d} "
            f"{d90_n:12,d} "
            f"{p50:15.3f} "
            f"{p95:15.3f}"
        )

    # --------------------------------------------------------
    # Additional diagnostics:
    # do catastrophic disagreements show stage-TC inconsistency?
    # --------------------------------------------------------

    catastrophic = (
        base
        &
        ~ref_accept
        &
        (p95_err > 90.0)
    )

    normal = (
        base
        &
        ref_accept
        &
        (p95_err <= 30.0)
    )

    def summary(mask):

        if not np.any(mask):
            return {"n": 0}

        return {
            "n":
                int(mask.sum()),

            "seq_full_tc_median":
                float(
                    np.median(
                        seq_full_tc[mask]
                    )
                ),

            "stage_min_median":
                float(
                    np.median(
                        stage_min[mask]
                    )
                ),

            "stage_min_p05":
                float(
                    np.percentile(
                        stage_min[mask],
                        5,
                    )
                ),

            "stage_spread_median":
                float(
                    np.median(
                        stage_spread[mask]
                    )
                ),

            "stage_spread_p95":
                float(
                    np.percentile(
                        stage_spread[mask],
                        95,
                    )
                ),

            "weighted_tc_median":
                float(
                    np.median(
                        weighted[mask]
                    )
                ),

            "phase_similarity_median":
                float(
                    np.median(
                        similarity[mask]
                    )
                ),
        }

    diagnostics = {
        "catastrophic_false_accept":
            summary(catastrophic),

        "stable_reference_accept":
            summary(normal),
    }

    print()
    print("=" * 100)
    print("INTERNAL-CONSISTENCY DIAGNOSTICS")
    print("=" * 100)

    for name, x in diagnostics.items():

        print(name)
        for k, v in x.items():
            print(f"  {k:<28s}: {v}")

    out = (
        seqdir
        / "u34f_hybrid_quality_router.json"
    )

    out.write_text(
        json.dumps(
            {
                "format":
                    "pyPSDS-GAMMA-U3.4f-hybrid-router-v1",

                "ministack_size":
                    16,

                "fullspan_tc_min":
                    args.fullspan_min,

                "threshold_scan":
                    rows,

                "diagnostics":
                    diagnostics,

                "decision":
                    "pending_router_frontier",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print("report:", out)
    print()
    print("U3.4f HYBRID ROUTER AUDIT: PASS")


if __name__ == "__main__":
    main()
