#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    p = np.percentile(
        x,
        [0, 1, 5, 25, 50, 75, 95, 99, 100],
    )

    names = (
        "min",
        "p01",
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
        for k, v in zip(names, p)
    }


def fraction_ge(x, threshold):
    x = np.asarray(x)
    good = np.isfinite(x)

    if not np.any(good):
        return np.nan

    return float(
        np.mean(
            x[good] >= threshold
        )
    )


def fraction_le(x, threshold):
    x = np.asarray(x)
    good = np.isfinite(x)

    if not np.any(good):
        return np.nan

    return float(
        np.mean(
            x[good] <= threshold
        )
    )


def group_summary(
    mask,
    similarity,
    median_err,
    p95_err,
    max_err,
):
    mask = np.asarray(
        mask,
        dtype=np.bool_,
    )

    n = int(
        np.count_nonzero(mask)
    )

    if n == 0:
        return {
            "n": 0,
        }

    s = similarity[mask]
    m = median_err[mask]
    p = p95_err[mask]
    x = max_err[mask]

    return {
        "n": n,

        "similarity":
            quantiles(s),

        "median_error_deg":
            quantiles(m),

        "p95_error_deg":
            quantiles(p),

        "max_error_deg":
            quantiles(x),

        "similarity_fractions": {
            "ge_0p99":
                fraction_ge(s, 0.99),

            "ge_0p995":
                fraction_ge(s, 0.995),

            "ge_0p999":
                fraction_ge(s, 0.999),
        },

        "median_error_fractions": {
            "le_5deg":
                fraction_le(m, 5.0),

            "le_10deg":
                fraction_le(m, 10.0),

            "le_15deg":
                fraction_le(m, 15.0),

            "le_20deg":
                fraction_le(m, 20.0),

            "gt_30deg":
                1.0
                -
                fraction_le(m, 30.0),

            "gt_60deg":
                1.0
                -
                fraction_le(m, 60.0),

            "gt_90deg":
                1.0
                -
                fraction_le(m, 90.0),
        },

        "p95_error_fractions": {
            "le_10deg":
                fraction_le(p, 10.0),

            "le_20deg":
                fraction_le(p, 20.0),

            "le_30deg":
                fraction_le(p, 30.0),

            "le_45deg":
                fraction_le(p, 45.0),

            "gt_60deg":
                1.0
                -
                fraction_le(p, 60.0),

            "gt_90deg":
                1.0
                -
                fraction_le(p, 90.0),
        },
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--tc-min",
        type=float,
        default=0.80,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (row0, col0, H, W),
    ) = open_from_config(
        args.config
    )

    processing = (
        Path(paths.output_dir)
        /
        "processing"
    )

    seqdir = (
        processing
        /
        "sequential"
    )

    metrics_path = (
        seqdir
        /
        "u34a_phase_parity_metrics.npz"
    )

    kmap_path = (
        seqdir
        /
        "compression_state_core_K24_effective_shp_count.npy"
    )

    pl_valid_path = (
        processing
        /
        "pl_valid.npy"
    )

    prior_path = (
        processing
        /
        "center_prior.npy"
    )

    ps_path = (
        processing
        /
        "ps_mask.npy"
    )

    tc_path = (
        processing
        /
        "temporal_coherence.npy"
    )

    required = (
        metrics_path,
        kmap_path,
        pl_valid_path,
        prior_path,
        ps_path,
        tc_path,
    )

    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)

    z = np.load(
        metrics_path
    )

    rows = z[
        "rows"
    ]

    cols = z[
        "cols"
    ]

    K = z[
        "effective_K"
    ]

    tc_points = z[
        "full_temporal_coherence"
    ]

    similarity = z[
        "phase_similarity"
    ]

    median_err = z[
        "median_abs_error_deg"
    ]

    p95_err = z[
        "p95_abs_error_deg"
    ]

    max_err = z[
        "max_abs_error_deg"
    ]

    # --------------------------------------------------------
    # Verify point-domain routing.
    # --------------------------------------------------------

    if np.any(
        K < 48
    ):
        raise RuntimeError(
            "U3.4a parity points unexpectedly contain K<48"
        )

    sequential_all = np.ones(
        rows.size,
        dtype=np.bool_,
    )

    production = (
        tc_points
        >=
        args.tc_min
    )

    tc_08_09 = (
        production
        &
        (tc_points < 0.90)
    )

    tc_09_095 = (
        tc_points >= 0.90
    ) & (
        tc_points < 0.95
    )

    tc_095_plus = (
        tc_points >= 0.95
    )

    # --------------------------------------------------------
    # Whole-scene routing counts.
    # --------------------------------------------------------

    kmap = np.load(
        kmap_path,
        mmap_mode="r",
    )

    pl_valid = np.load(
        pl_valid_path,
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    prior = np.load(
        prior_path,
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    ps = np.load(
        ps_path,
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    full_tc = np.load(
        tc_path,
        mmap_mode="r",
    )

    formal_ds = (
        pl_valid
        &
        prior
        &
        ~ps
    )

    accepted_ds = (
        formal_ds
        &
        np.isfinite(
            full_tc
        )
        &
        (
            full_tc
            >=
            args.tc_min
        )
    )

    sequential_route = (
        formal_ds
        &
        (
            kmap >= 48
        )
    )

    fallback_route = (
        formal_ds
        &
        ~sequential_route
    )

    accepted_seq = (
        accepted_ds
        &
        sequential_route
    )

    accepted_fallback = (
        accepted_ds
        &
        fallback_route
    )

    formal_n = int(
        np.count_nonzero(
            formal_ds
        )
    )

    accepted_n = int(
        np.count_nonzero(
            accepted_ds
        )
    )

    accepted_seq_n = int(
        np.count_nonzero(
            accepted_seq
        )
    )

    accepted_fallback_n = int(
        np.count_nonzero(
            accepted_fallback
        )
    )

    # Point-domain parity population must match
    # the scene routing for sequential accepted DS.
    production_n = int(
        np.count_nonzero(
            production
        )
    )

    if (
        production_n
        !=
        accepted_seq_n
    ):
        raise RuntimeError(
            "production parity population mismatch: "
            f"metrics={production_n:,}, "
            f"scene={accepted_seq_n:,}"
        )

    groups = {
        "all_sequential_Kge48":
            sequential_all,

        "production_TCge0p80":
            production,

        "TC0p80_0p90":
            tc_08_09,

        "TC0p90_0p95":
            tc_09_095,

        "TCge0p95":
            tc_095_plus,
    }

    summaries = {
        name:
            group_summary(
                mask,
                similarity,
                median_err,
                p95_err,
                max_err,
            )

        for name, mask
        in groups.items()
    }

    report = {
        "format":
            "pyPSDS-GAMMA-U3.4b-production-parity-gate-v1",

        "tc_min":
            args.tc_min,

        "formal_DS":
            formal_n,

        "accepted_DS":
            accepted_n,

        "accepted_sequential":
            accepted_seq_n,

        "accepted_full_scm_fallback":
            accepted_fallback_n,

        "accepted_sequential_fraction":
            (
                accepted_seq_n
                /
                accepted_n
                if accepted_n
                else 0.0
            ),

        "groups":
            summaries,

        "scientific_decision":
            "pending_ministack_size_sensitivity",
    }

    json_path = (
        seqdir
        /
        "u34b_production_parity_gate.json"
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

    print("=" * 118)

    print(
        "U3.4b PRODUCTION DS PARITY GATE"
    )

    print("=" * 118)

    print(
        "formal DS                  :",
        f"{formal_n:,}",
    )

    print(
        f"accepted DS TC>={args.tc_min:.2f}      :",
        f"{accepted_n:,}",
    )

    print(
        "accepted sequential        :",
        f"{accepted_seq_n:,}",
    )

    print(
        "accepted full-SCM fallback :",
        f"{accepted_fallback_n:,}",
    )

    if accepted_n:

        print(
            "sequential coverage        :",
            f"{100*accepted_seq_n/accepted_n:.6f}%",
        )

    print()

    print(
        "group              n        "
        "sim50      medErr50   medErr95   "
        "p95Err50   p95Err95"
    )

    print("-" * 118)

    for name in (
        "all_sequential_Kge48",
        "production_TCge0p80",
        "TC0p80_0p90",
        "TC0p90_0p95",
        "TCge0p95",
    ):

        x = summaries[
            name
        ]

        if not x.get(
            "n",
            0
        ):
            continue

        print(
            f"{name:<20s} "
            f"{x['n']:8,d} "
            f"{x['similarity']['median']:9.4f} "
            f"{x['median_error_deg']['median']:10.3f} "
            f"{x['median_error_deg']['p95']:10.3f} "
            f"{x['p95_error_deg']['median']:10.3f} "
            f"{x['p95_error_deg']['p95']:10.3f}"
        )

    print()

    x = summaries[
        "production_TCge0p80"
    ]

    print(
        "Production TC>=0.80 fractions"
    )

    print("-" * 118)

    for k, v in x[
        "similarity_fractions"
    ].items():

        print(
            f"{k:<24s}: "
            f"{100*v:8.3f}%"
        )

    for k, v in x[
        "median_error_fractions"
    ].items():

        print(
            f"median_{k:<17s}: "
            f"{100*v:8.3f}%"
        )

    for k, v in x[
        "p95_error_fractions"
    ].items():

        print(
            f"p95_{k:<20s}: "
            f"{100*v:8.3f}%"
        )

    print()

    print(
        "report:",
        json_path,
    )

    print()

    print(
        "U3.4b PRODUCTION PARITY STATISTICS: PASS"
    )


if __name__ == "__main__":
    main()
