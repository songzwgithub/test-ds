#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        _,
    ) = open_from_config(
        args.config
    )

    root = (
        Path(paths.output_dir)
        / "v09"
    )

    indir = (
        root
        / "ifg_visual_qa_v2"
    )

    csv_path = (
        indir
        / "visual_qa_v2_index.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            csv_path
        )

    rows = []

    with csv_path.open() as f:

        for r in csv.DictReader(f):

            branch_edges = int(
                r["branch_edges"]
            )

            branch_match = int(
                r[
                    "branch_with_raw_wrap_jump"
                ]
            )

            mismatch = (
                branch_edges
                -
                branch_match
            )

            if branch_edges:

                match_fraction = (
                    branch_match
                    /
                    branch_edges
                )

            else:

                match_fraction = 1.0

            rows.append({
                "pair_id":
                    int(
                        r["pair_id"]
                    ),

                "date1":
                    r["date1"],

                "date2":
                    r["date2"],

                "global_gauge":
                    int(
                        r["global_gauge"]
                    ),

                "branch_edges":
                    branch_edges,

                "branch_abs1":
                    int(
                        r["branch_abs1"]
                    ),

                "branch_abs2plus":
                    int(
                        r["branch_abs2plus"]
                    ),

                "branch_with_raw_wrap_jump":
                    branch_match,

                "branch_mismatch":
                    mismatch,

                "raw_wrap_match_fraction":
                    float(
                        match_fraction
                    ),

                "raw_wrap_match_percent":
                    float(
                        100.0
                        *
                        match_fraction
                    ),

                "safe_internal_bad":
                    int(
                        r["safe_internal_bad"]
                    ),

                "selected_non_exact":
                    int(
                        r["selected_non_exact"]
                    ),

                "rejected_cycle_outliers":
                    int(
                        r[
                            "rejected_cycle_outliers"
                        ]
                    ),

                "wrap_parity_max_rad":
                    float(
                        r["wrap_parity_max_rad"]
                    ),
            })

    if not rows:

        raise RuntimeError(
            "No IFGs in visual QA table."
        )

    n = len(rows)

    match = np.array(
        [
            r[
                "raw_wrap_match_fraction"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    mismatch = np.array(
        [
            r[
                "branch_mismatch"
            ]
            for r in rows
        ],
        dtype=np.int64,
    )

    branch = np.array(
        [
            r[
                "branch_edges"
            ]
            for r in rows
        ],
        dtype=np.int64,
    )

    abs2 = np.array(
        [
            r[
                "branch_abs2plus"
            ]
            for r in rows
        ],
        dtype=np.int64,
    )

    safe_bad = np.array(
        [
            r[
                "safe_internal_bad"
            ]
            for r in rows
        ],
        dtype=np.int64,
    )

    parity = np.array(
        [
            r[
                "wrap_parity_max_rad"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Weighted global branch-match fraction
    # --------------------------------------------------------

    total_branch = int(
        branch.sum()
    )

    total_mismatch = int(
        mismatch.sum()
    )

    if total_branch:

        global_match = (
            1.0
            -
            total_mismatch
            /
            total_branch
        )

    else:

        global_match = 1.0

    # --------------------------------------------------------
    # Sort by weakest raw-wrap agreement
    # --------------------------------------------------------

    weakest = sorted(
        rows,
        key=lambda r: (
            r[
                "raw_wrap_match_fraction"
            ],
            -r[
                "branch_mismatch"
            ],
        )
    )

    print("=" * 108)
    print(
        "Step 08y3 - Full-network visual branch QA summary"
    )
    print("=" * 108)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"IFGs audited               : "
        f"{n}"
    )

    print()
    print(
        "Spatial integer branch structure"
    )

    print(
        f"  total branch edges       : "
        f"{total_branch:,}"
    )

    print(
        f"  |dN| >= 2 edges         : "
        f"{abs2.sum():,}"
    )

    print(
        f"  IFGs with |dN| >= 2     : "
        f"{np.count_nonzero(abs2):d}/{n}"
    )

    print(
        f"  branch mismatch edges    : "
        f"{total_mismatch:,}"
    )

    print(
        f"  weighted global match    : "
        f"{100*global_match:.6f}%"
    )

    print()
    print(
        "Per-IFG raw-wrap branch match"
    )

    print(
        f"  min                      : "
        f"{100*match.min():.6f}%"
    )

    print(
        f"  p05                      : "
        f"{100*np.percentile(match,5):.6f}%"
    )

    print(
        f"  median                   : "
        f"{100*np.median(match):.6f}%"
    )

    print(
        f"  p95                      : "
        f"{100*np.percentile(match,95):.6f}%"
    )

    print(
        f"  max                      : "
        f"{100*match.max():.6f}%"
    )

    print()

    for threshold in (
        0.90,
        0.95,
        0.98,
        0.99,
        0.995,
        0.999,
    ):

        count = int(
            np.count_nonzero(
                match
                <
                threshold
            )
        )

        print(
            f"  IFGs below "
            f"{100*threshold:6.2f}%       : "
            f"{count}"
        )

    print()
    print(
        "Other QA"
    )

    print(
        f"  IFGs with SAFE bad       : "
        f"{np.count_nonzero(safe_bad):d}/{n}"
    )

    print(
        f"  total SAFE bad edges     : "
        f"{safe_bad.sum():,}"
    )

    print(
        f"  wrap parity max          : "
        f"{parity.max():.3e} rad"
    )

    print()
    print("=" * 108)
    print(
        "Lowest raw-wrap branch-match IFGs"
    )
    print("=" * 108)

    print(
        " pair  dates                  "
        "branches mismatch   match% "
        "|dN|>=2 safeBad"
    )

    for r in weakest[:20]:

        print(
            f" {r['pair_id']:4d}  "
            f"{r['date1']}->"
            f"{r['date2']} "
            f"{r['branch_edges']:8d} "
            f"{r['branch_mismatch']:8d} "
            f"{r['raw_wrap_match_percent']:8.4f} "
            f"{r['branch_abs2plus']:7d} "
            f"{r['safe_internal_bad']:7d}"
        )

    # --------------------------------------------------------
    # Conservative audit classification
    # --------------------------------------------------------

    if np.any(
        abs2
        >
        0
    ):

        status = (
            "REVIEW_ABS_DN_GE_2"
        )

    elif match.min() < 0.95:

        status = (
            "REVIEW_LOW_BRANCH_MATCH"
        )

    else:

        status = (
            "PASS"
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-full-visual-branch-qa-v09",

        "status":
            status,

        "ifgs":
            int(n),

        "total_branch_edges":
            total_branch,

        "branch_abs2plus_edges":
            int(
                abs2.sum()
            ),

        "ifgs_with_abs2plus":
            int(
                np.count_nonzero(
                    abs2
                )
            ),

        "branch_mismatch_edges":
            total_mismatch,

        "weighted_global_branch_match":
            float(
                global_match
            ),

        "per_ifg_branch_match": {
            "min":
                float(
                    match.min()
                ),

            "p05":
                float(
                    np.percentile(
                        match,
                        5
                    )
                ),

            "median":
                float(
                    np.median(
                        match
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        match,
                        95
                    )
                ),

            "max":
                float(
                    match.max()
                ),
        },

        "safe_bad": {
            "ifgs":
                int(
                    np.count_nonzero(
                        safe_bad
                    )
                ),

            "edges":
                int(
                    safe_bad.sum()
                ),
        },

        "wrap_parity_max_rad":
            float(
                parity.max()
            ),
    }

    manifest_path = (
        indir
        / "visual_qa_v3_summary.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n"
    )

    worst_csv = (
        indir
        / "visual_qa_v3_weakest_branch_match.csv"
    )

    with worst_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                weakest[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            weakest
        )

    print()
    print(
        f"summary manifest           : "
        f"{manifest_path}"
    )

    print(
        f"ranked table               : "
        f"{worst_csv}"
    )

    print()
    print(
        f"STEP 08y3 STATUS: {status}"
    )


if __name__ == "__main__":
    main()
