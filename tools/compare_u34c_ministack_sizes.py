#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def pct(x):
    return 100.0 * float(x)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--processing-dir",
        required=True,
    )

    args = ap.parse_args()

    seq = (
        Path(args.processing_dir)
        /
        "sequential"
    )

    rows = []

    for M in (
        12,
        16,
        19,
    ):

        u33 = read_json(
            seq
            /
            f"u34c_M{M}_u33b_report.json"
        )

        gate = read_json(
            seq
            /
            f"u34c_M{M}_production_gate.json"
        )

        phase = read_json(
            seq
            /
            f"u34c_M{M}_phase_report.json"
        )

        prod = gate[
            "groups"
        ][
            "production_TCge0p80"
        ]

        stages = u33[
            "stages"
        ]

        max_solver = max(
            int(x["solver_size"])
            for x in stages
        )

        stage_p95 = []

        for x in phase[
            "per_stage"
        ]:

            e = x[
                "point_epoch_error_deg"
            ]

            stage_p95.append(
                float(
                    e["p95"]
                )
            )

        row = {
            "M":
                M,

            "stages":
                len(stages),

            "max_solver":
                max_solver,

            "runtime_s":
                float(
                    u33[
                        "total_elapsed_seconds"
                    ]
                ),

            "coverage":
                float(
                    gate[
                        "accepted_sequential_fraction"
                    ]
                ),

            "sim50":
                float(
                    prod[
                        "similarity"
                    ][
                        "median"
                    ]
                ),

            "med50":
                float(
                    prod[
                        "median_error_deg"
                    ][
                        "median"
                    ]
                ),

            "med95":
                float(
                    prod[
                        "median_error_deg"
                    ][
                        "p95"
                    ]
                ),

            "p95_50":
                float(
                    prod[
                        "p95_error_deg"
                    ][
                        "median"
                    ]
                ),

            "p95_95":
                float(
                    prod[
                        "p95_error_deg"
                    ][
                        "p95"
                    ]
                ),

            "median_gt30":
                float(
                    prod[
                        "median_error_fractions"
                    ][
                        "gt_30deg"
                    ]
                ),

            "p95_gt60":
                float(
                    prod[
                        "p95_error_fractions"
                    ][
                        "gt_60deg"
                    ]
                ),

            "p95_gt90":
                float(
                    prod[
                        "p95_error_fractions"
                    ][
                        "gt_90deg"
                    ]
                ),

            "worst_stage_p95":
                max(
                    stage_p95
                ),

            "last_stage_p95":
                stage_p95[-1],
        }

        rows.append(
            row
        )

    base = rows[0]

    print("=" * 157)

    print(
        "U3.4c MINISTACK-SIZE SENSITIVITY"
    )

    print("=" * 157)

    print(
        "M  stages maxSolve runtime(s) "
        "seqCov%   sim50   medErr50 medErr95 "
        "p95Err50 p95Err95 "
        "med>30% p95>60% p95>90% "
        "worstStageP95"
    )

    print("-" * 157)

    for x in rows:

        print(
            f"{x['M']:2d} "
            f"{x['stages']:6d} "
            f"{x['max_solver']:8d} "
            f"{x['runtime_s']:10.2f} "
            f"{100*x['coverage']:7.3f} "
            f"{x['sim50']:8.4f} "
            f"{x['med50']:9.3f} "
            f"{x['med95']:8.3f} "
            f"{x['p95_50']:8.3f} "
            f"{x['p95_95']:8.3f} "
            f"{100*x['median_gt30']:7.3f} "
            f"{100*x['p95_gt60']:8.3f} "
            f"{100*x['p95_gt90']:8.3f} "
            f"{x['worst_stage_p95']:13.3f}"
        )

    print()

    print("=" * 100)
    print("CHANGE RELATIVE TO M=12")
    print("=" * 100)

    for x in rows[1:]:

        print(
            f"M={x['M']}:"
        )

        print(
            "  runtime factor       : "
            f"{x['runtime_s']/base['runtime_s']:.3f}x"
        )

        print(
            "  median error change  : "
            f"{x['med50']-base['med50']:+.3f} deg"
        )

        print(
            "  median-error p95     : "
            f"{x['med95']-base['med95']:+.3f} deg"
        )

        print(
            "  p95-error median     : "
            f"{x['p95_50']-base['p95_50']:+.3f} deg"
        )

        print(
            "  p95-error p95       : "
            f"{x['p95_95']-base['p95_95']:+.3f} deg"
        )

        print(
            "  p95>60 change       : "
            f"{100*(x['p95_gt60']-base['p95_gt60']):+.3f} pp"
        )

        print(
            "  worst-stage p95     : "
            f"{x['worst_stage_p95']-base['worst_stage_p95']:+.3f} deg"
        )

        print()

    out = (
        seq
        /
        "u34c_ministack_size_comparison.json"
    )

    out.write_text(
        json.dumps(
            rows,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        "json:",
        out,
    )

    print()

    print(
        "U3.4c MINISTACK-SIZE COMPARISON: PASS"
    )


if __name__ == "__main__":
    main()
