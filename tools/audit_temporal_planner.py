#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypsds.context import (
    open_from_config,
)

from pypsds.phase_linking.temporal_plan import (
    build_temporal_plan,
)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--max-compressed",
        type=int,
        default=5,
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

    dates = list(
        stack.dates
    )

    ndate = len(
        dates
    )

    # Current validated temporal reference.
    reference_index = 0

    full = build_temporal_plan(
        dates,
        strategy="full_scm",
        ministack_size=ndate,
        max_num_compressed=(
            args.max_compressed
        ),
        reference_index=(
            reference_index
        ),
    )

    collapse = build_temporal_plan(
        dates,
        strategy="sequential",
        ministack_size=ndate,
        max_num_compressed=(
            args.max_compressed
        ),
        reference_index=(
            reference_index
        ),
    )

    # --------------------------------------------------------
    # Exact-collapse structural gate.
    # --------------------------------------------------------

    collapse_pass = (
        collapse.exact_collapse
        and
        collapse.execution_ready
        and
        collapse.effective_strategy
        ==
        "full_scm"
        and
        len(
            collapse.stages
        )
        ==
        1
        and
        collapse.stages[0]
        .real_indices
        ==
        full.stages[0]
        .real_indices
        and
        collapse.stages[0]
        .real_dates
        ==
        full.stages[0]
        .real_dates
        and
        collapse.stages[0]
        .compressed_count
        ==
        0
        and
        collapse.stages[0]
        .solver_size
        ==
        ndate
    )

    candidates = []

    for M in (
        ndate,
        30,
        20,
        15,
    ):

        if M < 2:
            continue

        # avoid duplicate M=38 if N itself is 38 etc.
        if any(
            x[
                "ministack_size"
            ]
            ==
            M
            for x
            in candidates
        ):
            continue

        p = build_temporal_plan(
            dates,
            strategy="sequential",
            ministack_size=M,
            max_num_compressed=(
                args.max_compressed
            ),
            reference_index=(
                reference_index
            ),
        )

        candidates.append(
            p.as_dict()
        )

    auto = build_temporal_plan(
        dates,
        strategy="auto",
        ministack_size=30,
        max_num_compressed=(
            args.max_compressed
        ),
        reference_index=(
            reference_index
        ),
    )

    print(
        "=" * 100
    )

    print(
        "pyPSDS-GAMMA production planner temporal/ministack planner audit"
    )

    print(
        "=" * 100
    )

    print(
        "dates            :",
        ndate,
    )

    print(
        "reference index  :",
        reference_index,
    )

    print(
        "reference date   :",
        dates[
            reference_index
        ],
    )

    print(
        "max compressed   :",
        args.max_compressed,
    )

    print()

    print(
        "M=N exact collapse:",
        (
            "PASS"
            if
            collapse_pass
            else
            "FAIL"
        ),
    )

    print()

    print(
        f"{'M':>6s}"
        f"{'effective':>14s}"
        f"{'stages':>10s}"
        f"{'max solver':>12s}"
        f"{'max comp':>10s}"
        f"{'collapse':>12s}"
        f"{'ready':>10s}"
    )

    print(
        "-" * 78
    )

    for x in candidates:

        print(
            f"{x['ministack_size']:6d}"
            f"{x['effective_strategy']:>14s}"
            f"{x['stage_count']:10d}"
            f"{x['max_solver_size']:12d}"
            f"{x['max_compressed_inputs']:10d}"
            f"{str(x['exact_collapse']):>12s}"
            f"{str(x['execution_ready']):>10s}"
        )

    print()

    print(
        "Detailed M<N stage layouts"
    )

    print(
        "-" * 100
    )

    for x in candidates:

        if x[
            "exact_collapse"
        ]:
            continue

        print()
        print(
            f"M={x['ministack_size']}"
        )

        for stage in x[
            "stages"
        ]:

            real = stage[
                "real_indices"
            ]

            print(
                f"  stage {stage['stage_index']:2d}: "
                f"real={real[0]:2d}..{real[-1]:2d} "
                f"({stage['real_count']:2d})  "
                f"compressed="
                f"{stage['compressed_count']}  "
                f"solver_size="
                f"{stage['solver_size']:2d}  "
                f"reference="
                f"{stage['output_reference']}"
            )

    print()

    print(
        "AUTO strategy"
    )

    print(
        "  effective :",
        auto.effective_strategy,
    )

    print(
        "  ready     :",
        auto.execution_ready,
    )

    print(
        "  reason    :",
        auto.decision_reason,
    )

    report = {
        "format":
            "pyPSDS-GAMMA-temporal-planner-audit-v1",

        "config":
            str(
                config_path
            ),

        "ndate":
            ndate,

        "reference_index":
            reference_index,

        "reference_date":
            dates[
                reference_index
            ],

        "full_scm":
            full.as_dict(),

        "exact_collapse":
            collapse.as_dict(),

        "exact_collapse_pass":
            collapse_pass,

        "candidates":
            candidates,

        "auto":
            auto.as_dict(),
    }

    out = (
        Path(
            paths.output_dir
        )
        /
        "processing"
        /
        "temporal_planner_audit.json"
    )

    out.write_text(
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
        "saved:",
        out,
    )

    if not collapse_pass:

        raise SystemExit(
            "production planner EXACT COLLAPSE FAILED"
        )

    print()
    print(
        "production planner TEMPORAL PLANNER: PASS"
    )


if __name__ == "__main__":
    main()
