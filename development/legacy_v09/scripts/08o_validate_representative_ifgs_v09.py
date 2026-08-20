#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


def load_08i(path: Path):
    rows = []

    with path.open() as f:
        for r in csv.DictReader(f):

            rows.append({
                "pair_id":
                    int(r["pair_id"]),

                "date1":
                    r["date1"],

                "date2":
                    r["date2"],

                "frac_gt_pi2":
                    float(
                        r["local_frac_gt_pi_2"]
                    ),

                "p95":
                    float(
                        r["local_p95_abs_rad"]
                    ),

                "p99":
                    float(
                        r["local_p99_abs_rad"]
                    ),

                "median":
                    float(
                        r["local_median_abs_rad"]
                    ),
            })

    if not rows:
        raise RuntimeError(
            "08i table is empty."
        )

    return rows


def tag_of(row):
    return (
        f"pair{row['pair_id']:03d}_"
        f"{row['date1']}_"
        f"{row['date2']}"
    )


def select_representative(rows, n_rank):

    # Same primary difficulty logic used for
    # worst-case selection:
    #
    #   fraction |g| > pi/2
    #   then p95
    #
    ordered = sorted(
        rows,
        key=lambda r: (
            r["frac_gt_pi2"],
            r["p95"],
        ),
    )

    idx = np.rint(
        np.linspace(
            0,
            len(ordered) - 1,
            n_rank,
        )
    ).astype(int)

    selected = {
        ordered[i]["pair_id"]:
            ordered[i]
        for i in idx
    }

    # Ensure maximum p95 is represented too.
    max_p95 = max(
        rows,
        key=lambda r:
            r["p95"],
    )

    selected[
        max_p95["pair_id"]
    ] = max_p95

    # Ensure lexicographic worst IFG is included.
    worst = max(
        rows,
        key=lambda r: (
            r["frac_gt_pi2"],
            r["p95"],
        ),
    )

    selected[
        worst["pair_id"]
    ] = worst

    # Return in ascending difficulty order.
    return sorted(
        selected.values(),
        key=lambda r: (
            r["frac_gt_pi2"],
            r["p95"],
        ),
    )


def run_script(
    script,
    config,
    pair_id,
    log_path,
):
    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(config),
        "--pair-id",
        str(pair_id),
    ]

    with log_path.open(
        "w"
    ) as log:

        p = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if p.returncode != 0:
        raise RuntimeError(
            f"{script.name} failed for "
            f"pair {pair_id}. "
            f"See {log_path}"
        )


def read_json(path):
    return json.loads(
        path.read_text()
    )


def rejected_constraint_stats(
    csv_path: Path,
):
    rejected = []

    with csv_path.open() as f:

        for r in csv.DictReader(f):

            if (
                r["status"]
                ==
                "rejected_cycle_outlier"
            ):
                rejected.append(r)

    if not rejected:

        return {
            "rejected":
                0,

            "single_support_rejected":
                0,

            "max_rejected_edge_count":
                0,

            "min_rejected_consensus":
                None,
        }

    edge_counts = [
        int(r["edge_count"])
        for r in rejected
    ]

    ratios = [
        float(
            r["consensus_ratio"]
        )
        for r in rejected
    ]

    return {
        "rejected":
            len(rejected),

        "single_support_rejected":
            sum(
                x == 1
                for x in edge_counts
            ),

        "max_rejected_edge_count":
            max(edge_counts),

        "min_rejected_consensus":
            min(ratios),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--n-rank",
        type=int,
        default=9,
    )

    ap.add_argument(
        "--force",
        action="store_true",
    )

    args = ap.parse_args()

    if args.n_rank < 3:
        raise ValueError(
            "--n-rank must be >= 3"
        )

    (
        cfg,
        config_path,
        paths,
        stack,
        _,
    ) = open_from_config(
        args.config
    )

    repo = Path.cwd()

    script08l = (
        repo
        / "scripts"
        / "08l_safe_fragment_integer_audit_v09.py"
    )

    script08n = (
        repo
        / "scripts"
        / "08n_finalize_single_ifg_robust_solution_v09.py"
    )

    if not script08l.exists():
        raise FileNotFoundError(
            script08l
        )

    if not script08n.exists():
        raise FileNotFoundError(
            script08n
        )

    root = (
        Path(paths.output_dir)
        / "v09"
    )

    audit08i = (
        root
        / "spatial_phase_gradient_audit"
        / "per_ifg_spatial_gradient_qa.csv"
    )

    dir08l = (
        root
        / "safe_fragment_integer_audit"
    )

    dir08n = (
        root
        / "single_ifg_robust_solution"
    )

    outdir = (
        root
        / "representative_ifg_validation"
    )

    logdir = (
        outdir
        / "logs"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_08i(
        audit08i
    )

    selected = select_representative(
        rows,
        args.n_rank,
    )

    print("=" * 108)
    print(
        "Step 08o - Representative multi-IFG robustness validation"
    )
    print("=" * 108)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"total production IFGs      : "
        f"{len(rows)}"
    )

    print(
        f"selected validation IFGs   : "
        f"{len(selected)}"
    )

    print()

    print(
        "Selected pairs "
        "(ascending Step08i difficulty):"
    )

    for k, r in enumerate(
        selected,
        start=1,
    ):

        print(
            f"  {k:2d}. "
            f"pair {r['pair_id']:3d}  "
            f"{r['date1']}->{r['date2']}  "
            f">pi/2="
            f"{100*r['frac_gt_pi2']:.5f}%  "
            f"p95={r['p95']:.4f}"
        )

    summary_rows = []

    for k, r in enumerate(
        selected,
        start=1,
    ):

        pair_id = r[
            "pair_id"
        ]

        tag = tag_of(
            r
        )

        manifest08l = (
            dir08l
            / f"{tag}_manifest.json"
        )

        manifest08n = (
            dir08n
            / f"{tag}_manifest.json"
        )

        status_csv = (
            dir08n
            / f"{tag}_fragment_constraint_status.csv"
        )

        print()
        print(
            "-" * 108
        )

        print(
            f"[{k}/{len(selected)}] "
            f"pair {pair_id}: "
            f"{r['date1']} -> "
            f"{r['date2']}"
        )

        need08l = (
            args.force
            or
            not manifest08l.exists()
        )

        if need08l:

            log = (
                logdir
                / f"{tag}_08l.log"
            )

            print(
                "  running 08l ..."
            )

            run_script(
                script08l,
                config_path,
                pair_id,
                log,
            )

        else:

            print(
                "  08l existing result reused."
            )

        need08n = (
            args.force
            or
            not manifest08n.exists()
            or
            not status_csv.exists()
        )

        if need08n:

            log = (
                logdir
                / f"{tag}_08n.log"
            )

            print(
                "  running 08n ..."
            )

            run_script(
                script08n,
                config_path,
                pair_id,
                log,
            )

        else:

            print(
                "  08n existing result reused."
            )

        m_l = read_json(
            manifest08l
        )

        m_n = read_json(
            manifest08n
        )

        rej = rejected_constraint_stats(
            status_csv
        )

        safe_internal_bad = int(
            m_l[
                "safe_internal"
            ][
                "nonzero_integer_jump_edges"
            ]
        )

        safe_fragments = int(
            m_l[
                "topology"
            ][
                "safe_fragments"
            ]
        )

        required_merges = int(
            m_l[
                "topology"
            ][
                "required_fragment_merges"
            ]
        )

        fragment_pairs = int(
            m_l[
                "unsafe_structure"
            ][
                "fragment_pairs"
            ]
        )

        raw_conflicting_pairs = int(
            m_l[
                "unsafe_structure"
            ][
                "conflicting_pairs"
            ]
        )

        safe_bad_final = int(
            m_n[
                "local_edge_qa"
            ][
                "safe_bad"
            ]
        )

        unsafe_cross_bad = int(
            m_n[
                "local_edge_qa"
            ][
                "unsafe_cross_bad"
            ]
        )

        residual_conflict = int(
            m_n[
                "residual_registration"
            ][
                "conflict"
            ]
        )

        registered_fraction = float(
            m_n[
                "residual_registration"
            ][
                "registered_fraction"
            ]
        )

        wrap_error = float(
            m_n[
                "wrap_back_max_error_rad"
            ]
        )

        out = {
            "pair_id":
                pair_id,

            "date1":
                r["date1"],

            "date2":
                r["date2"],

            "step08i_frac_gt_pi2":
                r["frac_gt_pi2"],

            "step08i_p95_rad":
                r["p95"],

            "safe_fragments":
                safe_fragments,

            "required_fragment_merges":
                required_merges,

            "safe_internal_bad":
                safe_internal_bad,

            "fragment_pairs":
                fragment_pairs,

            "raw_conflicting_pair_groups":
                raw_conflicting_pairs,

            "rejected_cycle_outliers":
                rej[
                    "rejected"
                ],

            "single_support_rejected":
                rej[
                    "single_support_rejected"
                ],

            "max_rejected_edge_count":
                rej[
                    "max_rejected_edge_count"
                ],

            "min_rejected_consensus":
                rej[
                    "min_rejected_consensus"
                ],

            "final_safe_bad":
                safe_bad_final,

            "final_unsafe_cross_bad":
                unsafe_cross_bad,

            "residual_anchor_conflicts":
                residual_conflict,

            "registered_fraction":
                registered_fraction,

            "wrap_back_max_error_rad":
                wrap_error,
        }

        summary_rows.append(
            out
        )

        print(
            f"  safe fragments          : "
            f"{safe_fragments}"
        )

        print(
            f"  SAFE internal bad       : "
            f"{safe_internal_bad}"
        )

        print(
            f"  fragment pairs          : "
            f"{fragment_pairs}"
        )

        print(
            f"  rejected cycle outliers : "
            f"{rej['rejected']}"
        )

        print(
            f"  single-support rejected : "
            f"{rej['single_support_rejected']}"
        )

        print(
            f"  final SAFE bad          : "
            f"{safe_bad_final}"
        )

        print(
            f"  residual conflicts      : "
            f"{residual_conflict}"
        )

        print(
            f"  registered              : "
            f"{100*registered_fraction:.6f}%"
        )

    # ========================================================
    # Overall QA
    # ========================================================

    ntest = len(
        summary_rows
    )

    safe_internal_zero = sum(
        r["safe_internal_bad"]
        ==
        0
        for r in summary_rows
    )

    final_safe_zero = sum(
        r["final_safe_bad"]
        ==
        0
        for r in summary_rows
    )

    reg = np.array(
        [
            r["registered_fraction"]
            for r in summary_rows
        ],
        dtype=np.float64,
    )

    rejected = np.array(
        [
            r[
                "rejected_cycle_outliers"
            ]
            for r in summary_rows
        ],
        dtype=np.int32,
    )

    residual_conflict = np.array(
        [
            r[
                "residual_anchor_conflicts"
            ]
            for r in summary_rows
        ],
        dtype=np.int32,
    )

    print()
    print("=" * 108)
    print(
        "Representative validation summary"
    )
    print("=" * 108)

    print(
        f"tested IFGs                : "
        f"{ntest}"
    )

    print(
        f"SAFE internal bad = 0      : "
        f"{safe_internal_zero}/{ntest}"
    )

    print(
        f"final SAFE bad = 0         : "
        f"{final_safe_zero}/{ntest}"
    )

    print(
        f"cycle outliers "
        f"min/median/max      : "
        f"{rejected.min()} / "
        f"{np.median(rejected):.1f} / "
        f"{rejected.max()}"
    )

    print(
        f"residual conflicts "
        f"min/median/max      : "
        f"{residual_conflict.min()} / "
        f"{np.median(residual_conflict):.1f} / "
        f"{residual_conflict.max()}"
    )

    print(
        f"registered fraction "
        f"min/median/max      : "
        f"{100*reg.min():.6f}% / "
        f"{100*np.median(reg):.6f}% / "
        f"{100*reg.max():.6f}%"
    )

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / "representative_ifg_validation.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            summary_rows
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-representative-ifg-validation-v0.9",

        "status":
            "VALIDATION_NOT_PRODUCTION_FROZEN",

        "selection": {
            "method":
                (
                    "even ranks over Step08i "
                    "frac_gt_pi2 difficulty, "
                    "plus maximum p95 and worst IFG"
                ),

            "tested_pair_ids":
                [
                    r["pair_id"]
                    for r in summary_rows
                ],
        },

        "summary": {
            "tested_ifgs":
                ntest,

            "safe_internal_zero":
                safe_internal_zero,

            "final_safe_zero":
                final_safe_zero,

            "registered_fraction_min":
                float(
                    reg.min()
                ),

            "registered_fraction_median":
                float(
                    np.median(reg)
                ),

            "registered_fraction_max":
                float(
                    reg.max()
                ),

            "cycle_outliers_max":
                int(
                    rejected.max()
                ),

            "residual_conflicts_max":
                int(
                    residual_conflict.max()
                ),
        },
    }

    manifest_path = (
        outdir
        / "representative_ifg_validation.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print(
        f"summary CSV                : "
        f"{csv_path}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print(
        f"logs                       : "
        f"{logdir}"
    )

    print()
    print(
        "STEP 08o STATUS: PASS / "
        "REPRESENTATIVE VALIDATION COMPLETE"
    )

    print(
        "Do not run all 108 IFGs until "
        "the summary metrics are reviewed."
    )


if __name__ == "__main__":
    main()
