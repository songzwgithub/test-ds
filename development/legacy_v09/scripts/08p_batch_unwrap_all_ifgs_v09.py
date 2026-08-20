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


# ============================================================
# Utilities
# ============================================================

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
                        r[
                            "local_frac_gt_pi_2"
                        ]
                    ),

                "p95":
                    float(
                        r[
                            "local_p95_abs_rad"
                        ]
                    ),

                "p99":
                    float(
                        r[
                            "local_p99_abs_rad"
                        ]
                    ),
            })

    rows.sort(
        key=lambda x:
            x["pair_id"]
    )

    return rows


def tag_of(r):

    return (
        f"pair{r['pair_id']:03d}_"
        f"{r['date1']}_"
        f"{r['date2']}"
    )


def read_json(path: Path):

    return json.loads(
        path.read_text()
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
            f"{script.name} failed "
            f"for pair {pair_id}. "
            f"See {log_path}"
        )


# ============================================================
# Reconstruct the exact 08l fragment spanning forest
# and audit whether non-exact constraints were required.
# ============================================================

class DSU:

    def __init__(self, n):

        self.parent = list(
            range(n)
        )

        self.size = [
            1
        ] * n

    def find(self, x):

        while (
            self.parent[x]
            != x
        ):

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        if (
            self.size[a]
            <
            self.size[b]
        ):
            a, b = b, a

        self.parent[b] = a

        self.size[a] += (
            self.size[b]
        )

        return True


def audit_selected_superforest(
    group_csv: Path,
    nsafe: int,
    target_edges: int,
):

    rows = []

    with group_csv.open() as f:

        for r in csv.DictReader(f):

            rows.append({
                "fragment_a":
                    int(
                        r["fragment_a"]
                    ),

                "fragment_b":
                    int(
                        r["fragment_b"]
                    ),

                "edge_count":
                    int(
                        r["edge_count"]
                    ),

                "mode_count":
                    int(
                        r["mode_count"]
                    ),

                "consensus_ratio":
                    float(
                        r["consensus_ratio"]
                    ),

                "exact_consensus":
                    int(
                        r["exact_consensus"]
                    ),

                "median_abs_gradient_rad":
                    float(
                        r[
                            "median_abs_gradient_rad"
                        ]
                    ),

                "median_distance_m":
                    float(
                        r[
                            "median_distance_m"
                        ]
                    ),
            })

    # IMPORTANT:
    # 08l writes the rows after sorting by exactly this
    # quality order. Reproduce the same order explicitly.
    rows.sort(
        key=lambda r: (
            -r[
                "consensus_ratio"
            ],
            -r[
                "edge_count"
            ],
            r[
                "median_abs_gradient_rad"
            ],
            r[
                "median_distance_m"
            ],
            r[
                "fragment_a"
            ],
            r[
                "fragment_b"
            ],
        )
    )

    dsu = DSU(
        nsafe
    )

    selected = []

    for gid, r in enumerate(
        rows
    ):

        if dsu.union(
            r["fragment_a"],
            r["fragment_b"],
        ):

            selected.append(
                r
            )

            if (
                len(selected)
                ==
                target_edges
            ):
                break

    if (
        len(selected)
        !=
        target_edges
    ):

        raise RuntimeError(
            "Unable to reconstruct "
            "08l superforest."
        )

    non_exact = [
        r
        for r in selected
        if not r[
            "exact_consensus"
        ]
    ]

    weak_majority = [
        r
        for r in selected
        if r[
            "consensus_ratio"
        ] < 0.75
    ]

    single_support = [
        r
        for r in selected
        if r[
            "edge_count"
        ] == 1
    ]

    if selected:

        min_ratio = min(
            r[
                "consensus_ratio"
            ]
            for r in selected
        )

        min_support = min(
            r[
                "edge_count"
            ]
            for r in selected
        )

    else:

        min_ratio = 1.0
        min_support = 0

    return {
        "selected_edges":
            len(selected),

        "selected_non_exact":
            len(non_exact),

        "selected_ratio_lt_0p75":
            len(weak_majority),

        "selected_single_support":
            len(single_support),

        "selected_min_consensus":
            float(
                min_ratio
            ),

        "selected_min_support":
            int(
                min_support
            ),
    }


def rejected_constraint_stats(
    path: Path,
):

    rejected = []

    with path.open() as f:

        for r in csv.DictReader(f):

            if (
                r["status"]
                ==
                "rejected_cycle_outlier"
            ):

                rejected.append({
                    "edge_count":
                        int(
                            r[
                                "edge_count"
                            ]
                        ),

                    "mode_count":
                        int(
                            r[
                                "mode_count"
                            ]
                        ),

                    "ratio":
                        float(
                            r[
                                "consensus_ratio"
                            ]
                        ),
                })

    if not rejected:

        return {
            "count": 0,
            "single_support": 0,
            "multi_support": 0,
            "max_support": 0,
            "strong_rejected": 0,
        }

    return {
        "count":
            len(rejected),

        "single_support":
            sum(
                r[
                    "edge_count"
                ] == 1
                for r in rejected
            ),

        "multi_support":
            sum(
                r[
                    "edge_count"
                ] > 1
                for r in rejected
            ),

        "max_support":
            max(
                r[
                    "edge_count"
                ]
                for r in rejected
            ),

        # Strong rejected constraint:
        # at least 3 raw crossing edges and >=80% vote.
        "strong_rejected":
            sum(
                (
                    r[
                        "edge_count"
                    ] >= 3
                )
                and
                (
                    r[
                        "ratio"
                    ] >= 0.80
                )
                for r in rejected
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--force",
        action="store_true",
    )

    ap.add_argument(
        "--stop-on-red",
        action="store_true",
        help=(
            "Stop immediately when a production "
            "red-flag IFG is detected."
        ),
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
        / "batch_unwrap_validation"
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

    pairs = load_08i(
        audit08i
    )

    print("=" * 112)
    print(
        "Step 08p - Full 108-IFG robust unwrapping validation"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"production IFGs            : "
        f"{len(pairs)}"
    )

    print(
        f"force recompute            : "
        f"{args.force}"
    )

    results = []

    for idx, r in enumerate(
        pairs,
        start=1,
    ):

        pair_id = r[
            "pair_id"
        ]

        tag = tag_of(
            r
        )

        manifest_l = (
            dir08l
            / f"{tag}_manifest.json"
        )

        manifest_n = (
            dir08n
            / f"{tag}_manifest.json"
        )

        group_csv = (
            dir08l
            / f"{tag}_fragment_pair_consensus.csv"
        )

        constraint_csv = (
            dir08n
            / f"{tag}_fragment_constraint_status.csv"
        )

        print()
        print(
            "-" * 112
        )

        print(
            f"[{idx:3d}/{len(pairs):3d}] "
            f"pair {pair_id:3d}  "
            f"{r['date1']} -> "
            f"{r['date2']}  "
            f">pi/2="
            f"{100*r['frac_gt_pi2']:.5f}%"
        )

        # ----------------------------------------------------
        # Run / reuse 08l
        # ----------------------------------------------------

        if (
            args.force
            or
            not manifest_l.exists()
            or
            not group_csv.exists()
        ):

            log_path = (
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
                log_path,
            )

        else:

            print(
                "  reuse 08l"
            )

        # ----------------------------------------------------
        # Run / reuse 08n
        # ----------------------------------------------------

        if (
            args.force
            or
            not manifest_n.exists()
            or
            not constraint_csv.exists()
        ):

            log_path = (
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
                log_path,
            )

        else:

            print(
                "  reuse 08n"
            )

        ml = read_json(
            manifest_l
        )

        mn = read_json(
            manifest_n
        )

        nsafe = int(
            ml[
                "topology"
            ][
                "safe_fragments"
            ]
        )

        target_merges = int(
            ml[
                "topology"
            ][
                "required_fragment_merges"
            ]
        )

        forest = audit_selected_superforest(
            group_csv,
            nsafe,
            target_merges,
        )

        rejected = rejected_constraint_stats(
            constraint_csv
        )

        safe_internal_bad = int(
            ml[
                "safe_internal"
            ][
                "nonzero_integer_jump_edges"
            ]
        )

        final_safe_bad = int(
            mn[
                "local_edge_qa"
            ][
                "safe_bad"
            ]
        )

        unsafe_within_bad = int(
            mn[
                "local_edge_qa"
            ][
                "unsafe_within_bad"
            ]
        )

        unsafe_cross_bad = int(
            mn[
                "local_edge_qa"
            ][
                "unsafe_cross_bad"
            ]
        )

        residual_conflict = int(
            mn[
                "residual_registration"
            ][
                "conflict"
            ]
        )

        registered_fraction = float(
            mn[
                "residual_registration"
            ][
                "registered_fraction"
            ]
        )

        wrap_error = float(
            mn[
                "wrap_back_max_error_rad"
            ]
        )

        # ----------------------------------------------------
        # Production red flags
        # ----------------------------------------------------

        red_reasons = []

        if safe_internal_bad != 0:

            red_reasons.append(
                "safe_internal_integer_conflict"
            )

        if final_safe_bad != 0:

            red_reasons.append(
                "final_safe_integer_conflict"
            )

        if wrap_error > 1.0e-4:

            red_reasons.append(
                "wrap_back_error_gt_1e-4"
            )

        if (
            forest[
                "selected_non_exact"
            ]
            > 0
        ):

            red_reasons.append(
                "superforest_requires_non_exact_consensus"
            )

        if (
            forest[
                "selected_ratio_lt_0p75"
            ]
            > 0
        ):

            red_reasons.append(
                "superforest_requires_consensus_lt_0p75"
            )

        if (
            rejected[
                "strong_rejected"
            ]
            > 0
        ):

            red_reasons.append(
                "strong_cycle_constraint_rejected"
            )

        # Registered fraction is deliberately not
        # a hard red flag here. Unresolved residual
        # components are allowed and already masked.

        status = (
            "REVIEW"
            if red_reasons
            else
            "PASS"
        )

        row = {
            "pair_id":
                pair_id,

            "date1":
                r["date1"],

            "date2":
                r["date2"],

            "step08i_frac_gt_pi2":
                r[
                    "frac_gt_pi2"
                ],

            "step08i_p95_rad":
                r["p95"],

            "safe_fragments":
                nsafe,

            "required_fragment_merges":
                target_merges,

            "safe_internal_bad":
                safe_internal_bad,

            "final_safe_bad":
                final_safe_bad,

            "unsafe_within_bad":
                unsafe_within_bad,

            "unsafe_cross_bad":
                unsafe_cross_bad,

            "fragment_pairs":
                int(
                    ml[
                        "unsafe_structure"
                    ][
                        "fragment_pairs"
                    ]
                ),

            "selected_superforest_edges":
                forest[
                    "selected_edges"
                ],

            "selected_non_exact":
                forest[
                    "selected_non_exact"
                ],

            "selected_ratio_lt_0p75":
                forest[
                    "selected_ratio_lt_0p75"
                ],

            "selected_single_support":
                forest[
                    "selected_single_support"
                ],

            "selected_min_consensus":
                forest[
                    "selected_min_consensus"
                ],

            "selected_min_support":
                forest[
                    "selected_min_support"
                ],

            "rejected_cycle_outliers":
                rejected[
                    "count"
                ],

            "rejected_single_support":
                rejected[
                    "single_support"
                ],

            "rejected_multi_support":
                rejected[
                    "multi_support"
                ],

            "strong_rejected":
                rejected[
                    "strong_rejected"
                ],

            "residual_conflicts":
                residual_conflict,

            "registered_fraction":
                registered_fraction,

            "wrap_back_max_error_rad":
                wrap_error,

            "status":
                status,

            "red_reasons":
                ";".join(
                    red_reasons
                ),
        }

        results.append(
            row
        )

        print(
            f"  safe fragments          : "
            f"{nsafe}"
        )

        print(
            f"  SAFE internal/final bad : "
            f"{safe_internal_bad} / "
            f"{final_safe_bad}"
        )

        print(
            f"  forest non-exact        : "
            f"{forest['selected_non_exact']}"
        )

        print(
            f"  rejected cycle outliers : "
            f"{rejected['count']} "
            f"(strong={rejected['strong_rejected']})"
        )

        print(
            f"  residual conflicts      : "
            f"{residual_conflict}"
        )

        print(
            f"  registered              : "
            f"{100*registered_fraction:.6f}%"
        )

        print(
            f"  QA status               : "
            f"{status}"
        )

        if red_reasons:

            print(
                "  reasons                 : "
                + ", ".join(
                    red_reasons
                )
            )

            if args.stop_on_red:

                raise RuntimeError(
                    f"Production red flag "
                    f"at pair {pair_id}: "
                    f"{red_reasons}"
                )

    # ========================================================
    # Global summary
    # ========================================================

    n = len(
        results
    )

    passed = [
        r
        for r in results
        if r["status"] == "PASS"
    ]

    review = [
        r
        for r in results
        if r["status"] == "REVIEW"
    ]

    safe_bad_pairs = [
        r
        for r in results
        if (
            r[
                "safe_internal_bad"
            ] != 0
            or
            r[
                "final_safe_bad"
            ] != 0
        )
    ]

    non_exact_pairs = [
        r
        for r in results
        if r[
            "selected_non_exact"
        ] > 0
    ]

    strong_rejected_pairs = [
        r
        for r in results
        if r[
            "strong_rejected"
        ] > 0
    ]

    reg = np.asarray(
        [
            r[
                "registered_fraction"
            ]
            for r in results
        ],
        dtype=np.float64,
    )

    residual_conflicts = np.asarray(
        [
            r[
                "residual_conflicts"
            ]
            for r in results
        ],
        dtype=np.int32,
    )

    cycle_outliers = np.asarray(
        [
            r[
                "rejected_cycle_outliers"
            ]
            for r in results
        ],
        dtype=np.int32,
    )

    n_frag = np.asarray(
        [
            r[
                "safe_fragments"
            ]
            for r in results
        ],
        dtype=np.int32,
    )

    print()
    print("=" * 112)
    print(
        "FULL 108-IFG QA SUMMARY"
    )
    print("=" * 112)

    print(
        f"IFGs processed             : "
        f"{n}"
    )

    print(
        f"PASS                       : "
        f"{len(passed)}/{n}"
    )

    print(
        f"REVIEW                     : "
        f"{len(review)}/{n}"
    )

    print(
        f"SAFE-conflict IFGs         : "
        f"{len(safe_bad_pairs)}"
    )

    print(
        f"non-exact forest IFGs      : "
        f"{len(non_exact_pairs)}"
    )

    print(
        f"strong rejected IFGs       : "
        f"{len(strong_rejected_pairs)}"
    )

    print()

    print(
        f"safe fragments min/med/max : "
        f"{n_frag.min()} / "
        f"{np.median(n_frag):.1f} / "
        f"{n_frag.max()}"
    )

    print(
        f"cycle outliers min/med/max : "
        f"{cycle_outliers.min()} / "
        f"{np.median(cycle_outliers):.1f} / "
        f"{cycle_outliers.max()}"
    )

    print(
        f"residual conflicts min/med/max: "
        f"{residual_conflicts.min()} / "
        f"{np.median(residual_conflicts):.1f} / "
        f"{residual_conflicts.max()}"
    )

    print(
        f"registered min/med/max     : "
        f"{100*reg.min():.6f}% / "
        f"{100*np.median(reg):.6f}% / "
        f"{100*reg.max():.6f}%"
    )

    if review:

        print()
        print(
            "IFGs requiring review:"
        )

        for r in review:

            print(
                f"  pair {r['pair_id']:3d} "
                f"{r['date1']}->{r['date2']} : "
                f"{r['red_reasons']}"
            )

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / "all_ifg_unwrap_qa.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            results
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-full-ifg-unwrapping-validation-v0.9",

        "status":
            (
                "PASS"
                if not review
                else
                "REVIEW_REQUIRED"
            ),

        "ifgs":
            n,

        "pass":
            len(
                passed
            ),

        "review":
            len(
                review
            ),

        "safe_conflict_ifgs":
            len(
                safe_bad_pairs
            ),

        "non_exact_forest_ifgs":
            len(
                non_exact_pairs
            ),

        "strong_rejected_ifgs":
            len(
                strong_rejected_pairs
            ),

        "safe_fragments": {
            "min":
                int(
                    n_frag.min()
                ),

            "median":
                float(
                    np.median(
                        n_frag
                    )
                ),

            "max":
                int(
                    n_frag.max()
                ),
        },

        "cycle_outliers": {
            "min":
                int(
                    cycle_outliers.min()
                ),

            "median":
                float(
                    np.median(
                        cycle_outliers
                    )
                ),

            "max":
                int(
                    cycle_outliers.max()
                ),
        },

        "residual_conflicts": {
            "min":
                int(
                    residual_conflicts.min()
                ),

            "median":
                float(
                    np.median(
                        residual_conflicts
                    )
                ),

            "max":
                int(
                    residual_conflicts.max()
                ),
        },

        "registered_fraction": {
            "min":
                float(
                    reg.min()
                ),

            "median":
                float(
                    np.median(
                        reg
                    )
                ),

            "max":
                float(
                    reg.max()
                ),
        },

        "review_pair_ids":
            [
                r[
                    "pair_id"
                ]
                for r in review
            ],
    }

    manifest_path = (
        outdir
        / "all_ifg_unwrap_qa.json"
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
        f"QA CSV                     : "
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

    if review:

        print(
            "STEP 08p STATUS: COMPLETE / "
            "REVIEW REQUIRED"
        )

        print(
            "Do not proceed to time-series inversion."
        )

    else:

        print(
            "STEP 08p STATUS: PASS / "
            "ALL 108 IFGs VALIDATED"
        )

        print(
            "Step 08 unwrapping framework may now be frozen."
        )


if __name__ == "__main__":
    main()
