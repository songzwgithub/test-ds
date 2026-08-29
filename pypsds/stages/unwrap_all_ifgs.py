#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pypsds.config import cfg_get
from pypsds.context import open_from_config


# ============================================================
# Utilities
# ============================================================

def load_spatial_gradient_quality(path: Path):

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

    child_env = os.environ.copy()

    # IFG-level parallelism owns CPU scheduling. Prevent nested
    # BLAS/Numba oversubscription inside each independent pair.
    for key in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        child_env[key] = "1"

    with log_path.open(
        "w"
    ) as log:

        p = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env,
        )

    if p.returncode != 0:

        raise RuntimeError(
            f"{script.name} failed "
            f"for pair {pair_id}. "
            f"See {log_path}"
        )


# ============================================================
# PYPSDS_UNWRAP_PAIR_PARALLEL_V1
#
# IFG-level production precompute.
#
# Scientific operations are unchanged. Only orchestration changes:
# independent temporal-network IFGs are prepared concurrently,
# while final QA/aggregation remains deterministic in pair_id order.
# ============================================================

def ensure_pair_products(
    r,
    *,
    force,
    config_path,
    safe_fragment_script,
    single_ifg_solution_script,
    safe_fragment_quality_dir,
    single_ifg_solution_dir,
    logdir,
):
    pair_id = int(r["pair_id"])
    tag = tag_of(r)

    manifest_l = (
        safe_fragment_quality_dir
        / f"{tag}_manifest.json"
    )
    manifest_n = (
        single_ifg_solution_dir
        / f"{tag}_manifest.json"
    )
    group_csv = (
        safe_fragment_quality_dir
        / f"{tag}_fragment_pair_consensus.csv"
    )
    constraint_csv = (
        single_ifg_solution_dir
        / f"{tag}_fragment_constraint_status.csv"
    )

    actions = []

    if (
        force
        or not manifest_l.exists()
        or not group_csv.exists()
    ):
        run_script(
            safe_fragment_script,
            config_path,
            pair_id,
            logdir / f"{tag}_safe_fragment.log",
        )
        actions.append("safe")

    if (
        force
        or not manifest_n.exists()
        or not constraint_csv.exists()
    ):
        run_script(
            single_ifg_solution_script,
            config_path,
            pair_id,
            logdir / f"{tag}_single_ifg.log",
        )
        actions.append("solution")

    return {
        "pair_id": pair_id,
        "tag": tag,
        "actions": tuple(actions),
    }


def resolve_ifg_workers(
    cfg,
    *,
    pair_count,
):
    raw_env = os.environ.get(
        "PYPSDS_UNWRAP_IFG_WORKERS",
        "",
    ).strip()

    raw_cfg = cfg_get(
        cfg,
        "runtime.unwrap_ifg_workers",
        "auto",
    )

    raw = raw_env if raw_env else raw_cfg

    if raw in (
        None,
        "",
        "auto",
    ):
        workers = min(
            8,
            os.cpu_count() or 1,
            pair_count,
        )
    else:
        workers = int(raw)

    if workers < 1:
        raise ValueError(
            "runtime.unwrap_ifg_workers / "
            "PYPSDS_UNWRAP_IFG_WORKERS must be >=1"
        )

    return min(
        workers,
        pair_count,
    )



# ============================================================
# Reconstruct the exact safe_fragment_quality fragment spanning forest
# and quality whether non-exact constraints were required.
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


def quality_selected_superforest(
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
    # safe_fragment_quality writes the rows after sorting by exactly this
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
            "safe_fragment_quality superforest."
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

    stage_dir = Path(__file__).resolve().parent

    safe_fragment_script = (
        stage_dir / "build_safe_fragment_integer_quality.py"
    )

    single_ifg_solution_script = (
        stage_dir / "finalize_single_ifg_solution.py"
    )

    root = (
        Path(paths.output_dir)
        / "processing"
    )

    spatial_gradient_quality_csv = (
        root
        / "spatial_phase_gradient_quality"
        / "per_ifg_spatial_gradient_qa.csv"
    )

    safe_fragment_quality_dir = (
        root
        / "safe_fragment_integer_quality"
    )

    single_ifg_solution_dir = (
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

    pairs = load_spatial_gradient_quality(
        spatial_gradient_quality_csv
    )

    print("=" * 112)
    print(
        "Full temporal-network IFG robust unwrapping validation"
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

    ifg_workers = resolve_ifg_workers(
        cfg,
        pair_count=len(pairs),
    )

    # --stop-on-red keeps legacy immediate-stop semantics.
    effective_workers = (
        1
        if args.stop_on_red
        else ifg_workers
    )

    print(
        f"IFG parallel workers       : "
        f"{effective_workers}"
    )

    if (
        args.stop_on_red
        and ifg_workers != 1
    ):
        print(
            "stop-on-red               : "
            "serial semantics preserved"
        )

    parallel_precompute_done = False

    if effective_workers > 1:

        pending_pairs = []

        for r in pairs:
            tag = tag_of(r)

            manifest_l = (
                safe_fragment_quality_dir
                / f"{tag}_manifest.json"
            )
            manifest_n = (
                single_ifg_solution_dir
                / f"{tag}_manifest.json"
            )
            group_csv = (
                safe_fragment_quality_dir
                / f"{tag}_fragment_pair_consensus.csv"
            )
            constraint_csv = (
                single_ifg_solution_dir
                / f"{tag}_fragment_constraint_status.csv"
            )

            if (
                args.force
                or not manifest_l.exists()
                or not group_csv.exists()
                or not manifest_n.exists()
                or not constraint_csv.exists()
            ):
                pending_pairs.append(r)

        print(
            f"parallel pending IFGs      : "
            f"{len(pending_pairs)}/{len(pairs)}"
        )

        if pending_pairs:

            completed = 0

            with ThreadPoolExecutor(
                max_workers=effective_workers,
                thread_name_prefix="ifg",
            ) as executor:

                futures = {
                    executor.submit(
                        ensure_pair_products,
                        r,
                        force=args.force,
                        config_path=config_path,
                        safe_fragment_script=safe_fragment_script,
                        single_ifg_solution_script=single_ifg_solution_script,
                        safe_fragment_quality_dir=safe_fragment_quality_dir,
                        single_ifg_solution_dir=single_ifg_solution_dir,
                        logdir=logdir,
                    ):
                    int(r["pair_id"])
                    for r in pending_pairs
                }

                for future in as_completed(futures):
                    pair_id = futures[future]

                    try:
                        info = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            "Parallel IFG precompute failed "
                            f"for pair {pair_id}"
                        ) from exc

                    completed += 1

                    action_text = (
                        "+".join(info["actions"])
                        if info["actions"]
                        else "reuse"
                    )

                    print(
                        f"[parallel {completed:3d}/"
                        f"{len(pending_pairs):3d}] "
                        f"pair {pair_id:3d} : "
                        f"{action_text}",
                        flush=True,
                    )

        parallel_precompute_done = True

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
            safe_fragment_quality_dir
            / f"{tag}_manifest.json"
        )

        manifest_n = (
            single_ifg_solution_dir
            / f"{tag}_manifest.json"
        )

        group_csv = (
            safe_fragment_quality_dir
            / f"{tag}_fragment_pair_consensus.csv"
        )

        constraint_csv = (
            single_ifg_solution_dir
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
        # Run / reuse safe_fragment_quality
        # ----------------------------------------------------

        if (
            (
                args.force
                and not parallel_precompute_done
            )
            or
            not manifest_l.exists()
            or
            not group_csv.exists()
        ):

            log_path = (
                logdir
                / f"{tag}_safe_fragment.log"
            )

            print(
                "  running safe-fragment integer quality ..."
            )

            run_script(
                safe_fragment_script,
                config_path,
                pair_id,
                log_path,
            )

        else:

            print(
                "  reuse safe-fragment integer quality"
            )

        # ----------------------------------------------------
        # Run / reuse single_ifg_solution
        # ----------------------------------------------------

        if (
            (
                args.force
                and not parallel_precompute_done
            )
            or
            not manifest_n.exists()
            or
            not constraint_csv.exists()
        ):

            log_path = (
                logdir
                / f"{tag}_single_ifg.log"
            )

            print(
                "  running single-IFG robust solution ..."
            )

            run_script(
                single_ifg_solution_script,
                config_path,
                pair_id,
                log_path,
            )

        else:

            print(
                "  reuse single-IFG robust solution"
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

        forest = quality_selected_superforest(
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

            "stepspatial_gradient_quality_frac_gt_pi2":
                r[
                    "frac_gt_pi2"
                ],

            "stepspatial_gradient_quality_p95_rad":
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
        "FULL temporal-network IFG QA SUMMARY"
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
            "pyPSDS-GAMMA-full-ifg-unwrapping-validation-v1.0",

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
            "STEP unwrap_batch STATUS: COMPLETE / "
            "REVIEW REQUIRED"
        )

        print(
            "Do not proceed to time-series inversion."
        )

    else:

        print(
            "STEP unwrap_batch STATUS: PASS / "
            "ALL temporal-network IFGs VALIDATED"
        )

        print(
            "Step 08 unwrapping framework may now be frozen."
        )


if __name__ == "__main__":
    main()
