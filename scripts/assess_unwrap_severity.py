#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def load_08i(path: Path):

    out = []

    with path.open() as f:

        for r in csv.DictReader(f):

            out.append({
                "pair_id": int(r["pair_id"]),
                "date1": r["date1"],
                "date2": r["date2"],
            })

    out.sort(
        key=lambda x: x["pair_id"]
    )

    return out


def tag_of(r):

    return (
        f"pair{r['pair_id']:03d}_"
        f"{r['date1']}_"
        f"{r['date2']}"
    )


def read_groups(path: Path):

    rows = []

    with path.open() as f:

        for r in csv.DictReader(f):

            rows.append({
                "fragment_a":
                    int(r["fragment_a"]),

                "fragment_b":
                    int(r["fragment_b"]),

                "edge_count":
                    int(r["edge_count"]),

                "mode_count":
                    int(r["mode_count"]),

                "consensus_ratio":
                    float(r["consensus_ratio"]),

                "exact_consensus":
                    int(r["exact_consensus"]),

                "median_abs_gradient_rad":
                    float(
                        r["median_abs_gradient_rad"]
                    ),

                "median_distance_m":
                    float(
                        r["median_distance_m"]
                    ),
            })

    # Exact Step08l quality ordering.
    rows.sort(
        key=lambda r: (
            -r["consensus_ratio"],
            -r["edge_count"],
            r["median_abs_gradient_rad"],
            r["median_distance_m"],
            r["fragment_a"],
            r["fragment_b"],
        )
    )

    return rows


class DSU:

    def __init__(self, n):

        self.p = list(range(n))
        self.s = [1] * n

    def find(self, x):

        while self.p[x] != x:

            self.p[x] = self.p[
                self.p[x]
            ]

            x = self.p[x]

        return x

    def union(self, a, b):

        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        if self.s[a] < self.s[b]:
            a, b = b, a

        self.p[b] = a
        self.s[a] += self.s[b]

        return True


def rebuild_forest(
    groups,
    nsafe,
    target_edges,
):

    dsu = DSU(nsafe)

    selected = []

    for gid, r in enumerate(groups):

        if dsu.union(
            r["fragment_a"],
            r["fragment_b"],
        ):

            selected.append(
                (gid, r)
            )

            if len(selected) == target_edges:
                break

    if len(selected) != target_edges:

        raise RuntimeError(
            f"Forest reconstruction failed: "
            f"{len(selected)} != {target_edges}"
        )

    return selected


def smaller_side_impact(
    nsafe,
    selected,
    fragment_weights,
    test_gid,
):

    adj = [
        []
        for _ in range(nsafe)
    ]

    edge_row = None

    for gid, r in selected:

        a = r["fragment_a"]
        b = r["fragment_b"]

        if gid == test_gid:
            edge_row = r
            continue

        adj[a].append(b)
        adj[b].append(a)

    if edge_row is None:
        raise RuntimeError(
            "Selected edge not found."
        )

    start = edge_row[
        "fragment_a"
    ]

    q = [start]

    visited = np.zeros(
        nsafe,
        dtype=bool,
    )

    visited[start] = True

    while q:

        a = q.pop()

        for b in adj[a]:

            if visited[b]:
                continue

            visited[b] = True
            q.append(b)

    side1 = int(
        fragment_weights[
            visited
        ].sum()
    )

    # The forest has 103 independent local trees,
    # so the complement of "visited" contains other
    # unrelated local components too. We therefore
    # need the original tree containing test edge.

    # Rebuild complete selected-tree adjacency.
    full_adj = [
        []
        for _ in range(nsafe)
    ]

    for gid, r in selected:

        a = r["fragment_a"]
        b = r["fragment_b"]

        full_adj[a].append(b)
        full_adj[b].append(a)

    root = edge_row[
        "fragment_a"
    ]

    component_nodes = set(
        [root]
    )

    dq = [root]

    while dq:

        a = dq.pop()

        for b in full_adj[a]:

            if b in component_nodes:
                continue

            component_nodes.add(b)
            dq.append(b)

    total_tree_weight = int(
        sum(
            int(fragment_weights[x])
            for x in component_nodes
        )
    )

    side1_in_tree = int(
        sum(
            int(fragment_weights[x])
            for x in component_nodes
            if visited[x]
        )
    )

    side2_in_tree = (
        total_tree_weight
        -
        side1_in_tree
    )

    return min(
        side1_in_tree,
        side2_in_tree,
    )


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
        / "processing"
    )

    pps = (
        root
        / "point_phase_stack"
    )

    graph = (
        root
        / "spatial_graph"
    )

    quality08i = (
        root
        / "spatial_phase_gradient_quality"
        / "per_ifg_spatial_gradient_qa.csv"
    )

    dir08l = (
        root
        / "safe_fragment_integer_quality"
    )

    dir08n = (
        root
        / "single_ifg_robust_solution"
    )

    outdir = (
        root
        / "batch_unwrap_severity_quality"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase = np.load(
        pps
        / "phase_rad.npy",
        mmap_mode="r",
    )

    rows_point = np.load(
        pps
        / "rows.npy",
        mmap_mode="r",
    )

    cols_point = np.load(
        pps
        / "cols.npy",
        mmap_mode="r",
    )

    local_u = np.load(
        graph
        / "local_u.npy",
        mmap_mode="r",
    )

    local_v = np.load(
        graph
        / "local_v.npy",
        mmap_mode="r",
    )

    npoint, ndate = phase.shape

    pairs = load_08i(
        quality08i
    )

    all_rows = []

    safe_conflict_detail = []

    print("=" * 112)
    print(
        "Full-batch severity quality"
    )
    print("=" * 112)

    print(
        f"config                     : {config_path}"
    )

    print(
        f"points                     : {npoint:,}"
    )

    print(
        f"IFGs                       : {len(pairs)}"
    )

    # Need production temporal edges.
    temporal_edges = []

    with (
        root
        / "network"
        / "network.itab"
    ).open() as f:

        for raw in f:

            x = raw.split()

            if len(x) < 2:
                continue

            temporal_edges.append(
                (
                    int(x[0]) - 1,
                    int(x[1]) - 1,
                )
            )

    if len(temporal_edges) != len(pairs):

        raise RuntimeError(
            "Temporal edge count mismatch."
        )

    for kk, pair in enumerate(
        pairs,
        start=1,
    ):

        tag = tag_of(pair)

        ml_path = (
            dir08l
            / f"{tag}_manifest.json"
        )

        mn_path = (
            dir08n
            / f"{tag}_manifest.json"
        )

        group_path = (
            dir08l
            / f"{tag}_fragment_pair_consensus.csv"
        )

        frag_path = (
            dir08l
            / f"{tag}_safe_fragment.npy"
        )

        bad_ids_path = (
            dir08l
            / f"{tag}_bad_edge_ids.npy"
        )

        if not (
            ml_path.exists()
            and
            mn_path.exists()
            and
            group_path.exists()
            and
            frag_path.exists()
        ):
            raise RuntimeError(
                f"Missing Step08l/n products: {tag}"
            )

        ml = json.loads(
            ml_path.read_text()
        )

        mn = json.loads(
            mn_path.read_text()
        )

        safe_fragment = np.load(
            frag_path,
            mmap_mode="r",
        )

        frag_weights = np.bincount(
            np.asarray(
                safe_fragment,
                dtype=np.int32,
            )
        )

        nsafe = int(
            ml[
                "topology"
            ][
                "safe_fragments"
            ]
        )

        target = int(
            ml[
                "topology"
            ][
                "required_fragment_merges"
            ]
        )

        groups = read_groups(
            group_path
        )

        selected = rebuild_forest(
            groups,
            nsafe,
            target,
        )

        weak_selected = []

        for gid, r in selected:

            if (
                (not r["exact_consensus"])
                or
                r["consensus_ratio"] < 0.75
            ):

                impact = (
                    smaller_side_impact(
                        nsafe,
                        selected,
                        frag_weights,
                        gid,
                    )
                )

                weak_selected.append(
                    (
                        gid,
                        r,
                        impact,
                    )
                )

        non_exact = sum(
            not r["exact_consensus"]
            for _, r in selected
        )

        below075 = sum(
            r["consensus_ratio"] < 0.75
            for _, r in selected
        )

        weak_max_impact = (
            max(
                x[2]
                for x in weak_selected
            )
            if weak_selected
            else 0
        )

        weak_total_edge_impact = int(
            sum(
                x[2]
                for x in weak_selected
            )
        )

        min_selected_ratio = (
            min(
                r["consensus_ratio"]
                for _, r in selected
            )
            if selected
            else 1.0
        )

        min_selected_support = (
            min(
                r["edge_count"]
                for _, r in selected
            )
            if selected
            else 0
        )

        safe_bad = int(
            ml[
                "safe_internal"
            ][
                "nonzero_integer_jump_edges"
            ]
        )

        row = {
            "pair_id":
                pair["pair_id"],

            "date1":
                pair["date1"],

            "date2":
                pair["date2"],

            "safe_fragments":
                nsafe,

            "safe_internal_bad":
                safe_bad,

            "selected_edges":
                len(selected),

            "selected_non_exact":
                non_exact,

            "selected_ratio_lt_0p75":
                below075,

            "selected_min_ratio":
                min_selected_ratio,

            "selected_min_support":
                min_selected_support,

            "weak_selected_edges":
                len(weak_selected),

            "weak_edge_max_impact_points":
                weak_max_impact,

            "weak_edge_max_impact_fraction":
                weak_max_impact
                /
                npoint,

            "weak_edge_sum_impact_points":
                weak_total_edge_impact,

            "residual_conflicts":
                int(
                    mn[
                        "residual_registration"
                    ][
                        "conflict"
                    ]
                ),

            "registered_fraction":
                float(
                    mn[
                        "residual_registration"
                    ][
                        "registered_fraction"
                    ]
                ),
        }

        all_rows.append(row)

        # ====================================================
        # For only SAFE-conflict IFGs:
        # inspect actual conflicting SAFE edges.
        # ====================================================

        if safe_bad > 0:

            if not bad_ids_path.exists():

                raise RuntimeError(
                    f"Missing bad edge IDs: {tag}"
                )

            bad_ids = np.load(
                bad_ids_path
            ).astype(
                np.int64,
                copy=False,
            )

            ti, tj = temporal_edges[
                pair["pair_id"] - 1
            ]

            ifg = np.arctan2(
                np.sin(
                    np.asarray(
                        phase[:, tj],
                        dtype=np.float32,
                    )
                    -
                    np.asarray(
                        phase[:, ti],
                        dtype=np.float32,
                    )
                ),
                np.cos(
                    np.asarray(
                        phase[:, tj],
                        dtype=np.float32,
                    )
                    -
                    np.asarray(
                        phase[:, ti],
                        dtype=np.float32,
                    )
                ),
            ).astype(
                np.float32
            )

            bu = np.asarray(
                local_u[
                    bad_ids
                ],
                dtype=np.int32,
            )

            bv = np.asarray(
                local_v[
                    bad_ids
                ],
                dtype=np.int32,
            )

            bg = np.arctan2(
                np.sin(
                    ifg[bv]
                    -
                    ifg[bu]
                ),
                np.cos(
                    ifg[bv]
                    -
                    ifg[bu]
                ),
            )

            ba = np.abs(
                bg
            )

            bad_safe_mask = (
                ba
                <=
                np.pi / 2
            )

            ids_safe = bad_ids[
                bad_safe_mask
            ]

            a_safe = ba[
                bad_safe_mask
            ]

            su = np.asarray(
                local_u[
                    ids_safe
                ],
                dtype=np.int32,
            )

            sv = np.asarray(
                local_v[
                    ids_safe
                ],
                dtype=np.int32,
            )

            endpoints = np.unique(
                np.concatenate(
                    [su, sv]
                )
            )

            rr = np.asarray(
                rows_point[
                    endpoints
                ]
            )

            cc = np.asarray(
                cols_point[
                    endpoints
                ]
            )

            # Number of connected components made only
            # by the conflicting safe edges.
            endpoint_index = {
                int(p): i
                for i, p in enumerate(
                    endpoints.tolist()
                )
            }

            dsu_bad = DSU(
                len(endpoints)
            )

            for a, b in zip(
                su.tolist(),
                sv.tolist(),
            ):

                dsu_bad.union(
                    endpoint_index[a],
                    endpoint_index[b],
                )

            n_bad_clusters = len(
                {
                    dsu_bad.find(i)
                    for i in range(
                        len(endpoints)
                    )
                }
            )

            detail = {
                "pair_id":
                    pair["pair_id"],

                "date1":
                    pair["date1"],

                "date2":
                    pair["date2"],

                "safe_bad_edges":
                    int(
                        ids_safe.size
                    ),

                "safe_bad_unique_points":
                    int(
                        endpoints.size
                    ),

                "safe_bad_clusters":
                    int(
                        n_bad_clusters
                    ),

                "abs_g_min":
                    float(
                        a_safe.min()
                    ),

                "abs_g_median":
                    float(
                        np.median(
                            a_safe
                        )
                    ),

                "abs_g_max":
                    float(
                        a_safe.max()
                    ),

                "count_abs_g_gt_1p40":
                    int(
                        np.count_nonzero(
                            a_safe
                            >
                            1.40
                        )
                    ),

                "count_abs_g_gt_1p30":
                    int(
                        np.count_nonzero(
                            a_safe
                            >
                            1.30
                        )
                    ),

                "count_abs_g_gt_1p20":
                    int(
                        np.count_nonzero(
                            a_safe
                            >
                            1.20
                        )
                    ),

                "row_min":
                    int(
                        rr.min()
                    ),

                "row_max":
                    int(
                        rr.max()
                    ),

                "col_min":
                    int(
                        cc.min()
                    ),

                "col_max":
                    int(
                        cc.max()
                    ),
            }

            safe_conflict_detail.append(
                detail
            )

        if (
            kk == 1
            or
            kk % 10 == 0
            or
            kk == len(pairs)
        ):

            print(
                f"  processed "
                f"{kk:3d}/{len(pairs):3d}"
            )

    # ========================================================
    # Global summary
    # ========================================================

    safe_problem = [
        r
        for r in all_rows
        if r[
            "safe_internal_bad"
        ] > 0
    ]

    weak_max = np.array(
        [
            r[
                "weak_edge_max_impact_points"
            ]
            for r in all_rows
        ],
        dtype=np.int64,
    )

    non_exact = np.array(
        [
            r[
                "selected_non_exact"
            ]
            for r in all_rows
        ],
        dtype=np.int32,
    )

    below = np.array(
        [
            r[
                "selected_ratio_lt_0p75"
            ]
            for r in all_rows
        ],
        dtype=np.int32,
    )

    print()
    print("=" * 112)
    print(
        "A. Superforest weak-constraint severity"
    )
    print("=" * 112)

    print(
        f"IFGs with non-exact selected edges : "
        f"{np.count_nonzero(non_exact > 0)}/108"
    )

    print(
        f"selected non-exact min/med/max     : "
        f"{non_exact.min()} / "
        f"{np.median(non_exact):.1f} / "
        f"{non_exact.max()}"
    )

    print(
        f"selected ratio<0.75 min/med/max    : "
        f"{below.min()} / "
        f"{np.median(below):.1f} / "
        f"{below.max()}"
    )

    print(
        f"max smaller-side impact "
        f"min/med/max:"
    )

    print(
        f"  {weak_max.min():,} / "
        f"{np.median(weak_max):,.1f} / "
        f"{weak_max.max():,} points"
    )

    worst = max(
        all_rows,
        key=lambda r:
            r[
                "weak_edge_max_impact_points"
            ],
    )

    print()
    print(
        "Worst weak-forest edge impact:"
    )

    print(
        f"  pair                     : "
        f"{worst['pair_id']} "
        f"{worst['date1']}->{worst['date2']}"
    )

    print(
        f"  smaller-side points      : "
        f"{worst['weak_edge_max_impact_points']:,} "
        f"({100*worst['weak_edge_max_impact_fraction']:.6f}%)"
    )

    print(
        f"  minimum selected ratio   : "
        f"{worst['selected_min_ratio']:.3f}"
    )

    print()
    print("=" * 112)
    print(
        "B. SAFE-conflict severity"
    )
    print("=" * 112)

    print(
        f"SAFE-conflict IFGs         : "
        f"{len(safe_problem)}/108"
    )

    for d in safe_conflict_detail:

        print(
            f"  pair {d['pair_id']:3d} "
            f"{d['date1']}->{d['date2']}: "
            f"bad={d['safe_bad_edges']}, "
            f"points={d['safe_bad_unique_points']}, "
            f"clusters={d['safe_bad_clusters']}, "
            f"|g| min/med/max="
            f"{d['abs_g_min']:.3f}/"
            f"{d['abs_g_median']:.3f}/"
            f"{d['abs_g_max']:.3f}"
        )

    # ========================================================
    # Save
    # ========================================================

    csv1 = (
        outdir
        / "all_ifg_forest_severity.csv"
    )

    with csv1.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                all_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            all_rows
        )

    csv2 = (
        outdir
        / "safe_conflict_detail.csv"
    )

    if safe_conflict_detail:

        with csv2.open(
            "w",
            newline="",
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=list(
                    safe_conflict_detail[
                        0
                    ].keys()
                ),
            )

            w.writeheader()
            w.writerows(
                safe_conflict_detail
            )

    manifest = {
        "format":
            "pyPSDS-GAMMA-full-batch-severity-quality-v1.0",

        "ifgs":
            len(all_rows),

        "safe_conflict_ifgs":
            len(safe_problem),

        "non_exact_selected_ifgs":
            int(
                np.count_nonzero(
                    non_exact > 0
                )
            ),

        "ratio_lt_0p75_ifgs":
            int(
                np.count_nonzero(
                    below > 0
                )
            ),

        "weak_edge_impact_points": {
            "min":
                int(
                    weak_max.min()
                ),

            "median":
                float(
                    np.median(
                        weak_max
                    )
                ),

            "max":
                int(
                    weak_max.max()
                ),
        },
    }

    json_path = (
        outdir
        / "severity_quality.json"
    )

    json_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print(
        f"forest severity CSV        : "
        f"{csv1}"
    )

    print(
        f"safe-conflict CSV          : "
        f"{csv2}"
    )

    print(
        f"manifest                   : "
        f"{json_path}"
    )

    print()
    print(
        "STEP 08q STATUS: PASS / QUALITY ONLY"
    )


if __name__ == "__main__":
    main()
