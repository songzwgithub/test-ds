#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


TWOPI = 2.0 * np.pi


def wrap32(x):
    return np.arctan2(
        np.sin(x),
        np.cos(x),
    ).astype(
        np.float32,
        copy=False,
    )


def load_itab(path: Path, ndate: int):
    edges = []

    for raw in path.read_text().splitlines():
        f = raw.split()

        if len(f) < 2:
            continue

        i = int(f[0]) - 1
        j = int(f[1]) - 1

        if not (
            0 <= i < ndate
            and
            0 <= j < ndate
        ):
            raise RuntimeError(
                f"Invalid ITAB line: {raw}"
            )

        edges.append(
            (i, j)
        )

    return edges


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


def read_group_rows(path: Path):
    rows = []

    with path.open() as f:

        for r in csv.DictReader(f):

            rows.append({
                "fragment_a":
                    int(
                        r[
                            "fragment_a"
                        ]
                    ),

                "fragment_b":
                    int(
                        r[
                            "fragment_b"
                        ]
                    ),

                "edge_count":
                    int(
                        r[
                            "edge_count"
                        ]
                    ),

                "mode_shift":
                    int(
                        r[
                            "mode_shift_b_minus_a"
                        ]
                    ),

                "mode_count":
                    int(
                        r[
                            "mode_count"
                        ]
                    ),

                "consensus_ratio":
                    float(
                        r[
                            "consensus_ratio"
                        ]
                    ),

                "exact_consensus":
                    int(
                        r[
                            "exact_consensus"
                        ]
                    ),

                "median_abs_gradient_rad":
                    float(
                        r[
                            "median_abs_gradient_rad"
                        ]
                    ),

                "min_abs_gradient_rad":
                    float(
                        r[
                            "min_abs_gradient_rad"
                        ]
                    ),

                "median_distance_m":
                    float(
                        r[
                            "median_distance_m"
                        ]
                    ),
            })

    return rows


def tree_path(
    adjacency,
    start,
    goal,
):
    """
    Return:
      [(node0, edge_id, node1), ...]
    """

    q = deque(
        [start]
    )

    prev_node = {
        start: -1
    }

    prev_edge = {}

    while q:

        u = q.popleft()

        if u == goal:
            break

        for v, eid in adjacency[u]:

            if v in prev_node:
                continue

            prev_node[v] = u
            prev_edge[v] = eid

            q.append(v)

    if goal not in prev_node:
        raise RuntimeError(
            f"No tree path "
            f"{start}->{goal}"
        )

    rev = []

    cur = goal

    while cur != start:

        p = prev_node[cur]
        eid = prev_edge[cur]

        rev.append(
            (
                p,
                eid,
                cur,
            )
        )

        cur = p

    rev.reverse()

    return rev


def signed_shift(
    row,
    u,
    v,
):
    """
    row stores:
        shift_b - shift_a = y

    Return shift_v - shift_u.
    """

    a = row[
        "fragment_a"
    ]

    b = row[
        "fragment_b"
    ]

    y = row[
        "mode_shift"
    ]

    if (
        u == a
        and
        v == b
    ):
        return y

    if (
        u == b
        and
        v == a
    ):
        return -y

    raise RuntimeError(
        "Invalid edge orientation."
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--pair-id",
        type=int,
        default=19,
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

    pps = (
        root
        / "point_phase_stack"
    )

    network = (
        root
        / "network"
    )

    graph = (
        root
        / "spatial_graph"
    )

    auditdir = (
        root
        / "safe_fragment_integer_audit"
    )

    outdir = (
        root
        / "fragment_conflict_cycle_audit"
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

    npoint, ndate = (
        phase.shape
    )

    temporal_edges = load_itab(
        network
        / "network.itab",
        ndate,
    )

    pair_id = (
        args.pair_id
    )

    if not (
        1
        <= pair_id
        <= len(
            temporal_edges
        )
    ):
        raise RuntimeError(
            "Invalid pair ID."
        )

    ti, tj = (
        temporal_edges[
            pair_id - 1
        ]
    )

    tag = (
        f"pair{pair_id:03d}_"
        f"{stack.dates[ti]}_"
        f"{stack.dates[tj]}"
    )

    safe_fragment = np.load(
        auditdir
        / f"{tag}_safe_fragment.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    U = np.load(
        auditdir
        / f"{tag}_consensus_unwrapped.npy",
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

    group_csv = (
        auditdir
        / f"{tag}_fragment_pair_consensus.csv"
    )

    groups = read_group_rows(
        group_csv
    )

    nsafe = int(
        safe_fragment.max()
    ) + 1

    # ========================================================
    # Recalculate wrapped IFG / spatial gradients
    # ========================================================

    ifg = wrap32(
        np.asarray(
            phase[:, tj],
            dtype=np.float32,
        )
        -
        np.asarray(
            phase[:, ti],
            dtype=np.float32,
        )
    )

    u = np.asarray(
        local_u,
        dtype=np.int32,
    )

    v = np.asarray(
        local_v,
        dtype=np.int32,
    )

    g = wrap32(
        ifg[v]
        -
        ifg[u]
    )

    abs_g = np.abs(g)

    safe = (
        abs_g
        <=
        np.pi / 2
    )

    unsafe = ~safe

    delta = (
        np.asarray(
            U[v],
            dtype=np.float64,
        )
        -
        np.asarray(
            U[u],
            dtype=np.float64,
        )
        -
        g.astype(
            np.float64
        )
    )

    jump = np.rint(
        delta
        /
        TWOPI
    ).astype(
        np.int32
    )

    fu = safe_fragment[u]
    fv = safe_fragment[v]

    same_fragment = (
        fu == fv
    )

    cross_fragment = (
        fu != fv
    )

    bad = (
        jump != 0
    )

    # ========================================================
    # Bad-edge decomposition
    # ========================================================

    safe_bad = int(
        np.count_nonzero(
            safe
            &
            bad
        )
    )

    unsafe_within_total = int(
        np.count_nonzero(
            unsafe
            &
            same_fragment
        )
    )

    unsafe_within_bad = int(
        np.count_nonzero(
            unsafe
            &
            same_fragment
            &
            bad
        )
    )

    unsafe_cross_total = int(
        np.count_nonzero(
            unsafe
            &
            cross_fragment
        )
    )

    unsafe_cross_bad = int(
        np.count_nonzero(
            unsafe
            &
            cross_fragment
            &
            bad
        )
    )

    total_bad = int(
        np.count_nonzero(
            bad
        )
    )

    print("=" * 96)
    print(
        "Step 08m - Fragment conflict-cycle audit"
    )
    print("=" * 96)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"pair                       : "
        f"{pair_id}/"
        f"{len(temporal_edges)}"
    )

    print(
        f"dates                      : "
        f"{stack.dates[ti]} -> "
        f"{stack.dates[tj]}"
    )

    print(
        f"safe fragments             : "
        f"{nsafe}"
    )

    print(
        f"fragment-pair constraints  : "
        f"{len(groups)}"
    )

    print()
    print("=" * 96)
    print(
        "Bad-edge decomposition"
    )
    print("=" * 96)

    print(
        f"SAFE bad edges             : "
        f"{safe_bad:,}"
    )

    print(
        f"UNSAFE within-fragment     : "
        f"{unsafe_within_bad:,}/"
        f"{unsafe_within_total:,}"
    )

    print(
        f"UNSAFE cross-fragment      : "
        f"{unsafe_cross_bad:,}/"
        f"{unsafe_cross_total:,}"
    )

    print(
        f"total bad edges            : "
        f"{total_bad:,}"
    )

    if total_bad != (
        safe_bad
        +
        unsafe_within_bad
        +
        unsafe_cross_bad
    ):
        raise RuntimeError(
            "Bad-edge decomposition mismatch."
        )

    # ========================================================
    # Rebuild the exact quality-ordered superforest
    # used by Step 08l.
    #
    # CSV is already written in the sorted quality order.
    # ========================================================

    dsu = DSU(
        nsafe
    )

    selected = []
    non_tree = []

    adjacency = [
        []
        for _ in range(
            nsafe
        )
    ]

    for eid, row in enumerate(
        groups
    ):

        a = row[
            "fragment_a"
        ]

        b = row[
            "fragment_b"
        ]

        if dsu.union(
            a,
            b,
        ):
            selected.append(
                eid
            )

            adjacency[a].append(
                (
                    b,
                    eid,
                )
            )

            adjacency[b].append(
                (
                    a,
                    eid,
                )
            )

        else:
            non_tree.append(
                eid
            )

    roots = {
        dsu.find(i)
        for i in range(
            nsafe
        )
    }

    nroots = len(
        roots
    )

    cycle_rank = (
        len(groups)
        -
        nsafe
        +
        nroots
    )

    print()
    print("=" * 96)
    print(
        "Fragment supergraph topology"
    )
    print("=" * 96)

    print(
        f"nodes                      : "
        f"{nsafe}"
    )

    print(
        f"edges                      : "
        f"{len(groups)}"
    )

    print(
        f"forest edges               : "
        f"{len(selected)}"
    )

    print(
        f"components                 : "
        f"{nroots}"
    )

    print(
        f"cycle rank                 : "
        f"{cycle_rank}"
    )

    print(
        f"non-tree edges             : "
        f"{len(non_tree)}"
    )

    if cycle_rank != len(
        non_tree
    ):
        raise RuntimeError(
            "Cycle-rank/non-tree mismatch."
        )

    # ========================================================
    # Fundamental cycle audit
    # ========================================================

    cycle_rows = []

    print()
    print("=" * 96)
    print(
        "Fundamental integer conflict cycles"
    )
    print("=" * 96)

    for cycle_id, nt_eid in enumerate(
        non_tree,
        start=1,
    ):

        nt = groups[
            nt_eid
        ]

        a = nt[
            "fragment_a"
        ]

        b = nt[
            "fragment_b"
        ]

        observed = nt[
            "mode_shift"
        ]

        path = tree_path(
            adjacency,
            a,
            b,
        )

        path_sum = 0

        cycle_edge_ids = []

        for p, eid, q in path:

            path_sum += signed_shift(
                groups[eid],
                p,
                q,
            )

            cycle_edge_ids.append(
                eid
            )

        closure_misfit = (
            path_sum
            -
            observed
        )

        cycle_edge_ids.append(
            nt_eid
        )

        print()
        print(
            f"Cycle {cycle_id}"
        )

        print(
            f"  non-tree edge          : "
            f"{nt_eid}"
        )

        print(
            f"  fragments              : "
            f"{a} -> {b}"
        )

        print(
            f"  tree-path integer      : "
            f"{path_sum}"
        )

        print(
            f"  observed integer       : "
            f"{observed}"
        )

        print(
            f"  closure misfit         : "
            f"{closure_misfit:+d}"
        )

        print(
            f"  cycle edges            : "
            f"{len(cycle_edge_ids)}"
        )

        print()
        print(
            "  eid   tree?  frag_a frag_b  "
            "shift  support  ratio    exact  "
            "med|g|   medDist"
        )

        candidate_info = []

        for order_in_cycle, eid in enumerate(
            cycle_edge_ids
        ):

            r = groups[eid]

            is_non_tree = (
                eid == nt_eid
            )

            print(
                f"  {eid:3d}   "
                f"{'NO ':>4s}   "
                if is_non_tree
                else
                f"  {eid:3d}   "
                f"{'YES':>4s}   ",
                end=""
            )

            print(
                f"{r['fragment_a']:6d} "
                f"{r['fragment_b']:6d} "
                f"{r['mode_shift']:6d} "
                f"{r['mode_count']:3d}/"
                f"{r['edge_count']:<3d} "
                f"{r['consensus_ratio']:7.3f} "
                f"{r['exact_consensus']:5d} "
                f"{r['median_abs_gradient_rad']:8.3f} "
                f"{r['median_distance_m']:8.2f}"
            )

            candidate_info.append(
                (
                    eid,
                    r,
                    is_non_tree,
                    order_in_cycle,
                )
            )

            cycle_rows.append({
                "cycle_id":
                    cycle_id,

                "closure_misfit":
                    closure_misfit,

                "edge_id":
                    eid,

                "is_non_tree":
                    int(
                        is_non_tree
                    ),

                "fragment_a":
                    r[
                        "fragment_a"
                    ],

                "fragment_b":
                    r[
                        "fragment_b"
                    ],

                "mode_shift":
                    r[
                        "mode_shift"
                    ],

                "edge_count":
                    r[
                        "edge_count"
                    ],

                "mode_count":
                    r[
                        "mode_count"
                    ],

                "consensus_ratio":
                    r[
                        "consensus_ratio"
                    ],

                "exact_consensus":
                    r[
                        "exact_consensus"
                    ],

                "median_abs_gradient_rad":
                    r[
                        "median_abs_gradient_rad"
                    ],

                "median_distance_m":
                    r[
                        "median_distance_m"
                    ],
            })

        # Pure QA heuristic only.
        #
        # Weakest:
        #  1. non-exact consensus first
        #  2. lower consensus ratio
        #  3. fewer supporting edges
        #  4. larger median gradient
        #  5. larger distance
        weakest = min(
            candidate_info,
            key=lambda x: (
                x[1][
                    "exact_consensus"
                ],
                x[1][
                    "consensus_ratio"
                ],
                x[1][
                    "edge_count"
                ],
                -x[1][
                    "median_abs_gradient_rad"
                ],
                -x[1][
                    "median_distance_m"
                ],
            ),
        )

        wr = weakest[1]

        print()
        print(
            "  weakest statistical candidate:"
        )

        print(
            f"    edge_id              = "
            f"{weakest[0]}"
        )

        print(
            f"    fragment pair        = "
            f"{wr['fragment_a']}-"
            f"{wr['fragment_b']}"
        )

        print(
            f"    consensus            = "
            f"{wr['mode_count']}/"
            f"{wr['edge_count']} "
            f"({wr['consensus_ratio']:.3f})"
        )

        print(
            f"    median |g|           = "
            f"{wr['median_abs_gradient_rad']:.3f} rad"
        )

        print(
            f"    median distance      = "
            f"{wr['median_distance_m']:.2f} m"
        )

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / f"{tag}_fundamental_cycles.csv"
    )

    if cycle_rows:

        with csv_path.open(
            "w",
            newline="",
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=list(
                    cycle_rows[0].keys()
                ),
            )

            w.writeheader()
            w.writerows(
                cycle_rows
            )

    manifest = {
        "format":
            "pyPSDS-GAMMA-fragment-conflict-cycle-audit-v0.9",

        "status":
            "AUDIT_ONLY",

        "pair": {
            "pair_id":
                pair_id,

            "date1":
                str(
                    stack.dates[ti]
                ),

            "date2":
                str(
                    stack.dates[tj]
                ),
        },

        "bad_edge_decomposition": {
            "safe_bad":
                safe_bad,

            "unsafe_within_total":
                unsafe_within_total,

            "unsafe_within_bad":
                unsafe_within_bad,

            "unsafe_cross_total":
                unsafe_cross_total,

            "unsafe_cross_bad":
                unsafe_cross_bad,

            "total_bad":
                total_bad,
        },

        "supergraph": {
            "nodes":
                nsafe,

            "edges":
                len(groups),

            "components":
                nroots,

            "forest_edges":
                len(selected),

            "cycle_rank":
                cycle_rank,

            "non_tree_edges":
                len(non_tree),
        },

        "cycles": [],
    }

    for cid in range(
        1,
        len(non_tree) + 1,
    ):

        rr = [
            r
            for r in cycle_rows
            if r[
                "cycle_id"
            ]
            ==
            cid
        ]

        manifest[
            "cycles"
        ].append({
            "cycle_id":
                cid,

            "closure_misfit":
                (
                    rr[0][
                        "closure_misfit"
                    ]
                    if rr
                    else None
                ),

            "edge_count":
                len(rr),

            "edge_ids":
                [
                    r[
                        "edge_id"
                    ]
                    for r in rr
                ],
        })

    json_path = (
        outdir
        / f"{tag}_manifest.json"
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
    print("=" * 96)

    print(
        f"cycle table                : "
        f"{csv_path}"
    )

    print(
        f"manifest                   : "
        f"{json_path}"
    )

    print()
    print(
        "STEP 08m STATUS: PASS / AUDIT ONLY"
    )

    print(
        "Do not correct integer shifts yet."
    )


if __name__ == "__main__":
    main()
