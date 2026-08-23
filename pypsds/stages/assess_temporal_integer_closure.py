#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


TWOPI = 2.0 * np.pi


# ============================================================
# Temporal network
# ============================================================

def load_itab(path: Path, ndate: int):
    edges = []

    with path.open() as f:

        for raw in f:

            x = raw.split()

            if len(x) < 2:
                continue

            i = int(x[0]) - 1
            j = int(x[1]) - 1

            if not (
                0 <= i < ndate
                and
                0 <= j < ndate
            ):
                raise RuntimeError(
                    f"Invalid network edge: {raw}"
                )

            edges.append(
                (i, j)
            )

    return edges


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


def build_temporal_cycle_basis(
    ndate,
    edges,
):
    """
    Deterministic spanning tree from production edge order.

    For every non-tree edge e=(u,v):

        U_e - sum(tree path u->v) = 0

    A cycle row contains coefficients on the temporal-network IFGs.
    """

    dsu = DSU(
        ndate
    )

    tree_ids = []
    non_tree_ids = []

    for eid, (u, v) in enumerate(edges):

        if dsu.union(u, v):
            tree_ids.append(eid)
        else:
            non_tree_ids.append(eid)

    if len(tree_ids) != ndate - 1:

        raise RuntimeError(
            f"Temporal network is not connected: "
            f"tree edges={len(tree_ids)}, "
            f"expected={ndate-1}"
        )

    # Tree adjacency:
    # (neighbor acquisition, edge id, sign)
    #
    # sign is coefficient when traversing current -> neighbor:
    #
    # production IFG:
    #     Phi_e = Theta_v - Theta_u
    #
    # traversal u->v : +Phi_e
    # traversal v->u : -Phi_e
    adj = [
        []
        for _ in range(ndate)
    ]

    for eid in tree_ids:

        u, v = edges[eid]

        adj[u].append(
            (v, eid, +1)
        )

        adj[v].append(
            (u, eid, -1)
        )

    def find_path(start, goal):

        q = deque(
            [start]
        )

        prev = {
            start: None
        }

        while q:

            a = q.popleft()

            if a == goal:
                break

            for b, eid, sign in adj[a]:

                if b in prev:
                    continue

                prev[b] = (
                    a,
                    eid,
                    sign,
                )

                q.append(b)

        if goal not in prev:
            raise RuntimeError(
                f"No temporal tree path "
                f"{start}->{goal}"
            )

        rev = []

        cur = goal

        while cur != start:

            parent, eid, sign_parent_to_cur = (
                prev[cur]
            )

            rev.append(
                (
                    eid,
                    sign_parent_to_cur,
                )
            )

            cur = parent

        rev.reverse()

        return rev

    nedge = len(edges)

    C = np.zeros(
        (
            len(non_tree_ids),
            nedge,
        ),
        dtype=np.int8,
    )

    cycle_meta = []

    for cid, eid in enumerate(
        non_tree_ids
    ):

        u, v = edges[eid]

        # Non-tree IFG:
        # Phi_e = Theta_v - Theta_u
        #
        # path u->v gives same quantity, therefore:
        #
        # Phi_e - path = 0
        C[cid, eid] = +1

        path = find_path(
            u,
            v,
        )

        for peid, traversal_sign in path:

            C[
                cid,
                peid
            ] -= traversal_sign

        used = np.where(
            C[cid] != 0
        )[0]

        cycle_meta.append({
            "cycle_id":
                cid + 1,

            "non_tree_edge_id":
                eid + 1,

            "start_acquisition":
                u + 1,

            "end_acquisition":
                v + 1,

            "cycle_edge_count":
                int(
                    used.size
                ),

            "edge_ids":
                ",".join(
                    str(
                        int(x) + 1
                    )
                    for x in used
                ),
        })

    rank = np.linalg.matrix_rank(
        C.astype(
            np.float64
        )
    )

    return (
        C,
        tree_ids,
        non_tree_ids,
        cycle_meta,
        rank,
    )


def integer_mode(x):
    values, counts = np.unique(
        x,
        return_counts=True,
    )

    k = int(
        np.argmax(counts)
    )

    return (
        int(values[k]),
        int(counts[k]),
    )


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
        "--batch-size",
        type=int,
        default=12000,
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

    network_dir = (
        root
        / "network"
    )

    unwrap_dir = (
        root
        / "single_ifg_robust_solution"
    )

    policy_dir = (
        root
        / "unwrap_component_policy"
    )

    pps_dir = (
        root
        / "point_phase_stack"
    )

    qualityunwrap_conflict_quality_dir = (
        root
        / "safe_conflict_acquisition_quality"
    )

    outdir = (
        root
        / "temporal_integer_closure_quality"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_pt = np.load(
        pps_dir
        / "rows.npy",
        mmap_mode="r",
    )

    cols_pt = np.load(
        pps_dir
        / "cols.npy",
        mmap_mode="r",
    )

    npoint = rows_pt.size

    ndate = len(
        stack.dates
    )

    temporal_edges = load_itab(
        network_dir
        / "network.itab",
        ndate,
    )

    nedge = len(
        temporal_edges
    )

    (
        C,
        tree_ids,
        non_tree_ids,
        cycle_meta,
        cycle_rank,
    ) = build_temporal_cycle_basis(
        ndate,
        temporal_edges,
    )

    ncycle = C.shape[0]

    expected_cycle_rank = (
        nedge
        -
        ndate
        +
        1
    )

    if (
        ncycle
        != expected_cycle_rank
        or
        cycle_rank
        != expected_cycle_rank
    ):

        raise RuntimeError(
            "Temporal cycle basis rank mismatch: "
            f"cycles={ncycle}, "
            f"rank={cycle_rank}, "
            f"expected={expected_cycle_rank}"
        )

    # ========================================================
    # Main spatial component only
    # ========================================================

    main_mask = np.load(
        policy_dir
        / "main_component_mask.npy",
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    if main_mask.size != npoint:

        raise RuntimeError(
            "main component mask length mismatch"
        )

    main_ids = np.where(
        main_mask
    )[0].astype(
        np.int32
    )

    nmain = main_ids.size

    # ========================================================
    # Open all temporal-network unwrapped IFGs
    # ========================================================

    phase_files = []

    phase_maps = []

    for pair_id, (i, j) in enumerate(
        temporal_edges,
        start=1,
    ):

        tag = (
            f"pair{pair_id:03d}_"
            f"{stack.dates[i]}_"
            f"{stack.dates[j]}"
        )

        path = (
            unwrap_dir
            / (
                f"{tag}_"
                "unwrapped_phase_rad.npy"
            )
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        arr = np.load(
            path,
            mmap_mode="r",
        )

        if arr.size != npoint:

            raise RuntimeError(
                f"{path.name}: "
                f"point count mismatch"
            )

        phase_files.append(
            path
        )

        phase_maps.append(
            arr
        )

    print("=" * 96)
    print(
        "Temporal integer-closure quality "
        "of all unwrapped IFGs"
    )
    print("=" * 96)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"acquisitions               : "
        f"{ndate}"
    )

    print(
        f"IFGs                       : "
        f"{nedge}"
    )

    print(
        f"temporal tree edges        : "
        f"{len(tree_ids)}"
    )

    print(
        f"fundamental cycles         : "
        f"{ncycle}"
    )

    print(
        f"cycle basis rank           : "
        f"{cycle_rank}"
    )

    print(
        f"main spatial points        : "
        f"{nmain:,}/"
        f"{npoint:,} "
        f"({100*nmain/npoint:.4f}%)"
    )

    print(
        f"batch size                 : "
        f"{args.batch_size:,}"
    )

    # ========================================================
    # Pass 1:
    # determine modal temporal integer for every cycle.
    #
    # We cannot assume modal integer = 0 because each
    # independently spatially-unwrapped IFG may differ by
    # a global 2pi constant.
    # ========================================================

    # Closure integers should be small. Use dictionaries
    # instead of assuming a fixed integer range.
    hist = [
        {}
        for _ in range(
            ncycle
        )
    ]

    residual_max = np.zeros(
        ncycle,
        dtype=np.float64,
    )

    print()
    print(
        "Pass 1/2: estimating cycle-wise "
        "global integer modes ..."
    )

    for b0 in range(
        0,
        nmain,
        args.batch_size,
    ):

        b1 = min(
            b0
            +
            args.batch_size,
            nmain,
        )

        ids = main_ids[
            b0:b1
        ]

        B = ids.size

        # [B,nedge]
        X = np.empty(
            (
                B,
                nedge,
            ),
            dtype=np.float64,
        )

        for e in range(
            nedge
        ):

            X[:, e] = np.asarray(
                phase_maps[e][
                    ids
                ],
                dtype=np.float64,
            )

        # [B,71]
        closure = (
            X
            @
            C.T.astype(
                np.float64,
                copy=False,
            )
        )

        k = np.rint(
            closure
            /
            TWOPI
        ).astype(
            np.int32
        )

        residual = (
            closure
            -
            TWOPI
            *
            k
        )

        residual_max = np.maximum(
            residual_max,
            np.max(
                np.abs(
                    residual
                ),
                axis=0,
            ),
        )

        for c in range(
            ncycle
        ):

            values, counts = np.unique(
                k[:, c],
                return_counts=True,
            )

            h = hist[c]

            for value, count in zip(
                values.tolist(),
                counts.tolist(),
            ):

                h[value] = (
                    h.get(
                        value,
                        0
                    )
                    +
                    count
                )

        if (
            b0 == 0
            or
            b1 == nmain
            or
            (
                b1
                //
                args.batch_size
            )
            % 10
            ==
            0
        ):

            print(
                f"  {b1:,}/"
                f"{nmain:,}"
            )

    cycle_mode = np.zeros(
        ncycle,
        dtype=np.int32,
    )

    cycle_mode_count = np.zeros(
        ncycle,
        dtype=np.int64,
    )

    for c in range(
        ncycle
    ):

        if not hist[c]:

            raise RuntimeError(
                f"Empty histogram for cycle {c+1}"
            )

        mode_value, mode_count = max(
            hist[c].items(),
            key=lambda kv:
                kv[1],
        )

        cycle_mode[c] = int(
            mode_value
        )

        cycle_mode_count[c] = int(
            mode_count
        )

    # ========================================================
    # Pass 2:
    # spatial deviations from each cycle's global mode.
    # ========================================================

    print()
    print(
        "Pass 2/2: qualitying spatially varying "
        "temporal integer closure ..."
    )

    point_bad_count_main = np.zeros(
        nmain,
        dtype=np.uint16,
    )

    cycle_bad_count = np.zeros(
        ncycle,
        dtype=np.int64,
    )

    cycle_max_abs_integer_deviation = np.zeros(
        ncycle,
        dtype=np.int32,
    )

    total_bad_occurrences = 0

    for b0 in range(
        0,
        nmain,
        args.batch_size,
    ):

        b1 = min(
            b0
            +
            args.batch_size,
            nmain,
        )

        ids = main_ids[
            b0:b1
        ]

        B = ids.size

        X = np.empty(
            (
                B,
                nedge,
            ),
            dtype=np.float64,
        )

        for e in range(
            nedge
        ):

            X[:, e] = np.asarray(
                phase_maps[e][
                    ids
                ],
                dtype=np.float64,
            )

        closure = (
            X
            @
            C.T.astype(
                np.float64,
                copy=False,
            )
        )

        k = np.rint(
            closure
            /
            TWOPI
        ).astype(
            np.int32
        )

        deviation = (
            k
            -
            cycle_mode[
                None,
                :
            ]
        )

        bad = (
            deviation
            !=
            0
        )

        nb = np.sum(
            bad,
            axis=1,
            dtype=np.uint16,
        )

        point_bad_count_main[
            b0:b1
        ] = nb

        cb = np.sum(
            bad,
            axis=0,
            dtype=np.int64,
        )

        cycle_bad_count += cb

        total_bad_occurrences += int(
            cb.sum()
        )

        cycle_max_abs_integer_deviation = (
            np.maximum(
                cycle_max_abs_integer_deviation,
                np.max(
                    np.abs(
                        deviation
                    ),
                    axis=0,
                ),
            )
        )

        if (
            b0 == 0
            or
            b1 == nmain
            or
            (
                b1
                //
                args.batch_size
            )
            % 10
            ==
            0
        ):

            print(
                f"  {b1:,}/"
                f"{nmain:,}"
            )

    # ========================================================
    # Cycle summary
    # ========================================================

    cycle_rows = []

    for c in range(
        ncycle
    ):

        row = dict(
            cycle_meta[c]
        )

        row.update({
            "modal_integer":
                int(
                    cycle_mode[c]
                ),

            "modal_count":
                int(
                    cycle_mode_count[c]
                ),

            "modal_fraction":
                float(
                    cycle_mode_count[c]
                    /
                    nmain
                ),

            "spatial_bad_points":
                int(
                    cycle_bad_count[c]
                ),

            "spatial_bad_fraction":
                float(
                    cycle_bad_count[c]
                    /
                    nmain
                ),

            "max_abs_integer_deviation":
                int(
                    cycle_max_abs_integer_deviation[
                        c
                    ]
                ),

            "float_residual_max_rad":
                float(
                    residual_max[c]
                ),
        })

        cycle_rows.append(
            row
        )

    # ========================================================
    # Point summary
    # ========================================================

    bad_main = (
        point_bad_count_main
        >
        0
    )

    bad_main_ids = main_ids[
        bad_main
    ]

    unique_bad_points = int(
        bad_main_ids.size
    )

    point_bad_full = np.zeros(
        npoint,
        dtype=np.uint16,
    )

    point_bad_full[
        main_ids
    ] = point_bad_count_main

    # Recurrence distribution.
    print()
    print("=" * 96)
    print(
        "Temporal closure summary"
    )
    print("=" * 96)

    print(
        f"fundamental cycles         : "
        f"{ncycle}"
    )

    cycles_bad = int(
        np.count_nonzero(
            cycle_bad_count
            >
            0
        )
    )

    print(
        f"cycles with spatial "
        f"integer variation : "
        f"{cycles_bad}/{ncycle}"
    )

    print(
        f"cycle-point bad occurrences: "
        f"{total_bad_occurrences:,}"
    )

    print(
        f"unique bad main points     : "
        f"{unique_bad_points:,}/"
        f"{nmain:,} "
        f"({100*unique_bad_points/nmain:.8f}%)"
    )

    if unique_bad_points:

        vals = point_bad_count_main[
            bad_main
        ]

        print(
            f"bad-cycle count per affected "
            f"point min/med/max:"
        )

        print(
            f"  {vals.min()} / "
            f"{np.median(vals):.1f} / "
            f"{vals.max()}"
        )

        for threshold in (
            1,
            2,
            3,
            5,
            10,
            20,
            30,
            50,
        ):

            n = int(
                np.count_nonzero(
                    vals
                    >=
                    threshold
                )
            )

            print(
                f"points bad in >= "
                f"{threshold:2d} cycles : "
                f"{n}"
            )

    # ========================================================
    # Worst cycles
    # ========================================================

    worst_cycles = sorted(
        cycle_rows,
        key=lambda r:
            r[
                "spatial_bad_points"
            ],
        reverse=True,
    )

    print()
    print(
        "Worst temporal cycles:"
    )

    for r in worst_cycles[:12]:

        if (
            r[
                "spatial_bad_points"
            ]
            ==
            0
        ):
            break

        print(
            f"  cycle "
            f"{r['cycle_id']:2d}: "
            f"edges="
            f"{r['cycle_edge_count']:2d}, "
            f"mode="
            f"{r['modal_integer']:+d}, "
            f"bad="
            f"{r['spatial_bad_points']:6d} "
            f"({100*r['spatial_bad_fraction']:.6f}%), "
            f"max|dk|="
            f"{r['max_abs_integer_deviation']}"
        )

    # ========================================================
    # Compare with Stepunwrap_conflict_quality suspicious points
    # ========================================================

    overlap_info = None

    suspicious_edge_path = (
        qualityunwrap_conflict_quality_dir
        / "suspicious_edge_ids.npy"
    )

    if suspicious_edge_path.exists():

        suspicious_edge_ids = np.load(
            suspicious_edge_path
        ).astype(
            np.int64,
            copy=False,
        )

        graph_dir = (
            root
            / "spatial_graph"
        )

        local_u = np.load(
            graph_dir
            / "local_u.npy",
            mmap_mode="r",
        )

        local_v = np.load(
            graph_dir
            / "local_v.npy",
            mmap_mode="r",
        )

        suspicious_points = np.unique(
            np.concatenate(
                [
                    np.asarray(
                        local_u[
                            suspicious_edge_ids
                        ],
                        dtype=np.int32,
                    ),

                    np.asarray(
                        local_v[
                            suspicious_edge_ids
                        ],
                        dtype=np.int32,
                    ),
                ]
            )
        )

        overlap = np.intersect1d(
            bad_main_ids,
            suspicious_points,
            assume_unique=False,
        )

        overlap_info = {
            "stepunwrap_conflict_quality_suspicious_points":
                int(
                    suspicious_points.size
                ),

            "temporal_bad_main_points":
                int(
                    bad_main_ids.size
                ),

            "overlap_points":
                int(
                    overlap.size
                ),

            "fraction_of_temporal_bad_overlapping_unwrap_conflict_quality":
                (
                    float(
                        overlap.size
                        /
                        bad_main_ids.size
                    )
                    if bad_main_ids.size
                    else 0.0
                ),

            "fraction_of_unwrap_conflict_quality_points_temporally_bad":
                (
                    float(
                        overlap.size
                        /
                        suspicious_points.size
                    )
                    if suspicious_points.size
                    else 0.0
                ),
        }

        print()
        print("=" * 96)
        print(
            "Overlap with Stepunwrap_conflict_quality spatial "
            "SAFE-conflict points"
        )
        print("=" * 96)

        print(
            f"unwrap conflict quality suspicious points      : "
            f"{suspicious_points.size}"
        )

        print(
            f"temporal bad main points   : "
            f"{bad_main_ids.size}"
        )

        print(
            f"overlap                    : "
            f"{overlap.size}"
        )

        if bad_main_ids.size:

            print(
                f"temporal bad -> unwrap conflict quality overlap: "
                f"{100*overlap.size/bad_main_ids.size:.3f}%"
            )

    # ========================================================
    # Save point coordinates for bad points
    # ========================================================

    point_rows = []

    for pid in bad_main_ids.tolist():

        point_rows.append({
            "point_id":
                int(pid),

            "row":
                int(
                    rows_pt[
                        pid
                    ]
                ),

            "col":
                int(
                    cols_pt[
                        pid
                    ]
                ),

            "bad_cycle_count":
                int(
                    point_bad_full[
                        pid
                    ]
                ),
        })

    cycle_csv = (
        outdir
        / "temporal_cycle_qa.csv"
    )

    with cycle_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                cycle_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            cycle_rows
        )

    point_csv = (
        outdir
        / "temporal_bad_points.csv"
    )

    if point_rows:

        with point_csv.open(
            "w",
            newline="",
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=list(
                    point_rows[
                        0
                    ].keys()
                ),
            )

            w.writeheader()
            w.writerows(
                point_rows
            )

    np.save(
        outdir
        / "point_temporal_bad_cycle_count.npy",
        point_bad_full,
    )

    np.save(
        outdir
        / "cycle_modal_integer.npy",
        cycle_mode,
    )

    np.save(
        outdir
        / "cycle_matrix.npy",
        C,
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-temporal-integer-closure-quality-v1.0",

        "status":
            "QUALITY_ONLY",

        "acquisitions":
            ndate,

        "ifgs":
            nedge,

        "temporal_tree_edges":
            len(
                tree_ids
            ),

        "fundamental_cycles":
            ncycle,

        "cycle_basis_rank":
            int(
                cycle_rank
            ),

        "main_points":
            int(
                nmain
            ),

        "cycles_with_spatial_integer_variation":
            cycles_bad,

        "cycle_point_bad_occurrences":
            int(
                total_bad_occurrences
            ),

        "unique_bad_main_points":
            unique_bad_points,

        "bad_main_fraction":
            float(
                unique_bad_points
                /
                nmain
            ),

        "maximum_cycle_float_residual_rad":
            float(
                residual_max.max()
            ),

        "overlap_with_unwrap_conflict_quality":
            overlap_info,
    }

    json_path = (
        outdir
        / "temporal_integer_closure_quality.json"
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
        f"cycle QA CSV               : "
        f"{cycle_csv}"
    )

    print(
        f"bad-point CSV              : "
        f"{point_csv}"
    )

    print(
        f"manifest                   : "
        f"{json_path}"
    )

    print()
    print(
        "STEP temporal_closure STATUS: PASS / QUALITY ONLY"
    )


if __name__ == "__main__":
    main()
