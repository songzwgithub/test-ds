#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

from pypsds.prototype import open_from_config


TWOPI = 2.0 * np.pi


def wrap64(x):
    return np.arctan2(
        np.sin(x),
        np.cos(x),
    )


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
                f"Invalid network ITAB line: {raw}"
            )

        edges.append((i, j))

    return edges


def choose_pair(
    audit_csv: Path,
    requested_pair_id: int,
):
    rows = []

    with audit_csv.open() as f:
        reader = csv.DictReader(f)

        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError(
            "Empty 08i per-IFG QA table."
        )

    if requested_pair_id > 0:

        for r in rows:
            if int(r["pair_id"]) == requested_pair_id:
                return r

        raise RuntimeError(
            f"pair_id={requested_pair_id} "
            "not found in 08i table."
        )

    # Worst IFG:
    # primary = fraction > pi/2
    # secondary = p95
    return max(
        rows,
        key=lambda r: (
            float(
                r["local_frac_gt_pi_2"]
            ),
            float(
                r["local_p95_abs_rad"]
            ),
        ),
    )


@njit(cache=True)
def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[
            parent[x]
        ]
        x = parent[x]

    return x


@njit(cache=True)
def roots_from_edges(
    n,
    u,
    v,
):
    parent = np.arange(
        n,
        dtype=np.int32,
    )

    size = np.ones(
        n,
        dtype=np.int32,
    )

    for k in range(u.size):

        a = uf_find(
            parent,
            u[k],
        )

        b = uf_find(
            parent,
            v[k],
        )

        if a == b:
            continue

        if size[a] < size[b]:
            a, b = b, a

        parent[b] = a
        size[a] += size[b]

    roots = np.empty(
        n,
        dtype=np.int32,
    )

    for i in range(n):
        roots[i] = uf_find(
            parent,
            i,
        )

    return roots


@njit(cache=True)
def kruskal_select(
    n,
    u,
    v,
    order,
    target_edges,
):
    parent = np.arange(
        n,
        dtype=np.int32,
    )

    size = np.ones(
        n,
        dtype=np.int32,
    )

    selected = np.empty(
        target_edges,
        dtype=np.int64,
    )

    count = 0

    for z in range(order.size):

        eid = order[z]

        a = uf_find(
            parent,
            u[eid],
        )

        b = uf_find(
            parent,
            v[eid],
        )

        if a == b:
            continue

        if size[a] < size[b]:
            a, b = b, a

        parent[b] = a
        size[a] += size[b]

        selected[count] = eid
        count += 1

        if count == target_edges:
            break

    return selected[:count]


@njit(cache=True)
def build_tree_csr(
    n,
    u,
    v,
    g,
):
    m = u.size

    deg = np.zeros(
        n,
        dtype=np.int32,
    )

    for k in range(m):
        deg[u[k]] += 1
        deg[v[k]] += 1

    indptr = np.zeros(
        n + 1,
        dtype=np.int64,
    )

    for i in range(n):
        indptr[i + 1] = (
            indptr[i]
            + deg[i]
        )

    indices = np.empty(
        2 * m,
        dtype=np.int32,
    )

    values = np.empty(
        2 * m,
        dtype=np.float32,
    )

    cursor = indptr[:-1].copy()

    for k in range(m):

        a = u[k]
        b = v[k]
        gg = g[k]

        p = cursor[a]
        indices[p] = b
        values[p] = gg
        cursor[a] += 1

        p = cursor[b]
        indices[p] = a
        values[p] = -gg
        cursor[b] += 1

    return (
        indptr,
        indices,
        values,
        deg,
    )


@njit(cache=True)
def unwrap_forest(
    wrapped_phase,
    indptr,
    indices,
    values,
):
    n = wrapped_phase.size

    unwrapped = np.empty(
        n,
        dtype=np.float64,
    )

    visited = np.zeros(
        n,
        dtype=np.uint8,
    )

    queue = np.empty(
        n,
        dtype=np.int32,
    )

    roots = np.empty(
        n,
        dtype=np.int32,
    )

    nroot = 0

    for root in range(n):

        if visited[root]:
            continue

        roots[nroot] = root
        nroot += 1

        visited[root] = 1

        unwrapped[root] = float(
            wrapped_phase[root]
        )

        head = 0
        tail = 1

        queue[0] = root

        while head < tail:

            a = queue[head]
            head += 1

            for z in range(
                indptr[a],
                indptr[a + 1],
            ):

                b = indices[z]

                if visited[b]:
                    continue

                visited[b] = 1

                unwrapped[b] = (
                    unwrapped[a]
                    +
                    float(values[z])
                )

                queue[tail] = b
                tail += 1

    return (
        unwrapped,
        roots[:nroot],
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
        default=0,
        help=(
            "1-based production pair ID. "
            "0 = automatically use worst IFG from Step 08i."
        ),
    )

    ap.add_argument(
        "--distance-weight",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--edge-batch",
        type=int,
        default=250000,
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

    outroot = (
        Path(paths.output_dir)
        / "v09"
    )

    pps_dir = (
        outroot
        / "point_phase_stack"
    )

    netdir = (
        outroot
        / "network"
    )

    graphdir = (
        outroot
        / "spatial_graph"
    )

    policydir = (
        outroot
        / "unwrap_component_policy"
    )

    audit08i = (
        outroot
        / "spatial_phase_gradient_audit"
        / "per_ifg_spatial_gradient_qa.csv"
    )

    outdir = (
        outroot
        / "single_ifg_unwrap_prototype"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load PointPhaseStack / time network
    # ========================================================

    phase = np.load(
        pps_dir
        / "phase_rad.npy",
        mmap_mode="r",
    )

    npoint, ndate = phase.shape

    time_edges = load_itab(
        netdir
        / "network.itab",
        ndate,
    )

    pair_row = choose_pair(
        audit08i,
        args.pair_id,
    )

    pair_id = int(
        pair_row["pair_id"]
    )

    i, j = time_edges[
        pair_id - 1
    ]

    # ========================================================
    # Spatial graph
    # ========================================================

    local_u = np.load(
        graphdir
        / "local_u.npy",
        mmap_mode="r",
    )

    local_v = np.load(
        graphdir
        / "local_v.npy",
        mmap_mode="r",
    )

    local_dist = np.load(
        graphdir
        / "local_distance_m.npy",
        mmap_mode="r",
    )

    component = np.load(
        policydir
        / "local_component.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    point_tier = np.load(
        policydir
        / "point_unwrap_tier.npy",
    ).astype(
        np.uint8,
        copy=False,
    )

    anchor_component = np.load(
        policydir
        / "anchor_component.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    anchor_u = np.load(
        graphdir
        / "anchor_u.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    anchor_v = np.load(
        graphdir
        / "anchor_v.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    anchor_class = np.load(
        graphdir
        / "anchor_class.npy",
    ).astype(
        np.uint8,
        copy=False,
    )

    if not (
        local_u.size
        ==
        local_v.size
        ==
        local_dist.size
    ):
        raise RuntimeError(
            "Local spatial-edge arrays mismatch."
        )

    nedge = int(
        local_u.size
    )

    comp_ids, comp_counts = np.unique(
        component,
        return_counts=True,
    )

    ncomp = int(
        comp_ids.size
    )

    main_component = int(
        comp_ids[
            np.argmax(
                comp_counts
            )
        ]
    )

    main_points = int(
        comp_counts.max()
    )

    # ========================================================
    # Virtual IFG
    # ========================================================

    ifg = wrap32(
        np.asarray(
            phase[:, j],
            dtype=np.float32,
        )
        -
        np.asarray(
            phase[:, i],
            dtype=np.float32,
        )
    )

    # All local wrapped gradients.
    g = wrap32(
        ifg[
            np.asarray(local_v)
        ]
        -
        ifg[
            np.asarray(local_u)
        ]
    )

    abs_g = np.abs(g)

    print("=" * 92)
    print(
        "Step 08k - Single-IFG spatial unwrapping prototype"
    )
    print("=" * 92)

    print(
        f"config                    : {config_path}"
    )

    print(
        f"selected pair             : "
        f"{pair_id}/"
        f"{len(time_edges)}"
    )

    print(
        f"dates                     : "
        f"{stack.dates[i]} -> "
        f"{stack.dates[j]}"
    )

    print(
        f"selection mode            : "
        f"{'worst Step08i IFG' if args.pair_id == 0 else 'explicit'}"
    )

    print(
        f"08i local >pi/2           : "
        f"{100*float(pair_row['local_frac_gt_pi_2']):.4f}%"
    )

    print(
        f"08i local p95             : "
        f"{float(pair_row['local_p95_abs_rad']):.4f} rad"
    )

    print(
        f"points                    : "
        f"{npoint:,}"
    )

    print(
        f"local components          : "
        f"{ncomp}"
    )

    print(
        f"main component points     : "
        f"{main_points:,}"
    )

    print(
        f"local edges               : "
        f"{nedge:,}"
    )

    # ========================================================
    # Check whether edges <= pi/2 alone preserve all
    # R4-K8 local components.
    # ========================================================

    safe = (
        abs_g <= np.pi / 2
    )

    safe_count = int(
        np.count_nonzero(
            safe
        )
    )

    safe_u = np.asarray(
        local_u[safe],
        dtype=np.int32,
    )

    safe_v = np.asarray(
        local_v[safe],
        dtype=np.int32,
    )

    safe_roots = roots_from_edges(
        npoint,
        safe_u,
        safe_v,
    )

    safe_components = int(
        np.unique(
            safe_roots
        ).size
    )

    safe_preserves_partition = (
        safe_components == ncomp
    )

    del safe_roots

    print()
    print("=" * 92)
    print(
        "Safe-edge connectivity"
    )
    print("=" * 92)

    print(
        f"|g| <= pi/2 edges        : "
        f"{safe_count:,}/"
        f"{nedge:,} "
        f"({100*safe_count/nedge:.4f}%)"
    )

    print(
        f"components with safe edges: "
        f"{safe_components}"
    )

    print(
        f"preserves local partition : "
        f"{safe_preserves_partition}"
    )

    # ========================================================
    # Quality-guided minimum spanning forest
    #
    # Prefer:
    #   1. small wrapped spatial gradient
    #   2. short physical edge
    #
    # If <=pi/2 edges preserve all 103 components,
    # tree construction uses ONLY those safe edges.
    # ========================================================

    dmax = float(
        np.max(
            local_dist
        )
    )

    cost = (
        abs_g.astype(
            np.float32
        )
        /
        np.float32(np.pi)
        +
        np.float32(
            args.distance_weight
        )
        *
        (
            np.asarray(
                local_dist,
                dtype=np.float32,
            )
            /
            np.float32(dmax)
        )
    )

    if safe_preserves_partition:

        candidate_ids = np.where(
            safe
        )[0]

        local_order = np.argsort(
            cost[
                candidate_ids
            ],
            kind="stable",
        )

        order = candidate_ids[
            local_order
        ]

    else:

        # Very strong penalty for unsafe edges.
        cost2 = cost.copy()

        cost2[
            ~safe
        ] += 10.0

        order = np.argsort(
            cost2,
            kind="stable",
        )

        del cost2

    target_tree_edges = (
        npoint
        -
        ncomp
    )

    selected = kruskal_select(
        npoint,
        np.asarray(
            local_u,
            dtype=np.int32,
        ),
        np.asarray(
            local_v,
            dtype=np.int32,
        ),
        order.astype(
            np.int64,
            copy=False,
        ),
        target_tree_edges,
    )

    if selected.size != target_tree_edges:

        raise RuntimeError(
            f"Spanning forest incomplete: "
            f"{selected.size} edges, "
            f"expected {target_tree_edges}."
        )

    tree_u = np.asarray(
        local_u[
            selected
        ],
        dtype=np.int32,
    )

    tree_v = np.asarray(
        local_v[
            selected
        ],
        dtype=np.int32,
    )

    tree_g = g[
        selected
    ].astype(
        np.float32,
        copy=False,
    )

    tree_dist = np.asarray(
        local_dist[
            selected
        ],
        dtype=np.float32,
    )

    # ========================================================
    # Integrate tree gradients independently in every
    # local component.
    # ========================================================

    (
        indptr,
        indices,
        values,
        tree_degree,
    ) = build_tree_csr(
        npoint,
        tree_u,
        tree_v,
        tree_g,
    )

    (
        unwrapped,
        forest_roots,
    ) = unwrap_forest(
        ifg,
        indptr,
        indices,
        values,
    )

    if forest_roots.size != ncomp:

        raise RuntimeError(
            f"Forest roots={forest_roots.size}; "
            f"expected {ncomp}."
        )

    # Every solution must wrap exactly back to input.
    modulo_error = np.abs(
        wrap64(
            unwrapped
            -
            ifg.astype(
                np.float64
            )
        )
    )

    modulo_max = float(
        modulo_error.max()
    )

    if modulo_max > 1e-4:

        raise RuntimeError(
            f"Unwrapped->wrapped parity failed: "
            f"{modulo_max:.3e} rad"
        )

    # ========================================================
    # Tree quality
    # ========================================================

    tree_abs_g = np.abs(
        tree_g
    )

    tq = np.quantile(
        tree_abs_g,
        [
            0.50,
            0.90,
            0.95,
            0.99,
            1.00,
        ],
    )

    td = np.quantile(
        tree_dist,
        [
            0.50,
            0.90,
            0.95,
            0.99,
            1.00,
        ],
    )

    tree_gt_pi2 = int(
        np.count_nonzero(
            tree_abs_g
            >
            np.pi / 2
        )
    )

    print()
    print("=" * 92)
    print(
        "Quality-guided spanning forest"
    )
    print("=" * 92)

    print(
        f"tree edges                : "
        f"{selected.size:,}"
    )

    print(
        f"forest roots              : "
        f"{forest_roots.size}"
    )

    print(
        f"tree |g| median/p90/p95/"
        f"p99/max:"
    )

    print(
        "  "
        + " / ".join(
            f"{x:.4f}"
            for x in tq
        )
        + " rad"
    )

    print(
        f"tree edges > pi/2         : "
        f"{tree_gt_pi2:,}"
    )

    print(
        f"tree distance "
        f"median/p90/p95/p99/max:"
    )

    print(
        "  "
        + " / ".join(
            f"{x:.2f}"
            for x in td
        )
        + " m"
    )

    print(
        f"wrap-back max error       : "
        f"{modulo_max:.3e} rad"
    )

    # ========================================================
    # ALL non-tree edges:
    #
    # k_e = round(
    #   (U_v - U_u - wrapped_gradient) / 2pi
    # )
    #
    # k_e != 0 means the tree solution produces an
    # integer 2pi discontinuity across that short edge.
    # ========================================================

    is_tree = np.zeros(
        nedge,
        dtype=bool,
    )

    is_tree[
        selected
    ] = True

    bad_ids_parts = []
    bad_jump_parts = []

    residual_max = 0.0

    jump_nonzero = 0
    jump_abs1 = 0
    jump_abs2plus = 0

    non_tree_count = int(
        nedge
        -
        selected.size
    )

    for e0 in range(
        0,
        nedge,
        args.edge_batch,
    ):

        e1 = min(
            e0
            +
            args.edge_batch,
            nedge,
        )

        u = np.asarray(
            local_u[
                e0:e1
            ],
            dtype=np.int32,
        )

        v = np.asarray(
            local_v[
                e0:e1
            ],
            dtype=np.int32,
        )

        delta = (
            unwrapped[v]
            -
            unwrapped[u]
            -
            g[
                e0:e1
            ].astype(
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

        residual = (
            delta
            -
            TWOPI
            *
            jump
        )

        residual_max = max(
            residual_max,
            float(
                np.max(
                    np.abs(
                        residual
                    )
                )
            ),
        )

        nt = ~is_tree[
            e0:e1
        ]

        bad = (
            nt
            &
            (
                jump != 0
            )
        )

        jb = jump[
            bad
        ]

        jump_nonzero += int(
            jb.size
        )

        if jb.size:

            jump_abs1 += int(
                np.count_nonzero(
                    np.abs(jb)
                    == 1
                )
            )

            jump_abs2plus += int(
                np.count_nonzero(
                    np.abs(jb)
                    >= 2
                )
            )

            ids = (
                np.where(
                    bad
                )[0]
                +
                e0
            ).astype(
                np.int64
            )

            bad_ids_parts.append(
                ids
            )

            bad_jump_parts.append(
                jb.astype(
                    np.int16
                )
            )

    if bad_ids_parts:

        bad_edge_ids = np.concatenate(
            bad_ids_parts
        )

        bad_jumps = np.concatenate(
            bad_jump_parts
        )

    else:

        bad_edge_ids = np.empty(
            0,
            dtype=np.int64,
        )

        bad_jumps = np.empty(
            0,
            dtype=np.int16,
        )

    bad_fraction = (
        jump_nonzero
        /
        non_tree_count
        if non_tree_count > 0
        else 0.0
    )

    print()
    print("=" * 92)
    print(
        "Non-tree local-edge integer consistency"
    )
    print("=" * 92)

    print(
        f"non-tree edges            : "
        f"{non_tree_count:,}"
    )

    print(
        f"nonzero 2pi jump edges    : "
        f"{jump_nonzero:,} "
        f"({100*bad_fraction:.6f}%)"
    )

    print(
        f"|jump| = 1              : "
        f"{jump_abs1:,}"
    )

    print(
        f"|jump| >= 2             : "
        f"{jump_abs2plus:,}"
    )

    print(
        f"integer residual max      : "
        f"{residual_max:.3e} rad"
    )

    # ========================================================
    # Residual component integer registration
    # using TWO anchors.
    # ========================================================

    registered = np.zeros(
        npoint,
        dtype=bool,
    )

    registered[
        component
        ==
        main_component
    ] = True

    component_shift = np.zeros(
        ncomp,
        dtype=np.int32,
    )

    component_shift_valid = np.zeros(
        ncomp,
        dtype=bool,
    )

    component_shift_valid[
        main_component
    ] = True

    shift_rows = []

    conflict_components = []

    residual_comp_ids = [
        int(x)
        for x in comp_ids
        if int(x) != main_component
    ]

    tier_component_count = {
        1: 0,
        2: 0,
        3: 0,
    }

    tier_conflict_count = {
        1: 0,
        2: 0,
        3: 0,
    }

    tier_registered_points = {
        1: 0,
        2: 0,
        3: 0,
    }

    for comp in residual_comp_ids:

        pids = np.where(
            component
            ==
            comp
        )[0]

        tier = int(
            point_tier[
                pids[0]
            ]
        )

        tier_component_count[
            tier
        ] += 1

        aids = np.where(
            anchor_component
            ==
            comp
        )[0]

        if aids.size != 2:

            raise RuntimeError(
                f"Component {comp} has "
                f"{aids.size} anchors."
            )

        shifts = []

        anchor_rows = []

        for aid in aids:

            a = int(
                anchor_u[
                    aid
                ]
            )

            b = int(
                anchor_v[
                    aid
                ]
            )

            if (
                component[a]
                ==
                main_component
            ):
                main_p = a
                residual_p = b

            elif (
                component[b]
                ==
                main_component
            ):
                main_p = b
                residual_p = a

            else:
                raise RuntimeError(
                    "Anchor does not connect "
                    "residual to main component."
                )

            g_anchor = float(
                wrap64(
                    float(
                        ifg[
                            main_p
                        ]
                    )
                    -
                    float(
                        ifg[
                            residual_p
                        ]
                    )
                )
            )

            raw_n = (
                unwrapped[
                    main_p
                ]
                -
                unwrapped[
                    residual_p
                ]
                -
                g_anchor
            ) / TWOPI

            nshift = int(
                np.rint(
                    raw_n
                )
            )

            check = (
                unwrapped[
                    main_p
                ]
                -
                (
                    unwrapped[
                        residual_p
                    ]
                    +
                    TWOPI
                    *
                    nshift
                )
                -
                g_anchor
            )

            shifts.append(
                nshift
            )

            anchor_rows.append(
                (
                    int(aid),
                    nshift,
                    float(check),
                    int(
                        anchor_class[
                            aid
                        ]
                    ),
                )
            )

        agree = (
            shifts[0]
            ==
            shifts[1]
        )

        if agree:

            nshift = shifts[0]

            unwrapped[
                pids
            ] += (
                TWOPI
                *
                nshift
            )

            registered[
                pids
            ] = True

            component_shift[
                comp
            ] = nshift

            component_shift_valid[
                comp
            ] = True

            tier_registered_points[
                tier
            ] += int(
                pids.size
            )

        else:

            nshift = 0

            conflict_components.append(
                comp
            )

            tier_conflict_count[
                tier
            ] += 1

        shift_rows.append({
            "component_id":
                comp,

            "point_count":
                int(
                    pids.size
                ),

            "tier":
                tier,

            "anchor1_id":
                anchor_rows[0][0],

            "anchor1_shift":
                anchor_rows[0][1],

            "anchor1_check_rad":
                anchor_rows[0][2],

            "anchor1_class":
                anchor_rows[0][3],

            "anchor2_id":
                anchor_rows[1][0],

            "anchor2_shift":
                anchor_rows[1][1],

            "anchor2_check_rad":
                anchor_rows[1][2],

            "anchor2_class":
                anchor_rows[1][3],

            "anchors_agree":
                int(
                    agree
                ),

            "applied_shift":
                int(
                    nshift
                )
                if agree
                else 0,
        })

    registered_points = int(
        registered.sum()
    )

    print()
    print("=" * 92)
    print(
        "Residual two-anchor registration"
    )
    print("=" * 92)

    print(
        f"residual components       : "
        f"{len(residual_comp_ids)}"
    )

    print(
        f"anchor-agree components   : "
        f"{len(residual_comp_ids)-len(conflict_components)}"
    )

    print(
        f"anchor-conflict components: "
        f"{len(conflict_components)}"
    )

    print(
        f"globally registered points: "
        f"{registered_points:,}/"
        f"{npoint:,} "
        f"({100*registered_points/npoint:.5f}%)"
    )

    for tier, name in (
        (1, "normal"),
        (2, "extended"),
        (3, "long"),
    ):

        print(
            f"{name:8s}: "
            f"components="
            f"{tier_component_count[tier]:3d}, "
            f"conflicts="
            f"{tier_conflict_count[tier]:3d}, "
            f"registered residual points="
            f"{tier_registered_points[tier]:4d}"
        )

    # ========================================================
    # Save prototype products
    # ========================================================

    tag = (
        f"pair{pair_id:03d}_"
        f"{stack.dates[i]}_"
        f"{stack.dates[j]}"
    )

    np.save(
        outdir
        / f"{tag}_unwrapped_phase_rad.npy",
        unwrapped.astype(
            np.float32
        ),
    )

    np.save(
        outdir
        / f"{tag}_registered_mask.npy",
        registered,
    )

    np.save(
        outdir
        / f"{tag}_bad_local_edge_ids.npy",
        bad_edge_ids,
    )

    np.save(
        outdir
        / f"{tag}_bad_local_edge_jumps.npy",
        bad_jumps,
    )

    shift_csv = (
        outdir
        / f"{tag}_residual_component_shifts.csv"
    )

    with shift_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                shift_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            shift_rows
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-single-ifg-unwrapping-prototype-v0.9",

        "status":
            "PROTOTYPE_NOT_FROZEN",

        "pair": {
            "pair_id":
                pair_id,

            "date1":
                str(
                    stack.dates[i]
                ),

            "date2":
                str(
                    stack.dates[j]
                ),

            "selection":
                (
                    "worst_step08i"
                    if args.pair_id == 0
                    else "explicit"
                ),
        },

        "input": {
            "points":
                int(npoint),

            "local_edges":
                int(nedge),

            "local_components":
                int(ncomp),
        },

        "safe_edges": {
            "count":
                int(safe_count),

            "fraction":
                float(
                    safe_count
                    /
                    nedge
                ),

            "components":
                int(
                    safe_components
                ),

            "preserves_local_partition":
                bool(
                    safe_preserves_partition
                ),
        },

        "spanning_forest": {
            "edges":
                int(
                    selected.size
                ),

            "roots":
                int(
                    forest_roots.size
                ),

            "tree_edges_gt_pi_2":
                int(
                    tree_gt_pi2
                ),

            "tree_gradient_median_rad":
                float(
                    tq[0]
                ),

            "tree_gradient_p95_rad":
                float(
                    tq[2]
                ),

            "tree_gradient_max_rad":
                float(
                    tq[4]
                ),

            "wrap_back_max_error_rad":
                float(
                    modulo_max
                ),
        },

        "non_tree_consistency": {
            "non_tree_edges":
                int(
                    non_tree_count
                ),

            "nonzero_integer_jump_edges":
                int(
                    jump_nonzero
                ),

            "nonzero_integer_jump_fraction":
                float(
                    bad_fraction
                ),

            "jump_abs1":
                int(
                    jump_abs1
                ),

            "jump_abs2plus":
                int(
                    jump_abs2plus
                ),

            "integer_residual_max_rad":
                float(
                    residual_max
                ),
        },

        "residual_registration": {
            "components":
                int(
                    len(
                        residual_comp_ids
                    )
                ),

            "anchor_agree_components":
                int(
                    len(
                        residual_comp_ids
                    )
                    -
                    len(
                        conflict_components
                    )
                ),

            "anchor_conflict_components":
                int(
                    len(
                        conflict_components
                    )
                ),

            "registered_points":
                int(
                    registered_points
                ),

            "registered_fraction":
                float(
                    registered_points
                    /
                    npoint
                ),

            "tier_conflicts": {
                "normal":
                    int(
                        tier_conflict_count[1]
                    ),

                "extended":
                    int(
                        tier_conflict_count[2]
                    ),

                "long":
                    int(
                        tier_conflict_count[3]
                    ),
            },
        },
    }

    manifest_path = (
        outdir
        / f"{tag}_manifest.json"
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
        f"unwrapped phase           : "
        f"{outdir/f'{tag}_unwrapped_phase_rad.npy'}"
    )

    print(
        f"registered mask           : "
        f"{outdir/f'{tag}_registered_mask.npy'}"
    )

    print(
        f"bad local edges           : "
        f"{bad_edge_ids.size:,}"
    )

    print(
        f"component shifts          : "
        f"{shift_csv}"
    )

    print(
        f"manifest                  : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP 08k STATUS: PASS / PROTOTYPE ONLY"
    )

    print(
        "Do not run all 108 IFGs yet."
    )


if __name__ == "__main__":
    main()
