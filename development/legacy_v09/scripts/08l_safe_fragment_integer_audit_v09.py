#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import njit

from pypsds.prototype import open_from_config


TWOPI = 2.0 * np.pi


def wrap32(x):
    return np.arctan2(
        np.sin(x),
        np.cos(x),
    ).astype(np.float32, copy=False)


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
                f"Invalid network edge: {raw}"
            )

        edges.append((i, j))

    return edges


def choose_pair(audit_csv: Path, pair_id: int):
    rows = []

    with audit_csv.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if pair_id > 0:
        for r in rows:
            if int(r["pair_id"]) == pair_id:
                return r

        raise RuntimeError(
            f"pair {pair_id} not found"
        )

    return max(
        rows,
        key=lambda r: (
            float(r["local_frac_gt_pi_2"]),
            float(r["local_p95_abs_rad"]),
        ),
    )


@njit(cache=True)
def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def roots_from_edges(n, u, v):
    parent = np.arange(
        n,
        dtype=np.int32,
    )

    size = np.ones(
        n,
        dtype=np.int32,
    )

    for k in range(u.size):

        a = uf_find(parent, u[k])
        b = uf_find(parent, v[k])

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
        roots[i] = uf_find(parent, i)

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

        a = uf_find(parent, u[eid])
        b = uf_find(parent, v[eid])

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
            indptr[i] + deg[i]
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

        p = cursor[a]
        indices[p] = b
        values[p] = g[k]
        cursor[a] += 1

        p = cursor[b]
        indices[p] = a
        values[p] = -g[k]
        cursor[b] += 1

    return indptr, indices, values


@njit(cache=True)
def integrate_forest(
    wrapped_phase,
    indptr,
    indices,
    values,
):
    n = wrapped_phase.size

    out = np.empty(
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
        out[root] = float(
            wrapped_phase[root]
        )

        head = 0
        tail = 1

        queue[0] = root

        while head < tail:

            u = queue[head]
            head += 1

            for z in range(
                indptr[u],
                indptr[u + 1],
            ):

                v = indices[z]

                if visited[v]:
                    continue

                visited[v] = 1

                out[v] = (
                    out[u]
                    +
                    float(values[z])
                )

                queue[tail] = v
                tail += 1

    return out, roots[:nroot]


def audit_edge_jumps(
    U,
    u,
    v,
    g,
):
    delta = (
        U[v]
        -
        U[u]
        -
        g.astype(np.float64)
    )

    jump = np.rint(
        delta / TWOPI
    ).astype(np.int32)

    residual = (
        delta
        -
        TWOPI * jump
    )

    return (
        jump,
        float(
            np.max(
                np.abs(residual)
            )
        ),
    )


def mode_integer(values):
    values = np.asarray(
        values,
        dtype=np.int32,
    )

    uu, cc = np.unique(
        values,
        return_counts=True,
    )

    m = int(
        np.argmax(cc)
    )

    return (
        int(uu[m]),
        int(cc[m]),
        int(values.size),
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
    )

    ap.add_argument(
        "--distance-weight",
        type=float,
        default=0.05,
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

    net = (
        root
        / "network"
    )

    graph = (
        root
        / "spatial_graph"
    )

    policy = (
        root
        / "unwrap_component_policy"
    )

    audit08i = (
        root
        / "spatial_phase_gradient_audit"
        / "per_ifg_spatial_gradient_qa.csv"
    )

    outdir = (
        root
        / "safe_fragment_integer_audit"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase = np.load(
        pps / "phase_rad.npy",
        mmap_mode="r",
    )

    npoint, ndate = phase.shape

    temporal_edges = load_itab(
        net / "network.itab",
        ndate,
    )

    pair_row = choose_pair(
        audit08i,
        args.pair_id,
    )

    pair_id = int(
        pair_row["pair_id"]
    )

    ti, tj = temporal_edges[
        pair_id - 1
    ]

    local_u = np.load(
        graph / "local_u.npy",
        mmap_mode="r",
    )

    local_v = np.load(
        graph / "local_v.npy",
        mmap_mode="r",
    )

    local_dist = np.load(
        graph / "local_distance_m.npy",
        mmap_mode="r",
    )

    local_component = np.load(
        policy / "local_component.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    nedge = int(
        local_u.size
    )

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

    u_all = np.asarray(
        local_u,
        dtype=np.int32,
    )

    v_all = np.asarray(
        local_v,
        dtype=np.int32,
    )

    g_all = wrap32(
        ifg[v_all]
        -
        ifg[u_all]
    )

    abs_g = np.abs(
        g_all
    )

    safe = (
        abs_g
        <= np.pi / 2
    )

    unsafe = ~safe

    safe_ids = np.where(
        safe
    )[0]

    unsafe_ids = np.where(
        unsafe
    )[0]

    safe_u = u_all[
        safe
    ]

    safe_v = v_all[
        safe
    ]

    safe_g = g_all[
        safe
    ]

    print("=" * 96)
    print(
        "Step 08l - Safe-fragment integer consistency audit"
    )
    print("=" * 96)

    print(
        f"config                     : {config_path}"
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
        f"points                     : "
        f"{npoint:,}"
    )

    print(
        f"local edges                : "
        f"{nedge:,}"
    )

    print(
        f"safe edges <=pi/2          : "
        f"{safe_ids.size:,}"
    )

    print(
        f"unsafe edges >pi/2         : "
        f"{unsafe_ids.size:,}"
    )

    # ========================================================
    # Safe-fragment topology
    # ========================================================

    safe_roots = roots_from_edges(
        npoint,
        safe_u,
        safe_v,
    )

    _, safe_fragment = np.unique(
        safe_roots,
        return_inverse=True,
    )

    safe_fragment = safe_fragment.astype(
        np.int32
    )

    safe_counts = np.bincount(
        safe_fragment
    )

    nsafe = int(
        safe_counts.size
    )

    local_comp_ids = np.unique(
        local_component
    )

    nlocal = int(
        local_comp_ids.size
    )

    print()
    print("=" * 96)
    print(
        "Safe-fragment topology"
    )
    print("=" * 96)

    print(
        f"original local components  : "
        f"{nlocal}"
    )

    print(
        f"safe fragments             : "
        f"{nsafe}"
    )

    print(
        f"extra fragments after cut  : "
        f"{nsafe-nlocal}"
    )

    # ========================================================
    # Safe-only quality-guided spanning forest
    # ========================================================

    dmax = float(
        np.max(local_dist)
    )

    safe_cost = (
        abs_g[safe].astype(
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
                local_dist[safe],
                dtype=np.float32,
            )
            /
            np.float32(dmax)
        )
    )

    safe_order = np.argsort(
        safe_cost,
        kind="stable",
    )

    target_safe_tree = (
        npoint
        -
        nsafe
    )

    selected_safe_local = (
        kruskal_select(
            npoint,
            safe_u,
            safe_v,
            safe_order.astype(
                np.int64,
                copy=False,
            ),
            target_safe_tree,
        )
    )

    if (
        selected_safe_local.size
        !=
        target_safe_tree
    ):

        raise RuntimeError(
            "Safe spanning forest incomplete."
        )

    tree_u = safe_u[
        selected_safe_local
    ]

    tree_v = safe_v[
        selected_safe_local
    ]

    tree_g = safe_g[
        selected_safe_local
    ]

    (
        indptr,
        indices,
        values,
    ) = build_tree_csr(
        npoint,
        tree_u,
        tree_v,
        tree_g,
    )

    U0, roots = integrate_forest(
        ifg,
        indptr,
        indices,
        values,
    )

    if roots.size != nsafe:

        raise RuntimeError(
            f"Safe forest roots={roots.size}, "
            f"expected={nsafe}"
        )

    # ========================================================
    # Safe-edge internal integer consistency
    # ========================================================

    safe_jump, safe_resid_max = (
        audit_edge_jumps(
            U0,
            safe_u,
            safe_v,
            safe_g,
        )
    )

    safe_bad = int(
        np.count_nonzero(
            safe_jump != 0
        )
    )

    safe_bad_abs1 = int(
        np.count_nonzero(
            np.abs(safe_jump)
            == 1
        )
    )

    safe_bad_abs2 = int(
        np.count_nonzero(
            np.abs(safe_jump)
            >= 2
        )
    )

    print()
    print("=" * 96)
    print(
        "Safe-fragment internal consistency"
    )
    print("=" * 96)

    print(
        f"safe edges                 : "
        f"{safe_ids.size:,}"
    )

    print(
        f"safe-tree edges            : "
        f"{selected_safe_local.size:,}"
    )

    print(
        f"safe non-tree edges        : "
        f"{safe_ids.size-selected_safe_local.size:,}"
    )

    print(
        f"nonzero integer jumps      : "
        f"{safe_bad:,} "
        f"({100*safe_bad/safe_ids.size:.8f}%)"
    )

    print(
        f"|jump|=1                  : "
        f"{safe_bad_abs1:,}"
    )

    print(
        f"|jump|>=2                 : "
        f"{safe_bad_abs2:,}"
    )

    print(
        f"integer residual max       : "
        f"{safe_resid_max:.3e} rad"
    )

    # ========================================================
    # Unsafe edges -> safe-fragment relations
    # ========================================================

    unsafe_u = u_all[
        unsafe
    ]

    unsafe_v = v_all[
        unsafe
    ]

    unsafe_g = g_all[
        unsafe
    ]

    fu = safe_fragment[
        unsafe_u
    ]

    fv = safe_fragment[
        unsafe_v
    ]

    cross = (
        fu != fv
    )

    within = (
        ~cross
    )

    n_cross = int(
        np.count_nonzero(
            cross
        )
    )

    n_within = int(
        np.count_nonzero(
            within
        )
    )

    # Integer relation from independent safe-fragment
    # unwrapped phases.
    #
    # Need:
    #    shift_v - shift_u = -b
    #
    # where:
    #    b = round((U_v-U_u-g)/2pi)
    #
    delta_unsafe = (
        U0[
            unsafe_v
        ]
        -
        U0[
            unsafe_u
        ]
        -
        unsafe_g.astype(
            np.float64
        )
    )

    b_unsafe = np.rint(
        delta_unsafe
        /
        TWOPI
    ).astype(
        np.int32
    )

    required = (
        -b_unsafe
    )

    # ========================================================
    # Group cross-fragment constraints
    # ========================================================

    groups = defaultdict(
        list
    )

    cross_indices = np.where(
        cross
    )[0]

    for z in cross_indices.tolist():

        a = int(
            fu[z]
        )

        b = int(
            fv[z]
        )

        obs = int(
            required[z]
        )

        if a < b:
            key = (a, b)
            canonical_obs = obs
        else:
            key = (b, a)
            canonical_obs = -obs

        groups[
            key
        ].append(
            (
                canonical_obs,
                int(z),
            )
        )

    group_rows = []

    exact_groups = 0
    conflicting_groups = 0

    for (
        fa,
        fb,
    ), vals in groups.items():

        obs = [
            x[0]
            for x in vals
        ]

        mode, mode_count, total = (
            mode_integer(
                obs
            )
        )

        ratio = (
            mode_count
            /
            total
        )

        if mode_count == total:
            exact_groups += 1
        else:
            conflicting_groups += 1

        local_ids = np.array(
            [
                x[1]
                for x in vals
            ],
            dtype=np.int32,
        )

        global_unsafe_ids = (
            unsafe_ids[
                local_ids
            ]
        )

        group_rows.append({
            "fragment_a":
                fa,

            "fragment_b":
                fb,

            "edge_count":
                total,

            "mode_shift_b_minus_a":
                mode,

            "mode_count":
                mode_count,

            "consensus_ratio":
                ratio,

            "exact_consensus":
                int(
                    mode_count
                    ==
                    total
                ),

            "median_abs_gradient_rad":
                float(
                    np.median(
                        abs_g[
                            global_unsafe_ids
                        ]
                    )
                ),

            "min_abs_gradient_rad":
                float(
                    np.min(
                        abs_g[
                            global_unsafe_ids
                        ]
                    )
                ),

            "median_distance_m":
                float(
                    np.median(
                        np.asarray(
                            local_dist[
                                global_unsafe_ids
                            ]
                        )
                    )
                ),
        })

    print()
    print("=" * 96)
    print(
        "Unsafe-edge safe-fragment structure"
    )
    print("=" * 96)

    print(
        f"unsafe edges               : "
        f"{unsafe_ids.size:,}"
    )

    print(
        f"cross safe-fragment edges  : "
        f"{n_cross:,}"
    )

    print(
        f"within safe-fragment edges : "
        f"{n_within:,}"
    )

    print(
        f"distinct fragment pairs    : "
        f"{len(groups):,}"
    )

    print(
        f"exact-consensus pairs      : "
        f"{exact_groups:,}"
    )

    print(
        f"conflicting pair groups    : "
        f"{conflicting_groups:,}"
    )

    # ========================================================
    # Fragment-level consensus graph
    # ========================================================

    if not group_rows:
        raise RuntimeError(
            "No cross-fragment constraints."
        )

    # Higher consensus first,
    # then more supporting edges,
    # then lower median |g|,
    # then shorter distance.
    group_rows.sort(
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

    super_u = np.array(
        [
            r["fragment_a"]
            for r in group_rows
        ],
        dtype=np.int32,
    )

    super_v = np.array(
        [
            r["fragment_b"]
            for r in group_rows
        ],
        dtype=np.int32,
    )

    super_shift = np.array(
        [
            r[
                "mode_shift_b_minus_a"
            ]
            for r in group_rows
        ],
        dtype=np.int32,
    )

    # Already sorted by quality.
    super_order = np.arange(
        len(group_rows),
        dtype=np.int64,
    )

    target_super_tree = (
        nsafe
        -
        nlocal
    )

    selected_super = kruskal_select(
        nsafe,
        super_u,
        super_v,
        super_order,
        target_super_tree,
    )

    if (
        selected_super.size
        !=
        target_super_tree
    ):

        raise RuntimeError(
            f"Super-forest only selected "
            f"{selected_super.size}, "
            f"expected {target_super_tree}."
        )

    # ========================================================
    # Integrate fragment integer shifts
    # ========================================================

    adj = [
        []
        for _ in range(
            nsafe
        )
    ]

    for eid in selected_super.tolist():

        a = int(
            super_u[eid]
        )

        b = int(
            super_v[eid]
        )

        d = int(
            super_shift[eid]
        )

        # shift_b = shift_a + d
        adj[a].append(
            (b, d)
        )

        adj[b].append(
            (a, -d)
        )

    fragment_shift = np.zeros(
        nsafe,
        dtype=np.int32,
    )

    visited = np.zeros(
        nsafe,
        dtype=bool,
    )

    super_roots = []

    for root_frag in range(
        nsafe
    ):

        if visited[
            root_frag
        ]:
            continue

        super_roots.append(
            root_frag
        )

        visited[
            root_frag
        ] = True

        todo = [
            root_frag
        ]

        while todo:

            a = todo.pop()

            for b, d in adj[a]:

                if visited[b]:
                    continue

                fragment_shift[b] = (
                    fragment_shift[a]
                    +
                    d
                )

                visited[b] = True
                todo.append(b)

    if len(
        super_roots
    ) != nlocal:

        raise RuntimeError(
            f"Consensus superforest roots="
            f"{len(super_roots)}, "
            f"expected local components="
            f"{nlocal}"
        )

    U1 = (
        U0
        +
        TWOPI
        *
        fragment_shift[
            safe_fragment
        ].astype(
            np.float64
        )
    )

    # ========================================================
    # Full local-edge consistency after fragment registration
    # ========================================================

    full_jump, full_resid_max = (
        audit_edge_jumps(
            U1,
            u_all,
            v_all,
            g_all,
        )
    )

    full_bad = int(
        np.count_nonzero(
            full_jump != 0
        )
    )

    full_bad_abs1 = int(
        np.count_nonzero(
            np.abs(full_jump)
            == 1
        )
    )

    full_bad_abs2 = int(
        np.count_nonzero(
            np.abs(full_jump)
            >= 2
        )
    )

    safe_bad_after = int(
        np.count_nonzero(
            full_jump[
                safe
            ]
            != 0
        )
    )

    unsafe_bad_after = int(
        np.count_nonzero(
            full_jump[
                unsafe
            ]
            != 0
        )
    )

    print()
    print("=" * 96)
    print(
        "Consensus fragment registration"
    )
    print("=" * 96)

    print(
        f"required fragment merges   : "
        f"{target_super_tree}"
    )

    print(
        f"selected super-edges       : "
        f"{selected_super.size}"
    )

    print(
        f"superforest roots          : "
        f"{len(super_roots)}"
    )

    print()
    print(
        f"all local bad edges        : "
        f"{full_bad:,}/"
        f"{nedge:,} "
        f"({100*full_bad/nedge:.8f}%)"
    )

    print(
        f"bad SAFE edges             : "
        f"{safe_bad_after:,}"
    )

    print(
        f"bad UNSAFE edges           : "
        f"{unsafe_bad_after:,}"
    )

    print(
        f"|jump|=1                  : "
        f"{full_bad_abs1:,}"
    )

    print(
        f"|jump|>=2                 : "
        f"{full_bad_abs2:,}"
    )

    print(
        f"integer residual max       : "
        f"{full_resid_max:.3e} rad"
    )

    print()
    print(
        "08k baseline bad edges     : 132"
    )

    print(
        f"08l consensus bad edges    : "
        f"{full_bad}"
    )

    print(
        f"change vs 08k              : "
        f"{full_bad-132:+d}"
    )

    # ========================================================
    # Super-edge constraint audit
    # ========================================================

    super_group_bad = 0
    super_individual_bad = 0

    for r in group_rows:

        a = int(
            r["fragment_a"]
        )

        b = int(
            r["fragment_b"]
        )

        expected = int(
            r[
                "mode_shift_b_minus_a"
            ]
        )

        actual = int(
            fragment_shift[b]
            -
            fragment_shift[a]
        )

        if actual != expected:
            super_group_bad += 1

    for z in cross_indices.tolist():

        a = int(
            fu[z]
        )

        b = int(
            fv[z]
        )

        obs = int(
            required[z]
        )

        actual = int(
            fragment_shift[b]
            -
            fragment_shift[a]
        )

        if actual != obs:
            super_individual_bad += 1

    print()
    print(
        f"fragment-pair mode conflicts: "
        f"{super_group_bad}/"
        f"{len(group_rows)}"
    )

    print(
        f"individual cross-edge "
        f"conflicts: "
        f"{super_individual_bad}/"
        f"{n_cross}"
    )

    # ========================================================
    # Save
    # ========================================================

    tag = (
        f"pair{pair_id:03d}_"
        f"{stack.dates[ti]}_"
        f"{stack.dates[tj]}"
    )

    np.save(
        outdir
        / f"{tag}_safe_fragment.npy",
        safe_fragment,
    )

    np.save(
        outdir
        / f"{tag}_fragment_shift.npy",
        fragment_shift,
    )

    np.save(
        outdir
        / f"{tag}_consensus_unwrapped.npy",
        U1.astype(
            np.float32
        ),
    )

    bad_ids = np.where(
        full_jump != 0
    )[0].astype(
        np.int64
    )

    np.save(
        outdir
        / f"{tag}_bad_edge_ids.npy",
        bad_ids,
    )

    np.save(
        outdir
        / f"{tag}_bad_edge_jumps.npy",
        full_jump[
            bad_ids
        ].astype(
            np.int16
        ),
    )

    groups_csv = (
        outdir
        / f"{tag}_fragment_pair_consensus.csv"
    )

    with groups_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                group_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            group_rows
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-safe-fragment-integer-audit-v0.9",

        "status":
            "PROTOTYPE_NOT_FROZEN",

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

        "topology": {
            "original_local_components":
                nlocal,

            "safe_fragments":
                nsafe,

            "required_fragment_merges":
                nsafe - nlocal,
        },

        "safe_internal": {
            "safe_edges":
                int(
                    safe_ids.size
                ),

            "nonzero_integer_jump_edges":
                safe_bad,

            "jump_abs1":
                safe_bad_abs1,

            "jump_abs2plus":
                safe_bad_abs2,

            "integer_residual_max_rad":
                safe_resid_max,
        },

        "unsafe_structure": {
            "unsafe_edges":
                int(
                    unsafe_ids.size
                ),

            "cross_fragment_edges":
                n_cross,

            "within_fragment_edges":
                n_within,

            "fragment_pairs":
                len(
                    groups
                ),

            "exact_consensus_pairs":
                exact_groups,

            "conflicting_pairs":
                conflicting_groups,
        },

        "consensus_solution": {
            "selected_super_edges":
                int(
                    selected_super.size
                ),

            "superforest_roots":
                len(
                    super_roots
                ),

            "all_bad_edges":
                full_bad,

            "safe_bad_edges":
                safe_bad_after,

            "unsafe_bad_edges":
                unsafe_bad_after,

            "jump_abs1":
                full_bad_abs1,

            "jump_abs2plus":
                full_bad_abs2,

            "fragment_pair_mode_conflicts":
                super_group_bad,

            "individual_cross_edge_conflicts":
                super_individual_bad,

            "baseline_08k_bad_edges":
                132,
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
        f"fragment consensus table  : "
        f"{groups_csv}"
    )

    print(
        f"manifest                  : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP 08l STATUS: PASS / PROTOTYPE ONLY"
    )

    print(
        "Do not batch-unwrap 108 IFGs yet."
    )


if __name__ == "__main__":
    main()
