#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numba import njit

from pypsds.context import open_from_config


TWOPI = 2.0 * np.pi


def wrap32(x):
    return np.arctan2(
        np.sin(x),
        np.cos(x),
    ).astype(
        np.float32,
        copy=False,
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
def build_safe_forest(
    npoint,
    u,
    v,
    safe,
):
    """
    Scan safe edges in deterministic production-edge order.

    Every successful union is immediately a spanning-forest edge.
    Therefore:

        number of safe components
          = npoint - number of selected tree edges

    No quality sorting is needed for this quality.
    """

    parent = np.arange(
        npoint,
        dtype=np.int32,
    )

    size = np.ones(
        npoint,
        dtype=np.int32,
    )

    # At most Npoint-1 tree edges.
    selected = np.empty(
        npoint - 1,
        dtype=np.int64,
    )

    count = 0

    for k in range(
        u.size
    ):

        if not safe[k]:
            continue

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

        selected[count] = k
        count += 1

    # Compress.
    roots = np.empty(
        npoint,
        dtype=np.int32,
    )

    for i in range(
        npoint
    ):
        roots[i] = uf_find(
            parent,
            i,
        )

    return (
        selected[:count],
        roots,
    )


@njit(cache=True)
def build_tree_csr(
    npoint,
    u,
    v,
    g,
):
    m = u.size

    degree = np.zeros(
        npoint,
        dtype=np.int32,
    )

    for k in range(m):
        degree[u[k]] += 1
        degree[v[k]] += 1

    indptr = np.zeros(
        npoint + 1,
        dtype=np.int64,
    )

    for i in range(
        npoint
    ):
        indptr[i + 1] = (
            indptr[i]
            +
            degree[i]
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
    )


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

                out[b] = (
                    out[a]
                    +
                    float(values[z])
                )

                queue[tail] = b
                tail += 1

    return (
        out,
        roots[:nroot],
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

    policy = (
        root
        / "unwrap_component_policy"
    )

    outdir = (
        root
        / "acquisition_safe_fragment_quality"
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

    local_component = np.load(
        policy
        / "local_component.npy",
        mmap_mode="r",
    )

    npoint, ndate = phase.shape

    u = np.asarray(
        local_u,
        dtype=np.int32,
    )

    v = np.asarray(
        local_v,
        dtype=np.int32,
    )

    nedge = u.size

    nlocal = int(
        np.unique(
            local_component
        ).size
    )

    print("=" * 104)
    print(
        "Acquisition-domain SAFE-fragment quality"
    )
    print("=" * 104)

    print(
        f"config                     : {config_path}"
    )

    print(
        f"points                     : {npoint:,}"
    )

    print(
        f"acquisitions               : {ndate}"
    )

    print(
        f"local edges                : {nedge:,}"
    )

    print(
        f"static R4-K8 components    : {nlocal}"
    )

    print()

    results = []

    for t in range(
        ndate
    ):

        phi = np.asarray(
            phase[:, t],
            dtype=np.float32,
        )

        g = wrap32(
            phi[v]
            -
            phi[u]
        )

        abs_g = np.abs(
            g
        )

        safe = (
            abs_g
            <=
            np.pi / 2
        )

        unsafe_count = int(
            np.count_nonzero(
                ~safe
            )
        )

        (
            selected,
            roots,
        ) = build_safe_forest(
            npoint,
            u,
            v,
            safe,
        )

        _, safe_fragment = np.unique(
            roots,
            return_inverse=True,
        )

        safe_fragment = (
            safe_fragment.astype(
                np.int32
            )
        )

        nsafe = int(
            safe_fragment.max()
        ) + 1

        extra_fragments = (
            nsafe
            -
            nlocal
        )

        tree_u = u[
            selected
        ]

        tree_v = v[
            selected
        ]

        tree_g = g[
            selected
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

        (
            U,
            forest_roots,
        ) = integrate_forest(
            phi,
            indptr,
            indices,
            values,
        )

        if (
            forest_roots.size
            !=
            nsafe
        ):

            raise RuntimeError(
                f"{stack.dates[t]}: "
                f"forest roots mismatch "
                f"{forest_roots.size} != "
                f"{nsafe}"
            )

        # ----------------------------------------------------
        # Integer consistency on ALL safe edges.
        # ----------------------------------------------------

        safe_ids = np.where(
            safe
        )[0]

        su = u[
            safe_ids
        ]

        sv = v[
            safe_ids
        ]

        sg = g[
            safe_ids
        ].astype(
            np.float64
        )

        delta = (
            U[sv]
            -
            U[su]
            -
            sg
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

        bad_mask = (
            jump != 0
        )

        safe_bad = int(
            np.count_nonzero(
                bad_mask
            )
        )

        safe_bad_abs1 = int(
            np.count_nonzero(
                np.abs(
                    jump[
                        bad_mask
                    ]
                )
                ==
                1
            )
        )

        safe_bad_abs2plus = int(
            np.count_nonzero(
                np.abs(
                    jump[
                        bad_mask
                    ]
                )
                >=
                2
            )
        )

        if safe_bad:

            bad_safe_edge_ids = (
                safe_ids[
                    bad_mask
                ]
            )

            bad_abs_g = (
                abs_g[
                    bad_safe_edge_ids
                ]
            )

            bad_u = u[
                bad_safe_edge_ids
            ]

            bad_v = v[
                bad_safe_edge_ids
            ]

            bad_points = np.unique(
                np.concatenate(
                    [
                        bad_u,
                        bad_v,
                    ]
                )
            )

            bad_g_min = float(
                bad_abs_g.min()
            )

            bad_g_med = float(
                np.median(
                    bad_abs_g
                )
            )

            bad_g_max = float(
                bad_abs_g.max()
            )

            np.save(
                outdir
                / (
                    f"{stack.dates[t]}"
                    "_safe_bad_edge_ids.npy"
                ),
                bad_safe_edge_ids.astype(
                    np.int64
                ),
            )

        else:

            bad_points = np.empty(
                0,
                dtype=np.int32,
            )

            bad_g_min = 0.0
            bad_g_med = 0.0
            bad_g_max = 0.0

        wrap_back = np.abs(
            wrap32(
                U.astype(
                    np.float32
                )
                -
                phi
            )
        )

        wrap_back_max = float(
            wrap_back.max()
        )

        row = {
            "acquisition_index":
                t,

            "date":
                str(
                    stack.dates[t]
                ),

            "unsafe_edges":
                unsafe_count,

            "unsafe_fraction":
                unsafe_count
                /
                nedge,

            "safe_fragments":
                nsafe,

            "extra_fragments":
                extra_fragments,

            "safe_tree_edges":
                int(
                    selected.size
                ),

            "safe_internal_bad":
                safe_bad,

            "safe_bad_abs1":
                safe_bad_abs1,

            "safe_bad_abs2plus":
                safe_bad_abs2plus,

            "safe_bad_unique_points":
                int(
                    bad_points.size
                ),

            "safe_bad_abs_g_min":
                bad_g_min,

            "safe_bad_abs_g_median":
                bad_g_med,

            "safe_bad_abs_g_max":
                bad_g_max,

            "integer_residual_max_rad":
                float(
                    np.max(
                        np.abs(
                            residual
                        )
                    )
                ),

            "wrap_back_max_error_rad":
                wrap_back_max,
        }

        results.append(
            row
        )

        flag = (
            "CONFLICT"
            if safe_bad
            else
            "OK"
        )

        marker = (
            "  <=="
            if str(
                stack.dates[t]
            )
            in (
                "20141018",
                "20150110",
                "20160329",
            )
            else
            ""
        )

        print(
            f"{str(stack.dates[t]):10s}  "
            f"unsafe="
            f"{unsafe_count:6,d} "
            f"({100*unsafe_count/nedge:7.4f}%)  "
            f"fragments="
            f"{nsafe:4d}  "
            f"extra="
            f"{extra_fragments:3d}  "
            f"SAFE-bad="
            f"{safe_bad:4d}  "
            f"points="
            f"{bad_points.size:3d}  "
            f"{flag}"
            f"{marker}"
        )

    # ========================================================
    # Summary
    # ========================================================

    conflict = [
        r
        for r in results
        if r[
            "safe_internal_bad"
        ] > 0
    ]

    nfrag = np.array(
        [
            r[
                "safe_fragments"
            ]
            for r in results
        ],
        dtype=np.int32,
    )

    unsafe = np.array(
        [
            r[
                "unsafe_edges"
            ]
            for r in results
        ],
        dtype=np.int64,
    )

    print()
    print("=" * 104)
    print(
        "Acquisition-domain SAFE consistency summary"
    )
    print("=" * 104)

    print(
        f"acquisitions tested        : "
        f"{ndate}"
    )

    print(
        f"SAFE-conflict acquisitions : "
        f"{len(conflict)}/{ndate}"
    )

    print(
        f"safe fragments min/med/max : "
        f"{nfrag.min()} / "
        f"{np.median(nfrag):.1f} / "
        f"{nfrag.max()}"
    )

    print(
        f"unsafe edges min/med/max   : "
        f"{unsafe.min():,} / "
        f"{np.median(unsafe):,.1f} / "
        f"{unsafe.max():,}"
    )

    if conflict:

        print()
        print(
            "Acquisitions with SAFE integer conflicts:"
        )

        for r in conflict:

            print(
                f"  {r['date']}: "
                f"bad={r['safe_internal_bad']}, "
                f"points="
                f"{r['safe_bad_unique_points']}, "
                f"|g| min/med/max="
                f"{r['safe_bad_abs_g_min']:.3f}/"
                f"{r['safe_bad_abs_g_median']:.3f}/"
                f"{r['safe_bad_abs_g_max']:.3f}"
            )

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / "acquisition_safe_fragment_quality.csv"
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
            "pyPSDS-GAMMA-acquisition-safe-fragment-quality-v1.0",

        "status":
            "QUALITY_ONLY",

        "acquisitions":
            ndate,

        "static_local_components":
            nlocal,

        "safe_conflict_acquisitions":
            len(
                conflict
            ),

        "conflict_dates":
            [
                r["date"]
                for r in conflict
            ],

        "safe_fragments": {
            "min":
                int(
                    nfrag.min()
                ),

            "median":
                float(
                    np.median(
                        nfrag
                    )
                ),

            "max":
                int(
                    nfrag.max()
                ),
        },
    }

    json_path = (
        outdir
        / "acquisition_safe_fragment_quality.json"
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
        f"CSV                        : "
        f"{csv_path}"
    )

    print(
        f"manifest                   : "
        f"{json_path}"
    )

    print()
    print(
        "STEP 08s STATUS: PASS / QUALITY ONLY"
    )


if __name__ == "__main__":
    main()
