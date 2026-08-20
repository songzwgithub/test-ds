#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy import ndimage

from pypsds.prototype import open_from_config


@njit(cache=True)
def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def uf_union(parent, size, a, b):
    ra = uf_find(parent, a)
    rb = uf_find(parent, b)

    if ra == rb:
        return False

    if size[ra] < size[rb]:
        ra, rb = rb, ra

    parent[rb] = ra
    size[ra] += size[rb]

    return True


@njit(cache=True)
def add_shell_edges(
    index_grid,
    rows,
    cols,
    parent,
    size,
    radius,
):
    H, W = index_grid.shape
    n = rows.size

    n_edges = 0
    n_merge = 0

    r = radius

    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):

            if max(abs(dr), abs(dc)) != r:
                continue

            # Visit each undirected offset once.
            if dr < 0:
                continue

            if dr == 0 and dc <= 0:
                continue

            for p in range(n):

                rr = rows[p] + dr
                cc = cols[p] + dc

                if rr < 0 or rr >= H:
                    continue

                if cc < 0 or cc >= W:
                    continue

                q = index_grid[rr, cc]

                if q < 0:
                    continue

                n_edges += 1

                if uf_union(
                    parent,
                    size,
                    p,
                    q,
                ):
                    n_merge += 1

    return n_edges, n_merge


@njit(cache=True)
def get_roots(parent):
    n = parent.size

    out = np.empty(
        n,
        dtype=np.int32,
    )

    for i in range(n):
        out[i] = uf_find(
            parent,
            i,
        )

    return out


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--core-radius",
        type=int,
        default=4,
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

    outdir = (
        outroot
        / "spatial_graph_residual_audit"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = np.load(
        pps_dir / "rows.npy"
    ).astype(
        np.int32,
        copy=False,
    )

    cols = np.load(
        pps_dir / "cols.npy"
    ).astype(
        np.int32,
        copy=False,
    )

    npoint = rows.size

    if cols.size != npoint:
        raise RuntimeError(
            "rows/cols length mismatch"
        )

    H = int(rows.max()) + 1
    W = int(cols.max()) + 1

    index_grid = np.full(
        (H, W),
        -1,
        dtype=np.int32,
    )

    index_grid[
        rows,
        cols,
    ] = np.arange(
        npoint,
        dtype=np.int32,
    )

    parent = np.arange(
        npoint,
        dtype=np.int32,
    )

    size = np.ones(
        npoint,
        dtype=np.int32,
    )

    print("=" * 88)
    print(
        "Step 08d - R=4 residual spatial-component audit"
    )
    print("=" * 88)

    print(
        f"config                 : {config_path}"
    )

    print(
        f"points                 : {npoint:,}"
    )

    print(
        f"core radius            : {args.core_radius}"
    )

    print()

    cumulative_edges = 0

    for radius in range(
        1,
        args.core_radius + 1,
    ):

        ne, nm = add_shell_edges(
            index_grid,
            rows,
            cols,
            parent,
            size,
            radius,
        )

        cumulative_edges += int(ne)

        print(
            f"R={radius}: "
            f"shell_edges={ne:,}, "
            f"component_merges={nm:,}"
        )

    roots = get_roots(
        parent
    )

    unique_roots, inverse, counts = np.unique(
        roots,
        return_inverse=True,
        return_counts=True,
    )

    ncomp = len(
        unique_roots
    )

    largest_label = int(
        np.argmax(
            counts
        )
    )

    largest_size = int(
        counts[
            largest_label
        ]
    )

    component_label = inverse.astype(
        np.int32
    )

    main_points = (
        component_label
        ==
        largest_label
    )

    residual_points = (
        ~main_points
    )

    n_residual = int(
        residual_points.sum()
    )

    n_residual_comp = (
        ncomp - 1
    )

    print()
    print("=" * 88)
    print(
        "R=4 topology"
    )
    print("=" * 88)

    print(
        f"components             : {ncomp:,}"
    )

    print(
        f"largest component      : "
        f"{largest_size:,} "
        f"({100*largest_size/npoint:.4f}%)"
    )

    print(
        f"residual components    : "
        f"{n_residual_comp:,}"
    )

    print(
        f"residual points        : "
        f"{n_residual:,} "
        f"({100*n_residual/npoint:.4f}%)"
    )

    # ========================================================
    # Exact Chebyshev distance to the largest component
    # ========================================================

    main_mask = np.zeros(
        (H, W),
        dtype=bool,
    )

    main_mask[
        rows[main_points],
        cols[main_points],
    ] = True

    # distance_transform_cdt calculates the
    # distance of non-zero pixels to nearest zero.
    # Therefore ~main_mask has zero values exactly
    # on the largest component.
    distances, nearest_indices = (
        ndimage.distance_transform_cdt(
            ~main_mask,
            metric="chessboard",
            return_distances=True,
            return_indices=True,
        )
    )

    residual_ids = np.where(
        residual_points
    )[0]

    residual_distance = distances[
        rows[residual_ids],
        cols[residual_ids],
    ].astype(
        np.int32
    )

    print()
    print("=" * 88)
    print(
        "Residual-point distance to largest component"
    )
    print("=" * 88)

    if residual_ids.size:

        q = np.quantile(
            residual_distance,
            [
                0.0,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.0,
            ],
        )

        print(
            "Chebyshev radius "
            "min/p25/p50/p75/p90/p95/p99/max:"
        )

        print(
            "  "
            + " / ".join(
                f"{x:.1f}"
                for x in q
            )
        )

        for radius in (
            5,
            6,
            7,
            8,
            10,
            12,
            15,
            20,
        ):

            nn = int(
                np.count_nonzero(
                    residual_distance
                    <= radius
                )
            )

            print(
                f"distance <= {radius:2d}: "
                f"{nn:4d}/{n_residual:4d} "
                f"({100*nn/n_residual:7.3f}%)"
            )

    # ========================================================
    # Per residual component:
    # nearest direct bridge to giant component
    # ========================================================

    rows_out = []

    for label in range(
        ncomp
    ):

        if label == largest_label:
            continue

        ids = np.where(
            component_label
            ==
            label
        )[0]

        dd = distances[
            rows[ids],
            cols[ids],
        ]

        local = int(
            np.argmin(dd)
        )

        p = int(
            ids[local]
        )

        rp = int(
            rows[p]
        )

        cp = int(
            cols[p]
        )

        bridge_radius = int(
            distances[
                rp,
                cp
            ]
        )

        rq = int(
            nearest_indices[
                0,
                rp,
                cp
            ]
        )

        cq = int(
            nearest_indices[
                1,
                rp,
                cp
            ]
        )

        qid = int(
            index_grid[
                rq,
                cq
            ]
        )

        if qid < 0:
            raise RuntimeError(
                "Nearest largest-component "
                "coordinate has no point index."
            )

        if not main_points[
            qid
        ]:
            raise RuntimeError(
                "Nearest point is not in "
                "largest component."
            )

        cheb = max(
            abs(rp - rq),
            abs(cp - cq),
        )

        if cheb != bridge_radius:
            raise RuntimeError(
                "Distance-transform bridge mismatch."
            )

        rows_out.append({
            "component_label": int(
                label
            ),

            "component_size": int(
                ids.size
            ),

            "bridge_radius": int(
                bridge_radius
            ),

            "residual_point_id": p,

            "residual_row": rp,

            "residual_col": cp,

            "main_point_id": qid,

            "main_row": rq,

            "main_col": cq,
        })

    rows_out.sort(
        key=lambda x: (
            x["bridge_radius"],
            x["component_size"],
        )
    )

    component_radii = np.array(
        [
            x["bridge_radius"]
            for x in rows_out
        ],
        dtype=np.int32,
    )

    print()
    print("=" * 88)
    print(
        "Residual-component direct bridge radii"
    )
    print("=" * 88)

    if component_radii.size:

        print(
            f"min / median / max     : "
            f"{component_radii.min()} / "
            f"{np.median(component_radii):.1f} / "
            f"{component_radii.max()}"
        )

        for radius in (
            5,
            6,
            7,
            8,
            10,
            12,
            15,
            20,
        ):

            nn = int(
                np.count_nonzero(
                    component_radii
                    <= radius
                )
            )

            print(
                f"components bridgeable "
                f"at R<={radius:2d}: "
                f"{nn:3d}/"
                f"{n_residual_comp:3d}"
            )

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / "residual_components_to_main.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        fields = [
            "component_label",
            "component_size",
            "bridge_radius",
            "residual_point_id",
            "residual_row",
            "residual_col",
            "main_point_id",
            "main_row",
            "main_col",
        ]

        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()
        w.writerows(
            rows_out
        )

    np.save(
        outdir
        / "component_label_r4.npy",
        component_label,
    )

    summary = {
        "format":
            "pyPSDS-GAMMA-residual-spatial-audit-v0.9",

        "core_radius":
            int(args.core_radius),

        "points":
            int(npoint),

        "components":
            int(ncomp),

        "largest_component":
            int(largest_size),

        "largest_fraction":
            float(
                largest_size
                /
                npoint
            ),

        "residual_components":
            int(n_residual_comp),

        "residual_points":
            int(n_residual),

        "direct_bridge_radius": {
            "min": (
                int(component_radii.min())
                if component_radii.size
                else 0
            ),

            "median": (
                float(
                    np.median(
                        component_radii
                    )
                )
                if component_radii.size
                else 0.0
            ),

            "max": (
                int(component_radii.max())
                if component_radii.size
                else 0
            ),
        },
    }

    json_path = (
        outdir
        / "residual_spatial_audit.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print(
        f"component labels       : "
        f"{outdir/'component_label_r4.npy'}"
    )

    print(
        f"bridge table           : {csv_path}"
    )

    print(
        f"manifest               : {json_path}"
    )

    print()
    print(
        "STEP 08d STATUS: PASS"
    )


if __name__ == "__main__":
    main()
