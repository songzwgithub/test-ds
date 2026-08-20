#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit

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
    """
    Add all undirected point pairs with

        max(|dr|, |dc|) == radius

    exactly once.

    Only half of each symmetric offset shell
    is visited:
        dr > 0
        or
        dr == 0 and dc > 0
    """

    H, W = index_grid.shape
    n = rows.size

    candidate_edges = 0
    union_edges = 0

    r = radius

    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):

            if max(abs(dr), abs(dc)) != r:
                continue

            # Undirected half-plane only.
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

                candidate_edges += 1

                if uf_union(
                    parent,
                    size,
                    p,
                    q,
                ):
                    union_edges += 1

    return candidate_edges, union_edges


@njit(cache=True)
def compress_all(parent):
    n = parent.size

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


def topology_stats(parent):
    roots = compress_all(
        parent
    )

    _, counts = np.unique(
        roots,
        return_counts=True,
    )

    counts = np.sort(
        counts
    )[::-1]

    npoint = parent.size
    ncomp = counts.size

    largest = int(
        counts[0]
    )

    second = (
        int(counts[1])
        if ncomp > 1
        else 0
    )

    singleton_components = int(
        np.count_nonzero(
            counts == 1
        )
    )

    components_le10 = int(
        np.count_nonzero(
            counts <= 10
        )
    )

    components_le100 = int(
        np.count_nonzero(
            counts <= 100
        )
    )

    outside = (
        npoint
        -
        largest
    )

    return {
        "components": int(
            ncomp
        ),

        "largest": largest,

        "largest_fraction": float(
            largest / npoint
        ),

        "second_largest": second,

        "singleton_components":
            singleton_components,

        "components_le10":
            components_le10,

        "components_le100":
            components_le100,

        "points_outside_largest":
            int(outside),

        "outside_fraction": float(
            outside / npoint
        ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--max-radius",
        type=int,
        default=6,
    )

    args = ap.parse_args()

    if args.max_radius < 1:
        raise ValueError(
            "--max-radius must be >= 1"
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
        / "spatial_graph_radius_audit"
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

    # Important:
    # Point coordinates do not necessarily occupy
    # the last row/column, so extent is only an
    # index lookup grid here.
    H = int(
        rows.max()
    ) + 1

    W = int(
        cols.max()
    ) + 1

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
        "Step 08c - Local spatial bridge-radius audit"
    )
    print("=" * 88)

    print(
        f"config                 : {config_path}"
    )

    print(
        f"points                 : {npoint:,}"
    )

    print(
        f"lookup grid            : {H} x {W}"
    )

    print(
        f"maximum tested radius  : {args.max_radius}"
    )

    print()
    print(
        "Distance definition:"
    )

    print(
        "  Chebyshev radius R:"
    )

    print(
        "  max(|delta_row|, |delta_col|) <= R"
    )

    print()
    print(
        "R=1 is exactly the ordinary 8-neighbor graph."
    )

    results = []

    cumulative_edges = 0

    print()
    print("=" * 88)
    print(
        " Radius | pair edges | components | "
        "largest component       | outside largest | singletons"
    )
    print("=" * 88)

    for radius in range(
        1,
        args.max_radius + 1,
    ):

        shell_edges, union_edges = (
            add_shell_edges(
                index_grid,
                rows,
                cols,
                parent,
                size,
                radius,
            )
        )

        cumulative_edges += int(
            shell_edges
        )

        stat = topology_stats(
            parent
        )

        stat.update({
            "radius": int(
                radius
            ),

            "new_pair_edges": int(
                shell_edges
            ),

            "cumulative_pair_edges": int(
                cumulative_edges
            ),

            "new_component_merges": int(
                union_edges
            ),
        })

        results.append(
            stat
        )

        print(
            f"   {radius:2d}   | "
            f"{cumulative_edges:10,d} | "
            f"{stat['components']:10,d} | "
            f"{stat['largest']:10,d} "
            f"({100*stat['largest_fraction']:7.3f}%) | "
            f"{stat['points_outside_largest']:10,d} "
            f"({100*stat['outside_fraction']:7.3f}%) | "
            f"{stat['singleton_components']:8,d}"
        )

    # --------------------------------------------------------
    # Identify smallest useful radii.
    # Do NOT call these hard production thresholds yet.
    # --------------------------------------------------------

    thresholds = [
        0.90,
        0.95,
        0.99,
        0.995,
        0.999,
    ]

    crossing = {}

    for target in thresholds:

        rr = None

        for result in results:

            if (
                result[
                    "largest_fraction"
                ]
                >= target
            ):
                rr = result[
                    "radius"
                ]
                break

        crossing[
            str(target)
        ] = rr

    print()
    print("=" * 88)
    print(
        "Largest-component radius milestones"
    )
    print("=" * 88)

    for target in thresholds:

        rr = crossing[
            str(target)
        ]

        if rr is None:

            print(
                f">= {100*target:6.2f}% : "
                f"not reached by R="
                f"{args.max_radius}"
            )

        else:

            print(
                f">= {100*target:6.2f}% : "
                f"R={rr}"
            )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "format":
            "pyPSDS-GAMMA-spatial-radius-audit-v0.9",

        "points": int(
            npoint
        ),

        "distance":
            "Chebyshev pixel radius",

        "maximum_tested_radius": int(
            args.max_radius
        ),

        "results": results,

        "largest_component_milestones":
            crossing,
    }

    manifest_path = (
        outdir
        / "spatial_radius_audit.json"
    )

    manifest_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    # Also save easy CSV.
    import csv

    csv_path = (
        outdir
        / "spatial_radius_audit.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "radius",
            "new_pair_edges",
            "cumulative_pair_edges",
            "components",
            "largest",
            "largest_fraction",
            "second_largest",
            "points_outside_largest",
            "outside_fraction",
            "singleton_components",
            "components_le10",
            "components_le100",
        ])

        for s in results:

            w.writerow([
                s["radius"],
                s["new_pair_edges"],
                s["cumulative_pair_edges"],
                s["components"],
                s["largest"],
                s["largest_fraction"],
                s["second_largest"],
                s["points_outside_largest"],
                s["outside_fraction"],
                s["singleton_components"],
                s["components_le10"],
                s["components_le100"],
            ])

    print()
    print(
        f"CSV                    : {csv_path}"
    )

    print(
        f"manifest               : {manifest_path}"
    )

    print()
    print(
        "STEP 08c STATUS: PASS"
    )


if __name__ == "__main__":
    main()
