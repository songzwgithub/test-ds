#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

from pypsds.config import cfg_get
from pypsds.context import open_from_config


def read_gamma_value(path: Path, key: str):
    key0 = key.rstrip(":") + ":"

    for raw in path.read_text(
        errors="ignore"
    ).splitlines():

        f = raw.split()

        if not f:
            continue

        if f[0] == key0:

            for x in f[1:]:

                try:
                    return float(x)
                except ValueError:
                    continue

    return None


def build_offsets(
    radius,
    row_spacing,
    col_spacing,
):
    """
    row -> azimuth direction
    col -> range direction

    Candidate domain:
        Chebyshev radius <= radius

    Sorting:
        physical radar-plane distance,
        then Chebyshev radius,
        then deterministic dr/dc.
    """

    out = []

    for dr in range(
        -radius,
        radius + 1,
    ):

        for dc in range(
            -radius,
            radius + 1,
        ):

            if dr == 0 and dc == 0:
                continue

            cheb = max(
                abs(dr),
                abs(dc),
            )

            if cheb > radius:
                continue

            d = math.hypot(
                dr * row_spacing,
                dc * col_spacing,
            )

            out.append(
                (
                    d,
                    cheb,
                    dr,
                    dc,
                )
            )

    out.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
            x[3],
        )
    )

    drs = np.array(
        [x[2] for x in out],
        dtype=np.int16,
    )

    dcs = np.array(
        [x[3] for x in out],
        dtype=np.int16,
    )

    dist = np.array(
        [x[0] for x in out],
        dtype=np.float32,
    )

    return drs, dcs, dist


@njit(cache=True)
def select_directed_neighbors(
    index_grid,
    rows,
    cols,
    drs,
    dcs,
    kmax,
):
    """
    For every point, choose at most kmax nearest
    existing points inside the ordered offset list.

    Returns directed p -> q selections.
    They are converted to an undirected unique graph
    outside Numba.
    """

    n = rows.size
    H, W = index_grid.shape

    max_edge = n * kmax

    uu = np.empty(
        max_edge,
        dtype=np.int32,
    )

    vv = np.empty(
        max_edge,
        dtype=np.int32,
    )

    chosen_count = np.zeros(
        n,
        dtype=np.uint8,
    )

    ne = 0

    for p in range(n):

        rp = rows[p]
        cp = cols[p]

        nk = 0

        for z in range(
            drs.size
        ):

            rr = rp + drs[z]
            cc = cp + dcs[z]

            if rr < 0 or rr >= H:
                continue

            if cc < 0 or cc >= W:
                continue

            q = index_grid[
                rr,
                cc
            ]

            if q < 0:
                continue

            uu[ne] = p
            vv[ne] = q
            ne += 1

            nk += 1

            if nk >= kmax:
                break

        chosen_count[p] = nk

    return (
        uu[:ne],
        vv[:ne],
        chosen_count,
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
def uf_union(
    parent,
    size,
    a,
    b,
):
    ra = uf_find(
        parent,
        a,
    )

    rb = uf_find(
        parent,
        b,
    )

    if ra == rb:
        return

    if size[ra] < size[rb]:
        ra, rb = rb, ra

    parent[rb] = ra
    size[ra] += size[rb]


@njit(cache=True)
def connected_roots(
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

    for k in range(
        u.size
    ):

        uf_union(
            parent,
            size,
            u[k],
            v[k],
        )

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


def unique_undirected(
    u,
    v,
):
    """
    Encode undirected int32 point pairs into uint64,
    sort/unique, then decode.
    """

    a = np.minimum(
        u,
        v,
    ).astype(
        np.uint64,
        copy=False,
    )

    b = np.maximum(
        u,
        v,
    ).astype(
        np.uint64,
        copy=False,
    )

    key = (
        (a << np.uint64(32))
        |
        b
    )

    key = np.unique(
        key
    )

    uu = (
        key
        >> np.uint64(32)
    ).astype(
        np.int32
    )

    vv = (
        key
        &
        np.uint64(0xFFFFFFFF)
    ).astype(
        np.int32
    )

    return uu, vv


def graph_stats(
    n,
    u,
    v,
):
    roots = connected_roots(
        n,
        u,
        v,
    )

    _, counts = np.unique(
        roots,
        return_counts=True,
    )

    order = np.sort(
        counts
    )[::-1]

    ncomp = int(
        counts.size
    )

    largest = int(
        order[0]
    )

    second = (
        int(order[1])
        if ncomp > 1
        else 0
    )

    outside = int(
        n - largest
    )

    singleton = int(
        np.count_nonzero(
            counts == 1
        )
    )

    degree = (
        np.bincount(
            u,
            minlength=n,
        )
        +
        np.bincount(
            v,
            minlength=n,
        )
    )

    return {
        "edges": int(
            u.size
        ),

        "components": ncomp,

        "largest": largest,

        "largest_fraction":
            float(
                largest / n
            ),

        "second_largest":
            second,

        "outside_largest":
            outside,

        "outside_fraction":
            float(
                outside / n
            ),

        "singletons":
            singleton,

        "degree_min":
            int(
                degree.min()
            ),

        "degree_median":
            float(
                np.median(degree)
            ),

        "degree_mean":
            float(
                degree.mean()
            ),

        "degree_max":
            int(
                degree.max()
            ),

        "degree_zero":
            int(
                np.count_nonzero(
                    degree == 0
                )
            ),

        "degree_ge3":
            int(
                np.count_nonzero(
                    degree >= 3
                )
            ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--radius",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--k-values",
        default="4,6,8,10,12",
    )

    args = ap.parse_args()

    kvals = [
        int(x.strip())
        for x in
        args.k_values.split(",")
        if x.strip()
    ]

    if not kvals:
        raise ValueError(
            "No K values."
        )

    if min(kvals) <= 0:
        raise ValueError(
            "K must be >0."
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
        / "processing"
    )

    pps_dir = (
        outroot
        / "point_phase_stack"
    )

    outdir = (
        outroot
        / "spatial_sparse_graph_quality"
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
            "rows/cols mismatch"
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

    # ========================================================
    # Physical radar-grid spacing
    # ========================================================

    geometry_par = cfg_get(
        cfg,
        "phase_correction.radar_height.geometry_par",
        None,
    )

    row_spacing = 1.0
    col_spacing = 1.0
    spacing_source = "pixel_units"

    if geometry_par:

        gp = Path(
            geometry_par
        ).expanduser()

        if gp.exists():

            rg = read_gamma_value(
                gp,
                "range_pixel_spacing",
            )

            az = read_gamma_value(
                gp,
                "azimuth_pixel_spacing",
            )

            if (
                rg is not None
                and
                az is not None
                and
                rg > 0
                and
                az > 0
            ):

                # row = azimuth
                # col = range
                row_spacing = float(az)
                col_spacing = float(rg)

                spacing_source = str(gp)

    drs, dcs, offset_dist = build_offsets(
        args.radius,
        row_spacing,
        col_spacing,
    )

    print("=" * 92)
    print(
        "Step 08g - Sparse local spatial-graph quality"
    )
    print("=" * 92)

    print(
        f"config                     : {config_path}"
    )

    print(
        f"points                     : {npoint:,}"
    )

    print(
        f"lookup grid                : {H} x {W}"
    )

    print(
        f"local candidate radius     : R<={args.radius}"
    )

    print(
        f"K values                   : {kvals}"
    )

    print(
        f"row spacing                : "
        f"{row_spacing:.6f}"
    )

    print(
        f"column spacing             : "
        f"{col_spacing:.6f}"
    )

    print(
        f"spacing source             : "
        f"{spacing_source}"
    )

    print()
    print(
        "Neighbor ordering:"
    )

    print(
        "  physical radar-plane distance first,"
    )

    print(
        "  constrained to Chebyshev R<=4."
    )

    print()
    print("=" * 92)
    print(
        "  K | unique edges | components | "
        "largest component       | outside     | "
        "degree min/med/mean/max"
    )
    print("=" * 92)

    results = []

    for K in kvals:

        du, dv, chosen = (
            select_directed_neighbors(
                index_grid,
                rows,
                cols,
                drs,
                dcs,
                K,
            )
        )

        u, v = unique_undirected(
            du,
            dv,
        )

        del du, dv

        stat = graph_stats(
            npoint,
            u,
            v,
        )

        stat[
            "K"
        ] = int(K)

        stat[
            "points_with_lt_K_candidates"
        ] = int(
            np.count_nonzero(
                chosen < K
            )
        )

        stat[
            "selected_neighbor_count_min"
        ] = int(
            chosen.min()
        )

        stat[
            "selected_neighbor_count_median"
        ] = float(
            np.median(chosen)
        )

        stat[
            "selected_neighbor_count_max"
        ] = int(
            chosen.max()
        )

        results.append(
            stat
        )

        print(
            f" {K:2d} | "
            f"{stat['edges']:12,d} | "
            f"{stat['components']:10,d} | "
            f"{stat['largest']:10,d} "
            f"({100*stat['largest_fraction']:7.3f}%) | "
            f"{stat['outside_largest']:8,d} | "
            f"{stat['degree_min']:2d}/"
            f"{stat['degree_median']:4.1f}/"
            f"{stat['degree_mean']:5.2f}/"
            f"{stat['degree_max']:2d}"
        )

        print(
            f"      points with fewer than "
            f"{K} local candidates: "
            f"{stat['points_with_lt_K_candidates']:,}"
        )

        del u, v, chosen

        gc.collect()

    # ========================================================
    # Milestones
    # ========================================================

    print()
    print("=" * 92)
    print(
        "Sparse-graph connectivity milestones"
    )
    print("=" * 92)

    for target in [
        0.99,
        0.995,
        0.999,
    ]:

        match = None

        for s in results:

            if (
                s[
                    "largest_fraction"
                ]
                >= target
            ):

                match = s
                break

        if match is None:

            print(
                f">={100*target:6.2f}% largest "
                f"component : not reached"
            )

        else:

            print(
                f">={100*target:6.2f}% largest "
                f"component : K={match['K']} "
                f"({match['edges']:,} edges)"
            )

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / "sparse_local_graph_sweep.csv"
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

    summary = {
        "format":
            "pyPSDS-GAMMA-sparse-local-graph-quality-v1.0",

        "points":
            int(npoint),

        "local_chebyshev_radius":
            int(args.radius),

        "neighbor_order":
            "physical radar-plane distance",

        "row_spacing":
            float(row_spacing),

        "column_spacing":
            float(col_spacing),

        "spacing_source":
            spacing_source,

        "results":
            results,
    }

    json_path = (
        outdir
        / "sparse_local_graph_sweep.json"
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
        f"CSV                       : {csv_path}"
    )

    print(
        f"manifest                  : {json_path}"
    )

    print()
    print(
        "STEP 08g STATUS: PASS"
    )


if __name__ == "__main__":
    main()
