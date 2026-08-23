#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

from pypsds.config import cfg_get
from pypsds.context import open_from_config


def read_gamma_value(path: Path, key: str):
    target = key.rstrip(":") + ":"

    for raw in path.read_text(errors="ignore").splitlines():
        f = raw.split()

        if not f or f[0] != target:
            continue

        for x in f[1:]:
            try:
                return float(x)
            except ValueError:
                pass

    return None


def build_offsets(radius, row_spacing, col_spacing):
    items = []

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):

            if dr == 0 and dc == 0:
                continue

            if max(abs(dr), abs(dc)) > radius:
                continue

            distance_m = math.hypot(
                dr * row_spacing,
                dc * col_spacing,
            )

            items.append(
                (
                    distance_m,
                    max(abs(dr), abs(dc)),
                    dr,
                    dc,
                )
            )

    items.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
            x[3],
        )
    )

    return (
        np.asarray([x[2] for x in items], dtype=np.int16),
        np.asarray([x[3] for x in items], dtype=np.int16),
    )


@njit(cache=True)
def choose_neighbors(
    index_grid,
    rows,
    cols,
    drs,
    dcs,
    K,
):
    n = rows.size
    H, W = index_grid.shape

    u = np.empty(n * K, dtype=np.int32)
    v = np.empty(n * K, dtype=np.int32)

    ne = 0

    for p in range(n):

        rp = rows[p]
        cp = cols[p]

        nk = 0

        for z in range(drs.size):

            rr = rp + drs[z]
            cc = cp + dcs[z]

            if rr < 0 or rr >= H:
                continue

            if cc < 0 or cc >= W:
                continue

            q = index_grid[rr, cc]

            if q < 0:
                continue

            u[ne] = p
            v[ne] = q
            ne += 1

            nk += 1

            if nk >= K:
                break

    return u[:ne], v[:ne]


def unique_undirected(u, v):
    a = np.minimum(u, v).astype(np.uint64)
    b = np.maximum(u, v).astype(np.uint64)

    key = (
        (a << np.uint64(32))
        | b
    )

    key = np.unique(key)

    u2 = (
        key >> np.uint64(32)
    ).astype(np.int32)

    v2 = (
        key & np.uint64(0xFFFFFFFF)
    ).astype(np.int32)

    return u2, v2


@njit(cache=True)
def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def get_roots(n, u, v):
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    for k in range(u.size):

        a = uf_find(parent, u[k])
        b = uf_find(parent, v[k])

        if a == b:
            continue

        if size[a] < size[b]:
            a, b = b, a

        parent[b] = a
        size[a] += size[b]

    roots = np.empty(n, dtype=np.int32)

    for i in range(n):
        roots[i] = uf_find(parent, i)

    return roots


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--neighbors", type=int, default=8)

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        _,
    ) = open_from_config(args.config)

    outroot = Path(paths.output_dir) / "processing"

    pps_dir = outroot / "point_phase_stack"

    old_quality = (
        outroot
        / "spatial_graph_residual_quality"
    )

    anchor_dir = (
        outroot
        / "spatial_graph_two_anchor_quality"
    )

    outdir = (
        outroot
        / "spatial_graph"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = np.load(
        pps_dir / "rows.npy"
    ).astype(np.int32)

    cols = np.load(
        pps_dir / "cols.npy"
    ).astype(np.int32)

    npoint = rows.size

    H = int(rows.max()) + 1
    W = int(cols.max()) + 1

    index_grid = np.full(
        (H, W),
        -1,
        dtype=np.int32,
    )

    index_grid[rows, cols] = np.arange(
        npoint,
        dtype=np.int32,
    )

    # ---------------------------------------------------------
    # GAMMA radar-grid spacing
    # ---------------------------------------------------------

    geometry_par = Path(
        cfg_get(
            cfg,
            "phase_correction.radar_height.geometry_par",
        )
    )

    range_spacing = read_gamma_value(
        geometry_par,
        "range_pixel_spacing",
    )

    azimuth_spacing = read_gamma_value(
        geometry_par,
        "azimuth_pixel_spacing",
    )

    if (
        range_spacing is None
        or
        azimuth_spacing is None
    ):
        raise RuntimeError(
            "Cannot read GAMMA pixel spacing."
        )

    # rows = azimuth; cols = range
    row_spacing = float(azimuth_spacing)
    col_spacing = float(range_spacing)

    print("=" * 88)
    print(
        "Production spatial point graph"
    )
    print("=" * 88)

    print(f"points                 : {npoint:,}")
    print(f"core radius            : {args.radius}")
    print(f"local neighbors K      : {args.neighbors}")

    print(
        f"radar spacing           : "
        f"row={row_spacing:.6f} m, "
        f"col={col_spacing:.6f} m"
    )

    # =========================================================
    # Sparse K=8 local graph
    # =========================================================

    drs, dcs = build_offsets(
        args.radius,
        row_spacing,
        col_spacing,
    )

    du, dv = choose_neighbors(
        index_grid,
        rows,
        cols,
        drs,
        dcs,
        args.neighbors,
    )

    core_u, core_v = unique_undirected(
        du,
        dv,
    )

    del du, dv

    roots = get_roots(
        npoint,
        core_u,
        core_v,
    )

    _, sparse_labels = np.unique(
        roots,
        return_inverse=True,
    )

    sparse_labels = sparse_labels.astype(
        np.int32
    )

    sparse_counts = np.bincount(
        sparse_labels
    )

    sparse_main_label = int(
        np.argmax(sparse_counts)
    )

    sparse_main = (
        sparse_labels
        ==
        sparse_main_label
    )

    print()
    print(
        f"local graph edges       : "
        f"{core_u.size:,}"
    )

    print(
        f"local components        : "
        f"{sparse_counts.size}"
    )

    print(
        f"local main component    : "
        f"{sparse_main.sum():,} "
        f"({100*sparse_main.mean():.4f}%)"
    )

    # =========================================================
    # Exact parity with full R<=4 topology
    # =========================================================

    full_labels = np.load(
        old_quality
        / "component_label_r4.npy"
    ).astype(np.int32)

    if full_labels.size != npoint:
        raise RuntimeError(
            "R4 component-label length mismatch."
        )

    ns = int(sparse_labels.max()) + 1
    nf = int(full_labels.max()) + 1

    s_min = np.full(
        ns,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )

    s_max = np.full(
        ns,
        -1,
        dtype=np.int32,
    )

    np.minimum.at(
        s_min,
        sparse_labels,
        full_labels,
    )

    np.maximum.at(
        s_max,
        sparse_labels,
        full_labels,
    )

    f_min = np.full(
        nf,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )

    f_max = np.full(
        nf,
        -1,
        dtype=np.int32,
    )

    np.minimum.at(
        f_min,
        full_labels,
        sparse_labels,
    )

    np.maximum.at(
        f_max,
        full_labels,
        sparse_labels,
    )

    partition_equal = bool(
        np.all(s_min == s_max)
        and
        np.all(f_min == f_max)
    )

    print()
    print(
        f"exact R4 partition parity: "
        f"{partition_equal}"
    )

    if not partition_equal:
        raise RuntimeError(
            "K=8 sparse graph does not exactly reproduce "
            "the full R<=4 component partition. "
            "Do not reuse residual anchors."
        )

    # =========================================================
    # Read the two anchors for every residual component
    # =========================================================

    anchor_csv = (
        anchor_dir
        / "residual_two_anchor_quality.csv"
    )

    anchor_u = []
    anchor_v = []
    anchor_radius = []

    with anchor_csv.open() as f:

        reader = csv.DictReader(f)

        for r in reader:

            for k in (1, 2):

                p = int(
                    r[
                        f"anchor{k}_residual_point"
                    ]
                )

                q = int(
                    r[
                        f"anchor{k}_main_point"
                    ]
                )

                rad = int(
                    r[
                        f"anchor{k}_radius"
                    ]
                )

                if (
                    p < 0
                    or q < 0
                    or rad < 0
                ):
                    raise RuntimeError(
                        "Missing residual anchor."
                    )

                anchor_u.append(p)
                anchor_v.append(q)
                anchor_radius.append(rad)

    anchor_u = np.asarray(
        anchor_u,
        dtype=np.int32,
    )

    anchor_v = np.asarray(
        anchor_v,
        dtype=np.int32,
    )

    anchor_radius = np.asarray(
        anchor_radius,
        dtype=np.uint8,
    )

    # 1 = ordinary <=12
    # 2 = extended 13..20
    # 3 = long >20
    anchor_class = np.ones(
        anchor_radius.size,
        dtype=np.uint8,
    )

    anchor_class[
        anchor_radius > 12
    ] = 2

    anchor_class[
        anchor_radius > 20
    ] = 3

    # Physical distance.
    adr = (
        rows[anchor_u]
        -
        rows[anchor_v]
    ).astype(np.float64)

    adc = (
        cols[anchor_u]
        -
        cols[anchor_v]
    ).astype(np.float64)

    anchor_distance_m = np.hypot(
        adr * row_spacing,
        adc * col_spacing,
    ).astype(np.float32)

    # =========================================================
    # Core local-edge distances
    # =========================================================

    cdr = (
        rows[core_u]
        -
        rows[core_v]
    ).astype(np.float64)

    cdc = (
        cols[core_u]
        -
        cols[core_v]
    ).astype(np.float64)

    core_distance_m = np.hypot(
        cdr * row_spacing,
        cdc * col_spacing,
    ).astype(np.float32)

    # =========================================================
    # Save production graph
    # =========================================================

    # Separate arrays make mmap possible later.
    np.save(
        outdir / "local_u.npy",
        core_u,
    )

    np.save(
        outdir / "local_v.npy",
        core_v,
    )

    np.save(
        outdir / "local_distance_m.npy",
        core_distance_m,
    )

    np.save(
        outdir / "local_component.npy",
        sparse_labels,
    )

    np.save(
        outdir / "main_component_mask.npy",
        sparse_main,
    )

    np.save(
        outdir / "anchor_u.npy",
        anchor_u,
    )

    np.save(
        outdir / "anchor_v.npy",
        anchor_v,
    )

    np.save(
        outdir / "anchor_radius.npy",
        anchor_radius,
    )

    np.save(
        outdir / "anchor_distance_m.npy",
        anchor_distance_m,
    )

    np.save(
        outdir / "anchor_class.npy",
        anchor_class,
    )

    n_normal = int(
        np.count_nonzero(
            anchor_class == 1
        )
    )

    n_extended = int(
        np.count_nonzero(
            anchor_class == 2
        )
    )

    n_long = int(
        np.count_nonzero(
            anchor_class == 3
        )
    )

    # =========================================================
    # Final graph connectivity
    # =========================================================

    final_u = np.concatenate(
        [
            core_u,
            anchor_u,
        ]
    )

    final_v = np.concatenate(
        [
            core_v,
            anchor_v,
        ]
    )

    final_roots = get_roots(
        npoint,
        final_u,
        final_v,
    )

    n_final_components = int(
        np.unique(
            final_roots
        ).size
    )

    del final_u, final_v

    print()
    print("=" * 88)
    print(
        "Production spatial graph summary"
    )
    print("=" * 88)

    print(
        f"local edges             : "
        f"{core_u.size:,}"
    )

    print(
        f"residual anchor edges   : "
        f"{anchor_u.size:,}"
    )

    print(
        f"  normal <=R12          : "
        f"{n_normal:,}"
    )

    print(
        f"  extended R13..20      : "
        f"{n_extended:,}"
    )

    print(
        f"  long R>20             : "
        f"{n_long:,}"
    )

    print(
        f"total production edges  : "
        f"{core_u.size + anchor_u.size:,}"
    )

    print(
        f"final components        : "
        f"{n_final_components}"
    )

    print(
        f"anchor R min/med/max    : "
        f"{anchor_radius.min()} / "
        f"{np.median(anchor_radius):.1f} / "
        f"{anchor_radius.max()}"
    )

    print(
        f"anchor distance max     : "
        f"{anchor_distance_m.max():.2f} m"
    )

    if n_final_components != 1:
        raise RuntimeError(
            "Production spatial graph is not connected."
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-production-spatial-graph-v1.0",

        "status":
            "FROZEN",

        "points":
            int(npoint),

        "core": {
            "chebyshev_radius_pixels":
                int(args.radius),

            "nearest_neighbors":
                int(args.neighbors),

            "neighbor_order":
                "physical radar-plane distance",

            "local_edges":
                int(core_u.size),

            "components_before_anchors":
                int(sparse_counts.size),

            "main_component_points":
                int(sparse_main.sum()),

            "main_component_fraction":
                float(sparse_main.mean()),

            "exact_full_R4_partition_parity":
                partition_equal,
        },

        "radar_spacing_m": {
            "row_azimuth":
                row_spacing,

            "column_range":
                col_spacing,
        },

        "residual_anchors": {
            "components":
                int(
                    sparse_counts.size - 1
                ),

            "edges":
                int(anchor_u.size),

            "normal_R_le_12":
                n_normal,

            "extended_R_13_20":
                n_extended,

            "long_R_gt_20":
                n_long,

            "radius_max":
                int(anchor_radius.max()),

            "distance_max_m":
                float(
                    anchor_distance_m.max()
                ),
        },

        "final_components":
            n_final_components,

        "total_edges":
            int(
                core_u.size
                +
                anchor_u.size
            ),
    }

    manifest_path = (
        outdir
        / "spatial_graph_manifest.json"
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
        f"output directory        : {outdir}"
    )

    print(
        f"manifest                : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP spatial_graph STATUS: PASS / SPATIAL GRAPH FROZEN"
    )


if __name__ == "__main__":
    main()
