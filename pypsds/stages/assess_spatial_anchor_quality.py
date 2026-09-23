#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from pypsds.component_graph import (
    build_voronoi_component_candidates,
    build_component_forest,
    candidate_groups,
    select_forest_anchors,
)
from pypsds.config import cfg_get
from pypsds.context import open_from_config
from pypsds.geometry.gamma_par import gamma_par_scalar, read_gamma_par
from pypsds.geometry.inputs import resolve_geometry_inputs


def _resolve_int(cfg, key, default, *, minimum=1):
    raw = cfg_get(cfg, key, default)
    if raw in (None, "", "auto"):
        raw = default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _q(values):
    values = np.asarray(values)
    if values.size == 0:
        return []
    return [float(x) for x in np.quantile(values, [0, .25, .5, .75, .9, .95, .99, 1])]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build a scalable component-to-component bridge tree for the R=4 "
            "PS/DS spatial components. The candidate graph is derived from a "
            "global chessboard Voronoi partition and reduced to a deterministic "
            "locality-constrained minimum spanning forest. The forest containing "
            "the largest component defines the global-gauge domain; unsupported "
            "long bridges are never invented merely to force connectivity."
        )
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--block-rows", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--anchor-pool", type=int, default=None)
    ap.add_argument("--max-radius", type=int, default=None)
    ap.add_argument("--max-distance-m", type=float, default=None)
    args = ap.parse_args()

    cfg, config_path, paths, stack, _ = open_from_config(args.config)
    outroot = Path(paths.output_dir) / "processing"
    pps_dir = outroot / "point_phase_stack"
    quality_dir = outroot / "spatial_graph_residual_quality"
    outdir = outroot / "spatial_graph_two_anchor_quality"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = np.load(pps_dir / "rows.npy", mmap_mode="r").astype(np.int32, copy=False)
    cols = np.load(pps_dir / "cols.npy", mmap_mode="r").astype(np.int32, copy=False)
    labels = np.load(quality_dir / "component_label_r4.npy", mmap_mode="r").astype(np.int32, copy=False)

    npoint = int(rows.size)
    if cols.size != npoint or labels.size != npoint:
        raise RuntimeError("rows/cols/component length mismatch")
    if npoint == 0:
        raise RuntimeError("empty point set")

    ncomp = int(labels.max()) + 1
    component_sizes = np.bincount(labels, minlength=ncomp).astype(np.int64, copy=False)
    if np.count_nonzero(component_sizes) != ncomp:
        raise RuntimeError("component labels are not contiguous")
    root = int(np.argmax(component_sizes))

    geometry = resolve_geometry_inputs(cfg, paths)
    radar_par = Path(geometry.reference_rslc_par).resolve()
    par = read_gamma_par(radar_par)
    row_spacing = float(gamma_par_scalar(par, "azimuth_pixel_spacing"))
    col_spacing = float(gamma_par_scalar(par, "range_pixel_spacing"))
    if row_spacing <= 0 or col_spacing <= 0:
        raise RuntimeError("invalid reference-RSLC pixel spacing")

    cpu = os.cpu_count() or 1
    block_rows = args.block_rows or _resolve_int(
        cfg, "runtime.component_bridge_block_rows", 128
    )
    workers = args.workers or _resolve_int(
        cfg, "runtime.component_bridge_workers", min(16, cpu)
    )
    workers = min(workers, cpu)
    anchor_pool = args.anchor_pool or _resolve_int(
        cfg, "runtime.component_bridge_anchor_pool", 8, minimum=2
    )
    max_radius = args.max_radius or _resolve_int(
        cfg, "runtime.component_bridge_max_radius", 30, minimum=5
    )
    cfg_max_distance = cfg_get(cfg, "runtime.component_bridge_max_distance_m", None)
    max_distance_m = args.max_distance_m
    if max_distance_m is None and cfg_max_distance not in (None, "", "auto"):
        max_distance_m = float(cfg_max_distance)
    if max_distance_m is not None and max_distance_m <= 0:
        raise ValueError("component bridge max distance must be > 0")

    H = int(rows.max()) + 1
    W = int(cols.max()) + 1
    grid_gib = H * W * 4 / (1024 ** 3)
    nearest_gib = 2 * grid_gib

    print("=" * 96)
    print("Hierarchical component bridge quality")
    print("=" * 96)
    print(f"config                     : {config_path}")
    print(f"points                     : {npoint:,}")
    print(f"R=4 components             : {ncomp:,}")
    print(f"root component             : {root} ({component_sizes[root]:,} points)")
    print(f"radar extent               : {H} x {W}")
    print(f"radar spacing              : row={row_spacing:.6f} m, col={col_spacing:.6f} m")
    print(f"candidate metric           : chessboard Voronoi + physical-distance tie break")
    print(f"block rows / workers       : {block_rows} / {workers}")
    print(f"candidate anchors/pair     : {anchor_pool}")
    print(f"maximum bridge radius      : {max_radius} pixels")
    print(f"maximum bridge distance    : {max_distance_m if max_distance_m is not None else 'disabled'}")
    print(f"index-grid estimate        : {grid_gib:.2f} GiB")
    print(f"nearest-index estimate     : {nearest_gib:.2f} GiB")
    print()

    t0 = time.perf_counter()
    (ca, cb, pa, pb, radius, distance), index_grid = build_voronoi_component_candidates(
        rows,
        cols,
        labels,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        block_rows=block_rows,
        workers=workers,
        keep_per_pair=anchor_pool,
    )
    t_candidates = time.perf_counter() - t0

    groups = candidate_groups(ca, cb, radius, distance, ncomp)
    starts, stops, edge_a, edge_b, edge_radius, edge_distance, support = groups
    npair = int(edge_a.size)
    if npair == 0:
        raise RuntimeError("component candidate graph is empty")

    print(f"candidate component pairs  : {npair:,}")
    print(f"candidate anchor records   : {ca.size:,}")
    print(f"candidate build seconds    : {t_candidates:.2f}")

    t1 = time.perf_counter()
    forest, groups = build_component_forest(
        ca,
        cb,
        radius,
        distance,
        ncomp,
        component_sizes,
        root,
        max_radius=max_radius,
        max_distance_m=max_distance_m,
    )
    anchors, synthetic_count, duplicate_count = select_forest_anchors(
        forest,
        groups,
        ca,
        cb,
        pa,
        pb,
        radius,
        distance,
        component_sizes,
        index_grid,
        rows,
        cols,
        labels,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        core_radius=4,
        max_anchor_radius=max_radius,
    )
    t_tree = time.perf_counter() - t1

    if len(anchors) != forest.selected_edge_count:
        raise RuntimeError(
            f"forest anchor count mismatch: {len(anchors)} != {forest.selected_edge_count}"
        )

    starts, stops, edge_a, edge_b, edge_radius, edge_distance, support = groups
    out_rows = []
    edge_radii = []
    edge_distances = []
    weak_edges = 0
    long_edges = 0

    for child, parent, depth, aa, selected_weak in anchors:
        eid = int(forest.edge_group_index[child])
        s0 = int(starts[eid])
        e0 = int(stops[eid])
        pool_count = e0 - s0
        if selected_weak:
            weak_edges += 1

        a1, a2 = aa[0], aa[1]
        r1 = int(a1[2])
        r2 = int(a2[2])
        d1 = float(a1[3])
        d2 = float(a2[3])
        maxr = max(r1, r2)
        if maxr > 20:
            long_edges += 1
        edge_radii.append(maxr)
        edge_distances.append(max(d1, d2))

        pool_parent_points = set()
        pool_child_points = set()
        for k in range(s0, e0):
            if int(ca[k]) == child and int(cb[k]) == parent:
                pool_child_points.add(int(pa[k]))
                pool_parent_points.add(int(pb[k]))
            elif int(ca[k]) == parent and int(cb[k]) == child:
                pool_child_points.add(int(pb[k]))
                pool_parent_points.add(int(pa[k]))

        r3 = int(radius[s0 + 2]) if pool_count >= 3 else -1
        duplicate = int(a1[0] == a2[0] and a1[1] == a2[1])
        row = {
            "component_label": int(child),
            "component_size": int(component_sizes[child]),
            "parent_component_label": int(parent),
            "parent_component_size": int(component_sizes[parent]),
            "forest_root_component": int(forest.forest_root[child]),
            "component_depth": int(depth),
            "tree_edge_weak": int(bool(selected_weak)),
            "candidate_pool_count": int(pool_count),
            # Backward-compatible names retained for visualizers/old readers.
            "crossing_edges_Rmax": int(pool_count),
            "distinct_main_anchors_Rmax": int(len(pool_parent_points)),
            "distinct_parent_anchors_pool": int(len(pool_parent_points)),
            "distinct_child_anchors_pool": int(len(pool_child_points)),
            "min_radius_1_anchor": int(r1),
            "min_radius_2_anchor": int(r2) if not duplicate else -1,
            "min_radius_3_anchor": int(r3),
            "duplicate_anchor_pair": duplicate,
        }

        for k, anchor in enumerate((a1, a2), start=1):
            cp, pp, rad, dist_m, synthetic = anchor
            row[f"anchor{k}_radius"] = int(rad)
            row[f"anchor{k}_distance_m"] = float(dist_m)
            row[f"anchor{k}_residual_point"] = int(cp)
            row[f"anchor{k}_residual_row"] = int(rows[cp])
            row[f"anchor{k}_residual_col"] = int(cols[cp])
            # Legacy main_* columns now mean the tree-parent component.
            row[f"anchor{k}_main_point"] = int(pp)
            row[f"anchor{k}_main_row"] = int(rows[pp])
            row[f"anchor{k}_main_col"] = int(cols[pp])
            row[f"anchor{k}_parent_point"] = int(pp)
            row[f"anchor{k}_parent_row"] = int(rows[pp])
            row[f"anchor{k}_parent_col"] = int(cols[pp])
            row[f"anchor{k}_synthetic"] = int(bool(synthetic))

        out_rows.append(row)

    out_rows.sort(key=lambda x: (x["forest_root_component"], x["component_depth"], x["component_label"]))

    forest_roots = np.flatnonzero(forest.parent < 0).astype(np.int32)
    global_mask = forest.forest_root == root
    global_components = int(np.count_nonzero(global_mask))
    global_points = int(component_sizes[global_mask].sum())
    global_fraction = float(global_points / npoint)
    csv_path = outdir / "residual_two_anchor_quality.csv"
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "component_label", "component_size", "parent_component_label",
        "parent_component_size", "forest_root_component", "component_depth",
        "tree_edge_weak", "candidate_pool_count", "crossing_edges_Rmax",
        "distinct_main_anchors_Rmax", "distinct_parent_anchors_pool",
        "distinct_child_anchors_pool", "min_radius_1_anchor",
        "min_radius_2_anchor", "min_radius_3_anchor", "duplicate_anchor_pair",
        "anchor1_radius", "anchor1_distance_m", "anchor1_residual_point",
        "anchor1_residual_row", "anchor1_residual_col", "anchor1_main_point",
        "anchor1_main_row", "anchor1_main_col", "anchor1_parent_point",
        "anchor1_parent_row", "anchor1_parent_col", "anchor1_synthetic",
        "anchor2_radius", "anchor2_distance_m", "anchor2_residual_point",
        "anchor2_residual_row", "anchor2_residual_col", "anchor2_main_point",
        "anchor2_main_row", "anchor2_main_col", "anchor2_parent_point",
        "anchor2_parent_row", "anchor2_parent_col", "anchor2_synthetic",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    np.save(outdir / "component_parent_r4.npy", forest.parent.astype(np.int32))
    np.save(outdir / "component_depth_r4.npy", forest.depth.astype(np.int32))
    np.save(outdir / "component_forest_root_r4.npy", forest.forest_root.astype(np.int32))

    edge_radii_arr = np.asarray(edge_radii, dtype=np.int32)
    edge_distance_arr = np.asarray(edge_distances, dtype=np.float32)
    max_depth = int(forest.depth.max())

    summary = {
        "format": "pyPSDS-GAMMA-hierarchical-component-bridge-v2.0",
        "strategy": "global_voronoi_component_graph_plus_locality_constrained_msf",
        "distance": {
            "candidate_topology": "Chebyshev radar-grid Voronoi adjacency",
            "mst_primary": "bridge radius with support/singleton/physical-distance tie breaks",
            "tie_break": "physical radar-plane distance",
            "row_spacing_m": row_spacing,
            "col_spacing_m": col_spacing,
        },
        "points": npoint,
        "components": ncomp,
        "root_component": root,
        "root_points": int(component_sizes[root]),
        "forest_edges": int(forest.selected_edge_count),
        "forest_count": int(forest.forest_count),
        "forest_roots": [int(x) for x in forest_roots],
        "maximum_depth": max_depth,
        "global_gauge_forest": {
            "root_component": int(root),
            "components": global_components,
            "points": global_points,
            "point_fraction": global_fraction,
        },
        "bridge_limit": {
            "max_radius_pixels": int(max_radius),
            "max_distance_m": max_distance_m,
            "policy": "do_not_force unsupported long bridges",
        },
        "candidate_component_pairs": npair,
        "candidate_anchor_records": int(ca.size),
        "anchor_pool_per_pair": int(anchor_pool),
        "selected_weak_edges": int(weak_edges),
        "synthetic_second_anchors": int(synthetic_count),
        "duplicate_second_anchors": int(duplicate_count),
        "long_edges_R_gt_20": int(long_edges),
        "two_anchor": {
            "anchored_components": int(forest.selected_edge_count),
            "independent_two_anchor_components": int(forest.selected_edge_count - duplicate_count),
            "duplicate_anchor_components": int(duplicate_count),
            "minimum_radius": int(edge_radii_arr.min()) if edge_radii_arr.size else None,
            "median_radius": float(np.median(edge_radii_arr)) if edge_radii_arr.size else None,
            "maximum_radius": int(edge_radii_arr.max()) if edge_radii_arr.size else None,
        },
        "bridge_radius_quantiles": _q(edge_radii_arr),
        "bridge_distance_m_quantiles": _q(edge_distance_arr),
        "runtime_seconds": {
            "voronoi_candidates": t_candidates,
            "tree_and_anchor_selection": t_tree,
            "total": time.perf_counter() - t0,
        },
        "compatibility": {
            "legacy_filename": "residual_two_anchor_quality.csv",
            "legacy_main_columns_mean": "tree parent component, not necessarily the global root",
        },
    }

    json_path = outdir / "residual_two_anchor_quality.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print()
    print("=" * 96)
    print("Hierarchical component bridge summary")
    print("=" * 96)
    print(f"forest edges                : {forest.selected_edge_count:,}")
    print(f"forest count                : {forest.forest_count:,}")
    print(f"global-gauge forest         : {global_components:,} components, {global_points:,} points ({100*global_fraction:.3f}%)")
    print(f"forest max depth            : {max_depth}")
    print(f"weak candidate edges        : {weak_edges:,}")
    print(f"synthetic second anchors    : {synthetic_count:,}")
    print(f"duplicate second anchors    : {duplicate_count:,}")
    if edge_radii_arr.size:
        print(f"anchor R min/median/max     : {edge_radii_arr.min()} / {np.median(edge_radii_arr):.1f} / {edge_radii_arr.max()}")
        print(f"bridge distance max         : {edge_distance_arr.max():.2f} m")
    else:
        print("anchor R min/median/max     : none")
        print("bridge distance max         : none")
    print(f"component table             : {csv_path}")
    print(f"manifest                    : {json_path}")
    print()
    print("STEP spatial_anchor_quality STATUS: PASS")
    print("Component registration topology is hierarchical; unsupported long gaps remain detached forests.")


if __name__ == "__main__":
    main()
