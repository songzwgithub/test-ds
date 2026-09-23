#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.component_graph import build_component_csr
from pypsds.context import open_from_config


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Freeze hierarchical component-forest registration policy. Local R4 "
            "components remain independently unwrapped; offsets are propagated "
            "parent-to-child only inside locality-supported component forests. "
            "Detached forests are retained for QA but are not assigned an "
            "unsupported global 2pi gauge."
        )
    )
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg, config_path, paths, stack, _ = open_from_config(args.config)
    outroot = Path(paths.output_dir) / "processing"
    graphdir = outroot / "spatial_graph"
    anchor_dir = outroot / "spatial_graph_two_anchor_quality"
    outdir = outroot / "unwrap_component_policy"
    outdir.mkdir(parents=True, exist_ok=True)

    graph_manifest_path = graphdir / "spatial_graph_manifest.json"
    if not graph_manifest_path.is_file():
        raise FileNotFoundError(graph_manifest_path)
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    graph_core = graph_manifest.get("core", {})
    core_radius = int(graph_core["chebyshev_radius_pixels"])
    selected_local_k = int(graph_core["nearest_neighbors"])
    expected_residual_components = int(
        graph_manifest.get("residual_anchors", {}).get("components", -1)
    )

    rows = np.load(outroot / "point_phase_stack" / "rows.npy", mmap_mode="r")
    cols = np.load(outroot / "point_phase_stack" / "cols.npy", mmap_mode="r")
    component = np.load(graphdir / "local_component.npy", mmap_mode="r").astype(
        np.int32, copy=False
    )
    anchor_u = np.load(graphdir / "anchor_u.npy", mmap_mode="r").astype(
        np.int32, copy=False
    )
    anchor_v = np.load(graphdir / "anchor_v.npy", mmap_mode="r").astype(
        np.int32, copy=False
    )
    anchor_class = np.load(graphdir / "anchor_class.npy", mmap_mode="r").astype(
        np.uint8, copy=False
    )
    anchor_radius = np.load(graphdir / "anchor_radius.npy", mmap_mode="r").astype(
        np.int32, copy=False
    )
    anchor_distance = np.load(
        graphdir / "anchor_distance_m.npy", mmap_mode="r"
    ).astype(np.float32, copy=False)

    npoint = int(component.size)
    if rows.size != npoint or cols.size != npoint:
        raise RuntimeError("Point coordinate/component mismatch")
    if not (
        anchor_u.size
        == anchor_v.size
        == anchor_class.size
        == anchor_radius.size
        == anchor_distance.size
    ):
        raise RuntimeError("Anchor-array length mismatch")

    comp_counts = np.bincount(component)
    ncomp = int(comp_counts.size)
    if expected_residual_components >= 0 and ncomp - 1 != expected_residual_components:
        raise RuntimeError(
            "Residual component count does not match spatial_graph_manifest.json: "
            f"{ncomp-1} != {expected_residual_components}"
        )
    main_component = int(np.argmax(comp_counts))
    main_count = int(comp_counts[main_component])
    main_mask = component == main_component

    print("=" * 96)
    print("Hierarchical component-forest unwrapping policy")
    print("=" * 96)
    print(f"config                    : {config_path}")
    print(f"points                    : {npoint:,}")
    print(f"local components          : {ncomp:,}")
    print(f"global root component     : {main_component}")
    print(f"global root points        : {main_count:,} ({100*main_count/npoint:.4f}%)")
    print(f"component anchors         : {anchor_u.size:,}")

    # assess_spatial_anchor_quality writes anchor_u on the child component and
    # anchor_v on its tree parent. Exact R4 partition parity ensures the same
    # point-pair direction is valid after the sparse local graph relabel.
    anchor_child = component[np.asarray(anchor_u, dtype=np.int64)]
    anchor_parent = component[np.asarray(anchor_v, dtype=np.int64)]
    if np.any(anchor_child == anchor_parent):
        raise RuntimeError("Anchor edge is internal to one local component")

    component_parent = np.full(ncomp, -2, dtype=np.int32)
    component_anchor_count = np.zeros(ncomp, dtype=np.uint8)
    component_edge_tier = np.zeros(ncomp, dtype=np.uint8)

    for aid in range(anchor_u.size):
        child = int(anchor_child[aid])
        parent = int(anchor_parent[aid])
        old = int(component_parent[child])
        if old == -2:
            component_parent[child] = parent
        elif old != parent:
            raise RuntimeError(
                f"Component {child} has inconsistent parents {old}/{parent}"
            )
        component_anchor_count[child] += 1
        if anchor_class[aid] > component_edge_tier[child]:
            component_edge_tier[child] = anchor_class[aid]

    # Components never appearing as a child are roots of locality-supported
    # forests.  Only the forest rooted at the largest local component has the
    # global phase gauge. Other roots stay detached instead of receiving a
    # physically unsupported long bridge.
    roots = np.flatnonzero(component_parent == -2).astype(np.int32)
    if roots.size < 1:
        raise RuntimeError("Component forest has no roots")
    component_parent[roots] = -1
    if main_component not in set(int(x) for x in roots):
        raise RuntimeError(
            f"Largest local component {main_component} is not a forest root"
        )

    nonroot = component_parent >= 0
    if np.any(component_anchor_count[nonroot] != 2):
        bad = np.flatnonzero(nonroot & (component_anchor_count != 2))
        raise RuntimeError(
            "Every non-root component must carry exactly two bridge anchors; "
            f"bad={bad[:10].tolist()} count={bad.size}"
        )
    if np.any(component_anchor_count[~nonroot] != 0):
        raise RuntimeError("Forest root unexpectedly has bridge anchors")

    # Mark degenerate duplicated-anchor edges as long/low-confidence.
    duplicate_component = np.zeros(ncomp, dtype=bool)
    anchor_csv = anchor_dir / "residual_two_anchor_quality.csv"
    if anchor_csv.is_file():
        with anchor_csv.open() as f:
            for r in csv.DictReader(f):
                if int(r.get("duplicate_anchor_pair", "0") or 0) == 0:
                    continue
                pid = int(r["anchor1_residual_point"])
                duplicate_component[int(component[pid])] = True
    component_edge_tier[duplicate_component] = np.maximum(
        component_edge_tier[duplicate_component], np.uint8(3)
    )

    children = [[] for _ in range(ncomp)]
    for child in range(ncomp):
        parent = int(component_parent[child])
        if parent >= 0:
            children[parent].append(child)

    component_depth = np.full(ncomp, -1, dtype=np.int32)
    component_forest_root = np.full(ncomp, -1, dtype=np.int32)
    component_path_tier = np.zeros(ncomp, dtype=np.uint8)

    # Detached roots and all descendants are tier 3 by construction because
    # they are not globally registered. The tier is a confidence/status flag;
    # their local unwrapping products remain available for QA.
    queue = []
    for root0 in sorted(int(x) for x in roots):
        component_depth[root0] = 0
        component_forest_root[root0] = root0
        component_path_tier[root0] = np.uint8(0 if root0 == main_component else 3)
        queue.append(root0)

    head = 0
    while head < len(queue):
        parent = queue[head]
        head += 1
        for child in sorted(children[parent]):
            component_depth[child] = component_depth[parent] + 1
            component_forest_root[child] = component_forest_root[parent]
            component_path_tier[child] = np.uint8(
                max(
                    int(component_path_tier[parent]),
                    int(component_edge_tier[child]),
                )
            )
            queue.append(child)

    if len(queue) != ncomp or np.any(component_depth < 0):
        raise RuntimeError("Component parent graph is not a complete forest")

    global_forest_mask = component_forest_root == main_component
    global_component_count = int(np.count_nonzero(global_forest_mask))
    global_point_count = int(comp_counts[global_forest_mask].sum())
    detached_root_count = int(roots.size - 1)

    # One O(Npoint) counting-sort pass. This is the critical large-scene
    # performance contract and replaces repeated np.where(component == c).
    print("building component CSR point index (single O(N) pass)...", flush=True)
    component_point_offsets, component_point_order = build_component_csr(
        component, ncomp
    )
    if int(component_point_offsets[-1]) != npoint:
        raise RuntimeError("component CSR point count mismatch")

    point_tier = component_path_tier[component]
    if np.any(point_tier[main_mask] != 0):
        raise RuntimeError("Global root component received residual tier")
    if np.any((~main_mask) & (point_tier == 0)):
        raise RuntimeError("Residual points missing path tier")

    # Group anchor IDs once; no O(Nanchor*Ncomponent) scans.
    anchor_order = np.argsort(anchor_child, kind="stable").astype(
        np.int32, copy=False
    )
    anchor_counts = np.bincount(anchor_child, minlength=ncomp).astype(np.int64)
    anchor_offsets = np.zeros(ncomp + 1, dtype=np.int64)
    np.cumsum(anchor_counts, out=anchor_offsets[1:])

    component_rows = []
    tier_components = {1: 0, 2: 0, 3: 0}
    tier_points = {1: 0, 2: 0, 3: 0}

    order_by_depth = np.lexsort(
        (np.arange(ncomp, dtype=np.int32), component_depth, component_forest_root)
    )
    for comp0 in order_by_depth:
        comp = int(comp0)
        if comp == main_component:
            continue
        parent = int(component_parent[comp])
        depth = int(component_depth[comp])
        point_count = int(comp_counts[comp])
        tier = int(component_path_tier[comp])
        edge_tier = int(component_edge_tier[comp])
        is_root = parent < 0
        if tier not in (1, 2, 3):
            raise RuntimeError(f"Unexpected component path tier {tier} for {comp}")

        a0 = int(anchor_offsets[comp])
        a1 = int(anchor_offsets[comp + 1])
        aids = anchor_order[a0:a1]
        if is_root:
            if aids.size != 0:
                raise RuntimeError(f"Detached root {comp} unexpectedly has anchors")
        elif aids.size != 2:
            raise RuntimeError(f"Component {comp} has {aids.size} anchors; expected 2")

        tier_components[tier] += 1
        tier_points[tier] += point_count

        row = {
            "component_id": comp,
            "parent_component_id": parent,
            "forest_root_component_id": int(component_forest_root[comp]),
            "is_detached_root": int(is_root),
            "in_global_gauge_forest": int(component_forest_root[comp] == main_component),
            "depth": depth,
            "point_count": point_count,
            "edge_tier": edge_tier,
            "unwrap_tier": tier,
            "unwrap_tier_name": {1: "normal", 2: "extended", 3: "long"}[tier],
            "duplicate_anchor_pair": int(duplicate_component[comp]),
            "anchor1_id": int(aids[0]) if aids.size else -1,
            "anchor1_class": int(anchor_class[aids[0]]) if aids.size else 0,
            "anchor1_radius": int(anchor_radius[aids[0]]) if aids.size else -1,
            "anchor1_distance_m": float(anchor_distance[aids[0]]) if aids.size else float("nan"),
            "anchor2_id": int(aids[1]) if aids.size else -1,
            "anchor2_class": int(anchor_class[aids[1]]) if aids.size else 0,
            "anchor2_radius": int(anchor_radius[aids[1]]) if aids.size else -1,
            "anchor2_distance_m": float(anchor_distance[aids[1]]) if aids.size else float("nan"),
        }
        component_rows.append(row)

    print()
    print("=" * 96)
    print("Hierarchical registration forest")
    print("=" * 96)
    print(f"forest roots               : {roots.size:,}")
    print(f"detached roots             : {detached_root_count:,}")
    print(f"global-gauge components    : {global_component_count:,}/{ncomp:,}")
    print(
        f"global-gauge points        : {global_point_count:,}/{npoint:,} "
        f"({100*global_point_count/npoint:.3f}%)"
    )
    print(f"forest max depth           : {int(component_depth.max())}")
    print(f"duplicate-anchor components: {int(duplicate_component.sum()):,}")
    for tier, name in ((1, "normal"), (2, "extended"), (3, "long")):
        print(
            f"{name:8s}: components={tier_components[tier]:,}, "
            f"points={tier_points[tier]:,}"
        )

    np.save(outdir / "local_component.npy", component)
    np.save(outdir / "point_unwrap_tier.npy", point_tier.astype(np.uint8, copy=False))
    np.save(outdir / "main_component_mask.npy", main_mask)
    np.save(outdir / "anchor_component.npy", anchor_child.astype(np.int32, copy=False))
    np.save(
        outdir / "anchor_parent_component.npy",
        anchor_parent.astype(np.int32, copy=False),
    )
    np.save(outdir / "component_parent.npy", component_parent)
    np.save(outdir / "component_depth.npy", component_depth)
    np.save(outdir / "component_forest_root.npy", component_forest_root)
    np.save(outdir / "component_edge_tier.npy", component_edge_tier)
    np.save(outdir / "component_path_tier.npy", component_path_tier)
    np.save(outdir / "component_anchor_count.npy", component_anchor_count)
    np.save(outdir / "component_duplicate_anchor.npy", duplicate_component)
    np.save(outdir / "component_point_offsets.npy", component_point_offsets)
    np.save(outdir / "component_point_order.npy", component_point_order)

    csv_path = outdir / "residual_component_policy.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(component_rows[0].keys()))
        w.writeheader()
        w.writerows(component_rows)

    manifest = {
        "format": "pyPSDS-GAMMA-unwrapping-component-policy-v2.0",
        "status": "FROZEN",
        "registration_strategy": "locality_constrained_hierarchical_component_forest",
        "points": npoint,
        "local_components": ncomp,
        "main_component": {
            "id": main_component,
            "points": main_count,
            "fraction": float(main_count / npoint),
        },
        "forest": {
            "roots": [int(x) for x in roots],
            "count": int(roots.size),
            "detached_roots": detached_root_count,
            "maximum_depth": int(component_depth.max()),
            "global_gauge_components": global_component_count,
            "global_gauge_points": global_point_count,
            "global_gauge_fraction": float(global_point_count / npoint),
            "duplicate_anchor_components": int(duplicate_component.sum()),
        },
        "residual": {
            "components": ncomp - 1,
            "points": int(npoint - main_count),
            "normal": {"components": tier_components[1], "points": tier_points[1]},
            "extended": {"components": tier_components[2], "points": tier_points[2]},
            "long": {"components": tier_components[3], "points": tier_points[3]},
        },
        "performance_contract": {
            "component_membership": "CSR counting sort; O(Npoint) build, O(component_size) access",
            "forbidden_pattern": "repeated np.where(local_component == component) full-array scans",
        },
        "policy": {
            "core_radius_pixels": core_radius,
            "selected_local_k": selected_local_k,
            "within_component_edges": f"R{core_radius}-K{selected_local_k} local edges only",
            "spatial_graph_manifest": str(graph_manifest_path),
            "anchors_used_during_core_unwrap": False,
            "anchor_role": "parent-child integer 2pi registration after local-component unwrap",
            "anchor_consistency": "two anchors per selected forest edge; duplicate witness forces long tier",
            "registration_order": "global root to leaves by component depth",
            "detached_forest_policy": "retain local solution for QA; do not assign global 2pi gauge",
        },
    }
    manifest_path = outdir / "unwrap_component_policy.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print()
    print(f"component policy table    : {csv_path}")
    print(f"component CSR order       : {outdir/'component_point_order.npy'}")
    print(f"component CSR offsets     : {outdir/'component_point_offsets.npy'}")
    print(f"manifest                  : {manifest_path}")
    print()
    print("STEP unwrap_policy STATUS: PASS / HIERARCHICAL COMPONENT FOREST FROZEN")


if __name__ == "__main__":
    main()
