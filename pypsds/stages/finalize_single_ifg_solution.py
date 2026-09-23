#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


TWOPI = 2.0 * np.pi


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def load_itab(path: Path, ndate: int):
    out = []
    for raw in path.read_text().splitlines():
        f = raw.split()
        if len(f) < 2:
            continue
        i = int(f[0]) - 1
        j = int(f[1]) - 1
        if not (0 <= i < ndate and 0 <= j < ndate):
            raise RuntimeError(f"Invalid ITAB line: {raw}")
        out.append((i, j))
    return out


def read_groups(path: Path):
    rows = []
    with path.open() as f:
        for gid, r in enumerate(csv.DictReader(f)):
            rows.append({
                "group_id": gid,
                "fragment_a": int(r["fragment_a"]),
                "fragment_b": int(r["fragment_b"]),
                "mode_shift": int(r["mode_shift_b_minus_a"]),
                "edge_count": int(r["edge_count"]),
                "mode_count": int(r["mode_count"]),
                "consensus_ratio": float(r["consensus_ratio"]),
                "exact_consensus": int(r["exact_consensus"]),
                "median_abs_gradient_rad": float(r["median_abs_gradient_rad"]),
                "median_distance_m": float(r["median_distance_m"]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair-id", type=int, default=19)
    args = ap.parse_args()

    cfg, config_path, paths, stack, _ = open_from_config(args.config)
    root = Path(paths.output_dir) / "processing"
    pps = root / "point_phase_stack"
    network = root / "network"
    graph = root / "spatial_graph"
    policy = root / "unwrap_component_policy"
    qualitysafe = root / "safe_fragment_integer_quality"
    outdir = root / "single_ifg_robust_solution"
    outdir.mkdir(parents=True, exist_ok=True)

    phase = np.load(pps / "phase_rad.npy", mmap_mode="r")
    npoint, ndate = phase.shape
    temporal_edges = load_itab(network / "network.itab", ndate)
    pair_id = args.pair_id
    if not (1 <= pair_id <= len(temporal_edges)):
        raise RuntimeError("Invalid pair ID")
    ti, tj = temporal_edges[pair_id - 1]
    tag = f"pair{pair_id:03d}_{stack.dates[ti]}_{stack.dates[tj]}"

    U = np.load(qualitysafe / f"{tag}_consensus_unwrapped.npy").astype(np.float64, copy=True)
    safe_fragment = np.load(qualitysafe / f"{tag}_safe_fragment.npy").astype(np.int32, copy=False)
    fragment_shift = np.load(qualitysafe / f"{tag}_fragment_shift.npy").astype(np.int32, copy=False)
    groups = read_groups(qualitysafe / f"{tag}_fragment_pair_consensus.csv")

    local_component = np.load(policy / "local_component.npy", mmap_mode="r").astype(np.int32, copy=False)
    point_tier = np.load(policy / "point_unwrap_tier.npy", mmap_mode="r").astype(np.uint8, copy=False)
    component_parent = np.load(policy / "component_parent.npy").astype(np.int32, copy=False)
    component_depth = np.load(policy / "component_depth.npy").astype(np.int32, copy=False)
    component_forest_root = np.load(policy / "component_forest_root.npy").astype(np.int32, copy=False)
    component_duplicate_anchor = np.load(policy / "component_duplicate_anchor.npy", mmap_mode="r").astype(bool, copy=False)
    component_offsets = np.load(policy / "component_point_offsets.npy", mmap_mode="r").astype(np.int64, copy=False)
    component_order = np.load(policy / "component_point_order.npy", mmap_mode="r").astype(np.int32, copy=False)
    anchor_component = np.load(policy / "anchor_component.npy").astype(np.int32, copy=False)
    anchor_parent_component = np.load(policy / "anchor_parent_component.npy").astype(np.int32, copy=False)

    anchor_u = np.load(graph / "anchor_u.npy", mmap_mode="r").astype(np.int32, copy=False)
    anchor_v = np.load(graph / "anchor_v.npy", mmap_mode="r").astype(np.int32, copy=False)
    anchor_class = np.load(graph / "anchor_class.npy", mmap_mode="r").astype(np.uint8, copy=False)
    local_u = np.load(graph / "local_u.npy", mmap_mode="r")
    local_v = np.load(graph / "local_v.npy", mmap_mode="r")

    ncomp = int(component_parent.size)
    if (
        component_depth.size != ncomp
        or component_forest_root.size != ncomp
        or component_duplicate_anchor.size != ncomp
        or component_offsets.size != ncomp + 1
    ):
        raise RuntimeError("component policy array mismatch")
    if component_order.size != npoint or int(component_offsets[-1]) != npoint:
        raise RuntimeError("component CSR point index mismatch")
    if not (anchor_u.size == anchor_v.size == anchor_component.size == anchor_parent_component.size):
        raise RuntimeError("anchor policy/graph length mismatch")

    roots = np.flatnonzero(component_parent < 0).astype(np.int32)
    if roots.size < 1:
        raise RuntimeError("component forest has no roots")
    comp_counts = np.diff(component_offsets)
    main_component = int(np.argmax(comp_counts))
    if main_component not in set(int(x) for x in roots):
        raise RuntimeError("largest component is not a forest root")
    global_forest_components = int(
        np.count_nonzero(component_forest_root == main_component)
    )

    # Wrapped IFG.
    ifg = wrap(
        np.asarray(phase[:, tj], dtype=np.float64)
        - np.asarray(phase[:, ti], dtype=np.float64)
    )

    # 1. Fragment-pair global consistency.
    accepted_groups = []
    rejected_groups = []
    for r in groups:
        a = r["fragment_a"]
        b = r["fragment_b"]
        predicted = int(fragment_shift[b] - fragment_shift[a])
        observed = r["mode_shift"]
        residual = observed - predicted
        rr = dict(r)
        rr["predicted_shift"] = predicted
        rr["integer_residual"] = residual
        rr["status"] = "accepted_consistent" if residual == 0 else "rejected_cycle_outlier"
        (accepted_groups if residual == 0 else rejected_groups).append(rr)

    # 2. Verify local spatial-edge consistency.
    u = np.asarray(local_u, dtype=np.int32)
    v = np.asarray(local_v, dtype=np.int32)
    g = wrap(ifg[v] - ifg[u])
    delta = U[v] - U[u] - g
    jump = np.rint(delta / TWOPI).astype(np.int32)
    safe = np.abs(g) <= np.pi / 2
    same_safe_fragment = safe_fragment[u] == safe_fragment[v]
    bad = jump != 0
    safe_bad = int(np.count_nonzero(safe & bad))
    unsafe_within_bad = int(np.count_nonzero((~safe) & same_safe_fragment & bad))
    unsafe_cross_bad = int(np.count_nonzero((~safe) & (~same_safe_fragment) & bad))
    del g, delta, jump, safe, same_safe_fragment, bad

    # 3. Hierarchical child->parent component registration.
    registered = np.zeros(npoint, dtype=bool)
    registered_component = np.zeros(ncomp, dtype=bool)
    root0 = int(component_offsets[main_component])
    root1 = int(component_offsets[main_component + 1])
    root_points = np.asarray(component_order[root0:root1], dtype=np.int32)
    registered[root_points] = True
    registered_component[main_component] = True

    # Group anchors once (Nanchor ~ 2*Ncomponent), never scan per component.
    anchor_order = np.argsort(anchor_component, kind="stable").astype(np.int32, copy=False)
    anchor_counts = np.bincount(anchor_component, minlength=ncomp).astype(np.int64)
    anchor_offsets = np.zeros(ncomp + 1, dtype=np.int64)
    np.cumsum(anchor_counts, out=anchor_offsets[1:])

    registration_rows = []
    conflict_components = []
    dependency_components = []
    insufficient_components = []
    tier_components = {1: 0, 2: 0, 3: 0}
    tier_conflicts = {1: 0, 2: 0, 3: 0}
    tier_dependency = {1: 0, 2: 0, 3: 0}
    tier_registered_points = {1: 0, 2: 0, 3: 0}

    # Global-gauge forest first, then detached forests; parent-before-child
    # within each forest. This is deterministic and makes dependency propagation
    # explicit.
    global_first = (component_forest_root != main_component).astype(np.uint8)
    process_order = np.lexsort(
        (
            np.arange(ncomp, dtype=np.int32),
            component_depth,
            component_forest_root,
            global_first,
        )
    )
    for comp0 in process_order:
        comp = int(comp0)
        if comp == main_component:
            continue

        parent = int(component_parent[comp])
        depth = int(component_depth[comp])
        p0 = int(component_offsets[comp])
        p1 = int(component_offsets[comp + 1])
        pids = np.asarray(component_order[p0:p1], dtype=np.int32)
        if pids.size == 0:
            raise RuntimeError(f"empty local component {comp}")
        tier = int(point_tier[pids[0]])
        if tier not in (1, 2, 3):
            raise RuntimeError(f"invalid point tier {tier} in component {comp}")
        tier_components[tier] += 1

        a0 = int(anchor_offsets[comp])
        a1 = int(anchor_offsets[comp + 1])
        aids = np.asarray(anchor_order[a0:a1], dtype=np.int32)

        base_row = {
            "component_id": comp,
            "parent_component_id": parent,
            "forest_root_component_id": int(component_forest_root[comp]),
            "depth": depth,
            "point_count": int(pids.size),
            "tier": tier,
        }

        # A detached forest root deliberately has no global 2pi registration.
        # Preserve its local solution for QA, but do not mark it registered.
        if parent < 0:
            if aids.size != 0:
                raise RuntimeError(f"Detached root {comp} unexpectedly has anchors")
            dependency_components.append(comp)
            tier_dependency[tier] += 1
            registration_rows.append({
                **base_row,
                "status": "detached_forest_root",
                "anchor1_shift": 0,
                "anchor1_class": 0,
                "anchor1_residual_rad": float("nan"),
                "anchor2_shift": 0,
                "anchor2_class": 0,
                "anchor2_residual_rad": float("nan"),
                "anchors_agree": 0,
                "applied_shift": 0,
            })
            continue

        if component_duplicate_anchor[comp]:
            insufficient_components.append(comp)
            tier_dependency[tier] += 1
            registration_rows.append({
                **base_row,
                "status": "insufficient_independent_anchors",
                "anchor1_shift": 0,
                "anchor1_class": int(anchor_class[aids[0]]) if aids.size else 0,
                "anchor1_residual_rad": float("nan"),
                "anchor2_shift": 0,
                "anchor2_class": int(anchor_class[aids[1]]) if aids.size > 1 else 0,
                "anchor2_residual_rad": float("nan"),
                "anchors_agree": 0,
                "applied_shift": 0,
            })
            continue

        if aids.size != 2:
            raise RuntimeError(f"Component {comp} has {aids.size} anchors; expected 2")
        if np.any(anchor_parent_component[aids] != parent):
            raise RuntimeError(f"Component {comp} anchor parent mismatch")

        if not registered_component[parent]:
            dependency_components.append(comp)
            tier_dependency[tier] += 1
            registration_rows.append({
                **base_row,
                "status": "parent_unregistered",
                "anchor1_shift": 0,
                "anchor1_class": int(anchor_class[aids[0]]),
                "anchor1_residual_rad": float("nan"),
                "anchor2_shift": 0,
                "anchor2_class": int(anchor_class[aids[1]]),
                "anchor2_residual_rad": float("nan"),
                "anchors_agree": 0,
                "applied_shift": 0,
            })
            continue

        shifts = []
        details = []
        for aid0 in aids:
            aid = int(aid0)
            a = int(anchor_u[aid])
            b = int(anchor_v[aid])
            ca = int(local_component[a])
            cb = int(local_component[b])
            if ca == comp and cb == parent:
                child_p, parent_p = a, b
            elif cb == comp and ca == parent:
                child_p, parent_p = b, a
            else:
                raise RuntimeError(
                    f"Anchor {aid} does not connect child {comp} to parent {parent}: {ca}/{cb}"
                )

            g_anchor = float(wrap(ifg[parent_p] - ifg[child_p]))
            raw_shift = (U[parent_p] - U[child_p] - g_anchor) / TWOPI
            nshift = int(np.rint(raw_shift))
            residual_rad = U[parent_p] - (U[child_p] + TWOPI * nshift) - g_anchor
            shifts.append(nshift)
            details.append({
                "anchor_id": aid,
                "class": int(anchor_class[aid]),
                "shift": nshift,
                "residual_rad": float(residual_rad),
            })

        agree = shifts[0] == shifts[1]
        applied_shift = 0
        status = "anchor_conflict"
        if agree:
            applied_shift = int(shifts[0])
            U[pids] += TWOPI * applied_shift
            registered[pids] = True
            registered_component[comp] = True
            tier_registered_points[tier] += int(pids.size)
            status = "registered"
        else:
            conflict_components.append(comp)
            tier_conflicts[tier] += 1

        registration_rows.append({
            **base_row,
            "status": status,
            "anchor1_shift": details[0]["shift"],
            "anchor1_class": details[0]["class"],
            "anchor1_residual_rad": details[0]["residual_rad"],
            "anchor2_shift": details[1]["shift"],
            "anchor2_class": details[1]["class"],
            "anchor2_residual_rad": details[1]["residual_rad"],
            "anchors_agree": int(agree),
            "applied_shift": applied_shift,
        })

    registered_points = int(registered.sum())

    # 4. Final modulo parity. Integer component shifts must never change wrapped phase.
    wrap_error = np.abs(wrap(U - ifg))
    max_wrap_error = float(wrap_error.max())

    print("=" * 96)
    print("Final hierarchical single-IFG candidate")
    print("=" * 96)
    print(f"config                    : {config_path}")
    print(f"pair                      : {pair_id}/{len(temporal_edges)}")
    print(f"dates                     : {stack.dates[ti]} -> {stack.dates[tj]}")
    print(f"global component root     : {main_component}")
    print(f"component forests         : {roots.size:,}")
    print(f"global-forest components  : {global_forest_components:,}/{ncomp:,}")
    print(f"forest max depth          : {int(component_depth.max())}")
    print()
    print(f"fragment groups           : {len(groups):,}")
    print(f"accepted groups           : {len(accepted_groups):,}")
    print(f"rejected cycle outliers   : {len(rejected_groups):,}")
    print(f"SAFE bad                  : {safe_bad:,}")
    print(f"UNSAFE within bad         : {unsafe_within_bad:,}")
    print(f"UNSAFE cross bad          : {unsafe_cross_bad:,}")
    print()
    print(f"residual components       : {ncomp-1:,}")
    print(f"direct anchor conflicts   : {len(conflict_components):,}")
    print(f"insufficient 2-anchor     : {len(insufficient_components):,}")
    print(f"dependency skipped        : {len(dependency_components):,}")
    print(f"registered points         : {registered_points:,}/{npoint:,} ({100*registered_points/npoint:.5f}%)")
    for tier, name in ((1, "normal"), (2, "extended"), (3, "long")):
        print(
            f"{name:8s}: components={tier_components[tier]:,}, "
            f"conflicts={tier_conflicts[tier]:,}, "
            f"dependency={tier_dependency[tier]:,}, "
            f"registered={tier_registered_points[tier]:,}"
        )
    print(f"wrap-back max error       : {max_wrap_error:.3e} rad")

    np.save(outdir / f"{tag}_unwrapped_phase_rad.npy", U.astype(np.float32))
    np.save(outdir / f"{tag}_registered_mask.npy", registered)

    group_csv = outdir / f"{tag}_fragment_constraint_status.csv"
    all_groups = accepted_groups + rejected_groups
    all_groups.sort(key=lambda r: r["group_id"])
    if all_groups:
        with group_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_groups[0].keys()))
            w.writeheader(); w.writerows(all_groups)
    else:
        group_csv.write_text("")

    reg_csv = outdir / f"{tag}_residual_registration.csv"
    if registration_rows:
        with reg_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(registration_rows[0].keys()))
            w.writeheader(); w.writerows(registration_rows)
    else:
        reg_csv.write_text("")

    manifest = {
        "format": "pyPSDS-GAMMA-robust-single-ifg-candidate-v2.0",
        "status": "CANDIDATE_HIERARCHICAL_COMPONENT_FOREST",
        "pair": {
            "pair_id": pair_id,
            "date1": str(stack.dates[ti]),
            "date2": str(stack.dates[tj]),
        },
        "fragment_constraints": {
            "total": len(groups),
            "accepted": len(accepted_groups),
            "rejected_cycle_outliers": len(rejected_groups),
            "rejected_group_ids": [r["group_id"] for r in rejected_groups],
        },
        "local_edge_qa": {
            "safe_bad": safe_bad,
            "unsafe_within_bad": unsafe_within_bad,
            "unsafe_cross_bad": unsafe_cross_bad,
        },
        "component_registration": {
            "strategy": "global_forest_root_to_leaf_parent_child_two_anchor",
            "root_component": main_component,
            "forest_count": int(roots.size),
            "global_forest_components": global_forest_components,
            "maximum_depth": int(component_depth.max()),
            "components": ncomp - 1,
            "direct_anchor_conflicts": len(conflict_components),
            "insufficient_independent_anchors": len(insufficient_components),
            "dependency_skipped": len(dependency_components),
            "registered_components": int(registered_component.sum()),
            "registered_points": registered_points,
            "registered_fraction": float(registered_points / npoint),
        },
        # Preserve the legacy key for downstream readers that only inspect counts.
        "residual_registration": {
            "components": ncomp - 1,
            "agree": int(
                (ncomp - 1)
                - len(conflict_components)
                - len(insufficient_components)
                - len(dependency_components)
            ),
            "conflict": len(conflict_components),
            "insufficient_independent_anchors": len(insufficient_components),
            "dependency_skipped": len(dependency_components),
            "registered_points": registered_points,
            "registered_fraction": float(registered_points / npoint),
        },
        "performance_contract": {
            "component_point_lookup": "precomputed CSR; no per-component full-array np.where",
        },
        "wrap_back_max_error_rad": max_wrap_error,
    }
    manifest_path = outdir / f"{tag}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print()
    print(f"fragment status           : {group_csv}")
    print(f"component registration    : {reg_csv}")
    print(f"manifest                  : {manifest_path}")
    print()
    print("STEP single_ifg_solution STATUS: PASS / HIERARCHICAL-FOREST SINGLE-IFG CANDIDATE")


if __name__ == "__main__":
    main()
