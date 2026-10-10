"""Read-only candidate bridge assessment for disconnected occupied InSAR grids.

Geometric proximity and temporal smoothness identify *candidates*, never a
unique relative 2*pi gauge. This module does not modify unwrapped phases or
write production arrays. It is a preflight for cross-component constraints.

Run: python -m pypsds.unwrap.spatial_bridges --help
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

TWOPI = 2.0 * np.pi


def wrap_phase(x):
    return np.arctan2(np.sin(x), np.cos(x))


def occupied_components(node_grid):
    """Return occupied-node component IDs and sizes, using 4-connected cells."""
    from pypsds.unwrap.spatial_topology import _validated_node_grid

    g = _validated_node_grid(node_grid)
    mask = g >= 0
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    cc, ncomp = ndimage.label(mask, structure=structure)
    node_comp = np.empty(int(mask.sum()), dtype=np.int32)
    node_comp[g[mask]] = cc[mask].astype(np.int32) - 1
    sizes = np.bincount(node_comp, minlength=ncomp)
    xy = np.empty((node_comp.size, 2), dtype=np.float64)
    xy[g[mask], :] = np.argwhere(mask)
    return node_comp, sizes, xy


def candidate_pairs(node_grid, *, grid_size_m: float, max_distance_m: float):
    """Return all different-component node pairs within nominal metric radius.

    The range and azimuth grid axes are treated as nominal metric axes, exactly
    as the coarse-grid constructor does. These are NOT verified ground distances
    or terrain-crossability tests.
    """
    if not (np.isfinite(grid_size_m) and grid_size_m > 0):
        raise ValueError('grid_size_m must be positive')
    if not (np.isfinite(max_distance_m) and max_distance_m > grid_size_m):
        raise ValueError('max_distance_m must exceed grid_size_m')
    comp, sizes, xy = occupied_components(node_grid)
    if len(sizes) < 2:
        return comp, sizes, np.empty((0, 2), np.int32), np.empty(0, np.float32)
    node_pairs = cKDTree(xy).query_pairs(
        r=float(max_distance_m / grid_size_m) + 1e-9,
        output_type='ndarray',
    )
    if node_pairs.size == 0:
        return comp, sizes, np.empty((0, 2), np.int32), np.empty(0, np.float32)
    node_pairs = np.asarray(node_pairs, dtype=np.int32)
    cross = comp[node_pairs[:, 0]] != comp[node_pairs[:, 1]]
    node_pairs = node_pairs[cross]
    if not node_pairs.size:
        return comp, sizes, np.empty((0, 2), np.int32), np.empty(0, np.float32)
    length = np.linalg.norm(xy[node_pairs[:, 0]] - xy[node_pairs[:, 1]], axis=1)
    length = (length * grid_size_m).astype(np.float32)
    return comp, sizes, node_pairs, length


def _top_pairs(comp, node_pairs, distance, limit_per_component_pair=8):
    """Keep short candidate ties with explicit component-pair identities."""
    if limit_per_component_pair < 1:
        raise ValueError('limit_per_component_pair must be >= 1')
    if len(node_pairs) == 0:
        return (np.empty((0, 2), np.int32), np.empty(0, np.float32),
                np.empty((0, 2), np.int32))
    cc = np.column_stack((
        np.minimum(comp[node_pairs[:, 0]], comp[node_pairs[:, 1]]),
        np.maximum(comp[node_pairs[:, 0]], comp[node_pairs[:, 1]]),
    )).astype(np.int32)
    # Sort primarily by component pair, secondarily by separation.
    order = np.lexsort((distance, cc[:, 1], cc[:, 0]))
    ordered = cc[order]
    starts = np.r_[0, np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(order)]
    selected = np.concatenate([order[s:min(e, s + limit_per_component_pair)]
                               for s, e in zip(starts, ends)])
    return node_pairs[selected], distance[selected], cc[selected]


def _read_itab(path: Path, ndate: int):
    ifgs = []
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0].startswith('#'):
            continue
        i, j = int(parts[0]) - 1, int(parts[1]) - 1
        if not (0 <= i < ndate and 0 <= j < ndate and i != j):
            raise ValueError('Invalid network.itab acquisition index')
        ifgs.append((i, j))
    if not ifgs:
        raise ValueError('Empty network.itab')
    return ifgs


def _edge_time_residuals(phase, node_pairs, *, edges, dates, batch_size=1024):
    """StaMPS-style temporal innovation std per candidate tie (diagnostic)."""
    from pypsds.unwrap.statistical_cost import build_temporal_operator

    _, T, _ = build_temporal_operator(edges, dates)
    ii = np.asarray([i for i, _ in edges], dtype=np.intp)
    jj = np.asarray([j for _, j in edges], dtype=np.intp)
    out = np.empty(len(node_pairs), dtype=np.float32)
    for start in range(0, len(node_pairs), batch_size):
        end = min(len(node_pairs), start + batch_size)
        pair = node_pairs[start:end]
        d = wrap_phase(np.asarray(phase[pair[:, 1], :], dtype=np.float64) -
                       np.asarray(phase[pair[:, 0], :], dtype=np.float64))
        y = wrap_phase(d[:, jj] - d[:, ii])
        smooth = y @ T.T
        innovation = wrap_phase(y - smooth)
        out[start:end] = np.std(innovation, axis=1, ddof=1).astype(np.float32)
    return out


def _component_union(ncomp, sizes, component_pair_links):
    parent = np.arange(ncomp, dtype=np.int32)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for aa, bb in component_pair_links:
        a, b = find(int(aa)), find(int(bb))
        if a != b:
            parent[b] = a
    roots = np.fromiter((find(i) for i in range(ncomp)), dtype=np.int32, count=ncomp)
    _, inverse = np.unique(roots, return_inverse=True)
    w = np.bincount(inverse, weights=sizes, minlength=int(inverse.max()) + 1)
    return int(w.size), int(np.max(w))


def summarize_candidate_network(component_id, component_sizes, pairs, distances,
                                *, radii_m, node_support=None, noise_std=None,
                                min_support=8, max_noise_rad=1.3,
                                min_independent_pairs=3):
    """Report geometrically possible and quality-screened connection scenarios.

    Geometric/temporal gates are diagnostic and NEVER imply solved 2*pi gauges.
    """
    ncomp = len(component_sizes)
    nnode = int(np.sum(component_sizes))
    out = []
    for radius in radii_m:
        inside = distances <= radius + 1e-4
        relevant = pairs[inside]
        if relevant.size:
            cc = np.column_stack((
                np.minimum(component_id[relevant[:, 0]], component_id[relevant[:, 1]]),
                np.maximum(component_id[relevant[:, 0]], component_id[relevant[:, 1]]),
            ))
            raw_links = np.unique(cc, axis=0)
        else:
            raw_links = np.empty((0, 2), dtype=np.int32)
        geom_n, geom_dom = _component_union(ncomp, component_sizes, raw_links)

        reliable = inside.copy()
        if node_support is not None:
            reliable &= np.minimum(node_support[pairs[:, 0]], node_support[pairs[:, 1]]) >= min_support
        if noise_std is not None:
            reliable &= np.isfinite(noise_std) & (noise_std <= max_noise_rad)
        selected = pairs[reliable]
        if selected.size:
            cc = np.column_stack((
                np.minimum(component_id[selected[:, 0]], component_id[selected[:, 1]]),
                np.maximum(component_id[selected[:, 0]], component_id[selected[:, 1]]),
            ))
            labels, inv, cnt = np.unique(cc, axis=0, return_inverse=True, return_counts=True)
            # Redundancy must include different actual nodes on both sides.
            kept = np.zeros(len(labels), dtype=bool)
            for i in np.flatnonzero(cnt >= min_independent_pairs):
                group = selected[inv == i]
                left = np.where(component_id[group[:, 0]] == labels[i, 0], group[:, 0], group[:, 1])
                right = np.where(component_id[group[:, 0]] == labels[i, 1], group[:, 0], group[:, 1])
                kept[i] = len(np.unique(left)) >= 2 and len(np.unique(right)) >= 2
            robust_links = labels[kept]
        else:
            robust_links = np.empty((0, 2), dtype=np.int32)
        screened_n, screened_dom = _component_union(ncomp, component_sizes, robust_links)
        out.append({
            'nominal_distance_limit_m': float(radius),
            'candidate_node_pairs': int(np.count_nonzero(inside)),
            'candidate_component_pairs': int(len(raw_links)),
            'geometric_component_count': geom_n,
            'geometric_largest_node_fraction': geom_dom / max(nnode, 1),
            'screened_component_pairs': int(len(robust_links)),
            'screened_component_count': screened_n,
            'screened_largest_node_fraction': screened_dom / max(nnode, 1),
            'screening_note': 'Diagnostic only; screened component links do not fix integer gauges',
        })
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description='Read-only inter-component spatial bridge candidates')
    p.add_argument('--node-grid', type=Path, required=True)
    p.add_argument('--node-phase', type=Path, required=True)
    p.add_argument('--node-support', type=Path, default=None)
    p.add_argument('--dates', type=Path, required=True)
    p.add_argument('--itab', type=Path, required=True)
    p.add_argument('--grid-size-m', type=float, default=200.)
    p.add_argument('--radii-m', type=float, nargs='+', default=[285., 450., 600., 800.])
    p.add_argument('--per-component-pair', type=int, default=8)
    p.add_argument('--min-support', type=int, default=8)
    p.add_argument('--max-temporal-innovation-rad', type=float, default=1.3)
    p.add_argument('--min-independent-pairs', type=int, default=3)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)

    if any(x <= args.grid_size_m for x in args.radii_m):
        p.error('every radius must be greater than grid size')
    if args.min_support < 1 or args.min_independent_pairs < 1:
        p.error('support and independent-pairs minima must be positive')
    g = np.load(args.node_grid, mmap_mode='r')
    phase = np.load(args.node_phase, mmap_mode='r')
    c, sz, candidate, dist = candidate_pairs(
        g, grid_size_m=args.grid_size_m, max_distance_m=max(args.radii_m))
    if phase.ndim != 2 or phase.shape[0] != len(c):
        raise ValueError('node phase does not match node grid node count')
    dates = args.dates.read_text(encoding='utf-8').split()
    if len(dates) != phase.shape[1]:
        raise ValueError('dates do not match node phase acquisitions')
    itab = _read_itab(args.itab, len(dates))
    supp = None
    if args.node_support is not None:
        supp = np.asarray(np.load(args.node_support, mmap_mode='r'))
        if supp.shape != (len(c),):
            raise ValueError('node support does not match node count')
    pairs, distance, cc = _top_pairs(
        c, candidate, dist, limit_per_component_pair=args.per_component_pair)
    innovation = _edge_time_residuals(
        phase, pairs, edges=itab, dates=dates) if pairs.size else np.empty(0, np.float32)
    results = summarize_candidate_network(
        c, sz, pairs, distance,
        radii_m=args.radii_m, node_support=supp, noise_std=innovation,
        min_support=args.min_support,
        max_noise_rad=args.max_temporal_innovation_rad,
        min_independent_pairs=args.min_independent_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'spatial_bridge_candidates.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        w = csv.writer(handle)
        w.writerow(['component_a_0based','component_b_0based','node_a_0based',
                    'node_b_0based','nominal_separation_m','temporal_innovation_std_rad',
                    'minimum_node_support'])
        for i, pair in enumerate(pairs):
            s = min(supp[pair[0]], supp[pair[1]]) if supp is not None else ''
            w.writerow([int(cc[i,0]),int(cc[i,1]),int(pair[0]),int(pair[1]),
                        float(distance[i]),float(innovation[i]),s])
    summary = {
        'status':'DIAGNOSTIC_ONLY',
        'occupied_nodes':int(len(c)), 'initial_components':int(len(sz)),
        'largest_initial_component_fraction':float(sz.max()/len(c)),
        'max_radius_m':float(max(args.radii_m)),
        'retained_nearest_candidate_pairs':int(len(pairs)),
        'geometric_note':'Nominal radar-grid distance, not independently geocoded ground distance',
        'scientific_note':'Geometry and phase-noise screening cannot independently determine any relative 2*pi integer gauge; no phase is changed',
        'scenario_screening':{
            'minimum_node_support':args.min_support if supp is not None else None,
            'maximum_temporal_innovation_std_rad':args.max_temporal_innovation_rad,
            'minimum_distinct_node_pairs':args.min_independent_pairs,
            'minimum_nodes_each_component_side':2,
        },
        'scenarios': results,
        'csv':str(csv_path),
    }
    out = args.output_dir / 'spatial_bridge_summary.json'
    out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
