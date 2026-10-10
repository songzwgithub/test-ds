"""Build read-only, native-PS/DS-supported spatial constraints for InSAR unwrap.

This is an evidence-building stage, *not* an integer unwrapping algorithm.
It deliberately refuses to infer a relative 2*pi gauge from wrapped point
observations. The selected native point pairs form a reproducible sparse
spatial network usable by a subsequent integer-constraint solver.

GAMMA radar-coordinate distances are nominal. They must not be presented as
independently validated geodetic ground distances.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def _select_exposed_points(rows, cols, typ, quality, grid, row_spacing,
                           col_spacing, grid_size, anchors_per_side, side_window):
    """Select near-boundary points for each occupied node and cardinal face.

    Sides in order: north=0, south=1, west=2, east=3. The selection gives
    priority to proximity, with a bounded reliability bonus. No phase value
    or existing unwrapped result is used for selecting a candidate.
    """
    nnode = int(np.max(grid)) + 1
    selected = np.full((nnode, 4, anchors_per_side), -1, np.int32)
    best = np.full((nnode, 4, anchors_per_side), -1e30, np.float64)
    r0 = int(np.min(rows))
    c0 = int(np.min(cols))
    for p in range(len(rows)):
        y = (float(rows[p]) - r0) * row_spacing
        x = (float(cols[p]) - c0) * col_spacing
        gr = int(math.floor(y / grid_size))
        gc = int(math.floor(x / grid_size))
        if gr < 0 or gr >= grid.shape[0] or gc < 0 or gc >= grid.shape[1]:
            raise ValueError('Point coordinates are outside node grid')
        nid = int(grid[gr, gc])
        if nid < 0:
            raise ValueError('Observed point mapped to empty cell')
        yf = y - gr * grid_size
        xf = x - gc * grid_size
        distance = (yf, grid_size - yf, xf, grid_size - xf)
        q = float(quality[p])
        if typ[p] == 1:
            q = 0.95
        elif not np.isfinite(q):
            q = 0.0
        if q < 0.0:
            q = 0.0
        if q > 1.0:
            q = 1.0
        for side in range(4):
            if distance[side] > side_window:
                continue
            # Small, bounded quality term cannot dominate a large separation.
            score = -distance[side] + 15.0 * q + (2.0 if typ[p] == 1 else 0.0)
            for j in range(anchors_per_side):
                if score > best[nid, side, j]:
                    for k in range(anchors_per_side-1, j, -1):
                        best[nid, side, k] = best[nid, side, k-1]
                        selected[nid, side, k] = selected[nid, side, k-1]
                    best[nid, side, j] = score
                    selected[nid, side, j] = p
                    break
    return selected


def build_candidate_node_edges(node_grid, bridge_csv=None):
    """Occupied four-neighbor edges, optionally plus pre-existing bridge candidates."""
    g = np.asarray(node_grid)
    if g.ndim != 2 or not np.issubdtype(g.dtype, np.integer) or np.any(g < -1):
        raise ValueError('Invalid node grid')
    ids = np.sort(g[g >= 0]); nnode = int(ids.size)
    if nnode == 0 or not np.array_equal(ids, np.arange(nnode)):
        raise ValueError('Observed node IDs must be contiguous and unique')
    top, bot = g[:-1], g[1:]
    left, right = g[:, :-1], g[:, 1:]
    r = (top >= 0) & (bot >= 0)
    c = (left >= 0) & (right >= 0)
    a = np.concatenate((top[r], left[c])).astype(np.int32)
    b = np.concatenate((bot[r], right[c])).astype(np.int32)
    observed = np.stack((np.minimum(a,b), np.maximum(a,b)), axis=1)
    observed = np.unique(observed, axis=0)
    if bridge_csv is None:
        pairs = observed
    else:
        found = []
        with Path(bridge_csv).open(newline='', encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                u = int(rec['node_a_0based']); v = int(rec['node_b_0based'])
                if u == v or min(u,v) < 0 or max(u,v) >= nnode:
                    raise ValueError(f'Out-of-range bridge candidate: {u}, {v}')
                found.append((min(u,v),max(u,v)))
        allpairs = (np.vstack((observed, np.array(found, dtype=np.int32).reshape(-1, 2)))
                    if found else observed)
        pairs = np.unique(allpairs, axis=0)
    # Direct adjacency is a structural category, not evidence of an integer gauge.
    is_observed = np.zeros(len(pairs), dtype=np.bool_)
    if len(observed):
        keys0 = observed[:, 0].astype(np.int64) * nnode + observed[:, 1]
        keys = pairs[:, 0].astype(np.int64) * nnode + pairs[:, 1]
        is_observed = np.isin(keys, keys0, assume_unique=True)
    return pairs, is_observed


@njit(cache=True)
def _match_native_pairs(node_pairs, positions, anchors, rows, cols,
                        row_spacing, col_spacing, max_point_distance, n_pair):
    matched = np.full((len(node_pairs), n_pair, 2), -1, np.int32)
    lengths = np.full((len(node_pairs), n_pair), np.nan, np.float32)
    count = np.zeros(len(node_pairs), np.int8)
    for e in range(len(node_pairs)):
        a, b = int(node_pairs[e, 0]), int(node_pairs[e, 1])
        dy = int(positions[b, 0]) - int(positions[a, 0])
        dx = int(positions[b, 1]) - int(positions[a, 1])
        if abs(dy) > abs(dx):
            side_a = 1 if dy > 0 else 0
            side_b = 0 if dy > 0 else 1
        else:
            side_a = 3 if dx > 0 else 2
            side_b = 2 if dx > 0 else 3
        na = anchors.shape[2]
        for k in range(n_pair):
            bestd2 = max_point_distance ** 2
            best_a = -1
            best_b = -1
            for ia in range(na):
                pa = int(anchors[a, side_a, ia])
                if pa < 0:
                    continue
                used_a = False
                for old in range(k):
                    if matched[e, old, 0] == pa:
                        used_a = True
                if used_a:
                    continue
                for ib in range(na):
                    pb = int(anchors[b, side_b, ib])
                    if pb < 0:
                        continue
                    used_b = False
                    for old in range(k):
                        if matched[e, old, 1] == pb:
                            used_b = True
                    if used_b:
                        continue
                    ddy = (int(rows[pa]) - int(rows[pb])) * row_spacing
                    ddx = (int(cols[pa]) - int(cols[pb])) * col_spacing
                    d2 = ddy * ddy + ddx * ddx
                    if d2 <= bestd2:
                        bestd2 = d2
                        best_a = pa
                        best_b = pb
            if best_a < 0:
                break
            matched[e, k, 0] = best_a
            matched[e, k, 1] = best_b
            lengths[e, k] = math.sqrt(bestd2)
            count[e] += 1
    return matched, lengths, count


def build_native_edge_support(*, node_grid, point_rows, point_cols, point_type,
                              point_quality, row_spacing_m, col_spacing_m,
                              grid_size_m=200.0, anchors_per_side=4,
                              side_window_m=125.0, max_point_distance_m=250.0,
                              pairs_per_edge=3, bridge_csv=None):
    """Construct a sparse native-point observation graph without unwrapping."""
    r = np.asarray(point_rows); c = np.asarray(point_cols)
    typ = np.asarray(point_type); q = np.asarray(point_quality)
    if r.ndim != 1 or r.shape != c.shape or r.shape != typ.shape or r.shape != q.shape:
        raise ValueError('All point metadata must be one-dimensional and aligned')
    if not np.all(np.isfinite([row_spacing_m,col_spacing_m,grid_size_m,
                              side_window_m,max_point_distance_m])):
        raise ValueError('Geometry parameters must be finite')
    if min(row_spacing_m,col_spacing_m,grid_size_m,side_window_m,max_point_distance_m) <= 0:
        raise ValueError('Geometry parameters must be positive')
    if anchors_per_side < pairs_per_edge or pairs_per_edge < 2:
        raise ValueError('Require at least two independent pairs and sufficient anchors')
    g = np.asarray(node_grid, dtype=np.int32)
    pairs, observed = build_candidate_node_edges(g, bridge_csv)
    positions = np.empty((int(np.count_nonzero(g >= 0)), 2), dtype=np.int32)
    positions[g[g >= 0]] = np.argwhere(g >= 0)
    anchors = _select_exposed_points(np.asarray(r, dtype=np.int32),
                                     np.asarray(c, dtype=np.int32),
                                     np.asarray(typ, dtype=np.uint8),
                                     np.asarray(q, dtype=np.float32), g,
                                     float(row_spacing_m), float(col_spacing_m),
                                     float(grid_size_m), int(anchors_per_side),
                                     float(side_window_m))
    matched, distance, counts = _match_native_pairs(
        pairs, positions, anchors, np.asarray(r, dtype=np.int32),
        np.asarray(c, dtype=np.int32), float(row_spacing_m),
        float(col_spacing_m), float(max_point_distance_m), int(pairs_per_edge))
    return {'node_pairs':pairs, 'directly_adjacent':observed,
            'native_point_pairs':matched, 'native_pair_distances_m':distance,
            'independent_pair_count':counts,
            'node_anchor_point_ids':anchors}


def evaluate_native_pair_phase(*, point_phase, native_point_pairs,
                               batch_size=512):
    """Evaluate replicated native cross-edge wrapped phase observations.

    Low circular coherence indicates disagreement among pair samples; it is
    not a wrapped phase proof and must not directly change integer cycles.
    """
    phase = point_phase
    if phase.ndim != 2:
        raise ValueError('Expected [point,date] phase stack')
    pairs = np.asarray(native_point_pairs)
    if pairs.ndim != 3 or pairs.shape[2] != 2:
        raise ValueError('Expected [edge,independent pair,2] IDs')
    nedge, npair, _ = pairs.shape
    concentration = np.full(nedge, np.nan, np.float32)
    valid_count = np.zeros(nedge, np.int8)
    # The circular-mean result is a wrapped observation, not an absolute phase.
    for s in range(0,nedge,int(batch_size)):
        e = min(nedge,s+int(batch_size))
        sub = pairs[s:e]
        ok = (sub[:,:,0]>=0)&(sub[:,:,1]>=0)
        valid_count[s:e] = ok.sum(axis=1).astype(np.int8)
        if not ok.any():
            continue
        a = sub[:,:,0].clip(min=0)
        b = sub[:,:,1].clip(min=0)
        aa = np.asarray(phase[a.reshape(-1),:],dtype=np.float32).reshape(e-s,npair,-1)
        bb = np.asarray(phase[b.reshape(-1),:],dtype=np.float32).reshape(e-s,npair,-1)
        z = np.exp(1j*(bb-aa))
        z[~ok] = 0.0
        denom = np.maximum(1,ok.sum(axis=1))[:,None]
        amp = np.abs(z.sum(axis=1)/denom)
        med = np.median(amp,axis=1)
        med[~ok.any(axis=1)] = np.nan
        concentration[s:e] = med.astype(np.float32)
    return concentration, valid_count


def main(argv=None):
    p = argparse.ArgumentParser(description='Native PS/DS edge evidence; NO integer adjustment')
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--bridge-csv',type=Path,default=None)
    p.add_argument('--grid-size-m',type=float,default=200.0)
    p.add_argument('--max-point-distance-m',type=float,default=250.0)
    p.add_argument('--side-window-m',type=float,default=125.0)
    p.add_argument('--anchors-per-side',type=int,default=4)
    p.add_argument('--pairs-per-edge',type=int,default=3)
    p.add_argument('--minimum-circular-consistency',type=float,default=0.75)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    if not 0.0 <= args.minimum_circular_consistency <= 1.0:
        p.error('minimum-circular-consistency must lie in [0,1]')

    from pypsds.context import open_from_config
    from pypsds.geometry.inputs import resolve_geometry_inputs
    from pypsds.gamma.geometry import geometry_from_par
    cfg, config_path, paths, stack, _ = open_from_config(args.config)
    geom = geometry_from_par(resolve_geometry_inputs(cfg,paths).reference_rslc_par)
    pp = Path(paths.output_dir)/'processing'/'point_phase_stack'
    co = Path(paths.output_dir)/'processing'/'stamps3d_unwrap'
    r=np.load(pp/'rows.npy',mmap_mode='r')
    c=np.load(pp/'cols.npy',mmap_mode='r')
    typ=np.load(pp/'point_type.npy',mmap_mode='r')
    quality=np.load(pp/'temporal_coherence.npy',mmap_mode='r')
    phase=np.load(pp/'phase_rad.npy',mmap_mode='r')
    grid=np.load(co/'coarse_node_grid.npy',mmap_mode='r')
    if phase.shape[0] != len(r) or phase.shape[1] != len(stack.dates):
        raise ValueError('PointPhaseStack dates or point identity mismatch')
    print('Building native PS/DS edge support...',flush=True)
    model=build_native_edge_support(
        node_grid=grid,point_rows=r,point_cols=c,point_type=typ,
        point_quality=quality,row_spacing_m=geom.azimuth_spacing_m,
        col_spacing_m=geom.ground_range_spacing_m,
        grid_size_m=args.grid_size_m,side_window_m=args.side_window_m,
        max_point_distance_m=args.max_point_distance_m,
        anchors_per_side=args.anchors_per_side,
        pairs_per_edge=args.pairs_per_edge,bridge_csv=args.bridge_csv)
    print('Evaluating wrapped native pair coherence...',flush=True)
    coh,count=evaluate_native_pair_phase(point_phase=phase,
                                     native_point_pairs=model['native_point_pairs'])
    good=(count>=args.pairs_per_edge)&(coh>=args.minimum_circular_consistency)
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    for key,val in model.items():
        np.save(output/(key+'.npy'),val)
    np.save(output/'native_pair_circular_consistency.npy',coh)
    np.save(output/'supported_edge_mask.npy',good)
    observed=model['directly_adjacent']
    summary={
      'status':'DIAGNOSTIC_ONLY',
      'points':int(len(r)), 'acquisitions':int(phase.shape[1]),
      'grid_nodes':int(np.count_nonzero(grid>=0)),
      'candidate_node_edges':int(len(observed)),
      'occupied_four_neighbor_edges':int(np.count_nonzero(observed)),
      'bridge_candidate_edges':int(np.count_nonzero(~observed)),
      'edges_with_requested_native_pairs':int(np.count_nonzero(count>=args.pairs_per_edge)),
      'edges_with_phase_consistent_native_pairs':int(np.count_nonzero(good)),
      'occupied_edges_with_consistent_pairs':int(np.count_nonzero(good & observed)),
      'bridge_edges_with_consistent_pairs':int(np.count_nonzero(good & ~observed)),
      'minimum_circular_consistency':float(args.minimum_circular_consistency),
      'max_point_distance_m_nominal':float(args.max_point_distance_m),
      'concentration_p50_p90_p95':(
        [float(x) for x in np.percentile(coh[np.isfinite(coh)],[50,90,95])]
        if np.isfinite(coh).any() else [None]*3),
      'relative_integer_gauge_solved':False,
      'phase_modified':False,
      'caveat':'Replicated native wrapped phase agreement does not identify the correct 2*pi branch; neither these nominal radar-coordinate lengths nor this graph alone establish physical continuity.'
    }
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
