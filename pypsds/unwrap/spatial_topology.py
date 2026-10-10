"""Observed-grid spatial topology for scientific StaMPS-style unwrapping.

Nearest-neighbor filling may supply a phase to SNAPHU's rectangular raster,
but it is not an independent observation of phase across an empty area. This
module reports the actual four-connected support and builds statistical edges
only between genuinely occupied adjacent cells. Disconnected components carry
independent, unobservable spatial integer gauges until externally registered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def _validated_node_grid(node_grid):
    g = np.asarray(node_grid)
    if g.ndim != 2 or min(g.shape) < 1:
        raise ValueError("node_grid must be a nonempty 2D grid")
    if not np.issubdtype(g.dtype, np.integer):
        raise ValueError("node_grid requires integer node identifiers")
    if np.any(g < -1):
        raise ValueError("node_grid includes negative identifiers other than -1")
    node_ids = np.sort(g[g >= 0])
    if node_ids.size == 0:
        raise ValueError("node_grid contains no observed nodes")
    if not np.array_equal(node_ids, np.arange(node_ids.size)):
        raise ValueError("node_grid requires one unique, contiguous identifier per cell")
    return g


def build_occupied_spatial_edges(node_grid):
    """Build SNAPHU statistical edges only on observed four-neighbor pairs.

    Maps use -2 for unsupported dense-grid boundaries, -1 for a same-node
    equality constraint (not produced here), and >=0 for measured edges.
    Unlike the nearest-filled grid, this graph never links across blank cells.
    """
    g = _validated_node_grid(node_grid)
    a, b = g[:-1, :], g[1:, :]
    rvalid = (a >= 0) & (b >= 0) & (a != b)
    c, d = g[:, :-1], g[:, 1:]
    cvalid = (c >= 0) & (d >= 0) & (c != d)

    ra, rb = a[rvalid], b[rvalid]
    ca, cb = c[cvalid], d[cvalid]
    candidates_a = np.concatenate((ra, ca)).astype(np.int32)
    candidates_b = np.concatenate((rb, cb)).astype(np.int32)
    pairs = np.column_stack((np.minimum(candidates_a, candidates_b),
                             np.maximum(candidates_a, candidates_b)))
    if pairs.shape[0] == 0:
        raise RuntimeError("No physically occupied four-neighbor spatial edges")

    edge_nodes, inverse, occurrences = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True
    )
    nr = int(np.count_nonzero(rvalid))
    row_eid = np.full(rvalid.shape, -2, dtype=np.int32)
    col_eid = np.full(cvalid.shape, -2, dtype=np.int32)
    row_sign = np.zeros(rvalid.shape, dtype=np.int8)
    col_sign = np.zeros(cvalid.shape, dtype=np.int8)
    row_eid[rvalid] = inverse[:nr].astype(np.int32)
    col_eid[cvalid] = inverse[nr:].astype(np.int32)
    row_sign[rvalid] = np.where(ra < rb, 1, -1).astype(np.int8)
    col_sign[cvalid] = np.where(ca < cb, 1, -1).astype(np.int8)
    return {
        'edge_nodes': edge_nodes.astype(np.int32, copy=False),
        'edge_occurrences': occurrences.astype(np.int32, copy=False),
        'row_eid': row_eid,
        'row_sign': row_sign,
        'col_eid': col_eid,
        'col_sign': col_sign,
    }


def summarize_spatial_topology(node_grid, nearest_node=None):
    """Report independently observable spatial supports without guessing ties."""
    g = _validated_node_grid(node_grid)
    occupied = g >= 0
    # 4-connected; diagonal point contact alone cannot establish a spatial edge.
    cross = np.asarray([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    labeled, ncomp = ndimage.label(occupied, structure=cross)
    sizes = np.bincount(labeled[occupied], minlength=ncomp+1)[1:]
    largest = int(sizes.max())
    nr = int(np.count_nonzero(occupied[:-1,:] & occupied[1:,:]))
    nc = int(np.count_nonzero(occupied[:,:-1] & occupied[:,1:]))
    result = {
        'occupied_nodes': int(np.count_nonzero(occupied)),
        'grid_cells': int(g.size),
        'observed_fraction': float(np.mean(occupied)),
        'component_count': int(ncomp),
        'largest_component_nodes': largest,
        'dominant_node_fraction': largest / int(np.count_nonzero(occupied)),
        'nodes_outside_dominant_component': int(np.count_nonzero(occupied)) - largest,
        'component_sizes_top10': sorted(map(int, sizes), reverse=True)[:10],
        'component_count_at_least_10_nodes': int(np.count_nonzero(sizes >= 10)),
        'component_count_at_least_100_nodes': int(np.count_nonzero(sizes >= 100)),
        'occupied_grid_edges': nr + nc,
        'global_integer_gauge_identifiable_from_occupied_edges': bool(ncomp == 1),
    }
    if nearest_node is not None:
        nearest = np.asarray(nearest_node)
        if nearest.shape != g.shape:
            raise ValueError("nearest_node and node_grid shape differ")
        if np.any(nearest < 0) or np.any(nearest >= sizes.sum()):
            raise ValueError("nearest_node contains unknown node ids")
        r = nearest[:-1,:] != nearest[1:,:]
        c = nearest[:,:-1] != nearest[:,1:]
        real_r = occupied[:-1,:] & occupied[1:,:]
        real_c = occupied[:,:-1] & occupied[:,1:]
        n_synthetic = int(np.count_nonzero(r & ~real_r) +
                          np.count_nonzero(c & ~real_c))
        result.update({
            'nearest_fill_interfaces': int(np.count_nonzero(r) + np.count_nonzero(c)),
            'unsupported_nearest_fill_interfaces': n_synthetic,
        })
    return result


def main():
    p = argparse.ArgumentParser(description='Read-only spatial grid topology preflight')
    p.add_argument('--node-grid', type=Path, required=True)
    p.add_argument('--json', type=Path, default=None)
    args = p.parse_args()
    g = np.load(args.node_grid, mmap_mode='r')
    occ = g >= 0
    _, idx = ndimage.distance_transform_edt(~occ, return_indices=True)
    nearest = g[idx[0], idx[1]]
    profile = summarize_spatial_topology(g, nearest)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(profile, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
