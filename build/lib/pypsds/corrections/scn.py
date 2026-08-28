"""
Portable validated correction core.

Generated mechanically from the frozen authoritative production source.

Runtime project geometry/state previously stored in module globals is
represented as explicit function arguments. Numerical function bodies
are otherwise retained.
"""

import math
from numba import njit, prange, get_num_threads, set_num_threads
import numpy as np



def temporal_weight_matrix(day, master0, time_win):
    dt = day[:, None] - day[None, :]
    W = np.exp(-(dt * dt) / (2.0 * time_win * time_win))
    W[:, master0] = 0.0
    den = np.sum(W, axis=1)
    if np.any(den <= 0.0):
        raise RuntimeError('temporal Gaussian denominator <= 0')
    W /= den[:, None]
    return W

def build_cell_index(coords, cell_size):
    xmin = float(coords[:, 0].min())
    ymin = float(coords[:, 1].min())
    cx = np.floor((coords[:, 0] - xmin) / cell_size).astype(np.int32)
    cy = np.floor((coords[:, 1] - ymin) / cell_size).astype(np.int32)
    nx = int(cx.max()) + 1
    ny = int(cy.max()) + 1
    cell_id = cy.astype(np.int64) * nx + cx
    order = np.argsort(cell_id, kind='stable').astype(np.int32)
    ncell = nx * ny
    occupancy = np.bincount(cell_id, minlength=ncell).astype(np.int64)
    starts = np.empty(ncell + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(occupancy, out=starts[1:])
    return (cx, cy, nx, ny, order, starts)

def build_safe_offsets(cell_size, radius):
    kmax = int(math.ceil(radius / cell_size)) + 1
    radius_sq = radius * radius
    offsets = []
    for dy in range(-kmax, kmax + 1):
        min_y = max(abs(dy) - 1, 0) * cell_size
        for dx in range(-kmax, kmax + 1):
            min_x = max(abs(dx) - 1, 0) * cell_size
            if min_x * min_x + min_y * min_y < radius_sq:
                offsets.append((dx, dy))
    return np.asarray(offsets, dtype=np.int32)

@njit(parallel=True, fastmath=False, cache=False)
def spatial_gaussian_exact(coords, values, targets, cx, cy, nx, ny, order, starts, offsets, radius_sq, sigma2x2):
    ntarget = targets.size
    nepoch = values.shape[1]
    output = np.empty((ntarget, nepoch), dtype=np.float64)
    true_count = np.zeros(ntarget, dtype=np.int64)
    candidate_count = np.zeros(ntarget, dtype=np.int64)
    for ii in prange(ntarget):
        p = targets[ii]
        x0 = coords[p, 0]
        y0 = coords[p, 1]
        pcx = cx[p]
        pcy = cy[p]
        den = 0.0
        acc = np.zeros(nepoch, dtype=np.float64)
        ntrue = 0
        ncandidate = 0
        for kk in range(offsets.shape[0]):
            qx = pcx + offsets[kk, 0]
            qy = pcy + offsets[kk, 1]
            if qx < 0 or qx >= nx or qy < 0 or (qy >= ny):
                continue
            cell = qy * nx + qx
            q0 = starts[cell]
            q1 = starts[cell + 1]
            ncandidate += q1 - q0
            for jj in range(q0, q1):
                q = order[jj]
                dx = coords[q, 0] - x0
                dy = coords[q, 1] - y0
                dist_sq = dx * dx + dy * dy
                if dist_sq < radius_sq:
                    w = math.exp(-dist_sq / sigma2x2)
                    den += w
                    ntrue += 1
                    for e in range(nepoch):
                        acc[e] += w * values[q, e]
        if den <= 0.0:
            for e in range(nepoch):
                output[ii, e] = np.nan
        else:
            inv_den = 1.0 / den
            for e in range(nepoch):
                output[ii, e] = acc[e] * inv_den
        true_count[ii] = ntrue
        candidate_count[ii] = ncandidate
    return (output, true_count, candidate_count)

__all__ = ['temporal_weight_matrix', 'build_cell_index', 'build_safe_offsets', 'spatial_gaussian_exact']
