from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6371008.8


def local_xy_m(longitude_deg, latitude_deg):
    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)

    if lon.shape != lat.shape:
        raise ValueError("longitude/latitude shape mismatch")

    good = np.isfinite(lon) & np.isfinite(lat)
    if np.count_nonzero(good) < 3:
        raise ValueError("too few finite lon/lat points")

    lon0 = float(np.nanmedian(lon[good]))
    lat0 = float(np.nanmedian(lat[good]))

    x = (
        np.deg2rad(lon - lon0)
        * EARTH_RADIUS_M
        * np.cos(np.deg2rad(lat0))
    )
    y = np.deg2rad(lat - lat0) * EARTH_RADIUS_M

    return np.column_stack((x, y)), lon0, lat0


def select_balanced_anchors(
    coords_m,
    quality,
    *,
    cell_size_m=2000.0,
    anchors_per_cell=8,
):
    xy = np.asarray(coords_m, dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64).reshape(-1)

    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != q.size:
        raise ValueError("anchor geometry/quality shape mismatch")
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    if anchors_per_cell < 1:
        raise ValueError("anchors_per_cell must be >= 1")

    valid = (
        np.all(np.isfinite(xy), axis=1)
        & np.isfinite(q)
        & (q > 0)
    )
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        raise RuntimeError("no valid residual-ramp anchors")

    x0 = float(np.min(xy[ids, 0]))
    y0 = float(np.min(xy[ids, 1]))

    cx = np.floor((xy[:, 0] - x0) / float(cell_size_m)).astype(np.int64)
    cy = np.floor((xy[:, 1] - y0) / float(cell_size_m)).astype(np.int64)

    ny = int(np.max(cy[ids])) + 1
    cell = cx * max(ny, 1) + cy

    order = np.lexsort((-q[ids], cell[ids]))
    ids = ids[order]
    cells = cell[ids]

    keep = np.zeros(ids.size, dtype=bool)
    last = None
    count = 0

    for i, c in enumerate(cells):
        if last is None or c != last:
            last = c
            count = 0
        if count < int(anchors_per_cell):
            keep[i] = True
            count += 1

    anchors = np.sort(ids[keep])
    n_cells = int(np.unique(cell[anchors]).size)

    return anchors, n_cells


def weighted_plane(design, values, weights):
    X = np.asarray(design, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)

    good = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
    )

    Xv = X[good]
    yv = y[good]
    wv = w[good]

    if Xv.shape[0] < 10:
        return np.full(3, np.nan, dtype=np.float64)

    sw = np.sqrt(wv)
    beta, *_ = np.linalg.lstsq(
        Xv * sw[:, None],
        yv * sw,
        rcond=None,
    )
    return beta


def huber_plane(
    design,
    values,
    base_weights,
    *,
    iterations=5,
    delta=1.345,
):
    X = np.asarray(design, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    q = np.asarray(base_weights, dtype=np.float64).reshape(-1)

    beta = weighted_plane(X, y, q)
    if not np.all(np.isfinite(beta)):
        return beta, np.nan, 0

    scale = np.nan
    used = 0

    for _ in range(int(iterations)):
        residual = y - X @ beta
        good = np.isfinite(residual) & np.isfinite(q) & (q > 0)

        if np.count_nonzero(good) < 10:
            break

        r = residual[good]
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        scale = 1.4826 * mad

        if not np.isfinite(scale) or scale <= 1.0e-8:
            break

        u = np.abs(residual) / (float(delta) * scale)
        robust = np.ones_like(u)
        hi = u > 1.0
        robust[hi] = 1.0 / u[hi]

        beta_new = weighted_plane(X, y, q * robust)
        if not np.all(np.isfinite(beta_new)):
            break

        used += 1
        if np.max(np.abs(beta_new - beta)) < 1.0e-8:
            beta = beta_new
            break

        beta = beta_new

    return beta, float(scale), used


def fit_epoch_planes(
    design,
    phase_anchor,
    base_weights,
    *,
    iterations=5,
    delta=1.345,
    temporal_reference_index=0,
):
    X = np.asarray(design, dtype=np.float64)
    ph = np.asarray(phase_anchor, dtype=np.float64)
    q = np.asarray(base_weights, dtype=np.float64).reshape(-1)

    if ph.ndim != 2 or ph.shape[0] != X.shape[0]:
        raise ValueError("phase/design shape mismatch")
    if q.size != X.shape[0]:
        raise ValueError("weight/design shape mismatch")

    nepoch = ph.shape[1]
    coeff = np.full((nepoch, 3), np.nan, dtype=np.float64)
    scale = np.full(nepoch, np.nan, dtype=np.float64)
    used = np.zeros(nepoch, dtype=np.int32)

    for e in range(nepoch):
        if e == int(temporal_reference_index):
            coeff[e, :] = 0.0
            scale[e] = 0.0
            used[e] = 0
            continue

        b, s, it = huber_plane(
            X,
            ph[:, e],
            q,
            iterations=iterations,
            delta=delta,
        )

        if not np.all(np.isfinite(b)):
            raise RuntimeError(
                f"residual-ramp robust plane failed at epoch {e}"
            )

        coeff[e] = b
        scale[e] = s
        used[e] = it

    return coeff, scale, used


# PYPSDS_IFG_NETWORK_RAMP_FINAL_V1

def cell_balanced_weights(
    coords_m,
    *,
    cell_size_m=2000.0,
):
    """Keep all reliable points while equalizing total base weight per cell."""
    xy = np.asarray(coords_m, dtype=np.float64)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coords_m must have shape (N,2)")
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    if xy.shape[0] == 0:
        raise RuntimeError("no residual-ramp points")
    if not np.all(np.isfinite(xy)):
        raise ValueError("coords_m contains non-finite values")

    x0 = float(np.min(xy[:, 0]))
    y0 = float(np.min(xy[:, 1]))

    cx = np.floor(
        (xy[:, 0] - x0) / float(cell_size_m)
    ).astype(np.int64)
    cy = np.floor(
        (xy[:, 1] - y0) / float(cell_size_m)
    ).astype(np.int64)

    ny = int(np.max(cy)) + 1
    cell = cx * max(ny, 1) + cy

    unique, inverse, counts = np.unique(
        cell,
        return_inverse=True,
        return_counts=True,
    )

    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= float(np.mean(weights))

    meta = {
        "occupied_cells": int(unique.size),
        "cell_ps_count_min": int(np.min(counts)),
        "cell_ps_count_p50": float(np.median(counts)),
        "cell_ps_count_p95": float(np.percentile(counts, 95)),
        "cell_ps_count_max": int(np.max(counts)),
    }

    return weights, inverse.astype(np.int32), meta


def build_temporal_design(
    edges,
    ndate,
    *,
    reference_idx=0,
):
    """Build s_ij = q_j - q_i with q(reference_idx)=0."""
    if not (0 <= int(reference_idx) < int(ndate)):
        raise ValueError("invalid reference_idx")

    col = {}
    c = 0
    for t in range(int(ndate)):
        if t == int(reference_idx):
            continue
        col[t] = c
        c += 1

    A = np.zeros(
        (len(edges), int(ndate) - 1),
        dtype=np.float64,
    )

    for e, (i, j) in enumerate(edges):
        i = int(i)
        j = int(j)
        if i != int(reference_idx):
            A[e, col[i]] -= 1.0
        if j != int(reference_idx):
            A[e, col[j]] += 1.0

    return A


def network_project_ifg_slopes(
    edges,
    ndate,
    direct_slopes,
    *,
    reference_idx=0,
):
    """
    Project independently fitted IFG x/y slopes onto the connected
    acquisition network so the applied correction is exactly integrable.
    """
    s = np.asarray(direct_slopes, dtype=np.float64)

    if s.shape != (len(edges), 2):
        raise ValueError("direct_slopes must have shape (Nifg,2)")
    if not np.all(np.isfinite(s)):
        raise ValueError("direct_slopes contains non-finite values")

    A = build_temporal_design(
        edges,
        ndate,
        reference_idx=reference_idx,
    )

    rank = int(np.linalg.matrix_rank(A))
    expected_rank = int(ndate) - 1

    if rank != expected_rank:
        raise RuntimeError(
            f"temporal ramp network rank={rank}, expected={expected_rank}"
        )

    cond = float(np.linalg.cond(A))
    P = np.linalg.pinv(A, rcond=1.0e-12)
    unknown = P @ s
    projected = A @ unknown

    full = np.zeros((int(ndate), 2), dtype=np.float64)
    c = 0
    for t in range(int(ndate)):
        if t == int(reference_idx):
            continue
        full[t] = unknown[c]
        c += 1

    residual = s - projected

    denom = np.maximum(
        np.sqrt(np.sum(s * s, axis=1)),
        1.0e-12,
    )
    diff = np.sqrt(np.sum(residual * residual, axis=1))

    combined_corr = float(
        np.corrcoef(
            s.reshape(-1),
            projected.reshape(-1),
        )[0, 1]
    )

    meta = {
        "design_rank": rank,
        "design_condition_number": cond,
        "combined_xy_correlation": combined_corr,
        "coefficient_diff_rms_rad_per_km": float(
            np.sqrt(np.mean(residual * residual))
        ),
        "coefficient_diff_to_direct_ratio_p50_p95_p99": [
            float(x)
            for x in np.percentile(
                diff / denom,
                [50, 95, 99],
            )
        ],
        "projection_identity_max_abs": float(
            np.max(np.abs(projected - A @ unknown))
        ),
    }

    return projected, full, meta
