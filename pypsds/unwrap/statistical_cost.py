"""StaMPS-inspired temporal edge statistics and SNAPHU deformation costs.

This module builds statistical costs from phase-linked acquisition histories. It
never treats a wrapped-congruent solution as proof of the correct integer branch.
The numerical parameter choices follow the validated StaMPS 3D_QUICK A/B path.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

TWOPI = 2.0 * np.pi
NSHORTCYCLE = 200
COSTSCALE = 100.0
MAXSHORT = 32000
STAMPS_NOISE_STD_CUTOFF_RAD = 1.3
STAMPS_TIME_WIN_DAYS = 730.0

def wrap_phase(x):
    return np.arctan2(np.sin(x), np.cos(x))

def build_stamps_interp_edges(nearest_node: np.ndarray):
    """Python equivalent of the grid-edge part of StaMPS uw_interp.m."""
    nrow, ncol = nearest_node.shape

    rt = nearest_node[:-1, :].reshape(-1)
    rb = nearest_node[1:, :].reshape(-1)
    r_same = rt == rb

    cl = nearest_node[:, :-1].reshape(-1)
    cr = nearest_node[:, 1:].reshape(-1)
    c_same = cl == cr

    r_lo = np.minimum(rt[~r_same], rb[~r_same])
    r_hi = np.maximum(rt[~r_same], rb[~r_same])
    c_lo = np.minimum(cl[~c_same], cr[~c_same])
    c_hi = np.maximum(cl[~c_same], cr[~c_same])

    pairs = np.concatenate(
        [np.column_stack([r_lo, r_hi]), np.column_stack([c_lo, c_hi])],
        axis=0,
    ).astype(np.int32, copy=False)

    unique_pairs, inverse, counts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True
    )

    nr = r_lo.size
    r_inv = inverse[:nr]
    c_inv = inverse[nr:]

    row_eid = np.full(rt.size, -1, dtype=np.int32)
    col_eid = np.full(cl.size, -1, dtype=np.int32)
    row_sign = np.zeros(rt.size, dtype=np.int8)
    col_sign = np.zeros(cl.size, dtype=np.int8)

    r_idx = np.flatnonzero(~r_same)
    c_idx = np.flatnonzero(~c_same)

    row_eid[r_idx] = r_inv.astype(np.int32)
    col_eid[c_idx] = c_inv.astype(np.int32)

    row_sign[r_idx] = np.where(rt[~r_same] < rb[~r_same], 1, -1).astype(np.int8)
    col_sign[c_idx] = np.where(cl[~c_same] < cr[~c_same], 1, -1).astype(np.int8)

    return {
        "edge_nodes": unique_pairs,
        "edge_occurrences": counts.astype(np.int32),
        "row_eid": row_eid.reshape(nrow - 1, ncol),
        "row_sign": row_sign.reshape(nrow - 1, ncol),
        "col_eid": col_eid.reshape(nrow, ncol - 1),
        "col_sign": col_sign.reshape(nrow, ncol - 1),
    }


def build_temporal_operator(edges, dates, time_win_days=STAMPS_TIME_WIN_DAYS):
    """Linear part of StaMPS uw_sb_unwrap_space_time.m 3D_QUICK branch."""
    ndate = len(dates)
    nifg = len(edges)

    G = np.zeros((nifg, ndate), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        G[e, i] = -1.0
        G[e, j] = 1.0

    Gsub = G[:, 1:]
    if np.linalg.matrix_rank(Gsub) != ndate - 1:
        raise RuntimeError("StaMPS 3D_QUICK temporal design is rank deficient")

    P = np.linalg.pinv(Gsub, rcond=1.0e-12)
    R = np.zeros((ndate, nifg), dtype=np.float64)
    R[1:, :] = P

    d0 = datetime.strptime(str(dates[0]), "%Y%m%d")
    day = np.asarray(
        [(datetime.strptime(str(d), "%Y%m%d") - d0).days for d in dates],
        dtype=np.float64,
    )

    W = np.empty((ndate, ndate), dtype=np.float64)
    tw = float(time_win_days)
    for i in range(ndate):
        td2 = (day[i] - day) ** 2
        w = np.exp(-td2 / (2.0 * tw * tw))
        w /= np.sum(w)
        W[i, :] = w

    T = G @ W @ R
    return G, T, day


def build_statistical_edge_model(
    *, node_phase, temporal_edges, temporal_operator, spatial_edge_nodes,
    edge_occurrences, outdir: Path, edge_batch: int, force: bool,
):
    sigsq_path = outdir / "edge_sigsq_short.npy"
    offset_path = outdir / "edge_offset_short.npy"
    bad_path = outdir / "edge_no_stats_mask.npy"

    nedge = spatial_edge_nodes.shape[0]
    if nedge == 0:
        raise ValueError("No spatial edges for statistical-cost estimation")
    nifg = len(temporal_edges)

    if not force and sigsq_path.is_file() and offset_path.is_file() and bad_path.is_file():
        sig = np.load(sigsq_path, mmap_mode="r")
        off = np.load(offset_path, mmap_mode="r")
        bad = np.load(bad_path, mmap_mode="r")
        if sig.shape == (nedge,) and off.shape == (nedge, nifg) and bad.shape == (nedge,):
            print("reuse StaMPS statistical edge model")
            return sigsq_path, offset_path, bad_path

    edge_i = np.asarray([i for i, _ in temporal_edges], dtype=np.int64)
    edge_j = np.asarray([j for _, j in temporal_edges], dtype=np.int64)

    sigsq = np.lib.format.open_memmap(sigsq_path, mode="w+", dtype=np.int16, shape=(nedge,))
    offset = np.lib.format.open_memmap(offset_path, mode="w+", dtype=np.int16, shape=(nedge, nifg))
    bad = np.lib.format.open_memmap(bad_path, mode="w+", dtype=np.bool_, shape=(nedge,))

    for s in range(0, nedge, edge_batch):
        e = min(nedge, s + edge_batch)
        pair = spatial_edge_nodes[s:e]
        lo = pair[:, 0]
        hi = pair[:, 1]

        d_acq = wrap_phase(
            np.asarray(node_phase[hi, :], dtype=np.float64)
            - np.asarray(node_phase[lo, :], dtype=np.float64)
        )
        y = wrap_phase(d_acq[:, edge_j] - d_acq[:, edge_i])
        smooth = y @ temporal_operator.T
        noise = wrap_phase(y - smooth)

        noise_std = np.std(noise, axis=1, ddof=1)
        no_stats = noise_std > STAMPS_NOISE_STD_CUTOFF_RAD
        sigsq_noise = (noise_std / TWOPI) ** 2

        ss = np.rint(
            sigsq_noise * (NSHORTCYCLE ** 2) / COSTSCALE * edge_occurrences[s:e]
        )
        ss = np.clip(ss, 1, np.iinfo(np.int16).max).astype(np.int16)

        off = np.rint((y - smooth) / TWOPI * NSHORTCYCLE)
        off = np.clip(off, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

        sigsq[s:e] = ss
        offset[s:e, :] = off
        bad[s:e] = no_stats

        print(f"[STAT COST MODEL] {e:,}/{nedge:,} no-stats={100*np.mean(no_stats):.2f}%", flush=True)

    sigsq.flush(); offset.flush(); bad.flush()
    return sigsq_path, offset_path, bad_path


def write_cost_file(
    path: Path, pair_index: int, row_eid, row_sign, col_eid, col_sign,
    edge_sigsq, edge_offset, edge_bad,
):
    rowcost = np.zeros(row_eid.shape + (4,), dtype=np.int16)
    colcost = np.zeros(col_eid.shape + (4,), dtype=np.int16)

    rowcost[..., 1] = 1
    colcost[..., 1] = 1
    rowcost[..., 2] = MAXSHORT
    colcost[..., 2] = MAXSHORT

    row_stats = row_eid < 0
    col_stats = col_eid < 0
    r_non = row_eid >= 0
    c_non = col_eid >= 0

    r_good = np.zeros_like(r_non)
    c_good = np.zeros_like(c_non)
    r_good[r_non] = ~edge_bad[row_eid[r_non]]
    c_good[c_non] = ~edge_bad[col_eid[c_non]]

    row_stats |= r_good
    col_stats |= c_good

    rowcost[..., 3] = np.where(row_stats, -MAXSHORT, 1).astype(np.int16)
    colcost[..., 3] = np.where(col_stats, -MAXSHORT, 1).astype(np.int16)

    if np.any(r_good):
        eid = row_eid[r_good]
        rowcost[..., 1][r_good] = edge_sigsq[eid]
        signed = edge_offset[eid, pair_index].astype(np.int32) * row_sign[r_good].astype(np.int32)
        rowcost[..., 0][r_good] = np.clip(-signed, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

    if np.any(c_good):
        eid = col_eid[c_good]
        colcost[..., 1][c_good] = edge_sigsq[eid]
        signed = edge_offset[eid, pair_index].astype(np.int32) * col_sign[c_good].astype(np.int32)
        colcost[..., 0][c_good] = np.clip(signed, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

    with path.open("wb") as f:
        rowcost.tofile(f)
        colcost.tofile(f)


def snaphu_config_text():
    return "\n".join([
        "INFILE wrapped.cpx",
        "OUTFILE unwrapped.f32",
        "COSTINFILE snaphu.costinfile",
        "STATCOSTMODE DEFO",
        "INFILEFORMAT COMPLEX_DATA",
        "OUTFILEFORMAT FLOAT_DATA",
        "",
    ])



def summarize_spatial_integer_gradients(node_grid, acquisition_integer_cycles, dates):
    """Report gradients of the integer branch on occupied 4-neighbor grid edges.

    These are diagnostics, not proof that a discontinuity is unphysical. The
    trend is expressed in integer cycles per year without an assumed LOS sign.
    """
    grid = np.asarray(node_grid, dtype=np.int64)
    cycles = np.asarray(acquisition_integer_cycles)
    if grid.ndim != 2 or cycles.ndim != 2 or cycles.shape[1] != len(dates):
        raise ValueError("Invalid grid/cycle/date dimensions")
    if np.any(grid < -1) or (grid.size and np.max(grid) >= cycles.shape[0]):
        raise ValueError("Node grid index is out of bounds")
    if len(dates) < 2:
        raise ValueError("At least two acquisitions are required")
    t0 = datetime.strptime(str(dates[0]), "%Y%m%d")
    x = np.asarray([(datetime.strptime(str(d), "%Y%m%d") - t0).days
                    for d in dates], dtype=np.float64) / 365.2425
    x -= x.mean()
    denominator = float(x @ x)
    if denominator <= 0:
        raise ValueError("Acquisition dates have zero temporal span")
    rate = np.asarray(cycles, dtype=np.float64) @ x / denominator

    all_jumps = []
    a, b = grid[:, :-1], grid[:, 1:]
    mask = (a >= 0) & (b >= 0) & (a != b)
    if np.any(mask):
        all_jumps.append(np.abs(rate[b[mask]] - rate[a[mask]]))
    a, b = grid[:-1, :], grid[1:, :]
    mask = (a >= 0) & (b >= 0) & (a != b)
    if np.any(mask):
        all_jumps.append(np.abs(rate[b[mask]] - rate[a[mask]]))
    if not all_jumps:
        return {"edge_count": 0, "integer_gradient_cycles_per_year_p50_p90_p95_p99": [None]*4}
    y = np.concatenate(all_jumps)
    return {
        "edge_count": int(y.size),
        "integer_gradient_cycles_per_year_p50_p90_p95_p99":
            [float(v) for v in np.percentile(y, [50, 90, 95, 99])],
        "fraction_exceeding_0p2_cycles_per_year": float(np.mean(y > 0.2)),
        "scientific_note": "Integer-gradient magnitude alone cannot identify an unwrapping error",
    }
