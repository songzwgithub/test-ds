from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numba import njit
from scipy import ndimage

from pypsds.config import cfg_get
from pypsds.geometry.inputs import resolve_geometry_inputs
from pypsds.gamma.geometry import geometry_from_par

TWOPI = 2.0 * np.pi
TYPE_PS = np.uint8(1)


def wrap_phase(x):
    x = np.asarray(x)
    return np.arctan2(np.sin(x), np.cos(x))


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_itab(path: Path, ndate: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with path.open() as f:
        for raw in f:
            x = raw.split()
            if len(x) < 2:
                continue
            i = int(x[0]) - 1
            j = int(x[1]) - 1
            if not (0 <= i < ndate and 0 <= j < ndate and i != j):
                raise RuntimeError(f"Invalid temporal-network line: {raw.rstrip()}")
            edges.append((i, j))
    if not edges:
        raise RuntimeError(f"No temporal edges in {path}")
    return edges


def build_design_matrix(edges, ndate: int, reference_idx: int) -> np.ndarray:
    cols = {}
    k = 0
    for t in range(ndate):
        if t == reference_idx:
            continue
        cols[t] = k
        k += 1
    A = np.zeros((len(edges), ndate - 1), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        if i != reference_idx:
            A[e, cols[i]] -= 1.0
        if j != reference_idx:
            A[e, cols[j]] += 1.0
    return A


@njit(cache=True)
def _select_representatives(
    rows,
    cols,
    point_type,
    tc,
    row0,
    col0,
    row_spacing_m,
    col_spacing_m,
    grid_size_m,
    n_grid_cols,
    n_grid_cells,
):
    best_id = np.full(n_grid_cells, -1, dtype=np.int64)
    best_score = np.full(n_grid_cells, -1.0e30, dtype=np.float32)

    for p in range(rows.size):
        gr = int(math.floor((rows[p] - row0) * row_spacing_m / grid_size_m))
        gc = int(math.floor((cols[p] - col0) * col_spacing_m / grid_size_m))
        cell = gr * n_grid_cols + gc

        if point_type[p] == TYPE_PS:
            score = 3.0
        else:
            q = tc[p]
            if not np.isfinite(q):
                q = 0.0
            score = 1.0 + float(q)

        if score > best_score[cell]:
            best_score[cell] = score
            best_id[cell] = p

    return best_id, best_score


def build_representative_grid(
    rows: np.ndarray,
    cols: np.ndarray,
    point_type: np.ndarray,
    tc: np.ndarray,
    *,
    row_spacing_m: float,
    col_spacing_m: float,
    grid_size_m: float,
):
    if grid_size_m <= 0:
        raise ValueError("grid_size_m must be > 0")
    row0 = int(rows.min())
    col0 = int(cols.min())
    gr = np.floor((rows.astype(np.float64) - row0) * row_spacing_m / grid_size_m)
    gc = np.floor((cols.astype(np.float64) - col0) * col_spacing_m / grid_size_m)
    nrow = int(gr.max()) + 1
    ncol = int(gc.max()) + 1
    ncell = nrow * ncol

    best_id, best_score = _select_representatives(
        rows,
        cols,
        point_type,
        tc,
        row0,
        col0,
        float(row_spacing_m),
        float(col_spacing_m),
        float(grid_size_m),
        ncol,
        ncell,
    )

    rep_grid = best_id.reshape((nrow, ncol))
    occ = rep_grid >= 0
    rep_ids = rep_grid[occ].astype(np.int64, copy=False)

    node_grid = np.full((nrow, ncol), -1, dtype=np.int32)
    node_grid[occ] = np.arange(rep_ids.size, dtype=np.int32)

    return {
        "row0": row0,
        "col0": col0,
        "shape": (nrow, ncol),
        "rep_grid": rep_grid,
        "node_grid": node_grid,
        "occupied": occ,
        "rep_ids": rep_ids,
        "best_score": best_score.reshape((nrow, ncol)),
    }


def build_dense_support(
    node_grid: np.ndarray,
    node_quality: np.ndarray,
    *,
    grid_size_m: float,
    gap_scale_m: float,
    min_corr: float,
):
    occ = node_grid >= 0
    if not np.any(occ):
        raise RuntimeError("No occupied coarse-grid cells")

    distance_m, nearest_idx = ndimage.distance_transform_edt(
        ~occ,
        sampling=(float(grid_size_m), float(grid_size_m)),
        return_indices=True,
    )
    nearest_node = node_grid[nearest_idx[0], nearest_idx[1]]
    if np.any(nearest_node < 0):
        raise RuntimeError("Nearest-node construction failed")

    q = node_quality[nearest_node]
    if gap_scale_m > 0:
        q = q * np.exp(-((distance_m / float(gap_scale_m)) ** 2))
    corr = np.clip(q, float(min_corr), 0.999).astype(np.float32)

    return (
        nearest_node.astype(np.int32, copy=False),
        distance_m.astype(np.float32, copy=False),
        corr,
    )


def _snaphu_config_text(corr_path: Path) -> str:
    return "\n".join(
        [
            "INFILE wrapped.f32",
            "OUTFILE unwrapped.f32",
            f"CORRFILE {corr_path}",
            "STATCOSTMODE DEFO",
            "INFILEFORMAT FLOAT_DATA",
            "OUTFILEFORMAT FLOAT_DATA",
            "CORRFILEFORMAT FLOAT_DATA",
            "INITMETHOD MCF",
            "",
        ]
    )


def _run_one_snaphu(
    *,
    pair_index: int,
    edge: tuple[int, int],
    dates,
    node_phase: np.ndarray,
    nearest_node: np.ndarray,
    occupied: np.ndarray,
    corr_path: Path,
    pair_root: Path,
    cycles_dir: Path,
    snaphu_exe: str,
    force: bool,
    keep_scratch: bool,
):
    i, j = edge
    tag = f"pair{pair_index + 1:03d}_{dates[i]}_{dates[j]}"
    cycle_path = cycles_dir / f"{tag}_cycles.npy"

    if cycle_path.is_file() and not force:
        k = np.load(cycle_path, mmap_mode="r")
        if k.ndim == 1 and k.size == node_phase.shape[0]:
            return {
                "pair_index": pair_index,
                "tag": tag,
                "reused": True,
                "cycles_path": str(cycle_path),
                "max_abs_cycle": int(np.max(np.abs(k))) if k.size else 0,
                "wrap_back_max_error_rad": None,
                "seconds": 0.0,
            }

    t0 = time.perf_counter()
    pair_dir = pair_root / tag
    pair_dir.mkdir(parents=True, exist_ok=True)

    node_wrapped = wrap_phase(
        np.asarray(node_phase[:, j], dtype=np.float64)
        - np.asarray(node_phase[:, i], dtype=np.float64)
    ).astype(np.float32)

    dense = node_wrapped[
        np.asarray(nearest_node, dtype=np.int32).reshape(-1)
    ].reshape(nearest_node.shape)

    in_path = pair_dir / "wrapped.f32"
    out_path = pair_dir / "unwrapped.f32"
    conf_path = pair_dir / "snaphu.conf"
    log_path = pair_dir / "snaphu.log"

    np.ascontiguousarray(dense, dtype=np.float32).tofile(in_path)
    conf_path.write_text(_snaphu_config_text(corr_path), encoding="utf-8")

    cmd = [
        snaphu_exe,
        "-d",
        "-f",
        conf_path.name,
        str(dense.shape[1]),
    ]

    with log_path.open("w", encoding="utf-8") as log:
        cp = subprocess.run(
            cmd,
            cwd=pair_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if cp.returncode != 0:
        raise RuntimeError(
            f"SNAPHU failed for {tag} with exit code {cp.returncode}; log={log_path}"
        )

    raw = np.fromfile(out_path, dtype=np.float32)
    expected = int(dense.size)
    if raw.size != expected:
        raise RuntimeError(
            f"{tag}: SNAPHU output size {raw.size} != expected {expected}"
        )
    unwrapped = raw.reshape(dense.shape)
    u_node = np.asarray(unwrapped[occupied], dtype=np.float64)

    if not np.all(np.isfinite(u_node)):
        raise RuntimeError(f"{tag}: non-finite SNAPHU phase at occupied coarse nodes")

    wrap_error = float(
        np.max(
            np.abs(
                wrap_phase(
                    u_node
                    - node_wrapped.astype(np.float64)
                )
            )
        )
    )

    k64 = np.rint(
        (u_node - node_wrapped.astype(np.float64)) / TWOPI
    ).astype(np.int64)
    max_abs = int(np.max(np.abs(k64))) if k64.size else 0
    if max_abs >= np.iinfo(np.int16).max:
        raise RuntimeError(
            f"{tag}: coarse integer cycle magnitude {max_abs} exceeds int16"
        )

    atomic_save(cycle_path, k64.astype(np.int16))

    if not keep_scratch:
        for p in (in_path, out_path, conf_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    return {
        "pair_index": pair_index,
        "tag": tag,
        "reused": False,
        "cycles_path": str(cycle_path),
        "max_abs_cycle": max_abs,
        "wrap_back_max_error_rad": wrap_error,
        "seconds": float(time.perf_counter() - t0),
        "log": str(log_path),
    }


def _choose_temporal_edges(A: np.ndarray, bad_fraction: np.ndarray, threshold: float):
    n_unknown = A.shape[1]
    keep = np.asarray(bad_fraction <= float(threshold), dtype=bool)
    if np.count_nonzero(keep) >= n_unknown:
        rank = int(np.linalg.matrix_rank(A[keep, :]))
    else:
        rank = 0

    if rank < n_unknown:
        keep[:] = False
        order = np.argsort(bad_fraction, kind="stable")
        selected = []
        rank = 0
        for eid in order:
            selected.append(int(eid))
            test = A[np.asarray(selected, dtype=np.int64), :]
            rank = int(np.linalg.matrix_rank(test))
            if rank == n_unknown:
                break
        if rank != n_unknown:
            raise RuntimeError(
                f"Temporal integer network is rank deficient: rank={rank}/{n_unknown}"
            )
        keep[np.asarray(selected, dtype=np.int64)] = True

    return keep


def _threadpool_context(limits: int):
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return contextlib.nullcontext()
    return threadpool_limits(limits=max(1, int(limits)))


def _integer_solve_pass(
    K: np.ndarray,
    A_all: np.ndarray,
    keep: np.ndarray,
    gauge: np.ndarray,
    *,
    ndate: int,
    reference_idx: int,
    batch_size: int,
    iterations: int,
    blas_threads: int,
    cycles_out: np.memmap | None,
    mismatch_out: np.memmap | None,
):
    A = np.asarray(A_all[keep, :], dtype=np.float32)
    Aall = np.asarray(A_all, dtype=np.float32)
    rank = int(np.linalg.matrix_rank(A.astype(np.float64)))
    if rank != A.shape[1]:
        raise RuntimeError(
            f"Selected temporal integer graph rank={rank}, expected={A.shape[1]}"
        )
    P = np.linalg.pinv(A.astype(np.float64), rcond=1.0e-12).astype(np.float32)

    edge_bad = np.zeros(Aall.shape[0], dtype=np.int64)
    total_nodes = int(K.shape[0])
    unknown_dates = [t for t in range(ndate) if t != reference_idx]

    with _threadpool_context(blas_threads):
        for b0 in range(0, total_nodes, batch_size):
            b1 = min(total_nodes, b0 + batch_size)
            kb_all = np.asarray(K[b0:b1, :], dtype=np.float32)
            kb_all -= gauge[None, :].astype(np.float32)
            kb = kb_all[:, keep]

            x = np.rint(kb @ P.T).astype(np.float32)
            for _ in range(max(0, int(iterations))):
                residual = kb - x @ A.T
                delta = np.rint(residual @ P.T)
                if not np.any(delta):
                    break
                x += delta

            pred_all = x @ Aall.T
            resid_all = np.rint(kb_all - pred_all).astype(np.int16)
            edge_bad += np.count_nonzero(resid_all != 0, axis=0)

            resid_keep = resid_all[:, keep]
            mismatch_fraction = (
                np.count_nonzero(resid_keep != 0, axis=1)
                / max(1, resid_keep.shape[1])
            ).astype(np.float32)

            if mismatch_out is not None:
                mismatch_out[b0:b1] = mismatch_fraction

            if cycles_out is not None:
                full = np.zeros((b1 - b0, ndate), dtype=np.int16)
                xi = np.rint(x).astype(np.int64)
                if xi.size and int(np.max(np.abs(xi))) >= np.iinfo(np.int16).max:
                    raise RuntimeError(
                        "Temporal acquisition integer cycle exceeds int16 range"
                    )
                full[:, unknown_dates] = xi.astype(np.int16)
                cycles_out[b0:b1, :] = full

    return edge_bad / max(1, total_nodes)


def synchronize_temporal_integer_cycles(
    K: np.ndarray,
    edges,
    *,
    ndate: int,
    reference_idx: int,
    batch_size: int,
    iterations: int,
    edge_bad_threshold: float,
    strict_mismatch_fraction: float,
    blas_threads: int,
    work_dir: Path,
):
    A = build_design_matrix(edges, ndate, reference_idx)
    expected_rank = ndate - 1
    rank = int(np.linalg.matrix_rank(A))
    if rank != expected_rank:
        raise RuntimeError(
            f"Temporal design matrix rank={rank}, expected={expected_rank}"
        )

    # SNAPHU may add one spatially global integer offset to each IFG.  Its
    # *integrable* part is merely a common acquisition-time gauge and is
    # harmless (the later spatial reference removes it).  Estimate only the
    # common NON-integrable residual: solve a representative node sample first,
    # then take the robust per-edge median of the signed integer residual.
    nsample = min(int(K.shape[0]), 50000)
    if nsample == K.shape[0]:
        sample = np.asarray(K, dtype=np.float32)
    else:
        sample_ids = np.linspace(0, K.shape[0] - 1, nsample, dtype=np.int64)
        sample = np.asarray(K[sample_ids, :], dtype=np.float32)

    Af = A.astype(np.float32)
    P_all = np.linalg.pinv(A, rcond=1.0e-12).astype(np.float32)
    sx = np.rint(sample @ P_all.T).astype(np.float32)
    for _ in range(max(0, int(iterations))):
        sr = sample - sx @ Af.T
        sd = np.rint(sr @ P_all.T)
        if not np.any(sd):
            break
        sx += sd
    signed_residual = np.rint(sample - sx @ Af.T)
    gauge64 = np.rint(np.median(signed_residual, axis=0)).astype(np.int64)
    if gauge64.size and int(np.max(np.abs(gauge64))) >= np.iinfo(np.int16).max:
        raise RuntimeError("Common IFG integer gauge exceeds int16 range")
    gauge = gauge64.astype(np.int16)

    initial_keep = np.ones(len(edges), dtype=bool)
    initial_bad = _integer_solve_pass(
        K,
        A,
        initial_keep,
        gauge,
        ndate=ndate,
        reference_idx=reference_idx,
        batch_size=batch_size,
        iterations=iterations,
        blas_threads=blas_threads,
        cycles_out=None,
        mismatch_out=None,
    )

    keep = _choose_temporal_edges(A, initial_bad, edge_bad_threshold)

    cycles_path = work_dir / "node_acquisition_integer_cycles.npy"
    mismatch_path = work_dir / "node_temporal_integer_mismatch_fraction.npy"
    cycles = np.lib.format.open_memmap(
        cycles_path,
        mode="w+",
        dtype=np.int16,
        shape=(K.shape[0], ndate),
    )
    mismatch = np.lib.format.open_memmap(
        mismatch_path,
        mode="w+",
        dtype=np.float32,
        shape=(K.shape[0],),
    )

    final_bad = _integer_solve_pass(
        K,
        A,
        keep,
        gauge,
        ndate=ndate,
        reference_idx=reference_idx,
        batch_size=batch_size,
        iterations=iterations,
        blas_threads=blas_threads,
        cycles_out=cycles,
        mismatch_out=mismatch,
    )
    cycles.flush()
    mismatch.flush()

    node_valid = np.asarray(mismatch <= float(strict_mismatch_fraction), dtype=bool)

    atomic_save(work_dir / "temporal_edge_selected_mask.npy", keep)
    atomic_save(work_dir / "temporal_edge_initial_bad_fraction.npy", initial_bad)
    atomic_save(work_dir / "temporal_edge_final_bad_fraction.npy", final_bad)
    atomic_save(work_dir / "global_ifg_integer_gauge.npy", gauge)

    return {
        "cycles_path": cycles_path,
        "mismatch_path": mismatch_path,
        "node_valid": node_valid,
        "edge_keep": keep,
        "initial_edge_bad_fraction": initial_bad,
        "final_edge_bad_fraction": final_bad,
        "global_ifg_integer_gauge": gauge,
        "rank": rank,
    }


def _config_signature(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run_stamps3d_backend(*, cfg, config_path: Path, paths, stack, force: bool = False):
    t0 = time.perf_counter()
    processing = Path(paths.output_dir) / "processing"
    pps = processing / "point_phase_stack"
    network = processing / "network"
    outdir = processing / "stamps3d_unwrap"
    final_dir = processing / "final_unwrap"
    outdir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    phase = np.load(pps / "phase_rad.npy", mmap_mode="r")
    rows = np.load(pps / "rows.npy", mmap_mode="r")
    cols = np.load(pps / "cols.npy", mmap_mode="r")
    point_type = np.load(pps / "point_type.npy", mmap_mode="r")
    tc = np.load(pps / "temporal_coherence.npy", mmap_mode="r")

    npoint, ndate = phase.shape
    if rows.size != npoint or cols.size != npoint:
        raise RuntimeError("PointPhaseStack geometry/phase mismatch")
    if len(stack.dates) != ndate:
        raise RuntimeError("PointPhaseStack acquisition count mismatch")

    edges = load_itab(network / "network.itab", ndate)
    nifg = len(edges)
    reference_idx = int(cfg_get(cfg, "phase_linking.temporal_reference_index", 0))
    if not (0 <= reference_idx < ndate):
        raise RuntimeError("Invalid temporal reference index")

    prefix = "unwrap.stamps3d_snaphu"
    grid_size_m = float(cfg_get(cfg, f"{prefix}.grid_size_m", 200.0))
    gap_scale_m = float(cfg_get(cfg, f"{prefix}.gap_scale_m", 600.0))
    min_corr = float(cfg_get(cfg, f"{prefix}.min_corr", 0.03))
    strict_mismatch = float(
        cfg_get(cfg, f"{prefix}.strict_max_temporal_mismatch_fraction", 0.02)
    )
    edge_bad_threshold = float(
        cfg_get(cfg, f"{prefix}.temporal_edge_bad_fraction", 0.10)
    )
    sync_batch = int(cfg_get(cfg, f"{prefix}.temporal_sync_batch", 8192))
    sync_iterations = int(cfg_get(cfg, f"{prefix}.temporal_sync_iterations", 3))
    point_batch = int(cfg_get(cfg, f"{prefix}.point_batch", 131072))
    blas_threads = int(cfg_get(cfg, f"{prefix}.blas_threads", 8))
    keep_scratch = bool(cfg_get(cfg, f"{prefix}.keep_snaphu_scratch", False))

    raw_workers = cfg_get(cfg, f"{prefix}.snaphu_workers", "auto")
    if raw_workers in (None, "", "auto"):
        snaphu_workers = min(8, max(1, os.cpu_count() or 1), nifg)
    else:
        snaphu_workers = max(1, min(int(raw_workers), nifg))

    snaphu_name = str(cfg_get(cfg, f"{prefix}.snaphu", "snaphu"))
    snaphu_exe = shutil.which(snaphu_name)
    if snaphu_exe is None:
        raise RuntimeError(f"SNAPHU executable not found: {snaphu_name!r}")

    geom_inputs = resolve_geometry_inputs(cfg, paths)
    geom = geometry_from_par(geom_inputs.reference_rslc_par)
    row_spacing_m = float(geom.azimuth_spacing_m)
    col_spacing_m = float(geom.ground_range_spacing_m)

    print("=" * 104)
    print("pyPSDS StaMPS-style coarse-grid + SNAPHU + temporal-integer backend")
    print("=" * 104)
    print("config                     :", config_path)
    print("points / acquisitions / IFGs:", f"{npoint:,}", "/", ndate, "/", nifg)
    print("radar spacing [m]          :", row_spacing_m, "/", col_spacing_m)
    print("coarse grid size [m]       :", grid_size_m)
    print("SNAPHU workers             :", snaphu_workers)
    print("temporal sync BLAS threads :", blas_threads)
    print("force                      :", force)

    tg = time.perf_counter()
    grid = build_representative_grid(
        np.asarray(rows),
        np.asarray(cols),
        np.asarray(point_type),
        np.asarray(tc),
        row_spacing_m=row_spacing_m,
        col_spacing_m=col_spacing_m,
        grid_size_m=grid_size_m,
    )
    rep_ids = grid["rep_ids"]
    occupied = grid["occupied"]
    node_grid = grid["node_grid"]
    nnode = int(rep_ids.size)

    rep_type = np.asarray(point_type[rep_ids], dtype=np.uint8)
    rep_tc = np.asarray(tc[rep_ids], dtype=np.float32)
    node_quality = np.where(
        rep_type == TYPE_PS,
        np.float32(0.95),
        np.clip(np.nan_to_num(rep_tc, nan=0.80), 0.50, 0.99),
    ).astype(np.float32)

    nearest_node, distance_m, corr = build_dense_support(
        node_grid,
        node_quality,
        grid_size_m=grid_size_m,
        gap_scale_m=gap_scale_m,
        min_corr=min_corr,
    )

    atomic_save(outdir / "representative_point_ids.npy", rep_ids.astype(np.int64))
    atomic_save(outdir / "coarse_node_grid.npy", node_grid)
    atomic_save(outdir / "coarse_support_distance_m.npy", distance_m)
    atomic_save(outdir / "coarse_correlation.npy", corr)

    node_phase_path = outdir / "representative_phase_rad.npy"
    node_phase = np.lib.format.open_memmap(
        node_phase_path,
        mode="w+",
        dtype=np.float32,
        shape=(nnode, ndate),
    )
    node_copy_batch = max(4096, min(point_batch, 262144))
    for b0 in range(0, nnode, node_copy_batch):
        b1 = min(nnode, b0 + node_copy_batch)
        node_phase[b0:b1, :] = np.asarray(phase[rep_ids[b0:b1], :], dtype=np.float32)
    node_phase.flush()
    del node_phase
    node_phase = np.load(node_phase_path, mmap_mode="r")

    corr_path = outdir / "snaphu_correlation.f32"
    np.ascontiguousarray(corr, dtype=np.float32).tofile(corr_path)

    grid_seconds = time.perf_counter() - tg
    print("coarse grid shape          :", grid["shape"])
    print("occupied coarse nodes      :", f"{nnode:,}")
    print("grid build seconds         :", f"{grid_seconds:.2f}")

    signature_payload = {
        "algorithm": "pypsds-stamps3d-snaphu-v1",
        "dates": [str(x) for x in stack.dates],
        "edges": [[int(i), int(j)] for i, j in edges],
        "npoint": int(npoint),
        "grid_size_m": grid_size_m,
        "row_spacing_m": row_spacing_m,
        "col_spacing_m": col_spacing_m,
        "gap_scale_m": gap_scale_m,
        "min_corr": min_corr,
        "reference_idx": reference_idx,
        "representative_sha256": hashlib.sha256(
            np.asarray(rep_ids, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    signature = _config_signature(signature_payload)

    cycles_dir = outdir / "pair_cycles"
    pair_root = outdir / "snaphu_work"
    cycles_dir.mkdir(parents=True, exist_ok=True)
    pair_root.mkdir(parents=True, exist_ok=True)

    old_sig_path = outdir / "backend_signature.json"
    if old_sig_path.is_file() and not force:
        try:
            old_sig = json.loads(old_sig_path.read_text()).get("signature")
        except Exception:
            old_sig = None
        if old_sig != signature:
            raise RuntimeError(
                "Existing stamps3d pair-cycle cache has a different signature; "
                "rerun unwrap with --force"
            )
    atomic_json(old_sig_path, {"signature": signature, "payload": signature_payload})

    ts = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=snaphu_workers, thread_name_prefix="snaphu") as pool:
        futures = {
            pool.submit(
                _run_one_snaphu,
                pair_index=e,
                edge=edge,
                dates=stack.dates,
                node_phase=node_phase,
                nearest_node=nearest_node,
                occupied=occupied,
                corr_path=corr_path.resolve(),
                pair_root=pair_root,
                cycles_dir=cycles_dir,
                snaphu_exe=snaphu_exe,
                force=force,
                keep_scratch=keep_scratch,
            ): e
            for e, edge in enumerate(edges)
        }
        done = 0
        for future in as_completed(futures):
            info = future.result()
            results.append(info)
            done += 1
            if done == 1 or done % 8 == 0 or done == nifg:
                print(
                    f"[SNAPHU] {done:3d}/{nifg:3d} "
                    f"elapsed={time.perf_counter() - ts:.1f}s",
                    flush=True,
                )

    results.sort(key=lambda x: x["pair_index"])
    wrap_errors = [
        x["wrap_back_max_error_rad"]
        for x in results
        if x["wrap_back_max_error_rad"] is not None
    ]
    if wrap_errors and max(wrap_errors) > 1.0e-3:
        raise RuntimeError(
            f"SNAPHU coarse wrap-back error too large: {max(wrap_errors)} rad"
        )

    K_path = outdir / "coarse_ifg_integer_cycles.npy"
    K = np.lib.format.open_memmap(
        K_path,
        mode="w+",
        dtype=np.int16,
        shape=(nnode, nifg),
    )
    for e, info in enumerate(results):
        k = np.load(info["cycles_path"], mmap_mode="r")
        if k.size != nnode:
            raise RuntimeError("Pair integer-cycle cache shape mismatch")
        K[:, e] = np.asarray(k, dtype=np.int16)
    K.flush()
    snaphu_seconds = time.perf_counter() - ts

    ti = time.perf_counter()
    sync = synchronize_temporal_integer_cycles(
        K,
        edges,
        ndate=ndate,
        reference_idx=reference_idx,
        batch_size=sync_batch,
        iterations=sync_iterations,
        edge_bad_threshold=edge_bad_threshold,
        strict_mismatch_fraction=strict_mismatch,
        blas_threads=blas_threads,
        work_dir=outdir,
    )
    integer_cycles = np.load(sync["cycles_path"], mmap_mode="r")
    node_valid = sync["node_valid"]
    sync_seconds = time.perf_counter() - ti

    node_unw_path = outdir / "coarse_acquisition_phase_unwrapped_rad.npy"
    node_unw = np.lib.format.open_memmap(
        node_unw_path,
        mode="w+",
        dtype=np.float32,
        shape=(nnode, ndate),
    )
    for b0 in range(0, nnode, sync_batch):
        b1 = min(nnode, b0 + sync_batch)
        node_unw[b0:b1, :] = (
            np.asarray(node_phase[b0:b1, :], dtype=np.float32)
            + np.float32(TWOPI)
            * np.asarray(integer_cycles[b0:b1, :], dtype=np.float32)
        )
    node_unw.flush()

    print("temporal selected edges     :", int(np.count_nonzero(sync["edge_keep"])), "/", nifg)
    print("valid coarse nodes          :", f"{int(node_valid.sum()):,}/{nnode:,}")
    print("temporal sync seconds       :", f"{sync_seconds:.2f}")

    tr = time.perf_counter()
    full_path = outdir / "acquisition_phase_unwrapped_rad.npy"
    full = np.lib.format.open_memmap(
        full_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint, ndate),
    )
    strict_path = final_dir / "strict_unwrap_valid_mask.npy"
    strict_mm = np.lib.format.open_memmap(
        strict_path,
        mode="w+",
        dtype=np.bool_,
        shape=(npoint,),
    )

    node_unw = np.load(node_unw_path, mmap_mode="r")
    node_grid_flat = node_grid.reshape(-1)
    ncol = grid["shape"][1]
    max_wrap_parity = 0.0

    for p0 in range(0, npoint, point_batch):
        p1 = min(npoint, p0 + point_batch)
        rr = np.asarray(rows[p0:p1], dtype=np.int64)
        cc = np.asarray(cols[p0:p1], dtype=np.int64)
        gr = np.floor((rr - grid["row0"]) * row_spacing_m / grid_size_m).astype(np.int64)
        gc = np.floor((cc - grid["col0"]) * col_spacing_m / grid_size_m).astype(np.int64)
        cell = gr * ncol + gc
        nid = node_grid_flat[cell]
        if np.any(nid < 0):
            raise RuntimeError("Point mapped to empty coarse cell")

        ph = np.asarray(phase[p0:p1, :], dtype=np.float32)
        rep_wrapped = np.asarray(node_phase[nid, :], dtype=np.float32)
        rep_unwrapped = np.asarray(node_unw[nid, :], dtype=np.float32)

        restored = (
            rep_unwrapped
            + wrap_phase(
                ph.astype(np.float64)
                - rep_wrapped.astype(np.float64)
            ).astype(np.float32)
        )
        full[p0:p1, :] = restored
        strict_mm[p0:p1] = node_valid[nid]

        err = float(
            np.max(
                np.abs(
                    wrap_phase(
                        restored.astype(np.float64)
                        - ph.astype(np.float64)
                    )
                )
            )
        )
        max_wrap_parity = max(max_wrap_parity, err)

        if p1 == npoint or p1 % 1_000_000 < point_batch:
            print(
                f"[RESTORE] {p1:,}/{npoint:,} "
                f"({100*p1/npoint:.2f}%)",
                flush=True,
            )

    full.flush()
    strict_mm.flush()

    strict_arr = np.asarray(strict_mm, dtype=bool)
    strict_ids = np.flatnonzero(strict_arr).astype(np.int32)
    atomic_save(final_dir / "strict_point_ids.npy", strict_ids)
    atomic_save(
        final_dir / "global_ifg_integer_delta.npy",
        np.zeros(nifg, dtype=np.int32),
    )

    strict_fraction = float(strict_ids.size / max(1, npoint))
    restore_seconds = time.perf_counter() - tr

    mismatch = np.load(sync["mismatch_path"], mmap_mode="r")
    mismatch_q = np.percentile(np.asarray(mismatch), [50, 90, 95, 99, 100]).tolist()
    final_bad = np.asarray(sync["final_edge_bad_fraction"])
    bad_q = np.percentile(final_bad, [50, 90, 95, 99, 100]).tolist()

    manifest = {
        "status": "PASS" if max_wrap_parity <= 1.0e-4 else "REVIEW",
        "backend": "stamps3d_snaphu",
        "algorithm": (
            "StaMPS-inspired metric coarse grid + SNAPHU spatial unwrap + "
            "redundant temporal integer-network synchronization + exact "
            "wrapped point-phase restoration"
        ),
        "not_bitwise_pystamps_stage6": True,
        "points": int(npoint),
        "acquisitions": int(ndate),
        "ifgs": int(nifg),
        "coarse_grid_shape": [int(x) for x in grid["shape"]],
        "coarse_grid_size_m": float(grid_size_m),
        "coarse_nodes": int(nnode),
        "coarse_node_fraction_valid": float(np.mean(node_valid)),
        "strict_points": int(strict_ids.size),
        "strict_point_fraction": strict_fraction,
        "temporal_reference_index_0based": int(reference_idx),
        "temporal_reference_date": str(stack.dates[reference_idx]),
        "temporal_design_rank": int(sync["rank"]),
        "selected_temporal_edges": int(np.count_nonzero(sync["edge_keep"])),
        "temporal_edges_total": int(nifg),
        "node_temporal_mismatch_fraction_p50_p90_p95_p99_max": [float(x) for x in mismatch_q],
        "edge_bad_fraction_p50_p90_p95_p99_max": [float(x) for x in bad_q],
        "point_wrap_back_max_error_rad": float(max_wrap_parity),
        "snaphu_wrap_back_max_error_rad": float(max(wrap_errors)) if wrap_errors else None,
        "runtime_seconds": {
            "grid": float(grid_seconds),
            "snaphu": float(snaphu_seconds),
            "temporal_sync": float(sync_seconds),
            "point_restore": float(restore_seconds),
            "total": float(time.perf_counter() - t0),
        },
        "outputs": {
            "full_acquisition_phase": str(full_path),
            "strict_mask": str(strict_path),
            "strict_point_ids": str(final_dir / "strict_point_ids.npy"),
            "coarse_integer_cycles": str(sync["cycles_path"]),
        },
        "scientific_note": (
            "The coarse grid determines spatial integer ambiguity at a metric support scale. "
            "Temporal redundancy is enforced through the full selected acquisition graph. "
            "The final point phase remains congruent, modulo floating precision, with "
            "PointPhaseStack."
        ),
    }
    atomic_json(outdir / "stamps3d_unwrap_manifest.json", manifest)
    atomic_json(final_dir / "final_unwrap_manifest.json", manifest)

    print()
    print("=" * 104)
    print("STAMPS3D/SNAPHU FINAL QA")
    print("=" * 104)
    print("strict coverage             :", f"{100*strict_fraction:.3f}%")
    print("point wrap-back max [rad]   :", f"{max_wrap_parity:.3e}")
    print("node mismatch p50/90/95/99/max:", mismatch_q)
    print("edge bad p50/90/95/99/max  :", bad_q)
    print("total seconds               :", f"{manifest['runtime_seconds']['total']:.1f}")
    print("manifest                    :", outdir / "stamps3d_unwrap_manifest.json")
    print("=" * 104)

    if max_wrap_parity > 1.0e-4:
        raise RuntimeError(
            f"Point wrap-back error {max_wrap_parity} rad exceeds 1e-4"
        )

    return manifest


def is_stamps3d_backend(cfg) -> bool:
    return str(
        cfg_get(cfg, "unwrap.backend", "legacy_hierarchical")
    ).strip().lower() in {
        "stamps3d_snaphu",
        "stamps_grid_snaphu",
    }


def run_stamps3d_residual_ramp(*, cfg, config_path: Path, paths, stack):
    """Residual-ramp stage for the acquisition-phase stamps3d backend.

    It fits the same robust IFG-domain degree-1 ramp model as the legacy stage,
    but derives each IFG on demand from the compact Npoint x Ndate acquisition
    phase product. No Npoint x Nifg unwrapped archive is written.
    """
    import csv

    from pypsds.corrections.residual_ramp import (
        cell_balanced_weights,
        huber_plane,
        local_xy_m,
        network_project_ifg_slopes,
    )

    root = Path(paths.output_dir) / "processing"
    final_dir = root / "final_unwrap"
    geom_dir = root / "point_geometry"
    pps_dir = root / "point_phase_stack"
    network_dir = root / "network"
    stamps_dir = root / "stamps3d_unwrap"
    outdir = root / "residual_ramp"
    outdir.mkdir(parents=True, exist_ok=True)

    strict_ids = np.asarray(
        np.load(final_dir / "strict_point_ids.npy"), dtype=np.int64
    )
    strict_mask = np.asarray(
        np.load(final_dir / "strict_unwrap_valid_mask.npy"), dtype=bool
    )
    if not np.array_equal(strict_ids, np.flatnonzero(strict_mask)):
        raise RuntimeError("stamps3d strict-point contract mismatch")

    full_phase = np.load(
        stamps_dir / "acquisition_phase_unwrapped_rad.npy", mmap_mode="r"
    )
    point_type = np.load(pps_dir / "point_type.npy", mmap_mode="r")
    strict_type = np.asarray(point_type[strict_ids], dtype=np.uint8)
    ps_strict = np.flatnonzero(strict_type == TYPE_PS).astype(np.int64)
    anchor_ids = strict_ids[ps_strict]

    lon = np.asarray(
        np.load(geom_dir / "longitude_deg.npy", mmap_mode="r"), dtype=np.float64
    )
    lat = np.asarray(
        np.load(geom_dir / "latitude_deg.npy", mmap_mode="r"), dtype=np.float64
    )
    if lon.size != strict_ids.size or lat.size != strict_ids.size:
        raise RuntimeError("stamps3d strict geometry size mismatch")

    edges = load_itab(network_dir / "network.itab", full_phase.shape[1])
    ndate = full_phase.shape[1]
    nifg = len(edges)
    tref = int(cfg_get(cfg, "phase_linking.temporal_reference_index", 0))

    mode = str(
        cfg_get(cfg, "corrections.residual_ramp.mode", "disabled")
    ).strip().lower()
    coeff_direct_path = outdir / "ifg_ramp_direct_coefficients_rad_per_km.npy"
    coeff_projected_path = outdir / "ifg_ramp_projected_coefficients_rad_per_km.npy"
    acq_coeff_path = outdir / "acquisition_ramp_coefficients_rad_per_km.npy"
    anchor_idx_path = outdir / "anchor_strict_indices.npy"
    anchor_pid_path = outdir / "anchor_point_ids.npy"
    anchor_weight_path = outdir / "anchor_base_weight.npy"
    stats_path = outdir / "residual_ramp_ifg_stats.csv"
    manifest_path = outdir / "residual_ramp_manifest.json"

    print("=" * 104)
    print("STAMPS3D DIRECT IFG-DOMAIN RESIDUAL RAMP")
    print("=" * 104)
    print("strict points / PS         :", f"{strict_ids.size:,}", "/", f"{ps_strict.size:,}")
    print("acquisitions / IFGs        :", ndate, "/", nifg)
    print("mode                       :", mode)

    if mode in ("disabled", "none", "off", "false", "0"):
        np.save(coeff_direct_path, np.zeros((nifg, 3), dtype=np.float64))
        np.save(coeff_projected_path, np.zeros((nifg, 3), dtype=np.float64))
        np.save(acq_coeff_path, np.zeros((ndate, 2), dtype=np.float64))
        np.save(anchor_idx_path, np.empty(0, dtype=np.int64))
        np.save(anchor_pid_path, np.empty(0, dtype=np.int64))
        np.save(anchor_weight_path, np.empty(0, dtype=np.float64))
        with stats_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "ifg_index_0based", "pair_id", "date1", "date2",
                "direct_ax_rad_per_km", "direct_by_rad_per_km",
                "direct_intercept_rad", "projected_ax_rad_per_km",
                "projected_by_rad_per_km",
            ])
        atomic_json(
            manifest_path,
            {
                "status": "PASS_DISABLED",
                "mode": "disabled",
                "backend": "stamps3d_snaphu",
                "points": int(strict_ids.size),
                "strict_ps": int(ps_strict.size),
                "ifgs": int(nifg),
                "acquisitions": int(ndate),
                "materialized_ifg_stack": False,
            },
        )
        print("RESIDUAL RAMP STATUS: PASS_DISABLED")
        return

    if mode not in ("robust_huber_balanced", "ifg_network_huber"):
        raise RuntimeError(f"Unsupported residual_ramp mode: {mode}")

    min_anchors = int(cfg_get(cfg, "corrections.residual_ramp.min_anchors", 30))
    if ps_strict.size < min_anchors:
        raise RuntimeError(f"strict PS={ps_strict.size} < min_anchors={min_anchors}")

    cell_size_m = float(cfg_get(cfg, "corrections.residual_ramp.cell_size_m", 2000.0))
    min_cells = int(cfg_get(cfg, "corrections.residual_ramp.min_occupied_cells", 6))
    delta = float(cfg_get(cfg, "corrections.residual_ramp.huber_delta", 1.345))
    iterations = int(cfg_get(cfg, "corrections.residual_ramp.huber_iterations", 5))

    coords_m, lon0, lat0 = local_xy_m(lon, lat)
    x_km = coords_m[:, 0] / 1000.0
    y_km = coords_m[:, 1] / 1000.0
    base_weight, cell_index, cell_meta = cell_balanced_weights(
        coords_m[ps_strict], cell_size_m=cell_size_m
    )
    if int(cell_meta["occupied_cells"]) < min_cells:
        raise RuntimeError("Too few occupied ramp anchor cells")

    Xa = np.column_stack((
        x_km[ps_strict],
        y_km[ps_strict],
        np.ones(ps_strict.size, dtype=np.float64),
    ))

    direct = np.full((nifg, 3), np.nan, dtype=np.float64)
    scale_all = np.full(nifg, np.nan, dtype=np.float64)
    used_all = np.zeros(nifg, dtype=np.int32)

    t0 = time.perf_counter()
    for e, (i, j) in enumerate(edges):
        z_anchor = (
            np.asarray(full_phase[anchor_ids, j], dtype=np.float64)
            - np.asarray(full_phase[anchor_ids, i], dtype=np.float64)
        )
        beta, scale, used = huber_plane(
            Xa,
            z_anchor,
            base_weight,
            iterations=iterations,
            delta=delta,
        )
        if not np.all(np.isfinite(beta)):
            raise RuntimeError(f"IFG ramp fit failed at edge {e}")
        direct[e] = beta
        scale_all[e] = scale
        used_all[e] = used
        if e == 0 or (e + 1) % 10 == 0 or e + 1 == nifg:
            print(
                f"[RAMP] {e+1:3d}/{nifg} "
                f"|g|={math.hypot(beta[0], beta[1]):.5f} rad/km "
                f"scale={scale:.5f}",
                flush=True,
            )

    projected_xy, acquisition_xy, projection = network_project_ifg_slopes(
        edges,
        ndate,
        direct[:, :2],
        reference_idx=tref,
    )
    projected = direct.copy()
    projected[:, :2] = projected_xy

    np.save(coeff_direct_path, direct)
    np.save(coeff_projected_path, projected)
    np.save(acq_coeff_path, acquisition_xy)
    np.save(anchor_idx_path, ps_strict.astype(np.int64))
    np.save(anchor_pid_path, anchor_ids.astype(np.int64))
    np.save(anchor_weight_path, base_weight.astype(np.float64))
    np.save(outdir / "anchor_cell_index.npy", cell_index.astype(np.int32))

    rows_csv = []
    direct_rms = np.empty(nifg, dtype=np.float64)
    projected_rms = np.empty(nifg, dtype=np.float64)
    diff_ratio = np.empty(nifg, dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        dax, dby = direct[e, :2]
        pax, pby = projected[e, :2]
        dramp = dax * x_km + dby * y_km
        pramp = pax * x_km + pby * y_km
        diff = dramp - pramp
        drms = float(np.sqrt(np.mean(dramp * dramp)))
        prms = float(np.sqrt(np.mean(pramp * pramp)))
        xrms = float(np.sqrt(np.mean(diff * diff)))
        direct_rms[e] = drms
        projected_rms[e] = prms
        diff_ratio[e] = xrms / max(drms, 1.0e-12)
        rows_csv.append({
            "ifg_index_0based": e,
            "pair_id": e + 1,
            "date1": str(stack.dates[i]),
            "date2": str(stack.dates[j]),
            "direct_ax_rad_per_km": float(dax),
            "direct_by_rad_per_km": float(dby),
            "direct_intercept_rad": float(direct[e, 2]),
            "projected_ax_rad_per_km": float(pax),
            "projected_by_rad_per_km": float(pby),
            "huber_scale_rad": float(scale_all[e]),
            "huber_iterations": int(used_all[e]),
            "direct_slope_rms_rad": drms,
            "projected_slope_rms_rad": prms,
            "projection_diff_to_direct_ratio": float(diff_ratio[e]),
        })

    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    combined_corr = float(
        np.corrcoef(direct[:, :2].reshape(-1), projected_xy.reshape(-1))[0, 1]
    )
    ratio_q = np.percentile(diff_ratio, [50, 95, 99]).tolist()
    recommendation = (
        "PASS" if combined_corr >= 0.90 and ratio_q[1] <= 0.40 else "REVIEW"
    )

    atomic_json(
        manifest_path,
        {
            "status": (
                "PASS_IFG_NETWORK_PROJECTED_HUBER"
                if recommendation == "PASS"
                else "REVIEW_IFG_NETWORK_PROJECTION"
            ),
            "backend": "stamps3d_snaphu",
            "mode": mode,
            "domain": "ifg",
            "points": int(strict_ids.size),
            "strict_ps": int(ps_strict.size),
            "anchors": int(ps_strict.size),
            "occupied_cells": int(cell_meta["occupied_cells"]),
            "cell_size_m": cell_size_m,
            "huber_delta": delta,
            "huber_iterations": iterations,
            "acquisitions": int(ndate),
            "ifgs": int(nifg),
            "temporal_reference_index_0based": int(tref),
            "network_projection": {
                **projection,
                "combined_xy_correlation": combined_corr,
                "spatial_diff_to_direct_ratio_p50_p95_p99": ratio_q,
                "recommendation": recommendation,
            },
            "materialized_ifg_stack": False,
            "source": str(stamps_dir / "acquisition_phase_unwrapped_rad.npy"),
            "coordinate_origin_lon_deg": float(lon0),
            "coordinate_origin_lat_deg": float(lat0),
            "elapsed_seconds": float(time.perf_counter() - t0),
        },
    )
    print("RESIDUAL RAMP STATUS:", recommendation)


def run_stamps3d_timeseries_inversion(
    *, cfg, config_path: Path, paths, stack, batch_size: int
):
    """Direct acquisition-phase handoff for the stamps3d backend.

    The legacy solver rebuilds acquisition phase from Npoint x Nifg products.
    stamps3d already solved the acquisition integer gauge, so this path streams
    the compact Npoint x Ndate phase once and applies the acquisition-domain
    residual-ramp correction directly.
    """
    root = Path(paths.output_dir) / "processing"
    pps_dir = root / "point_phase_stack"
    final_dir = root / "final_unwrap"
    stamps_dir = root / "stamps3d_unwrap"
    ramp_dir = root / "residual_ramp"
    geom_dir = root / "point_geometry"
    outdir = root / "network_inversion"
    outdir.mkdir(parents=True, exist_ok=True)

    phase_pl = np.load(pps_dir / "phase_rad.npy", mmap_mode="r")
    phase_unw = np.load(
        stamps_dir / "acquisition_phase_unwrapped_rad.npy", mmap_mode="r"
    )
    strict_ids = np.asarray(
        np.load(final_dir / "strict_point_ids.npy"), dtype=np.int64
    )
    npoint, ndate = phase_unw.shape
    if phase_pl.shape != phase_unw.shape:
        raise RuntimeError("stamps3d wrapped/unwrapped acquisition shape mismatch")
    if len(stack.dates) != ndate:
        raise RuntimeError("stamps3d acquisition count mismatch")

    ramp_mode = str(
        cfg_get(cfg, "corrections.residual_ramp.mode", "disabled")
    ).strip().lower()
    ramp_enabled = ramp_mode not in ("disabled", "none", "off", "false", "0")

    if ramp_enabled:
        coeff = np.asarray(
            np.load(ramp_dir / "acquisition_ramp_coefficients_rad_per_km.npy"),
            dtype=np.float64,
        )
        lon = np.asarray(
            np.load(geom_dir / "longitude_deg.npy", mmap_mode="r"), dtype=np.float64
        )
        lat = np.asarray(
            np.load(geom_dir / "latitude_deg.npy", mmap_mode="r"), dtype=np.float64
        )
        from pypsds.corrections.residual_ramp import local_xy_m
        coords_m, _, _ = local_xy_m(lon, lat)
        x_km = coords_m[:, 0] / 1000.0
        y_km = coords_m[:, 1] / 1000.0
        if coeff.shape != (ndate, 2):
            raise RuntimeError("stamps3d ramp coefficient shape mismatch")
        if x_km.size != strict_ids.size:
            raise RuntimeError("stamps3d strict geometry mismatch")
    else:
        coeff = np.zeros((ndate, 2), dtype=np.float64)
        x_km = np.zeros(strict_ids.size, dtype=np.float64)
        y_km = np.zeros(strict_ids.size, dtype=np.float64)

    reference_idx = int(cfg_get(cfg, "phase_linking.temporal_reference_index", 0))
    if not (0 <= reference_idx < ndate):
        raise RuntimeError("Invalid temporal reference index")

    out_path = outdir / "acquisition_phase_l2_candidate_rad.npy"
    out = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(strict_ids.size, ndate),
    )
    parity_by_point = np.lib.format.open_memmap(
        outdir / "stamps3d_wrap_parity_max_abs_rad.npy",
        mode="w+",
        dtype=np.float32,
        shape=(strict_ids.size,),
    )

    effective_batch = int(
        cfg_get(cfg, "unwrap.stamps3d_snaphu.point_batch", max(batch_size, 131072))
    )
    global_max = 0.0
    global_ss = 0.0
    global_n = 0
    t0 = time.perf_counter()

    for b0 in range(0, strict_ids.size, effective_batch):
        b1 = min(strict_ids.size, b0 + effective_batch)
        ids = strict_ids[b0:b1]
        theta = np.asarray(phase_unw[ids, :], dtype=np.float64)
        expected_wrapped = np.asarray(phase_pl[ids, :], dtype=np.float64)

        if ramp_enabled:
            ramp = (
                x_km[b0:b1, None] * coeff[None, :, 0]
                + y_km[b0:b1, None] * coeff[None, :, 1]
            )
            theta -= ramp
            expected_wrapped -= ramp

        theta -= theta[:, reference_idx:reference_idx + 1]
        expected_wrapped -= expected_wrapped[:, reference_idx:reference_idx + 1]

        parity = wrap_phase(theta - expected_wrapped)
        pmax = np.max(np.abs(parity), axis=1)
        parity_by_point[b0:b1] = pmax.astype(np.float32)
        global_max = max(global_max, float(pmax.max()))
        global_ss += float(np.sum(parity * parity))
        global_n += int(parity.size)
        out[b0:b1, :] = theta.astype(np.float32)

        if b1 == strict_ids.size or b1 % 1_000_000 < effective_batch:
            print(
                f"[DIRECT ACQ] {b1:,}/{strict_ids.size:,} "
                f"({100*b1/max(1, strict_ids.size):.2f}%)",
                flush=True,
            )

    out.flush()
    parity_by_point.flush()
    np.save(outdir / "strict_point_ids.npy", strict_ids.astype(np.int32))

    rms = math.sqrt(global_ss / max(1, global_n))
    manifest = {
        "status": "PASS" if global_max <= 1.0e-4 else "REVIEW",
        "backend": "stamps3d_snaphu",
        "method": "direct_acquisition_phase_handoff",
        "points": int(strict_ids.size),
        "acquisitions": int(ndate),
        "reference_index_0based": int(reference_idx),
        "residual_ramp_enabled": bool(ramp_enabled),
        "wrapped_parity_rms_rad": float(rms),
        "wrapped_parity_max_rad": float(global_max),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "scientific_note": (
            "No second IFG-to-acquisition inversion is performed because the "
            "stamps3d backend already solves acquisition integer cycles using "
            "the redundant temporal network."
        ),
    }
    atomic_json(outdir / "stamps3d_direct_inversion_manifest.json", manifest)

    print("=" * 96)
    print("STAMPS3D DIRECT ACQUISITION HANDOFF")
    print("=" * 96)
    print("strict points              :", f"{strict_ids.size:,}")
    print("wrapped parity RMS/max     :", f"{rms:.3e}", "/", f"{global_max:.3e}")
    print("elapsed seconds            :", f"{manifest['elapsed_seconds']:.1f}")
    print("=" * 96)

    if global_max > 1.0e-4:
        raise RuntimeError(
            f"stamps3d direct acquisition parity error {global_max} > 1e-4 rad"
        )
