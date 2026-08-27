from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from pypsds.config import cfg_get
from pypsds.context import open_from_config

TWOPI = 2.0 * np.pi


def load_itab(path: Path, ndate: int) -> list[tuple[int, int]]:
    edges = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        f = raw.split()
        if len(f) < 2:
            continue
        i = int(f[0]) - 1
        j = int(f[1]) - 1
        if not (0 <= i < ndate and 0 <= j < ndate):
            raise RuntimeError(f"Invalid ITAB row: {raw}")
        edges.append((i, j))
    return edges


def build_design_matrix(edges, ndate: int, reference_idx: int = 0):
    col = {}
    k = 0
    for t in range(ndate):
        if t == reference_idx:
            continue
        col[t] = k
        k += 1
    A = np.zeros((len(edges), ndate - 1), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        if i != reference_idx:
            A[e, col[i]] -= 1.0
        if j != reference_idx:
            A[e, col[j]] += 1.0
    return A


def weighted_operator(A, weights):
    A = np.asarray(A, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if A.ndim != 2 or w.shape != (A.shape[0],):
        raise ValueError("A/weights shape mismatch")
    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("weights must be finite and >0")
    normal = A.T @ (w[:, None] * A)
    rhs = A.T * w[None, :]
    try:
        return np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Weighted temporal design is singular") from exc


def _open_ifg_maps(root, dates, edges, npoint):
    d = root / "single_ifg_robust_solution"
    out = []
    for pair_id, (i, j) in enumerate(edges, start=1):
        tag = f"pair{pair_id:03d}_{dates[i]}_{dates[j]}"
        p = d / f"{tag}_unwrapped_phase_rad.npy"
        if not p.is_file():
            raise FileNotFoundError(p)
        a = np.load(p, mmap_mode="r")
        if a.size != npoint:
            raise RuntimeError(f"{p.name}: point count mismatch")
        out.append(a)
    return out


def _observations(maps, ids, gauge):
    Y = np.empty((ids.size, len(maps)), dtype=np.float64)
    for e, arr in enumerate(maps):
        Y[:, e] = np.asarray(arr[ids], dtype=np.float64) + TWOPI * gauge[e]
    return Y


def upgrade_network_inversion(config_path, batch_size: int = 12000):
    """
    Conservative feasible WLS upgrade of the validated ordinary-L2 solution.

    If strict-network residuals are below the numerical floor, the existing
    ordinary-L2 acquisition phase is preserved bit-for-bit.
    """
    cfg, _, paths, stack, _ = open_from_config(config_path)
    root = Path(paths.output_dir) / "processing"
    inv = root / "network_inversion"
    net = root / "network"
    final = root / "final_unwrap"
    pps = root / "point_phase_stack"

    phase_path = inv / "acquisition_phase_l2_candidate_rad.npy"
    strict_path = inv / "strict_point_ids.npy"
    gauge_path = final / "global_ifg_integer_delta.npy"
    for p in (phase_path, strict_path, gauge_path, net / "network.itab"):
        if not p.is_file():
            raise FileNotFoundError(p)

    phase_ols = np.load(phase_path, mmap_mode="r")
    strict_ids = np.asarray(np.load(strict_path), dtype=np.int32)
    gauge = np.asarray(np.load(gauge_path), dtype=np.int32)
    nstrict, ndate = phase_ols.shape
    npoint = int(np.load(pps / "phase_rad.npy", mmap_mode="r").shape[0])
    edges = load_itab(net / "network.itab", ndate)
    nifg = len(edges)
    if strict_ids.size != nstrict or gauge.shape != (nifg,):
        raise RuntimeError("network-inversion contract mismatch")

    A = build_design_matrix(edges, ndate)
    if np.linalg.matrix_rank(A) != ndate - 1:
        raise RuntimeError("Temporal design matrix is rank deficient")
    maps = _open_ifg_maps(root, stack.dates, edges, npoint)

    ss = np.zeros(nifg, dtype=np.float64)
    nn = np.zeros(nifg, dtype=np.int64)
    for b0 in range(0, nstrict, batch_size):
        b1 = min(b0 + batch_size, nstrict)
        ids = strict_ids[b0:b1]
        Y = _observations(maps, ids, gauge)
        theta = np.asarray(phase_ols[b0:b1, 1:], dtype=np.float64)
        residual = Y - theta @ A.T
        good = np.isfinite(residual)
        ss += np.sum(np.where(good, residual * residual, 0.0), axis=0)
        nn += np.sum(good, axis=0)

    if np.any(nn == 0):
        raise RuntimeError("At least one IFG has no residual observations")
    sigma = np.sqrt(ss / nn)
    if not np.all(np.isfinite(sigma)):
        raise RuntimeError("Non-finite IFG residual sigma")

    requested = str(
        cfg_get(cfg, "timeseries.inversion.method", "weighted_l2")
    ).strip().lower()
    if requested not in {"weighted_l2", "ordinary_l2"}:
        raise ValueError("inversion method must be weighted_l2 or ordinary_l2")

    min_sigma = float(
        cfg_get(cfg, "timeseries.inversion.min_auto_sigma_rad", 1.0e-4)
    )
    clip_min = float(cfg_get(cfg, "timeseries.inversion.weight_clip_min", 0.5))
    clip_max = float(cfg_get(cfg, "timeseries.inversion.weight_clip_max", 2.0))
    if min_sigma <= 0 or not (0 < clip_min <= 1 <= clip_max):
        raise ValueError("invalid inversion uncertainty/weight settings")

    median_sigma = float(np.median(sigma))
    floor_dominated = median_sigma <= min_sigma

    if requested == "weighted_l2" and not floor_dominated:
        sigma_w = np.maximum(sigma, min_sigma)
        weights = np.clip((median_sigma / sigma_w) ** 2, clip_min, clip_max)
        weights /= float(np.median(weights))
        effective = "weighted_l2"
    else:
        weights = np.ones(nifg, dtype=np.float64)
        effective = "ordinary_l2"

    # Formal absolute covariance with a conservative numerical floor.
    sigma_abs = np.maximum(sigma, min_sigma)
    w_abs = 1.0 / (sigma_abs * sigma_abs)
    cov_sub = np.linalg.inv(A.T @ (w_abs[:, None] * A))
    cov_full = np.zeros((ndate, ndate), dtype=np.float64)
    cov_full[1:, 1:] = cov_sub
    se_full = np.sqrt(np.maximum(np.diag(cov_full), 0.0))

    max_diff = 0.0
    if effective == "weighted_l2":
        P = weighted_operator(A, weights)
        tmp = inv / ".acquisition_phase_network.tmp.npy"
        if tmp.exists():
            tmp.unlink()
        out = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(nstrict, ndate)
        )
        for b0 in range(0, nstrict, batch_size):
            b1 = min(b0 + batch_size, nstrict)
            ids = strict_ids[b0:b1]
            Y = _observations(maps, ids, gauge)
            theta = Y @ P.T
            full = np.zeros((ids.size, ndate), dtype=np.float64)
            full[:, 1:] = theta
            old = np.asarray(phase_ols[b0:b1], dtype=np.float64)
            max_diff = max(max_diff, float(np.max(np.abs(full - old))))
            out[b0:b1] = full.astype(np.float32)
        out.flush()
        del out
        del phase_ols
        os.replace(tmp, phase_path)

    np.save(inv / "ifg_residual_sigma_rad.npy", sigma.astype(np.float32))
    np.save(inv / "ifg_weights.npy", weights.astype(np.float32))
    np.save(
        inv / "acquisition_phase_standard_error_rad.npy",
        se_full.astype(np.float32),
    )
    np.save(inv / "acquisition_phase_covariance_rad2.npy", cov_full)

    manifest = {
        "status": "PASS_MONITORING_NETWORK_INVERSION",
        "version": "1.3.0",
        "requested_method": requested,
        "effective_method": effective,
        "ordinary_solution_preserved": effective == "ordinary_l2",
        "floor_dominated": bool(floor_dominated),
        "min_auto_sigma_rad": min_sigma,
        "ifg_residual_sigma_rad": {
            "min": float(np.min(sigma)),
            "median": median_sigma,
            "max": float(np.max(sigma)),
        },
        "relative_weights": {
            "min": float(np.min(weights)),
            "median": float(np.median(weights)),
            "max": float(np.max(weights)),
            "clip_min": clip_min,
            "clip_max": clip_max,
        },
        "max_abs_ordinary_vs_effective_phase_rad": max_diff,
        "formal_uncertainty_note": (
            "Network-inversion uncertainty from a global diagonal IFG residual "
            "model; correlated/systematic atmosphere/orbit/model errors are "
            "not fully represented."
        ),
    }
    (inv / "monitoring_inversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 96)
    print("MONITORING NETWORK INVERSION")
    print("=" * 96)
    print("requested/effective :", requested, "/", effective)
    print("IFG sigma min/med/max:", np.min(sigma), median_sigma, np.max(sigma))
    print("weights min/med/max :", np.min(weights), np.median(weights), np.max(weights))
    print("OLS/WLS max diff    :", max_diff, "rad")
    print("=" * 96)
    return manifest


def _config_from_argv(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(args):
        if token == "--config" and i + 1 < len(args):
            return args[i + 1]
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    raise RuntimeError("--config not found in stage argv")


def upgrade_from_argv():
    upgrade_network_inversion(_config_from_argv())
