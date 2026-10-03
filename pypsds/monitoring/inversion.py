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



# PYPSDS_STAMPS3D_DIRECT_MONITORING_V1
def _upgrade_stamps3d_direct_monitoring(
    *,
    cfg,
    paths,
    stack,
):
    """
    Monitoring/uncertainty contract for the stamps3d direct-acquisition
    backend.

    The stamps3d backend has already solved acquisition integer cycles
    from the redundant temporal network.  Therefore it must NOT rebuild
    acquisition phase from a materialized Npoint x Nifg archive.

    Network covariance here represents the conservative numerical handoff
    floor only.  It is not a complete physical deformation uncertainty.
    """

    root = (
        Path(paths.output_dir)
        /
        "processing"
    )

    inv = (
        root
        /
        "network_inversion"
    )

    net = (
        root
        /
        "network"
    )

    final = (
        root
        /
        "final_unwrap"
    )

    phase_path = (
        inv
        /
        "acquisition_phase_l2_candidate_rad.npy"
    )

    strict_path = (
        inv
        /
        "strict_point_ids.npy"
    )

    direct_manifest_path = (
        inv
        /
        "stamps3d_direct_inversion_manifest.json"
    )

    parity_path = (
        inv
        /
        "stamps3d_wrap_parity_max_abs_rad.npy"
    )

    for path in (
        phase_path,
        strict_path,
        direct_manifest_path,
        parity_path,
        net / "network.itab",
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    direct = json.loads(
        direct_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if direct.get("status") != "PASS":
        raise RuntimeError(
            "stamps3d direct acquisition handoff "
            f"is not PASS: {direct.get('status')}"
        )

    phase = np.load(
        phase_path,
        mmap_mode="r",
    )

    strict_ids = np.asarray(
        np.load(
            strict_path
        ),
        dtype=np.int32,
    )

    strict_mask = np.asarray(
        np.load(
            final
            /
            "strict_unwrap_valid_mask.npy"
        ),
        dtype=bool,
    )

    expected_ids = np.flatnonzero(
        strict_mask
    ).astype(
        np.int32
    )

    if not np.array_equal(
        strict_ids,
        expected_ids,
    ):
        raise RuntimeError(
            "stamps3d direct monitoring strict-point "
            "contract mismatch"
        )

    if phase.ndim != 2:
        raise RuntimeError(
            "direct acquisition phase must be 2-D"
        )

    nstrict, ndate = phase.shape

    if strict_ids.size != nstrict:
        raise RuntimeError(
            "direct phase / strict-point count mismatch"
        )

    if len(stack.dates) != ndate:
        raise RuntimeError(
            "direct phase acquisition count mismatch"
        )

    edges = load_itab(
        net
        /
        "network.itab",
        ndate,
    )

    nifg = len(
        edges
    )

    reference_idx = int(
        cfg_get(
            cfg,
            "phase_linking.temporal_reference_index",
            0,
        )
    )

    if not (
        0
        <=
        reference_idx
        <
        ndate
    ):
        raise RuntimeError(
            "invalid temporal reference index"
        )

    A = build_design_matrix(
        edges,
        ndate,
        reference_idx,
    )

    rank = int(
        np.linalg.matrix_rank(
            A
        )
    )

    if rank != ndate - 1:
        raise RuntimeError(
            "stamps3d monitoring temporal "
            f"design rank={rank}/{ndate - 1}"
        )

    parity_rms = float(
        direct[
            "wrapped_parity_rms_rad"
        ]
    )

    parity_max = float(
        direct[
            "wrapped_parity_max_rad"
        ]
    )

    if (
        not np.isfinite(parity_rms)
        or
        not np.isfinite(parity_max)
    ):
        raise RuntimeError(
            "non-finite stamps3d wrap parity"
        )

    if parity_max > 1.0e-4:
        raise RuntimeError(
            "stamps3d wrap parity exceeds "
            f"production tolerance: {parity_max}"
        )

    parity_by_point = np.load(
        parity_path,
        mmap_mode="r",
    )

    if parity_by_point.shape != (
        nstrict,
    ):
        raise RuntimeError(
            "stamps3d parity-vector shape mismatch"
        )

    min_sigma = float(
        cfg_get(
            cfg,
            "timeseries.inversion.min_auto_sigma_rad",
            1.0e-4,
        )
    )

    if min_sigma <= 0:
        raise ValueError(
            "timeseries.inversion.min_auto_sigma_rad "
            "must be > 0"
        )

    # --------------------------------------------------------
    # Direct backend uncertainty contract
    #
    # We do NOT invent 236 full-resolution IFG residual maps.
    #
    # The acquisition solution already satisfies the wrapped
    # acquisition contract to parity_rms.  Use the existing
    # production numerical floor as a conservative lower bound
    # for the network covariance.
    # --------------------------------------------------------

    sigma_floor = max(
        min_sigma,
        parity_rms,
    )

    sigma = np.full(
        nifg,
        sigma_floor,
        dtype=np.float64,
    )

    weights = np.ones(
        nifg,
        dtype=np.float64,
    )

    normal = (
        A.T
        @
        A
    )

    cov_sub = (
        sigma_floor
        *
        sigma_floor
        *
        np.linalg.inv(
            normal
        )
    )

    cov_full = np.zeros(
        (
            ndate,
            ndate,
        ),
        dtype=np.float64,
    )

    nonref = [
        i
        for i in range(ndate)
        if i != reference_idx
    ]

    cov_full[
        np.ix_(
            nonref,
            nonref,
        )
    ] = cov_sub

    se_full = np.sqrt(
        np.maximum(
            np.diag(
                cov_full
            ),
            0.0,
        )
    )

    np.save(
        inv
        /
        "ifg_residual_sigma_rad.npy",
        sigma.astype(
            np.float32
        ),
    )

    np.save(
        inv
        /
        "ifg_weights.npy",
        weights.astype(
            np.float32
        ),
    )

    np.save(
        inv
        /
        "acquisition_phase_standard_error_rad.npy",
        se_full.astype(
            np.float32
        ),
    )

    np.save(
        inv
        /
        "acquisition_phase_covariance_rad2.npy",
        cov_full,
    )

    requested = str(
        cfg_get(
            cfg,
            "timeseries.inversion.method",
            "ordinary_l2",
        )
    ).strip().lower()

    manifest = {
        "status":
            "PASS_MONITORING_STAMPS3D_DIRECT",

        "version":
            "1.3.5",

        "backend":
            "stamps3d_snaphu",

        "requested_method":
            requested,

        "effective_method":
            "direct_acquisition_phase_handoff",

        "ordinary_solution_preserved":
            True,

        "second_ifg_inversion_performed":
            False,

        "materialized_ifg_stack":
            False,

        "points":
            int(nstrict),

        "acquisitions":
            int(ndate),

        "ifgs":
            int(nifg),

        "reference_index_0based":
            int(reference_idx),

        "temporal_design_rank":
            rank,

        "wrapped_parity_rms_rad":
            parity_rms,

        "wrapped_parity_max_rad":
            parity_max,

        "min_auto_sigma_rad":
            min_sigma,

        "numerical_sigma_floor_rad":
            sigma_floor,

        "floor_dominated":
            bool(
                parity_rms
                <=
                min_sigma
            ),

        "relative_weights": {
            "min":
                1.0,

            "median":
                1.0,

            "max":
                1.0,

            "interpretation":
                (
                    "No second weighted IFG inversion is "
                    "performed for the direct acquisition backend."
                ),
        },

        "max_abs_ordinary_vs_effective_phase_rad":
            0.0,

        "scientific_contract": (
            "The stamps3d backend already solved acquisition "
            "integer cycles using the redundant temporal network. "
            "Monitoring therefore preserves the direct acquisition "
            "phase exactly and does not reconstruct it from an "
            "Npoint x Nifg archive."
        ),

        "formal_uncertainty_note": (
            "Network covariance is a conservative numerical-handoff "
            "floor derived from the validated wrapped-parity error "
            "and min_auto_sigma_rad. It does not represent complete "
            "physical uncertainty from atmosphere, orbit, geocoding, "
            "deformation model, or unwrap ambiguity."
        ),
    }

    manifest_path = (
        inv
        /
        "monitoring_inversion_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print(
        "MONITORING NETWORK INVERSION "
        "- STAMPS3D DIRECT"
    )
    print("=" * 96)

    print(
        "points / acquisitions / IFGs :",
        f"{nstrict:,}",
        "/",
        ndate,
        "/",
        nifg,
    )

    print(
        "wrapped parity RMS/max       :",
        f"{parity_rms:.3e}",
        "/",
        f"{parity_max:.3e}",
        "rad",
    )

    print(
        "numerical sigma floor        :",
        f"{sigma_floor:.3e}",
        "rad",
    )

    print(
        "second IFG inversion         : NO"
    )

    print(
        "materialized IFG stack       : NO"
    )

    print(
        "acquisition phase modified   : NO"
    )

    print(
        "manifest                     :",
        manifest_path,
    )

    print("=" * 96)

    return manifest

def upgrade_network_inversion(config_path, batch_size: int = 12000):
    """
    Conservative feasible WLS upgrade of the validated ordinary-L2 solution.

    If strict-network residuals are below the numerical floor, the existing
    ordinary-L2 acquisition phase is preserved bit-for-bit.
    """
    cfg, _, paths, stack, _ = open_from_config(config_path)

    # PYPSDS_STAMPS3D_DIRECT_MONITORING_V1
    from pypsds.stamps3d_backend import is_stamps3d_backend

    if is_stamps3d_backend(
        cfg
    ):
        return _upgrade_stamps3d_direct_monitoring(
            cfg=cfg,
            paths=paths,
            stack=stack,
        )

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
