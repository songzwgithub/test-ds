from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pypsds.config import cfg_get
from pypsds.context import open_from_config


def rank01(values):
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    good = np.flatnonzero(np.isfinite(x))
    if good.size == 0:
        return out
    if good.size == 1:
        out[good] = 1.0
        return out
    order = good[np.argsort(x[good], kind="mergesort")]
    out[order] = np.linspace(0.0, 1.0, order.size)
    return out


def choose_reference_region(
    xy_m,
    rate_abs,
    residual,
    *,
    radius_m,
    cell_size_m,
    min_points,
    rate_weight=0.60,
    residual_weight=0.30,
    density_weight=0.10,
):
    xy = np.asarray(xy_m, dtype=np.float64)
    rate = np.asarray(rate_abs, dtype=np.float64)
    rms = np.asarray(residual, dtype=np.float64)

    valid = (
        np.all(np.isfinite(xy), axis=1)
        & np.isfinite(rate)
        & np.isfinite(rms)
    )
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size < min_points:
        raise RuntimeError("Too few finite points for automatic reference")

    vxy = xy[valid_ids]
    tree = cKDTree(vxy)
    x0 = float(np.min(vxy[:, 0]))
    y0 = float(np.min(vxy[:, 1]))
    cx = np.floor((vxy[:, 0] - x0) / cell_size_m).astype(np.int64)
    cy = np.floor((vxy[:, 1] - y0) / cell_size_m).astype(np.int64)
    cells, counts = np.unique(
        np.column_stack((cx, cy)), axis=0, return_counts=True
    )

    candidates = []
    seen = set()
    seed_min = max(3, min_points // 10)

    for (cell_x, cell_y), count in zip(cells, counts):
        if int(count) < seed_min:
            continue
        local = np.flatnonzero((cx == cell_x) & (cy == cell_y))
        centre = np.median(vxy[local], axis=0)
        region_local = np.asarray(
            tree.query_ball_point(centre, r=radius_m),
            dtype=np.int64,
        )
        if region_local.size < min_points:
            continue
        region = np.sort(valid_ids[region_local])
        signature = tuple(region.tolist())
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(
            {
                "indices": region,
                "n_points": int(region.size),
                "median_abs_rate": float(np.median(rate[region])),
                "median_residual": float(np.median(rms[region])),
            }
        )

    if not candidates:
        raise RuntimeError(
            "No automatic reference region satisfies radius/min_points constraints"
        )

    q_rate = rank01(-np.asarray([x["median_abs_rate"] for x in candidates]))
    q_rms = rank01(-np.asarray([x["median_residual"] for x in candidates]))
    q_den = rank01(
        np.log1p(np.asarray([x["n_points"] for x in candidates], dtype=float))
    )

    weights = np.asarray(
        [rate_weight, residual_weight, density_weight],
        dtype=np.float64,
    )
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("automatic-reference weights must be non-negative")
    weights /= weights.sum()
    score = weights[0] * q_rate + weights[1] * q_rms + weights[2] * q_den

    best = int(np.nanargmax(score))
    for i, s in enumerate(score):
        candidates[i]["score"] = float(s)

    return candidates[best], sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True,
    )


def _local_xy(lon, lat):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    R = 6371008.8
    x = np.deg2rad(lon - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return np.column_stack((x, y))


def select_auto_reference(config_path):
    """
    Select a high-quality *relative* reference region.

    The acquisition phase is first centered by a robust scene epoch median,
    then regions are ranked by low relative rate, low residual, and density.
    This cannot prove absolute physical stability.
    """
    cfg, _, paths, stack, _ = open_from_config(config_path)
    proc = Path(paths.output_dir) / "processing"
    inv = proc / "network_inversion"
    geom = proc / "point_geometry"
    outdir = proc / "referenced_timeseries"
    outdir.mkdir(parents=True, exist_ok=True)

    strict_ids = np.asarray(
        np.load(inv / "strict_point_ids.npy"),
        dtype=np.int32,
    )
    phase = np.load(
        inv
        / "acquisition_phase_l2_candidate_rad.npy",
        mmap_mode="r",
    )
    lon = np.asarray(
        np.load(geom / "longitude_deg.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    lat = np.asarray(
        np.load(geom / "latitude_deg.npy", mmap_mode="r"),
        dtype=np.float64,
    )

    n, ndate = phase.shape
    if strict_ids.size != n or lon.size != n or lat.size != n:
        raise RuntimeError("automatic-reference geometry/phase contract mismatch")

    dates = [datetime.strptime(str(x), "%Y%m%d") for x in stack.dates]
    years = np.asarray(
        [(x - dates[0]).days / 365.25 for x in dates],
        dtype=np.float64,
    )
    tc = years - years.mean()
    denom = float(tc @ tc)
    if denom <= 0:
        raise RuntimeError("invalid time axis")

    sample_n = min(
        n,
        int(cfg_get(cfg, "reference.auto.scene_median_sample", 100000)),
    )
    sample_idx = np.linspace(0, n - 1, sample_n, dtype=np.int64)
    scene_epoch_median = np.median(
        np.asarray(phase[sample_idx], dtype=np.float64),
        axis=0,
    )

    rate = np.empty(n, dtype=np.float64)
    temporal_rms = np.empty(n, dtype=np.float64)

    for b0 in range(0, n, 50000):
        b1 = min(b0 + 50000, n)
        Y = (
            np.asarray(phase[b0:b1], dtype=np.float64)
            - scene_epoch_median[None, :]
        )
        ym = np.mean(Y, axis=1)
        slope = ((Y - ym[:, None]) @ tc) / denom
        intercept = ym - slope * float(np.mean(years))
        fit = intercept[:, None] + slope[:, None] * years[None, :]
        rr = Y - fit
        rate[b0:b1] = np.abs(slope)
        temporal_rms[b0:b1] = np.sqrt(np.mean(rr * rr, axis=1))

    network_rms_path = inv / "l2_network_residual_rms_rad.npy"
    if network_rms_path.is_file():
        nr = np.asarray(np.load(network_rms_path, mmap_mode="r"), dtype=np.float64)
        quality_rms = (
            np.hypot(temporal_rms, nr)
            if nr.shape == temporal_rms.shape
            else temporal_rms
        )
    else:
        quality_rms = temporal_rms

    xy = _local_xy(lon, lat)
    radius = float(cfg_get(cfg, "reference.auto.radius_m", 500.0))
    cell = float(cfg_get(cfg, "reference.auto.cell_size_m", 500.0))
    min_points = int(
        cfg_get(
            cfg,
            "reference.auto.min_points",
            cfg_get(cfg, "reference.min_points", 100),
        )
    )

    best, candidates = choose_reference_region(
        xy,
        rate,
        quality_rms,
        radius_m=radius,
        cell_size_m=cell,
        min_points=min_points,
        rate_weight=float(cfg_get(cfg, "reference.auto.rate_weight", 0.60)),
        residual_weight=float(
            cfg_get(cfg, "reference.auto.residual_weight", 0.30)
        ),
        density_weight=float(
            cfg_get(cfg, "reference.auto.density_weight", 0.10)
        ),
    )

    idx = np.asarray(best["indices"], dtype=np.int64)
    selected = outdir / "auto_reference_point_ids.npy"
    np.save(selected, strict_ids[idx].astype(np.int32))

    report = {
        "status": "PASS_AUTO_REFERENCE",
        "version": "1.3.0",
        "method": "auto_stable_relative_region",
        "points": int(idx.size),
        "longitude_median": float(np.median(lon[idx])),
        "latitude_median": float(np.median(lat[idx])),
        "radius_m": radius,
        "cell_size_m": cell,
        "median_abs_relative_phase_rate_rad_per_year": best["median_abs_rate"],
        "median_quality_residual_rad": best["median_residual"],
        "score": best["score"],
        "candidate_count": len(candidates),
        "top_candidates": [
            {k: v for k, v in row.items() if k != "indices"}
            for row in candidates[:10]
        ],
        "note": (
            "Automatic selection establishes a high-quality relative InSAR "
            "datum; it does not prove zero physical deformation. Prefer an "
            "externally validated stable reference when available."
        ),
        "point_ids_file": str(selected),
    }
    (outdir / "reference_selection.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("AUTOMATIC STABLE REFERENCE")
    print("=" * 96)
    print("points       :", idx.size)
    print("median lon   :", np.median(lon[idx]))
    print("median lat   :", np.median(lat[idx]))
    print("relative rate:", best["median_abs_rate"], "rad/yr")
    print("quality RMS  :", best["median_residual"], "rad")
    print("score        :", best["score"])
    print("=" * 96)
    return selected
