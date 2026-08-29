#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from pypsds.config import cfg_get
from pypsds.context import open_from_config
from pypsds.corrections.residual_ramp import (
    cell_balanced_weights,
    huber_plane,
    local_xy_m,
    network_project_ifg_slopes,
)

TYPE_PS = np.uint8(1)


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_itab(path: Path, ndate: int):
    edges = []
    with path.open() as f:
        for raw in f:
            x = raw.split()
            if len(x) < 2:
                continue
            i = int(x[0]) - 1
            j = int(x[1]) - 1
            if not (0 <= i < ndate and 0 <= j < ndate):
                raise RuntimeError(f"Invalid ITAB line: {raw}")
            edges.append((i, j))
    if not edges:
        raise RuntimeError(f"No temporal edges found in {path}")
    return edges


def pair_tag(pair_id: int, i: int, j: int, dates) -> str:
    return f"pair{pair_id:03d}_{dates[i]}_{dates[j]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg, config_path, paths, stack, _ = open_from_config(args.config)

    root = Path(paths.output_dir) / "processing"
    final_dir = root / "final_unwrap"
    geom_dir = root / "point_geometry"
    pps_dir = root / "point_phase_stack"
    network_dir = root / "network"
    unwrap_dir = root / "single_ifg_robust_solution"
    outdir = root / "residual_ramp"
    ifg_outdir = outdir / "ifgs"

    outdir.mkdir(parents=True, exist_ok=True)
    ifg_outdir.mkdir(parents=True, exist_ok=True)

    strict_ids = np.asarray(
        np.load(final_dir / "strict_point_ids.npy"),
        dtype=np.int64,
    )
    strict_mask = np.asarray(
        np.load(final_dir / "strict_unwrap_valid_mask.npy"),
        dtype=bool,
    )
    expected_ids = np.flatnonzero(strict_mask).astype(np.int64)

    if not np.array_equal(strict_ids, expected_ids):
        raise RuntimeError(
            "final_unwrap strict_point_ids.npy does not match "
            "strict_unwrap_valid_mask.npy"
        )

    lon = np.asarray(
        np.load(geom_dir / "longitude_deg.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    lat = np.asarray(
        np.load(geom_dir / "latitude_deg.npy", mmap_mode="r"),
        dtype=np.float64,
    )

    if lon.size != strict_ids.size or lat.size != strict_ids.size:
        raise RuntimeError("strict geometry / strict-point mismatch")

    point_type_full = np.asarray(
        np.load(pps_dir / "point_type.npy", mmap_mode="r"),
        dtype=np.uint8,
    )
    strict_type = point_type_full[strict_ids]
    ps_strict = np.flatnonzero(strict_type == TYPE_PS).astype(np.int64)

    mode = str(
        cfg_get(
            cfg,
            "corrections.residual_ramp.mode",
            "disabled",
        )
    ).strip().lower()

    domain = str(
        cfg_get(
            cfg,
            "corrections.residual_ramp.domain",
            "ifg",
        )
    ).strip().lower()

    spatial_balance = str(
        cfg_get(
            cfg,
            "corrections.residual_ramp.spatial_balance",
            "equal_cell_total_weight",
        )
    ).strip().lower()

    anchor_source = str(
        cfg_get(
            cfg,
            "corrections.residual_ramp.anchor_source",
            "ps",
        )
    ).strip().lower()

    if domain != "ifg":
        raise RuntimeError("Production residual_ramp requires domain=ifg")
    if spatial_balance != "equal_cell_total_weight":
        raise RuntimeError(
            "Production residual_ramp requires "
            "spatial_balance=equal_cell_total_weight"
        )
    if anchor_source != "ps":
        raise RuntimeError(
            "Production residual_ramp requires anchor_source=ps"
        )

    cell_size_m = float(
        cfg_get(
            cfg,
            "corrections.residual_ramp.cell_size_m",
            2000.0,
        )
    )
    min_anchors = int(
        cfg_get(
            cfg,
            "corrections.residual_ramp.min_anchors",
            30,
        )
    )
    min_cells = int(
        cfg_get(
            cfg,
            "corrections.residual_ramp.min_occupied_cells",
            6,
        )
    )
    delta = float(
        cfg_get(
            cfg,
            "corrections.residual_ramp.huber_delta",
            1.345,
        )
    )
    iterations = int(
        cfg_get(
            cfg,
            "corrections.residual_ramp.huber_iterations",
            5,
        )
    )
    tref = int(
        cfg_get(
            cfg,
            "phase_linking.temporal_reference_index",
            0,
        )
    )

    nstrict = strict_ids.size
    ndate = len(stack.dates)
    edges = load_itab(network_dir / "network.itab", ndate)
    nifg = len(edges)

    coeff_direct_path = (
        outdir / "ifg_ramp_direct_coefficients_rad_per_km.npy"
    )
    coeff_projected_path = (
        outdir / "ifg_ramp_projected_coefficients_rad_per_km.npy"
    )
    acq_coeff_path = (
        outdir / "acquisition_ramp_coefficients_rad_per_km.npy"
    )
    anchor_idx_path = outdir / "anchor_strict_indices.npy"
    anchor_pid_path = outdir / "anchor_point_ids.npy"
    anchor_weight_path = outdir / "anchor_base_weight.npy"
    stats_path = outdir / "residual_ramp_ifg_stats.csv"
    manifest_path = outdir / "residual_ramp_manifest.json"

    print("=" * 104)
    print("PRODUCTION IFG-DOMAIN RESIDUAL DEGREE-1 RAMP")
    print("=" * 104)
    print("config                     :", config_path)
    print("mode                       :", mode)
    print("domain                     :", domain)
    print("strict points              :", f"{nstrict:,}")
    print("strict PS                  :", f"{ps_strict.size:,}")
    print("acquisitions / IFGs        :", ndate, "/", nifg)
    print("anchor source              : all final strict PS")
    print("spatial balance            : equal total base weight per 2-km cell")
    print("Huber delta / iterations   :", delta, "/", iterations)
    print("applied correction         : network-projected ax*x + by*y only")
    print("intercept                  : fitted, diagnostic only, NOT removed")
    print()

    if mode in ("disabled", "none", "off", "false", "0"):
        np.save(coeff_direct_path, np.zeros((nifg, 3)))
        np.save(coeff_projected_path, np.zeros((nifg, 3)))
        np.save(acq_coeff_path, np.zeros((ndate, 2)))
        np.save(anchor_idx_path, np.empty(0, dtype=np.int64))
        np.save(anchor_pid_path, np.empty(0, dtype=np.int64))
        np.save(anchor_weight_path, np.empty(0, dtype=np.float64))

        with stats_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "ifg_index_0based",
                "pair_id",
                "date1",
                "date2",
                "direct_ax_rad_per_km",
                "direct_by_rad_per_km",
                "direct_intercept_rad",
                "projected_ax_rad_per_km",
                "projected_by_rad_per_km",
            ])

        atomic_json(
            manifest_path,
            {
                "status": "PASS_DISABLED",
                "mode": "disabled",
                "domain": "ifg",
                "points": int(nstrict),
                "strict_ps": int(ps_strict.size),
                "ifgs": int(nifg),
                "acquisitions": int(ndate),
                "subtract_intercept": False,
            },
        )
        print("RESIDUAL RAMP STATUS: PASS_DISABLED")
        return

    if mode not in ("robust_huber_balanced", "ifg_network_huber"):
        raise RuntimeError(f"Unsupported residual_ramp mode: {mode}")

    if ps_strict.size < min_anchors:
        raise RuntimeError(
            f"strict PS={ps_strict.size} < min_anchors={min_anchors}"
        )

    t0 = time.perf_counter()

    coords_m, lon0, lat0 = local_xy_m(lon, lat)
    x_km = coords_m[:, 0] / 1000.0
    y_km = coords_m[:, 1] / 1000.0

    base_weight, cell_index, cell_meta = cell_balanced_weights(
        coords_m[ps_strict],
        cell_size_m=cell_size_m,
    )

    occupied_cells = int(cell_meta["occupied_cells"])
    if occupied_cells < min_cells:
        raise RuntimeError(
            f"occupied cells={occupied_cells} < min_occupied_cells={min_cells}"
        )

    Xa = np.column_stack((
        x_km[ps_strict],
        y_km[ps_strict],
        np.ones(ps_strict.size, dtype=np.float64),
    ))

    design_rank = int(np.linalg.matrix_rank(Xa))
    if design_rank != 3:
        raise RuntimeError(
            f"anchor design rank={design_rank}, expected 3"
        )

    Xn = Xa.copy()
    for j in (0, 1):
        s = float(np.std(Xn[:, j]))
        if s <= 0:
            raise RuntimeError("zero PS spatial spread")
        Xn[:, j] = (Xn[:, j] - np.mean(Xn[:, j])) / s

    design_cond = float(np.linalg.cond(Xn))

    direct = np.full((nifg, 3), np.nan, dtype=np.float64)
    robust_scale = np.full(nifg, np.nan, dtype=np.float64)
    robust_used = np.zeros(nifg, dtype=np.int32)

    for e, (i, j) in enumerate(edges):
        tag = pair_tag(e + 1, i, j, stack.dates)
        src = unwrap_dir / f"{tag}_unwrapped_phase_rad.npy"

        if not src.is_file():
            raise FileNotFoundError(src)

        u = np.load(src, mmap_mode="r")
        if strict_ids.size and int(strict_ids.max()) >= u.size:
            raise RuntimeError(
                f"{src.name}: strict IDs exceed IFG domain"
            )

        z = np.asarray(u[strict_ids], dtype=np.float64)

        beta, scale, used = huber_plane(
            Xa,
            z[ps_strict],
            base_weight,
            iterations=iterations,
            delta=delta,
        )

        if not np.all(np.isfinite(beta)):
            raise RuntimeError(
                f"IFG robust plane failed: {src.name}"
            )

        direct[e] = beta
        robust_scale[e] = scale
        robust_used[e] = used

        if e == 0 or (e + 1) % 10 == 0 or e + 1 == nifg:
            print(
                f"[IFG RAMP FIT] {e+1:3d}/{nifg} "
                f"|g|={math.hypot(beta[0], beta[1]):.5f} rad/km "
                f"scale={scale:.5f} rad",
                flush=True,
            )

    projected_xy, acquisition_xy, projection = (
        network_project_ifg_slopes(
            edges,
            ndate,
            direct[:, :2],
            reference_idx=tref,
        )
    )

    projected = direct.copy()
    projected[:, :2] = projected_xy

    np.save(coeff_direct_path, direct)
    np.save(coeff_projected_path, projected)
    np.save(acq_coeff_path, acquisition_xy)
    np.save(anchor_idx_path, ps_strict.astype(np.int64))
    np.save(
        anchor_pid_path,
        strict_ids[ps_strict].astype(np.int64),
    )
    np.save(anchor_weight_path, base_weight.astype(np.float64))
    np.save(
        outdir / "anchor_cell_index.npy",
        cell_index.astype(np.int32),
    )

    rows = []
    direct_rms_all = np.empty(nifg, dtype=np.float64)
    projected_rms_all = np.empty(nifg, dtype=np.float64)
    diff_rms_all = np.empty(nifg, dtype=np.float64)
    ratio_all = np.empty(nifg, dtype=np.float64)

    for e, (i, j) in enumerate(edges):
        tag = pair_tag(e + 1, i, j, stack.dates)
        src = unwrap_dir / f"{tag}_unwrapped_phase_rad.npy"
        dst = ifg_outdir / f"{tag}_unwrapped_phase_rad.npy"

        u = np.load(src, mmap_mode="r")
        z = np.asarray(u[strict_ids], dtype=np.float64)

        dax, dby = direct[e, :2]
        pax, pby = projected[e, :2]

        direct_ramp = dax * x_km + dby * y_km
        projected_ramp = pax * x_km + pby * y_km
        projection_diff = direct_ramp - projected_ramp

        corrected = (z - projected_ramp).astype(np.float32)

        tmp = dst.with_name("." + dst.name + ".tmp")
        with tmp.open("wb") as f:
            np.save(f, corrected)
        os.replace(tmp, dst)

        drms = float(np.sqrt(np.mean(direct_ramp * direct_ramp)))
        prms = float(np.sqrt(np.mean(projected_ramp * projected_ramp)))
        xrms = float(np.sqrt(np.mean(projection_diff * projection_diff)))
        ratio = xrms / max(drms, 1.0e-12)

        direct_rms_all[e] = drms
        projected_rms_all[e] = prms
        diff_rms_all[e] = xrms
        ratio_all[e] = ratio

        rows.append({
            "ifg_index_0based": e,
            "pair_id": e + 1,
            "date1": str(stack.dates[i]),
            "date2": str(stack.dates[j]),
            "direct_ax_rad_per_km": float(dax),
            "direct_by_rad_per_km": float(dby),
            "direct_intercept_rad": float(direct[e, 2]),
            "projected_ax_rad_per_km": float(pax),
            "projected_by_rad_per_km": float(pby),
            "huber_scale_rad": float(robust_scale[e]),
            "huber_iterations": int(robust_used[e]),
            "direct_slope_rms_rad": drms,
            "projected_slope_rms_rad": prms,
            "projection_diff_rms_rad": xrms,
            "projection_diff_to_direct_ratio": ratio,
        })

    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    combined_corr = float(
        np.corrcoef(
            direct[:, :2].reshape(-1),
            projected_xy.reshape(-1),
        )[0, 1]
    )
    ratio_q = np.percentile(ratio_all, [50, 95, 99]).tolist()

    projection_recommendation = (
        "PASS"
        if combined_corr >= 0.90 and ratio_q[1] <= 0.40
        else "REVIEW"
    )

    stage_status = (
        "PASS_IFG_NETWORK_PROJECTED_HUBER"
        if projection_recommendation == "PASS"
        else "REVIEW_IFG_NETWORK_PROJECTION"
    )

    manifest = {
        "status": stage_status,
        "mode": mode,
        "domain": "ifg",
        "method": (
            "all_final_strict_PS_equal_cell_total_weight_"
            "Huber_degree1_IFG_fit_network_projected_slopes"
        ),
        "points": int(nstrict),
        "strict_ps": int(ps_strict.size),
        "anchors": int(ps_strict.size),
        "anchor_policy": "all_final_strict_PS",
        "anchor_weighting": "equal_total_base_weight_per_metric_cell",
        "cell_size_m": float(cell_size_m),
        "occupied_cells": occupied_cells,
        "cell_ps_count_min": int(cell_meta["cell_ps_count_min"]),
        "cell_ps_count_p50": float(cell_meta["cell_ps_count_p50"]),
        "cell_ps_count_p95": float(cell_meta["cell_ps_count_p95"]),
        "cell_ps_count_max": int(cell_meta["cell_ps_count_max"]),
        "anchor_design_rank": design_rank,
        "anchor_normalized_condition_number": design_cond,
        "huber_delta": float(delta),
        "huber_iterations": int(iterations),
        "acquisitions": int(ndate),
        "ifgs": int(nifg),
        "temporal_reference_index_0based": int(tref),
        "temporal_reference_date": str(stack.dates[tref]),
        "coordinate_origin_lon_deg": float(lon0),
        "coordinate_origin_lat_deg": float(lat0),
        "subtract_intercept": False,
        "network_projection": {
            **projection,
            "combined_xy_correlation": combined_corr,
            "spatial_diff_to_direct_ratio_p50_p95_p99": ratio_q,
            "recommendation": projection_recommendation,
        },
        "direct_slope_rms_rad_p50_p95_p99": [
            float(x)
            for x in np.percentile(
                direct_rms_all,
                [50, 95, 99],
            )
        ],
        "projected_slope_rms_rad_p50_p95_p99": [
            float(x)
            for x in np.percentile(
                projected_rms_all,
                [50, 95, 99],
            )
        ],
        "projection_diff_rms_rad_p50_p95_p99": [
            float(x)
            for x in np.percentile(
                diff_rms_all,
                [50, 95, 99],
            )
        ],
        "elapsed_seconds": float(time.perf_counter() - t0),
        "scientific_note": (
            "All final strict PS participate. Each 2-km cell receives "
            "equal total base weight; Huber IRLS suppresses localized "
            "deformation/outliers. IFG slopes are estimated directly, "
            "then projected onto the connected acquisition network so "
            "the applied correction is temporally integrable. The fitted "
            "intercept is diagnostic only and is not removed."
        ),
    }

    atomic_json(manifest_path, manifest)

    print()
    print("=" * 104)
    print("RESIDUAL-RAMP FINAL QA")
    print("=" * 104)
    print("anchors / occupied cells   :", ps_strict.size, "/", occupied_cells)
    print("PS/cell min/p50/p95/max    :", [
        cell_meta["cell_ps_count_min"],
        cell_meta["cell_ps_count_p50"],
        cell_meta["cell_ps_count_p95"],
        cell_meta["cell_ps_count_max"],
    ])
    print("anchor design rank         :", design_rank, "/ 3")
    print("normalized condition       :", design_cond)
    print("direct/projected xy corr   :", combined_corr)
    print("projection ratio p50/p95/99:", ratio_q)
    print("projection recommendation  :", projection_recommendation)
    print("corrected IFGs             :", ifg_outdir)
    print("manifest                   :", manifest_path)
    print("=" * 104)
    print("RESIDUAL RAMP STATUS:", stage_status)
    print("=" * 104)


if __name__ == "__main__":
    main()
