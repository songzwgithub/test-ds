#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from pypsds.config import cfg_get
from pypsds.context import open_from_config
from pypsds.corrections.residual_ramp import (
    fit_epoch_planes,
    local_xy_m,
    select_balanced_anchors,
)

TYPE_PS = np.uint8(1)


def atomic_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def compute_ps_adi(
    stack,
    *,
    roi_row0,
    roi_col0,
    H,
    W,
    ps_strict,
    strict_rows,
    strict_cols,
    tile_rows,
):
    """
    Exact PS amplitude-dispersion index from raw RSLC.

    Matches the validated PS statistic:
      amplitude = abs(complex64) -> float32
      mean/std  = float64, ddof=0
      ADI       = std / mean
    """
    ps_rows = strict_rows[ps_strict]
    ps_cols = strict_cols[ps_strict]

    adi = np.full(ps_strict.size, np.nan, dtype=np.float64)

    order = np.argsort(ps_rows, kind="stable")
    sorted_rows = ps_rows[order]

    for r0 in range(0, H, tile_rows):
        r1 = min(r0 + tile_rows, H)

        lo = int(np.searchsorted(sorted_rows, r0, side="left"))
        hi = int(np.searchsorted(sorted_rows, r1, side="left"))

        if hi <= lo:
            continue

        local_ord = order[lo:hi]
        rr = ps_rows[local_ord] - r0
        cc = ps_cols[local_ord]

        raw = stack.read_window(
            row0=int(roi_row0 + r0),
            col0=int(roi_col0),
            rows=int(r1 - r0),
            cols=int(W),
        )

        z = raw[:, rr, cc]

        finite = np.all(
            np.isfinite(z.real) & np.isfinite(z.imag),
            axis=0,
        )
        nonzero = ~np.any(
            (z.real == 0) & (z.imag == 0),
            axis=0,
        )
        good = finite & nonzero

        amp = np.abs(z).astype(np.float32, copy=False)
        x = amp[:, good].astype(np.float64, copy=False)

        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0, ddof=0)

        values = np.full(local_ord.size, np.nan, dtype=np.float64)

        good_mean = np.isfinite(mean) & (mean > 0)
        good_pos = np.flatnonzero(good)

        values[good_pos[good_mean]] = std[good_mean] / mean[good_mean]
        adi[local_ord] = values

        print(
            f"[RESIDUAL RAMP ADI] rows {r0:4d}:{r1:4d} "
            f"PS={local_ord.size:5d} "
            f"finite={np.count_nonzero(np.isfinite(values)):5d}",
            flush=True,
        )

    return adi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg, config_path, paths, stack, roi = open_from_config(args.config)
    roi_row0, roi_col0, H, W = roi

    proc = Path(paths.output_dir) / "processing"
    inv = proc / "network_inversion"
    geom = proc / "point_geometry"
    pps = proc / "point_phase_stack"
    outdir = proc / "residual_ramp"
    outdir.mkdir(parents=True, exist_ok=True)

    phase = np.load(
        inv / "acquisition_phase_l2_candidate_rad.npy",
        mmap_mode="r",
    )
    strict_ids = np.asarray(
        np.load(inv / "strict_point_ids.npy"),
        dtype=np.int64,
    )

    lon = np.asarray(
        np.load(geom / "longitude_deg.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    lat = np.asarray(
        np.load(geom / "latitude_deg.npy", mmap_mode="r"),
        dtype=np.float64,
    )

    rows_full = np.asarray(
        np.load(pps / "rows.npy", mmap_mode="r"),
        dtype=np.int64,
    )
    cols_full = np.asarray(
        np.load(pps / "cols.npy", mmap_mode="r"),
        dtype=np.int64,
    )
    type_full = np.asarray(
        np.load(pps / "point_type.npy", mmap_mode="r"),
        dtype=np.uint8,
    )

    npoint, nepoch = phase.shape

    if strict_ids.size != npoint:
        raise RuntimeError("strict-point/phase mismatch")
    if lon.size != npoint or lat.size != npoint:
        raise RuntimeError("geometry/phase mismatch")

    strict_rows = rows_full[strict_ids]
    strict_cols = cols_full[strict_ids]
    strict_type = type_full[strict_ids]

    mode = str(
        cfg_get(cfg, "corrections.residual_ramp.mode", "disabled")
    ).strip().lower()

    tref = int(
        cfg_get(cfg, "phase_linking.temporal_reference_index", 0)
    )

    out_path = outdir / "acquisition_phase_deramped_rad.npy"
    coeff_path = outdir / "ramp_coefficients_rad_per_km.npy"
    anchor_idx_path = outdir / "anchor_strict_indices.npy"
    anchor_pid_path = outdir / "anchor_point_ids.npy"
    csv_path = outdir / "residual_ramp_epoch_stats.csv"
    manifest_path = outdir / "residual_ramp_manifest.json"

    print("=" * 96)
    print("PRODUCTION RESIDUAL DEGREE-1 PHASE RAMP")
    print("=" * 96)
    print("config                    :", config_path)
    print("mode                      :", mode)
    print("points / epochs           :", f"{npoint:,}", "/", nepoch)

    if mode in ("disabled", "none", "off", "false", "0"):
        coeff = np.zeros((nepoch, 3), dtype=np.float64)

        tmp = outdir / ".acquisition_phase_deramped_rad.disabled.tmp"
        if tmp.exists():
            tmp.unlink()

        out = np.lib.format.open_memmap(
            tmp,
            mode="w+",
            dtype=np.float64,
            shape=phase.shape,
        )
        out[:] = np.asarray(phase, dtype=np.float64)
        out[:, tref] = 0.0
        out.flush()
        del out
        os.replace(tmp, out_path)

        np.save(coeff_path, coeff)
        np.save(anchor_idx_path, np.empty(0, dtype=np.int64))
        np.save(anchor_pid_path, np.empty(0, dtype=np.int64))

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch_index_0based", "date",
                "ax_rad_per_km", "by_rad_per_km", "intercept_rad",
                "huber_scale_rad", "huber_iterations",
                "spatial_slope_rms_rad",
                "spatial_slope_peak_to_peak_rad",
            ])
            for e, date in enumerate(stack.dates):
                w.writerow([e, str(date), 0, 0, 0, 0, 0, 0, 0])

        atomic_json(
            manifest_path,
            {
                "status": "PASS_DISABLED",
                "mode": "disabled",
                "points": int(npoint),
                "epochs": int(nepoch),
                "anchors": 0,
                "temporal_reference_index_0based": int(tref),
            },
        )
        print("RESIDUAL RAMP STATUS: PASS_DISABLED")
        return

    if mode not in ("robust_huber_balanced", "adi_ranked_huber_balanced"):
        raise RuntimeError(f"unsupported residual_ramp mode: {mode}")

    anchor_source = str(
        cfg_get(cfg, "corrections.residual_ramp.anchor_source", "ps")
    ).strip().lower()

    rank_metric = str(
        cfg_get(
            cfg,
            "corrections.residual_ramp.anchor_rank_metric",
            "amplitude_dispersion_index",
        )
    ).strip().lower()

    weighting = str(
        cfg_get(cfg, "corrections.residual_ramp.anchor_weighting", "equal")
    ).strip().lower()

    if anchor_source != "ps":
        raise RuntimeError("residual_ramp requires anchor_source=ps")
    if rank_metric not in ("amplitude_dispersion_index", "adi"):
        raise RuntimeError(
            "residual_ramp requires anchor_rank_metric=amplitude_dispersion_index"
        )
    if weighting != "equal":
        raise RuntimeError("residual_ramp requires anchor_weighting=equal")

    cell_size_m = float(
        cfg_get(cfg, "corrections.residual_ramp.cell_size_m", 2000.0)
    )
    per_cell = int(
        cfg_get(cfg, "corrections.residual_ramp.anchors_per_cell", 8)
    )
    min_anchors = int(
        cfg_get(cfg, "corrections.residual_ramp.min_anchors", 30)
    )
    delta = float(
        cfg_get(cfg, "corrections.residual_ramp.huber_delta", 1.345)
    )
    iterations = int(
        cfg_get(cfg, "corrections.residual_ramp.huber_iterations", 5)
    )
    chunk = max(
        1024,
        int(cfg_get(cfg, "corrections.residual_ramp.chunk_points", 131072)),
    )
    adi_tile_rows = max(
        16,
        int(cfg_get(cfg, "corrections.residual_ramp.adi_tile_rows", 96)),
    )

    ps_strict = np.flatnonzero(strict_type == TYPE_PS).astype(np.int64)

    if ps_strict.size < min_anchors:
        raise RuntimeError(
            f"strict PS={ps_strict.size} < min_anchors={min_anchors}"
        )

    print("strict PS                 :", f"{ps_strict.size:,}")
    print("anchor policy             : lowest ADI PS per metric cell")
    print("anchor weighting          : equal")
    print("cell / anchors per cell   :", cell_size_m, "/", per_cell)
    print("Huber delta / iterations  :", delta, "/", iterations)

    t0 = time.perf_counter()

    adi = compute_ps_adi(
        stack,
        roi_row0=roi_row0,
        roi_col0=roi_col0,
        H=H,
        W=W,
        ps_strict=ps_strict,
        strict_rows=strict_rows,
        strict_cols=strict_cols,
        tile_rows=adi_tile_rows,
    )

    if not np.all(np.isfinite(adi)):
        raise RuntimeError("strict PS ADI contains non-finite values")

    coords_m, lon0, lat0 = local_xy_m(lon, lat)
    x_km = coords_m[:, 0] / 1000.0
    y_km = coords_m[:, 1] / 1000.0

    selection_quality = np.full(npoint, np.nan, dtype=np.float64)
    selection_quality[ps_strict] = np.clip(1.0 - adi, 0.05, 1.0)

    anchors, occupied_cells = select_balanced_anchors(
        coords_m,
        selection_quality,
        cell_size_m=cell_size_m,
        anchors_per_cell=per_cell,
    )

    if anchors.size < min_anchors:
        raise RuntimeError(
            f"selected anchors={anchors.size} < min_anchors={min_anchors}"
        )
    if np.any(strict_type[anchors] != TYPE_PS):
        raise RuntimeError("non-PS point entered anchor set")

    lookup = np.full(npoint, -1, dtype=np.int64)
    lookup[ps_strict] = np.arange(ps_strict.size, dtype=np.int64)
    anchor_adi = adi[lookup[anchors]]

    Xa = np.column_stack((
        x_km[anchors],
        y_km[anchors],
        np.ones(anchors.size, dtype=np.float64),
    ))

    rank = int(np.linalg.matrix_rank(Xa))
    if rank != 3:
        raise RuntimeError(f"anchor design rank={rank}, expected 3")

    Xn = Xa.copy()
    for j in (0, 1):
        s = float(np.std(Xn[:, j]))
        if s <= 0:
            raise RuntimeError("zero anchor spatial spread")
        Xn[:, j] = (Xn[:, j] - np.mean(Xn[:, j])) / s
    cond = float(np.linalg.cond(Xn))

    # ADI ranks/selects anchors only. Base weights are deliberately equal.
    weights = np.ones(anchors.size, dtype=np.float64)

    coeff, robust_scale, used = fit_epoch_planes(
        Xa,
        np.asarray(phase[anchors, :], dtype=np.float64),
        weights,
        iterations=iterations,
        delta=delta,
        temporal_reference_index=tref,
    )

    tmp = outdir / ".acquisition_phase_deramped_rad.tmp"
    if tmp.exists():
        tmp.unlink()

    out = np.lib.format.open_memmap(
        tmp,
        mode="w+",
        dtype=np.float64,
        shape=(npoint, nepoch),
    )

    slope_ss = np.zeros(nepoch, dtype=np.float64)
    slope_min = np.full(nepoch, np.inf, dtype=np.float64)
    slope_max = np.full(nepoch, -np.inf, dtype=np.float64)
    nsum = 0

    for p0 in range(0, npoint, chunk):
        p1 = min(p0 + chunk, npoint)

        X = np.column_stack((
            x_km[p0:p1],
            y_km[p0:p1],
            np.ones(p1-p0, dtype=np.float64),
        ))

        ramp = X @ coeff.T
        corrected = np.asarray(
            phase[p0:p1, :], dtype=np.float64
        ) - ramp

        corrected[:, tref] = 0.0
        out[p0:p1, :] = corrected

        spatial = ramp - coeff[:, 2][None, :]
        slope_ss += np.sum(spatial * spatial, axis=0)
        slope_min = np.minimum(slope_min, np.min(spatial, axis=0))
        slope_max = np.maximum(slope_max, np.max(spatial, axis=0))
        nsum += p1 - p0

        print(
            f"[RESIDUAL RAMP] {p1:,}/{npoint:,} "
            f"({100.0*p1/npoint:.1f}%)",
            flush=True,
        )

    out.flush()
    del out
    os.replace(tmp, out_path)

    slope_rms = np.sqrt(slope_ss / max(1, nsum))
    slope_ptp = slope_max - slope_min

    np.save(coeff_path, coeff)
    np.save(anchor_idx_path, anchors.astype(np.int64))
    np.save(anchor_pid_path, strict_ids[anchors].astype(np.int64))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch_index_0based", "date",
            "ax_rad_per_km", "by_rad_per_km", "intercept_rad",
            "huber_scale_rad", "huber_iterations",
            "spatial_slope_rms_rad",
            "spatial_slope_peak_to_peak_rad",
        ])
        for e, date in enumerate(stack.dates):
            w.writerow([
                e, str(date),
                f"{coeff[e,0]:.12e}",
                f"{coeff[e,1]:.12e}",
                f"{coeff[e,2]:.12e}",
                f"{robust_scale[e]:.12e}",
                int(used[e]),
                f"{slope_rms[e]:.12e}",
                f"{slope_ptp[e]:.12e}",
            ])

    manifest = {
        "status": "PASS_ADI_RANKED_EQUAL_HUBER",
        "mode": "robust_huber_balanced",
        "method": "spatially_balanced_lowest_ADI_PS_equal_weight_Huber_degree1",
        "points": int(npoint),
        "epochs": int(nepoch),
        "strict_ps": int(ps_strict.size),
        "anchors": int(anchors.size),
        "occupied_cells": int(occupied_cells),
        "anchor_design_rank": int(rank),
        "anchor_normalized_condition_number": float(cond),
        "anchor_source": "ps",
        "anchor_rank_metric": "amplitude_dispersion_index",
        "anchor_weighting": "equal",
        "anchor_ps": int(anchors.size),
        "anchor_ds": 0,
        "strict_ps_adi_p01_p05_p50_p95_p99": [
            float(x) for x in np.percentile(adi, [1,5,50,95,99])
        ],
        "anchor_adi_p01_p50_p99": [
            float(x) for x in np.percentile(anchor_adi, [1,50,99])
        ],
        "cell_size_m": float(cell_size_m),
        "anchors_per_cell": int(per_cell),
        "min_anchors": int(min_anchors),
        "huber_delta": float(delta),
        "huber_iterations": int(iterations),
        "adi_tile_rows": int(adi_tile_rows),
        "temporal_reference_index_0based": int(tref),
        "temporal_reference_date": str(stack.dates[tref]),
        "coordinate_origin_lon_deg": float(lon0),
        "coordinate_origin_lat_deg": float(lat0),
        "ax_abs_p50_p95_rad_per_km": [
            float(x) for x in np.percentile(np.abs(coeff[:,0]), [50,95])
        ],
        "by_abs_p50_p95_rad_per_km": [
            float(x) for x in np.percentile(np.abs(coeff[:,1]), [50,95])
        ],
        "spatial_slope_rms_p50_p95_rad": [
            float(x) for x in np.percentile(slope_rms, [50,95])
        ],
        "spatial_slope_peak_to_peak_p50_p95_rad": [
            float(x) for x in np.percentile(slope_ptp, [50,95])
        ],
        "elapsed_seconds": float(time.perf_counter() - t0),
        "scientific_note": (
            "ADI ranks/selects strict PS anchors only. "
            "Selected anchors use equal base weight in the Huber fit. "
            "Spatial reference is applied later by the reference stage."
        ),
    }

    atomic_json(manifest_path, manifest)

    check = np.load(out_path, mmap_mode="r")
    epoch0 = float(
        np.max(np.abs(np.asarray(check[:, tref], dtype=np.float64)))
    )
    if epoch0 != 0.0:
        raise RuntimeError(f"temporal gauge failed: {epoch0}")

    print()
    print("=" * 96)
    print("RESIDUAL-RAMP FINAL QA")
    print("=" * 96)
    print("anchors / cells             :", anchors.size, "/", occupied_cells)
    print("anchor ADI p01/p50/p99      :", manifest["anchor_adi_p01_p50_p99"])
    print("design rank                 :", rank, "/ 3")
    print("normalized condition        :", cond)
    print("|ax| p50/p95 rad/km         :", manifest["ax_abs_p50_p95_rad_per_km"])
    print("|by| p50/p95 rad/km         :", manifest["by_abs_p50_p95_rad_per_km"])
    print("slope RMS p50/p95 rad       :", manifest["spatial_slope_rms_p50_p95_rad"])
    print("epoch0 max |rad|            :", epoch0)
    print("=" * 96)
    print("RESIDUAL RAMP STATUS: PASS_ADI_RANKED_EQUAL_HUBER")
    print("=" * 96)


if __name__ == "__main__":
    main()
