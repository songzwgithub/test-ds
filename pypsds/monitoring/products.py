from __future__ import annotations

import csv
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from pypsds.config import cfg_get
from pypsds.context import open_from_config
from .vertical import vertical_factor


def _pct(x, q=(1, 5, 50, 95, 99)):
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return [float(v) for v in np.percentile(a, q)]


def _annual_metrics(los, dates, min_obs, min_span_days, outdir):
    dobj = [datetime.strptime(str(x), "%Y%m%d") for x in dates]
    groups = []
    years = []
    for year in sorted({x.year for x in dobj}):
        ids = np.asarray(
            [i for i, x in enumerate(dobj) if x.year == year],
            dtype=np.int32,
        )
        if ids.size < min_obs:
            continue
        if (dobj[int(ids[-1])] - dobj[int(ids[0])]).days < min_span_days:
            continue
        years.append(year)
        groups.append(ids)

    npoint = los.shape[0]
    ny = len(groups)
    vpath = outdir / "annual_velocity_toward_satellite_mm_per_year.npy"
    rpath = outdir / "annual_velocity_residual_rms_mm.npy"
    if ny == 0:
        np.save(vpath, np.empty((npoint, 0), dtype=np.float32))
        np.save(rpath, np.empty((npoint, 0), dtype=np.float32))
        np.save(outdir / "annual_velocity_years.npy", np.empty(0, dtype=np.int32))
        return years, vpath, rpath

    vout = np.lib.format.open_memmap(
        vpath, mode="w+", dtype=np.float32, shape=(npoint, ny)
    )
    rout = np.lib.format.open_memmap(
        rpath, mode="w+", dtype=np.float32, shape=(npoint, ny)
    )

    for j, ids in enumerate(groups):
        t0 = dobj[int(ids[0])]
        t = np.asarray(
            [(dobj[int(k)] - t0).days / 365.25 for k in ids],
            dtype=np.float64,
        )
        tc = t - t.mean()
        denom = float(tc @ tc)
        for b0 in range(0, npoint, 100000):
            b1 = min(b0 + 100000, npoint)
            Y = np.asarray(los[b0:b1, :][:, ids], dtype=np.float64)
            ym = np.mean(Y, axis=1)
            slope = ((Y - ym[:, None]) @ tc) / denom
            intercept = ym - slope * float(t.mean())
            rr = Y - (intercept[:, None] + slope[:, None] * t[None, :])
            vout[b0:b1, j] = slope.astype(np.float32)
            rout[b0:b1, j] = np.sqrt(
                np.mean(rr * rr, axis=1)
            ).astype(np.float32)

    vout.flush()
    rout.flush()
    del vout, rout
    np.save(outdir / "annual_velocity_years.npy", np.asarray(years, dtype=np.int32))
    return years, vpath, rpath


def _scatter(path, lon, lat, values, title, label, max_points=250000):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"figure export skipped: {exc}")
        return False

    values = np.asarray(values, dtype=np.float64)
    ids = np.flatnonzero(
        np.isfinite(lon) & np.isfinite(lat) & np.isfinite(values)
    )
    if ids.size == 0:
        return False
    if ids.size > max_points:
        ids = ids[
            np.linspace(0, ids.size - 1, max_points, dtype=np.int64)
        ]

    lo, hi = np.percentile(values[ids], [2, 98])
    vmax = max(abs(float(lo)), abs(float(hi)), 1e-6)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sc = ax.scatter(
        lon[ids], lat[ids], c=values[ids], s=2.0,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        linewidths=0, rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, shrink=0.85).set_label(label)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return True


def _utm_epsg(lon, lat):
    zone = int(math.floor((float(lon) + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if float(lat) >= 0 else 32700) + zone


def _geotiff(path, lon, lat, values, resolution_m):
    """
    Non-interpolated engineering raster. Points in an occupied UTM cell are
    averaged; empty cells remain nodata.
    """
    try:
        import rasterio
        from pyproj import Transformer
        from rasterio.transform import from_origin
    except Exception as exc:
        warnings.warn(f"GeoTIFF export skipped: {exc}")
        return None

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    if not np.any(good):
        return None

    lon, lat, val = lon[good], lat[good], val[good]
    epsg = _utm_epsg(np.median(lon), np.median(lat))
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = tr.transform(lon, lat)
    x, y = np.asarray(x), np.asarray(y)

    res = float(resolution_m)
    xmin = math.floor(float(np.min(x)) / res) * res
    xmax = math.ceil(float(np.max(x)) / res) * res
    ymin = math.floor(float(np.min(y)) / res) * res
    ymax = math.ceil(float(np.max(y)) / res) * res
    width = max(1, int(math.ceil((xmax - xmin) / res)))
    height = max(1, int(math.ceil((ymax - ymin) / res)))

    col = np.clip(np.floor((x - xmin) / res).astype(np.int64), 0, width - 1)
    row = np.clip(np.floor((ymax - y) / res).astype(np.int64), 0, height - 1)
    linear = row * width + col
    count = np.bincount(linear, minlength=height * width)
    total = np.bincount(linear, weights=val, minlength=height * width)

    grid = np.full(height * width, np.nan, dtype=np.float32)
    occupied = count > 0
    grid[occupied] = (total[occupied] / count[occupied]).astype(np.float32)
    grid = grid.reshape(height, width)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=f"EPSG:{epsg}",
        transform=from_origin(xmin, ymax, res, res),
        nodata=np.nan,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(grid, 1)

    return {
        "path": str(path),
        "epsg": epsg,
        "resolution_m": res,
        "width": width,
        "height": height,
        "occupied_cells": int(np.count_nonzero(occupied)),
        "interpolation": False,
        "cell_statistic": "arithmetic_mean",
    }


def _monitoring_csv(path, columns):
    try:
        import pandas as pd
        pd.DataFrame(columns).to_csv(path, index=False)
        return
    except Exception:
        pass

    names = list(columns)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(names)
        n = len(columns[names[0]])
        for i in range(n):
            w.writerow([columns[k][i] for k in names])


def _geojson(path, columns):
    names = [k for k in columns if k not in {"longitude_deg", "latitude_deg"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","features":[\n')
        for i in range(len(columns["point_id"])):
            props = {}
            for k in names:
                v = columns[k][i]
                props[k] = v.item() if isinstance(v, np.generic) else v
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(columns["longitude_deg"][i]),
                        float(columns["latitude_deg"][i]),
                    ],
                },
                "properties": props,
            }
            if i:
                f.write(",\n")
            f.write(json.dumps(feature, ensure_ascii=False))
        f.write("\n]}\n")


def _shapefile(path_base, columns):
    try:
        import shapefile
    except Exception as exc:
        warnings.warn(f"Shapefile export skipped: {exc}")
        return False

    path_base.parent.mkdir(parents=True, exist_ok=True)
    w = shapefile.Writer(str(path_base), shapeType=shapefile.POINT)
    w.field("point_id", "N", 12, 0)
    w.field("los_vel", "F", 18, 6)
    w.field("cum_mm", "F", 18, 6)
    w.field("rms_mm", "F", 18, 6)
    w.field("unc_mm_y", "F", 18, 6)
    for i in range(len(columns["point_id"])):
        w.point(
            float(columns["longitude_deg"][i]),
            float(columns["latitude_deg"][i]),
        )
        w.record(
            int(columns["point_id"][i]),
            float(columns["los_velocity_toward_satellite_mm_per_year"][i]),
            float(columns["los_cumulative_toward_satellite_mm"][i]),
            float(columns["linear_residual_rms_mm"][i]),
            float(columns["velocity_formal_uncertainty_mm_per_year"][i]),
        )
    w.close()
    path_base.with_suffix(".prj").write_text(
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
        encoding="ascii",
    )
    path_base.with_suffix(".cpg").write_text("UTF-8\n", encoding="ascii")
    return True


def _html(path, report):
    rows = []
    for key, value in report.items():
        shown = (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else str(value)
        )
        rows.append(
            f"<tr><th>{key}</th><td><pre>{shown}</pre></td></tr>"
        )
    path.write_text(
        """<!doctype html><html><head><meta charset="utf-8">
<title>pyPSDS-GAMMA monitoring QA</title>
<style>body{font-family:Arial;margin:2rem;max-width:1200px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:.5rem;text-align:left;vertical-align:top}
th{width:28%;background:#f4f4f4}pre{white-space:pre-wrap;margin:0}</style>
</head><body><h1>pyPSDS-GAMMA monitoring quality summary</h1><table>"""
        + "\n".join(rows)
        + "</table></body></html>\n",
        encoding="utf-8",
    )


def build_monitoring_products(config_path):
    cfg, _, paths, stack, _ = open_from_config(config_path)
    output = Path(paths.output_dir)
    proc = output / "processing"
    products = Path(paths.products_dir)
    products.mkdir(parents=True, exist_ok=True)

    inv = proc / "network_inversion"
    ref = proc / "referenced_timeseries"
    geom = proc / "point_geometry"
    final = proc / "final_los"

    strict_ids = np.asarray(np.load(inv / "strict_point_ids.npy"), dtype=np.int32)
    lon = np.asarray(np.load(geom / "longitude_deg.npy", mmap_mode="r"))
    lat = np.asarray(np.load(geom / "latitude_deg.npy", mmap_mode="r"))
    incidence = np.asarray(
        np.load(geom / "incidence_rad.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    los = np.load(
        final / "los_displacement_toward_satellite_mm.npy",
        mmap_mode="r",
    )
    vel = np.load(
        products / "los_velocity_toward_satellite_mm_per_year.npy",
        mmap_mode="r",
    )
    cum = np.load(products / "los_cumulative_toward_satellite_mm.npy", mmap_mode="r")
    rms = np.load(products / "linear_residual_rms_mm.npy", mmap_mode="r")
    ols_se = np.load(
        products / "velocity_slope_standard_error_mm_per_year.npy",
        mmap_mode="r",
    )

    npoint, ndate = los.shape
    if strict_ids.size != npoint or lon.size != npoint or incidence.size != npoint:
        raise RuntimeError("monitoring product geometry contract failed")

    time_contract = np.load(products / "time_axis_contract.npz")
    slope_w = np.asarray(
        time_contract[
            "slope_weights_per_year"
            if "slope_weights_per_year" in time_contract.files
            else "slope_weights"
        ],
        dtype=np.float64,
    )
    cov_phase = np.asarray(
        np.load(inv / "acquisition_phase_covariance_rad2.npy"),
        dtype=np.float64,
    )

    final_manifest = json.loads(
        (final / "final_los_manifest.json").read_text(encoding="utf-8")
    )
    mm_per_rad = float(
        final_manifest["scientific_contract"]["los_factor_mm_per_rad"]
    )

    network_velocity_se = (
        max(float(slope_w @ cov_phase @ slope_w), 0.0) ** 0.5
    ) * abs(mm_per_rad)

    ref_sigma = np.asarray(
        np.load(ref / "reference_phase_mad_sigma_rad.npy"),
        dtype=np.float64,
    )
    ref_idx = np.asarray(
        np.load(ref / "reference_strict_indices.npy"),
        dtype=np.int64,
    )
    nref = int(ref_idx.size)
    if nref < 1:
        raise RuntimeError("reference set is empty")

    ref_median_se = 1.2533141373155 * ref_sigma / np.sqrt(float(nref))
    reference_velocity_se = (
        np.sqrt(np.sum((slope_w * ref_median_se) ** 2))
        * abs(mm_per_rad)
    )

    network_arr = np.full(npoint, network_velocity_se, dtype=np.float32)
    reference_arr = np.full(npoint, reference_velocity_se, dtype=np.float32)
    formal_unc = np.sqrt(
        np.asarray(ols_se, dtype=np.float64) ** 2
        + network_velocity_se**2
        + reference_velocity_se**2
    ).astype(np.float32)

    np.save(
        products / "network_velocity_standard_error_mm_per_year.npy",
        network_arr,
    )
    np.save(
        products / "reference_velocity_standard_error_mm_per_year.npy",
        reference_arr,
    )
    np.save(
        products / "velocity_formal_uncertainty_mm_per_year.npy",
        formal_unc,
    )

    if bool(cfg_get(cfg, "products.monitoring.annual_velocity", True)):
        annual_years, annual_vpath, annual_rpath = _annual_metrics(
            los,
            stack.dates,
            int(cfg_get(cfg, "products.monitoring.annual_min_obs", 6)),
            float(
                cfg_get(
                    cfg,
                    "products.monitoring.annual_min_span_days",
                    180.0,
                )
            ),
            products,
        )
    else:
        annual_years, annual_vpath, annual_rpath = [], None, None

    vertical_enabled = bool(cfg_get(cfg, "products.vertical.enabled", False))
    vertical_info = {"enabled": False}
    vertical_vel = vertical_cum = vertical_unc = None

    if vertical_enabled:
        positive = str(cfg_get(cfg, "products.vertical.positive", "down"))
        vf = vertical_factor(incidence, positive)
        vertical_vel = (np.asarray(vel, dtype=np.float64) * vf).astype(np.float32)
        vertical_cum = (np.asarray(cum, dtype=np.float64) * vf).astype(np.float32)
        vertical_unc = (
            np.asarray(formal_unc, dtype=np.float64) * np.abs(vf)
        ).astype(np.float32)

        np.save(products / "vertical_velocity_mm_per_year.npy", vertical_vel)
        np.save(products / "vertical_cumulative_mm.npy", vertical_cum)
        np.save(
            products / "vertical_velocity_formal_uncertainty_mm_per_year.npy",
            vertical_unc,
        )

        if bool(cfg_get(cfg, "products.vertical.timeseries", True)):
            tmp = products / ".vertical_displacement.tmp.npy"
            target = products / "vertical_displacement_mm.npy"
            if tmp.exists():
                tmp.unlink()
            out = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.float32, shape=los.shape
            )
            for b0 in range(0, npoint, 100000):
                b1 = min(b0 + 100000, npoint)
                out[b0:b1] = (
                    np.asarray(los[b0:b1], dtype=np.float64)
                    * vf[b0:b1, None]
                ).astype(np.float32)
            out.flush()
            del out
            os.replace(tmp, target)

        vertical_info = {
            "enabled": True,
            "positive": positive,
            "formula": (
                "vertical_up = LOS_toward / cos(incidence); "
                "vertical_down = -vertical_up"
            ),
            "assumption": "horizontal deformation is negligible",
        }

    columns = {
        "point_id": strict_ids,
        "longitude_deg": lon,
        "latitude_deg": lat,
        "los_velocity_toward_satellite_mm_per_year": np.asarray(vel),
        "los_cumulative_toward_satellite_mm": np.asarray(cum),
        "linear_residual_rms_mm": np.asarray(rms),
        "velocity_temporal_ols_standard_error_mm_per_year": np.asarray(ols_se),
        "network_velocity_standard_error_mm_per_year": network_arr,
        "reference_velocity_standard_error_mm_per_year": reference_arr,
        "velocity_formal_uncertainty_mm_per_year": formal_unc,
    }
    if vertical_enabled:
        columns.update(
            {
                "vertical_velocity_mm_per_year": vertical_vel,
                "vertical_cumulative_mm": vertical_cum,
                "vertical_velocity_formal_uncertainty_mm_per_year": vertical_unc,
            }
        )

    monitoring_csv = products / "monitoring_points.csv"
    _monitoring_csv(monitoring_csv, columns)
    created = [str(monitoring_csv)]

    figures = products / "figures"
    rasters = products / "rasters"
    gis = products / "gis"

    if bool(cfg_get(cfg, "products.monitoring.figures", True)):
        figures.mkdir(parents=True, exist_ok=True)
        for name, values, title, label in (
            ("velocity_map.png", vel, "LOS velocity (toward positive)", "mm/year"),
            ("cumulative_map.png", cum, "LOS cumulative displacement", "mm"),
            (
                "velocity_uncertainty_map.png",
                formal_unc,
                "Formal velocity uncertainty",
                "mm/year",
            ),
        ):
            p = figures / name
            if _scatter(p, lon, lat, values, title, label):
                created.append(str(p))

        if annual_years:
            av = np.load(annual_vpath, mmap_mode="r")
            for j, year in enumerate(annual_years):
                p = figures / f"annual_velocity_{year}.png"
                if _scatter(
                    p, lon, lat, av[:, j],
                    f"LOS annual velocity {year}", "mm/year",
                ):
                    created.append(str(p))

    geotiffs = []
    if bool(cfg_get(cfg, "products.monitoring.geotiff", True)):
        res = float(cfg_get(cfg, "products.monitoring.grid_resolution_m", 100.0))
        for name, values in (
            ("velocity_mm_per_year.tif", vel),
            ("cumulative_mm.tif", cum),
            ("residual_rms_mm.tif", rms),
            ("velocity_formal_uncertainty_mm_per_year.tif", formal_unc),
        ):
            meta = _geotiff(rasters / name, lon, lat, values, res)
            if meta:
                geotiffs.append(meta)
                created.append(meta["path"])

        if annual_years:
            av = np.load(annual_vpath, mmap_mode="r")
            for j, year in enumerate(annual_years):
                meta = _geotiff(
                    rasters / f"annual_velocity_{year}_mm_per_year.tif",
                    lon, lat, av[:, j], res,
                )
                if meta:
                    geotiffs.append(meta)
                    created.append(meta["path"])

        if vertical_enabled:
            for name, values in (
                ("vertical_velocity_mm_per_year.tif", vertical_vel),
                ("vertical_cumulative_mm.tif", vertical_cum),
            ):
                meta = _geotiff(rasters / name, lon, lat, values, res)
                if meta:
                    geotiffs.append(meta)
                    created.append(meta["path"])

    if bool(cfg_get(cfg, "products.monitoring.geojson", False)):
        p = gis / "monitoring_points.geojson"
        _geojson(p, columns)
        created.append(str(p))

    if bool(cfg_get(cfg, "products.monitoring.shapefile", False)):
        p = gis / "monitoring_points"
        if _shapefile(p, columns):
            created.append(str(p.with_suffix(".shp")))

    inv_manifest_path = inv / "monitoring_inversion_manifest.json"
    inv_manifest = (
        json.loads(inv_manifest_path.read_text(encoding="utf-8"))
        if inv_manifest_path.is_file()
        else {}
    )
    strict_mask_path = proc / "final_unwrap" / "strict_unwrap_valid_mask.npy"
    if strict_mask_path.is_file():
        sm = np.load(strict_mask_path, mmap_mode="r")
        strict_valid, strict_total = int(np.count_nonzero(sm)), int(sm.size)
    else:
        strict_valid = strict_total = npoint

    report = {
        "status": "PASS_MONITORING_PRODUCTS",
        "version": "1.3.0",
        "points": int(npoint),
        "acquisitions": int(ndate),
        "strict_unwrap_valid": {
            "valid": strict_valid,
            "total": strict_total,
            "fraction": float(strict_valid / strict_total),
        },
        "network_inversion": {
            "requested_method": inv_manifest.get("requested_method"),
            "effective_method": inv_manifest.get("effective_method"),
            "relative_weights": inv_manifest.get("relative_weights"),
        },
        "reference_points": nref,
        "uncertainty": {
            "temporal_ols_p50_p95_p99_mm_per_year": _pct(
                ols_se, (50, 95, 99)
            ),
            "network_component_mm_per_year": network_velocity_se,
            "reference_component_mm_per_year": reference_velocity_se,
            "formal_composite_p50_p95_p99_mm_per_year": _pct(
                formal_unc, (50, 95, 99)
            ),
            "interpretation": (
                "Formal monitoring uncertainty combines temporal OLS scatter, "
                "network-inversion covariance, and reference-median sampling "
                "uncertainty under an independence approximation. It is not a "
                "complete bound on correlated/systematic atmospheric, orbital, "
                "geocoding, or deformation-model errors."
            ),
        },
        "velocity_p01_p05_p50_p95_p99_mm_per_year": _pct(vel),
        "cumulative_p01_p05_p50_p95_p99_mm": _pct(cum),
        "annual_velocity_years": annual_years,
        "vertical": vertical_info,
        "geotiff": geotiffs,
        "generated": created,
    }

    qa_json = products / "monitoring_quality.json"
    qa_html = products / "monitoring_quality.html"
    qa_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _html(qa_html, report)
    created.extend([str(qa_json), str(qa_html)])

    manifest = {
        "status": "PASS_MONITORING_PRODUCTS",
        "version": "1.3.0",
        "outputs": created,
        "annual_years": annual_years,
        "vertical": vertical_info,
    }
    mpath = products / "monitoring_products_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    legacy_path = products / "point_products_manifest.json"
    if legacy_path.is_file():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        legacy["monitoring"] = manifest
        legacy_path.write_text(
            json.dumps(legacy, indent=2) + "\n", encoding="utf-8"
        )

    print("=" * 96)
    print("GROUND-DEFORMATION MONITORING PRODUCTS")
    print("=" * 96)
    print("points/acquisitions :", npoint, "/", ndate)
    print("network velocity SE :", network_velocity_se, "mm/yr")
    print("reference velocity SE:", reference_velocity_se, "mm/yr")
    print("formal uncertainty p50:", float(np.median(formal_unc)), "mm/yr")
    print("annual years        :", annual_years)
    print("vertical enabled    :", vertical_enabled)
    print("QA                  :", qa_json)
    print("=" * 96)
    return manifest
