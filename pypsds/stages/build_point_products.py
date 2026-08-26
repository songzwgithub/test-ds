from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import numpy as np

from pypsds.stages._v11_common import cfg_get, load_context, run_runtime, write_json


def _table_columns(ctx):
    proc = ctx["proc"]
    products = ctx["products_dir"]
    strict_ids = np.load(proc / "network_inversion" / "strict_point_ids.npy", mmap_mode="r")
    lon = np.load(proc / "point_geometry" / "longitude_deg.npy", mmap_mode="r")
    lat = np.load(proc / "point_geometry" / "latitude_deg.npy", mmap_mode="r")
    vel = np.load(products / "los_velocity_toward_satellite_mm_per_year.npy", mmap_mode="r")
    cum = np.load(products / "los_cumulative_toward_satellite_mm.npy", mmap_mode="r")
    rms = np.load(products / "linear_residual_rms_mm.npy", mmap_mode="r")
    se = np.load(products / "velocity_slope_standard_error_mm_per_year.npy", mmap_mode="r")
    return {
        "point_id": np.asarray(strict_ids),
        "longitude_deg": np.asarray(lon),
        "latitude_deg": np.asarray(lat),
        "los_velocity_toward_satellite_mm_per_year": np.asarray(vel),
        "los_cumulative_toward_satellite_mm": np.asarray(cum),
        "linear_residual_rms_mm": np.asarray(rms),
        "velocity_slope_standard_error_mm_per_year": np.asarray(se),
    }


def _write_tables(ctx):
    cfg = ctx["cfg"]
    if not bool(cfg_get(cfg, "products.point.enabled", True)):
        return []

    formats = [
        str(x).strip().lower()
        for x in cfg_get(cfg, "products.point.formats", ["parquet", "geopackage", "csv"])
    ]
    cols = _table_columns(ctx)
    out = ctx["products_dir"]
    created = []

    try:
        import pandas as pd
        frame = pd.DataFrame(cols)
    except Exception as exc:
        if "csv" in formats:
            path = out / "point_products.csv"
            names = list(cols)
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(names)
                n = len(cols[names[0]])
                for i in range(n):
                    w.writerow([cols[k][i] for k in names])
            created.append(str(path))
        warnings.warn(f"pandas unavailable; only CSV fallback can be written: {exc}")
        return created

    if "csv" in formats:
        path = out / "point_products.csv"
        frame.to_csv(path, index=False)
        created.append(str(path))

    if "parquet" in formats:
        path = out / "point_products.parquet"
        try:
            frame.to_parquet(path, index=False)
            created.append(str(path))
        except Exception as exc:
            warnings.warn(f"Parquet export skipped: {exc}")

    if "geopackage" in formats or "gpkg" in formats:
        path = out / "point_products.gpkg"
        try:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame(
                frame,
                geometry=gpd.points_from_xy(frame["longitude_deg"], frame["latitude_deg"]),
                crs=str(cfg_get(cfg, "products.point.crs", "EPSG:4326")),
            )
            if path.exists():
                path.unlink()
            gdf.to_file(path, layer="points", driver="GPKG")
            created.append(str(path))
        except Exception as exc:
            warnings.warn(f"GeoPackage export skipped: {exc}")

    return created


def main():
    ap = argparse.ArgumentParser(description="Build final full-resolution point products.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    ctx = load_context(args.config)
    run_runtime("point_metrics_runtime.py", ctx)

    required = (
        "los_velocity_toward_satellite_mm_per_year.npy",
        "los_cumulative_toward_satellite_mm.npy",
        "linear_residual_rms_mm.npy",
        "velocity_slope_standard_error_mm_per_year.npy",
        "time_axis_contract.npz",
        "point_products_manifest.json",
    )
    for name in required:
        p = ctx["products_dir"] / name
        if not p.is_file():
            raise FileNotFoundError(p)

    created = _write_tables(ctx)

    print("=" * 88)
    print("POINT PRODUCTS STATUS: PASS")
    print("primary directory :", ctx["products_dir"])
    print("tables            :", created)
    print("=" * 88)


if __name__ == "__main__":
    main()
