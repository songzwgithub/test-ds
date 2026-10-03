#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export Ningbo TP-LA final products to many small QGIS-ready GeoPackage files.

Design goals
------------
1. Never modify upstream InSAR products.
2. Keep strict-point order and scientific lineage.
3. Split output by theme AND by point-count shard to avoid giant GPKGs.
4. Use atomic .tmp.gpkg -> .gpkg replacement for safe resume.
5. Keep every thematic file self-contained with Point geometry (EPSG:4326).
6. Do NOT export the 75-epoch time series here; that is intentionally separate
   because repeating geometry for all epochs would multiply disk use.

Default output:
  <root>/output/gis_tp_la/
      00_geometry/
      01_velocity/
      02_seasonal/
      03_cumulative/
      04_quality/
      manifest/
          shard_index.csv
          fields.json
          README_QGIS.txt
      shard_index.gpkg

Point identifiers
-----------------
SID : row index in the final strict TP-LA arrays, 0-based.
PID : original PointPhaseStack point id, equal to strict_point_ids.npy, 0-based.
PTYPE: 1=PS, 2=DS.

LOS sign
--------
Positive LOS displacement/velocity = toward satellite.

Vertical approximation
----------------------
V_UP ~= LOS / cos(incidence)
SUBS_DN = -V_UP

This is ONLY a single-track vertical approximation assuming negligible
horizontal motion. It is not a rigorous 2D/3D decomposition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def die(msg: str) -> None:
    raise RuntimeError(msg)


def need_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_npy(path: Path):
    return np.load(need_file(path), mmap_mode="r", allow_pickle=False)


def human_gib(n: int) -> float:
    return float(n) / (1024.0 ** 3)


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def import_gis():
    try:
        import pandas as pd
        import geopandas as gpd
        import pyogrio
        import shapely
        from shapely.geometry import box
    except Exception as exc:
        raise RuntimeError(
            "GIS Python dependencies are required.\n"
            "Install in the current environment with:\n"
            "  pip install -U pandas geopandas shapely pyogrio\n"
            f"Original import error: {exc}"
        ) from exc
    return pd, gpd, pyogrio, shapely, box


def write_gpkg_atomic(
    *,
    gpd,
    pyogrio,
    frame,
    geometry,
    path: Path,
    layer: str,
    crs: str = "EPSG:4326",
    overwrite_existing: bool = False,
) -> str:
    """
    Write one standalone GPKG atomically.
    Existing complete files are skipped unless overwrite_existing=True.
    """
    if path.exists() and not overwrite_existing:
        return "SKIP"

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.gpkg")

    for p in (tmp,):
        if p.exists():
            p.unlink()

    gdf = gpd.GeoDataFrame(
        frame,
        geometry=geometry,
        crs=crs,
    )

    # GPKG spatial index is useful for QGIS. Each file is already bounded
    # by max-points, so index build remains manageable.
    pyogrio.write_dataframe(
        gdf,
        tmp,
        layer=layer,
        driver="GPKG",
        layer_options={"SPATIAL_INDEX": "YES"},
    )

    os.replace(tmp, path)
    return "WRITE"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export final Ningbo TP-LA products to split QGIS GeoPackages."
        )
    )
    p.add_argument(
        "--root",
        required=True,
        help="Project root, e.g. /mnt/ningbo_process/ds",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=1_000_000,
        help=(
            "Maximum points per GeoPackage shard. "
            "Default: 1,000,000. Use 500000 for smaller files."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory. Default: <root>/output/gis_tp_la"
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files.",
    )
    p.add_argument(
        "--no-vertical",
        action="store_true",
        help="Do not export single-track vertical approximation fields.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    root = Path(args.root).expanduser().resolve()
    out = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "output" / "gis_tp_la"
    )

    if args.max_points < 50_000:
        die("--max-points is unrealistically small; use >= 50000")

    pd, gpd, pyogrio, shapely, box = import_gis()

    proc = root / "output" / "processing"
    geom_dir = proc / "point_geometry"
    pps_dir = proc / "point_phase_stack"
    unwrap_dir = proc / "final_unwrap"
    prod = root / "output" / "products_tp_la"

    # ------------------------------------------------------------------
    # Geometry / identity
    # ------------------------------------------------------------------
    lon = load_npy(geom_dir / "longitude_deg.npy")
    lat = load_npy(geom_dir / "latitude_deg.npy")
    hgt = load_npy(geom_dir / "height_m.npy")
    inc = load_npy(geom_dir / "incidence_rad.npy")
    row = load_npy(geom_dir / "radar_row.npy")
    col = load_npy(geom_dir / "radar_col.npy")

    strict_ids = load_npy(unwrap_dir / "strict_point_ids.npy")
    point_type_full = load_npy(pps_dir / "point_type.npy")

    # Quality metadata live in full PointPhaseStack order.
    tc_full = load_npy(pps_dir / "temporal_coherence.npy")
    pair_full = load_npy(pps_dir / "median_pair_coherence.npy")
    shp_full = load_npy(pps_dir / "glrt_support_K.npy")
    est_full = load_npy(pps_dir / "estimator_code.npy")
    emi_full = load_npy(pps_dir / "emi_eigenvalue.npy")
    evd_full = load_npy(pps_dir / "evd_eigenvalue.npy")
    gamma_full = load_npy(pps_dir / "gamma_min_eigenvalue.npy")

    # ------------------------------------------------------------------
    # TP-LA scientific products
    # ------------------------------------------------------------------
    vel_la = load_npy(
        prod / "los_velocity_linear_annual_mm_per_year.npy"
    )
    vel_ols = load_npy(
        prod / "los_velocity_ols_mm_per_year.npy"
    )
    ann_sin = load_npy(
        prod / "annual_sin_coefficient_mm.npy"
    )
    ann_cos = load_npy(
        prod / "annual_cos_coefficient_mm.npy"
    )
    ann_amp = load_npy(
        prod / "annual_amplitude_mm.npy"
    )
    ann_phase = load_npy(
        prod / "annual_phase_rad.npy"
    )
    ann_peak = load_npy(
        prod / "annual_peak_lag_days_from_temporal_reference.npy"
    )
    rms_la = load_npy(
        prod / "linear_annual_residual_rms_mm.npy"
    )
    vel_se = load_npy(
        prod / "velocity_linear_annual_standard_error_mm_per_year.npy"
    )
    cumulative = load_npy(
        prod / "los_cumulative_toward_satellite_mm.npy"
    )

    manifest_path = need_file(
        prod / "point_products_tp_la_manifest.json"
    )
    product_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    time_contract_path = need_file(
        prod / "time_axis_and_model_contract.npz"
    )
    with np.load(time_contract_path, allow_pickle=False) as z:
        dates = [str(x) for x in z["acquisition_dates"]]
        temporal_reference_date = str(
            np.asarray(z["temporal_reference_date"]).item()
        )
        geometric_master_date = str(
            np.asarray(z["geometric_master_date"]).item()
        )

    n = int(lon.size)

    arrays_strict = {
        "lat": lat,
        "hgt": hgt,
        "inc": inc,
        "row": row,
        "col": col,
        "vel_la": vel_la,
        "vel_ols": vel_ols,
        "ann_sin": ann_sin,
        "ann_cos": ann_cos,
        "ann_amp": ann_amp,
        "ann_phase": ann_phase,
        "ann_peak": ann_peak,
        "rms_la": rms_la,
        "vel_se": vel_se,
        "cumulative": cumulative,
    }

    for name, a in arrays_strict.items():
        if int(a.size) != n:
            die(f"{name} size={a.size:,}, expected strict n={n:,}")

    if strict_ids.ndim != 1 or strict_ids.size != n:
        die(
            f"strict_point_ids shape={strict_ids.shape}, "
            f"expected ({n},)"
        )

    if strict_ids.size:
        if strict_ids.min() < 0 or strict_ids.max() >= point_type_full.size:
            die("strict_point_ids exceed PointPhaseStack domain")

    for name, a in {
        "point_type": point_type_full,
        "temporal_coherence": tc_full,
        "median_pair_coherence": pair_full,
        "glrt_support_K": shp_full,
        "estimator_code": est_full,
        "emi_eigenvalue": emi_full,
        "evd_eigenvalue": evd_full,
        "gamma_min_eigenvalue": gamma_full,
    }.items():
        if a.ndim != 1:
            die(f"{name} must be 1-D, got {a.shape}")
        if a.size != point_type_full.size:
            die(
                f"{name} size={a.size:,}, "
                f"point_type size={point_type_full.size:,}"
            )

    ptype_strict = np.asarray(
        point_type_full[strict_ids],
        dtype=np.uint8,
    )

    nps = int(np.count_nonzero(ptype_strict == 1))
    nds = int(np.count_nonzero(ptype_strict == 2))

    # Incidence QA.
    inc_deg_all = np.rad2deg(
        np.asarray(inc, dtype=np.float64)
    )
    if not np.all(np.isfinite(inc_deg_all)):
        die("incidence contains non-finite values")
    if np.any((inc_deg_all <= 0.0) | (inc_deg_all >= 90.0)):
        die("incidence outside physical 0..90 degree range")

    # Output tree.
    d_geom = out / "00_geometry"
    d_vel = out / "01_velocity"
    d_sea = out / "02_seasonal"
    d_cum = out / "03_cumulative"
    d_qa = out / "04_quality"
    d_man = out / "manifest"
    for d in (d_geom, d_vel, d_sea, d_cum, d_qa, d_man):
        d.mkdir(parents=True, exist_ok=True)

    nshard = int(math.ceil(n / args.max_points))

    print("=" * 100)
    print("NINGBO TP-LA -> SPLIT QGIS GEOPACKAGES")
    print("=" * 100)
    print("root                    :", root)
    print("output                  :", out)
    print("strict points           :", f"{n:,}")
    print("PS / DS                 :", f"{nps:,} / {nds:,}")
    print("acquisitions            :", len(dates))
    print("temporal reference      :", temporal_reference_date)
    print("geometric master        :", geometric_master_date)
    print("max points / file       :", f"{args.max_points:,}")
    print("shards / theme          :", nshard)
    print("themes                  :", 5)
    print("expected GPKG count     :", nshard * 5)
    print("vertical approximation  :", not args.no_vertical)
    print("=" * 100)

    rows_manifest = []
    t_all = time.perf_counter()

    for shard in range(nshard):
        start = shard * args.max_points
        stop = min(start + args.max_points, n)
        m = stop - start

        sid = np.arange(start, stop, dtype=np.int64)
        pid = np.asarray(strict_ids[start:stop], dtype=np.int64)

        lo = np.asarray(lon[start:stop], dtype=np.float64)
        la = np.asarray(lat[start:stop], dtype=np.float64)

        if not (
            np.all(np.isfinite(lo))
            and np.all(np.isfinite(la))
        ):
            die(f"non-finite lon/lat in shard {shard}")

        geometry = gpd.points_from_xy(lo, la)

        ptype = np.asarray(ptype_strict[start:stop], dtype=np.uint8)
        inc_deg = np.rad2deg(
            np.asarray(inc[start:stop], dtype=np.float64)
        ).astype(np.float32)

        cos_inc = np.cos(
            np.asarray(inc[start:stop], dtype=np.float64)
        )
        if np.any(cos_inc <= 0.0):
            die(f"invalid cos(incidence) in shard {shard}")

        tag = (
            f"part_{shard+1:03d}_"
            f"s{start:08d}_e{stop-1:08d}"
        )

        # --------------------------------------------------------------
        # 00 geometry
        # --------------------------------------------------------------
        f_geom = d_geom / f"geometry_{tag}.gpkg"
        df_geom = pd.DataFrame({
            "SID": sid,
            "PID": pid,
            "PTYPE": ptype,
            "RAD_ROW": np.asarray(row[start:stop], dtype=np.int32),
            "RAD_COL": np.asarray(col[start:stop], dtype=np.int32),
            "LON": lo,
            "LAT": la,
            "HGT_M": np.asarray(hgt[start:stop], dtype=np.float32),
            "INC_DEG": inc_deg,
        })
        s0 = time.perf_counter()
        act_geom = write_gpkg_atomic(
            gpd=gpd,
            pyogrio=pyogrio,
            frame=df_geom,
            geometry=geometry,
            path=f_geom,
            layer="geometry",
            overwrite_existing=args.overwrite,
        )
        del df_geom

        # --------------------------------------------------------------
        # 01 velocity
        # --------------------------------------------------------------
        vla = np.asarray(vel_la[start:stop], dtype=np.float32)
        vols = np.asarray(vel_ols[start:stop], dtype=np.float32)
        vse = np.asarray(vel_se[start:stop], dtype=np.float32)
        rr = np.asarray(rms_la[start:stop], dtype=np.float32)

        data_vel = {
            "SID": sid,
            "PID": pid,
            "PTYPE": ptype,
            "VEL_LA": vla,
            "VEL_OLS": vols,
            "VEL_SE": vse,
            "RMS_LA": rr,
        }

        if not args.no_vertical:
            vup = (
                vla.astype(np.float64) / cos_inc
            ).astype(np.float32)
            data_vel["VEL_VUP"] = vup
            data_vel["SUBS_DN"] = (-vup).astype(np.float32)

        f_vel = d_vel / f"velocity_{tag}.gpkg"
        df_vel = pd.DataFrame(data_vel)
        act_vel = write_gpkg_atomic(
            gpd=gpd,
            pyogrio=pyogrio,
            frame=df_vel,
            geometry=geometry,
            path=f_vel,
            layer="velocity",
            overwrite_existing=args.overwrite,
        )
        del df_vel, data_vel

        # --------------------------------------------------------------
        # 02 seasonal
        # --------------------------------------------------------------
        aa = np.asarray(ann_amp[start:stop], dtype=np.float32)

        data_sea = {
            "SID": sid,
            "PID": pid,
            "PTYPE": ptype,
            "ANN_AMP": aa,
            "ANN_SIN": np.asarray(
                ann_sin[start:stop], dtype=np.float32
            ),
            "ANN_COS": np.asarray(
                ann_cos[start:stop], dtype=np.float32
            ),
            "ANN_PHASE": np.asarray(
                ann_phase[start:stop], dtype=np.float32
            ),
            "PEAK_LAG_D": np.asarray(
                ann_peak[start:stop], dtype=np.float32
            ),
        }

        if not args.no_vertical:
            data_sea["ANN_VAMP"] = (
                aa.astype(np.float64) / cos_inc
            ).astype(np.float32)

        f_sea = d_sea / f"seasonal_{tag}.gpkg"
        df_sea = pd.DataFrame(data_sea)
        act_sea = write_gpkg_atomic(
            gpd=gpd,
            pyogrio=pyogrio,
            frame=df_sea,
            geometry=geometry,
            path=f_sea,
            layer="seasonal",
            overwrite_existing=args.overwrite,
        )
        del df_sea, data_sea

        # --------------------------------------------------------------
        # 03 cumulative
        # --------------------------------------------------------------
        cc = np.asarray(
            cumulative[start:stop],
            dtype=np.float32,
        )
        data_cum = {
            "SID": sid,
            "PID": pid,
            "PTYPE": ptype,
            "CUM_LOS": cc,
        }

        if not args.no_vertical:
            cup = (
                cc.astype(np.float64) / cos_inc
            ).astype(np.float32)
            data_cum["CUM_VUP"] = cup
            data_cum["CUM_SUB_DN"] = (-cup).astype(np.float32)

        f_cum = d_cum / f"cumulative_{tag}.gpkg"
        df_cum = pd.DataFrame(data_cum)
        act_cum = write_gpkg_atomic(
            gpd=gpd,
            pyogrio=pyogrio,
            frame=df_cum,
            geometry=geometry,
            path=f_cum,
            layer="cumulative",
            overwrite_existing=args.overwrite,
        )
        del df_cum, data_cum

        # --------------------------------------------------------------
        # 04 quality
        # --------------------------------------------------------------
        # Pull quality arrays from original PointPhaseStack with PID.
        qid = pid
        df_qa = pd.DataFrame({
            "SID": sid,
            "PID": pid,
            "PTYPE": ptype,
            "TC": np.asarray(
                tc_full[qid], dtype=np.float32
            ),
            "PAIR_COH": np.asarray(
                pair_full[qid], dtype=np.float32
            ),
            "SHP_K": np.asarray(
                shp_full[qid], dtype=np.int16
            ),
            "EST_CODE": np.asarray(
                est_full[qid], dtype=np.int16
            ),
            "EMI_EIG": np.asarray(
                emi_full[qid], dtype=np.float32
            ),
            "EVD_EIG": np.asarray(
                evd_full[qid], dtype=np.float32
            ),
            "GAMMA_MIN": np.asarray(
                gamma_full[qid], dtype=np.float32
            ),
            "VEL_SE": vse,
            "RMS_LA": rr,
        })

        f_qa = d_qa / f"quality_{tag}.gpkg"
        act_qa = write_gpkg_atomic(
            gpd=gpd,
            pyogrio=pyogrio,
            frame=df_qa,
            geometry=geometry,
            path=f_qa,
            layer="quality",
            overwrite_existing=args.overwrite,
        )
        del df_qa

        elapsed = time.perf_counter() - s0

        rec = {
            "shard": shard + 1,
            "start_sid": start,
            "end_sid": stop - 1,
            "points": m,
            "lon_min": float(np.min(lo)),
            "lon_max": float(np.max(lo)),
            "lat_min": float(np.min(la)),
            "lat_max": float(np.max(la)),
            "geometry": str(f_geom.relative_to(out)),
            "velocity": str(f_vel.relative_to(out)),
            "seasonal": str(f_sea.relative_to(out)),
            "cumulative": str(f_cum.relative_to(out)),
            "quality": str(f_qa.relative_to(out)),
            "geometry_gib": (
                human_gib(f_geom.stat().st_size)
                if f_geom.exists() else math.nan
            ),
            "velocity_gib": (
                human_gib(f_vel.stat().st_size)
                if f_vel.exists() else math.nan
            ),
            "seasonal_gib": (
                human_gib(f_sea.stat().st_size)
                if f_sea.exists() else math.nan
            ),
            "cumulative_gib": (
                human_gib(f_cum.stat().st_size)
                if f_cum.exists() else math.nan
            ),
            "quality_gib": (
                human_gib(f_qa.stat().st_size)
                if f_qa.exists() else math.nan
            ),
        }
        rows_manifest.append(rec)

        print(
            f"[GIS] shard {shard+1:03d}/{nshard:03d} "
            f"{start:,}:{stop:,} "
            f"({100.0*stop/n:6.2f}%) "
            f"geom={act_geom} vel={act_vel} "
            f"sea={act_sea} cum={act_cum} qa={act_qa} "
            f"{elapsed:.1f}s",
            flush=True,
        )

        del geometry, sid, pid, lo, la, ptype, inc_deg, cos_inc
        del vla, vols, vse, rr, aa, cc

    # ------------------------------------------------------------------
    # Manifests
    # ------------------------------------------------------------------
    shard_csv = d_man / "shard_index.csv"
    with shard_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows_manifest[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows_manifest)

    field_contract = {
        "crs": "EPSG:4326",
        "point_type": {
            "PTYPE": {
                "1": "PS",
                "2": "DS",
            }
        },
        "identity": {
            "SID": (
                "0-based row index in final strict TP-LA arrays; "
                "use this to join all final products."
            ),
            "PID": (
                "0-based original PointPhaseStack point id "
                "(strict_point_ids.npy)."
            ),
            "RAD_ROW": "0-based original RSLC azimuth row.",
            "RAD_COL": "0-based original RSLC range column.",
        },
        "geometry": {
            "LON": "longitude [deg]",
            "LAT": "latitude [deg]",
            "HGT_M": "sampled height [m]",
            "INC_DEG": "local incidence angle [deg]",
        },
        "velocity": {
            "VEL_LA": (
                "PRIMARY LOS secular velocity from linear+annual model "
                "[mm/yr], positive toward satellite."
            ),
            "VEL_OLS": (
                "ordinary linear-only LOS velocity [mm/yr], "
                "positive toward satellite; QC comparison."
            ),
            "VEL_SE": (
                "formal linear+annual secular-velocity standard error "
                "[mm/yr]; not total geodetic uncertainty."
            ),
            "RMS_LA": (
                "linear+annual residual RMS [mm]."
            ),
            "VEL_VUP": (
                "approximate vertical-up velocity = VEL_LA/cos(INC) "
                "[mm/yr]."
            ),
            "SUBS_DN": (
                "approximate subsidence-down velocity = -VEL_VUP "
                "[mm/yr], positive downward."
            ),
        },
        "seasonal": {
            "ANN_AMP": "annual LOS amplitude [mm]",
            "ANN_SIN": "annual sine coefficient [mm]",
            "ANN_COS": "annual cosine coefficient [mm]",
            "ANN_PHASE": (
                "annual phase [rad] for A*sin(2*pi*t + phase)"
            ),
            "PEAK_LAG_D": (
                "annual peak lag [days] after temporal reference "
                f"{temporal_reference_date}, modulo 365.25 d."
            ),
            "ANN_VAMP": (
                "approximate vertical annual amplitude "
                "= ANN_AMP/cos(INC) [mm]."
            ),
        },
        "cumulative": {
            "CUM_LOS": (
                f"LOS displacement {dates[0]} -> {dates[-1]} [mm], "
                "positive toward satellite."
            ),
            "CUM_VUP": (
                "approximate vertical-up cumulative displacement [mm]."
            ),
            "CUM_SUB_DN": (
                "approximate downward cumulative subsidence [mm], "
                "positive downward."
            ),
        },
        "quality": {
            "TC": (
                "DS temporal coherence from PointPhaseStack; "
                "PS values are NaN by construction."
            ),
            "PAIR_COH": (
                "DS median pair coherence; PS values are NaN."
            ),
            "SHP_K": (
                "DS statistical homogeneous pixel support count; "
                "PS value is -1."
            ),
            "EST_CODE": (
                "phase-linking estimator code; PS value is -1."
            ),
            "EMI_EIG": "EMI eigenvalue metadata; DS metric.",
            "EVD_EIG": "EVD eigenvalue metadata; DS metric.",
            "GAMMA_MIN": "minimum Gamma eigenvalue metadata; DS metric.",
            "VEL_SE": "same formal velocity SE as velocity theme.",
            "RMS_LA": "same linear+annual residual RMS as velocity theme.",
        },
        "vertical_warning": (
            "VEL_VUP, SUBS_DN, CUM_VUP, CUM_SUB_DN and ANN_VAMP are "
            "single-track approximations assuming negligible horizontal "
            "motion. Do not describe them as rigorous vertical deformation."
        ),
        "scientific_source": {
            "products_manifest": str(manifest_path),
            "status": product_manifest.get("status"),
            "temporal_reference_date": temporal_reference_date,
            "geometric_master_date": geometric_master_date,
            "first_acquisition": dates[0],
            "last_acquisition": dates[-1],
            "acquisitions": len(dates),
        },
    }
    atomic_json(d_man / "fields.json", field_contract)

    # ------------------------------------------------------------------
    # Tiny footprint/index GPKG for quickly locating all shards in QGIS.
    # ------------------------------------------------------------------
    idx_records = []
    idx_geom = []
    for r in rows_manifest:
        idx_records.append({
            "SHARD": int(r["shard"]),
            "SID0": int(r["start_sid"]),
            "SID1": int(r["end_sid"]),
            "NPTS": int(r["points"]),
            "VEL_FILE": r["velocity"],
            "SEA_FILE": r["seasonal"],
            "CUM_FILE": r["cumulative"],
            "QA_FILE": r["quality"],
        })
        idx_geom.append(
            box(
                r["lon_min"],
                r["lat_min"],
                r["lon_max"],
                r["lat_max"],
            )
        )

    idx_gdf = gpd.GeoDataFrame(
        pd.DataFrame(idx_records),
        geometry=idx_geom,
        crs="EPSG:4326",
    )
    idx_tmp = out / "shard_index.tmp.gpkg"
    idx_final = out / "shard_index.gpkg"
    if idx_tmp.exists():
        idx_tmp.unlink()
    pyogrio.write_dataframe(
        idx_gdf,
        idx_tmp,
        layer="shards",
        driver="GPKG",
        layer_options={"SPATIAL_INDEX": "YES"},
    )
    os.replace(idx_tmp, idx_final)

    total_bytes = 0
    for d in (d_geom, d_vel, d_sea, d_cum, d_qa):
        total_bytes += sum(
            p.stat().st_size for p in d.glob("*.gpkg")
        )

    readme = f"""NINGBO TP-LA QGIS EXPORT

Status
------
Final science source : PASS_TP_LA_PRODUCTION
Strict points        : {n:,}
PS / DS              : {nps:,} / {nds:,}
Acquisitions         : {len(dates)}
Date range           : {dates[0]} -> {dates[-1]}
Temporal reference   : {temporal_reference_date}
Geometric master     : {geometric_master_date}

Sharding
--------
Max points/file      : {args.max_points:,}
Shards/theme         : {nshard}
Themes               : 5
Expected GPKGs       : {nshard * 5}

Folders
-------
00_geometry  : point identity, radar row/col, height, incidence
01_velocity  : primary TP-LA velocity, OLS velocity, formal SE, RMS,
               plus optional single-track vertical approximation
02_seasonal  : annual amplitude/sine/cosine/phase/peak timing
03_cumulative: cumulative LOS and optional vertical approximation
04_quality   : DS Phase-Linking quality metadata + final RMS/SE

Open shard_index.gpkg first in QGIS. It contains shard footprints and the
corresponding file names.

Join key
--------
SID is the authoritative row id for final strict TP-LA arrays.
PID is the original PointPhaseStack id.

Sign convention
---------------
LOS positive = toward satellite.
VEL_VUP/CUM_VUP positive = upward.
SUBS_DN/CUM_SUB_DN positive = downward.

IMPORTANT
---------
Vertical values are single-track approximations:
  vertical_up ~= LOS / cos(incidence)
They assume negligible horizontal motion and are not rigorous vertical
decomposition products.

The 75-epoch time series is intentionally NOT duplicated into these GPKGs.
Keep:
  output/processing/final_los_tp_la/los_displacement_toward_satellite_mm.npy
as the canonical full-resolution time series. Exporting all 75 epochs with
geometry should be done separately by AOI/date groups to avoid hundreds of
gigabytes of redundant GIS data.

Field definitions:
  manifest/fields.json

Shard table:
  manifest/shard_index.csv
"""
    (d_man / "README_QGIS.txt").write_text(
        readme,
        encoding="utf-8",
    )

    elapsed_all = time.perf_counter() - t_all

    print()
    print("=" * 100)
    print("FINAL GIS EXPORT: PASS")
    print("=" * 100)
    print("output                  :", out)
    print("strict points           :", f"{n:,}")
    print("shards / theme          :", nshard)
    print("GPKG files              :", nshard * 5)
    print("total GPKG size         :", f"{human_gib(total_bytes):.2f} GiB")
    print("shard index             :", idx_final)
    print("field contract          :", d_man / "fields.json")
    print("readme                  :", d_man / "README_QGIS.txt")
    print("seconds                 :", f"{elapsed_all:.1f}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
