from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def los_geometry_eu(incidence_rad, heading_deg):
    """
    Ground-to-satellite LOS unit vector reduced to East/Up.

    heading is satellite flight azimuth clockwise from north.
    Right-looking radar and North=0 are assumed.
    """
    inc = np.asarray(incidence_rad, dtype=np.float64)
    h = np.deg2rad(float(heading_deg))
    return -np.sin(inc) * np.cos(h), np.cos(inc)


def solve_east_up(
    los_a,
    los_d,
    inc_a,
    inc_d,
    heading_a,
    heading_d,
    sigma_a=None,
    sigma_d=None,
):
    los_a = np.asarray(los_a, dtype=np.float64)
    los_d = np.asarray(los_d, dtype=np.float64)
    ue_a, uu_a = los_geometry_eu(inc_a, heading_a)
    ue_d, uu_d = los_geometry_eu(inc_d, heading_d)

    det = ue_a * uu_d - ue_d * uu_a
    if np.any(np.abs(det) < 1.0e-6):
        raise ValueError("ascending/descending geometry is ill-conditioned")

    east = (los_a * uu_d - los_d * uu_a) / det
    up = (ue_a * los_d - ue_d * los_a) / det
    out = {"east": east, "up": up, "determinant": det}

    if sigma_a is not None and sigma_d is not None:
        sa = np.asarray(sigma_a, dtype=np.float64)
        sd = np.asarray(sigma_d, dtype=np.float64)
        east_var = (uu_d / det) ** 2 * sa**2 + (uu_a / det) ** 2 * sd**2
        up_var = (ue_d / det) ** 2 * sa**2 + (ue_a / det) ** 2 * sd**2
        out["east_sigma"] = np.sqrt(np.maximum(east_var, 0.0))
        out["up_sigma"] = np.sqrt(np.maximum(up_var, 0.0))
    return out


def _xy(lon, lat, lon0, lat0):
    R = 6371008.8
    x = np.deg2rad(lon - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return np.column_stack((x, y))


def _load_track(output_root):
    output_root = Path(output_root)
    p = output_root / "products"
    g = output_root / "processing" / "point_geometry"
    vel = np.load(p / "los_velocity_toward_satellite_mm_per_year.npy")
    unc_path = p / "velocity_formal_uncertainty_mm_per_year.npy"
    unc = np.load(
        unc_path
        if unc_path.is_file()
        else p / "velocity_slope_standard_error_mm_per_year.npy"
    )
    lon = np.load(g / "longitude_deg.npy")
    lat = np.load(g / "latitude_deg.npy")
    inc = np.load(g / "incidence_rad.npy")
    return lon, lat, inc, vel, unc


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Matched ascending/descending LOS velocity decomposition into "
            "East and Up under North=0."
        )
    )
    ap.add_argument("--ascending-output", required=True)
    ap.add_argument("--descending-output", required=True)
    ap.add_argument("--ascending-heading-deg", required=True, type=float)
    ap.add_argument("--descending-heading-deg", required=True, type=float)
    ap.add_argument("--max-distance-m", type=float, default=100.0)
    ap.add_argument("--output-dir", default="decomposition")
    args = ap.parse_args(argv)

    ao = Path(args.ascending_output).expanduser().resolve()
    do = Path(args.descending_output).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    alon, alat, ainc, avel, aunc = _load_track(ao)
    dlon, dlat, dinc, dvel, dunc = _load_track(do)

    lon0 = float(np.median(np.r_[alon, dlon]))
    lat0 = float(np.median(np.r_[alat, dlat]))
    axy = _xy(alon, alat, lon0, lat0)
    dxy = _xy(dlon, dlat, lon0, lat0)

    dist, did = cKDTree(dxy).query(axy, k=1)
    keep = np.isfinite(dist) & (dist <= args.max_distance_m)
    ai = np.flatnonzero(keep)
    di = did[keep].astype(np.int64)
    if ai.size == 0:
        raise RuntimeError("No ascending/descending points matched")

    sol = solve_east_up(
        avel[ai], dvel[di],
        ainc[ai], dinc[di],
        args.ascending_heading_deg,
        args.descending_heading_deg,
        aunc[ai], dunc[di],
    )
    lon = 0.5 * (alon[ai] + dlon[di])
    lat = 0.5 * (alat[ai] + dlat[di])

    np.savez_compressed(
        out / "east_up_velocity.npz",
        longitude_deg=lon,
        latitude_deg=lat,
        ascending_index=ai,
        descending_index=di,
        match_distance_m=dist[keep],
        east_velocity_mm_per_year=sol["east"],
        up_velocity_mm_per_year=sol["up"],
        subsidence_down_mm_per_year=-sol["up"],
        east_standard_error_mm_per_year=sol["east_sigma"],
        up_standard_error_mm_per_year=sol["up_sigma"],
    )

    with (out / "east_up_velocity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "longitude_deg", "latitude_deg", "match_distance_m",
                "east_velocity_mm_per_year", "up_velocity_mm_per_year",
                "subsidence_down_mm_per_year",
                "east_standard_error_mm_per_year",
                "up_standard_error_mm_per_year",
            ]
        )
        dd = dist[keep]
        for k in range(ai.size):
            w.writerow(
                [
                    lon[k], lat[k], dd[k],
                    sol["east"][k], sol["up"][k], -sol["up"][k],
                    sol["east_sigma"][k], sol["up_sigma"][k],
                ]
            )

    manifest = {
        "status": "PASS_EAST_UP_DECOMPOSITION",
        "version": "1.3.0",
        "matched_points": int(ai.size),
        "max_distance_m": float(args.max_distance_m),
        "ascending_heading_deg": float(args.ascending_heading_deg),
        "descending_heading_deg": float(args.descending_heading_deg),
        "model": "two-track East/Up velocity decomposition with North=0",
        "los_sign": "positive_toward_satellite",
        "up_sign": "positive_up",
        "subsidence_sign": "positive_down = -up",
    }
    (out / "decomposition_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print("matched points :", ai.size)
    print("output         :", out)
    print("DECOMPOSITION STATUS: PASS")


if __name__ == "__main__":
    main()
