from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pypsds.config import load_config
from pypsds.project import resolve_project_paths
from pypsds.geometry import (
    compute_incidence_rad,
    geolocate_points,
    resolve_data2pt,
    resolve_geometry_inputs,
    resolve_height_raster,
    sample_height_m,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full-resolution strict-point geometry."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg, config_path = load_config(
        Path(args.config)
    )
    paths = resolve_project_paths(
        cfg,
        config_path,
    )

    geometry = resolve_geometry_inputs(
        cfg,
        paths,
    )
    height_raster = resolve_height_raster(
        cfg,
        paths,
        geometry,
    )
    data2pt = resolve_data2pt()

    proc = (
        Path(paths.output_dir)
        / "processing"
    )

    strict_ids_path = (
        proc
        / "network_inversion"
        / "strict_point_ids.npy"
    )
    rows_path = (
        proc
        / "point_phase_stack"
        / "rows.npy"
    )
    cols_path = (
        proc
        / "point_phase_stack"
        / "cols.npy"
    )

    for path in (
        strict_ids_path,
        rows_path,
        cols_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    strict_ids = np.load(
        strict_ids_path,
        mmap_mode="r",
    )
    all_rows = np.load(
        rows_path,
        mmap_mode="r",
    )
    all_cols = np.load(
        cols_path,
        mmap_mode="r",
    )

    if strict_ids.ndim != 1:
        raise RuntimeError(
            "strict_point_ids.npy must be 1-D."
        )

    if all_rows.shape != all_cols.shape:
        raise RuntimeError(
            "point-stack rows/cols shape mismatch."
        )

    if (
        strict_ids.size
        and
        (
            strict_ids.min() < 0
            or
            strict_ids.max() >= all_rows.size
        )
    ):
        raise RuntimeError(
            "strict point IDs exceed point-stack domain."
        )

    rows = np.asarray(
        all_rows[strict_ids],
        dtype=np.int32,
    )
    cols = np.asarray(
        all_cols[strict_ids],
        dtype=np.int32,
    )

    n = int(strict_ids.size)

    out = (
        proc
        / "point_geometry"
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    geo = geolocate_points(
        rows=rows,
        cols=cols,
        geometry=geometry,
        work_dir=out,
        data2pt=data2pt,
    )

    height_gamma = (
        out
        / "height_m.gamma_pt"
    )

    height = sample_height_m(
        height_raster=height_raster,
        geometry=geometry,
        point_list=geo.point_list,
        output_path=height_gamma,
        expected_count=n,
        data2pt=data2pt,
    )

    incidence = compute_incidence_rad(
        longitude_deg=geo.longitude_deg,
        latitude_deg=geo.latitude_deg,
        height_m=height,
        radar_row=rows,
        reference_rslc_par=
            geometry.reference_rslc_par,
    )

    if not geo.valid_mask.all():
        raise RuntimeError(
            "Invalid longitude/latitude in strict domain."
        )

    np.save(
        out / "radar_row.npy",
        rows,
    )
    np.save(
        out / "radar_col.npy",
        cols,
    )
    np.save(
        out / "longitude_deg.npy",
        geo.longitude_deg,
    )
    np.save(
        out / "latitude_deg.npy",
        geo.latitude_deg,
    )
    np.save(
        out / "height_m.npy",
        np.asarray(
            height,
            dtype=np.float64,
        ),
    )
    np.save(
        out / "incidence_rad.npy",
        incidence,
    )

    manifest = {
        "contract":
            "pyPSDS-GAMMA-v1.1-point-geometry",

        "point_count":
            n,

        "reference_date":
            geometry.reference_date,

        "inputs": {
            "strict_point_ids":
                str(strict_ids_path),

            "rows":
                str(rows_path),

            "cols":
                str(cols_path),

            "reference_rslc_par":
                str(
                    geometry.reference_rslc_par
                ),

            "geometry_par":
                str(
                    geometry.geometry_par
                ),

            "longitude_raster":
                str(
                    geometry.longitude_raster
                ),

            "latitude_raster":
                str(
                    geometry.latitude_raster
                ),

            "height_raster":
                str(
                    height_raster
                ),

            "data2pt":
                str(
                    data2pt
                ),
        },

        "outputs": {
            "radar_row":
                "radar_row.npy",

            "radar_col":
                "radar_col.npy",

            "longitude":
                "longitude_deg.npy",

            "latitude":
                "latitude_deg.npy",

            "height":
                "height_m.npy",

            "incidence":
                "incidence_rad.npy",
        },

        "statistics": {
            "longitude_min":
                float(
                    geo.longitude_deg.min()
                ),

            "longitude_max":
                float(
                    geo.longitude_deg.max()
                ),

            "latitude_min":
                float(
                    geo.latitude_deg.min()
                ),

            "latitude_max":
                float(
                    geo.latitude_deg.max()
                ),

            "height_min_m":
                float(
                    height.min()
                ),

            "height_median_m":
                float(
                    np.median(height)
                ),

            "height_max_m":
                float(
                    height.max()
                ),

            "incidence_min_rad":
                float(
                    incidence.min()
                ),

            "incidence_median_rad":
                float(
                    np.median(incidence)
                ),

            "incidence_max_rad":
                float(
                    incidence.max()
                ),
        },
    }

    manifest_path = (
        out
        / "point_geometry_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("POINT GEOMETRY")
    print("=" * 88)
    print("strict points  :", n)
    print("reference date :", geometry.reference_date)
    print("height raster  :", height_raster)
    print("data2pt        :", data2pt)
    print("output         :", out)
    print("manifest       :", manifest_path)
    print("=" * 88)
    print("POINT GEOMETRY STATUS: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
