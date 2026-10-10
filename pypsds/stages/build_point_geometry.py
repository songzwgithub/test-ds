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
    resolve_geometry_inputs,
    resolve_height_raster,
    resolve_radar_look_factors,
    sample_full_resolution_point_geometry,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def geographic_collapse_qa(
    longitude_deg,
    latitude_deg,
    *,
    max_sample: int = 1_000_000,
) -> dict:
    """
    Detect accidental many-to-one geolocation of full-resolution points.

    The historical 10x2-nearest-cell bug produced a collapse ratio near 11
    for Ningbo. A deterministic sample keeps this QA cheap on 20M+ points.
    """

    lon = np.asarray(
        longitude_deg,
        dtype=np.float64,
    )
    lat = np.asarray(
        latitude_deg,
        dtype=np.float64,
    )

    if lon.shape != lat.shape or lon.ndim != 1:
        raise RuntimeError(
            "longitude/latitude shape mismatch."
        )

    n = int(lon.size)

    if n == 0:
        return {
            "sample_points": 0,
            "unique_lonlat": 0,
            "collapse_ratio": 1.0,
        }

    ns = min(
        n,
        int(max_sample),
    )

    if ns == n:
        idx = np.arange(
            n,
            dtype=np.int64,
        )
    else:
        idx = np.linspace(
            0,
            n - 1,
            ns,
            dtype=np.int64,
        )

    pairs = np.empty(
        ns,
        dtype=[
            ("lon", "<f8"),
            ("lat", "<f8"),
        ],
    )
    pairs["lon"] = lon[idx]
    pairs["lat"] = lat[idx]

    unique = int(
        np.unique(
            pairs
        ).size
    )

    ratio = (
        float(ns) / float(unique)
        if unique
        else float("inf")
    )

    return {
        "sample_points":
            int(ns),

        "unique_lonlat":
            unique,

        "collapse_ratio":
            ratio,
    }


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

    range_looks, azimuth_looks = (
        resolve_radar_look_factors(
            geometry.geometry_par,
            geometry.reference_rslc_par,
        )
    )

    proc = (
        Path(paths.output_dir)
        / "processing"
    )

    strict_ids_path = (
        proc
        / "final_unwrap"
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

    geo = sample_full_resolution_point_geometry(
        rows=rows,
        cols=cols,
        geometry=geometry,
        height_raster=height_raster,
        work_dir=out,
    )

    height = geo.height_m

    incidence = compute_incidence_rad(
        longitude_deg=geo.longitude_deg,
        latitude_deg=geo.latitude_deg,
        height_m=height,
        radar_row=rows,
        reference_rslc_par=
            geometry.reference_rslc_par,
    )

    if not geo.valid_mask.all():
        bad = int(
            np.count_nonzero(
                ~geo.valid_mask
            )
        )
        raise RuntimeError(
            "Invalid longitude/latitude in strict domain: "
            f"{bad:,}/{n:,}."
        )

    collapse = geographic_collapse_qa(
        geo.longitude_deg,
        geo.latitude_deg,
    )

    # Historical Ningbo failure:
    # 24,537,124 raw points -> 2,224,092 lon/lat positions,
    # sampled collapse ratio ~11 for a 10x2 MLI geometry.
    if (
        collapse["sample_points"] >= 10_000
        and
        collapse["collapse_ratio"] > 1.05
    ):
        raise RuntimeError(
            "Full-resolution geometry collapse detected: "
            f"sample={collapse['sample_points']:,}, "
            f"unique={collapse['unique_lonlat']:,}, "
            f"ratio={collapse['collapse_ratio']:.6f}. "
            "Point geometry must not be quantized to the "
            "multilook raster lattice."
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
        np.asarray(
            geo.longitude_deg,
            dtype=np.float64,
        ),
    )
    np.save(
        out / "latitude_deg.npy",
        np.asarray(
            geo.latitude_deg,
            dtype=np.float64,
        ),
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
            "pyPSDS-GAMMA-v1.2-fullres-point-geometry",

        "point_count":
            n,

        "reference_date":
            geometry.reference_date,

        "sampling": {
            "method":
                "masked_joint_bilinear_multilook_raster_at_singlelook_points",

            "range_looks":
                int(range_looks),

            "azimuth_looks":
                int(azimuth_looks),

            "continuous_coordinate":
                "u=raw_col/range_looks; "
                "v=raw_row/azimuth_looks",

            "edge_policy":
                "linear_extrapolation_from_last_two_mli_centers",

            "downstream_geometry_dtype":
                "float64",

            "validity_policy":
                "joint lon/lat/height validity; invalid MLI nodes excluded "
                "and valid bilinear weights renormalized",

            "support_counts": {
                "full": int(geo.full_support_count),
                "partial_2_or_3": int(geo.partial_support_count),
                "single": int(geo.single_support_count),
                "zero": int(geo.zero_support_count),
            },
        },

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

        "quality": {
            "invalid_lonlat_points":
                int(
                    np.count_nonzero(
                        ~geo.valid_mask
                    )
                ),

            "geographic_collapse_sample_points":
                int(
                    collapse[
                        "sample_points"
                    ]
                ),

            "geographic_collapse_unique_lonlat":
                int(
                    collapse[
                        "unique_lonlat"
                    ]
                ),

            "geographic_collapse_ratio":
                float(
                    collapse[
                        "collapse_ratio"
                    ]
                ),
        },

        "statistics": {
            "radar_row_min":
                int(rows.min()) if n else None,

            "radar_row_max":
                int(rows.max()) if n else None,

            "radar_col_min":
                int(cols.min()) if n else None,

            "radar_col_max":
                int(cols.max()) if n else None,

            "longitude_min":
                float(
                    geo.longitude_deg.min()
                ) if n else None,

            "longitude_max":
                float(
                    geo.longitude_deg.max()
                ) if n else None,

            "latitude_min":
                float(
                    geo.latitude_deg.min()
                ) if n else None,

            "latitude_max":
                float(
                    geo.latitude_deg.max()
                ) if n else None,

            "height_min_m":
                float(
                    height.min()
                ) if n else None,

            "height_median_m":
                float(
                    np.median(height)
                ) if n else None,

            "height_max_m":
                float(
                    height.max()
                ) if n else None,

            "incidence_min_rad":
                float(
                    incidence.min()
                ) if n else None,

            "incidence_median_rad":
                float(
                    np.median(incidence)
                ) if n else None,

            "incidence_max_rad":
                float(
                    incidence.max()
                ) if n else None,
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
    print("strict points       :", f"{n:,}")
    print("reference date      :", geometry.reference_date)
    print(
        "geometry looks      :",
        f"{range_looks} range x {azimuth_looks} azimuth",
    )
    print(
        "sampling            :",
        "bilinear MLI -> single-look points",
    )
    print(
        "geometry support    :",
        f"full={geo.full_support_count:,} "
        f"partial={geo.partial_support_count:,} "
        f"single={geo.single_support_count:,} "
        f"zero={geo.zero_support_count:,}",
    )
    print(
        "collapse sample     :",
        f"{collapse['sample_points']:,}",
    )
    print(
        "collapse unique     :",
        f"{collapse['unique_lonlat']:,}",
    )
    print(
        "collapse ratio      :",
        f"{collapse['collapse_ratio']:.6f}",
    )
    print("height raster       :", height_raster)
    print("output              :", out)
    print("manifest            :", manifest_path)
    print("=" * 88)
    print("POINT GEOMETRY STATUS: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
