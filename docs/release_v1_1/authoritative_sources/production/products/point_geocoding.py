#!/usr/bin/env python3

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


PSDS = Path(
    "/home/ubuntu/Downloads/psds"
)

PROC = (
    PSDS /
    "output/processing"
)

GEOM = (
    PROC /
    "gacos_geometry"
)

PRODUCT = (
    PROC /
    "final_los_products"
)

OUT = (
    PROC /
    "final_point_geocoding"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# input
# ------------------------------------------------------------

lon_file = GEOM / "longitude_deg.npy"
lat_file = GEOM / "latitude_deg.npy"

vel_file = (
    PRODUCT /
    "los_velocity_toward_satellite_mm_per_year.npy"
)

cum_file = (
    PRODUCT /
    "los_cumulative_toward_satellite_mm.npy"
)

rms_file = (
    PRODUCT /
    "linear_residual_rms_mm.npy"
)

se_file = (
    PRODUCT /
    "velocity_slope_standard_error_mm_per_year.npy"
)


t0=time.time()


# ------------------------------------------------------------
# load
# ------------------------------------------------------------

lon=np.load(
    lon_file,
    mmap_mode="r"
)

lat=np.load(
    lat_file,
    mmap_mode="r"
)

vel=np.load(
    vel_file,
    mmap_mode="r"
)

cum=np.load(
    cum_file,
    mmap_mode="r"
)

rms=np.load(
    rms_file,
    mmap_mode="r"
)

se=np.load(
    se_file,
    mmap_mode="r"
)


n=len(lon)


assert n==len(lat)
assert n==len(vel)


# ------------------------------------------------------------
# valid
# ------------------------------------------------------------

valid = (
    np.isfinite(lon)
    &
    np.isfinite(lat)
    &
    np.isfinite(vel)
)

idx=np.where(valid)[0]


print("="*80)
print("P15-7B2 POINT GEOCODING")
print("="*80)

print(
    "total points:",
    n
)

print(
    "valid points:",
    len(idx),
    f"({100*len(idx)/n:.6f}%)"
)


# ------------------------------------------------------------
# dataframe
# ------------------------------------------------------------

df=pd.DataFrame(
    {
        "point_id":
            idx.astype(np.int64),

        "longitude_deg":
            lon[idx],

        "latitude_deg":
            lat[idx],

        "los_velocity_mm_yr":
            vel[idx],

        "los_cumulative_mm":
            cum[idx],

        "residual_rms_mm":
            rms[idx],

        "velocity_se_mm_yr":
            se[idx],
    }
)


# ------------------------------------------------------------
# csv
# ------------------------------------------------------------

csv_out=(
    OUT /
    "los_point_products.csv"
)


df.to_csv(
    csv_out,
    index=False
)


print(
    "CSV:",
    csv_out
)



# ------------------------------------------------------------
# geopackage
# ------------------------------------------------------------

gpkg_ok=False

try:

    import geopandas as gpd
    from shapely.geometry import Point


    geom=[
        Point(x,y)
        for x,y in zip(
            df.longitude_deg,
            df.latitude_deg
        )
    ]


    gdf=gpd.GeoDataFrame(
        df,
        geometry=geom,
        crs="EPSG:4326"
    )


    gpkg_out=(
        OUT /
        "los_point_products.gpkg"
    )


    gdf.to_file(
        gpkg_out,
        layer="LOS_velocity",
        driver="GPKG"
    )


    gpkg_ok=True


    print(
        "GPKG:",
        gpkg_out
    )


except Exception as e:

    print(
        "GeoPackage skipped:",
        e
    )


# ------------------------------------------------------------
# manifest
# ------------------------------------------------------------

manifest={

    "stage":
        "P15-7B2_POINT_GEOCODING",

    "points":
        int(n),

    "valid_points":
        int(len(idx)),

    "crs":
        "EPSG:4326",

    "geometry":
        "point_based",

    "gamma_geocode":
        False,

    "lookup_table":
        False,

    "rasterization":
        False,

    "velocity_sign":
        "positive_toward_satellite",

    "gpkg_created":
        gpkg_ok,

    "elapsed_seconds":
        time.time()-t0
}


with open(
    OUT /
    "p15_7b2_point_geocoding_manifest.json",
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


print()
print(
    "elapsed:",
    time.time()-t0,
    "s"
)

print("="*80)
print(
    "P15-7B2 FINAL RESULT: PASS_POINT_GEOCODING"
)
print("="*80)

