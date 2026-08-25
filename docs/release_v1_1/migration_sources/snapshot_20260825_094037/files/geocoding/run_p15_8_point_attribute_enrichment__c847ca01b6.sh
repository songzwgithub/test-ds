#!/bin/bash
set -euo pipefail


echo "================================================================================"
echo "P15-8 FINAL POINT ATTRIBUTE ENRICHMENT"
echo "================================================================================"


PY=/tmp/p15_8_point_attribute_enrichment.py


cat > ${PY} <<'PY'


from pathlib import Path
import json
import time
import shutil
import subprocess

import pandas as pd



# ==========================================================
# PATH
# ==========================================================

BASE = Path(
    "/home/ubuntu/Downloads/psds/output/processing"
)


INPUT = (
    BASE
    /
    "final_point_products"
    /
    "los_points.parquet"
)


OUT = (
    BASE
    /
    "final_point_products"
)


OUT.mkdir(
    parents=True,
    exist_ok=True
)


t0 = time.time()


print("Reading point products")

df = pd.read_parquet(
    INPUT
)


n = len(df)


print(
    "points:",
    n
)



# ==========================================================
# geometry
# ==========================================================

GEOM = (
    BASE
    /
    "gacos_geometry"
)


lon_file = (
    GEOM
    /
    "longitude_deg.npy"
)

lat_file = (
    GEOM
    /
    "latitude_deg.npy"
)

inc_file = (
    GEOM
    /
    "incidence_gamma_compatible_fast_rad.npy"
)



# numpy only here
import numpy as np


lon = np.load(
    lon_file
)

lat = np.load(
    lat_file
)

inc = np.load(
    inc_file
)


if len(lon) != n:
    raise RuntimeError(
        "longitude size mismatch"
    )


df["longitude_deg"] = lon
df["latitude_deg"] = lat


df["incidence_deg"] = (
    np.degrees(inc)
)



# ==========================================================
# LOS products
# ==========================================================

LOS = (
    BASE
    /
    "final_los_products"
)


def append_array(
    column,
    filename
):

    arr = np.load(
        LOS / filename
    )

    if len(arr) != n:
        raise RuntimeError(
            f"{column} size mismatch"
        )

    df[column] = arr



append_array(
    "los_velocity_mm_yr",
    "los_velocity_toward_satellite_mm_per_year.npy"
)


append_array(
    "los_cumulative_mm",
    "los_cumulative_toward_satellite_mm.npy"
)


append_array(
    "linear_residual_rms_mm",
    "linear_residual_rms_mm.npy"
)


append_array(
    "velocity_se_mm_yr",
    "velocity_slope_standard_error_mm_per_year.npy"
)



# ==========================================================
# time series statistics
# ==========================================================

TS = (
    BASE
    /
    "final_los_timeseries"
)


phase_file = (
    TS
    /
    "acquisition_phase_final_rad.npy"
)


phase = np.load(
    phase_file,
    mmap_mode="r"
)


if phase.shape[0] != n:
    raise RuntimeError(
        "phase point dimension mismatch"
    )


df["n_epochs"] = np.sum(
    np.isfinite(phase),
    axis=1
)


df["phase_std_rad"] = np.nanstd(
    phase,
    axis=1
)



# ==========================================================
# point id
# ==========================================================

if "point_id" not in df.columns:

    df.insert(
        0,
        "point_id",
        range(n)
    )



# ==========================================================
# save parquet
# ==========================================================

OUT_PARQUET = (
    OUT
    /
    "psds_final_points.parquet"
)


df.to_parquet(
    OUT_PARQUET,
    index=False,
    compression="zstd"
)


print(
    "PARQUET:",
    OUT_PARQUET
)



# ==========================================================
# save csv
# ==========================================================

OUT_CSV = (
    OUT
    /
    "psds_final_points.csv"
)


df.to_csv(
    OUT_CSV,
    index=False
)


print(
    "CSV:",
    OUT_CSV
)



# ==========================================================
# GPKG
# ==========================================================

OUT_GPKG = (
    OUT
    /
    "psds_final_points.gpkg"
)


ogr = shutil.which(
    "ogr2ogr"
)


gpkg_ok = False


if ogr:

    cmd = [

        ogr,

        "-f",
        "GPKG",

        str(OUT_GPKG),

        str(OUT_CSV),

        "-oo",
        "X_POSSIBLE_NAMES=longitude_deg",

        "-oo",
        "Y_POSSIBLE_NAMES=latitude_deg",

        "-a_srs",
        "EPSG:4326"

    ]


    try:

        subprocess.run(
            cmd,
            check=True
        )

        gpkg_ok=True

        print(
            "GPKG:",
            OUT_GPKG
        )


    except Exception as e:

        print(
            "GPKG failed:",
            e
        )


else:

    print(
        "ogr2ogr not found"
    )



# ==========================================================
# metadata
# ==========================================================


meta = {

    "stage":
    "P15-8_FINAL_POINT_ATTRIBUTE_ENRICHMENT",

    "points":
    int(n),

    "crs":
    "EPSG:4326",

    "geometry":
    "original PS/DS point geometry",

    "products":
    {

        "parquet":
        str(OUT_PARQUET),

        "csv":
        str(OUT_CSV),

        "gpkg":
        str(OUT_GPKG)
        if gpkg_ok
        else None

    },

    "columns":
    list(df.columns),

    "elapsed_seconds":
    time.time()-t0

}


with open(
    OUT /
    "p15_8_point_product_metadata.json",
    "w"
) as f:

    json.dump(
        meta,
        f,
        indent=2
    )



print()
print("="*80)
print(
"P15-8 FINAL RESULT: PASS_FINAL_POINT_ATTRIBUTE_PRODUCT"
)
print("="*80)



PY


echo
echo "------------------------------------------------------------"
echo "Python syntax check"
echo "------------------------------------------------------------"


python -m py_compile ${PY}


echo "SYNTAX PASS"



echo
echo "------------------------------------------------------------"
echo "RUN P15-8"
echo "------------------------------------------------------------"


python ${PY}


