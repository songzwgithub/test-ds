#!/bin/bash
set -euo pipefail

echo "================================================================================"
echo "P15-7C POINT PRODUCT PACKAGING"
echo "================================================================================"


PY=/tmp/p15_7c_point_packaging.py


cat > ${PY} <<'PYCODE'

from pathlib import Path
import json
import time
import shutil
import subprocess

import numpy as np
import pandas as pd


# ==========================================================
# PATH
# ==========================================================

BASE = Path(
    "/home/ubuntu/Downloads/psds/output/processing"
)


INPUT = (
    BASE /
    "final_point_geocoding" /
    "los_point_products.csv"
)


OUT = (
    BASE /
    "final_point_products"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


t0=time.time()


# ==========================================================
# LOAD
# ==========================================================

print("Reading point products")

df = pd.read_csv(
    INPUT
)


n=len(df)


print(
    "points:",
    n
)



# ==========================================================
# CSV COPY
# ==========================================================

csv_out = (
    OUT /
    "los_points.csv"
)


df.to_csv(
    csv_out,
    index=False
)


print(
    "CSV:",
    csv_out
)



# ==========================================================
# PARQUET
# ==========================================================

parquet_out = (
    OUT /
    "los_points.parquet"
)


try:

    df.to_parquet(
        parquet_out,
        index=False,
        compression="zstd"
    )

    parquet_status=True

    print(
        "PARQUET:",
        parquet_out
    )


except Exception as e:

    parquet_status=False

    print(
        "PARQUET skipped:",
        e
    )



# ==========================================================
# GPKG using ogr2ogr
# ==========================================================

gpkg_out = (
    OUT /
    "los_points.gpkg"
)


ogr = shutil.which(
    "ogr2ogr"
)


gpkg_status=False


if ogr:


    cmd=[

        ogr,

        "-f",
        "GPKG",

        str(gpkg_out),

        str(csv_out),

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
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        gpkg_status=True

        print(
            "GPKG:",
            gpkg_out
        )


    except Exception as e:

        print(
            "ogr2ogr failed:",
            e
        )


else:

    print(
        "ogr2ogr not found"
    )



# ==========================================================
# METADATA
# ==========================================================


meta={

    "stage":
        "P15-7C_POINT_PRODUCT_PACKAGING",

    "points":
        int(n),

    "crs":
        "EPSG:4326",

    "geometry":
        "original PS/DS point geometry",

    "rasterization":
        False,

    "gamma_geocode":
        False,

    "lookup_table":
        False,

    "products":
    {

        "csv":
            str(csv_out),

        "parquet":
            str(parquet_out)
            if parquet_status
            else None,

        "gpkg":
            str(gpkg_out)
            if gpkg_status
            else None

    },


    "elapsed_seconds":
        time.time()-t0

}



with open(
    OUT /
    "p15_7c_point_product_manifest.json",
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
"P15-7C FINAL RESULT: PASS_POINT_PRODUCT_PACKAGING"
)
print("="*80)


PYCODE



echo
echo "------------------------------------------------------------"
echo "Python syntax check"
echo "------------------------------------------------------------"


python -m py_compile ${PY}

echo "SYNTAX PASS"



echo
echo "------------------------------------------------------------"
echo "RUN P15-7C"
echo "------------------------------------------------------------"


python ${PY}


