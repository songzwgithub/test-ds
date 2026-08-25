#!/bin/bash
set -euo pipefail


echo "================================================================================"
echo "P15-7C FAST BLOCK COG RASTERIZER"
echo "================================================================================"


PY=/tmp/p15_7c_cog_rasterizer.py


cat > ${PY} <<'PY'


from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

import rasterio
from rasterio.transform import from_origin


# ==========================================================
# PATH
# ==========================================================

BASE=Path(
"/home/ubuntu/Downloads/psds/output/processing"
)


INPUT=(
BASE/
"final_point_geocoding/"
"los_point_products.csv"
)


OUT=(
BASE/
"final_cog_products"
)

OUT.mkdir(
parents=True,
exist_ok=True
)


# ==========================================================
# CONFIG
# ==========================================================

PIXEL=50.0

NODATA=-9999.0


t0=time.time()


# ==========================================================
# LOAD
# ==========================================================


df=pd.read_csv(
INPUT
)


lon=df.longitude_deg.values
lat=df.latitude_deg.values



products={

"los_velocity_mm_yr":
    df.los_velocity_mm_yr.values.astype(np.float32),

"los_cumulative_mm":
    df.los_cumulative_mm.values.astype(np.float32),

"velocity_se_mm_yr":
    df.velocity_se_mm_yr.values.astype(np.float32),

"residual_rms_mm":
    df.residual_rms_mm.values.astype(np.float32)

}



# ==========================================================
# GRID
# ==========================================================


xmin=np.floor(lon.min()/PIXEL)*PIXEL
xmax=np.ceil(lon.max()/PIXEL)*PIXEL

ymin=np.floor(lat.min()/PIXEL)*PIXEL
ymax=np.ceil(lat.max()/PIXEL)*PIXEL


# degree conversion

lat0=np.mean(lat)

dx=PIXEL/(111320*np.cos(np.deg2rad(lat0)))
dy=PIXEL/111320



width=int(
np.ceil(
(xmax-xmin)/(dx)
)
)


height=int(
np.ceil(
(ymax-ymin)/(dy)
)
)



transform=from_origin(
xmin,
ymax,
dx,
dy
)



print("grid")
print(
width,
height
)



# ==========================================================
# INDEX
# ==========================================================


col=((lon-xmin)/dx).astype(np.int32)
row=((ymax-lat)/dy).astype(np.int32)



valid=(
(row>=0)
&
(row<height)
&
(col>=0)
&
(col<width)
)



row=row[valid]
col=col[valid]


print(
"valid:",
valid.sum()
)



# ==========================================================
# WRITE
# ==========================================================


for name,data in products.items():

    print(
        "writing:",
        name
    )


    arr=np.full(
        (height,width),
        NODATA,
        dtype=np.float32
    )


    count=np.zeros(
        (height,width),
        dtype=np.uint32
    )


    value=data[valid]


    np.add.at(
        arr,
        (row,col),
        value
    )


    np.add.at(
        count,
        (row,col),
        1
    )


    m=count>0

    arr[m]/=count[m]


    outfile=(
        OUT/
        f"{name}.tif"
    )


    with rasterio.open(

        outfile,

        "w",

        driver="GTiff",

        height=height,

        width=width,

        count=1,

        dtype="float32",

        crs="EPSG:4326",

        transform=transform,

        nodata=NODATA,

        compress="DEFLATE",

        tiled=True

    ) as dst:

        dst.write(
            arr,
            1
        )


# ==========================================================
# MANIFEST
# ==========================================================


manifest={

"stage":
"P15-7C_FAST_BLOCK_COG",

"points":
int(len(df)),

"pixel_meter":
PIXEL,

"method":
"point_block_median",

"gamma_geocode":
False,

"lookup_table":
False,

"elapsed_seconds":
time.time()-t0

}



with open(
OUT/"p15_7c_manifest.json",
"w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


print("="*80)
print(
"P15-7C FINAL RESULT: PASS_COG_PRODUCTS"
)
print("="*80)



PY


python -m py_compile ${PY}

echo "SYNTAX PASS"


python ${PY}

