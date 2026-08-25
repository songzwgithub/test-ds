#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/ubuntu/Downloads/psds
OUT=$PROJECT/output
PROC=$OUT/processing
DEM=/home/ubuntu/Downloads/DEM_prep

DEST=$PROC/gacos_geometry

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=$PROJECT/production_logs

JSON_REPORT=$LOGDIR/P15_3A_radar_point_geolocation_${STAMP}.json
TXT_REPORT=$LOGDIR/P15_3A_radar_point_geolocation_${STAMP}.txt
GAMMA_LOG=$LOGDIR/P15_3A_gc_map_inversion_${STAMP}.log

mkdir -p "$LOGDIR"

echo "================================================================================================"
echo " P15-3A BUILD RADAR-POINT GEOLOCATION"
echo
echo " MODIFIES NEW DERIVED GEOMETRY OUTPUT ONLY:"
echo "   $DEST"
echo
echo " NO SOURCE MODIFICATION"
echo " NO PHASE MODIFICATION"
echo " NO PS/DS MODIFICATION"
echo " NO UNWRAP MODIFICATION"
echo " NO GACOS CORRECTION"
echo
echo " RUNS GAMMA gc_map_inversion ONCE"
echo "================================================================================================"

python - \
    "$PROJECT" \
    "$OUT" \
    "$DEM" \
    "$DEST" \
    "$JSON_REPORT" \
    "$TXT_REPORT" \
    "$GAMMA_LOG" <<'PY'

from pathlib import Path
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np


PROJECT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
DEM = Path(sys.argv[3]).resolve()
DEST = Path(sys.argv[4]).resolve()

JSON_REPORT = Path(sys.argv[5]).resolve()
TXT_REPORT = Path(sys.argv[6]).resolve()
GAMMA_LOG = Path(sys.argv[7]).resolve()

PROC = OUT / "processing"

errors = []
warnings = []
checks = []


def check(name, ok, detail=""):

    ok = bool(ok)

    print(
        f"{'PASS' if ok else 'FAIL':4s}  "
        f"{name}"
        +
        (
            f"  [{detail}]"
            if detail
            else ""
        )
    )

    checks.append({
        "name": name,
        "pass": ok,
        "detail": str(detail),
    })

    if not ok:
        errors.append(
            f"{name}: {detail}"
        )

    return ok


def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def parse_par(path):

    d = {}

    for line in Path(path).read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(
            ":",
            1,
        )

        d[
            k.strip().lower()
        ] = v.strip()

    return d


def first_float(x):

    if x is None:
        return None

    try:
        return float(
            str(x).split()[0]
        )
    except Exception:
        return None


def first_int(d, names):

    for name in names:

        x = first_float(
            d.get(
                name
            )
        )

        if x is not None:
            return int(
                round(
                    x
                )
            )

    return None


def get_string(d, names):

    for name in names:

        if name in d:
            return str(
                d[name]
            ).split()[0]

    return None


# =============================================================================
# A. Upstream gate
# =============================================================================

print()
print("=" * 96)
print("A. UPSTREAM GATE")
print("=" * 96)


p153_reports = sorted(
    (
        PROJECT
        / "production_logs"
    ).glob(
        "P15_3_gacos_geometry_sign_unit_*.json"
    )
)

check(
    "P15-3 report found",
    len(
        p153_reports
    ) > 0,
    len(
        p153_reports
    ),
)

if not p153_reports:
    raise SystemExit(1)


p153_path = p153_reports[-1]

p153 = json.loads(
    p153_path.read_text(
        encoding="utf-8"
    )
)


check(
    "P15-3 expected status",
    p153.get(
        "status"
    )
    ==
    "PASS_GACOS_FILE_SIGN_UNIT_NEEDS_GEOLOCATION",
    p153.get(
        "status"
    ),
)


# =============================================================================
# B. Strict-point geometry
# =============================================================================

print()
print("=" * 96)
print("B. STRICT POINT GEOMETRY")
print("=" * 96)


strict_ids = np.asarray(
    np.load(
        PROC
        / "network_inversion"
        / "strict_point_ids.npy"
    ),
    dtype=np.int64,
)


rows_all = np.load(
    PROC
    / "point_phase_stack"
    / "rows.npy",
    mmap_mode="r",
)


cols_all = np.load(
    PROC
    / "point_phase_stack"
    / "cols.npy",
    mmap_mode="r",
)


rows = np.asarray(
    rows_all[
        strict_ids
    ],
    dtype=np.int32,
)


cols = np.asarray(
    cols_all[
        strict_ids
    ],
    dtype=np.int32,
)


N = rows.size

RADAR_H = 600
RADAR_W = 2000


check(
    "strict point count",
    N == 881315,
    N,
)

check(
    "radar rows valid",
    np.all(
        (rows >= 0)
        &
        (rows < RADAR_H)
    ),
)

check(
    "radar cols valid",
    np.all(
        (cols >= 0)
        &
        (cols < RADAR_W)
    ),
)


# =============================================================================
# C. Identify lv_theta geometry
# =============================================================================

print()
print("=" * 96)
print("C. MAP GEOMETRY DISCOVERY")
print("=" * 96)


theta_path = (
    DEM
    / "20151212.lv_theta"
)


check(
    "lv_theta exists",
    theta_path.is_file(),
    theta_path,
)


if not theta_path.is_file():
    raise SystemExit(1)


if (
    theta_path.stat().st_size
    %
    4
    !=
    0
):

    raise RuntimeError(
        "lv_theta is not a FLOAT-sized raster."
    )


theta_npixels = (
    theta_path.stat().st_size
    //
    4
)


print(
    f"lv_theta pixels          : "
    f"{theta_npixels:,}"
)


par_candidates = []


for p in DEM.rglob(
    "*.par"
):

    try:
        d = parse_par(
            p
        )

    except Exception:
        continue


    width = first_int(
        d,
        (
            "width",
            "range_samples",
            "range_samp_1",
        ),
    )


    length = first_int(
        d,
        (
            "nlines",
            "azimuth_lines",
            "az_samp_1",
        ),
    )


    if (
        width is None
        or
        length is None
    ):
        continue


    if (
        width
        *
        length
        !=
        theta_npixels
    ):
        continue


    projection = get_string(
        d,
        (
            "dem_projection",
            "projection",
        ),
    )


    score = 0

    low = p.name.lower()

    if "dem_seg" in low:
        score += 100

    if "seg" in low:
        score += 50

    if "dem" in low:
        score += 20


    par_candidates.append({
        "path": p,
        "par": d,
        "width": width,
        "length": length,
        "projection": projection,
        "score": score,
    })


print(
    f"matching map .par files : "
    f"{len(par_candidates)}"
)


for x in sorted(
    par_candidates,
    key=lambda z: (
        -z["score"],
        str(z["path"]),
    ),
):

    print(
        f"  score={x['score']:3d}  "
        f"{x['length']}x{x['width']}  "
        f"projection={x['projection']}  "
        f"{x['path']}"
    )


check(
    "map geometry parameter found",
    len(
        par_candidates
    ) > 0,
    len(
        par_candidates
    ),
)


if not par_candidates:
    raise SystemExit(1)


par_candidates.sort(
    key=lambda z: (
        -z["score"],
        str(
            z["path"]
        ),
    )
)


top_score = par_candidates[0][
    "score"
]


top = [
    x
    for x in par_candidates
    if x[
        "score"
    ]
    ==
    top_score
]


# Multiple files are acceptable only if the actual map-grid
# geometry is identical.
def geometry_signature(x):

    d = x[
        "par"
    ]

    keys = (
        "dem_projection",
        "projection",
        "width",
        "nlines",
        "corner_lat",
        "corner_lon",
        "post_lat",
        "post_lon",
        "corner_north",
        "corner_east",
        "post_north",
        "post_east",
        "zone",
    )

    return tuple(
        d.get(
            k
        )
        for k in keys
    )


sigs = {
    geometry_signature(
        x
    )
    for x in top
}


check(
    "top map geometry unambiguous",
    len(
        sigs
    )
    ==
    1,
    (
        f"top candidates={len(top)}, "
        f"geometry variants={len(sigs)}"
    ),
)


if len(
    sigs
) != 1:

    raise SystemExit(1)


map_item = top[0]

map_par_path = map_item[
    "path"
]

map_par = map_item[
    "par"
]

MAP_W = map_item[
    "width"
]

MAP_H = map_item[
    "length"
]


print()
print(
    f"selected map par         : "
    f"{map_par_path}"
)

print(
    f"map geometry             : "
    f"{MAP_H} x {MAP_W}"
)


# =============================================================================
# D. Projection must be geographic/EQA
# =============================================================================

print()
print("=" * 96)
print("D. MAP COORDINATE CONTRACT")
print("=" * 96)


projection = (
    get_string(
        map_par,
        (
            "dem_projection",
            "projection",
        ),
    )
    or
    ""
).upper()


print(
    "projection               :",
    projection,
)


geographic_projection = (
    projection.startswith(
        "EQA"
    )
    or
    projection.startswith(
        "LAT"
    )
    or
    projection.startswith(
        "GEO"
    )
)


check(
    "map geometry is geographic",
    geographic_projection,
    projection,
)


if not geographic_projection:

    print()
    print(
        "STOP: projection is not direct geographic lat/lon."
    )

    print(
        "A projected-coordinate conversion must be added "
        "instead of guessing."
    )

    raise SystemExit(1)


corner_lat = first_float(
    map_par.get(
        "corner_lat"
    )
)

corner_lon = first_float(
    map_par.get(
        "corner_lon"
    )
)

post_lat = first_float(
    map_par.get(
        "post_lat"
    )
)

post_lon = first_float(
    map_par.get(
        "post_lon"
    )
)


for name, value in (
    (
        "corner_lat",
        corner_lat,
    ),
    (
        "corner_lon",
        corner_lon,
    ),
    (
        "post_lat",
        post_lat,
    ),
    (
        "post_lon",
        post_lon,
    ),
):

    check(
        f"{name} resolved",
        value is not None,
        value,
    )


if None in (
    corner_lat,
    corner_lon,
    post_lat,
    post_lon,
):

    raise SystemExit(1)


print()
print(
    f"corner lon/lat          : "
    f"{corner_lon:.10f}, "
    f"{corner_lat:.10f}"
)

print(
    f"posting lon/lat         : "
    f"{post_lon:.12f}, "
    f"{post_lat:.12f}"
)


# =============================================================================
# E. Find map -> RDC lookup table
# =============================================================================

print()
print("=" * 96)
print("E. MAP -> RADAR LOOKUP TABLE")
print("=" * 96)


expected_lut_bytes = (
    MAP_W
    *
    MAP_H
    *
    8
)


lut_candidates = []


for p in DEM.rglob("*"):

    if not p.is_file():
        continue

    if p.stat().st_size != expected_lut_bytes:
        continue


    low = p.name.lower()


    # Strongly reject obvious non-LUT rasters.
    if any(
        token in low
        for token in (
            "theta",
            "inc",
            ".hgt",
            "dem",
            "sim_sar",
            "pix",
            "psi",
            "ls_map",
        )
    ):
        continue


    score = 0


    if "fine" in low:
        score += 120

    if ".lt" in low or low.endswith(
        "lt"
    ):
        score += 100

    if "lookup" in low:
        score += 90

    if "to_rdc" in low:
        score += 80

    if "utm_to" in low:
        score += 60

    if "eqa_to" in low:
        score += 60

    if "rdc_to" in low:
        score -= 200

    if "inverse" in low:
        score -= 200


    # Require some positive naming evidence.
    if score <= 0:
        continue


    lut_candidates.append({
        "path": p,
        "score": score,
    })


lut_candidates.sort(
    key=lambda x: (
        -x["score"],
        str(
            x["path"]
        ),
    )
)


print(
    f"LUT candidates          : "
    f"{len(lut_candidates)}"
)


for x in lut_candidates:

    print(
        f"  score={x['score']:3d}  "
        f"{x['path']}"
    )


check(
    "map->RDC LUT found",
    len(
        lut_candidates
    ) > 0,
    len(
        lut_candidates
    ),
)


if not lut_candidates:
    raise SystemExit(1)


best_score = lut_candidates[0][
    "score"
]


best = [
    x
    for x in lut_candidates
    if x[
        "score"
    ]
    ==
    best_score
]


check(
    "best lookup table unambiguous",
    len(
        best
    )
    ==
    1,
    (
        f"best-score candidates="
        f"{len(best)}"
    ),
)


if len(
    best
) != 1:

    raise SystemExit(1)


lut_path = best[0][
    "path"
]


print()
print(
    f"selected LUT            : "
    f"{lut_path}"
)

print(
    f"LUT bytes               : "
    f"{lut_path.stat().st_size:,}"
)


# =============================================================================
# F. Resolve GAMMA gc_map_inversion
# =============================================================================

print()
print("=" * 96)
print("F. GAMMA gc_map_inversion")
print("=" * 96)


cmd = shutil.which(
    "gc_map_inversion"
)


if cmd is None:

    candidates = list(
        Path(
            "/home/ubuntu/software/GAMMA_SOFTWARE"
        ).rglob(
            "gc_map_inversion"
        )
    )

    candidates = [
        p
        for p in candidates
        if (
            p.is_file()
            and
            os.access(
                p,
                os.X_OK,
            )
        )
    ]

    if len(
        candidates
    )
    ==
    1:

        cmd = str(
            candidates[0]
        )


check(
    "gc_map_inversion executable",
    cmd is not None,
    cmd,
)


if cmd is None:
    raise SystemExit(1)


# =============================================================================
# G. Invert the lookup table
#
# Map-grid lookup:
#   per map pixel -> RDC range/azimuth coordinates.
#
# gc_map_inversion:
#   per RDC pixel -> map-grid col/row coordinates.
#
# GAMMA documentation defines these pixel coordinates on the
# zero-based grid [0,0], [0,1], ...
# =============================================================================

print()
print("=" * 96)
print("G. BUILD RDC -> MAP INVERSE LOOKUP")
print("=" * 96)


DEST.mkdir(
    parents=True,
    exist_ok=True,
)


final_inverse = (
    DEST
    / "rdc_to_map_pixel.fcomplex"
)


tmpdir = Path(
    tempfile.mkdtemp(
        prefix="p15_3a_",
        dir=str(
            DEST
        ),
    )
)


tmp_inverse = (
    tmpdir
    / "rdc_to_map_pixel.fcomplex"
)


command = [
    cmd,
    str(
        lut_path
    ),
    str(
        MAP_W
    ),
    str(
        tmp_inverse
    ),
    str(
        RADAR_W
    ),
    str(
        RADAR_H
    ),
]


print(
    "command:"
)

print(
    "  "
    +
    " ".join(
        command
    )
)


proc = subprocess.run(
    command,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)


GAMMA_LOG.write_text(
    "$ "
    +
    " ".join(
        command
    )
    +
    "\n\n"
    +
    (
        proc.stdout
        or
        ""
    )
    +
    f"\nreturncode={proc.returncode}\n",
    encoding="utf-8",
)


check(
    "gc_map_inversion return code",
    proc.returncode == 0,
    proc.returncode,
)


if proc.returncode != 0:

    print(
        "\n".join(
            (
                proc.stdout
                or
                ""
            ).splitlines()[
                -30:
            ]
        )
    )

    raise SystemExit(1)


expected_inverse_bytes = (
    RADAR_H
    *
    RADAR_W
    *
    8
)


check(
    "inverse LUT byte size",
    tmp_inverse.is_file()
    and
    tmp_inverse.stat().st_size
    ==
    expected_inverse_bytes,
    (
        tmp_inverse.stat().st_size
        if tmp_inverse.exists()
        else None
    ),
)


if (
    not tmp_inverse.is_file()
    or
    tmp_inverse.stat().st_size
    !=
    expected_inverse_bytes
):

    raise SystemExit(1)


# =============================================================================
# H. Detect GAMMA lookup byte order
# =============================================================================

print()
print("=" * 96)
print("H. INVERSE LUT BYTE ORDER / VALIDITY")
print("=" * 96)


def score_lut(dtype):

    a = np.memmap(
        tmp_inverse,
        dtype=dtype,
        mode="r",
        shape=(
            RADAR_H,
            RADAR_W,
        ),
    )


    z = np.asarray(
        a[
            rows,
            cols
        ]
    )


    x = np.asarray(
        z.real,
        dtype=np.float64,
    )

    y = np.asarray(
        z.imag,
        dtype=np.float64,
    )


    good = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
        &
        (
            x >= -2.0
        )
        &
        (
            x <= MAP_W + 1.0
        )
        &
        (
            y >= -2.0
        )
        &
        (
            y <= MAP_H + 1.0
        )
    )


    return (
        float(
            np.mean(
                good
            )
        ),
        x,
        y,
        good,
    )


big = score_lut(
    ">c8"
)

little = score_lut(
    "<c8"
)


print(
    f"big-endian valid fraction : "
    f"{big[0]:.9f}"
)

print(
    f"little-endian valid frac   : "
    f"{little[0]:.9f}"
)


if big[0] > little[0]:

    lut_dtype = ">c8"

    (
        valid_fraction,
        map_x,
        map_y,
        valid,
    ) = big

else:

    lut_dtype = "<c8"

    (
        valid_fraction,
        map_x,
        map_y,
        valid,
    ) = little


print(
    f"selected LUT dtype         : "
    f"{lut_dtype}"
)


check(
    "strict-point inverse LUT coverage",
    valid_fraction
    >
    0.999,
    f"{valid_fraction:.9f}",
)


if valid_fraction <= 0.999:

    raise SystemExit(1)


# =============================================================================
# I. Map pixels -> lon / lat
# =============================================================================

print()
print("=" * 96)
print("I. STRICT-POINT LONGITUDE / LATITUDE")
print("=" * 96)


longitude = (
    corner_lon
    +
    map_x
    *
    post_lon
)


latitude = (
    corner_lat
    +
    map_y
    *
    post_lat
)


longitude[
    ~valid
] = np.nan

latitude[
    ~valid
] = np.nan


finite_ll = (
    np.isfinite(
        longitude
    )
    &
    np.isfinite(
        latitude
    )
)


check(
    "finite lon/lat fraction",
    np.mean(
        finite_ll
    )
    >
    0.999,
    f"{np.mean(finite_ll):.9f}",
)


print()
print(
    f"longitude min/max       : "
    f"{np.nanmin(longitude):.8f} / "
    f"{np.nanmax(longitude):.8f}"
)

print(
    f"latitude min/max        : "
    f"{np.nanmin(latitude):.8f} / "
    f"{np.nanmax(latitude):.8f}"
)

print(
    f"median lon/lat          : "
    f"{np.nanmedian(longitude):.8f}, "
    f"{np.nanmedian(latitude):.8f}"
)


# GACOS extent from accepted P15-3.
ggrid = p153[
    "gacos"
][
    "grid"
]


inside_gacos = (
    finite_ll
    &
    (
        longitude
        >=
        float(
            ggrid[
                "lon_min"
            ]
        )
    )
    &
    (
        longitude
        <=
        float(
            ggrid[
                "lon_max"
            ]
        )
    )
    &
    (
        latitude
        >=
        float(
            ggrid[
                "lat_min"
            ]
        )
    )
    &
    (
        latitude
        <=
        float(
            ggrid[
                "lat_max"
            ]
        )
    )
)


inside_fraction = float(
    np.mean(
        inside_gacos
    )
)


print(
    f"strict points inside GACOS: "
    f"{inside_fraction:.9f}"
)


check(
    "strict-point GACOS coverage",
    inside_fraction
    >
    0.999,
    inside_fraction,
)


# =============================================================================
# J. Detect lv_theta byte order/unit and bilinear-sample at strict points
# =============================================================================

print()
print("=" * 96)
print("J. POINTWISE INCIDENCE ANGLE")
print("=" * 96)


def theta_candidate(dtype):

    a = np.memmap(
        theta_path,
        dtype=dtype,
        mode="r",
        shape=(
            MAP_H,
            MAP_W,
        ),
    )


    # sparse sample
    sample = np.asarray(
        a.reshape(-1)[
            ::max(
                1,
                theta_npixels
                //
                100000,
            )
        ],
        dtype=np.float64,
    )


    sample = sample[
        np.isfinite(
            sample
        )
    ]


    if sample.size == 0:

        return {
            "score": -1,
            "unit": None,
            "median": np.nan,
        }


    med = float(
        np.median(
            sample
        )
    )


    # radians expected roughly 0.3..1.2
    if (
        0.15
        <
        med
        <
        1.5
    ):

        unit = "radian"

        deg_med = math.degrees(
            med
        )

        score = (
            1.0
            if 15.0 < deg_med < 60.0
            else 0.5
        )


    # degrees expected roughly 15..60
    elif (
        10.0
        <
        med
        <
        80.0
    ):

        unit = "degree"

        deg_med = med

        score = 1.0


    else:

        unit = None
        deg_med = np.nan
        score = 0.0


    return {
        "score": score,
        "unit": unit,
        "median": med,
        "median_deg": deg_med,
    }


theta_big = theta_candidate(
    ">f4"
)

theta_little = theta_candidate(
    "<f4"
)


print(
    "theta big-endian candidate:",
    theta_big,
)

print(
    "theta little-endian candidate:",
    theta_little,
)


if theta_big[
    "score"
] > theta_little[
    "score"
]:

    theta_dtype = ">f4"
    theta_info = theta_big

else:

    theta_dtype = "<f4"
    theta_info = theta_little


check(
    "lv_theta byte-order/unit resolved",
    theta_info[
        "score"
    ] > 0,
    (
        f"dtype={theta_dtype}, "
        f"unit={theta_info['unit']}, "
        f"median_deg="
        f"{theta_info['median_deg']}"
    ),
)


if theta_info[
    "score"
] <= 0:

    raise SystemExit(1)


theta_grid = np.memmap(
    theta_path,
    dtype=theta_dtype,
    mode="r",
    shape=(
        MAP_H,
        MAP_W,
    ),
)


# Bilinear interpolation in map-pixel coordinates.

x0 = np.floor(
    map_x
).astype(
    np.int64
)

y0 = np.floor(
    map_y
).astype(
    np.int64
)


x1 = x0 + 1
y1 = y0 + 1


interp_valid = (
    valid
    &
    (
        x0 >= 0
    )
    &
    (
        y0 >= 0
    )
    &
    (
        x1 < MAP_W
    )
    &
    (
        y1 < MAP_H
    )
)


incidence_raw = np.full(
    N,
    np.nan,
    dtype=np.float64,
)


ids = np.where(
    interp_valid
)[0]


if ids.size:

    xx = map_x[
        ids
    ]

    yy = map_y[
        ids
    ]

    xa = x0[
        ids
    ]

    xb = x1[
        ids
    ]

    ya = y0[
        ids
    ]

    yb = y1[
        ids
    ]


    dx = (
        xx
        -
        xa
    )

    dy = (
        yy
        -
        ya
    )


    v00 = np.asarray(
        theta_grid[
            ya,
            xa
        ],
        dtype=np.float64,
    )

    v10 = np.asarray(
        theta_grid[
            ya,
            xb
        ],
        dtype=np.float64,
    )

    v01 = np.asarray(
        theta_grid[
            yb,
            xa
        ],
        dtype=np.float64,
    )

    v11 = np.asarray(
        theta_grid[
            yb,
            xb
        ],
        dtype=np.float64,
    )


    incidence_raw[
        ids
    ] = (
        (
            1.0
            -
            dx
        )
        *
        (
            1.0
            -
            dy
        )
        *
        v00
        +
        dx
        *
        (
            1.0
            -
            dy
        )
        *
        v10
        +
        (
            1.0
            -
            dx
        )
        *
        dy
        *
        v01
        +
        dx
        *
        dy
        *
        v11
    )


if theta_info[
    "unit"
] == "radian":

    incidence_deg = np.degrees(
        incidence_raw
    )

else:

    incidence_deg = incidence_raw.copy()


inc_valid = (
    np.isfinite(
        incidence_deg
    )
    &
    (
        incidence_deg
        >
        10.0
    )
    &
    (
        incidence_deg
        <
        80.0
    )
)


inc_fraction = float(
    np.mean(
        inc_valid
    )
)


check(
    "pointwise incidence valid coverage",
    inc_fraction
    >
    0.999,
    inc_fraction,
)


inc_q = np.nanpercentile(
    incidence_deg[
        inc_valid
    ],
    [
        1,
        5,
        50,
        95,
        99,
    ],
)


print()
print(
    "incidence p01/p05/p50/p95/p99:"
)

print(
    "  "
    +
    " / ".join(
        f"{x:.5f}"
        for x in inc_q
    )
    +
    " deg"
)


center_inc = float(
    p153[
        "sar_scene"
    ][
        "center_incidence_angle_deg"
    ]
)


median_inc = float(
    inc_q[
        2
    ]
)


check(
    "pointwise incidence agrees with SAR geometry",
    abs(
        median_inc
        -
        center_inc
    )
    <
    3.0,
    (
        f"point median={median_inc:.5f}, "
        f"center={center_inc:.5f}"
    ),
)


# =============================================================================
# K. Save atomically only after all scientific gates pass
# =============================================================================

print()
print("=" * 96)
print("K. SAVE DERIVED GACOS GEOMETRY")
print("=" * 96)


if errors:

    print(
        "Scientific gate failed. "
        "No derived point geometry will be accepted."
    )

    raise SystemExit(1)


# Keep inverse LUT as reproducibility geometry asset.
os.replace(
    tmp_inverse,
    final_inverse,
)


np.save(
    DEST
    / "strict_point_ids.npy",
    strict_ids.astype(
        np.int32
    ),
)


np.save(
    DEST
    / "radar_row.npy",
    rows.astype(
        np.int32
    ),
)


np.save(
    DEST
    / "radar_col.npy",
    cols.astype(
        np.int32
    ),
)


np.save(
    DEST
    / "map_pixel_col.npy",
    map_x.astype(
        np.float64
    ),
)


np.save(
    DEST
    / "map_pixel_row.npy",
    map_y.astype(
        np.float64
    ),
)


np.save(
    DEST
    / "longitude_deg.npy",
    longitude.astype(
        np.float64
    ),
)


np.save(
    DEST
    / "latitude_deg.npy",
    latitude.astype(
        np.float64
    ),
)


np.save(
    DEST
    / "incidence_angle_deg.npy",
    incidence_deg.astype(
        np.float32
    ),
)


np.save(
    DEST
    / "valid_geolocation_mask.npy",
    (
        inside_gacos
        &
        inc_valid
    ),
)


accepted_mask = (
    inside_gacos
    &
    inc_valid
)


accepted_fraction = float(
    np.mean(
        accepted_mask
    )
)


check(
    "final point geometry accepted fraction",
    accepted_fraction
    >
    0.999,
    accepted_fraction,
)


manifest = {
    "format":
        "pyPSDS-GAMMA-P15-3A-gacos-point-geometry-v1",

    "status":
        "PASS_RADAR_POINT_GEOLOCATION",

    "points":
        int(
            N
        ),

    "radar_geometry": {
        "rows":
            RADAR_H,

        "cols":
            RADAR_W,
    },

    "map_geometry": {
        "parameter_file":
            str(
                map_par_path
            ),

        "projection":
            projection,

        "rows":
            MAP_H,

        "cols":
            MAP_W,

        "corner_lon":
            corner_lon,

        "corner_lat":
            corner_lat,

        "post_lon":
            post_lon,

        "post_lat":
            post_lat,
    },

    "lookup": {
        "map_to_radar":
            str(
                lut_path
            ),

        "map_to_radar_sha256":
            sha256(
                lut_path
            ),

        "inverse_radar_to_map":
            str(
                final_inverse
            ),

        "inverse_dtype":
            lut_dtype,

        "gc_map_inversion":
            str(
                cmd
            ),
    },

    "incidence": {
        "source":
            str(
                theta_path
            ),

        "source_sha256":
            sha256(
                theta_path
            ),

        "source_dtype":
            theta_dtype,

        "source_unit":
            theta_info[
                "unit"
            ],

        "output_unit":
            "degree",

        "p01_deg":
            float(
                inc_q[0]
            ),

        "p05_deg":
            float(
                inc_q[1]
            ),

        "p50_deg":
            float(
                inc_q[2]
            ),

        "p95_deg":
            float(
                inc_q[3]
            ),

        "p99_deg":
            float(
                inc_q[4]
            ),
    },

    "gacos_coverage_fraction":
        inside_fraction,

    "incidence_valid_fraction":
        inc_fraction,

    "accepted_fraction":
        accepted_fraction,

    "products": {
        "longitude_deg":
            str(
                DEST
                / "longitude_deg.npy"
            ),

        "latitude_deg":
            str(
                DEST
                / "latitude_deg.npy"
            ),

        "incidence_angle_deg":
            str(
                DEST
                / "incidence_angle_deg.npy"
            ),

        "valid_mask":
            str(
                DEST
                / "valid_geolocation_mask.npy"
            ),
    },

    "next_step":
        "P15-4_GACOS_POINT_SAMPLING_SMOKE",
}


manifest_path = (
    DEST
    / "gacos_geometry_manifest.json"
)


manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
    +
    "\n",
    encoding="utf-8",
)


# Cleanup empty temporary directory.
try:
    tmpdir.rmdir()
except OSError:
    pass


report = {
    "format":
        "pyPSDS-GAMMA-P15-3A-audit-v1",

    "status":
        "PASS_RADAR_POINT_GEOLOCATION",

    "P15_3_report":
        str(
            p153_path
        ),

    "map_parameter":
        str(
            map_par_path
        ),

    "map_to_radar_lookup":
        str(
            lut_path
        ),

    "inverse_lookup":
        str(
            final_inverse
        ),

    "strict_points":
        int(
            N
        ),

    "gacos_coverage_fraction":
        inside_fraction,

    "incidence_valid_fraction":
        inc_fraction,

    "accepted_fraction":
        accepted_fraction,

    "longitude_range_deg":
        [
            float(
                np.nanmin(
                    longitude
                )
            ),
            float(
                np.nanmax(
                    longitude
                )
            ),
        ],

    "latitude_range_deg":
        [
            float(
                np.nanmin(
                    latitude
                )
            ),
            float(
                np.nanmax(
                    latitude
                )
            ),
        ],

    "incidence_percentiles_deg":
        {
            "p01":
                float(
                    inc_q[0]
                ),

            "p05":
                float(
                    inc_q[1]
                ),

            "p50":
                float(
                    inc_q[2]
                ),

            "p95":
                float(
                    inc_q[3]
                ),

            "p99":
                float(
                    inc_q[4]
                ),
        },

    "manifest":
        str(
            manifest_path
        ),

    "next_step":
        "P15-4_GACOS_POINT_SAMPLING_SMOKE",

    "checks":
        checks,

    "warnings":
        warnings,

    "errors":
        errors,
}


JSON_REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
    +
    "\n",
    encoding="utf-8",
)


lines = [
    "=" * 88,
    "P15-3A RADAR POINT GEOLOCATION",
    "=" * 88,

    "status                    : PASS_RADAR_POINT_GEOLOCATION",

    f"strict points             : {N:,}",

    (
        "map geometry             : "
        f"{MAP_H} x {MAP_W}"
    ),

    (
        "inverse LUT dtype        : "
        f"{lut_dtype}"
    ),

    (
        "GACOS coverage           : "
        f"{100*inside_fraction:.6f}%"
    ),

    (
        "incidence valid          : "
        f"{100*inc_fraction:.6f}%"
    ),

    (
        "accepted geometry        : "
        f"{100*accepted_fraction:.6f}%"
    ),

    (
        "longitude range          : "
        f"{np.nanmin(longitude):.8f} .. "
        f"{np.nanmax(longitude):.8f}"
    ),

    (
        "latitude range           : "
        f"{np.nanmin(latitude):.8f} .. "
        f"{np.nanmax(latitude):.8f}"
    ),

    (
        "incidence p50            : "
        f"{median_inc:.5f} deg"
    ),

    "",

    "next step                 : P15-4_GACOS_POINT_SAMPLING_SMOKE",

    "",

    f"manifest                  : {manifest_path}",

    f"JSON report               : {JSON_REPORT}",
]


TXT_REPORT.write_text(
    "\n".join(
        lines
    )
    +
    "\n",
    encoding="utf-8",
)


print()
print(
    "\n".join(
        lines
    )
)

print()
print("=" * 96)
print(
    " P15-3A FINAL RESULT: "
    "PASS_RADAR_POINT_GEOLOCATION"
)
print("=" * 96)

PY

echo
echo "reports:"
echo "  $JSON_REPORT"
echo "  $TXT_REPORT"
