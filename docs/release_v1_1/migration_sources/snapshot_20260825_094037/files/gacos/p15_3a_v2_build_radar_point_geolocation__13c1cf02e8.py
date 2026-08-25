from pathlib import Path
import hashlib
import json
import math
import os
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
            if detail != ""
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


def first_int(d, *names):

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


def sha256(path):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        for b in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                b
            )

    return h.hexdigest()


print("=" * 96)
print("P15-3A v2 BUILD RADAR-POINT GEOLOCATION")
print("=" * 96)


# =============================================================================
# A. P15-3 upstream gate
# =============================================================================

reports = sorted(
    (
        PROJECT
        / "production_logs"
    ).glob(
        "P15_3_gacos_geometry_sign_unit_*.json"
    )
)


check(
    "P15-3 report found",
    bool(
        reports
    ),
    len(
        reports
    ),
)


if not reports:
    raise SystemExit(1)


p153_path = reports[-1]


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


if errors:
    raise SystemExit(1)


# =============================================================================
# B. Strict production points
# =============================================================================

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
    "radar coordinate bounds",
    np.all(
        (rows >= 0)
        &
        (rows < RADAR_H)
        &
        (cols >= 0)
        &
        (cols < RADAR_W)
    ),
)


if errors:
    raise SystemExit(1)


# =============================================================================
# C. lv_theta + matching map-grid parameter
# =============================================================================

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


check(
    "lv_theta FLOAT-size",
    theta_path.stat().st_size % 4 == 0,
    theta_path.stat().st_size,
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


map_candidates = []


for p in DEM.rglob(
    "*.par"
):

    try:

        d = parse_par(
            p
        )

    except Exception:

        continue


    w = first_int(
        d,
        "width",
        "range_samples",
        "range_samp_1",
    )


    h = first_int(
        d,
        "nlines",
        "azimuth_lines",
        "az_samp_1",
    )


    if (
        w is None
        or
        h is None
        or
        w * h
        !=
        theta_npixels
    ):
        continue


    projection = str(
        d.get(
            "dem_projection",
            d.get(
                "projection",
                "",
            ),
        )
    ).split()[0].upper()


    low = p.name.lower()

    score = 0


    if "dem_seg" in low:
        score += 100

    if "seg" in low:
        score += 50

    if "dem" in low:
        score += 20


    map_candidates.append(
        (
            score,
            str(
                p
            ),
            p,
            d,
            h,
            w,
            projection,
        )
    )


map_candidates.sort(
    key=lambda x: (
        -x[0],
        x[1],
    )
)


print(
    f"matching map pars        : "
    f"{len(map_candidates)}"
)


for x in map_candidates[:20]:

    print(
        f"  score={x[0]:3d} "
        f"{x[4]}x{x[5]} "
        f"{x[6]} "
        f"{x[2]}"
    )


check(
    "map geometry parameter found",
    bool(
        map_candidates
    ),
    len(
        map_candidates
    ),
)


if not map_candidates:
    raise SystemExit(1)


top_score = map_candidates[
    0
][
    0
]


top = [
    x
    for x
    in map_candidates
    if x[
        0
    ]
    ==
    top_score
]


def geometry_signature(item):

    d = item[
        3
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
        for k
        in keys
    )


variants = {
    geometry_signature(
        x
    )
    for x
    in top
}


check(
    "top map geometry unambiguous",
    len(
        variants
    )
    ==
    1,
    (
        f"top={len(top)}, "
        f"variants={len(variants)}"
    ),
)


if errors:
    raise SystemExit(1)


(
    _,
    _,
    map_par_path,
    map_par,
    MAP_H,
    MAP_W,
    projection,
) = top[
    0
]


check(
    "map projection geographic",
    projection.startswith(
        (
            "EQA",
            "LAT",
            "GEO",
        )
    ),
    projection,
)


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


if errors:
    raise SystemExit(1)


print()
print(
    f"selected map par         : "
    f"{map_par_path}"
)

print(
    f"map geometry             : "
    f"{MAP_H} x {MAP_W}"
)

print(
    f"corner lon/lat           : "
    f"{corner_lon:.10f}, "
    f"{corner_lat:.10f}"
)

print(
    f"post lon/lat             : "
    f"{post_lon:.12f}, "
    f"{post_lat:.12f}"
)


# =============================================================================
# D. Discover map -> radar lookup table
# =============================================================================

expected_lut_bytes = (
    MAP_H
    *
    MAP_W
    *
    8
)


lut_candidates = []


for p in DEM.rglob("*"):

    if not p.is_file():
        continue


    try:

        if (
            p.stat().st_size
            !=
            expected_lut_bytes
        ):
            continue

    except OSError:

        continue


    low = p.name.lower()


    if any(
        token in low
        for token in (
            "theta",
            "inc",
            ".hgt",
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

    if (
        low.endswith(
            ".lt"
        )
        or
        ".lt."
        in low
        or
        low.endswith(
            "lt"
        )
    ):
        score += 100

    if "lookup" in low:
        score += 90

    if "to_rdc" in low:
        score += 80

    if (
        "utm_to" in low
        or
        "eqa_to" in low
    ):
        score += 60

    if (
        "rdc_to" in low
        or
        "inverse" in low
    ):
        score -= 200


    if score > 0:

        lut_candidates.append(
            (
                score,
                str(
                    p
                ),
                p,
            )
        )


lut_candidates.sort(
    key=lambda x: (
        -x[0],
        x[1],
    )
)


print()
print(
    f"LUT candidates           : "
    f"{len(lut_candidates)}"
)


for score, _, p in lut_candidates[:20]:

    print(
        f"  score={score:3d} "
        f"{p}"
    )


check(
    "map->radar LUT found",
    bool(
        lut_candidates
    ),
    len(
        lut_candidates
    ),
)


if not lut_candidates:
    raise SystemExit(1)


best_score = lut_candidates[
    0
][
    0
]


best = [
    x
    for x
    in lut_candidates
    if x[
        0
    ]
    ==
    best_score
]


check(
    "best LUT unambiguous",
    len(
        best
    )
    ==
    1,
    len(
        best
    ),
)


if errors:
    raise SystemExit(1)


lut_path = best[
    0
][
    2
]


print(
    f"selected LUT             : "
    f"{lut_path}"
)


# =============================================================================
# E. Resolve GAMMA executable
# =============================================================================

cmd = shutil.which(
    "gc_map_inversion"
)


if cmd is None:

    found = [
        p
        for p
        in Path(
            "/home/ubuntu/software/GAMMA_SOFTWARE"
        ).rglob(
            "gc_map_inversion"
        )
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
        found
    )
    ==
    1:

        cmd = str(
            found[
                0
            ]
        )


check(
    "gc_map_inversion executable",
    cmd is not None,
    cmd,
)


if cmd is None:
    raise SystemExit(1)


# =============================================================================
# F. Run gc_map_inversion
# =============================================================================

DEST.mkdir(
    parents=True,
    exist_ok=True,
)


tmpdir = Path(
    tempfile.mkdtemp(
        prefix=".p15_3a_v2_",
        dir=DEST,
    )
)


tmp_inverse = (
    tmpdir
    / "rdc_to_map_pixel.fcomplex"
)


final_inverse = (
    DEST
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


print()
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


expected_inverse_bytes = (
    RADAR_H
    *
    RADAR_W
    *
    8
)


check(
    "inverse LUT byte size",
    (
        tmp_inverse.is_file()
        and
        tmp_inverse.stat().st_size
        ==
        expected_inverse_bytes
    ),
    (
        tmp_inverse.stat().st_size
        if tmp_inverse.exists()
        else None
    ),
)


if errors:

    print()
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


# =============================================================================
# G. Detect inverse LUT endian
# =============================================================================

def read_inverse(dtype):

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
            x >= -1.5
        )
        &
        (
            x <= MAP_W + 0.5
        )
        &
        (
            y >= -1.5
        )
        &
        (
            y <= MAP_H + 0.5
        )
    )


    return (
        float(
            good.mean()
        ),
        x,
        y,
        good,
    )


be = read_inverse(
    ">c8"
)

le = read_inverse(
    "<c8"
)


print()
print(
    f"big-endian valid fraction: "
    f"{be[0]:.9f}"
)

print(
    f"little-endian valid frac : "
    f"{le[0]:.9f}"
)


if be[
    0
] >= le[
    0
]:

    lut_dtype = ">c8"

    chosen = be

else:

    lut_dtype = "<c8"

    chosen = le


(
    coverage,
    map_x,
    map_y,
    valid,
) = chosen


print(
    f"selected inverse dtype   : "
    f"{lut_dtype}"
)


check(
    "inverse LUT strict-point coverage",
    coverage > 0.999,
    coverage,
)


if errors:
    raise SystemExit(1)


# =============================================================================
# H. Map-pixel -> lon/lat
# =============================================================================

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


finite_fraction = float(
    finite_ll.mean()
)


check(
    "finite lon/lat coverage",
    finite_fraction > 0.999,
    finite_fraction,
)


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
    inside_gacos.mean()
)


check(
    "strict points inside GACOS",
    inside_fraction > 0.999,
    inside_fraction,
)


print()
print(
    f"longitude min/max        : "
    f"{np.nanmin(longitude):.8f} / "
    f"{np.nanmax(longitude):.8f}"
)

print(
    f"latitude min/max         : "
    f"{np.nanmin(latitude):.8f} / "
    f"{np.nanmax(latitude):.8f}"
)


# =============================================================================
# I. Resolve lv_theta endian + units
# =============================================================================

def theta_probe(dtype):

    a = np.memmap(
        theta_path,
        dtype=dtype,
        mode="r",
        shape=(
            MAP_H,
            MAP_W,
        ),
    )


    step = max(
        1,
        theta_npixels
        //
        100000,
    )


    sample = np.asarray(
        a.reshape(
            -1
        )[
            ::step
        ],
        dtype=np.float64,
    )


    sample = sample[
        np.isfinite(
            sample
        )
    ]


    if sample.size == 0:

        return (
            -1.0,
            None,
            np.nan,
            np.nan,
        )


    med = float(
        np.median(
            sample
        )
    )


    if (
        0.15
        <
        med
        <
        1.5
    ):

        deg = math.degrees(
            med
        )


        score = (
            1.0
            if 15
            <
            deg
            <
            60
            else 0.5
        )


        return (
            score,
            "radian",
            med,
            deg,
        )


    if (
        10
        <
        med
        <
        80
    ):

        return (
            1.0,
            "degree",
            med,
            med,
        )


    return (
        0.0,
        None,
        med,
        np.nan,
    )


theta_be = theta_probe(
    ">f4"
)

theta_le = theta_probe(
    "<f4"
)


print()
print(
    "theta big-endian candidate:",
    theta_be,
)

print(
    "theta little-endian candidate:",
    theta_le,
)


if theta_be[
    0
] >= theta_le[
    0
]:

    theta_dtype = ">f4"

    theta_info = theta_be

else:

    theta_dtype = "<f4"

    theta_info = theta_le


check(
    "theta byte order/unit resolved",
    theta_info[
        0
    ]
    >
    0,
    (
        f"dtype={theta_dtype}, "
        f"unit={theta_info[1]}, "
        f"median_deg={theta_info[3]}"
    ),
)


if errors:
    raise SystemExit(1)


# =============================================================================
# J. Bilinear sample pointwise incidence
# =============================================================================

theta_grid = np.memmap(
    theta_path,
    dtype=theta_dtype,
    mode="r",
    shape=(
        MAP_H,
        MAP_W,
    ),
)


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


ids = np.flatnonzero(
    interp_valid
)


if ids.size:

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
        map_x[
            ids
        ]
        -
        xa
    )


    dy = (
        map_y[
            ids
        ]
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
    1
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
        incidence_deg > 10
    )
    &
    (
        incidence_deg < 80
    )
)


inc_fraction = float(
    inc_valid.mean()
)


check(
    "pointwise incidence coverage",
    inc_fraction > 0.999,
    inc_fraction,
)


q = np.nanpercentile(
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


center_inc = float(
    p153[
        "sar_scene"
    ][
        "center_incidence_angle_deg"
    ]
)


check(
    "incidence median agrees with center",
    abs(
        float(
            q[
                2
            ]
        )
        -
        center_inc
    )
    <
    3.0,
    (
        f"p50={q[2]:.5f}, "
        f"center={center_inc:.5f}"
    ),
)


print()
print(
    "incidence p01/p05/p50/p95/p99:"
)

print(
    "  "
    +
    " / ".join(
        f"{v:.5f}"
        for v
        in q
    )
    +
    " deg"
)


accepted = (
    inside_gacos
    &
    inc_valid
)


accepted_fraction = float(
    accepted.mean()
)


check(
    "accepted geometry fraction",
    accepted_fraction > 0.999,
    accepted_fraction,
)


if errors:
    raise SystemExit(1)


# =============================================================================
# K. Save only after all gates pass
# =============================================================================

temp_products = (
    tmpdir
    / "products"
)


temp_products.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    temp_products
    / "strict_point_ids.npy",
    strict_ids.astype(
        np.int32
    ),
)


np.save(
    temp_products
    / "radar_row.npy",
    rows,
)


np.save(
    temp_products
    / "radar_col.npy",
    cols,
)


np.save(
    temp_products
    / "map_pixel_col.npy",
    map_x.astype(
        np.float64
    ),
)


np.save(
    temp_products
    / "map_pixel_row.npy",
    map_y.astype(
        np.float64
    ),
)


np.save(
    temp_products
    / "longitude_deg.npy",
    longitude.astype(
        np.float64
    ),
)


np.save(
    temp_products
    / "latitude_deg.npy",
    latitude.astype(
        np.float64
    ),
)


np.save(
    temp_products
    / "incidence_angle_deg.npy",
    incidence_deg.astype(
        np.float32
    ),
)


np.save(
    temp_products
    / "valid_geolocation_mask.npy",
    accepted,
)


for p in temp_products.iterdir():

    os.replace(
        p,
        DEST
        /
        p.name,
    )


os.replace(
    tmp_inverse,
    final_inverse,
)


manifest = {
    "format":
        "pyPSDS-GAMMA-P15-3A-gacos-point-geometry-v2",

    "status":
        "PASS_RADAR_POINT_GEOLOCATION",

    "points":
        int(
            N
        ),

    "map_parameter":
        str(
            map_par_path
        ),

    "map_to_radar_lookup":
        str(
            lut_path
        ),

    "map_to_radar_sha256":
        sha256(
            lut_path
        ),

    "radar_to_map_lookup":
        str(
            final_inverse
        ),

    "inverse_dtype":
        lut_dtype,

    "theta_source":
        str(
            theta_path
        ),

    "theta_dtype":
        theta_dtype,

    "theta_unit":
        theta_info[
            1
        ],

    "gacos_coverage_fraction":
        inside_fraction,

    "incidence_valid_fraction":
        inc_fraction,

    "accepted_fraction":
        accepted_fraction,

    "longitude_range_deg": [
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

    "latitude_range_deg": [
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

    "incidence_percentiles_deg": {
        "p01":
            float(
                q[
                    0
                ]
            ),

        "p05":
            float(
                q[
                    1
                ]
            ),

        "p50":
            float(
                q[
                    2
                ]
            ),

        "p95":
            float(
                q[
                    3
                ]
            ),

        "p99":
            float(
                q[
                    4
                ]
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
    )
    +
    "\n",
    encoding="utf-8",
)


report = {
    "format":
        "pyPSDS-GAMMA-P15-3A-audit-v2",

    "status":
        "PASS_RADAR_POINT_GEOLOCATION",

    "P15_3_report":
        str(
            p153_path
        ),

    "manifest":
        str(
            manifest_path
        ),

    "strict_points":
        int(
            N
        ),

    "accepted_fraction":
        accepted_fraction,

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
    )
    +
    "\n",
    encoding="utf-8",
)


lines = [
    "=" * 88,

    "P15-3A v2 RADAR POINT GEOLOCATION",

    "=" * 88,

    "status                    : PASS_RADAR_POINT_GEOLOCATION",

    f"strict points             : {N:,}",

    f"map geometry              : {MAP_H} x {MAP_W}",

    f"inverse LUT dtype         : {lut_dtype}",

    f"GACOS coverage            : {100*inside_fraction:.6f}%",

    f"incidence valid           : {100*inc_fraction:.6f}%",

    f"accepted geometry         : {100*accepted_fraction:.6f}%",

    (
        "longitude range           : "
        f"{np.nanmin(longitude):.8f} .. "
        f"{np.nanmax(longitude):.8f}"
    ),

    (
        "latitude range            : "
        f"{np.nanmin(latitude):.8f} .. "
        f"{np.nanmax(latitude):.8f}"
    ),

    f"incidence p50             : {q[2]:.5f} deg",

    "next step                 : P15-4_GACOS_POINT_SAMPLING_SMOKE",

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


try:

    temp_products.rmdir()
    tmpdir.rmdir()

except OSError:

    pass


print()
print("=" * 96)
print(
    " P15-3A FINAL RESULT: "
    "PASS_RADAR_POINT_GEOLOCATION"
)
print("=" * 96)
