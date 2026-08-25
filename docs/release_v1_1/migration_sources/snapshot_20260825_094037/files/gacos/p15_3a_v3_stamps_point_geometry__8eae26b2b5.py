from pathlib import Path
import json
import math
import shutil
import subprocess

import numpy as np


PROJECT = Path("/home/ubuntu/Downloads/psds")
OUT = PROJECT / "output"
PROC = OUT / "processing"

RSLC_PAR = Path(
    "/home/ubuntu/Downloads/RSLC/20151212.rslc.par"
)

GEO_PAR = Path(
    "/home/ubuntu/Downloads/DEM_prep/20151212_4_1.vv.mli.par"
)

LON_RASTER = Path(
    "/home/ubuntu/Downloads/DEM_prep/20151212_4_1.rdc.lon"
)

LAT_RASTER = Path(
    "/home/ubuntu/Downloads/DEM_prep/20151212_4_1.rdc.lat"
)

DEST = PROC / "gacos_geometry"

STRICT_IDS = (
    PROC
    / "network_inversion"
    / "strict_point_ids.npy"
)

ROWS = (
    PROC
    / "point_phase_stack"
    / "rows.npy"
)

COLS = (
    PROC
    / "point_phase_stack"
    / "cols.npy"
)


errors = []


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

    if not ok:
        errors.append(
            f"{name}: {detail}"
        )

    return ok


def parse_par(path):

    out = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(
            ":",
            1,
        )

        out[
            k.strip().lower()
        ] = v.strip()

    return out


def number(d, key):

    if key not in d:
        return None

    try:
        return float(
            d[key].split()[0]
        )

    except Exception:
        return None


print("=" * 96)
print("P15-3A v3 StaMPS-style point geometry")
print()
print("USES EXISTING RADAR-COORDINATE lon/lat GRIDS")
print("USES GAMMA data2pt")
print("NO PHASE MODIFICATION")
print("NO GACOS CORRECTION")
print("=" * 96)


# =============================================================================
# 1. Input contracts
# =============================================================================

for p in (
    RSLC_PAR,
    GEO_PAR,
    LON_RASTER,
    LAT_RASTER,
    STRICT_IDS,
    ROWS,
    COLS,
):

    check(
        f"exists: {p.name}",
        p.is_file(),
        p,
    )


if errors:
    raise SystemExit(1)


geo = parse_par(
    GEO_PAR
)


ref = parse_par(
    RSLC_PAR
)


geo_width = int(
    number(
        geo,
        "range_samples",
    )
)

geo_length = int(
    number(
        geo,
        "azimuth_lines",
    )
)


expected_bytes = (
    geo_width
    *
    geo_length
    *
    4
)


print()
print(
    f"multilook geometry       : "
    f"{geo_length} x {geo_width}"
)

print(
    f"expected FLOAT bytes     : "
    f"{expected_bytes:,}"
)


check(
    "longitude raster geometry",
    LON_RASTER.stat().st_size
    ==
    expected_bytes,
    LON_RASTER.stat().st_size,
)


check(
    "latitude raster geometry",
    LAT_RASTER.stat().st_size
    ==
    expected_bytes,
    LAT_RASTER.stat().st_size,
)


if errors:
    raise SystemExit(1)


# =============================================================================
# 2. Strict points
# =============================================================================

strict_ids = np.asarray(
    np.load(
        STRICT_IDS
    ),
    dtype=np.int64,
)


all_rows = np.load(
    ROWS,
    mmap_mode="r",
)


all_cols = np.load(
    COLS,
    mmap_mode="r",
)


rows = np.asarray(
    all_rows[
        strict_ids
    ],
    dtype=np.int32,
)


cols = np.asarray(
    all_cols[
        strict_ids
    ],
    dtype=np.int32,
)


N = strict_ids.size


check(
    "strict point count",
    N == 881315,
    N,
)


print(
    f"row min/max              : "
    f"{rows.min()} / {rows.max()}"
)

print(
    f"col min/max              : "
    f"{cols.min()} / {cols.max()}"
)


# =============================================================================
# 3. IPTA point list
#
# Production pyPSDS convention:
#
#   plist[:,0] = range = col
#   plist[:,1] = azimuth = row
#
# big-endian INT32
# =============================================================================

DEST.mkdir(
    parents=True,
    exist_ok=True,
)


plist = (
    DEST
    / "strict_points.plist"
)


plist_array = np.column_stack(
    (
        cols,
        rows,
    )
).astype(
    ">i4",
    copy=False,
)


plist_array.tofile(
    plist
)


check(
    "plist byte size",
    plist.stat().st_size
    ==
    N
    *
    2
    *
    4,
    plist.stat().st_size,
)


# =============================================================================
# 4. Resolve data2pt
# =============================================================================

data2pt = shutil.which(
    "data2pt"
)


if data2pt is None:

    candidates = list(
        Path(
            "/home/ubuntu/software/GAMMA_SOFTWARE"
        ).rglob(
            "data2pt"
        )
    )

    candidates = [
        p
        for p in candidates
        if p.is_file()
    ]

    if len(
        candidates
    )
    ==
    1:

        data2pt = str(
            candidates[0]
        )


check(
    "data2pt resolved",
    data2pt is not None,
    data2pt,
)


if errors:
    raise SystemExit(1)


# =============================================================================
# 5. Sample longitude + latitude exactly like pyPSDS height sampling
# =============================================================================

lon_pt = (
    DEST
    / "longitude_deg.gamma_pt"
)

lat_pt = (
    DEST
    / "latitude_deg.gamma_pt"
)


commands = [
    (
        "longitude",
        [
            data2pt,
            str(
                LON_RASTER
            ),
            str(
                GEO_PAR
            ),
            str(
                plist
            ),
            str(
                RSLC_PAR
            ),
            str(
                lon_pt
            ),
            "1",
            "2",
        ],
    ),
    (
        "latitude",
        [
            data2pt,
            str(
                LAT_RASTER
            ),
            str(
                GEO_PAR
            ),
            str(
                plist
            ),
            str(
                RSLC_PAR
            ),
            str(
                lat_pt
            ),
            "1",
            "2",
        ],
    ),
]


for label, cmd in commands:

    print()
    print(
        label,
        ":",
    )

    print(
        "  "
        +
        " ".join(
            cmd
        )
    )


    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


    print(
        "\n".join(
            (
                proc.stdout
                or
                ""
            ).splitlines()[
                -10:
            ]
        )
    )


    check(
        f"data2pt {label}",
        proc.returncode == 0,
        proc.returncode,
    )


if errors:
    raise SystemExit(1)


# GAMMA FLOAT output
lon = np.fromfile(
    lon_pt,
    dtype=">f4",
).astype(
    np.float64
)


lat = np.fromfile(
    lat_pt,
    dtype=">f4",
).astype(
    np.float64
)


check(
    "longitude point count",
    lon.size == N,
    lon.size,
)


check(
    "latitude point count",
    lat.size == N,
    lat.size,
)


if errors:
    raise SystemExit(1)


# =============================================================================
# 6. Geographic QA
# =============================================================================

valid_ll = (
    np.isfinite(
        lon
    )
    &
    np.isfinite(
        lat
    )
    &
    (
        lon > -180
    )
    &
    (
        lon < 180
    )
    &
    (
        lat > -90
    )
    &
    (
        lat < 90
    )
)


ll_fraction = float(
    valid_ll.mean()
)


check(
    "valid lon/lat fraction",
    ll_fraction > 0.999,
    f"{ll_fraction:.9f}",
)


print()
print(
    f"longitude min/max        : "
    f"{np.nanmin(lon):.8f} / "
    f"{np.nanmax(lon):.8f}"
)

print(
    f"latitude min/max         : "
    f"{np.nanmin(lat):.8f} / "
    f"{np.nanmax(lat):.8f}"
)

print(
    f"median lon/lat           : "
    f"{np.nanmedian(lon):.8f}, "
    f"{np.nanmedian(lat):.8f}"
)


# GACOS region from accepted P15-3
inside_gacos = (
    valid_ll
    &
    (lon >= 23.6)
    &
    (lon <= 23.8)
    &
    (lat >= 38.0)
    &
    (lat <= 38.2)
)


gacos_fraction = float(
    inside_gacos.mean()
)


check(
    "strict points inside GACOS",
    gacos_fraction > 0.999,
    f"{gacos_fraction:.9f}",
)


# =============================================================================
# 7. Pointwise incidence angle -- StaMPS/GAMMA analytical geometry
#
# Our point columns are 1x1 RSLC range pixels, so use RSLC.par,
# NOT the 4:1 MLI range spacing.
# =============================================================================

required = (
    "near_range_slc",
    "range_pixel_spacing",
    "sar_to_earth_center",
    "earth_radius_below_sensor",
    "center_range_slc",
    "incidence_angle",
)


for key in required:

    check(
        f"RSLC parameter {key}",
        number(
            ref,
            key,
        )
        is not None,
        ref.get(
            key,
            None,
        ),
    )


if errors:
    raise SystemExit(1)


near_range = number(
    ref,
    "near_range_slc",
)

range_spacing = number(
    ref,
    "range_pixel_spacing",
)

sat_radius = number(
    ref,
    "sar_to_earth_center",
)

earth_radius = number(
    ref,
    "earth_radius_below_sensor",
)

center_range = number(
    ref,
    "center_range_slc",
)

center_incidence_par = number(
    ref,
    "incidence_angle",
)


# First verify the spherical geometry equation independently at center range.
center_cos = (
    sat_radius**2
    -
    earth_radius**2
    -
    center_range**2
) / (
    2.0
    *
    earth_radius
    *
    center_range
)


center_cos = np.clip(
    center_cos,
    -1.0,
    1.0,
)


center_incidence_calc = math.degrees(
    math.acos(
        center_cos
    )
)


print()
print(
    f"center incidence PAR     : "
    f"{center_incidence_par:.6f} deg"
)

print(
    f"center incidence formula : "
    f"{center_incidence_calc:.6f} deg"
)


check(
    "StaMPS incidence geometry parity",
    abs(
        center_incidence_calc
        -
        center_incidence_par
    )
    <
    0.1,
    (
        f"diff="
        f"{abs(center_incidence_calc-center_incidence_par):.6f} deg"
    ),
)


if errors:
    raise SystemExit(1)


# 0-based numpy col -> first RSLC sample at col=0.
slant_range = (
    near_range
    +
    cols.astype(
        np.float64
    )
    *
    range_spacing
)


cos_inc = (
    sat_radius**2
    -
    earth_radius**2
    -
    slant_range**2
) / (
    2.0
    *
    earth_radius
    *
    slant_range
)


cos_inc = np.clip(
    cos_inc,
    -1.0,
    1.0,
)


incidence_deg = np.degrees(
    np.arccos(
        cos_inc
    )
)


valid_inc = (
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
    valid_inc.mean()
)


check(
    "valid point incidence fraction",
    inc_fraction > 0.999,
    f"{inc_fraction:.9f}",
)


q = np.percentile(
    incidence_deg[
        valid_inc
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
        f"{x:.6f}"
        for x in q
    )
    +
    " deg"
)


# =============================================================================
# 8. Final accepted geometry
# =============================================================================

accepted = (
    inside_gacos
    &
    valid_inc
)


accepted_fraction = float(
    accepted.mean()
)


check(
    "final GACOS geometry coverage",
    accepted_fraction > 0.999,
    f"{accepted_fraction:.9f}",
)


if errors:
    raise SystemExit(1)


# =============================================================================
# 9. Save derived point geometry
# =============================================================================

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
    rows,
)


np.save(
    DEST
    / "radar_col.npy",
    cols,
)


np.save(
    DEST
    / "longitude_deg.npy",
    lon,
)


np.save(
    DEST
    / "latitude_deg.npy",
    lat,
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
    / "valid_gacos_geometry_mask.npy",
    accepted,
)


manifest = {
    "format":
        "pyPSDS-GAMMA-P15-3A-StaMPS-point-geometry-v3",

    "status":
        "PASS_RADAR_POINT_GEOLOCATION",

    "method":
        "StaMPS_GAMMA_style_point_geometry",

    "points":
        int(
            N
        ),

    "longitude_source":
        str(
            LON_RASTER
        ),

    "latitude_source":
        str(
            LAT_RASTER
        ),

    "source_geometry_par":
        str(
            GEO_PAR
        ),

    "reference_rslc_par":
        str(
            RSLC_PAR
        ),

    "sampling":
        {
            "command":
                "GAMMA data2pt",

            "plist_order":
                "range_col, azimuth_row",

            "plist_dtype":
                "big-endian int32",

            "output_dtype":
                "big-endian float32",
        },

    "incidence":
        {
            "method":
                "StaMPS_spherical_radar_geometry",

            "formula":
                (
                    "acos((se^2-re^2-rg^2)"
                    "/(2*re*rg))"
                ),

            "range_geometry":
                (
                    "rg=near_range_slc"
                    "+radar_col*range_pixel_spacing"
                ),

            "p01_deg":
                float(
                    q[0]
                ),

            "p05_deg":
                float(
                    q[1]
                ),

            "p50_deg":
                float(
                    q[2]
                ),

            "p95_deg":
                float(
                    q[3]
                ),

            "p99_deg":
                float(
                    q[4]
                ),
        },

    "valid_lonlat_fraction":
        ll_fraction,

    "gacos_coverage_fraction":
        gacos_fraction,

    "incidence_valid_fraction":
        inc_fraction,

    "accepted_fraction":
        accepted_fraction,

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
    "\n"
)


print()
print("=" * 96)
print("P15-3A v3 RESULT")
print("=" * 96)

print(
    "status                    : "
    "PASS_RADAR_POINT_GEOLOCATION"
)

print(
    f"strict points             : "
    f"{N:,}"
)

print(
    f"valid lon/lat             : "
    f"{100*ll_fraction:.6f}%"
)

print(
    f"inside GACOS              : "
    f"{100*gacos_fraction:.6f}%"
)

print(
    f"valid incidence           : "
    f"{100*inc_fraction:.6f}%"
)

print(
    f"accepted geometry         : "
    f"{100*accepted_fraction:.6f}%"
)

print(
    f"incidence p50             : "
    f"{q[2]:.6f} deg"
)

print(
    f"manifest                  : "
    f"{manifest_path}"
)

print(
    "next step                 : "
    "P15-4_GACOS_POINT_SAMPLING_SMOKE"
)

print("=" * 96)
print(
    "P15-3A FINAL RESULT: "
    "PASS_RADAR_POINT_GEOLOCATION"
)
print("=" * 96)

