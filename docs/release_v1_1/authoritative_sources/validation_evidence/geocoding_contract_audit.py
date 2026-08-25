from pathlib import Path
import json
import re
import os

import numpy as np


ROOT = Path("/home/ubuntu/Downloads")
PSDS = ROOT / "psds"
PROC = PSDS / "output/processing"
DEM = ROOT / "DEM_prep"

LON = PROC / "gacos_geometry/longitude_deg.npy"
LAT = PROC / "gacos_geometry/latitude_deg.npy"
PLIST = PROC / "gacos_geometry/strict_points.plist"

VEL = (
    PROC
    / "final_los_products"
    / "los_velocity_toward_satellite_mm_per_year.npy"
)

CUM = (
    PROC
    / "final_los_products"
    / "los_cumulative_toward_satellite_mm.npy"
)

P7A = (
    PROC
    / "final_los_products"
    / "p15_7a_final_point_products_manifest.json"
)

OUTDIR = (
    PROC
    / "final_los_geocoding"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT = (
    OUTDIR
    / "p15_7b0_geocoding_contract_audit.json"
)


NUM = re.compile(
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)


def parse_par(path):
    d = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(
            ":",
            1,
        )

        d[k.strip().lower()] = v.strip()

    return d


def scalar(d, names):

    for name in names:

        v = d.get(
            name.lower()
        )

        if v is None:
            continue

        m = NUM.search(v)

        if m:
            return float(
                m.group(0)
            )

    return None


def integer(d, names):

    x = scalar(
        d,
        names
    )

    if x is None:
        return None

    return int(
        round(x)
    )


# ================================================================
# Inputs
# ================================================================

for p in (
    DEM,
    LON,
    LAT,
    PLIST,
    VEL,
    CUM,
    P7A,
):
    if not p.exists():
        raise FileNotFoundError(p)


m7a = json.loads(
    P7A.read_text()
)

if (
    m7a.get("status")
    != "PASS_FINAL_LOS_POINT_PRODUCTS"
):
    raise RuntimeError(
        "P15-7A not PASS"
    )


lon = np.load(
    LON,
    mmap_mode="r",
).astype(
    np.float64
)

lat = np.load(
    LAT,
    mmap_mode="r",
).astype(
    np.float64
)

vel = np.load(
    VEL,
    mmap_mode="r",
)

cum = np.load(
    CUM,
    mmap_mode="r",
)

plist = np.fromfile(
    PLIST,
    dtype=">i4",
).reshape(
    -1,
    2,
)


n = lon.size


if not (
    lat.size == n
    and vel.size == n
    and cum.size == n
    and plist.shape == (n, 2)
):
    raise RuntimeError(
        "point-product contract failed"
    )


# ================================================================
# Inventory all GAMMA parameter files
# ================================================================

par_files = sorted(
    DEM.rglob("*.par")
)


candidates = []


for p in par_files:

    d = parse_par(p)

    width = integer(
        d,
        (
            "width",
            "range_samples",
            "samples",
        ),
    )

    nlines = integer(
        d,
        (
            "nlines",
            "azimuth_lines",
            "lines",
        ),
    )

    projection_raw = (
        d.get("dem_projection")
        or d.get("map_projection")
        or d.get("projection")
        or ""
    ).strip()

    projection = (
        projection_raw.split()[0]
        if projection_raw.split()
        else ""
    )

    corner_lon = scalar(
        d,
        (
            "corner_lon",
            "longitude",
        ),
    )

    corner_lat = scalar(
        d,
        (
            "corner_lat",
            "latitude",
        ),
    )

    post_lon = scalar(
        d,
        (
            "post_lon",
            "longitude_post",
        ),
    )

    post_lat = scalar(
        d,
        (
            "post_lat",
            "latitude_post",
        ),
    )

    corner_east = scalar(
        d,
        (
            "corner_east",
            "corner_easting",
        ),
    )

    corner_north = scalar(
        d,
        (
            "corner_north",
            "corner_northing",
        ),
    )

    post_east = scalar(
        d,
        (
            "post_east",
            "easting_post",
        ),
    )

    post_north = scalar(
        d,
        (
            "post_north",
            "northing_post",
        ),
    )

    map_like = (
        width is not None
        and nlines is not None
        and (
            (
                corner_lon is not None
                and corner_lat is not None
                and post_lon is not None
                and post_lat is not None
            )
            or
            (
                corner_east is not None
                and corner_north is not None
                and post_east is not None
                and post_north is not None
            )
            or
            projection != ""
        )
    )

    if not map_like:
        continue

    candidates.append(
        {
            "path": str(p),
            "name": p.name,
            "width": width,
            "nlines": nlines,
            "projection": projection,
            "corner_lon": corner_lon,
            "corner_lat": corner_lat,
            "post_lon": post_lon,
            "post_lat": post_lat,
            "corner_east": corner_east,
            "corner_north": corner_north,
            "post_east": post_east,
            "post_north": post_north,
        }
    )


# ================================================================
# Lookup / geocoding related files
# ================================================================

lookup_files = []

keywords = (
    ".lt",
    "lookup",
    "map_to_rdc",
    "rdc_to_map",
    "seg",
)


for p in DEM.rglob("*"):

    if not p.is_file():
        continue

    nl = p.name.lower()

    if any(
        k in nl
        for k in keywords
    ):
        lookup_files.append(
            {
                "path": str(p),
                "size_bytes": p.stat().st_size,
            }
        )


# ================================================================
# Rank geographic grids by whether all strict points fall inside.
#
# This is audit only. No interpolation and no raster is generated.
# ================================================================

evaluated = []


for c in candidates:

    if None in (
        c["corner_lon"],
        c["corner_lat"],
        c["post_lon"],
        c["post_lat"],
    ):
        continue

    if (
        c["post_lon"] == 0
        or c["post_lat"] == 0
    ):
        continue

    width = c["width"]
    nlines = c["nlines"]

    x0 = c["corner_lon"]
    y0 = c["corner_lat"]

    dx = c["post_lon"]
    dy = c["post_lat"]

    # GAMMA DEM parameter convention is grid-node coordinate.
    col_float = (
        lon - x0
    ) / dx

    row_float = (
        lat - y0
    ) / dy

    inside = (
        (col_float >= -0.5)
        &
        (col_float <= width - 0.5)
        &
        (row_float >= -0.5)
        &
        (row_float <= nlines - 0.5)
    )

    inside_fraction = float(
        np.mean(inside)
    )

    nearest_col = np.rint(
        col_float
    )

    nearest_row = np.rint(
        row_float
    )

    valid = inside

    if np.any(valid):

        grid_lon = (
            x0
            +
            nearest_col[valid]
            *
            dx
        )

        grid_lat = (
            y0
            +
            nearest_row[valid]
            *
            dy
        )

        # Local metric approximation only for reporting subpixel
        # geolocation mismatch; not used to generate products.
        lat0 = float(
            np.median(
                lat[valid]
            )
        )

        dlon_m = (
            np.deg2rad(
                lon[valid]
                -
                grid_lon
            )
            *
            6371008.8
            *
            np.cos(
                np.deg2rad(lat0)
            )
        )

        dlat_m = (
            np.deg2rad(
                lat[valid]
                -
                grid_lat
            )
            *
            6371008.8
        )

        dist = np.hypot(
            dlon_m,
            dlat_m
        )

        q = np.percentile(
            dist,
            (
                50,
                95,
                99,
                100,
            ),
        )

        # Collision audit for nearest-node scatter.
        rc = (
            nearest_row[valid].astype(
                np.int64
            )
            *
            width
            +
            nearest_col[valid].astype(
                np.int64
            )
        )

        unique_cells = np.unique(
            rc
        ).size

        collision_points = int(
            np.count_nonzero(valid)
            -
            unique_cells
        )

        collision_fraction = float(
            collision_points
            /
            np.count_nonzero(valid)
        )

    else:

        q = np.asarray(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        )

        unique_cells = 0
        collision_points = 0
        collision_fraction = np.nan


    pixel_ns_m = abs(
        dy
    ) * 111320.0

    pixel_ew_m = (
        abs(dx)
        *
        111320.0
        *
        np.cos(
            np.deg2rad(
                np.median(lat)
            )
        )
    )


    score = (
        inside_fraction
        -
        (
            0.05
            *
            collision_fraction
            if np.isfinite(
                collision_fraction
            )
            else 1.0
        )
    )


    evaluated.append(
        {
            **c,

            "inside_fraction":
                inside_fraction,

            "approx_pixel_ew_m":
                float(pixel_ew_m),

            "approx_pixel_ns_m":
                float(pixel_ns_m),

            "nearest_grid_distance_p50_m":
                float(q[0]),

            "nearest_grid_distance_p95_m":
                float(q[1]),

            "nearest_grid_distance_p99_m":
                float(q[2]),

            "nearest_grid_distance_max_m":
                float(q[3]),

            "unique_cells":
                int(unique_cells),

            "collision_points":
                collision_points,

            "collision_fraction":
                collision_fraction,

            "ranking_score":
                float(score),
        }
    )


evaluated.sort(
    key=lambda x:
        x[
            "ranking_score"
        ],
    reverse=True,
)


best = (
    evaluated[0]
    if evaluated
    else None
)


# ================================================================
# Binary size cross-check against candidate map grids
#
# Common GAMMA lookup table:
# complex float32 = 8 bytes / map pixel.
#
# Do not assume semantics; only report exact size matches.
# ================================================================

size_matches = []


for lf in lookup_files:

    size = lf[
        "size_bytes"
    ]

    matches = []

    for c in candidates:

        npix = (
            c["width"]
            *
            c["nlines"]
        )

        if size in (
            npix * 4,
            npix * 8,
            npix * 16,
        ):
            matches.append(
                {
                    "par": c["path"],
                    "width": c["width"],
                    "nlines": c["nlines"],
                    "bytes_per_pixel": (
                        size / npix
                    ),
                }
            )

    if matches:

        size_matches.append(
            {
                "file": lf["path"],
                "size_bytes": size,
                "matches": matches,
            }
        )


# ================================================================
# Decision
# ================================================================

if best is None:

    recommendation = (
        "NO_GEOGRAPHIC_GAMMA_GRID_IDENTIFIED"
    )

elif (
    best["inside_fraction"] == 1.0
    and
    best["collision_fraction"] <= 0.01
):

    recommendation = (
        "USE_NATIVE_GAMMA_GRID_POINT_PRESERVING_SCATTER"
    )

else:

    recommendation = (
        "REQUIRE_LOOKUP_TABLE_GEOCODING_BEFORE_GEOTIFF"
    )


manifest = {
    "status":
        "PASS_GEOCODING_CONTRACT_AUDIT",

    "points":
        int(n),

    "point_extent":
        {
            "lon_min": float(lon.min()),
            "lon_max": float(lon.max()),
            "lat_min": float(lat.min()),
            "lat_max": float(lat.max()),
        },

    "dem_prep":
        str(DEM),

    "map_parameter_candidates":
        candidates,

    "evaluated_geographic_grids":
        evaluated,

    "best_geographic_grid":
        best,

    "lookup_related_files":
        lookup_files,

    "lookup_size_matches":
        size_matches,

    "recommendation":
        recommendation,

    "products_modified":
        False,
}


REPORT.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n"
)


# ================================================================
# Print
# ================================================================

print("=" * 100)
print("P15-7B0 GAMMA GEOCODING CONTRACT AUDIT")
print("=" * 100)

print(
    "points                         :",
    f"{n:,}",
)

print(
    "point longitude                :",
    f"{lon.min():.10f} .. {lon.max():.10f}",
)

print(
    "point latitude                 :",
    f"{lat.min():.10f} .. {lat.max():.10f}",
)

print()

print(
    "parameter files scanned        :",
    len(par_files),
)

print(
    "map-grid candidates            :",
    len(candidates),
)

print(
    "lookup-related files           :",
    len(lookup_files),
)

print(
    "lookup exact-size matches      :",
    len(size_matches),
)

print()

if evaluated:

    print("TOP GEOGRAPHIC GRID CANDIDATES")
    print("-" * 100)

    for c in evaluated[:10]:

        print(
            Path(
                c["path"]
            ).name
        )

        print(
            "  path                         :",
            c["path"],
        )

        print(
            "  projection                   :",
            c["projection"],
        )

        print(
            "  size                         :",
            f"{c['width']} x {c['nlines']}",
        )

        print(
            "  corner lon/lat               :",
            c["corner_lon"],
            c["corner_lat"],
        )

        print(
            "  post lon/lat                 :",
            c["post_lon"],
            c["post_lat"],
        )

        print(
            "  approx pixel EW/NS           :",
            (
                f"{c['approx_pixel_ew_m']:.3f} / "
                f"{c['approx_pixel_ns_m']:.3f} m"
            ),
        )

        print(
            "  strict points inside         :",
            f"{100*c['inside_fraction']:.6f}%",
        )

        print(
            "  nearest grid dist p50/95/99  :",
            (
                f"{c['nearest_grid_distance_p50_m']:.3f} / "
                f"{c['nearest_grid_distance_p95_m']:.3f} / "
                f"{c['nearest_grid_distance_p99_m']:.3f} m"
            ),
        )

        print(
            "  unique cells                 :",
            f"{c['unique_cells']:,}",
        )

        print(
            "  collisions                   :",
            (
                f"{c['collision_points']:,} "
                f"({100*c['collision_fraction']:.6f}%)"
            ),
        )

        print()

else:

    print(
        "No longitude/latitude GAMMA "
        "map-grid candidate found."
    )


print("-" * 100)

print(
    "recommendation                 :",
    recommendation,
)

print(
    "report                         :",
    REPORT,
)

print(
    "products modified              :",
    False,
)

print("=" * 100)

print(
    "P15-7B0 FINAL RESULT: "
    "PASS_GEOCODING_CONTRACT_AUDIT"
)

print("=" * 100)
