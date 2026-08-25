from pathlib import Path
import json, math, shutil, subprocess
import numpy as np

PROJECT = Path("/home/ubuntu/Downloads/psds")
PROC = PROJECT / "output" / "processing"
DEM = Path("/home/ubuntu/Downloads/DEM_prep")

RSLC_PAR = Path("/home/ubuntu/Downloads/RSLC/20151212.rslc.par")
GEO_PAR = DEM / "20151212_4_1.vv.mli.par"
LON = DEM / "20151212_4_1.rdc.lon"
LAT = DEM / "20151212_4_1.rdc.lat"

DEST = PROC / "gacos_geometry"
DEST.mkdir(parents=True, exist_ok=True)


def read_par(path):
    d = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip().lower()] = v.strip()
    return d


def val(d, key):
    return float(d[key].split()[0])


# ----------------------------------------------------------------------
# 1. Strict production points
# ----------------------------------------------------------------------

strict_ids = np.asarray(
    np.load(PROC / "network_inversion" / "strict_point_ids.npy"),
    dtype=np.int64,
)

all_rows = np.load(
    PROC / "point_phase_stack" / "rows.npy",
    mmap_mode="r",
)

all_cols = np.load(
    PROC / "point_phase_stack" / "cols.npy",
    mmap_mode="r",
)

rows = np.asarray(all_rows[strict_ids], dtype=np.int32)
cols = np.asarray(all_cols[strict_ids], dtype=np.int32)

n = len(strict_ids)

assert n == 881315, n

print("strict points             :", n)
print("row min/max               :", rows.min(), rows.max())
print("col min/max               :", cols.min(), cols.max())


# ----------------------------------------------------------------------
# 2. Verify 4:1 lon/lat raster geometry
# ----------------------------------------------------------------------

geo = read_par(GEO_PAR)

gw = int(val(geo, "range_samples"))
gh = int(val(geo, "azimuth_lines"))

expected = gw * gh * 4

print("multilook geometry        :", gh, "x", gw)
print("expected FLOAT bytes      :", expected)

assert LON.stat().st_size == expected, (
    LON.stat().st_size,
    expected,
)

assert LAT.stat().st_size == expected, (
    LAT.stat().st_size,
    expected,
)


# ----------------------------------------------------------------------
# 3. IPTA point list
#
# GAMMA/IPTA:
#   first  = range pixel  = col
#   second = azimuth line = row
# ----------------------------------------------------------------------

plist = DEST / "strict_points.plist"

np.column_stack(
    (cols, rows)
).astype(
    ">i4"
).tofile(
    plist
)

assert plist.stat().st_size == n * 8


# ----------------------------------------------------------------------
# 4. Resolve data2pt
# ----------------------------------------------------------------------

data2pt = shutil.which("data2pt")

if data2pt is None:
    cands = [
        p
        for p in Path(
            "/home/ubuntu/software/GAMMA_SOFTWARE"
        ).rglob("data2pt")
        if p.is_file()
    ]

    if len(cands) != 1:
        raise RuntimeError(
            f"Cannot resolve data2pt uniquely: {cands[:10]}"
        )

    data2pt = str(cands[0])

print("data2pt                  :", data2pt)


# ----------------------------------------------------------------------
# 5. Sample lon / lat
# ----------------------------------------------------------------------

lon_bin = DEST / "longitude_deg.gamma_pt"
lat_bin = DEST / "latitude_deg.gamma_pt"

jobs = [
    (LON, lon_bin, "longitude"),
    (LAT, lat_bin, "latitude"),
]

for src, dst, name in jobs:

    cmd = [
        data2pt,
        str(src),
        str(GEO_PAR),
        str(plist),
        str(RSLC_PAR),
        str(dst),
        "1",
        "2",
    ]

    print()
    print("$", " ".join(cmd))

    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(
        "\n".join(
            (p.stdout or "").splitlines()[-12:]
        )
    )

    if p.returncode != 0:
        raise RuntimeError(
            f"data2pt {name} failed: {p.returncode}"
        )


lon = np.fromfile(
    lon_bin,
    dtype=">f4",
).astype(
    np.float64
)

lat = np.fromfile(
    lat_bin,
    dtype=">f4",
).astype(
    np.float64
)

assert lon.size == n, (lon.size, n)
assert lat.size == n, (lat.size, n)


# ----------------------------------------------------------------------
# 6. Lon/lat QA
# ----------------------------------------------------------------------

valid_ll = (
    np.isfinite(lon)
    & np.isfinite(lat)
    & (lon > -180)
    & (lon < 180)
    & (lat > -90)
    & (lat < 90)
)

inside_gacos = (
    valid_ll
    & (lon >= 23.6)
    & (lon <= 23.8)
    & (lat >= 38.0)
    & (lat <= 38.2)
)

ll_frac = float(valid_ll.mean())
gacos_frac = float(inside_gacos.mean())

print()
print("valid lon/lat             :", f"{ll_frac*100:.6f}%")
print("inside GACOS              :", f"{gacos_frac*100:.6f}%")
print(
    "lon range                 :",
    f"{np.nanmin(lon):.8f} .. {np.nanmax(lon):.8f}",
)
print(
    "lat range                 :",
    f"{np.nanmin(lat):.8f} .. {np.nanmax(lat):.8f}",
)


# ----------------------------------------------------------------------
# 7. StaMPS/GAMMA pointwise incidence
#
# rg = near_range_slc + range_pixel * range_pixel_spacing
#
# incidence =
# acos((se^2 - re^2 - rg^2) / (2 re rg))
# ----------------------------------------------------------------------

r = read_par(RSLC_PAR)

near = val(r, "near_range_slc")
dr = val(r, "range_pixel_spacing")
se = val(r, "sar_to_earth_center")
re = val(r, "earth_radius_below_sensor")

center_range = val(r, "center_range_slc")
center_inc_par = val(r, "incidence_angle")


def incidence_from_range(rg):
    c = (
        se * se
        - re * re
        - rg * rg
    ) / (
        2.0 * re * rg
    )

    return np.degrees(
        np.arccos(
            np.clip(c, -1.0, 1.0)
        )
    )


# First validate the formula against the parameter-file center angle.
center_inc_calc = float(
    incidence_from_range(center_range)
)

print()
print(
    "center incidence PAR      :",
    f"{center_inc_par:.6f} deg",
)
print(
    "center incidence formula  :",
    f"{center_inc_calc:.6f} deg",
)

center_diff = abs(
    center_inc_calc
    - center_inc_par
)

print(
    "center incidence diff     :",
    f"{center_diff:.6f} deg",
)

if center_diff >= 0.1:
    raise RuntimeError(
        "StaMPS incidence formula does not match RSLC geometry: "
        f"{center_diff:.6f} deg"
    )


# Current cols are 0-based RSLC range pixels.
slant = (
    near
    + cols.astype(np.float64) * dr
)

inc = incidence_from_range(
    slant
)

valid_inc = (
    np.isfinite(inc)
    & (inc > 10.0)
    & (inc < 80.0)
)

inc_frac = float(
    valid_inc.mean()
)

q = np.percentile(
    inc[valid_inc],
    [1, 5, 50, 95, 99],
)

print()
print(
    "incidence p01/p05/p50/p95/p99:"
)
print(
    "  "
    + " / ".join(
        f"{x:.6f}"
        for x in q
    )
    + " deg"
)

print(
    "valid incidence           :",
    f"{inc_frac*100:.6f}%",
)


# ----------------------------------------------------------------------
# 8. Final geometry gate
# ----------------------------------------------------------------------

accepted = (
    inside_gacos
    & valid_inc
)

accepted_fraction = float(
    accepted.mean()
)

print(
    "accepted geometry         :",
    f"{accepted_fraction*100:.6f}%",
)

if ll_frac <= 0.999:
    raise RuntimeError(
        f"lon/lat coverage too low: {ll_frac}"
    )

if gacos_frac <= 0.999:
    raise RuntimeError(
        f"GACOS coverage too low: {gacos_frac}"
    )

if inc_frac <= 0.999:
    raise RuntimeError(
        f"incidence coverage too low: {inc_frac}"
    )

if accepted_fraction <= 0.999:
    raise RuntimeError(
        f"accepted geometry too low: {accepted_fraction}"
    )


# ----------------------------------------------------------------------
# 9. Save derived geometry only after all gates pass
# ----------------------------------------------------------------------

np.save(
    DEST / "strict_point_ids.npy",
    strict_ids.astype(np.int32),
)

np.save(
    DEST / "radar_row.npy",
    rows,
)

np.save(
    DEST / "radar_col.npy",
    cols,
)

np.save(
    DEST / "longitude_deg.npy",
    lon,
)

np.save(
    DEST / "latitude_deg.npy",
    lat,
)

np.save(
    DEST / "incidence_angle_deg.npy",
    inc.astype(np.float32),
)

np.save(
    DEST / "valid_gacos_geometry_mask.npy",
    accepted,
)

manifest = {
    "status": "PASS_RADAR_POINT_GEOLOCATION",
    "method": (
        "StaMPS_GAMMA_style_"
        "data2pt_lonlat_plus_analytic_incidence"
    ),
    "points": int(n),
    "valid_lonlat_fraction": ll_frac,
    "gacos_coverage_fraction": gacos_frac,
    "incidence_valid_fraction": inc_frac,
    "accepted_fraction": accepted_fraction,
    "center_incidence_par_deg": center_inc_par,
    "center_incidence_calc_deg": center_inc_calc,
    "incidence_p01_p05_p50_p95_p99_deg": [
        float(x)
        for x in q
    ],
    "next_step": "P15-4_GACOS_POINT_SAMPLING_SMOKE",
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
    + "\n"
)

print()
print("=" * 80)
print("P15-3A FINAL RESULT: PASS_RADAR_POINT_GEOLOCATION")
print("=" * 80)
print("manifest :", manifest_path)
