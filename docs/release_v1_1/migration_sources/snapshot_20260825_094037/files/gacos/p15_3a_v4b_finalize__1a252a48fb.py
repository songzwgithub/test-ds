from pathlib import Path
import json
import math
import numpy as np

PROJECT = Path("/home/ubuntu/Downloads/psds")
PROC = PROJECT / "output" / "processing"
DEST = PROC / "gacos_geometry"

RSLC_PAR = Path(
    "/home/ubuntu/Downloads/RSLC/20151212.rslc.par"
)

lon_bin = DEST / "longitude_deg.gamma_pt"
lat_bin = DEST / "latitude_deg.gamma_pt"


def read_par(path):
    d = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip().lower()] = v.strip()
    return d


def val(d, key):
    return float(d[key].split()[0])


strict_ids = np.asarray(
    np.load(
        PROC / "network_inversion" / "strict_point_ids.npy"
    ),
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

n = strict_ids.size

assert n == 881315


# ------------------------------------------------------------
# Existing data2pt products
# ------------------------------------------------------------

lon = np.fromfile(
    lon_bin,
    dtype=">f4",
).astype(np.float64)

lat = np.fromfile(
    lat_bin,
    dtype=">f4",
).astype(np.float64)

assert lon.size == n
assert lat.size == n


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


# ------------------------------------------------------------
# Exact StaMPS/GAMMA formula
# mt_ml_select_gamma.m
# ------------------------------------------------------------

p = read_par(RSLC_PAR)

rgn = val(p, "near_range_slc")
rps = val(p, "range_pixel_spacing")
se = val(p, "sar_to_earth_center")
re = val(p, "earth_radius_below_sensor")

rgc = val(p, "center_range_slc")
gamma_center_inc = val(p, "incidence_angle")


def stamps_incidence(rg):
    x = (
        se * se
        - re * re
        - rg * rg
    ) / (
        2.0 * re * rg
    )

    return np.degrees(
        np.arccos(
            np.clip(x, -1.0, 1.0)
        )
    )


# Current Python radar columns are zero-based.
# One-pixel convention difference is negligible, but use col directly
# consistently with the existing pyPSDS radar-coordinate system.
rg = (
    rgn
    + cols.astype(np.float64) * rps
)

inc = stamps_incidence(rg)


# Diagnostic only.
stamps_center_inc = float(
    stamps_incidence(rgc)
)

center_difference = (
    stamps_center_inc
    - gamma_center_inc
)


valid_inc = (
    np.isfinite(inc)
    & (inc > 10.0)
    & (inc < 80.0)
)

accepted = (
    inside_gacos
    & valid_inc
)


ll_frac = float(valid_ll.mean())
gacos_frac = float(inside_gacos.mean())
inc_frac = float(valid_inc.mean())
accepted_frac = float(accepted.mean())


q = np.percentile(
    inc[valid_inc],
    [1, 5, 50, 95, 99],
)


print("=" * 88)
print("P15-3A v4b StaMPS/GAMMA geometry finalize")
print("=" * 88)

print(f"strict points             : {n:,}")
print(f"valid lon/lat             : {ll_frac*100:.6f}%")
print(f"inside GACOS              : {gacos_frac*100:.6f}%")
print()

print(
    "longitude range          : "
    f"{lon[valid_ll].min():.8f} .. "
    f"{lon[valid_ll].max():.8f}"
)

print(
    "latitude range           : "
    f"{lat[valid_ll].min():.8f} .. "
    f"{lat[valid_ll].max():.8f}"
)

print()
print(
    f"GAMMA parameter incidence : "
    f"{gamma_center_inc:.6f} deg"
)

print(
    f"StaMPS spherical center   : "
    f"{stamps_center_inc:.6f} deg"
)

print(
    f"diagnostic difference     : "
    f"{center_difference:+.6f} deg"
)

print(
    "NOTE                      : "
    "diagnostic only; NOT a failure gate"
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
    f"valid incidence           : "
    f"{inc_frac*100:.6f}%"
)

print(
    f"accepted geometry         : "
    f"{accepted_frac*100:.6f}%"
)


if ll_frac <= 0.999:
    raise RuntimeError("lon/lat coverage failed")

if gacos_frac <= 0.999:
    raise RuntimeError("GACOS geographic coverage failed")

if inc_frac <= 0.999:
    raise RuntimeError("StaMPS incidence coverage failed")

if accepted_frac <= 0.999:
    raise RuntimeError("final point geometry coverage failed")


# ------------------------------------------------------------
# Save accepted derived geometry
# ------------------------------------------------------------

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
    DEST / "incidence_angle_stamps_deg.npy",
    inc.astype(np.float32),
)

np.save(
    DEST / "valid_gacos_geometry_mask.npy",
    accepted,
)


manifest = {
    "format":
        "pyPSDS-GAMMA-P15-3A-StaMPS-geometry-v4b",

    "status":
        "PASS_RADAR_POINT_GEOLOCATION",

    "longitude_latitude_method":
        "GAMMA_data2pt_existing_rdc_lon_lat",

    "incidence_method":
        "StaMPS_mt_ml_select_gamma_spherical_geometry",

    "incidence_formula":
        (
            "acos((se^2-re^2-rg^2)/(2*re*rg))"
        ),

    "points":
        int(n),

    "valid_lonlat_fraction":
        ll_frac,

    "gacos_coverage_fraction":
        gacos_frac,

    "incidence_valid_fraction":
        inc_frac,

    "accepted_fraction":
        accepted_frac,

    "gamma_parameter_center_incidence_deg":
        gamma_center_inc,

    "stamps_spherical_center_incidence_deg":
        stamps_center_inc,

    "center_difference_deg":
        center_difference,

    "center_difference_role":
        "diagnostic_only_not_failure_gate",

    "incidence_percentiles_deg": {
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p50": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
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
    + "\n"
)


print()
print(
    "manifest                  :",
    manifest_path,
)

print()
print("=" * 88)
print(
    "P15-3A FINAL RESULT: "
    "PASS_RADAR_POINT_GEOLOCATION"
)
print("=" * 88)
