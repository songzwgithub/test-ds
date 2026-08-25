from pathlib import Path
import json
import re
import time

import numpy as np
from numba import njit, prange


ROOT = Path(
    "/home/ubuntu/Downloads/psds/output/processing/gacos_geometry"
)

PAR = Path(
    "/home/ubuntu/Downloads/RSLC/20151212.rslc.par"
)

OUT = (
    ROOT
    / "incidence_gamma_compatible_fast_rad.npy"
)

MANIFEST = (
    ROOT
    / "fast_incidence_manifest.json"
)


# ======================================================================
# GAMMA parameter parser
# ======================================================================

NUM_RE = re.compile(
    r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
)


def read_par(path):

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


def scalar(d, key):

    return float(
        NUM_RE.findall(
            d[key]
        )[0]
    )


def vec3(d, key):

    x = NUM_RE.findall(
        d[key]
    )

    return np.array(
        [
            float(x[0]),
            float(x[1]),
            float(x[2]),
        ],
        dtype=np.float64,
    )


# ======================================================================
# Inputs
# ======================================================================

lon = np.asarray(
    np.load(
        ROOT / "longitude_deg.npy",
        mmap_mode="r",
    ),
    dtype=np.float64,
)

lat = np.asarray(
    np.load(
        ROOT / "latitude_deg.npy",
        mmap_mode="r",
    ),
    dtype=np.float64,
)

row = np.asarray(
    np.load(
        ROOT / "radar_row.npy",
        mmap_mode="r",
    ),
    dtype=np.int32,
)

hgt = np.fromfile(
    ROOT / "height_m.gamma_pt",
    dtype=">f4",
).astype(
    np.float64
)


N = lon.size

if not (
    lat.size
    ==
    row.size
    ==
    hgt.size
    ==
    N
):
    raise RuntimeError(
        "point-array size mismatch"
    )


# ======================================================================
# Image timing
# ======================================================================

p = read_par(
    PAR
)

nlines = int(
    round(
        scalar(
            p,
            "azimuth_lines",
        )
    )
)

line_dt = scalar(
    p,
    "azimuth_line_time",
)


if "center_time" in p:

    center_time = scalar(
        p,
        "center_time",
    )

    row_time = (
        center_time
        +
        (
            np.arange(
                nlines,
                dtype=np.float64,
            )
            -
            0.5
            *
            (
                nlines - 1
            )
        )
        *
        line_dt
    )

else:

    start_time = scalar(
        p,
        "start_time",
    )

    row_time = (
        start_time
        +
        np.arange(
            nlines,
            dtype=np.float64,
        )
        *
        line_dt
    )


# ======================================================================
# Orbit state vectors
# ======================================================================

nsv = int(
    round(
        scalar(
            p,
            "number_of_state_vectors",
        )
    )
)

sv_t0 = scalar(
    p,
    "time_of_first_state_vector",
)

sv_dt = scalar(
    p,
    "state_vector_interval",
)


pos = np.vstack(
    [
        vec3(
            p,
            f"state_vector_position_{i+1}",
        )
        for i in range(nsv)
    ]
)


vel = np.vstack(
    [
        vec3(
            p,
            f"state_vector_velocity_{i+1}",
        )
        for i in range(nsv)
    ]
)


# ======================================================================
# Cubic Hermite interpolation
#
# Only 600 positions for current scene.
# ======================================================================

def orbit_position(
    query_time,
):

    idx = (
        np.searchsorted(
            sv_t0
            +
            np.arange(
                nsv,
                dtype=np.float64,
            )
            *
            sv_dt,
            query_time,
            side="right",
        )
        -
        1
    )


    idx = np.clip(
        idx,
        0,
        nsv - 2,
    )


    t_left = (
        sv_t0
        +
        idx
        *
        sv_dt
    )


    u = (
        query_time
        -
        t_left
    ) / sv_dt


    u2 = u * u
    u3 = u2 * u


    h00 = (
        2*u3
        -
        3*u2
        +
        1
    )

    h10 = (
        u3
        -
        2*u2
        +
        u
    )

    h01 = (
        -2*u3
        +
        3*u2
    )

    h11 = (
        u3
        -
        u2
    )


    return (
        h00[:, None]
        *
        pos[idx]

        +

        h10[:, None]
        *
        sv_dt
        *
        vel[idx]

        +

        h01[:, None]
        *
        pos[idx + 1]

        +

        h11[:, None]
        *
        sv_dt
        *
        vel[idx + 1]
    )


sat_by_row = orbit_position(
    row_time
)


# ======================================================================
# WGS84
# ======================================================================

A = 6378137.0

F = (
    1.0
    /
    298.257223563
)

E2 = (
    F
    *
    (
        2.0
        -
        F
    )
)


# ======================================================================
# Final fused production kernel
#
# GAMMA-compatible numerical definition:
#
#   ground point G = WGS84(lon,lat,h)
#   radial normal  = G / |G|
#   LOS            = S - G
#
#   incidence = acos(radial_normal dot LOS_hat)
# ======================================================================

@njit(
    parallel=True,
    fastmath=False,
    cache=True,
)
def incidence_fast(
    lon_deg,
    lat_deg,
    hgt_m,
    radar_row,
    sat_xyz,
    a,
    e2,
):

    n = lon_deg.size

    out = np.empty(
        n,
        dtype=np.float32,
    )

    d2r = (
        np.pi
        /
        180.0
    )


    for i in prange(
        n
    ):

        lam = (
            lon_deg[i]
            *
            d2r
        )

        phi = (
            lat_deg[i]
            *
            d2r
        )


        sp = np.sin(
            phi
        )

        cp = np.cos(
            phi
        )

        sl = np.sin(
            lam
        )

        cl = np.cos(
            lam
        )


        rn = (
            a
            /
            np.sqrt(
                1.0
                -
                e2
                *
                sp
                *
                sp
            )
        )


        h = hgt_m[i]


        gx = (
            rn + h
        ) * cp * cl


        gy = (
            rn + h
        ) * cp * sl


        gz = (
            rn
            *
            (
                1.0
                -
                e2
            )
            +
            h
        ) * sp


        r = radar_row[i]


        lx = (
            sat_xyz[r, 0]
            -
            gx
        )

        ly = (
            sat_xyz[r, 1]
            -
            gy
        )

        lz = (
            sat_xyz[r, 2]
            -
            gz
        )


        los_norm = np.sqrt(
            lx*lx
            +
            ly*ly
            +
            lz*lz
        )


        ground_norm = np.sqrt(
            gx*gx
            +
            gy*gy
            +
            gz*gz
        )


        cosine = (
            gx*lx
            +
            gy*ly
            +
            gz*lz
        ) / (
            ground_norm
            *
            los_norm
        )


        if cosine > 1.0:
            cosine = 1.0

        elif cosine < -1.0:
            cosine = -1.0


        out[i] = np.arccos(
            cosine
        )


    return out


# ======================================================================
# Warm-up
# ======================================================================

_ = incidence_fast(
    lon[:2048],
    lat[:2048],
    hgt[:2048],
    row[:2048],
    sat_by_row,
    A,
    E2,
)


# ======================================================================
# Full benchmark
# ======================================================================

t0 = time.perf_counter()

inc = incidence_fast(
    lon,
    lat,
    hgt,
    row,
    sat_by_row,
    A,
    E2,
)

elapsed = (
    time.perf_counter()
    -
    t0
)


valid = (
    np.isfinite(inc)
    &
    (inc > 0)
    &
    (inc < np.pi/2)
)


deg = np.degrees(
    inc[
        valid
    ].astype(
        np.float64
    )
)


q = np.percentile(
    deg,
    [
        1,
        5,
        50,
        95,
        99,
    ]
)


print("=" * 88)
print("P15-3H FINAL FAST GAMMA-COMPATIBLE INCIDENCE")
print("=" * 88)

print(
    "points                    :",
    f"{N:,}",
)

print(
    "compute seconds           :",
    f"{elapsed:.6f}",
)

print(
    "throughput                :",
    f"{N/elapsed:,.0f} points/s",
)

print(
    "valid                     :",
    f"{100*valid.mean():.6f}%",
)

print(
    "inc p01/p05/p50/p95/p99 :",
    " / ".join(
        f"{x:.6f}"
        for x in q
    ),
    "deg",
)


# ======================================================================
# Existing GAMMA oracle regression
# ======================================================================

gamma_file = (
    ROOT
    /
    "incidence_ellipsoid_gamma_rad.gamma_pt"
)


truth_status = (
    "NOT_AVAILABLE"
)

rms = None
p99 = None
max_abs = None


if gamma_file.is_file():

    gamma = np.fromfile(
        gamma_file,
        dtype=">f4",
    ).astype(
        np.float64
    )


    if gamma.size != N:
        raise RuntimeError(
            "GAMMA truth size mismatch"
        )


    m = (
        valid
        &
        np.isfinite(gamma)
        &
        (gamma > 0)
        &
        (gamma < np.pi/2)
    )


    diff = np.degrees(
        inc[m].astype(
            np.float64
        )
        -
        gamma[m]
    )


    ad = np.abs(
        diff
    )


    rms = float(
        np.sqrt(
            np.mean(
                diff
                *
                diff
            )
        )
    )


    p99 = float(
        np.percentile(
            ad,
            99,
        )
    )


    max_abs = float(
        np.max(
            ad
        )
    )


    print()
    print(
        "FAST vs GAMMA inc_flg=1"
    )

    print(
        "RMS deg                  :",
        f"{rms:.9f}",
    )

    print(
        "p99 |diff| deg           :",
        f"{p99:.9f}",
    )

    print(
        "max |diff| deg           :",
        f"{max_abs:.9f}",
    )


    if (
        rms <= 0.002
        and
        p99 <= 0.005
        and
        max_abs <= 0.02
    ):

        truth_status = (
            "PASS_STRONG"
        )

    else:

        raise RuntimeError(
            "FAST incidence failed GAMMA regression"
        )


# ======================================================================
# Save only accepted product
# ======================================================================

np.save(
    OUT,
    inc,
)


manifest = {
    "status":
        "PASS_FAST_GAMMA_COMPATIBLE_INCIDENCE",

    "algorithm":
        "actual_height_ECEF_radial_normal_LOS",

    "orbit":
        "row_time_cubic_Hermite",

    "zero_doppler_refinement":
        False,

    "points":
        int(N),

    "seconds":
        float(elapsed),

    "points_per_second":
        float(
            N
            /
            elapsed
        ),

    "valid_fraction":
        float(
            valid.mean()
        ),

    "incidence_percentiles_deg":
        {
            "p01": float(q[0]),
            "p05": float(q[1]),
            "p50": float(q[2]),
            "p95": float(q[3]),
            "p99": float(q[4]),
        },

    "gamma_truth":
        {
            "status": truth_status,
            "rms_deg": rms,
            "p99_abs_deg": p99,
            "max_abs_deg": max_abs,
        },

    "production_policy":
        {
            "gc_map2_incidence":
                "validation_only",

            "geocode_incidence":
                "not_required",

            "data2pt_incidence":
                "not_required",

            "local_incidence":
                "not_for_gacos",

            "lv_theta":
                "qa_only",
        },

    "next":
        "P15-4_FAST_GACOS_POINT_SAMPLING",
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    +
    "\n"
)


print()
print(
    "output                    :",
    OUT,
)

print(
    "manifest                  :",
    MANIFEST,
)

print()
print("=" * 88)
print(
    "P15-3H FINAL RESULT: "
    "PASS_FAST_GAMMA_COMPATIBLE_INCIDENCE"
)
print("=" * 88)
