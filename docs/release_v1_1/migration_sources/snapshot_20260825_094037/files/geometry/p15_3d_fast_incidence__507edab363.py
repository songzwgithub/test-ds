from pathlib import Path
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

lon = np.load(ROOT / "longitude_deg.npy", mmap_mode="r")
lat = np.load(ROOT / "latitude_deg.npy", mmap_mode="r")
row = np.load(ROOT / "radar_row.npy", mmap_mode="r")

hgt = np.fromfile(
    ROOT / "height_m.gamma_pt",
    dtype=">f4",
).astype(np.float64)

if not (
    lon.size
    == lat.size
    == row.size
    == hgt.size
):
    raise RuntimeError(
        (
            lon.size,
            lat.size,
            row.size,
            hgt.size,
        )
    )

N = lon.size


# ------------------------------------------------------------
# GAMMA PAR parser
# ------------------------------------------------------------

def read_par(path):
    d = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(":", 1)

        d[
            k.strip().lower()
        ] = v.strip()

    return d


def scalar(d, key):
    s = d[key]

    m = re.search(
        r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?",
        s,
    )

    return float(
        m.group(0)
    )


def vec3(d, key):
    vals = re.findall(
        r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?",
        d[key],
    )

    return np.array(
        [
            float(vals[0]),
            float(vals[1]),
            float(vals[2]),
        ],
        dtype=np.float64,
    )


p = read_par(PAR)


# ------------------------------------------------------------
# Image azimuth timing
# ------------------------------------------------------------

nlines = int(
    round(
        scalar(
            p,
            "azimuth_lines",
        )
    )
)

dt_line = scalar(
    p,
    "azimuth_line_time",
)


# Prefer center_time because it avoids ambiguity about
# whether start_time denotes first-line center or boundary.
if "center_time" in p:

    tc = scalar(
        p,
        "center_time",
    )

    row_time = (
        tc
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
                nlines
                -
                1
            )
        )
        *
        dt_line
    )

elif "start_time" in p:

    t0 = scalar(
        p,
        "start_time",
    )

    row_time = (
        t0
        +
        np.arange(
            nlines,
            dtype=np.float64,
        )
        *
        dt_line
    )

else:
    raise RuntimeError(
        "No center_time/start_time in RSLC.par"
    )


# ------------------------------------------------------------
# Orbit state vectors
# ------------------------------------------------------------

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

sv_time = (
    sv_t0
    +
    np.arange(
        nsv,
        dtype=np.float64,
    )
    *
    sv_dt
)

sv_pos = np.empty(
    (nsv, 3),
    dtype=np.float64,
)

sv_vel = np.empty(
    (nsv, 3),
    dtype=np.float64,
)

have_velocity = True


for i in range(nsv):

    sv_pos[i] = vec3(
        p,
        f"state_vector_position_{i+1}",
    )

    key = (
        f"state_vector_velocity_{i+1}"
    )

    if key in p:

        sv_vel[i] = vec3(
            p,
            key,
        )

    else:
        have_velocity = False


if not have_velocity:

    # Fallback only.
    sv_vel[:] = np.gradient(
        sv_pos,
        sv_time,
        axis=0,
    )


# ------------------------------------------------------------
# Cubic Hermite orbit interpolation
#
# Only nlines satellite positions are required.
# For this scene: only 600 positions.
# ------------------------------------------------------------

def hermite_position(
    tq,
    ts,
    pos,
    vel,
):

    idx = (
        np.searchsorted(
            ts,
            tq,
            side="right",
        )
        -
        1
    )

    idx = np.clip(
        idx,
        0,
        len(ts) - 2,
    )

    t0 = ts[idx]
    t1 = ts[idx + 1]

    hh = t1 - t0

    u = (
        tq - t0
    ) / hh

    h00 = (
        2*u**3
        -
        3*u**2
        +
        1
    )

    h10 = (
        u**3
        -
        2*u**2
        +
        u
    )

    h01 = (
        -2*u**3
        +
        3*u**2
    )

    h11 = (
        u**3
        -
        u**2
    )

    out = np.empty(
        (tq.size, 3),
        dtype=np.float64,
    )

    for k in range(3):

        out[:, k] = (
            h00
            *
            pos[idx, k]

            +

            h10
            *
            hh
            *
            vel[idx, k]

            +

            h01
            *
            pos[idx + 1, k]

            +

            h11
            *
            hh
            *
            vel[idx + 1, k]
        )

    return out


sat_by_row = hermite_position(
    row_time,
    sv_time,
    sv_pos,
    sv_vel,
)


# ------------------------------------------------------------
# WGS84
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Fused Numba point kernel
# ------------------------------------------------------------

@njit(
    parallel=True,
    fastmath=False,
    cache=True,
)
def incidence_kernel(
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

    for i in prange(n):

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

        h = hgt_m[i]

        sp = np.sin(phi)
        cp = np.cos(phi)

        sl = np.sin(lam)
        cl = np.cos(lam)

        # WGS84 prime vertical radius
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

        # Ground ECEF
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

        sx = sat_xyz[r, 0]
        sy = sat_xyz[r, 1]
        sz = sat_xyz[r, 2]


        # Ground -> satellite LOS
        lx = sx - gx
        ly = sy - gy
        lz = sz - gz

        ll = np.sqrt(
            lx*lx
            +
            ly*ly
            +
            lz*lz
        )


        # Ellipsoid outward normal.
        #
        # Since phi is geodetic latitude,
        # this is directly:
        nx = cp * cl
        ny = cp * sl
        nz = sp


        c = (
            nx*lx
            +
            ny*ly
            +
            nz*lz
        ) / ll


        if c > 1.0:
            c = 1.0

        elif c < -1.0:
            c = -1.0


        out[i] = np.arccos(
            c
        )


    return out


# ------------------------------------------------------------
# JIT warmup -- excluded from benchmark
# ------------------------------------------------------------

_ = incidence_kernel(
    np.asarray(
        lon[:1024],
        dtype=np.float64,
    ),
    np.asarray(
        lat[:1024],
        dtype=np.float64,
    ),
    hgt[:1024],
    np.asarray(
        row[:1024],
        dtype=np.int32,
    ),
    sat_by_row,
    A,
    E2,
)


# ------------------------------------------------------------
# Production benchmark
# ------------------------------------------------------------

t0 = time.perf_counter()

inc = incidence_kernel(
    np.asarray(
        lon,
        dtype=np.float64,
    ),
    np.asarray(
        lat,
        dtype=np.float64,
    ),
    hgt,
    np.asarray(
        row,
        dtype=np.int32,
    ),
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
    (
        inc
        <
        np.pi/2
    )
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
    ],
)


print("=" * 88)
print("P15-3D FAST ELLIPSOID INCIDENCE")
print("=" * 88)

print(
    "points                    :",
    f"{N:,}",
)

print(
    "azimuth rows              :",
    nlines,
)

print(
    "state vectors             :",
    nsv,
)

print(
    "state-vector velocities   :",
    have_velocity,
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


# ------------------------------------------------------------
# Compare with existing GAMMA look_vector result
# ------------------------------------------------------------

lv_file = (
    ROOT
    /
    "lv_theta_rad.gamma_pt"
)

if lv_file.is_file():

    lv = np.fromfile(
        lv_file,
        dtype=">f4",
    ).astype(
        np.float64
    )

    if lv.size == inc.size:

        lv_inc = (
            np.pi/2
            -
            lv
        )

        m = (
            valid
            &
            np.isfinite(
                lv_inc
            )
        )

        diff = np.degrees(
            inc[m].astype(
                np.float64
            )
            -
            lv_inc[m]
        )

        print()
        print(
            "FAST minus (90-lv_theta)"
        )

        print(
            "diff p01/p50/p99 deg     :",
            " / ".join(
                f"{x:.8f}"
                for x in np.percentile(
                    diff,
                    [1, 50, 99],
                )
            ),
        )

        print(
            "diff RMS deg              :",
            f"{np.sqrt(np.mean(diff**2)):.8f}",
        )

        print(
            "diff max abs deg          :",
            f"{np.max(np.abs(diff)):.8f}",
        )


np.save(
    ROOT
    /
    "incidence_ellipsoid_fast_rad.npy",
    inc,
)


print()
print(
    "output                    :",
    ROOT
    /
    "incidence_ellipsoid_fast_rad.npy",
)

print("=" * 88)
