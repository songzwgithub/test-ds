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


# =============================================================================
# Parameter parsing
# =============================================================================

def read_par(path):
    d = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" in line:
            k, v = line.split(
                ":",
                1,
            )

            d[
                k.strip().lower()
            ] = v.strip()

    return d


NUM_RE = re.compile(
    r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
)


def scalar(d, key):
    m = NUM_RE.search(
        d[key]
    )

    if not m:
        raise KeyError(
            key
        )

    return float(
        m.group(0)
    )


def vec3(d, key):
    vals = NUM_RE.findall(
        d[key]
    )

    if len(vals) < 3:
        raise KeyError(
            key
        )

    return np.array(
        [
            float(vals[0]),
            float(vals[1]),
            float(vals[2]),
        ],
        dtype=np.float64,
    )


# =============================================================================
# Inputs
# =============================================================================

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

gamma = np.fromfile(
    ROOT / "incidence_ellipsoid_gamma_rad.gamma_pt",
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
    gamma.size
    ==
    N
):

    raise RuntimeError(
        (
            lon.size,
            lat.size,
            row.size,
            hgt.size,
            gamma.size,
        )
    )


# =============================================================================
# RSLC timing + orbit
# =============================================================================

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


dt_line = scalar(
    p,
    "azimuth_line_time",
)


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


sv_pos = np.empty(
    (
        nsv,
        3,
    ),
    dtype=np.float64,
)


sv_vel = np.empty(
    (
        nsv,
        3,
    ),
    dtype=np.float64,
)


for i in range(
    nsv
):

    sv_pos[i] = vec3(
        p,
        f"state_vector_position_{i+1}",
    )

    sv_vel[i] = vec3(
        p,
        f"state_vector_velocity_{i+1}",
    )


# =============================================================================
# WGS84
# =============================================================================

A_WGS84 = 6378137.0

F_WGS84 = (
    1.0
    /
    298.257223563
)

E2_WGS84 = (
    F_WGS84
    *
    (
        2.0
        -
        F_WGS84
    )
)


# =============================================================================
# Cubic Hermite orbit position / velocity / acceleration
# =============================================================================

@njit(
    inline="always"
)
def orbit_pva(
    t,
    sv_t0,
    sv_dt,
    pos,
    vel,
):

    n = pos.shape[0]

    j = int(
        np.floor(
            (
                t
                -
                sv_t0
            )
            /
            sv_dt
        )
    )


    if j < 0:

        j = 0

    elif j > n - 2:

        j = n - 2


    t0 = (
        sv_t0
        +
        j
        *
        sv_dt
    )


    u = (
        t
        -
        t0
    ) / sv_dt


    u2 = (
        u
        *
        u
    )

    u3 = (
        u2
        *
        u
    )


    # Position basis
    h00 = (
        2.0*u3
        -
        3.0*u2
        +
        1.0
    )

    h10 = (
        u3
        -
        2.0*u2
        +
        u
    )

    h01 = (
        -2.0*u3
        +
        3.0*u2
    )

    h11 = (
        u3
        -
        u2
    )


    # First derivatives
    dh00 = (
        6.0*u2
        -
        6.0*u
    )

    dh10 = (
        3.0*u2
        -
        4.0*u
        +
        1.0
    )

    dh01 = (
        -6.0*u2
        +
        6.0*u
    )

    dh11 = (
        3.0*u2
        -
        2.0*u
    )


    # Second derivatives
    d2h00 = (
        12.0*u
        -
        6.0
    )

    d2h10 = (
        6.0*u
        -
        4.0
    )

    d2h01 = (
        -12.0*u
        +
        6.0
    )

    d2h11 = (
        6.0*u
        -
        2.0
    )


    sx = (
        h00
        *
        pos[j, 0]

        +

        h10
        *
        sv_dt
        *
        vel[j, 0]

        +

        h01
        *
        pos[j+1, 0]

        +

        h11
        *
        sv_dt
        *
        vel[j+1, 0]
    )


    sy = (
        h00
        *
        pos[j, 1]

        +

        h10
        *
        sv_dt
        *
        vel[j, 1]

        +

        h01
        *
        pos[j+1, 1]

        +

        h11
        *
        sv_dt
        *
        vel[j+1, 1]
    )


    sz = (
        h00
        *
        pos[j, 2]

        +

        h10
        *
        sv_dt
        *
        vel[j, 2]

        +

        h01
        *
        pos[j+1, 2]

        +

        h11
        *
        sv_dt
        *
        vel[j+1, 2]
    )


    vx = (
        dh00
        *
        pos[j, 0]

        +

        dh10
        *
        sv_dt
        *
        vel[j, 0]

        +

        dh01
        *
        pos[j+1, 0]

        +

        dh11
        *
        sv_dt
        *
        vel[j+1, 0]
    ) / sv_dt


    vy = (
        dh00
        *
        pos[j, 1]

        +

        dh10
        *
        sv_dt
        *
        vel[j, 1]

        +

        dh01
        *
        pos[j+1, 1]

        +

        dh11
        *
        sv_dt
        *
        vel[j+1, 1]
    ) / sv_dt


    vz = (
        dh00
        *
        pos[j, 2]

        +

        dh10
        *
        sv_dt
        *
        vel[j, 2]

        +

        dh01
        *
        pos[j+1, 2]

        +

        dh11
        *
        sv_dt
        *
        vel[j+1, 2]
    ) / sv_dt


    inv_h2 = (
        1.0
        /
        (
            sv_dt
            *
            sv_dt
        )
    )


    ax = (
        d2h00
        *
        pos[j, 0]

        +

        d2h10
        *
        sv_dt
        *
        vel[j, 0]

        +

        d2h01
        *
        pos[j+1, 0]

        +

        d2h11
        *
        sv_dt
        *
        vel[j+1, 0]
    ) * inv_h2


    ay = (
        d2h00
        *
        pos[j, 1]

        +

        d2h10
        *
        sv_dt
        *
        vel[j, 1]

        +

        d2h01
        *
        pos[j+1, 1]

        +

        d2h11
        *
        sv_dt
        *
        vel[j+1, 1]
    ) * inv_h2


    az = (
        d2h00
        *
        pos[j, 2]

        +

        d2h10
        *
        sv_dt
        *
        vel[j, 2]

        +

        d2h01
        *
        pos[j+1, 2]

        +

        d2h11
        *
        sv_dt
        *
        vel[j+1, 2]
    ) * inv_h2


    return (
        sx,
        sy,
        sz,
        vx,
        vy,
        vz,
        ax,
        ay,
        az,
    )


# =============================================================================
# Fused point kernel
#
# Solve:
#
#       (S(t) - G) dot V(t) = 0
#
# with Newton iteration.
# =============================================================================

@njit(
    parallel=True,
    fastmath=False,
)
def incidence_zero_doppler(
    lon_deg,
    lat_deg,
    hgt_m,
    radar_row,
    row_time,
    sv_t0,
    sv_dt,
    sv_pos,
    sv_vel,
    a,
    e2,
    niter,
):

    n = lon_deg.size


    inc = np.empty(
        n,
        dtype=np.float32,
    )


    dt_out = np.empty(
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

        h = hgt_m[i]


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


        gx = (
            rn
            +
            h
        ) * cp * cl


        gy = (
            rn
            +
            h
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


        t_init = row_time[
            radar_row[i]
        ]


        t = t_init


        # Newton iterations
        for _ in range(
            niter
        ):

            (
                sx,
                sy,
                sz,
                vx,
                vy,
                vz,
                ax,
                ay,
                az,
            ) = orbit_pva(
                t,
                sv_t0,
                sv_dt,
                sv_pos,
                sv_vel,
            )


            lx = (
                sx
                -
                gx
            )

            ly = (
                sy
                -
                gy
            )

            lz = (
                sz
                -
                gz
            )


            f = (
                lx*vx
                +
                ly*vy
                +
                lz*vz
            )


            df = (
                vx*vx
                +
                vy*vy
                +
                vz*vz
                +
                lx*ax
                +
                ly*ay
                +
                lz*az
            )


            if np.abs(
                df
            ) < 1.0e-12:

                break


            step = (
                -f
                /
                df
            )


            # Safety only.
            # A correct initial image-row time should need
            # substantially less than this.
            if step > 0.25:

                step = 0.25

            elif step < -0.25:

                step = -0.25


            t += step


        (
            sx,
            sy,
            sz,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = orbit_pva(
            t,
            sv_t0,
            sv_dt,
            sv_pos,
            sv_vel,
        )


        lx = (
            sx
            -
            gx
        )

        ly = (
            sy
            -
            gy
        )

        lz = (
            sz
            -
            gz
        )


        ll = np.sqrt(
            lx*lx
            +
            ly*ly
            +
            lz*lz
        )


        # WGS84 ellipsoid outward normal
        nx = (
            cp
            *
            cl
        )

        ny = (
            cp
            *
            sl
        )

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


        inc[i] = np.arccos(
            c
        )


        dt_out[i] = (
            t
            -
            t_init
        )


    return (
        inc,
        dt_out,
    )


# =============================================================================
# JIT warm-up
# =============================================================================

_ = incidence_zero_doppler(
    lon[:2048],
    lat[:2048],
    hgt[:2048],
    row[:2048],
    row_time,
    sv_t0,
    sv_dt,
    sv_pos,
    sv_vel,
    A_WGS84,
    E2_WGS84,
    3,
)


# =============================================================================
# Full benchmark
# =============================================================================

t0 = time.perf_counter()


inc, dt_corr = incidence_zero_doppler(
    lon,
    lat,
    hgt,
    row,
    row_time,
    sv_t0,
    sv_dt,
    sv_pos,
    sv_vel,
    A_WGS84,
    E2_WGS84,
    3,
)


elapsed = (
    time.perf_counter()
    -
    t0
)


# =============================================================================
# Compare directly with existing GAMMA inc_flg=1 truth
# =============================================================================

valid = (
    np.isfinite(
        inc
    )
    &
    np.isfinite(
        gamma
    )
    &
    (
        inc > 0
    )
    &
    (
        inc < np.pi/2
    )
    &
    (
        gamma > 0
    )
    &
    (
        gamma < np.pi/2
    )
)


fd = np.degrees(
    inc[
        valid
    ].astype(
        np.float64
    )
)


gd = np.degrees(
    gamma[
        valid
    ]
)


diff = (
    fd
    -
    gd
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


qd = np.percentile(
    diff,
    [
        1,
        5,
        50,
        95,
        99,
    ],
)


qa = np.percentile(
    ad,
    [
        50,
        95,
        99,
        100,
    ],
)


qdt = np.percentile(
    dt_corr[
        valid
    ].astype(
        np.float64
    ),
    [
        1,
        5,
        50,
        95,
        99,
    ],
)


# GACOS mapping-factor error
mf = (
    1.0
    /
    np.cos(
        inc[
            valid
        ].astype(
            np.float64
        )
    )
)


mg = (
    1.0
    /
    np.cos(
        gamma[
            valid
        ]
    )
)


ppm = (
    (
        mf
        /
        mg
    )
    -
    1.0
) * 1.0e6


print(
    "=" * 88
)

print(
    "P15-3F ZERO-DOPPLER FAST INCIDENCE"
)

print(
    "=" * 88
)


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
    "valid comparison          :",
    f"{100*valid.mean():.6f}%",
)


print()
print(
    "zero-Doppler dt p01/p05/p50/p95/p99 s:"
)

print(
    " ",
    qdt,
)


print()
print(
    "FAST-ZD incidence p01/p50/p99:"
)

print(
    " ",
    np.percentile(
        fd,
        [
            1,
            50,
            99,
        ],
    ),
)


print(
    "GAMMA incidence p01/p50/p99:"
)

print(
    " ",
    np.percentile(
        gd,
        [
            1,
            50,
            99,
        ],
    ),
)


print()
print(
    "FAST-ZD - GAMMA diff "
    "p01/p05/p50/p95/p99 deg:"
)

print(
    " ",
    qd,
)


print(
    "RMS difference deg       :",
    f"{rms:.9f}",
)


print(
    "|diff| p50/p95/p99/max   :",
    " / ".join(
        f"{x:.9f}"
        for x in qa
    ),
    "deg",
)


print()
print(
    "mapping error p01/p50/p99 ppm:"
)

print(
    " ",
    np.percentile(
        ppm,
        [
            1,
            50,
            99,
        ],
    ),
)


print(
    "mapping max abs ppm      :",
    f"{np.max(np.abs(ppm)):.3f}",
)


# =============================================================================
# Save audit products
# =============================================================================

np.save(
    ROOT
    /
    "incidence_ellipsoid_zd_fast_rad.npy",
    inc,
)


np.save(
    ROOT
    /
    "zero_doppler_time_correction_s.npy",
    dt_corr,
)


# =============================================================================
# Gate
# =============================================================================

if valid.mean() < 0.99999:

    status = (
        "FAIL_COVERAGE"
    )


elif (
    rms <= 0.002
    and
    qa[2] <= 0.005
    and
    qa[3] <= 0.02
):

    status = (
        "PASS_ZERO_DOPPLER_STRONG"
    )


elif (
    rms <= 0.01
    and
    qa[2] <= 0.02
    and
    qa[3] <= 0.05
):

    status = (
        "PASS_ZERO_DOPPLER_PRODUCTION_EQUIVALENT"
    )


else:

    status = (
        "FAIL_ZERO_DOPPLER_EQUIVALENCE"
    )


print()
print(
    "=" * 88
)

print(
    "P15-3F FINAL RESULT:",
    status,
)

print(
    "=" * 88
)
