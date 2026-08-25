from pathlib import Path
import re
import numpy as np

ROOT = Path(
    "/home/ubuntu/Downloads/psds/output/processing/gacos_geometry"
)

PAR = Path(
    "/home/ubuntu/Downloads/RSLC/20151212.rslc.par"
)


def read_par(path):
    d = {}

    for line in path.read_text(errors="ignore").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip().lower()] = v.strip()

    return d


num_re = re.compile(
    r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
)


def scalar(d, key):
    return float(
        num_re.findall(d[key])[0]
    )


def vec3(d, key):
    return np.array(
        [
            float(x)
            for x in num_re.findall(d[key])[:3]
        ],
        dtype=np.float64,
    )


# ----------------------------------------------------------------------
# Existing point geometry + GAMMA truth
# ----------------------------------------------------------------------

p = read_par(PAR)

lon = np.load(
    ROOT / "longitude_deg.npy"
).astype(np.float64)

lat = np.load(
    ROOT / "latitude_deg.npy"
).astype(np.float64)

row = np.load(
    ROOT / "radar_row.npy"
).astype(np.int32)

hgt = np.fromfile(
    ROOT / "height_m.gamma_pt",
    dtype=">f4",
).astype(np.float64)

gamma = np.fromfile(
    ROOT / "incidence_ellipsoid_gamma_rad.gamma_pt",
    dtype=">f4",
).astype(np.float64)


assert (
    lon.size
    == lat.size
    == row.size
    == hgt.size
    == gamma.size
)


# ----------------------------------------------------------------------
# Radar line timing
# ----------------------------------------------------------------------

nlines = int(
    round(
        scalar(p, "azimuth_lines")
    )
)

dt = scalar(
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
                nlines - 1
            )
        )
        *
        dt
    )

else:

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
        dt
    )


# ----------------------------------------------------------------------
# Orbit
# ----------------------------------------------------------------------

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


# Cubic Hermite interpolation
def satellite_position(t):

    j = np.floor(
        (t - sv_t0)
        /
        sv_dt
    ).astype(np.int64)

    j = np.clip(
        j,
        0,
        nsv - 2,
    )

    u = (
        t
        -
        (
            sv_t0
            +
            j
            *
            sv_dt
        )
    ) / sv_dt

    u2 = u * u
    u3 = u2 * u

    h00 = 2*u3 - 3*u2 + 1
    h10 = u3 - 2*u2 + u
    h01 = -2*u3 + 3*u2
    h11 = u3 - u2

    return (
        h00[:, None]
        *
        pos[j]

        +

        h10[:, None]
        *
        sv_dt
        *
        vel[j]

        +

        h01[:, None]
        *
        pos[j+1]

        +

        h11[:, None]
        *
        sv_dt
        *
        vel[j+1]
    )


sat = satellite_position(
    row_time[row]
)


# ----------------------------------------------------------------------
# WGS84 ground geometry
# ----------------------------------------------------------------------

A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)

lam = np.deg2rad(lon)
phi = np.deg2rad(lat)

sp = np.sin(phi)
cp = np.cos(phi)

sl = np.sin(lam)
cl = np.cos(lam)

rn = (
    A
    /
    np.sqrt(
        1.0
        -
        E2
        *
        sp
        *
        sp
    )
)


ellipsoid_normal = np.column_stack(
    (
        cp * cl,
        cp * sl,
        sp,
    )
)


def incidence(height, normal_mode):

    ground = np.column_stack(
        (
            (rn + height)
            *
            cp
            *
            cl,

            (rn + height)
            *
            cp
            *
            sl,

            (
                rn
                *
                (1.0 - E2)
                +
                height
            )
            *
            sp,
        )
    )


    los = (
        sat
        -
        ground
    )

    los /= np.linalg.norm(
        los,
        axis=1,
    )[:, None]


    if normal_mode == "ellipsoid":

        normal = ellipsoid_normal


    elif normal_mode == "radial":

        normal = (
            ground
            /
            np.linalg.norm(
                ground,
                axis=1,
            )[:, None]
        )


    else:

        raise ValueError(
            normal_mode
        )


    cosine = np.sum(
        normal
        *
        los,
        axis=1,
    )


    return np.arccos(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )


zero = np.zeros_like(
    hgt
)


variants = {
    "actual_h_ellipsoid":
        incidence(
            hgt,
            "ellipsoid",
        ),

    "zero_h_ellipsoid":
        incidence(
            zero,
            "ellipsoid",
        ),

    "actual_h_radial":
        incidence(
            hgt,
            "radial",
        ),

    "zero_h_radial":
        incidence(
            zero,
            "radial",
        ),
}


valid_gamma = (
    np.isfinite(gamma)
    &
    (gamma > 0)
    &
    (gamma < np.pi/2)
)


print("=" * 88)
print("P15-3G INCIDENCE DEFINITION DECOMPOSITION")
print("=" * 88)

hv = hgt[
    np.isfinite(hgt)
]

print(
    "height m p01/p05/p50/p95/p99:"
)

print(
    " ",
    np.percentile(
        hv,
        [1, 5, 50, 95, 99],
    )
)

print()


best = None


for name, candidate in variants.items():

    valid = (
        valid_gamma
        &
        np.isfinite(candidate)
        &
        (candidate > 0)
        &
        (candidate < np.pi/2)
    )


    diff = np.degrees(
        candidate[valid]
        -
        gamma[valid]
    )

    abs_diff = np.abs(
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
            abs_diff,
            99,
        )
    )

    maximum = float(
        np.max(
            abs_diff
        )
    )


    score = (
        rms,
        p99,
        maximum,
    )


    if (
        best is None
        or
        rms < best[0]
    ):

        best = (
            rms,
            name,
            score,
        )


    print(name)

    print(
        "  incidence p01/p50/p99:",
        np.percentile(
            np.degrees(
                candidate[valid]
            ),
            [1, 50, 99],
        ),
    )

    print(
        "  FAST-GAMMA p01/p50/p99:",
        np.percentile(
            diff,
            [1, 50, 99],
        ),
    )

    print(
        "  RMS / p99abs / maxabs:",
        score,
    )

    print()


# ----------------------------------------------------------------------
# Explicit height contribution
# ----------------------------------------------------------------------

height_effect = np.degrees(
    variants["actual_h_ellipsoid"]
    -
    variants["zero_h_ellipsoid"]
)


print(
    "actual_h - zero_h incidence "
    "p01/p50/p99 deg:"
)

print(
    " ",
    np.percentile(
        height_effect,
        [1, 50, 99],
    )
)


print()
print(
    "BEST VARIANT:",
    best[1],
)

print(
    "BEST RMS / p99abs / maxabs:",
    best[2],
)

print("=" * 88)
