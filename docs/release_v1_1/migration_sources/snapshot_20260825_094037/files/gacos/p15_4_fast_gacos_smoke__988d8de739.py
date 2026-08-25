from pathlib import Path
import csv
import json
import math
import re
import time

import numpy as np
from numba import njit, prange


PSDS = Path("/home/ubuntu/Downloads/psds")

ROOT = (
    PSDS
    / "output/processing/gacos_geometry"
)

OUT = (
    PSDS
    / "output/processing/gacos_fast_smoke"
)

GACOS = Path(
    "/home/ubuntu/Downloads/GACOS"
)

RSLC_PAR = Path(
    "/home/ubuntu/Downloads/RSLC/20151212.rslc.par"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


REF_DATE = "20141006"

REF_ROW = 539
REF_COL = 337

REF_ROWS = 21
REF_COLS = 31

EXPECTED_REF_POINTS = 607

C0 = 299792458.0


# ======================================================================
# Small parsers
# ======================================================================

def par_scalar(path, key):

    rx = re.compile(
        r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
    )

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(
            ":",
            1,
        )

        if k.strip().lower() == key.lower():

            m = rx.search(
                v
            )

            if m:
                return float(
                    m.group(0)
                )

    raise KeyError(
        key
    )


def read_rsc(path):

    d = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        s = line.strip()

        if not s:
            continue

        if s.startswith("#"):
            continue

        p = s.split()

        if len(p) >= 2:
            d[
                p[0].upper()
            ] = p[1]

    return d


def req(
    d,
    key,
    cast=float,
):

    if key not in d:
        raise KeyError(
            f"missing {key}"
        )

    return cast(
        d[key]
    )


# ======================================================================
# GACOS date / geometry contract
# ======================================================================

ztd_files = sorted(
    GACOS.glob(
        "*.ztd"
    )
)

dates = [
    p.stem
    for p in ztd_files
]


if (
    len(ztd_files) != 38
    or
    len(set(dates)) != 38
    or
    dates[0] != REF_DATE
):

    raise RuntimeError(
        (
            "GACOS date contract failed: "
            f"n={len(dates)}, "
            f"first={dates[:1]}"
        )
    )


r0 = read_rsc(
    Path(
        str(
            ztd_files[0]
        )
        +
        ".rsc"
    )
)


width = req(
    r0,
    "WIDTH",
    int,
)

length = req(
    r0,
    "FILE_LENGTH",
    int,
)

x_first = req(
    r0,
    "X_FIRST",
)

y_first = req(
    r0,
    "Y_FIRST",
)

x_step = req(
    r0,
    "X_STEP",
)

y_step = req(
    r0,
    "Y_STEP",
)


geom0 = (
    width,
    length,
    x_first,
    y_first,
    x_step,
    y_step,
)


expected_bytes = (
    width
    *
    length
    *
    4
)


for ztd in ztd_files:

    rsc = Path(
        str(ztd)
        +
        ".rsc"
    )

    if not rsc.is_file():

        raise RuntimeError(
            f"missing RSC: {rsc}"
        )


    rr = read_rsc(
        rsc
    )


    geom = (
        req(
            rr,
            "WIDTH",
            int,
        ),
        req(
            rr,
            "FILE_LENGTH",
            int,
        ),
        req(
            rr,
            "X_FIRST",
        ),
        req(
            rr,
            "Y_FIRST",
        ),
        req(
            rr,
            "X_STEP",
        ),
        req(
            rr,
            "Y_STEP",
        ),
    )


    if geom != geom0:

        raise RuntimeError(
            f"RSC geometry mismatch: {ztd.name}"
        )


    if (
        ztd.stat().st_size
        !=
        expected_bytes
    ):

        raise RuntimeError(
            f"ZTD byte-size mismatch: {ztd.name}"
        )


# ======================================================================
# Point geometry
# ======================================================================

lon = np.load(
    ROOT
    /
    "longitude_deg.npy",
    mmap_mode="r",
)

lat = np.load(
    ROOT
    /
    "latitude_deg.npy",
    mmap_mode="r",
)

inc = np.load(
    ROOT
    /
    "incidence_gamma_compatible_fast_rad.npy",
    mmap_mode="r",
)


n = lon.size


if not (
    lat.size
    ==
    inc.size
    ==
    n
):

    raise RuntimeError(
        "point-array size mismatch"
    )


# ======================================================================
# Reconstruct the exact 607-point reference region
#
# plist is:
#
#   col, row
#
# in 1x1 radar coordinates.
# ======================================================================

plist = np.fromfile(
    ROOT
    /
    "strict_points.plist",
    dtype=">i4",
)


if plist.size != 2*n:

    raise RuntimeError(
        "strict_points.plist size mismatch"
    )


plist = plist.reshape(
    -1,
    2,
)


col = plist[:, 0]
row = plist[:, 1]


hr = REF_ROWS // 2
hc = REF_COLS // 2


ref_idx = np.flatnonzero(
    (
        row
        >=
        REF_ROW - hr
    )
    &
    (
        row
        <=
        REF_ROW + hr
    )
    &
    (
        col
        >=
        REF_COL - hc
    )
    &
    (
        col
        <=
        REF_COL + hc
    )
).astype(
    np.int32
)


if (
    ref_idx.size
    !=
    EXPECTED_REF_POINTS
):

    raise RuntimeError(
        (
            f"reference points "
            f"{ref_idx.size} "
            f"!= "
            f"{EXPECTED_REF_POINTS}"
        )
    )


# ======================================================================
# Incidence -> LOS mapping factor
# ======================================================================

valid_inc = (
    np.isfinite(
        inc
    )
    &
    (
        inc > 0
    )
    &
    (
        inc < np.pi/2
    )
)


if not np.all(
    valid_inc
):

    raise RuntimeError(
        "invalid incidence"
    )


sec_inc = (
    1.0
    /
    np.cos(
        np.asarray(
            inc,
            dtype=np.float64,
        )
    )
).astype(
    np.float32
)


# ======================================================================
# Build project-local GACOS interpolation geometry
#
# Important:
#
# This is NOT cross-project cache.
#
# Every new project / ROI regenerates it once.
#
# Stored per point:
#
#   base index
#   fx
#   fy
#
# Only 12 bytes / point.
# ======================================================================

@njit(
    parallel=True,
    fastmath=False,
    cache=True,
)
def build_interp(
    lon,
    lat,
    x0,
    y0,
    dx,
    dy,
    width,
    length,
):

    n = lon.size


    base = np.empty(
        n,
        np.int32,
    )

    fx = np.empty(
        n,
        np.float32,
    )

    fy = np.empty(
        n,
        np.float32,
    )

    bad = np.zeros(
        n,
        np.uint8,
    )


    for k in prange(
        n
    ):

        u = (
            lon[k]
            -
            x0
        ) / dx


        v = (
            lat[k]
            -
            y0
        ) / dy


        if (
            not np.isfinite(u)
            or
            not np.isfinite(v)
            or
            u < -1e-9
            or
            v < -1e-9
            or
            u > (width-1)+1e-9
            or
            v > (length-1)+1e-9
        ):

            bad[k] = 1


        u = min(
            max(
                u,
                0.0,
            ),
            width - 1.0,
        )


        v = min(
            max(
                v,
                0.0,
            ),
            length - 1.0,
        )


        j = int(
            math.floor(
                u
            )
        )

        i = int(
            math.floor(
                v
            )
        )


        # Last grid node:
        # use the last valid bilinear cell
        # with fractional coordinate 1.
        if j >= width - 1:
            j = width - 2

        if i >= length - 1:
            i = length - 2


        base[k] = (
            i
            *
            width
            +
            j
        )


        fx[k] = (
            u - j
        )


        fy[k] = (
            v - i
        )


    return (
        base,
        fx,
        fy,
        bad,
    )


# ======================================================================
# Fused bilinear interpolation + ZTD -> LOS
#
# No 4-neighbour temporary arrays.
# ======================================================================

@njit(
    parallel=True,
    fastmath=False,
    cache=True,
)
def sample_los(
    z,
    base,
    fx,
    fy,
    sec_inc,
    width,
    out,
):

    for k in prange(
        base.size
    ):

        b = base[k]

        x = fx[k]
        y = fy[k]


        z00 = z[b]
        z01 = z[b + 1]

        z10 = z[
            b + width
        ]

        z11 = z[
            b + width + 1
        ]


        a = (
            z00
            +
            x
            *
            (
                z01
                -
                z00
            )
        )


        c = (
            z10
            +
            x
            *
            (
                z11
                -
                z10
            )
        )


        ztd = (
            a
            +
            y
            *
            (
                c
                -
                a
            )
        )


        out[k] = (
            ztd
            *
            sec_inc[k]
        )


# ======================================================================
# JIT warmup -- excluded from benchmark
# ======================================================================

_ = build_interp(
    np.asarray(
        lon[:1024],
        np.float64,
    ),
    np.asarray(
        lat[:1024],
        np.float64,
    ),
    x_first,
    y_first,
    x_step,
    y_step,
    width,
    length,
)


dummy = np.zeros(
    width * length,
    np.float32,
)

tmp = np.empty(
    1024,
    np.float32,
)


sample_los(
    dummy,
    np.zeros(
        1024,
        np.int32,
    ),
    np.zeros(
        1024,
        np.float32,
    ),
    np.zeros(
        1024,
        np.float32,
    ),
    np.ones(
        1024,
        np.float32,
    ),
    width,
    tmp,
)


# ======================================================================
# Build interpolation geometry once
# ======================================================================

tg = time.perf_counter()


base, fx, fy, bad = build_interp(
    lon,
    lat,
    x_first,
    y_first,
    x_step,
    y_step,
    width,
    length,
)


geometry_seconds = (
    time.perf_counter()
    -
    tg
)


if np.any(
    bad
):

    raise RuntimeError(
        (
            "GACOS coverage failed: "
            f"bad={int(bad.sum())}"
        )
    )


# ======================================================================
# Project-local cache only
#
# Useful for 38 dates in THIS project.
# Never reused between unrelated projects.
# ======================================================================

cache_dir = (
    OUT
    /
    "project_local_interp"
)

cache_dir.mkdir(
    exist_ok=True
)


np.save(
    cache_dir
    /
    "base.npy",
    base,
)

np.save(
    cache_dir
    /
    "fx.npy",
    fx,
)

np.save(
    cache_dir
    /
    "fy.npy",
    fy,
)

np.save(
    cache_dir
    /
    "sec_inc.npy",
    sec_inc,
)

np.save(
    cache_dir
    /
    "ref_idx.npy",
    ref_idx,
)


# ======================================================================
# Wavelength / correction sign
# ======================================================================

freq = par_scalar(
    RSLC_PAR,
    "radar_frequency",
)


wavelength = (
    C0
    /
    freq
)


phase_factor = (
    4.0
    *
    math.pi
    /
    wavelength
)


# pyPSDS convention already frozen:
#
# phi_corr =
#
# phi_obs + (4*pi/lambda) * dL_ref
#
# so phase_factor is POSITIVE.


# ======================================================================
# ZTD reader
# ======================================================================

def read_ztd(
    path,
):

    z = np.fromfile(
        path,
        dtype="<f4",
    )


    if (
        z.size
        !=
        width * length
    ):

        raise RuntimeError(
            f"bad ZTD size: {path}"
        )


    if not np.all(
        np.isfinite(
            z
        )
    ):

        raise RuntimeError(
            f"non-finite ZTD: {path}"
        )


    med = float(
        np.median(
            z
        )
    )


    # Broad sanity gate only.
    if not (
        0.5
        <=
        med
        <=
        5.0
    ):

        raise RuntimeError(
            (
                "suspicious ZTD median "
                f"{med}: {path}"
            )
        )


    return (
        z,
        med,
    )


# ======================================================================
# Reference epoch
# ======================================================================

los0 = np.empty(
    n,
    np.float32,
)


work = np.empty(
    n,
    np.float32,
)


z0, _ = read_ztd(
    GACOS
    /
    f"{REF_DATE}.ztd"
)


sample_los(
    z0,
    base,
    fx,
    fy,
    sec_inc,
    width,
    los0,
)


# ======================================================================
# Diagnostic percentile subset
#
# Hard invariants still use ALL points.
#
# Exact percentiles on tens/hundreds of millions
# of points would dominate runtime for no scientific benefit.
# ======================================================================

diag_n = min(
    n,
    200_000,
)


diag_idx = np.linspace(
    0,
    n - 1,
    diag_n,
    dtype=np.int64,
)


# ======================================================================
# Stream 38 epochs
#
# Memory stays O(Npoints), not O(Npoints x Ndates).
# ======================================================================

rows_out = []

first_max = None
max_ref_resid = 0.0


t0 = time.perf_counter()


for e, (
    date,
    path,
) in enumerate(
    zip(
        dates,
        ztd_files,
    )
):

    te = time.perf_counter()


    z, zmed = read_ztd(
        path
    )


    sample_los(
        z,
        base,
        fx,
        fy,
        sec_inc,
        width,
        work,
    )


    # --------------------------------------------------------------
    # Temporal reference:
    #
    # L(t) - L(20141006)
    # --------------------------------------------------------------

    np.subtract(
        work,
        los0,
        out=work,
    )


    # --------------------------------------------------------------
    # Spatial reference:
    #
    # exact same 607-point region
    # --------------------------------------------------------------

    ref_before = float(
        np.median(
            work[
                ref_idx
            ]
        )
    )


    work -= np.float32(
        ref_before
    )


    ref_after = float(
        np.median(
            work[
                ref_idx
            ]
        )
    )


    max_ref_resid = max(
        max_ref_resid,
        abs(
            ref_after
        ),
    )


    if e == 0:

        first_max = float(
            np.max(
                np.abs(
                    work
                )
            )
        )


    # --------------------------------------------------------------
    # Diagnostics only
    # --------------------------------------------------------------

    q = np.percentile(
        work[
            diag_idx
        ].astype(
            np.float64,
            copy=False,
        ),
        [
            1,
            5,
            50,
            95,
            99,
        ],
    )


    rows_out.append(
        {
            "date":
                date,

            "ztd_median_m":
                zmed,

            "ref_before_m":
                ref_before,

            "ref_after_m":
                ref_after,

            "dlos_p01_m":
                float(q[0]),

            "dlos_p50_m":
                float(q[2]),

            "dlos_p99_m":
                float(q[4]),

            "phase_add_p01_rad":
                float(
                    q[0]
                    *
                    phase_factor
                ),

            "phase_add_p50_rad":
                float(
                    q[2]
                    *
                    phase_factor
                ),

            "phase_add_p99_rad":
                float(
                    q[4]
                    *
                    phase_factor
                ),

            "epoch_seconds":
                float(
                    time.perf_counter()
                    -
                    te
                ),
        }
    )


loop_seconds = (
    time.perf_counter()
    -
    t0
)


# ======================================================================
# Hard invariants
# ======================================================================

if (
    first_max is None
    or
    first_max > 1e-7
):

    raise RuntimeError(
        (
            "first epoch not zero: "
            f"{first_max}"
        )
    )


if (
    max_ref_resid
    >
    1e-7
):

    raise RuntimeError(
        (
            "reference residual too large: "
            f"{max_ref_resid}"
        )
    )


# ======================================================================
# Audit outputs
# ======================================================================

csv_path = (
    OUT
    /
    "gacos_smoke_epoch_stats.csv"
)


with csv_path.open(
    "w",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            rows_out[0].keys()
        ),
    )

    w.writeheader()

    w.writerows(
        rows_out
    )


manifest = {
    "status":
        "PASS_P15_4_FAST_GACOS_SMOKE",

    "production_phase_modified":
        False,

    "points":
        int(n),

    "epochs":
        len(dates),

    "reference_date":
        REF_DATE,

    "reference_points":
        int(
            ref_idx.size
        ),

    "gacos_grid":
        {
            "width":
                width,

            "length":
                length,

            "x_first":
                x_first,

            "y_first":
                y_first,

            "x_step":
                x_step,

            "y_step":
                y_step,

            "coverage_fraction":
                1.0,
        },

    "incidence_source":
        str(
            ROOT
            /
            "incidence_gamma_compatible_fast_rad.npy"
        ),

    "sec_inc_p01_p50_p99":
        [
            float(x)
            for x in np.percentile(
                sec_inc.astype(
                    np.float64
                ),
                [
                    1,
                    50,
                    99,
                ],
            )
        ],

    "wavelength_m":
        wavelength,

    "phase_factor_rad_per_m":
        phase_factor,

    "correction_sign":
        "PLUS",

    "formula":
        (
            "phi_corr = "
            "phi_obs + "
            "(4*pi/lambda)*dL_ref"
        ),

    "geometry_build_seconds":
        geometry_seconds,

    "epoch_loop_seconds":
        loop_seconds,

    "point_epochs_per_second":
        (
            n
            *
            len(dates)
            /
            loop_seconds
        ),

    "first_epoch_max_abs_dlos_m":
        first_max,

    "max_abs_reference_median_after_m":
        max_ref_resid,

    "interpolation_cache_scope":
        "project_local_only",

    "next":
        "P15-5_STREAM_GACOS_CORRECTION_AND_SCLA",
}


manifest_path = (
    OUT
    /
    "gacos_fast_smoke_manifest.json"
)


manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    +
    "\n"
)


# ======================================================================
# Summary
# ======================================================================

print(
    "=" * 88
)

print(
    "P15-4 FAST GACOS POINT-SAMPLING SMOKE"
)

print(
    "=" * 88
)


print(
    "points                    :",
    f"{n:,}",
)

print(
    "epochs                    :",
    len(dates),
)

print(
    "grid                      :",
    f"{length} x {width}",
)

print(
    "coverage                  :",
    "100.000000%",
)

print(
    "reference points          :",
    ref_idx.size,
)

print(
    "geometry build seconds    :",
    f"{geometry_seconds:.6f}",
)

print(
    "epoch-loop seconds        :",
    f"{loop_seconds:.6f}",
)

print(
    "throughput                :",
    (
        f"{n*len(dates)/loop_seconds:,.0f} "
        "point-epochs/s"
    ),
)

print(
    "wavelength m              :",
    f"{wavelength:.15f}",
)

print(
    "phase factor rad/m        :",
    f"+{phase_factor:.12f}",
)

print(
    "first epoch max |dLOS| m  :",
    f"{first_max:.3e}",
)

print(
    "max |ref median| m        :",
    f"{max_ref_resid:.3e}",
)

print(
    "production phase modified :",
    False,
)

print(
    "stats                     :",
    csv_path,
)

print(
    "manifest                  :",
    manifest_path,
)

print(
    "=" * 88
)

print(
    "P15-4 FINAL RESULT: "
    "PASS_FAST_GACOS_SMOKE"
)

print(
    "=" * 88
)
