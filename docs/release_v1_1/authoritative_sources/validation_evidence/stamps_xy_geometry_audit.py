from pathlib import Path
import json
import time

import numpy as np


ROOT = Path("/home/ubuntu/Downloads")
PSDS = ROOT / "psds"
PROC = PSDS / "output/processing"

LON = (
    PROC
    / "gacos_geometry"
    / "longitude_deg.npy"
)

LAT = (
    PROC
    / "gacos_geometry"
    / "latitude_deg.npy"
)

PLIST = (
    PROC
    / "gacos_geometry"
    / "strict_points.plist"
)

OUTDIR = (
    PROC
    / "stamps_stage8"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

XY_OUT = (
    OUTDIR
    / "stamps_xy_unrotated_m.npy"
)

AFFINE_OUT = (
    OUTDIR
    / "stamps_xy_affine_transform.npz"
)

MANIFEST = (
    OUTDIR
    / "p15_6a_stamps_xy_grid_audit.json"
)


# ================================================================
# StaMPS llh2local exact constants
# ================================================================

A = 6378137.0
E = 0.08209443794970


def meridian_arc(lat):
    """
    Exact translation of StaMPS llh2local.m.
    Input radians. Output metres.
    """

    e2 = E * E
    e4 = e2 * e2
    e6 = e4 * e2

    return A * (
        (
            1
            - e2 / 4
            - 3 * e4 / 64
            - 5 * e6 / 256
        )
        * lat

        -
        (
            3 * e2 / 8
            + 3 * e4 / 32
            + 45 * e6 / 1024
        )
        * np.sin(2 * lat)

        +
        (
            15 * e4 / 256
            + 45 * e6 / 1024
        )
        * np.sin(4 * lat)

        -
        (
            35 * e6 / 3072
        )
        * np.sin(6 * lat)
    )


def stamps_llh2local_m(
    lon_deg,
    lat_deg,
    origin_lon_deg,
    origin_lat_deg,
):
    """
    Exact vectorised equivalent of StaMPS llh2local.m,
    except final km conversion is omitted because Stage 8
    ultimately uses metres.
    """

    lon = np.deg2rad(
        np.asarray(
            lon_deg,
            dtype=np.float64,
        )
    )

    lat = np.deg2rad(
        np.asarray(
            lat_deg,
            dtype=np.float64,
        )
    )

    lon0 = np.deg2rad(
        float(
            origin_lon_deg
        )
    )

    lat0 = np.deg2rad(
        float(
            origin_lat_deg
        )
    )

    M = meridian_arc(
        lat
    )

    M0 = float(
        meridian_arc(
            np.asarray(
                lat0,
                dtype=np.float64,
            )
        )
    )

    N = (
        A
        /
        np.sqrt(
            1.0
            -
            E * E
            *
            np.sin(lat) ** 2
        )
    )

    dlambda = (
        lon
        -
        lon0
    )

    xy = np.empty(
        (
            lon.size,
            2,
        ),
        dtype=np.float64,
    )

    nonzero = (
        lat != 0.0
    )

    Eang = (
        dlambda[
            nonzero
        ]
        *
        np.sin(
            lat[
                nonzero
            ]
        )
    )

    xy[
        nonzero,
        0
    ] = (
        N[
            nonzero
        ]
        /
        np.tan(
            lat[
                nonzero
            ]
        )
        *
        np.sin(
            Eang
        )
    )

    xy[
        nonzero,
        1
    ] = (
        M[
            nonzero
        ]
        -
        M0
        +
        N[
            nonzero
        ]
        /
        np.tan(
            lat[
                nonzero
            ]
        )
        *
        (
            1.0
            -
            np.cos(
                Eang
            )
        )
    )

    # latitude = 0 special case from official code
    if np.any(
        ~nonzero
    ):

        xy[
            ~nonzero,
            0
        ] = (
            A
            *
            dlambda[
                ~nonzero
            ]
        )

        xy[
            ~nonzero,
            1
        ] = (
            -M0
        )

    return xy


# ================================================================
# Inputs
# ================================================================

for p in (
    LON,
    LAT,
    PLIST,
):
    if not p.is_file():
        raise FileNotFoundError(
            p
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


plist = np.fromfile(
    PLIST,
    dtype=">i4",
).reshape(
    -1,
    2,
)


if (
    lon.shape != lat.shape
    or
    lon.ndim != 1
    or
    plist.shape != (
        lon.size,
        2,
    )
):

    raise RuntimeError(
        (
            "point contract failed: "
            f"lon={lon.shape}, "
            f"lat={lat.shape}, "
            f"plist={plist.shape}"
        )
    )


n = lon.size


# IPTA plist convention already frozen:
# column 0 = range/col
# column 1 = azimuth/row

col = plist[
    :,
    0
].astype(
    np.float64
)

row = plist[
    :,
    1
].astype(
    np.float64
)


# ================================================================
# Official StaMPS origin:
#
# ll0 = (max(lonlat)+min(lonlat))/2
#
# Translation/rotation do not affect Stage-8 pairwise distances.
# ================================================================

lon0 = float(
    (
        np.max(lon)
        +
        np.min(lon)
    )
    /
    2.0
)


lat0 = float(
    (
        np.max(lat)
        +
        np.min(lat)
    )
    /
    2.0
)


t0 = time.perf_counter()


xy64 = stamps_llh2local_m(
    lon,
    lat,
    lon0,
    lat0,
)


# Official:
#
# xy=single(xy')
# xy(:,2:3)=round(xy(:,2:3)*1000)/1000
#
# Reproduce the coordinates actually seen downstream.
xy = (
    np.round(
        xy64.astype(
            np.float32
        ).astype(
            np.float64
        )
        *
        1000.0
    )
    /
    1000.0
)


xy_seconds = (
    time.perf_counter()
    -
    t0
)


np.save(
    XY_OUT,
    xy
)


# ================================================================
# Global affine fit:
#
# [x y] =
# [1 row col] @ M
#
# If residual is tiny relative to 100 m wavelength, we can
# replace KDTree all-neighbour filtering by a raster Gaussian.
# ================================================================

# Avoid fitting 881k rows unnecessarily.
sample_n = min(
    200000,
    n,
)


sample_idx = np.linspace(
    0,
    n - 1,
    sample_n,
    dtype=np.int64,
)


X = np.column_stack(
    (
        np.ones(
            sample_n,
            dtype=np.float64,
        ),

        row[
            sample_idx
        ],

        col[
            sample_idx
        ],
    )
)


Y = xy[
    sample_idx,
    :
]


Mcoef = np.linalg.lstsq(
    X,
    Y,
    rcond=None,
)[0]


# Evaluate ALL points chunked.
CHUNK = 262144


ss = 0.0
count = 0
max_res = 0.0

res_sample = np.empty(
    n,
    dtype=np.float32,
)


for start in range(
    0,
    n,
    CHUNK,
):

    stop = min(
        start
        +
        CHUNK,
        n,
    )


    XX = np.column_stack(
        (
            np.ones(
                stop - start,
                dtype=np.float64,
            ),

            row[
                start:stop
            ],

            col[
                start:stop
            ],
        )
    )


    pred = (
        XX
        @
        Mcoef
    )


    d = (
        xy[
            start:stop,
            :
        ]
        -
        pred
    )


    r = np.sqrt(
        np.sum(
            d
            *
            d,
            axis=1,
        )
    )


    res_sample[
        start:stop
    ] = r.astype(
        np.float32
    )


    ss += float(
        np.sum(
            r
            *
            r
        )
    )

    count += int(
        r.size
    )

    max_res = max(
        max_res,
        float(
            np.max(
                r
            )
        ),
    )


affine_rms = float(
    np.sqrt(
        ss
        /
        count
    )
)


affine_q = np.percentile(
    res_sample,
    [
        50,
        90,
        95,
        99,
        99.9,
    ],
)


# ================================================================
# Metric induced by one row/column pixel
#
# x = b_row*row + b_col*col
# y = ...
#
# squared physical distance:
#
# d^2 = [dr dc] Q [dr dc]^T
# ================================================================

J = np.asarray(
    [
        Mcoef[
            1,
            :
        ],
        Mcoef[
            2,
            :
        ],
    ],
    dtype=np.float64,
)
# J:
# row increment -> [dx,dy]
# col increment -> [dx,dy]


Q = (
    J
    @
    J.T
)


row_step_m = float(
    np.linalg.norm(
        J[
            0,
            :
        ]
    )
)


col_step_m = float(
    np.linalg.norm(
        J[
            1,
            :
        ]
    )
)


cos_angle = float(
    (
        J[
            0,
            :
        ]
        @
        J[
            1,
            :
        ]
    )
    /
    (
        row_step_m
        *
        col_step_m
    )
)


angle_deg = float(
    np.rad2deg(
        np.arccos(
            np.clip(
                cos_angle,
                -1.0,
                1.0,
            )
        )
    )
)


# ================================================================
# Pairwise distance parity test:
#
# Compare exact StaMPS xy distance against affine-grid distance.
# Use local random offsets representative of Stage8 <=400m.
# ================================================================

H = int(
    np.max(
        row
    )
) + 1

W = int(
    np.max(
        col
    )
) + 1


index_grid = np.full(
    (
        H,
        W,
    ),
    -1,
    dtype=np.int32,
)


ri = row.astype(
    np.int64
)

ci = col.astype(
    np.int64
)


index_grid[
    ri,
    ci
] = np.arange(
    n,
    dtype=np.int32,
)


rng = np.random.default_rng(
    20260824
)


target_pairs = 200000

d_exact = []
d_affine = []


# Conservative pixel windows around 400m.
max_dr = int(
    np.ceil(
        420.0
        /
        max(
            row_step_m,
            1e-9,
        )
    )
) + 3


max_dc = int(
    np.ceil(
        420.0
        /
        max(
            col_step_m,
            1e-9,
        )
    )
) + 3


attempts = 0


while (
    sum(
        len(x)
        for x in d_exact
    )
    <
    target_pairs
    and
    attempts < 20
):

    attempts += 1

    m = 50000


    a = rng.integers(
        0,
        n,
        size=m,
    )


    dr = rng.integers(
        -max_dr,
        max_dr + 1,
        size=m,
    )


    dc = rng.integers(
        -max_dc,
        max_dc + 1,
        size=m,
    )


    r2 = (
        ri[
            a
        ]
        +
        dr
    )

    c2 = (
        ci[
            a
        ]
        +
        dc
    )


    inside = (
        (r2 >= 0)
        &
        (r2 < H)
        &
        (c2 >= 0)
        &
        (c2 < W)
    )


    a = a[
        inside
    ]

    dr = dr[
        inside
    ]

    dc = dc[
        inside
    ]

    r2 = r2[
        inside
    ]

    c2 = c2[
        inside
    ]


    b = index_grid[
        r2,
        c2
    ]


    valid = (
        b >= 0
    )


    a = a[
        valid
    ]

    b = b[
        valid
    ]

    dr = dr[
        valid
    ]

    dc = dc[
        valid
    ]


    de = np.linalg.norm(
        xy[
            a
        ]
        -
        xy[
            b
        ],
        axis=1,
    )


    dd = np.column_stack(
        (
            dr.astype(
                np.float64
            ),

            dc.astype(
                np.float64
            ),
        )
    )


    da = np.sqrt(
        np.einsum(
            "ni,ij,nj->n",
            dd,
            Q,
            dd,
        )
    )


    # Stage8 relevant range.
    keep = (
        de < 400.0
    )


    if np.any(
        keep
    ):

        d_exact.append(
            de[
                keep
            ]
        )

        d_affine.append(
            da[
                keep
            ]
        )


if not d_exact:

    raise RuntimeError(
        "no local pairwise samples"
    )


d_exact = np.concatenate(
    d_exact
)[:target_pairs]


d_affine = np.concatenate(
    d_affine
)[:target_pairs]


distance_error = (
    d_affine
    -
    d_exact
)


dist_max = float(
    np.max(
        np.abs(
            distance_error
        )
    )
)


dist_rms = float(
    np.sqrt(
        np.mean(
            distance_error
            *
            distance_error
        )
    )
)


dist_q = np.percentile(
    np.abs(
        distance_error
    ),
    [
        50,
        95,
        99,
        99.9,
    ],
)


# Gaussian-weight relative error at sigma=100m.
sigma = 100.0


w_exact = np.exp(
    -
    d_exact
    *
    d_exact
    /
    (
        2.0
        *
        sigma
        *
        sigma
    )
)


w_affine = np.exp(
    -
    d_affine
    *
    d_affine
    /
    (
        2.0
        *
        sigma
        *
        sigma
    )
)


# Ignore vanishing weights for relative metric.
use = (
    w_exact
    >
    1e-6
)


w_abs_max = float(
    np.max(
        np.abs(
            w_affine
            -
            w_exact
        )
    )
)


w_rel_q = np.percentile(
    np.abs(
        w_affine[
            use
        ]
        -
        w_exact[
            use
        ]
    )
    /
    w_exact[
        use
    ],
    [
        50,
        95,
        99,
        99.9,
    ],
)


# ================================================================
# Decision
#
# Do NOT automatically accept an approximation here.
# This stage only tells us which implementation P15-6B should use.
# ================================================================

if (
    dist_q[2]
    <= 0.05
    and
    w_rel_q[2]
    <= 1e-3
):

    recommendation = (
        "RASTER_GAUSSIAN_CANDIDATE_VALIDATE_AGAINST_KDTREE"
    )

else:

    recommendation = (
        "USE_EXACT_COORDINATE_NEIGHBOUR_ENGINE"
    )


np.savez(
    AFFINE_OUT,

    origin_lonlat_deg=
        np.asarray(
            [
                lon0,
                lat0,
            ],
            dtype=np.float64,
        ),

    affine_coeff=
        Mcoef,

    J_row_col_to_xy_m=
        J,

    metric_Q=
        Q,
)


manifest = {

    "status":
        "PASS_STAMPS_XY_AUDIT",

    "source_semantics":
        (
            "StaMPS ps_load_initial_gamma "
            "llh2local WGS84"
        ),

    "points":
        int(
            n
        ),

    "origin_lonlat_deg":
        [
            lon0,
            lat0,
        ],

    "xy_seconds":
        xy_seconds,

    "xy_extent_m":
        {
            "xmin":
                float(
                    xy[:,0].min()
                ),

            "xmax":
                float(
                    xy[:,0].max()
                ),

            "ymin":
                float(
                    xy[:,1].min()
                ),

            "ymax":
                float(
                    xy[:,1].max()
                ),
        },

    "affine":
        {
            "coeff":
                Mcoef.tolist(),

            "rms_position_residual_m":
                affine_rms,

            "max_position_residual_m":
                max_res,

            "position_residual_p50_p90_p95_p99_p999_m":
                [
                    float(x)
                    for x in affine_q
                ],

            "row_step_m":
                row_step_m,

            "col_step_m":
                col_step_m,

            "row_col_angle_deg":
                angle_deg,

            "metric_Q":
                Q.tolist(),
        },

    "local_distance_parity":
        {
            "samples":
                int(
                    d_exact.size
                ),

            "max_abs_error_m":
                dist_max,

            "rms_error_m":
                dist_rms,

            "abs_error_p50_p95_p99_p999_m":
                [
                    float(x)
                    for x in dist_q
                ],

            "gaussian_weight_max_abs_error":
                w_abs_max,

            "gaussian_weight_rel_error_p50_p95_p99_p999":
                [
                    float(x)
                    for x in w_rel_q
                ],
        },

    "stage8":
        {
            "scn_wavelength_m":
                100.0,

            "radius_m":
                400.0,

            "recommendation":
                recommendation,
        },

    "outputs":
        {
            "xy":
                str(
                    XY_OUT
                ),

            "affine_transform":
                str(
                    AFFINE_OUT
                ),
        },
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    +
    "\n"
)


print("=" * 92)
print("P15-6A STAMPS XY / STAGE-8 GRID ACCELERATION AUDIT")
print("=" * 92)

print(
    "points                         :",
    f"{n:,}",
)

print(
    "StaMPS ll0 lon/lat             :",
    f"{lon0:.10f}, {lat0:.10f}",
)

print(
    "xy generation seconds          :",
    f"{xy_seconds:.6f}",
)

print()

print(
    "row metric step                :",
    f"{row_step_m:.6f} m",
)

print(
    "col metric step                :",
    f"{col_step_m:.6f} m",
)

print(
    "row/col physical angle         :",
    f"{angle_deg:.6f} deg",
)

print()

print(
    "affine position RMS            :",
    f"{affine_rms:.6f} m",
)

print(
    "affine position max            :",
    f"{max_res:.6f} m",
)

print(
    "affine position p50/90/95/99/999:",
    affine_q,
)

print()

print(
    "local distance samples         :",
    f"{d_exact.size:,}",
)

print(
    "distance error RMS             :",
    f"{dist_rms:.9f} m",
)

print(
    "distance error max             :",
    f"{dist_max:.9f} m",
)

print(
    "distance |err| p50/95/99/999  :",
    dist_q,
)

print(
    "Gaussian weight max abs error  :",
    f"{w_abs_max:.12e}",
)

print(
    "Gaussian rel err p50/95/99/999 :",
    w_rel_q,
)

print()

print(
    "Stage8 recommendation          :",
    recommendation,
)

print(
    "xy output                      :",
    XY_OUT,
)

print(
    "manifest                       :",
    MANIFEST,
)

print("=" * 92)
print(
    "P15-6A FINAL RESULT: "
    "PASS_STAMPS_XY_AUDIT"
)
print("=" * 92)
