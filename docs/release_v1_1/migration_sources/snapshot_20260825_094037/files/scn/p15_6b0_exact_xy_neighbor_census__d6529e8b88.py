from pathlib import Path
import json
import re
import time

import numpy as np
from scipy.spatial import cKDTree


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

RSLC_PAR = (
    ROOT
    / "RSLC"
    / "20151212.rslc.par"
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
    / "stamps_xy_exact_float32_m.npy"
)

SORT_OUT = (
    OUTDIR
    / "stamps_sort_index.npy"
)

COUNT_OUT = (
    OUTDIR
    / "stage8_neighbor_count_r400m.npy"
)

MANIFEST = (
    OUTDIR
    / "p15_6b0_exact_neighbor_census.json"
)


RADIUS = 400.0
SIGMA = 100.0

QUERY_CHUNK = 65536


# ======================================================================
# StaMPS llh2local
# ======================================================================

A = 6378137.0
E = 0.08209443794970


NUM_RE = re.compile(
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)


def read_par(path):

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


def par_scalar(
    pars,
    names,
):

    for name in names:

        x = pars.get(
            name.lower()
        )

        if x is None:
            continue

        m = NUM_RE.search(x)

        if m:
            return float(
                m.group(0)
            )

    raise KeyError(
        " / ".join(names)
    )


def meridian_arc(lat):

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
        * np.sin(
            2 * lat
        )

        +
        (
            15 * e4 / 256
            + 45 * e6 / 1024
        )
        * np.sin(
            4 * lat
        )

        -
        (
            35 * e6 / 3072
        )
        * np.sin(
            6 * lat
        )
    )


def llh2local_exact_m(
    lon_deg,
    lat_deg,
    lon0_deg,
    lat0_deg,
):

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
        float(lon0_deg)
    )

    lat0 = np.deg2rad(
        float(lat0_deg)
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


    nz = (
        lat != 0.0
    )


    Ee = (
        dlambda[nz]
        *
        np.sin(
            lat[nz]
        )
    )


    xy[
        nz,
        0
    ] = (
        N[nz]
        /
        np.tan(
            lat[nz]
        )
        *
        np.sin(
            Ee
        )
    )


    xy[
        nz,
        1
    ] = (
        M[nz]
        -
        M0
        +
        N[nz]
        /
        np.tan(
            lat[nz]
        )
        *
        (
            1.0
            -
            np.cos(
                Ee
            )
        )
    )


    if np.any(
        ~nz
    ):

        xy[
            ~nz,
            0
        ] = (
            A
            *
            dlambda[
                ~nz
            ]
        )

        xy[
            ~nz,
            1
        ] = (
            -M0
        )


    return xy


# ======================================================================
# Inputs
# ======================================================================

for p in (
    LON,
    LAT,
    PLIST,
    RSLC_PAR,
):

    if not p.is_file():
        raise FileNotFoundError(p)


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


n = lon.size


if (
    lat.shape != lon.shape
    or
    plist.shape != (
        n,
        2,
    )
):

    raise RuntimeError(
        "point geometry contract failed"
    )


pars = read_par(
    RSLC_PAR
)


heading = par_scalar(
    pars,
    (
        "heading",
    ),
)


# ======================================================================
# Official StaMPS origin
# ======================================================================

lon0 = float(
    (
        lon.max()
        +
        lon.min()
    )
    /
    2.0
)


lat0 = float(
    (
        lat.max()
        +
        lat.min()
    )
    /
    2.0
)


t_xy = time.perf_counter()


xy_raw = llh2local_exact_m(
    lon,
    lat,
    lon0,
    lat0,
)


# ======================================================================
# Official StaMPS optional rotation:
#
# theta=(180-heading)*pi/180
# if theta>pi
#     theta=theta-2*pi
# end
#
# rotm=[
#   cos(theta), sin(theta)
#  -sin(theta), cos(theta)
# ]
#
# Keep rotation only if BOTH x and y extents shrink.
# ======================================================================

theta = np.deg2rad(
    180.0
    -
    heading
)


if theta > np.pi:

    theta -= (
        2.0
        *
        np.pi
    )


rotm = np.asarray(
    [
        [
            np.cos(theta),
            np.sin(theta),
        ],
        [
            -np.sin(theta),
            np.cos(theta),
        ],
    ],
    dtype=np.float64,
)


xynew = (
    rotm
    @
    xy_raw.T
).T


raw_span = np.ptp(
    xy_raw,
    axis=0,
)


rot_span = np.ptp(
    xynew,
    axis=0,
)


rotation_accepted = bool(
    (
        rot_span[0]
        <
        raw_span[0]
    )
    and
    (
        rot_span[1]
        <
        raw_span[1]
    )
)


xy_selected = (
    xynew
    if rotation_accepted
    else xy_raw
)


# ======================================================================
# Official:
#
# xy=single(xy')
# xy(:,2:3)=round(xy(:,2:3)*1000)/1000
#
# Keep the actual single-precision coordinates.
# ======================================================================

xy32 = xy_selected.astype(
    np.float32
)


xy32 = (
    np.round(
        xy32
        *
        np.float32(
            1000.0
        )
    )
    /
    np.float32(
        1000.0
    )
).astype(
    np.float32
)


xy_seconds = (
    time.perf_counter()
    -
    t_xy
)


if not np.all(
    np.isfinite(
        xy32
    )
):

    raise RuntimeError(
        "non-finite StaMPS XY"
    )


np.save(
    XY_OUT,
    xy32
)


# ======================================================================
# Official StaMPS ordering:
#
# sortrows(xy,[2,1])
#
# Primary key y, secondary key x.
#
# We keep production arrays in their current order; sort_ix is only
# needed to identify the official StaMPS first PS / parity gauge.
# ======================================================================

sort_ix = np.lexsort(
    (
        xy32[:, 0],
        xy32[:, 1],
    )
).astype(
    np.int32
)


np.save(
    SORT_OUT,
    sort_ix
)


first_ps_current_index = int(
    sort_ix[0]
)


# ======================================================================
# Exact-coordinate cKDTree
# ======================================================================

coords = xy32.astype(
    np.float64
)


t_tree = time.perf_counter()


tree = cKDTree(
    coords,
    compact_nodes=True,
    balanced_tree=True,
)


tree_seconds = (
    time.perf_counter()
    -
    t_tree
)


# ======================================================================
# Exact 400 m neighbour census
#
# return_length=True is critical:
# no giant neighbour lists are materialised.
# ======================================================================

counts = np.lib.format.open_memmap(
    COUNT_OUT,
    mode="w+",
    dtype=np.int32,
    shape=(
        n,
    ),
)


t_query = time.perf_counter()


for start in range(
    0,
    n,
    QUERY_CHUNK,
):

    stop = min(
        start
        +
        QUERY_CHUNK,
        n,
    )


    c = tree.query_ball_point(
        coords[
            start:stop
        ],
        r=RADIUS,
        workers=-1,
        return_length=True,
    )


    c = np.asarray(
        c,
        dtype=np.int64,
    )


    if (
        c.size
        !=
        stop - start
    ):

        raise RuntimeError(
            "KDTree count size mismatch"
        )


    if np.any(
        c <= 0
    ):

        raise RuntimeError(
            "point without self neighbour"
        )


    counts[
        start:stop
    ] = c.astype(
        np.int32
    )


    print(
        "[NEIGHBOUR COUNT] "
        f"{stop:,}/{n:,} "
        f"({100*stop/n:.1f}%)",
        flush=True,
    )


counts.flush()


query_seconds = (
    time.perf_counter()
    -
    t_query
)


count64 = np.asarray(
    counts,
    dtype=np.int64,
)


count_q = np.percentile(
    count64,
    [
        1,
        5,
        50,
        90,
        95,
        99,
        99.9,
    ],
)


mean_count = float(
    np.mean(
        count64
    )
)


max_count = int(
    count64.max()
)


directed_interactions = int(
    np.sum(
        count64,
        dtype=np.int64,
    )
)


# Directed count includes:
# i->j and j->i, plus self once.
undirected_nonself = (
    directed_interactions
    -
    n
) // 2


undirected_with_self = (
    undirected_nonself
    +
    n
)


# ======================================================================
# Exact cell-list candidate census
#
# Cell width = radius.
#
# Any true neighbour within 400 m must lie in one of the 3x3
# surrounding cells. This gives an upper bound on exact pair
# evaluations for a Numba cell-list implementation.
# ======================================================================

xmin = float(
    coords[:, 0].min()
)

ymin = float(
    coords[:, 1].min()
)


bx = np.floor(
    (
        coords[:, 0]
        -
        xmin
    )
    /
    RADIUS
).astype(
    np.int32
)


by = np.floor(
    (
        coords[:, 1]
        -
        ymin
    )
    /
    RADIUS
).astype(
    np.int32
)


nx = int(
    bx.max()
) + 1

ny = int(
    by.max()
) + 1


occupancy = np.zeros(
    (
        ny,
        nx,
    ),
    dtype=np.int64,
)


np.add.at(
    occupancy,
    (
        by,
        bx,
    ),
    1,
)


pad = np.pad(
    occupancy,
    1,
    mode="constant",
)


neighbour_cell_population = np.zeros_like(
    occupancy,
    dtype=np.int64,
)


for dy in (
    0,
    1,
    2,
):

    for dx in (
        0,
        1,
        2,
    ):

        neighbour_cell_population += (
            pad[
                dy:
                dy + ny,
                dx:
                dx + nx,
            ]
        )


cell_candidate_directed = int(
    np.sum(
        occupancy
        *
        neighbour_cell_population,
        dtype=np.int64,
    )
)


candidate_to_true_ratio = float(
    cell_candidate_directed
    /
    directed_interactions
)


occ_nonzero = occupancy[
    occupancy > 0
]


occ_q = np.percentile(
    occ_nonzero,
    [
        50,
        90,
        95,
        99,
    ],
)


# ======================================================================
# Memory/work estimates
# ======================================================================

# If the whole sparse neighbour matrix were persisted:
#
# int32 column + float32 weight ~= 8 bytes / interaction
# int32 column + float64 weight ~= 12 bytes / interaction

csr_float32_gib = float(
    directed_interactions
    *
    8
    /
    1024**3
)


csr_float64_gib = float(
    directed_interactions
    *
    12
    /
    1024**3
)


chunk_4096_mean_nnz = float(
    mean_count
    *
    4096.0
)


chunk_8192_mean_nnz = float(
    mean_count
    *
    8192.0
)


# Each true neighbour contributes to 38 epochs if naively multiplied.
naive_point_epoch_interactions = (
    directed_interactions
    *
    38
)


# ======================================================================
# Routing recommendation
# ======================================================================

if (
    directed_interactions
    <=
    500_000_000
):

    recommendation = (
        "BENCHMARK_CKDTREE_STREAMING_EXACT"
    )

elif (
    candidate_to_true_ratio
    <=
    1.8
):

    recommendation = (
        "BENCHMARK_NUMBA_CELL_LIST_EXACT_FIRST"
    )

else:

    recommendation = (
        "BENCHMARK_CELL_LIST_VS_CKDTREE_EXACT"
    )


# ======================================================================
# Manifest
# ======================================================================

manifest = {

    "status":
        "PASS_EXACT_STAMPS_XY_NEIGHBOUR_CENSUS",

    "points":
        int(
            n
        ),

    "coordinate_contract":
        {
            "origin_lonlat_deg":
                [
                    lon0,
                    lat0,
                ],

            "heading_deg":
                heading,

            "theta_deg":
                float(
                    np.rad2deg(
                        theta
                    )
                ),

            "rotation_accepted":
                rotation_accepted,

            "raw_span_xy_m":
                [
                    float(x)
                    for x in raw_span
                ],

            "rotated_span_xy_m":
                [
                    float(x)
                    for x in rot_span
                ],

            "dtype":
                "float32",

            "rounding":
                "1 mm",

            "official_first_ps_current_index":
                first_ps_current_index,

            "xy_file":
                str(
                    XY_OUT
                ),

            "sort_index_file":
                str(
                    SORT_OUT
                ),
        },

    "exact_neighbour_census":
        {
            "radius_m":
                RADIUS,

            "sigma_m":
                SIGMA,

            "mean":
                mean_count,

            "max":
                max_count,

            "p01_p05_p50_p90_p95_p99_p999":
                [
                    float(x)
                    for x in count_q
                ],

            "directed_interactions":
                directed_interactions,

            "undirected_nonself_pairs":
                int(
                    undirected_nonself
                ),

            "undirected_pairs_with_self":
                int(
                    undirected_with_self
                ),

            "count_file":
                str(
                    COUNT_OUT
                ),
        },

    "cell_list_upper_bound":
        {
            "cell_size_m":
                RADIUS,

            "grid_nx":
                nx,

            "grid_ny":
                ny,

            "occupied_cells":
                int(
                    occ_nonzero.size
                ),

            "occupancy_p50_p90_p95_p99":
                [
                    float(x)
                    for x in occ_q
                ],

            "candidate_directed_interactions":
                cell_candidate_directed,

            "candidate_to_true_ratio":
                candidate_to_true_ratio,
        },

    "memory_estimate":
        {
            "global_CSR_float32_weight_GiB":
                csr_float32_gib,

            "global_CSR_float64_weight_GiB":
                csr_float64_gib,

            "mean_nnz_per_4096_point_chunk":
                chunk_4096_mean_nnz,

            "mean_nnz_per_8192_point_chunk":
                chunk_8192_mean_nnz,
        },

    "work_estimate":
        {
            "naive_38_epoch_weighted_interactions":
                int(
                    naive_point_epoch_interactions
                ),
        },

    "timing":
        {
            "xy_seconds":
                xy_seconds,

            "tree_build_seconds":
                tree_seconds,

            "count_query_seconds":
                query_seconds,
        },

    "next_engine_recommendation":
        recommendation,

    "phase_modified":
        False,
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    +
    "\n"
)


# ======================================================================
# Print
# ======================================================================

print("=" * 96)
print("P15-6B0 EXACT STAMPS XY + 400 m NEIGHBOUR CENSUS")
print("=" * 96)

print(
    "points                         :",
    f"{n:,}",
)

print(
    "heading                        :",
    f"{heading:.6f} deg",
)

print(
    "StaMPS rotation theta          :",
    f"{np.rad2deg(theta):.6f} deg",
)

print(
    "rotation accepted              :",
    rotation_accepted,
)

print(
    "raw xy span                    :",
    raw_span,
)

print(
    "rotated xy span                :",
    rot_span,
)

print(
    "official first PS current idx  :",
    first_ps_current_index,
)

print()

print(
    "KDTree build seconds           :",
    f"{tree_seconds:.6f}",
)

print(
    "neighbour census seconds       :",
    f"{query_seconds:.6f}",
)

print()

print(
    "neighbours mean                :",
    f"{mean_count:,.2f}",
)

print(
    "neighbours max                 :",
    f"{max_count:,}",
)

print(
    "neigh p01/05/50/90/95/99/999 :",
    count_q,
)

print()

print(
    "directed interactions          :",
    f"{directed_interactions:,}",
)

print(
    "undirected non-self pairs      :",
    f"{undirected_nonself:,}",
)

print(
    "naive interactions ×38 epochs :",
    f"{naive_point_epoch_interactions:,}",
)

print()

print(
    "400m cell grid                 :",
    f"{ny} x {nx}",
)

print(
    "occupied cells                 :",
    f"{occ_nonzero.size:,}",
)

print(
    "cell occupancy p50/90/95/99    :",
    occ_q,
)

print(
    "cell candidate interactions    :",
    f"{cell_candidate_directed:,}",
)

print(
    "candidate / true ratio         :",
    f"{candidate_to_true_ratio:.4f}",
)

print()

print(
    "global CSR f32 estimate        :",
    f"{csr_float32_gib:.2f} GiB",
)

print(
    "global CSR f64 estimate        :",
    f"{csr_float64_gib:.2f} GiB",
)

print(
    "mean nnz / 4096-point chunk    :",
    f"{chunk_4096_mean_nnz:,.0f}",
)

print()

print(
    "recommended next benchmark     :",
    recommendation,
)

print(
    "exact XY                       :",
    XY_OUT,
)

print(
    "neighbor count                 :",
    COUNT_OUT,
)

print(
    "manifest                       :",
    MANIFEST,
)

print("=" * 96)

print(
    "P15-6B0 FINAL RESULT: "
    "PASS_EXACT_STAMPS_XY_NEIGHBOUR_CENSUS"
)

print("=" * 96)
print("AUDIT ONLY -- NO PHASE MODIFIED")
