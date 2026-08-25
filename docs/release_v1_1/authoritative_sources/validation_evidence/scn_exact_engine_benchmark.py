from pathlib import Path
from datetime import datetime
import json
import math
import time

import numpy as np
from scipy.spatial import cKDTree

from numba import (
    njit,
    prange,
    set_num_threads,
    get_num_threads,
)


ROOT = Path("/home/ubuntu/Downloads")
PSDS = ROOT / "psds"
PROC = PSDS / "output/processing"

PRE = (
    PROC
    / "stamps_pre_scn_phase"
    / "acquisition_phase_pre_scn_rad.npy"
)

GMAN = (
    PROC
    / "gacos_corrected_phase"
    / "gacos_correction_manifest.json"
)

XY = (
    PROC
    / "stamps_stage8"
    / "stamps_xy_exact_float32_m.npy"
)

SORT_IX = (
    PROC
    / "stamps_stage8"
    / "stamps_sort_index.npy"
)

COUNTS = (
    PROC
    / "stamps_stage8"
    / "stage8_neighbor_count_r400m.npy"
)

B0_MANIFEST = (
    PROC
    / "stamps_stage8"
    / "p15_6b0_exact_neighbor_census.json"
)

OUT = (
    PROC
    / "stamps_stage8"
    / "p15_6b1_exact_engine_benchmark.json"
)


MASTER_DATE = "20151212"

TIME_WIN_DAYS = 365.0
SIGMA_M = 100.0
RADIUS_M = 400.0

CELL_SIZES = (
    400.0,
    200.0,
    100.0,
    50.0,
)

SAMPLE_N = 512

PARITY_TOL_RAD = 1.0e-9

TEMP_CHUNK = 131072


# ================================================================
# Exact temporal weight matrix from pySTAMPS stage8_sbas.py
# ================================================================

def temporal_weight_matrix(
    day,
    master0,
    time_win,
):

    dt = (
        day[:, None]
        -
        day[None, :]
    )

    W = np.exp(
        -(
            dt * dt
        )
        /
        (
            2.0
            *
            time_win
            *
            time_win
        )
    )

    # Official:
    # weight_factor(master_ix)=0
    W[:, master0] = 0.0

    den = np.sum(
        W,
        axis=1,
    )

    if np.any(
        den <= 0
    ):
        raise RuntimeError(
            "zero temporal weight denominator"
        )

    W /= den[:, None]

    return W


# ================================================================
# Exact cell-list support
# ================================================================

def build_cell_index(
    coords,
    cell_size,
):

    xmin = float(
        coords[:, 0].min()
    )

    ymin = float(
        coords[:, 1].min()
    )

    cx = np.floor(
        (
            coords[:, 0]
            -
            xmin
        )
        /
        cell_size
    ).astype(
        np.int32
    )

    cy = np.floor(
        (
            coords[:, 1]
            -
            ymin
        )
        /
        cell_size
    ).astype(
        np.int32
    )

    nx = int(
        cx.max()
    ) + 1

    ny = int(
        cy.max()
    ) + 1

    cell_id = (
        cy.astype(
            np.int64
        )
        *
        nx
        +
        cx
    )

    order = np.argsort(
        cell_id,
        kind="stable",
    ).astype(
        np.int32
    )

    ncell = (
        nx
        *
        ny
    )

    occ = np.bincount(
        cell_id,
        minlength=ncell,
    ).astype(
        np.int64
    )

    starts = np.empty(
        ncell + 1,
        dtype=np.int64,
    )

    starts[0] = 0

    np.cumsum(
        occ,
        out=starts[1:],
    )

    return (
        cx,
        cy,
        nx,
        ny,
        order,
        starts,
    )


def build_safe_offsets(
    cell_size,
    radius,
):
    """
    Include every cell offset whose two cell rectangles can
    contain at least one pair with distance < radius.

    For offset k:
      minimum axis separation =
          max(|k|-1, 0) * cell_size
    """

    kmax = int(
        math.ceil(
            radius
            /
            cell_size
        )
    ) + 1

    out = []

    r2 = (
        radius
        *
        radius
    )

    for dy in range(
        -kmax,
        kmax + 1,
    ):

        min_y = (
            max(
                abs(dy) - 1,
                0,
            )
            *
            cell_size
        )

        for dx in range(
            -kmax,
            kmax + 1,
        ):

            min_x = (
                max(
                    abs(dx) - 1,
                    0,
                )
                *
                cell_size
            )

            if (
                min_x * min_x
                +
                min_y * min_y
                <
                r2
            ):
                out.append(
                    (
                        dx,
                        dy,
                    )
                )

    return np.asarray(
        out,
        dtype=np.int32,
    )


@njit(
    parallel=True,
    fastmath=False,
    cache=False,
)
def cell_gaussian_exact(
    coords,
    values,
    targets,
    cx,
    cy,
    nx,
    ny,
    order,
    starts,
    offsets,
    radius_sq,
    sigma2x2,
):
    """
    Exact Stage-8 spatial Gaussian for selected target points.

    - true physical coordinates
    - strict distance < 400 m
    - no KNN truncation
    - no approximate distance
    - float64 Gaussian accumulation
    """

    nt = targets.size
    nepoch = values.shape[1]

    out = np.empty(
        (
            nt,
            nepoch,
        ),
        dtype=np.float64,
    )

    true_count = np.zeros(
        nt,
        dtype=np.int64,
    )

    candidate_count = np.zeros(
        nt,
        dtype=np.int64,
    )

    for ii in prange(
        nt
    ):

        p = targets[ii]

        x0 = coords[p, 0]
        y0 = coords[p, 1]

        pcx = cx[p]
        pcy = cy[p]

        den = 0.0

        acc = np.zeros(
            nepoch,
            dtype=np.float64,
        )

        ntotal = 0
        ncandidate = 0

        for ko in range(
            offsets.shape[0]
        ):

            qx = (
                pcx
                +
                offsets[ko, 0]
            )

            qy = (
                pcy
                +
                offsets[ko, 1]
            )

            if (
                qx < 0
                or
                qx >= nx
                or
                qy < 0
                or
                qy >= ny
            ):
                continue

            cid = (
                qy
                *
                nx
                +
                qx
            )

            q0 = starts[cid]
            q1 = starts[cid + 1]

            ncandidate += (
                q1
                -
                q0
            )

            for kk in range(
                q0,
                q1,
            ):

                q = order[kk]

                dx = (
                    coords[q, 0]
                    -
                    x0
                )

                dy = (
                    coords[q, 1]
                    -
                    y0
                )

                d2 = (
                    dx * dx
                    +
                    dy * dy
                )

                # Official strict radius condition.
                if d2 < radius_sq:

                    w = math.exp(
                        -d2
                        /
                        sigma2x2
                    )

                    den += w
                    ntotal += 1

                    for e in range(
                        nepoch
                    ):
                        acc[e] += (
                            w
                            *
                            values[q, e]
                        )

        if den <= 0.0:
            for e in range(
                nepoch
            ):
                out[ii, e] = np.nan

        else:
            invden = (
                1.0
                /
                den
            )

            for e in range(
                nepoch
            ):
                out[ii, e] = (
                    acc[e]
                    *
                    invden
                )

        true_count[ii] = ntotal
        candidate_count[ii] = ncandidate

    return (
        out,
        true_count,
        candidate_count,
    )


# ================================================================
# Inputs
# ================================================================

for p in (
    PRE,
    GMAN,
    XY,
    SORT_IX,
    COUNTS,
    B0_MANIFEST,
):
    if not p.is_file():
        raise FileNotFoundError(
            p
        )


pre = np.load(
    PRE,
    mmap_mode="r",
)

coords = np.load(
    XY,
    mmap_mode="r",
).astype(
    np.float64
)

sort_ix = np.load(
    SORT_IX
).astype(
    np.int64
)

global_counts = np.load(
    COUNTS,
    mmap_mode="r",
)


gman = json.loads(
    GMAN.read_text()
)

b0 = json.loads(
    B0_MANIFEST.read_text()
)


dates = list(
    gman[
        "acquisition_dates"
    ]
)


npoint, nepoch = pre.shape


if (
    coords.shape != (
        npoint,
        2,
    )
    or
    nepoch != 38
    or
    len(dates) != 38
    or
    sort_ix.size != npoint
):

    raise RuntimeError(
        "input contract failed"
    )


master0 = dates.index(
    MASTER_DATE
)


official_first = int(
    sort_ix[0]
)


expected_first = int(
    b0[
        "coordinate_contract"
    ][
        "official_first_ps_current_index"
    ]
)


if (
    official_first
    !=
    expected_first
):

    raise RuntimeError(
        (
            "official first PS mismatch: "
            f"{official_first} != "
            f"{expected_first}"
        )
    )


# ================================================================
# Temporal high-pass
#
# Official:
#
# H = ph_all - ph_all @ W.T
# ph_hpt = H - H(first_PS)
#
# ph_hpt is stored float32.
# ================================================================

dobj = [
    datetime.strptime(
        d,
        "%Y%m%d",
    )
    for d in dates
]


day = np.asarray(
    [
        (
            x
            -
            dobj[0]
        ).days
        for x in dobj
    ],
    dtype=np.float64,
)


W = temporal_weight_matrix(
    day,
    master0,
    TIME_WIN_DAYS,
)


first = np.asarray(
    pre[
        official_first,
        :
    ],
    dtype=np.float64,
)


h0 = (
    first
    -
    first
    @
    W.T
)


ph_hpt = np.empty(
    (
        npoint,
        nepoch,
    ),
    dtype=np.float32,
)


t_temporal = (
    time.perf_counter()
)


for start in range(
    0,
    npoint,
    TEMP_CHUNK,
):

    stop = min(
        start
        +
        TEMP_CHUNK,
        npoint,
    )

    y = np.asarray(
        pre[
            start:stop,
            :
        ],
        dtype=np.float64,
    )

    h = (
        y
        -
        y
        @
        W.T
        -
        h0[
            None,
            :
        ]
    )

    ph_hpt[
        start:stop,
        :
    ] = h.astype(
        np.float32
    )


temporal_seconds = (
    time.perf_counter()
    -
    t_temporal
)


first_hpt_max = float(
    np.max(
        np.abs(
            ph_hpt[
                official_first,
                :
            ]
        )
    )
)


TEMPORAL_REFERENCE_TOL_RAD = 1.0e-12

if first_hpt_max > TEMPORAL_REFERENCE_TOL_RAD:

    raise RuntimeError(
        (
            "official first PS temporal "
            "reference failed: "
            f"{first_hpt_max:.12e} rad "
            f"> {TEMPORAL_REFERENCE_TOL_RAD:.1e} rad"
        )
    )


# ================================================================
# Deterministic representative sample
# ================================================================

rng = np.random.default_rng(
    20260824
)


targets = rng.choice(
    npoint,
    size=SAMPLE_N,
    replace=False,
).astype(
    np.int64
)


sample_global_counts = np.asarray(
    global_counts[
        targets
    ],
    dtype=np.int64,
)


# ================================================================
# KDTree oracle
# ================================================================

tree = cKDTree(
    coords,
    compact_nodes=True,
    balanced_tree=True,
)


t0 = time.perf_counter()


neighbour_lists = (
    tree.query_ball_point(
        coords[
            targets
        ],
        r=RADIUS_M,
        workers=-1,
    )
)


oracle = np.empty(
    (
        SAMPLE_N,
        nepoch,
    ),
    dtype=np.float64,
)


oracle_counts = np.empty(
    SAMPLE_N,
    dtype=np.int64,
)


oracle_true_interactions = 0


for ii in range(
    SAMPLE_N
):

    p = targets[ii]

    q = np.asarray(
        neighbour_lists[ii],
        dtype=np.int64,
    )

    dxy = (
        coords[q, :]
        -
        coords[
            p,
            :
        ]
    )

    d2 = np.sum(
        dxy
        *
        dxy,
        axis=1,
    )

    # pySTAMPS uses strict < radius^2 after KDTree lookup.
    use = (
        d2
        <
        RADIUS_M
        *
        RADIUS_M
    )

    q = q[
        use
    ]

    d2 = d2[
        use
    ]

    w = np.exp(
        -d2
        /
        (
            2.0
            *
            SIGMA_M
            *
            SIGMA_M
        )
    )

    den = np.sum(
        w,
        dtype=np.float64,
    )

    oracle[ii, :] = (
        w
        @
        ph_hpt[
            q,
            :
        ].astype(
            np.float64
        )
    ) / den

    oracle_counts[ii] = (
        q.size
    )

    oracle_true_interactions += int(
        q.size
    )


kdtree_seconds = (
    time.perf_counter()
    -
    t0
)


del neighbour_lists


# return_length census used <=R.
# Exact-boundary points are allowed to differ, but report it.
global_vs_strict_count_diff = (
    sample_global_counts
    -
    oracle_counts
)


# ================================================================
# Numba setup
# ================================================================

available_threads = int(
    get_num_threads()
)


threads = min(
    32,
    available_threads,
)


set_num_threads(
    threads
)


# Compile outside benchmark.
cx0, cy0, nx0, ny0, order0, starts0 = (
    build_cell_index(
        coords,
        100.0,
    )
)


offsets0 = build_safe_offsets(
    100.0,
    RADIUS_M,
)


_ = cell_gaussian_exact(
    coords,
    ph_hpt,
    targets[:2],
    cx0,
    cy0,
    nx0,
    ny0,
    order0,
    starts0,
    offsets0,
    RADIUS_M
    *
    RADIUS_M,
    2.0
    *
    SIGMA_M
    *
    SIGMA_M,
)


del (
    cx0,
    cy0,
    order0,
    starts0,
    offsets0,
)


# ================================================================
# Cell-size sweep
# ================================================================

results = []


full_directed = int(
    b0[
        "exact_neighbour_census"
    ][
        "directed_interactions"
    ]
)


for cell_size in CELL_SIZES:

    t_build = time.perf_counter()

    (
        cx,
        cy,
        nx,
        ny,
        order,
        starts,
    ) = build_cell_index(
        coords,
        cell_size,
    )

    offsets = build_safe_offsets(
        cell_size,
        RADIUS_M,
    )

    build_seconds = (
        time.perf_counter()
        -
        t_build
    )


    t_kernel = (
        time.perf_counter()
    )


    (
        got,
        true_count,
        candidate_count,
    ) = cell_gaussian_exact(
        coords,
        ph_hpt,
        targets,
        cx,
        cy,
        nx,
        ny,
        order,
        starts,
        offsets,
        RADIUS_M
        *
        RADIUS_M,
        2.0
        *
        SIGMA_M
        *
        SIGMA_M,
    )


    kernel_seconds = (
        time.perf_counter()
        -
        t_kernel
    )


    # ------------------------------------------------------------
    # HARD neighbour parity
    # ------------------------------------------------------------

    count_diff = (
        true_count
        -
        oracle_counts
    )


    count_max_abs = int(
        np.max(
            np.abs(
                count_diff
            )
        )
    )


    if count_max_abs != 0:

        raise RuntimeError(
            (
                f"cell={cell_size}: "
                "exact neighbour count parity FAILED; "
                f"max diff={count_max_abs}"
            )
        )


    # ------------------------------------------------------------
    # HARD phase parity
    # ------------------------------------------------------------

    diff = (
        got
        -
        oracle
    )


    max_abs = float(
        np.max(
            np.abs(
                diff
            )
        )
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


    if (
        max_abs
        >
        PARITY_TOL_RAD
    ):

        raise RuntimeError(
            (
                f"cell={cell_size}: "
                "phase parity FAILED: "
                f"{max_abs}"
            )
        )


    candidate_total = int(
        np.sum(
            candidate_count,
            dtype=np.int64,
        )
    )


    true_total = int(
        np.sum(
            true_count,
            dtype=np.int64,
        )
    )


    ratio = float(
        candidate_total
        /
        true_total
    )


    true_per_sec = float(
        true_total
        /
        kernel_seconds
    )


    candidate_per_sec = float(
        candidate_total
        /
        kernel_seconds
    )


    phase_accum_per_sec = float(
        true_total
        *
        nepoch
        /
        kernel_seconds
    )


    estimated_full_seconds = float(
        kernel_seconds
        *
        full_directed
        /
        true_total
    )


    results.append(
        {
            "cell_size_m":
                cell_size,

            "grid_nx":
                nx,

            "grid_ny":
                ny,

            "offset_cells":
                int(
                    offsets.shape[0]
                ),

            "build_seconds":
                build_seconds,

            "kernel_seconds":
                kernel_seconds,

            "true_interactions":
                true_total,

            "candidate_interactions":
                candidate_total,

            "candidate_to_true_ratio":
                ratio,

            "true_interactions_per_second":
                true_per_sec,

            "candidate_evaluations_per_second":
                candidate_per_sec,

            "phase_accumulations_per_second":
                phase_accum_per_sec,

            "estimated_full_spatial_seconds":
                estimated_full_seconds,

            "estimated_full_spatial_minutes":
                estimated_full_seconds
                /
                60.0,

            "neighbor_count_max_abs_diff":
                count_max_abs,

            "phase_max_abs_diff_rad":
                max_abs,

            "phase_rms_diff_rad":
                rms,
        }
    )


    print(
        f"[CELL {cell_size:6.1f} m] "
        f"kernel={kernel_seconds:.4f}s "
        f"candidate/true={ratio:.4f} "
        f"parity={max_abs:.3e} rad "
        f"est_full={estimated_full_seconds/60:.2f} min",
        flush=True,
    )


# ================================================================
# Ranking
# ================================================================

results.sort(
    key=lambda x:
        x[
            "estimated_full_spatial_seconds"
        ]
)


best = results[0]


kdtree_estimated_full_seconds = float(
    kdtree_seconds
    *
    full_directed
    /
    oracle_true_interactions
)


speedup_vs_kdtree = float(
    kdtree_estimated_full_seconds
    /
    best[
        "estimated_full_spatial_seconds"
    ]
)


# ================================================================
# Decision
# ================================================================

if (
    best[
        "phase_max_abs_diff_rad"
    ]
    <=
    PARITY_TOL_RAD
    and
    best[
        "neighbor_count_max_abs_diff"
    ]
    ==
    0
):

    recommendation = (
        "USE_NUMBA_EXACT_CELL_LIST"
    )

else:

    recommendation = (
        "USE_CKDTREE_STREAMING_ORACLE"
    )


manifest = {
    "status":
        "PASS_EXACT_ENGINE_BENCHMARK",

    "scientific_contract":
        {
            "coordinates":
                "exact StaMPS float32+1mm XY",

            "temporal_time_window_days":
                TIME_WIN_DAYS,

            "spatial_sigma_m":
                SIGMA_M,

            "strict_radius_m":
                RADIUS_M,

            "epochs":
                nepoch,

            "official_first_ps_current_index":
                official_first,

            "temporal_first_ps_max_abs_rad":
                first_hpt_max,
        },

    "sample":
        {
            "points":
                SAMPLE_N,

            "strict_true_interactions":
                oracle_true_interactions,

            "neighbor_p50_p95_p99":
                [
                    float(x)
                    for x in np.percentile(
                        oracle_counts,
                        [
                            50,
                            95,
                            99,
                        ],
                    )
                ],

            "census_minus_strict_min":
                int(
                    global_vs_strict_count_diff.min()
                ),

            "census_minus_strict_max":
                int(
                    global_vs_strict_count_diff.max()
                ),
        },

    "temporal":
        {
            "seconds":
                temporal_seconds,

            "first_ps_max_abs_rad":
                first_hpt_max,
        },

    "kdtree_oracle":
        {
            "sample_seconds":
                kdtree_seconds,

            "true_interactions":
                oracle_true_interactions,

            "estimated_full_spatial_seconds":
                kdtree_estimated_full_seconds,

            "estimated_full_spatial_minutes":
                kdtree_estimated_full_seconds
                /
                60.0,
        },

    "numba":
        {
            "threads":
                threads,

            "available_threads":
                available_threads,

            "parity_tolerance_rad":
                PARITY_TOL_RAD,

            "cell_size_sweep":
                results,
        },

    "best":
        {
            **best,

            "estimated_speedup_vs_kdtree":
                speedup_vs_kdtree,
        },

    "full_problem":
        {
            "points":
                npoint,

            "directed_interactions":
                full_directed,

            "point_epoch_interactions":
                full_directed
                *
                nepoch,
        },

    "recommendation":
        recommendation,

    "phase_modified":
        False,
}


OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    +
    "\n"
)


print()
print("=" * 96)
print("P15-6B1 EXACT STAGE-8 ENGINE BENCHMARK")
print("=" * 96)

print(
    "sample points                   :",
    SAMPLE_N,
)

print(
    "sample strict interactions      :",
    f"{oracle_true_interactions:,}",
)

print(
    "sample neighbours p50/p95/p99   :",
    np.percentile(
        oracle_counts,
        [
            50,
            95,
            99,
        ],
    ),
)

print()

print(
    "temporal high-pass seconds      :",
    f"{temporal_seconds:.6f}",
)

print(
    "official first PS max |H|       :",
    f"{first_hpt_max:.12e}",
)

print()

print(
    "KDTree oracle sample seconds    :",
    f"{kdtree_seconds:.6f}",
)

print(
    "KDTree extrapolated full        :",
    (
        f"{kdtree_estimated_full_seconds/60:.2f} "
        "min"
    ),
)

print()

for x in results:

    print(
        "cell "
        f"{x['cell_size_m']:>6.1f} m | "
        f"offsets={x['offset_cells']:>3d} | "
        f"cand/true={x['candidate_to_true_ratio']:.4f} | "
        f"time={x['kernel_seconds']:.4f}s | "
        f"parity={x['phase_max_abs_diff_rad']:.3e} | "
        f"full≈{x['estimated_full_spatial_minutes']:.2f} min"
    )

print()

print(
    "BEST cell size                 :",
    f"{best['cell_size_m']:.1f} m",
)

print(
    "BEST candidate/true            :",
    f"{best['candidate_to_true_ratio']:.4f}",
)

print(
    "BEST phase parity max          :",
    f"{best['phase_max_abs_diff_rad']:.12e} rad",
)

print(
    "BEST neighbour count diff      :",
    best[
        "neighbor_count_max_abs_diff"
    ],
)

print(
    "BEST extrapolated full         :",
    (
        f"{best['estimated_full_spatial_minutes']:.2f} "
        "min"
    ),
)

print(
    "speedup vs KDTree oracle       :",
    f"{speedup_vs_kdtree:.3f}x",
)

print(
    "Numba threads                  :",
    threads,
)

print()

print(
    "recommended production engine :",
    recommendation,
)

print(
    "report                         :",
    OUT,
)

print("=" * 96)
print(
    "P15-6B1 FINAL RESULT: "
    "PASS_EXACT_ENGINE_BENCHMARK"
)
print("=" * 96)
print("BENCHMARK ONLY -- NO PHASE MODIFIED")
