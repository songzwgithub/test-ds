from pathlib import Path
import json
import math
import os
import re
import time
from datetime import datetime, timezone

import numpy as np
from numba import njit, prange


PSDS = Path(
    "/home/ubuntu/Downloads/psds"
)

PROC = (
    PSDS
    / "output/processing"
)

PRE = (
    PROC
    / "referenced_timeseries"
    / "acquisition_phase_referenced_rad.npy"
)

POST = (
    PROC
    / "gacos_corrected_phase"
    / "acquisition_phase_gacos_corrected_rad.npy"
)

GMAN = (
    PROC
    / "gacos_corrected_phase"
    / "gacos_correction_manifest.json"
)

GEOM = (
    PROC
    / "gacos_geometry"
)

OUT = (
    PROC
    / "scla_residual_dem_estimation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


BETA = (
    OUT
    / "scla_beta_rad_per_m_bperp.npy"
)

SIGMA = (
    OUT
    / "scla_beta_sigma_rad_per_m_bperp.npy"
)

IMPROVE = (
    OUT
    / "scla_partial_r2.npy"
)

MANIFEST = (
    OUT
    / "scla_estimation_manifest.json"
)


CHUNK = int(
    os.environ.get(
        "P15_SCLA_CHUNK_POINTS",
        "262144",
    )
)


# ======================================================================
# Input contract
# ======================================================================

for p in (
    PRE,
    POST,
    GMAN,
    GEOM / "strict_points.plist",
):

    if not p.is_file():

        raise FileNotFoundError(
            p
        )


gman = json.loads(
    GMAN.read_text()
)


dates = list(
    gman[
        "acquisition_dates"
    ]
)


if (
    len(dates) != 38
    or
    dates[0] != "20141006"
):

    raise RuntimeError(
        (
            "date contract failed: "
            f"n={len(dates)}, "
            f"first={dates[:1]}"
        )
    )


src_pre = np.load(
    PRE,
    mmap_mode="r",
)


src_post = np.load(
    POST,
    mmap_mode="r",
)


if (
    src_pre.shape
    !=
    src_post.shape
):

    raise RuntimeError(
        (
            "pre/post phase shape mismatch: "
            f"{src_pre.shape} "
            f"{src_post.shape}"
        )
    )


if (
    src_post.ndim != 2
    or
    src_post.shape[1] != len(dates)
):

    raise RuntimeError(
        f"phase shape contract failed: {src_post.shape}"
    )


if (
    src_pre.dtype != np.float32
    or
    src_post.dtype != np.float32
):

    raise RuntimeError(
        (
            "phase dtype contract failed: "
            f"{src_pre.dtype} "
            f"{src_post.dtype}"
        )
    )


npoint, ndate = src_post.shape


# ======================================================================
# Locate production acquisition Bperp
#
# All valid copies must numerically agree.
# ======================================================================

candidates = []


for base in (
    PROC,
    PSDS / "output",
):

    if not base.exists():
        continue


    for p in base.rglob(
        "acquisition_bperp_m.npy"
    ):

        try:

            x = np.load(
                p
            )

            x = np.asarray(
                x,
                dtype=np.float64,
            ).reshape(
                -1
            )

        except Exception:

            continue


        if (
            x.size == ndate
            and
            np.all(
                np.isfinite(
                    x
                )
            )
        ):

            candidates.append(
                (
                    p,
                    x,
                )
            )


# de-duplicate paths
uniq = {}


for p, x in candidates:

    uniq[
        str(
            p.resolve()
        )
    ] = (
        p,
        x,
    )


candidates = list(
    uniq.values()
)


if not candidates:

    raise RuntimeError(
        "No valid acquisition_bperp_m.npy "
        "with 38 finite values found"
    )


b_path, b = candidates[0]


for p, x in candidates[1:]:

    if not np.allclose(
        x,
        b,
        rtol=0.0,
        atol=1e-5,
    ):

        raise RuntimeError(
            (
                "Conflicting acquisition_bperp_m.npy files:\n"
                f"  {b_path}\n"
                f"  {p}\n"
                f"max diff={np.max(np.abs(x-b))}"
            )
        )


# ----------------------------------------------------------------------
# Phase epoch 0 = 20141006.
#
# Therefore residual-topography Bperp basis must use
# the SAME temporal origin.
# ----------------------------------------------------------------------

brel = (
    b
    -
    b[0]
)


# ======================================================================
# Time/deformation nuisance basis
#
# All basis functions are exactly zero at epoch 0.
#
#   linear
#   annual sine
#   annual cosine - 1
#
# This prevents seasonal/linear physical deformation from leaking
# unnecessarily into the Bperp-correlated SCLA coefficient.
# ======================================================================

def parse_date(
    s,
):

    return datetime.strptime(
        s,
        "%Y%m%d",
    ).replace(
        tzinfo=timezone.utc
    )


t0 = parse_date(
    dates[0]
)


days = np.array(
    [
        (
            parse_date(d)
            -
            t0
        ).total_seconds()
        /
        86400.0

        for d in dates
    ],
    dtype=np.float64,
)


ty = (
    days
    /
    365.2425
)


omega = (
    2.0
    *
    np.pi
)


X0 = np.column_stack(
    [
        ty,

        np.sin(
            omega
            *
            ty
        ),

        np.cos(
            omega
            *
            ty
        )
        -
        1.0,
    ]
).astype(
    np.float64
)


if X0.shape[1] != 3:

    raise RuntimeError(
        "unexpected nuisance design"
    )


rank0 = int(
    np.linalg.matrix_rank(
        X0
    )
)


if rank0 != 3:

    raise RuntimeError(
        f"nuisance design rank={rank0}"
    )


# Orthogonal basis
Q0, _ = np.linalg.qr(
    X0,
    mode="reduced",
)


# ======================================================================
# Frisch-Waugh-Lovell:
#
# residualize Bperp against nuisance deformation ONCE.
# ======================================================================

bres = (
    brel
    -
    Q0
    @
    (
        Q0.T
        @
        brel
    )
)


den = float(
    np.dot(
        bres,
        bres,
    )
)


if (
    not np.isfinite(
        den
    )
    or
    den <= 0.0
):

    raise RuntimeError(
        "Bperp is not identifiable after nuisance residualization"
    )


Xfull = np.column_stack(
    [
        X0,
        brel,
    ]
)


rank_full = int(
    np.linalg.matrix_rank(
        Xfull
    )
)


# normalized condition number
column_norm = np.linalg.norm(
    Xfull,
    axis=0,
)


Xnorm = (
    Xfull
    /
    column_norm[
        None,
        :
    ]
)


cond_full = float(
    np.linalg.cond(
        Xnorm
    )
)


if (
    rank_full
    !=
    Xfull.shape[1]
):

    raise RuntimeError(
        (
            "full design rank deficient: "
            f"{rank_full}/"
            f"{Xfull.shape[1]}"
        )
    )


dof = (
    ndate
    -
    rank_full
)


if dof <= 0:

    raise RuntimeError(
        f"invalid dof={dof}"
    )


# ======================================================================
# FAST estimator
#
# No per-point lstsq.
# No per-point temporary arrays.
#
# For each point:
#
#   beta = y' bres / (bres' bres)
#
# Also calculate:
#
#   sigma(beta)
#   partial R^2 of Bperp after removing nuisance basis
# ======================================================================

@njit(
    parallel=True,
    fastmath=False,
    cache=True,
)
def estimate_block(
    y,
    q0,
    bres,
    den,
    dof,
    beta,
    sigma,
    partial_r2,
):

    n = y.shape[0]

    nt = y.shape[1]


    for i in prange(
        n
    ):

        y2 = 0.0

        num = 0.0


        c0 = 0.0
        c1 = 0.0
        c2 = 0.0


        for t in range(
            nt
        ):

            v = float(
                y[
                    i,
                    t,
                ]
            )


            y2 += (
                v
                *
                v
            )


            num += (
                v
                *
                bres[t]
            )


            c0 += (
                v
                *
                q0[
                    t,
                    0,
                ]
            )

            c1 += (
                v
                *
                q0[
                    t,
                    1,
                ]
            )

            c2 += (
                v
                *
                q0[
                    t,
                    2,
                ]
            )


        proj = (
            c0*c0
            +
            c1*c1
            +
            c2*c2
        )


        sse0 = (
            y2
            -
            proj
        )


        if (
            sse0 < 0.0
            and
            sse0 > -1e-9
        ):

            sse0 = 0.0


        bt = (
            num
            /
            den
        )


        reduction = (
            num
            *
            num
            /
            den
        )


        sse1 = (
            sse0
            -
            reduction
        )


        if (
            sse1 < 0.0
            and
            sse1 > -1e-9
        ):

            sse1 = 0.0


        beta[i] = bt


        if sse1 >= 0.0:

            sigma[i] = math.sqrt(
                (
                    sse1
                    /
                    dof
                )
                /
                den
            )

        else:

            sigma[i] = np.nan


        if sse0 > 0.0:

            r2 = (
                reduction
                /
                sse0
            )


            if r2 < 0.0:

                r2 = 0.0

            elif r2 > 1.0:

                r2 = 1.0


            partial_r2[i] = r2

        else:

            partial_r2[i] = 0.0


# ======================================================================
# JIT warm-up
# ======================================================================

warm_n = min(
    2048,
    npoint,
)


yw = np.asarray(
    src_post[
        :warm_n,
        :,
    ],
    dtype=np.float32,
)


bw = np.empty(
    warm_n,
    np.float32,
)

sw = np.empty(
    warm_n,
    np.float32,
)

rw = np.empty(
    warm_n,
    np.float32,
)


estimate_block(
    yw,
    Q0,
    bres,
    den,
    dof,
    bw,
    sw,
    rw,
)


# ======================================================================
# Output arrays
# ======================================================================

beta_post = np.lib.format.open_memmap(
    BETA,
    mode="w+",
    dtype=np.float32,
    shape=(
        npoint,
    ),
)


sigma_post = np.lib.format.open_memmap(
    SIGMA,
    mode="w+",
    dtype=np.float32,
    shape=(
        npoint,
    ),
)


r2_post = np.lib.format.open_memmap(
    IMPROVE,
    mode="w+",
    dtype=np.float32,
    shape=(
        npoint,
    ),
)


# Pre-GACOS beta is diagnostic only.
# It is not persisted.
beta_pre = np.empty(
    npoint,
    dtype=np.float32,
)


# ======================================================================
# Chunked estimation
# ======================================================================

t_est = time.perf_counter()


for s in range(
    0,
    npoint,
    CHUNK,
):

    e = min(
        npoint,
        s + CHUNK,
    )


    m = (
        e - s
    )


    # --------------------------------------------------------------
    # POST GACOS
    # --------------------------------------------------------------

    yp = np.asarray(
        src_post[
            s:e,
            :,
        ],
        dtype=np.float32,
    )


    btmp = np.empty(
        m,
        np.float32,
    )

    stmp = np.empty(
        m,
        np.float32,
    )

    rtmp = np.empty(
        m,
        np.float32,
    )


    estimate_block(
        yp,
        Q0,
        bres,
        den,
        dof,
        btmp,
        stmp,
        rtmp,
    )


    beta_post[
        s:e
    ] = btmp


    sigma_post[
        s:e
    ] = stmp


    r2_post[
        s:e
    ] = rtmp


    # --------------------------------------------------------------
    # PRE GACOS
    #
    # Same estimator, only for comparison of atmospheric leakage.
    # --------------------------------------------------------------

    y0 = np.asarray(
        src_pre[
            s:e,
            :,
        ],
        dtype=np.float32,
    )


    b0 = np.empty(
        m,
        np.float32,
    )

    s0 = np.empty(
        m,
        np.float32,
    )

    r0 = np.empty(
        m,
        np.float32,
    )


    estimate_block(
        y0,
        Q0,
        bres,
        den,
        dof,
        b0,
        s0,
        r0,
    )


    beta_pre[
        s:e
    ] = b0


beta_post.flush()
sigma_post.flush()
r2_post.flush()


estimate_seconds = (
    time.perf_counter()
    -
    t_est
)


beta = np.asarray(
    beta_post
)

sigma = np.asarray(
    sigma_post
)

partial_r2 = np.asarray(
    r2_post
)


# ======================================================================
# Numerical QA
# ======================================================================

finite = (
    np.isfinite(
        beta
    )
    &
    np.isfinite(
        sigma
    )
    &
    np.isfinite(
        partial_r2
    )
)


finite_fraction = float(
    finite.mean()
)


if (
    finite_fraction
    <
    0.999999
):

    raise RuntimeError(
        (
            "non-finite SCLA outputs: "
            f"{finite_fraction:.9f}"
        )
    )


# ======================================================================
# Same frozen 607-point reference region
# ======================================================================

ref_idx = np.load(
    PROC
    / "gacos_fast_smoke"
    / "project_local_interp"
    / "ref_idx.npy"
)


if ref_idx.size != 607:

    raise RuntimeError(
        (
            "reference point count "
            f"{ref_idx.size}"
        )
    )


ref_beta_med = float(
    np.median(
        beta[
            ref_idx
        ].astype(
            np.float64
        )
    )
)


# ======================================================================
# Spatial-structure QA
#
# Real residual DEM/SCLA should be spatially structured.
# ======================================================================

plist = np.fromfile(
    GEOM
    / "strict_points.plist",
    dtype=">i4",
).reshape(
    -1,
    2,
)


if plist.shape[0] != npoint:

    raise RuntimeError(
        "plist point count mismatch"
    )


col = plist[
    :,
    0
].astype(
    np.int64
)


row = plist[
    :,
    1
].astype(
    np.int64
)


H = (
    int(
        row.max()
    )
    +
    1
)


W = (
    int(
        col.max()
    )
    +
    1
)


grid = np.full(
    (
        H,
        W,
    ),
    np.nan,
    dtype=np.float32,
)


grid[
    row,
    col
] = beta


dh = np.abs(
    grid[
        :,
        1:
    ]
    -
    grid[
        :,
        :-1
    ]
)


dv = np.abs(
    grid[
        1:,
        :
    ]
    -
    grid[
        :-1,
        :
    ]
)


adj_h = dh[
    np.isfinite(
        dh
    )
]


adj_v = dv[
    np.isfinite(
        dv
    )
]


adj = np.concatenate(
    (
        adj_h,
        adj_v,
    )
)


adj_med = float(
    np.median(
        adj
    )
)


# Deterministic random-pair control
rng = np.random.default_rng(
    20260824
)


nr = min(
    500_000,
    npoint,
)


ia = rng.integers(
    0,
    npoint,
    size=nr,
)


ib = rng.integers(
    0,
    npoint,
    size=nr,
)


rand_med = float(
    np.median(
        np.abs(
            beta[
                ia
            ].astype(
                np.float64
            )
            -
            beta[
                ib
            ].astype(
                np.float64
            )
        )
    )
)


spatial_ratio = (
    adj_med
    /
    rand_med
    if rand_med > 0.0
    else np.nan
)


# ======================================================================
# PRE vs POST GACOS audit
# ======================================================================

bp = beta.astype(
    np.float64
)


bpre = beta_pre.astype(
    np.float64
)


corr_pre_post = float(
    np.corrcoef(
        bpre,
        bp,
    )[0, 1]
)


dbeta = (
    bp
    -
    bpre
)


# ======================================================================
# Statistics
# ======================================================================

abs_beta = np.abs(
    bp
)


beta_q = np.percentile(
    bp,
    [
        1,
        5,
        50,
        95,
        99,
    ],
)


abs_beta_q = np.percentile(
    abs_beta,
    [
        50,
        95,
        99,
    ],
)


sigma_q = np.percentile(
    sigma.astype(
        np.float64
    ),
    [
        50,
        95,
        99,
    ],
)


r2_q = np.percentile(
    partial_r2.astype(
        np.float64
    ),
    [
        50,
        95,
        99,
    ],
)


frac1 = float(
    np.mean(
        partial_r2
        >=
        0.01
    )
)


frac5 = float(
    np.mean(
        partial_r2
        >=
        0.05
    )
)


frac10 = float(
    np.mean(
        partial_r2
        >=
        0.10
    )
)


baseline_span = float(
    np.ptp(
        brel
    )
)


# Approximate total baseline-correlated phase range
# across the acquisition baseline span.
topo_phase_span_abs = (
    abs_beta
    *
    baseline_span
)


topo_span_q = np.percentile(
    topo_phase_span_abs,
    [
        50,
        95,
        99,
    ],
)


corr_t_b = float(
    np.corrcoef(
        ty,
        brel,
    )[0, 1]
)


# ======================================================================
# Acceptance
#
# This stage ESTIMATES ONLY.
# No phase correction is applied here.
# ======================================================================

status = (
    "PASS_SCLA_ESTIMATION_CANDIDATE"
)


if (
    not np.isfinite(
        spatial_ratio
    )
    or
    spatial_ratio >= 0.25
):

    status = (
        "FAIL_SCLA_SPATIAL_STRUCTURE"
    )


if cond_full >= 10.0:

    status = (
        "FAIL_SCLA_DESIGN_CONDITION"
    )


# ======================================================================
# Manifest
# ======================================================================

manifest = {

    "status":
        status,

    "production_phase_modified":
        False,

    "source_phase":
        str(
            POST
        ),

    "comparison_source_pre_gacos":
        str(
            PRE
        ),

    "points":
        int(
            npoint
        ),

    "epochs":
        int(
            ndate
        ),

    "dates":
        dates,

    "baseline_source":
        str(
            b_path
        ),

    "baseline_reference":
        {
            "temporal_reference_date":
                dates[0],

            "bperp_reference_value_m":
                float(
                    b[0]
                ),

            "relative_bperp_min_m":
                float(
                    brel.min()
                ),

            "relative_bperp_max_m":
                float(
                    brel.max()
                ),

            "relative_bperp_span_m":
                baseline_span,
        },

    "model":
        {
            "method":
                (
                    "Frisch-Waugh-Lovell "
                    "partial regression"
                ),

            "target":
                (
                    "baseline-correlated phase "
                    "coefficient beta [rad/m Bperp]"
                ),

            "nuisance_basis":
                [
                    "linear time",
                    "annual sine anchored at epoch0",
                    "annual cosine-minus-one anchored at epoch0",
                ],

            "phase_model":
                (
                    "phi = nuisance_deformation "
                    "+ beta*Bperp_relative "
                    "+ residual"
                ),

            "rank_nuisance":
                rank0,

            "rank_full":
                rank_full,

            "normalized_condition_number":
                cond_full,

            "corr_time_bperp":
                corr_t_b,

            "degrees_of_freedom":
                dof,
        },

    "performance":
        {
            "chunk_points":
                CHUNK,

            "estimate_pre_and_post_seconds":
                estimate_seconds,

            "point_fits_per_second":
                float(
                    2
                    *
                    npoint
                    /
                    estimate_seconds
                ),
        },

    "qa":
        {
            "finite_fraction":
                finite_fraction,

            "beta_rad_per_m_p01_p05_p50_p95_p99":
                [
                    float(x)
                    for x in beta_q
                ],

            "abs_beta_rad_per_m_p50_p95_p99":
                [
                    float(x)
                    for x in abs_beta_q
                ],

            "sigma_beta_p50_p95_p99":
                [
                    float(x)
                    for x in sigma_q
                ],

            "partial_r2_p50_p95_p99":
                [
                    float(x)
                    for x in r2_q
                ],

            "fraction_partial_r2_ge_0p01":
                frac1,

            "fraction_partial_r2_ge_0p05":
                frac5,

            "fraction_partial_r2_ge_0p10":
                frac10,

            "abs_topographic_phase_span_rad_p50_p95_p99":
                [
                    float(x)
                    for x in topo_span_q
                ],

            "reference_beta_median_rad_per_m":
                ref_beta_med,

            "adjacent_abs_dbeta_median":
                adj_med,

            "random_abs_dbeta_median":
                rand_med,

            "adjacent_to_random_ratio":
                spatial_ratio,

            "pre_post_gacos_beta_correlation":
                corr_pre_post,

            "post_minus_pre_beta_p01_p50_p99":
                [
                    float(x)
                    for x in np.percentile(
                        dbeta,
                        [
                            1,
                            50,
                            99,
                        ],
                    )
                ],
        },

    "outputs":
        {
            "beta":
                str(
                    BETA
                ),

            "sigma_beta":
                str(
                    SIGMA
                ),

            "partial_r2":
                str(
                    IMPROVE
                ),
        },

    "policy":
        {
            "correction_applied":
                False,

            "signed_dem_height_conversion":
                (
                    "deferred; beta*Bperp correction "
                    "does not require signed DEM-height conversion"
                ),

            "next":
                "P15-5C_SCLA_CORRECTION_GATE_AND_APPLICATION",
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


# ======================================================================
# Summary
# ======================================================================

print(
    "=" * 88
)

print(
    "P15-5B FAST RESIDUAL DEM / SCLA ESTIMATION"
)

print(
    "=" * 88
)


print(
    "points                         :",
    f"{npoint:,}",
)

print(
    "epochs                         :",
    ndate,
)

print(
    "baseline source                :",
    b_path,
)

print(
    "Brel min/max/span m            :",
    (
        f"{brel.min():.6f} / "
        f"{brel.max():.6f} / "
        f"{baseline_span:.6f}"
    ),
)

print(
    "corr(time,Brel)                :",
    f"{corr_t_b:.6f}",
)

print(
    "full design rank / columns     :",
    f"{rank_full} / {Xfull.shape[1]}",
)

print(
    "normalized condition number    :",
    f"{cond_full:.6f}",
)

print(
    "estimate pre+post seconds      :",
    f"{estimate_seconds:.6f}",
)

print(
    "fit throughput                 :",
    (
        f"{2*npoint/estimate_seconds:,.0f} "
        "point-fits/s"
    ),
)


print()

print(
    "beta p01/p05/p50/p95/p99       :",
    beta_q,
)

print(
    "|beta| p50/p95/p99             :",
    abs_beta_q,
)

print(
    "sigma_beta p50/p95/p99         :",
    sigma_q,
)

print(
    "partial R2 p50/p95/p99         :",
    r2_q,
)

print(
    "partial R2 >=1/5/10%           :",
    (
        f"{100*frac1:.3f}% / "
        f"{100*frac5:.3f}% / "
        f"{100*frac10:.3f}%"
    ),
)

print(
    "topo phase span |rad| p50/95/99:",
    topo_span_q,
)


print()

print(
    "reference beta median          :",
    f"{ref_beta_med:.9e} rad/m",
)

print(
    "adjacent |dbeta| median        :",
    f"{adj_med:.9e}",
)

print(
    "random   |dbeta| median        :",
    f"{rand_med:.9e}",
)

print(
    "adjacent/random ratio          :",
    f"{spatial_ratio:.6f}",
)


print()

print(
    "pre/post GACOS beta correlation:",
    f"{corr_pre_post:.6f}",
)

print(
    "post-pre beta p01/p50/p99      :",
    np.percentile(
        dbeta,
        [
            1,
            50,
            99,
        ],
    ),
)


print()

print(
    "production phase modified      :",
    False,
)

print(
    "manifest                       :",
    MANIFEST,
)


print(
    "=" * 88
)

print(
    "P15-5B FINAL RESULT:",
    status,
)

print(
    "=" * 88
)


if (
    status
    !=
    "PASS_SCLA_ESTIMATION_CANDIDATE"
):

    raise SystemExit(
        2
    )
