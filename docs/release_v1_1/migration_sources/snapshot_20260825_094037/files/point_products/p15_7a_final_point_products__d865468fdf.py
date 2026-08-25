from pathlib import Path
from datetime import datetime
import json
import os
import time

import numpy as np


# ======================================================================
# Paths
# ======================================================================

ROOT = Path("/home/ubuntu/Downloads")
PSDS = ROOT / "psds"
PROC = PSDS / "output/processing"

LOS = (
    PROC
    / "final_los_timeseries"
    / "los_displacement_toward_satellite_mm.npy"
)

FINAL_MANIFEST = (
    PROC
    / "final_los_timeseries"
    / "p15_6c_final_los_timeseries_manifest.json"
)

GMAN = (
    PROC
    / "gacos_corrected_phase"
    / "gacos_correction_manifest.json"
)

REF_FILE = (
    PROC
    / "gacos_fast_smoke"
    / "project_local_interp"
    / "ref_idx.npy"
)

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
    / "final_los_products"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


VEL = (
    OUTDIR
    / "los_velocity_toward_satellite_mm_per_year.npy"
)

CUM = (
    OUTDIR
    / "los_cumulative_toward_satellite_mm.npy"
)

RMS = (
    OUTDIR
    / "linear_residual_rms_mm.npy"
)

VEL_SE = (
    OUTDIR
    / "velocity_slope_standard_error_mm_per_year.npy"
)


TMP_VEL = (
    OUTDIR
    / ".velocity.tmp.npy"
)

TMP_CUM = (
    OUTDIR
    / ".cumulative.tmp.npy"
)

TMP_RMS = (
    OUTDIR
    / ".rms.tmp.npy"
)

TMP_VEL_SE = (
    OUTDIR
    / ".velocity_se.tmp.npy"
)

TIME_CONTRACT = (
    OUTDIR
    / "time_axis_contract.npz"
)

MANIFEST = (
    OUTDIR
    / "p15_7a_final_point_products_manifest.json"
)


CHUNK = 131072
SAMPLE_N = 4096

YEAR_DAYS = 365.25

PARITY_TOL_VEL = 1.0e-10
CUM_PARITY_TOL_MM = 1.0e-7


# ======================================================================
# Inputs
# ======================================================================

for p in (
    LOS,
    FINAL_MANIFEST,
    GMAN,
    REF_FILE,
    LON,
    LAT,
    PLIST,
):

    if not p.is_file():
        raise FileNotFoundError(p)


final_manifest = json.loads(
    FINAL_MANIFEST.read_text()
)


if (
    final_manifest.get("status")
    !=
    "PASS_FINAL_REFERENCED_LOS_TIMESERIES"
):

    raise RuntimeError(
        "P15-6C is not frozen PASS"
    )


los = np.load(
    LOS,
    mmap_mode="r",
)


gman = json.loads(
    GMAN.read_text()
)


dates = list(
    gman[
        "acquisition_dates"
    ]
)


npoint, nepoch = los.shape


if (
    nepoch != 38
    or
    len(dates) != 38
):

    raise RuntimeError(
        "time-axis contract failed"
    )


if dates[0] != "20141006":

    raise RuntimeError(
        (
            "frozen temporal datum changed: "
            f"{dates[0]}"
        )
    )


# Temporal datum must still be bit-exact.
epoch0_max = float(
    np.max(
        np.abs(
            los[:, 0]
        )
    )
)


if epoch0_max != 0.0:

    raise RuntimeError(
        (
            "final LOS temporal reference "
            f"changed: {epoch0_max}"
        )
    )


ref_idx = np.load(
    REF_FILE
).astype(
    np.int64
)


if ref_idx.size != 607:

    raise RuntimeError(
        f"reference point count={ref_idx.size}"
    )


lon = np.load(
    LON,
    mmap_mode="r",
)


lat = np.load(
    LAT,
    mmap_mode="r",
)


plist = np.fromfile(
    PLIST,
    dtype=">i4",
).reshape(
    -1,
    2,
)


if (
    lon.size != npoint
    or
    lat.size != npoint
    or
    plist.shape != (npoint, 2)
):

    raise RuntimeError(
        "point geometry contract failed"
    )


# ======================================================================
# Time axis
# ======================================================================

date_obj = [
    datetime.strptime(
        d,
        "%Y%m%d",
    )
    for d in dates
]


days = np.asarray(
    [
        (
            d
            -
            date_obj[0]
        ).days
        for d in date_obj
    ],
    dtype=np.float64,
)


years = (
    days
    /
    YEAR_DAYS
)


# OLS with intercept:
#
# slope =
#   sum((t-tmean)*y) /
#   sum((t-tmean)^2)
#
# Because sum(t-tmean)=0, this is simply:
#
#   y @ tc / Sxx
#
t_mean = float(
    np.mean(years)
)


tc = (
    years
    -
    t_mean
)


Sxx = float(
    tc
    @
    tc
)


if Sxx <= 0.0:

    raise RuntimeError(
        "invalid temporal design"
    )


slope_weights = (
    tc
    /
    Sxx
)


if abs(
    np.sum(
        slope_weights
    )
) > 1e-14:

    raise RuntimeError(
        "slope weights do not annihilate constant"
    )


time_span_days = float(
    days[-1]
    -
    days[0]
)


time_span_years = float(
    years[-1]
    -
    years[0]
)


# ======================================================================
# Independent sample parity against explicit lstsq
# ======================================================================

sample_n = min(
    SAMPLE_N,
    npoint,
)


sample_idx = np.linspace(
    0,
    npoint - 1,
    sample_n,
    dtype=np.int64,
)


Ysample = np.asarray(
    los[
        sample_idx,
        :
    ],
    dtype=np.float64,
)


vel_fast_sample = (
    Ysample
    @
    slope_weights
)


A = np.column_stack(
    (
        np.ones(
            nepoch,
            dtype=np.float64,
        ),
        years,
    )
)


coef_sample = np.linalg.lstsq(
    A,
    Ysample.T,
    rcond=None,
)[0]


vel_explicit_sample = (
    coef_sample[
        1,
        :
    ]
)


vel_parity = float(
    np.max(
        np.abs(
            vel_fast_sample
            -
            vel_explicit_sample
        )
    )
)


if (
    vel_parity
    >
    PARITY_TOL_VEL
):

    raise RuntimeError(
        (
            "velocity fused/lstsq parity failed: "
            f"{vel_parity}"
        )
    )


cum_fast_sample = (
    Ysample[:, -1]
    -
    Ysample[:, 0]
)


cum_explicit_sample = (
    Ysample[:, -1]
)


cum_parity = float(
    np.max(
        np.abs(
            cum_fast_sample
            -
            cum_explicit_sample
        )
    )
)


if (
    cum_parity
    >
    CUM_PARITY_TOL_MM
):

    raise RuntimeError(
        (
            "cumulative parity failed: "
            f"{cum_parity}"
        )
    )


# ======================================================================
# Atomic output preparation
# ======================================================================

for p in (
    TMP_VEL,
    TMP_CUM,
    TMP_RMS,
    TMP_VEL_SE,
):

    if p.exists():
        p.unlink()


vel_out = np.lib.format.open_memmap(
    TMP_VEL,
    mode="w+",
    dtype=np.float32,
    shape=(npoint,),
)


cum_out = np.lib.format.open_memmap(
    TMP_CUM,
    mode="w+",
    dtype=np.float32,
    shape=(npoint,),
)


rms_out = np.lib.format.open_memmap(
    TMP_RMS,
    mode="w+",
    dtype=np.float32,
    shape=(npoint,),
)


vel_se_out = np.lib.format.open_memmap(
    TMP_VEL_SE,
    mode="w+",
    dtype=np.float32,
    shape=(npoint,),
)


# ======================================================================
# Production
# ======================================================================

t0 = time.perf_counter()


for start in range(
    0,
    npoint,
    CHUNK,
):

    stop = min(
        start + CHUNK,
        npoint,
    )


    Y = np.asarray(
        los[
            start:stop,
            :
        ],
        dtype=np.float64,
    )


    # ----------------------------------------------------------
    # Velocity [mm/yr]
    # ----------------------------------------------------------

    slope = (
        Y
        @
        slope_weights
    )


    # ----------------------------------------------------------
    # Intercept
    # ----------------------------------------------------------

    y_mean = np.mean(
        Y,
        axis=1,
    )


    intercept = (
        y_mean
        -
        slope
        *
        t_mean
    )


    # ----------------------------------------------------------
    # Model / residual
    # ----------------------------------------------------------

    model = (
        intercept[
            :,
            None
        ]
        +
        slope[
            :,
            None
        ]
        *
        years[
            None,
            :
        ]
    )


    residual = (
        Y
        -
        model
    )


    SSE = np.sum(
        residual
        *
        residual,
        axis=1,
    )


    rms = np.sqrt(
        SSE
        /
        nepoch
    )


    # ----------------------------------------------------------
    # OLS slope standard error
    #
    # This is a regression-model diagnostic,
    # NOT total geodetic uncertainty.
    # ----------------------------------------------------------

    sigma2 = (
        SSE
        /
        (
            nepoch
            -
            2
        )
    )


    slope_se = np.sqrt(
        sigma2
        /
        Sxx
    )


    # ----------------------------------------------------------
    # Last-minus-first cumulative displacement
    # ----------------------------------------------------------

    cumulative = (
        Y[:, -1]
        -
        Y[:, 0]
    )


    if not (
        np.all(np.isfinite(slope))
        and
        np.all(np.isfinite(cumulative))
        and
        np.all(np.isfinite(rms))
        and
        np.all(np.isfinite(slope_se))
    ):

        raise RuntimeError(
            (
                "non-finite point product "
                f"in {start}:{stop}"
            )
        )


    vel_out[
        start:stop
    ] = slope.astype(
        np.float32
    )


    cum_out[
        start:stop
    ] = cumulative.astype(
        np.float32
    )


    rms_out[
        start:stop
    ] = rms.astype(
        np.float32
    )


    vel_se_out[
        start:stop
    ] = slope_se.astype(
        np.float32
    )


    print(
        "[POINT PRODUCTS] "
        f"{stop:,}/{npoint:,} "
        f"({100*stop/npoint:.1f}%)",
        flush=True,
    )


vel_out.flush()
cum_out.flush()
rms_out.flush()
vel_se_out.flush()


seconds = (
    time.perf_counter()
    -
    t0
)


del vel_out
del cum_out
del rms_out
del vel_se_out


# ======================================================================
# Publish
# ======================================================================

os.replace(
    TMP_VEL,
    VEL,
)


os.replace(
    TMP_CUM,
    CUM,
)


os.replace(
    TMP_RMS,
    RMS,
)


os.replace(
    TMP_VEL_SE,
    VEL_SE,
)


# ======================================================================
# Final QA
# ======================================================================

vel = np.load(
    VEL,
    mmap_mode="r",
)


cum = np.load(
    CUM,
    mmap_mode="r",
)


rms = np.load(
    RMS,
    mmap_mode="r",
)


vel_se = np.load(
    VEL_SE,
    mmap_mode="r",
)


for name, arr in (
    ("velocity", vel),
    ("cumulative", cum),
    ("residual_rms", rms),
    ("velocity_se", vel_se),
):

    if arr.shape != (npoint,):

        raise RuntimeError(
            f"{name} shape={arr.shape}"
        )


    if not np.all(
        np.isfinite(
            arr
        )
    ):

        raise RuntimeError(
            f"{name} contains non-finite values"
        )


# ======================================================================
# Statistics
# ======================================================================

vel_q = np.percentile(
    np.asarray(
        vel,
        dtype=np.float64,
    ),
    [
        1,
        5,
        50,
        95,
        99,
    ],
)


abs_vel_q = np.percentile(
    np.abs(
        np.asarray(
            vel,
            dtype=np.float64,
        )
    ),
    [
        50,
        95,
        99,
    ],
)


cum_q = np.percentile(
    np.asarray(
        cum,
        dtype=np.float64,
    ),
    [
        1,
        5,
        50,
        95,
        99,
    ],
)


rms_q = np.percentile(
    np.asarray(
        rms,
        dtype=np.float64,
    ),
    [
        50,
        95,
        99,
    ],
)


se_q = np.percentile(
    np.asarray(
        vel_se,
        dtype=np.float64,
    ),
    [
        50,
        95,
        99,
    ],
)


reference_velocity_median = float(
    np.median(
        np.asarray(
            vel[
                ref_idx
            ],
            dtype=np.float64,
        )
    )
)


reference_cumulative_median = float(
    np.median(
        np.asarray(
            cum[
                ref_idx
            ],
            dtype=np.float64,
        )
    )
)


# Reference cumulative should be extremely close to zero because
# each epoch is spatially median-referenced.
if abs(
    reference_cumulative_median
) > 1e-5:

    raise RuntimeError(
        (
            "reference cumulative median "
            "unexpectedly nonzero: "
            f"{reference_cumulative_median}"
        )
    )


# ======================================================================
# Store immutable time-axis coefficients
# ======================================================================

np.savez(
    TIME_CONTRACT,

    acquisition_dates=
        np.asarray(
            dates,
            dtype="U8",
        ),

    days_since_reference=
        days,

    years_since_reference=
        years,

    centered_years=
        tc,

    slope_weights_per_year=
        slope_weights,

    temporal_reference_date=
        np.asarray(
            dates[0]
        ),

    time_span_days=
        np.asarray(
            time_span_days,
            dtype=np.float64,
        ),

    time_span_years=
        np.asarray(
            time_span_years,
            dtype=np.float64,
        ),
)


# ======================================================================
# Manifest
# ======================================================================

manifest = {

    "status":
        "PASS_FINAL_LOS_POINT_PRODUCTS",

    "points":
        int(
            npoint
        ),

    "acquisitions":
        int(
            nepoch
        ),

    "scientific_contract":
        {
            "los_positive":
                "toward_satellite",

            "velocity_method":
                (
                    "ordinary least squares "
                    "with intercept over all acquisitions"
                ),

            "velocity_unit":
                "mm/year",

            "year_days":
                YEAR_DAYS,

            "cumulative_definition":
                (
                    "last acquisition minus "
                    "20141006 reference acquisition"
                ),

            "temporal_reference":
                dates[0],

            "time_span_days":
                time_span_days,

            "time_span_years":
                time_span_years,

            "regression_standard_error_note":
                (
                    "Temporal OLS slope standard error only; "
                    "not total InSAR/geodetic uncertainty."
                ),
        },

    "hard_parity":
        {
            "sample_points":
                sample_n,

            "velocity_fused_vs_lstsq_max_abs_mm_per_year":
                vel_parity,

            "velocity_tolerance":
                PARITY_TOL_VEL,

            "cumulative_last_minus_first_parity_mm":
                cum_parity,

            "epoch0_los_max_abs_mm":
                epoch0_max,
        },

    "statistics":
        {
            "velocity_p01_p05_p50_p95_p99_mm_per_year":
                [
                    float(x)
                    for x in vel_q
                ],

            "abs_velocity_p50_p95_p99_mm_per_year":
                [
                    float(x)
                    for x in abs_vel_q
                ],

            "cumulative_p01_p05_p50_p95_p99_mm":
                [
                    float(x)
                    for x in cum_q
                ],

            "linear_residual_rms_p50_p95_p99_mm":
                [
                    float(x)
                    for x in rms_q
                ],

            "velocity_standard_error_p50_p95_p99_mm_per_year":
                [
                    float(x)
                    for x in se_q
                ],

            "reference_velocity_median_mm_per_year":
                reference_velocity_median,

            "reference_cumulative_median_mm":
                reference_cumulative_median,
        },

    "geometry":
        {
            "longitude":
                str(LON),

            "latitude":
                str(LAT),

            "strict_plist":
                str(PLIST),

            "plist_semantics":
                "range/column, azimuth/row",
        },

    "performance":
        {
            "seconds":
                seconds,

            "points_per_second":
                (
                    npoint
                    /
                    seconds
                ),
        },

    "outputs":
        {
            "velocity":
                str(VEL),

            "cumulative":
                str(CUM),

            "linear_residual_rms":
                str(RMS),

            "velocity_slope_standard_error":
                str(VEL_SE),

            "time_axis_contract":
                str(TIME_CONTRACT),

            "dtype":
                "float32",
        },

    "upstream_phase_modified":
        False,

    "next":
        (
            "P15-7B geocoding contract audit "
            "and exact GeoTIFF export"
        ),
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
print("P15-7A FINAL LOS POINT PRODUCTS")
print("=" * 96)

print(
    "points / acquisitions           :",
    f"{npoint:,} / {nepoch}",
)

print(
    "time span                       :",
    (
        f"{time_span_days:.0f} days / "
        f"{time_span_years:.6f} years"
    ),
)

print(
    "velocity                        :",
    "OLS all 38 epochs + intercept",
)

print(
    "LOS positive                    :",
    "toward satellite",
)

print()

print(
    "velocity parity max             :",
    f"{vel_parity:.12e} mm/yr",
)

print(
    "cumulative parity max           :",
    f"{cum_parity:.12e} mm",
)

print()

print(
    "velocity p01/05/50/95/99        :",
    vel_q,
)

print(
    "|velocity| p50/p95/p99          :",
    abs_vel_q,
)

print(
    "cumulative p01/05/50/95/99      :",
    cum_q,
)

print(
    "linear residual RMS p50/95/p99  :",
    rms_q,
)

print(
    "velocity SE p50/p95/p99         :",
    se_q,
)

print()

print(
    "reference velocity median       :",
    (
        f"{reference_velocity_median:.12e} "
        "mm/yr"
    ),
)

print(
    "reference cumulative median     :",
    (
        f"{reference_cumulative_median:.12e} "
        "mm"
    ),
)

print()

print(
    "production seconds              :",
    f"{seconds:.6f}",
)

print(
    "throughput                      :",
    f"{npoint/seconds:,.0f} points/s",
)

print()

print(
    "velocity output                 :",
    VEL,
)

print(
    "cumulative output               :",
    CUM,
)

print(
    "residual RMS output             :",
    RMS,
)

print(
    "velocity SE output              :",
    VEL_SE,
)

print(
    "upstream phase modified         :",
    False,
)

print(
    "manifest                        :",
    MANIFEST,
)

print("=" * 96)

print(
    "P15-7A FINAL RESULT: "
    "PASS_FINAL_LOS_POINT_PRODUCTS"
)

print("=" * 96)
