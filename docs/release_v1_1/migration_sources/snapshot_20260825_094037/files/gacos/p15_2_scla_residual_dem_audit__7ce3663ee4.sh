#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/ubuntu/Downloads/psds
OUT=$PROJECT/output
PROC=$OUT/processing
NET=$PROJECT/prototype_outputs/v09/network
LOGDIR=$PROJECT/production_logs

STAMP=$(date +%Y%m%d_%H%M%S)

JSON_REPORT=$LOGDIR/P15_2_scla_residual_dem_audit_${STAMP}.json
TXT_REPORT=$LOGDIR/P15_2_scla_residual_dem_audit_${STAMP}.txt

mkdir -p "$LOGDIR"

echo "================================================================================================"
echo " P15-2 SCLA / RESIDUAL DEM IDENTIFIABILITY AUDIT"
echo
echo " READ ONLY: production scientific products"
echo " NO SOURCE MODIFICATION"
echo " NO PRODUCTION PRODUCT MODIFICATION"
echo " NO GAMMA EXECUTION"
echo " NO SCLA CORRECTION"
echo " NO ATMOSPHERIC CORRECTION"
echo "================================================================================================"

python - \
    "$PROJECT" \
    "$OUT" \
    "$NET" \
    "$JSON_REPORT" \
    "$TXT_REPORT" <<'PY'

from pathlib import Path
import json
import math
import re
import sys

import numpy as np


PROJECT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
NET = Path(sys.argv[3]).resolve()

JSON_REPORT = Path(sys.argv[4]).resolve()
TXT_REPORT = Path(sys.argv[5]).resolve()

PROC = OUT / "processing"

errors = []
warnings = []
checks = []


def check(name, ok, detail=""):

    ok = bool(ok)

    checks.append({
        "name": name,
        "pass": ok,
        "detail": str(detail),
    })

    print(
        f"{'PASS' if ok else 'FAIL':4s}  "
        f"{name}"
        +
        (
            f"  [{detail}]"
            if detail
            else ""
        )
    )

    if not ok:
        errors.append(
            f"{name}: {detail}"
        )

    return ok


def percentile_summary(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return None

    q = np.percentile(
        x,
        [
            1,
            5,
            25,
            50,
            75,
            95,
            99,
        ],
    )

    return {
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
    }


def parse_yyyymmdd(s):

    s = str(s)

    return np.datetime64(
        f"{s[0:4]}-{s[4:6]}-{s[6:8]}",
        "D",
    )


print()
print("=" * 96)
print("A. ACCEPTED INPUTS")
print("=" * 96)


# ============================================================================
# P15-1 sign report
# ============================================================================

p15_1_reports = sorted(
    (
        PROJECT
        / "production_logs"
    ).glob(
        "P15_1_los_sign_convention_*.json"
    )
)

check(
    "P15-1 report found",
    len(p15_1_reports) > 0,
    len(p15_1_reports),
)

if not p15_1_reports:
    raise SystemExit(1)

p15_1_path = p15_1_reports[-1]

p15_1 = json.loads(
    p15_1_path.read_text(
        encoding="utf-8"
    )
)

check(
    "P15-1 accepted",
    p15_1.get("status")
    ==
    "PASS_SIGN_CONVENTION_FROZEN",
    p15_1.get("status"),
)

wavelength = float(
    p15_1[
        "sensor"
    ][
        "wavelength_m"
    ]
)

check(
    "wavelength valid",
    0.05 < wavelength < 0.06,
    wavelength,
)


# ============================================================================
# Referenced phase
# ============================================================================

phase_path = (
    PROC
    / "referenced_timeseries"
    / "acquisition_phase_referenced_rad.npy"
)

dates_path = (
    PROC
    / "network_inversion"
    / "dates.txt"
)

strict_ids_path = (
    PROC
    / "network_inversion"
    / "strict_point_ids.npy"
)

rows_path = (
    PROC
    / "point_phase_stack"
    / "rows.npy"
)

cols_path = (
    PROC
    / "point_phase_stack"
    / "cols.npy"
)


for p in (
    phase_path,
    dates_path,
    strict_ids_path,
    rows_path,
    cols_path,
):
    check(
        f"input exists: {p.name}",
        p.is_file(),
        p,
    )


phase = np.load(
    phase_path,
    mmap_mode="r",
)

strict_ids = np.load(
    strict_ids_path,
    mmap_mode="r",
)

all_rows = np.load(
    rows_path,
    mmap_mode="r",
)

all_cols = np.load(
    cols_path,
    mmap_mode="r",
)

dates = [
    x.strip()
    for x in dates_path.read_text().splitlines()
    if x.strip()
]


check(
    "phase shape",
    phase.shape
    ==
    (
        881315,
        38,
    ),
    phase.shape,
)

check(
    "dates",
    len(dates) == 38,
    len(dates),
)


# ============================================================================
# B. Acquisition Bperp
# ============================================================================

print()
print("=" * 96)
print("B. ACQUISITION PERPENDICULAR BASELINE")
print("=" * 96)


bperp_path = (
    NET
    / "acquisition_bperp_m.npy"
)

check(
    "acquisition_bperp_m.npy exists",
    bperp_path.is_file(),
    bperp_path,
)

if not bperp_path.is_file():
    raise SystemExit(1)


bperp = np.asarray(
    np.load(
        bperp_path
    ),
    dtype=np.float64,
).reshape(-1)


check(
    "Bperp length",
    bperp.size == 38,
    bperp.size,
)

check(
    "Bperp finite",
    np.all(
        np.isfinite(
            bperp
        )
    ),
)

print()
print(
    f"Bperp min/max             : "
    f"{bperp.min():.4f} / "
    f"{bperp.max():.4f} m"
)

print(
    f"Bperp span                : "
    f"{np.ptp(bperp):.4f} m"
)

print(
    f"Bperp median              : "
    f"{np.median(bperp):.4f} m"
)


# ============================================================================
# C. Verify against GAMMA base_calc stdout
# ============================================================================

print()
print("=" * 96)
print("C. GAMMA BASE_CALC PARITY")
print("=" * 96)


base_stdout = (
    NET
    / "gamma_base_calc"
    / "base_calc_stdout.log"
)

check(
    "base_calc_stdout.log exists",
    base_stdout.is_file(),
    base_stdout,
)


parsed = {}

if base_stdout.is_file():

    pat = re.compile(
        r"ref\.\:\s*(\d{8})\s+"
        r"(\d{8})\s+"
        r"Bperp:\s*"
        r"([-+0-9.Ee]+)"
    )

    for line in base_stdout.read_text(
        errors="ignore"
    ).splitlines():

        m = pat.search(
            line
        )

        if m:

            refdate = m.group(1)
            secdate = m.group(2)
            bp = float(
                m.group(3)
            )

            parsed[
                secdate
            ] = bp


print(
    "parsed GAMMA acquisition Bperp:",
    len(parsed),
)


# Geometric master is not listed as secondary in the 37 reference-to-all pairs.
geom_ref = "20151212"

check(
    "geometric reference date present in stack",
    geom_ref in dates,
    geom_ref,
)

geom_idx = dates.index(
    geom_ref
)

print(
    "geometric reference index :",
    geom_idx,
)


check(
    "geometric-master Bperp approximately zero",
    abs(
        bperp[
            geom_idx
        ]
    )
    <
    1.0e-6,
    f"{bperp[geom_idx]:.9f} m",
)


gamma_diffs = []

missing_gamma = []

for i, d in enumerate(
    dates
):

    if d == geom_ref:
        continue

    if d not in parsed:
        missing_gamma.append(
            d
        )

        continue

    gamma_diffs.append(
        bperp[i]
        -
        parsed[d]
    )


check(
    "all 37 secondary GAMMA Bperp parsed",
    len(
        gamma_diffs
    )
    ==
    37,
    (
        f"matched={len(gamma_diffs)}, "
        f"missing={missing_gamma}"
    ),
)


if gamma_diffs:

    gamma_diffs = np.asarray(
        gamma_diffs,
        dtype=np.float64,
    )

    max_bp_diff = float(
        np.max(
            np.abs(
                gamma_diffs
            )
        )
    )

else:

    max_bp_diff = float(
        "nan"
    )


check(
    "acquisition_bperp / GAMMA parity",
    np.isfinite(
        max_bp_diff
    )
    and
    max_bp_diff < 1.0e-3,
    f"max diff={max_bp_diff:.6e} m",
)


# ============================================================================
# D. Correct temporal-reference baseline
# ============================================================================

print()
print("=" * 96)
print("D. TEMPORAL-REFERENCE BASELINE")
print("=" * 96)


# Network inversion / PointPhaseStack uses acquisition 0 as temporal phase zero.
temporal_ref_idx = 0
temporal_ref_date = dates[
    temporal_ref_idx
]


check(
    "temporal reference date",
    temporal_ref_date == "20141006",
    temporal_ref_date,
)


# The residual-topography term in a phase history referenced to acquisition 0
# must use baseline DIFFERENCE relative to acquisition 0.
b_rel = (
    bperp
    -
    bperp[
        temporal_ref_idx
    ]
)


check(
    "relative Bperp first epoch zero",
    abs(
        b_rel[0]
    )
    <
    1.0e-12,
    b_rel[0],
)


print(
    f"Bperp temporal reference  : "
    f"{bperp[0]:.6f} m"
)

print(
    f"relative Bperp min/max    : "
    f"{b_rel.min():.4f} / "
    f"{b_rel.max():.4f} m"
)

print(
    f"relative Bperp span       : "
    f"{np.ptp(b_rel):.4f} m"
)


# ============================================================================
# E. Time/Bperp identifiability
# ============================================================================

print()
print("=" * 96)
print("E. TIME / BPERP IDENTIFIABILITY")
print("=" * 96)


dates64 = np.asarray(
    [
        parse_yyyymmdd(
            x
        )
        for x in dates
    ],
    dtype="datetime64[D]",
)


t_days = (
    dates64
    -
    dates64[0]
).astype(
    "timedelta64[D]"
).astype(
    np.float64
)


t_year = (
    t_days
    /
    365.25
)


corr_tb = float(
    np.corrcoef(
        t_year,
        b_rel,
    )[0, 1]
)


print(
    f"corr(time, Bperp)         : "
    f"{corr_tb:+.6f}"
)


if abs(
    corr_tb
) >= 0.90:

    warnings.append(
        "Bperp is very strongly correlated with time; "
        "linear deformation and SCLA are poorly separable."
    )

elif abs(
    corr_tb
) >= 0.75:

    warnings.append(
        "Bperp/time correlation is moderately high; "
        "SCLA estimation requires conservative validation."
    )


# Build normalized design to assess conditioning, not physical coefficient.
tc = (
    t_year
    -
    np.mean(
        t_year
    )
)

bc = (
    b_rel
    -
    np.mean(
        b_rel
    )
)


t_std = float(
    np.std(
        tc
    )
)

b_std = float(
    np.std(
        bc
    )
)


check(
    "time variance nonzero",
    t_std > 0,
    t_std,
)

check(
    "Bperp variance nonzero",
    b_std > 1.0,
    b_std,
)


X_norm = np.column_stack([
    np.ones(
        len(dates)
    ),
    tc / t_std,
    bc / b_std,
])


cond_norm = float(
    np.linalg.cond(
        X_norm
    )
)


print(
    f"normalized [1,t,B] cond  : "
    f"{cond_norm:.4f}"
)


check(
    "time/Bperp design condition",
    cond_norm < 10.0,
    cond_norm,
)


# ============================================================================
# F. Full-scene baseline-correlated phase audit
#
# Model 0:
#   phi(t) = a + v*t
#
# Model 1:
#   phi(t) = a + v*t + beta * Bperp_rel(t)
#
# beta has units rad / meter-baseline.
#
# This is an IDENTIFIABILITY TEST ONLY.
# It must NOT yet be interpreted as a final SCLA estimate because:
#   - GACOS is not applied,
#   - nonlinear deformation may remain,
#   - no temporal filtering has yet been applied.
# ============================================================================

print()
print("=" * 96)
print("F. BASELINE-CORRELATED PHASE SIGNATURE")
print("=" * 96)


X0 = np.column_stack([
    np.ones(
        len(dates)
    ),
    t_year,
])


X1 = np.column_stack([
    np.ones(
        len(dates)
    ),
    t_year,
    b_rel,
])


P0 = np.linalg.pinv(
    X0
)

P1 = np.linalg.pinv(
    X1
)


npoint = phase.shape[0]

BATCH = 32768


beta_all = np.empty(
    npoint,
    dtype=np.float32,
)

rate0_all = np.empty(
    npoint,
    dtype=np.float32,
)

rate1_all = np.empty(
    npoint,
    dtype=np.float32,
)

rms0_all = np.empty(
    npoint,
    dtype=np.float32,
)

rms1_all = np.empty(
    npoint,
    dtype=np.float32,
)


for b0 in range(
    0,
    npoint,
    BATCH,
):

    b1 = min(
        b0 + BATCH,
        npoint,
    )

    Y = np.asarray(
        phase[
            b0:b1,
            :
        ],
        dtype=np.float64,
    )


    coef0 = (
        Y
        @
        P0.T
    )


    coef1 = (
        Y
        @
        P1.T
    )


    fit0 = (
        coef0
        @
        X0.T
    )


    fit1 = (
        coef1
        @
        X1.T
    )


    e0 = (
        Y
        -
        fit0
    )


    e1 = (
        Y
        -
        fit1
    )


    rms0 = np.sqrt(
        np.mean(
            e0 * e0,
            axis=1,
        )
    )


    rms1 = np.sqrt(
        np.mean(
            e1 * e1,
            axis=1,
        )
    )


    rate0_all[
        b0:b1
    ] = coef0[
        :,
        1
    ].astype(
        np.float32
    )


    rate1_all[
        b0:b1
    ] = coef1[
        :,
        1
    ].astype(
        np.float32
    )


    beta_all[
        b0:b1
    ] = coef1[
        :,
        2
    ].astype(
        np.float32
    )


    rms0_all[
        b0:b1
    ] = rms0.astype(
        np.float32
    )


    rms1_all[
        b0:b1
    ] = rms1.astype(
        np.float32
    )


    if (
        b0 == 0
        or
        b1 == npoint
        or
        (
            b1
            //
            BATCH
        )
        %
        5
        ==
        0
    ):

        print(
            f"  {b1:,}/"
            f"{npoint:,}"
        )


beta = np.asarray(
    beta_all,
    dtype=np.float64,
)

rms0 = np.asarray(
    rms0_all,
    dtype=np.float64,
)

rms1 = np.asarray(
    rms1_all,
    dtype=np.float64,
)

rate0 = np.asarray(
    rate0_all,
    dtype=np.float64,
)

rate1 = np.asarray(
    rate1_all,
    dtype=np.float64,
)


improvement = (
    rms0
    -
    rms1
)


improvement_fraction = np.divide(
    improvement,
    rms0,
    out=np.zeros_like(
        improvement
    ),
    where=(
        rms0 > 0
    ),
)


rate_change = (
    rate1
    -
    rate0
)


print()
print(
    "beta [rad/m-baseline] distribution:"
)

beta_q = percentile_summary(
    beta
)

print(
    json.dumps(
        beta_q,
        indent=2,
    )
)


print()
print(
    "RMS improvement fraction distribution:"
)

improve_q = percentile_summary(
    improvement_fraction
)

print(
    json.dumps(
        improve_q,
        indent=2,
    )
)


print()
print(
    "linear-rate change after adding Bperp [rad/yr]:"
)

rate_change_q = percentile_summary(
    rate_change
)

print(
    json.dumps(
        rate_change_q,
        indent=2,
    )
)


fraction_improved_01 = float(
    np.mean(
        improvement_fraction
        >
        0.01
    )
)

fraction_improved_05 = float(
    np.mean(
        improvement_fraction
        >
        0.05
    )
)

fraction_improved_10 = float(
    np.mean(
        improvement_fraction
        >
        0.10
    )
)


print()
print(
    f"points RMS improvement >1% : "
    f"{100*fraction_improved_01:.3f}%"
)

print(
    f"points RMS improvement >5% : "
    f"{100*fraction_improved_05:.3f}%"
)

print(
    f"points RMS improvement >10%: "
    f"{100*fraction_improved_10:.3f}%"
)


# ============================================================================
# G. Approximate equivalent DEM-error magnitude
#
# Linearized magnitude:
#
# |phi_topo| ~= (4*pi/lambda)
#                * |Bperp|/(R*sin(theta))
#                * |dh|
#
# Therefore:
#
# |dh| ~= |beta| * lambda * R*sin(theta)/(4*pi)
#
# Sign is intentionally NOT frozen here.
# ============================================================================

print()
print("=" * 96)
print("G. APPROXIMATE RESIDUAL DEM MAGNITUDE")
print("=" * 96)


center_range_m = 839056.5760
incidence_deg = 33.9481

theta = math.radians(
    incidence_deg
)


height_scale = (
    wavelength
    *
    center_range_m
    *
    math.sin(
        theta
    )
    /
    (
        4.0
        *
        math.pi
    )
)


height_abs = (
    np.abs(
        beta
    )
    *
    height_scale
)


height_q = percentile_summary(
    height_abs
)


print(
    f"center slant range        : "
    f"{center_range_m:.3f} m"
)

print(
    f"incidence angle           : "
    f"{incidence_deg:.4f} deg"
)

print(
    f"|dh| scale               : "
    f"{height_scale:.6f} "
    f"m per (rad/m-baseline)"
)

print()
print(
    "approx |residual DEM| distribution [m]:"
)

print(
    json.dumps(
        height_q,
        indent=2,
    )
)


# ============================================================================
# H. Spatial smoothness diagnostic
#
# Real residual DEM/SCLA should be spatially structured rather than random.
#
# Compare beta differences for nearest radar-coordinate horizontal/vertical
# neighbours against differences from random point pairs.
# ============================================================================

print()
print("=" * 96)
print("H. SPATIAL STRUCTURE OF BASELINE COEFFICIENT")
print("=" * 96)


strict_rows = np.asarray(
    all_rows[
        np.asarray(
            strict_ids,
            dtype=np.int64,
        )
    ],
    dtype=np.int32,
)

strict_cols = np.asarray(
    all_cols[
        np.asarray(
            strict_ids,
            dtype=np.int64,
        )
    ],
    dtype=np.int32,
)


grid = np.full(
    (
        600,
        2000,
    ),
    -1,
    dtype=np.int32,
)


grid[
    strict_rows,
    strict_cols
] = np.arange(
    npoint,
    dtype=np.int32,
)


neighbor_diffs = []


# right neighbour
valid = (
    strict_cols
    <
    1999
)

ids = np.where(
    valid
)[0]

nid = grid[
    strict_rows[
        ids
    ],
    strict_cols[
        ids
    ]
    +
    1
]

good = (
    nid >= 0
)

if np.any(
    good
):

    a = ids[
        good
    ]

    b = nid[
        good
    ]

    neighbor_diffs.append(
        np.abs(
            beta[a]
            -
            beta[b]
        )
    )


# down neighbour
valid = (
    strict_rows
    <
    599
)

ids = np.where(
    valid
)[0]

nid = grid[
    strict_rows[
        ids
    ]
    +
    1,
    strict_cols[
        ids
    ],
]

good = (
    nid >= 0
)

if np.any(
    good
):

    a = ids[
        good
    ]

    b = nid[
        good
    ]

    neighbor_diffs.append(
        np.abs(
            beta[a]
            -
            beta[b]
        )
    )


if neighbor_diffs:

    neighbor_diff = np.concatenate(
        neighbor_diffs
    )

else:

    neighbor_diff = np.empty(
        0,
        dtype=np.float64,
    )


rng = np.random.default_rng(
    20260824
)

nr = min(
    max(
        neighbor_diff.size,
        100000,
    ),
    1000000,
)

ra = rng.integers(
    0,
    npoint,
    size=nr,
)

rb = rng.integers(
    0,
    npoint,
    size=nr,
)

random_diff = np.abs(
    beta[
        ra
    ]
    -
    beta[
        rb
    ]
)


neighbor_median = float(
    np.median(
        neighbor_diff
    )
) if neighbor_diff.size else float(
    "nan"
)

random_median = float(
    np.median(
        random_diff
    )
)


smooth_ratio = (
    neighbor_median
    /
    random_median
    if (
        np.isfinite(
            neighbor_median
        )
        and
        random_median > 0
    )
    else float(
        "nan"
    )
)


print(
    f"adjacent beta pairs       : "
    f"{neighbor_diff.size:,}"
)

print(
    f"adjacent |dbeta| median   : "
    f"{neighbor_median:.6e}"
)

print(
    f"random   |dbeta| median   : "
    f"{random_median:.6e}"
)

print(
    f"spatial smoothness ratio  : "
    f"{smooth_ratio:.6f}"
)


if np.isfinite(
    smooth_ratio
):

    check(
        "baseline coefficient spatial structure",
        smooth_ratio < 0.75,
        (
            f"neighbor/random median "
            f"ratio={smooth_ratio:.4f}"
        ),
    )


# ============================================================================
# I. Reference region differential nature
# ============================================================================

print()
print("=" * 96)
print("I. REFERENCE-REGION EFFECT")
print("=" * 96)


ref_indices_path = (
    PROC
    / "referenced_timeseries"
    / "reference_strict_indices.npy"
)

ref_indices = np.asarray(
    np.load(
        ref_indices_path
    ),
    dtype=np.int64,
)


ref_beta = beta[
    ref_indices
]


ref_beta_median = float(
    np.median(
        ref_beta
    )
)


global_beta_median = float(
    np.median(
        beta
    )
)


print(
    f"reference beta median     : "
    f"{ref_beta_median:+.6e} rad/m"
)

print(
    f"global beta median        : "
    f"{global_beta_median:+.6e} rad/m"
)


# Because the phase was referenced epoch-by-epoch using the reference-region
# median, the recovered beta is a DIFFERENTIAL SCLA coefficient.
check(
    "reference beta finite",
    np.isfinite(
        ref_beta_median
    ),
    ref_beta_median,
)


# ============================================================================
# J. Conservative classification
# ============================================================================

print()
print("=" * 96)
print("J. SCLA IDENTIFIABILITY CLASSIFICATION")
print("=" * 96)


baseline_verified = (
    len(
        gamma_diffs
    )
    ==
    37
    and
    max_bp_diff
    <
    1.0e-3
)


design_good = (
    cond_norm
    <
    10.0
    and
    abs(
        corr_tb
    )
    <
    0.90
)


spatial_signal = (
    np.isfinite(
        smooth_ratio
    )
    and
    smooth_ratio
    <
    0.75
)


# We deliberately do NOT require large residual-RMS improvement.
# A physically real SCLA can be small after accurate DEM correction.
identifiable = (
    baseline_verified
    and
    design_good
)


if not identifiable:

    status = (
        "REVIEW_SCLA_IDENTIFIABILITY"
    )

    next_step = (
        "STOP_AND_REVIEW_BASELINE_TIME_CONFOUNDING"
    )

else:

    status = (
        "PASS_SCLA_INPUT_IDENTIFIABILITY"
    )

    # GACOS is complete 38/38 from P15-0.
    #
    # Before estimating/removing beta as SCLA, atmospheric delay
    # should be audited at the exact point/date geometry so that
    # atmosphere is not accidentally absorbed into the Bperp term.
    next_step = (
        "P15-3_GACOS_GEOMETRY_SIGN_UNIT_AUDIT"
    )


print(
    f"baseline verified         : "
    f"{baseline_verified}"
)

print(
    f"time/Bperp identifiable  : "
    f"{design_good}"
)

print(
    f"spatial beta structure    : "
    f"{spatial_signal}"
)

print(
    f"status                    : "
    f"{status}"
)

print(
    f"next step                 : "
    f"{next_step}"
)


report = {
    "format":
        "pyPSDS-GAMMA-P15-2-SCLA-identifiability-v1",

    "status":
        status,

    "P15_1_report":
        str(
            p15_1_path
        ),

    "inputs": {
        "phase":
            str(
                phase_path
            ),

        "bperp":
            str(
                bperp_path
            ),

        "dates":
            dates,

        "strict_points":
            int(
                npoint
            ),
    },

    "baseline": {
        "geometric_reference_date":
            geom_ref,

        "geometric_reference_index":
            int(
                geom_idx
            ),

        "temporal_reference_date":
            temporal_ref_date,

        "temporal_reference_index":
            temporal_ref_idx,

        "geometric_reference_bperp_m":
            float(
                bperp[
                    geom_idx
                ]
            ),

        "temporal_reference_bperp_m":
            float(
                bperp[
                    temporal_ref_idx
                ]
            ),

        "relative_baseline_min_m":
            float(
                b_rel.min()
            ),

        "relative_baseline_max_m":
            float(
                b_rel.max()
            ),

        "relative_baseline_span_m":
            float(
                np.ptp(
                    b_rel
                )
            ),

        "gamma_parity_max_abs_m":
            max_bp_diff,
    },

    "identifiability": {
        "corr_time_bperp":
            corr_tb,

        "normalized_design_condition":
            cond_norm,

        "baseline_verified":
            baseline_verified,

        "design_good":
            design_good,
    },

    "simple_regression_diagnostic": {
        "model_without_SCLA":
            "phi = a + v*t",

        "model_with_SCLA":
            "phi = a + v*t + beta*Bperp_relative",

        "beta_rad_per_m_baseline":
            beta_q,

        "rms_improvement_fraction":
            improve_q,

        "fraction_RMS_improvement_gt_1pct":
            fraction_improved_01,

        "fraction_RMS_improvement_gt_5pct":
            fraction_improved_05,

        "fraction_RMS_improvement_gt_10pct":
            fraction_improved_10,

        "linear_rate_change_rad_per_year":
            rate_change_q,

        "warning":
            (
                "Diagnostic only. Atmospheric delay and nonlinear "
                "deformation have not been removed."
            ),
    },

    "approximate_residual_DEM_magnitude": {
        "formula_magnitude":
            (
                "|dh| ~= |beta| * lambda * "
                "R * sin(theta)/(4*pi)"
            ),

        "sign_interpretation":
            "not_frozen",

        "center_slant_range_m":
            center_range_m,

        "incidence_angle_deg":
            incidence_deg,

        "scale_m_per_rad_per_mbaseline":
            height_scale,

        "abs_height_m":
            height_q,
    },

    "spatial_structure": {
        "adjacent_pairs":
            int(
                neighbor_diff.size
            ),

        "adjacent_abs_dbeta_median":
            neighbor_median,

        "random_abs_dbeta_median":
            random_median,

        "neighbor_random_ratio":
            smooth_ratio,

        "spatial_structure_detected":
            spatial_signal,
    },

    "reference_region": {
        "reference_points":
            int(
                ref_indices.size
            ),

        "reference_beta_median_rad_per_m":
            ref_beta_median,

        "global_beta_median_rad_per_m":
            global_beta_median,

        "interpretation":
            (
                "beta is differential relative to the "
                "computational reference region"
            ),
    },

    "next_step":
        next_step,

    "checks":
        checks,

    "warnings":
        warnings,

    "errors":
        errors,
}


JSON_REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
    +
    "\n",
    encoding="utf-8",
)


lines = [
    "=" * 88,
    "P15-2 SCLA / RESIDUAL DEM IDENTIFIABILITY",
    "=" * 88,

    f"status                    : {status}",

    f"points                    : {npoint:,}",

    f"acquisitions              : {len(dates)}",

    "",

    f"geometric ref             : {geom_ref} / index {geom_idx}",

    f"temporal phase ref        : {temporal_ref_date} / index 0",

    (
        "Bperp GAMMA max diff    : "
        f"{max_bp_diff:.6e} m"
    ),

    (
        "relative Bperp span     : "
        f"{np.ptp(b_rel):.3f} m"
    ),

    (
        "corr(time,Bperp)        : "
        f"{corr_tb:+.6f}"
    ),

    (
        "design condition        : "
        f"{cond_norm:.4f}"
    ),

    "",

    (
        "beta p50                : "
        f"{beta_q['p50']:+.6e} rad/m"
    ),

    (
        "RMS improve >5%         : "
        f"{100*fraction_improved_05:.3f}%"
    ),

    (
        "approx |DEM error| p50  : "
        f"{height_q['p50']:.3f} m"
    ),

    (
        "approx |DEM error| p95  : "
        f"{height_q['p95']:.3f} m"
    ),

    (
        "beta spatial ratio      : "
        f"{smooth_ratio:.4f}"
    ),

    "",

    f"next step                 : {next_step}",

    "",

    f"JSON report               : {JSON_REPORT}",
]


TXT_REPORT.write_text(
    "\n".join(
        lines
    )
    +
    "\n",
    encoding="utf-8",
)


print()
print(
    "\n".join(
        lines
    )
)


if errors:

    print()
    print("=" * 96)
    print(
        " P15-2 FINAL RESULT: FAIL_INPUT_CONTRACT"
    )
    print("=" * 96)

    raise SystemExit(1)


print()
print("=" * 96)

if identifiable:

    print(
        " P15-2 FINAL RESULT: "
        "PASS_SCLA_INPUT_IDENTIFIABILITY"
    )

else:

    print(
        " P15-2 FINAL RESULT: "
        "REVIEW_SCLA_IDENTIFIABILITY"
    )

print("=" * 96)

PY

echo
echo "reports:"
echo "  $JSON_REPORT"
echo "  $TXT_REPORT"
