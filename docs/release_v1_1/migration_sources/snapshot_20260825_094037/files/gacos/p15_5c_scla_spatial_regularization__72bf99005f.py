from pathlib import Path
import csv
import json
import math
import re
import time

import numpy as np
from scipy.ndimage import uniform_filter


PSDS = Path("/home/ubuntu/Downloads/psds")
PROC = PSDS / "output/processing"

PHASE = (
    PROC
    / "gacos_corrected_phase"
    / "acquisition_phase_gacos_corrected_rad.npy"
)

GMAN = (
    PROC
    / "gacos_corrected_phase"
    / "gacos_correction_manifest.json"
)

SCLA = (
    PROC
    / "scla_residual_dem_estimation"
)

BETA_RAW = (
    SCLA
    / "scla_beta_rad_per_m_bperp.npy"
)

SIGMA_RAW = (
    SCLA
    / "scla_beta_sigma_rad_per_m_bperp.npy"
)

R2_RAW = (
    SCLA
    / "scla_partial_r2.npy"
)

BPERP = (
    PROC
    / "network"
    / "acquisition_bperp_m.npy"
)

GEOM = (
    PROC
    / "gacos_geometry"
)

REF_IDX_FILE = (
    PROC
    / "gacos_fast_smoke"
    / "project_local_interp"
    / "ref_idx.npy"
)

RSLC_PAR = Path(
    "/home/ubuntu/Downloads/RSLC/20151212.rslc.par"
)

OUT = (
    PROC
    / "scla_spatial_regularization"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BETA_OUT = (
    OUT
    / "scla_beta_regularized_rad_per_m_bperp.npy"
)

CSV_OUT = (
    OUT
    / "regularization_scale_audit.csv"
)

MANIFEST = (
    OUT
    / "scla_regularization_manifest.json"
)


RADII_M = [
    0.0,
    25.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
    300.0,
]

MIN_STABILITY_CORR = 0.40
PLATEAU_FRACTION = 0.98
MIN_SCALE_RETENTION = 0.50


# ============================================================
# Helpers
# ============================================================

NUM = re.compile(
    r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
)


def par_scalar(path, key):

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(":", 1)

        if k.strip().lower() == key.lower():

            m = NUM.search(v)

            if m:
                return float(m.group(0))

    raise KeyError(key)


def robust_scale(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    med = np.median(x)

    mad = np.median(
        np.abs(
            x - med
        )
    )

    return (
        mad / 0.6744897501960817
    )


def centered_corr(a, b):

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    a = a - np.median(a)
    b = b - np.median(b)

    sa = np.sqrt(
        np.dot(a, a)
    )

    sb = np.sqrt(
        np.dot(b, b)
    )

    if sa == 0.0 or sb == 0.0:
        return np.nan

    return float(
        np.dot(a, b)
        /
        (sa * sb)
    )


# ============================================================
# Inputs
# ============================================================

for p in (
    PHASE,
    GMAN,
    BETA_RAW,
    SIGMA_RAW,
    R2_RAW,
    BPERP,
    GEOM / "strict_points.plist",
    REF_IDX_FILE,
    RSLC_PAR,
):

    if not p.is_file():
        raise FileNotFoundError(p)


gman = json.loads(
    GMAN.read_text()
)

dates = list(
    gman["acquisition_dates"]
)

if len(dates) != 38:
    raise RuntimeError(
        f"expected 38 dates, got {len(dates)}"
    )


phase = np.load(
    PHASE,
    mmap_mode="r",
)

beta_full = np.load(
    BETA_RAW,
    mmap_mode="r",
)

sigma_full = np.load(
    SIGMA_RAW,
    mmap_mode="r",
)

r2_full = np.load(
    R2_RAW,
    mmap_mode="r",
)

b = np.load(
    BPERP
).astype(
    np.float64
).reshape(-1)

ref_idx = np.load(
    REF_IDX_FILE
).astype(
    np.int64
)


npoint, ndate = phase.shape


if not (
    beta_full.size
    == sigma_full.size
    == r2_full.size
    == npoint
):

    raise RuntimeError(
        "SCLA array size mismatch"
    )


if b.size != ndate:
    raise RuntimeError(
        "Bperp size mismatch"
    )


# ============================================================
# Radar grid
# ============================================================

plist = np.fromfile(
    GEOM / "strict_points.plist",
    dtype=">i4",
).reshape(
    -1,
    2,
)

if plist.shape[0] != npoint:
    raise RuntimeError(
        "plist size mismatch"
    )


col = plist[:, 0].astype(
    np.int64
)

row = plist[:, 1].astype(
    np.int64
)

H = int(
    row.max()
) + 1

W = int(
    col.max()
) + 1


# ============================================================
# Physical pixel spacing
# ============================================================

range_spacing = par_scalar(
    RSLC_PAR,
    "range_pixel_spacing",
)

try:
    incidence_deg = par_scalar(
        RSLC_PAR,
        "incidence_angle",
    )

except KeyError:

    inc = np.load(
        GEOM
        / "incidence_gamma_compatible_fast_rad.npy",
        mmap_mode="r",
    )

    incidence_deg = float(
        np.degrees(
            np.median(inc)
        )
    )


ground_range_spacing = (
    range_spacing
    /
    np.sin(
        np.deg2rad(
            incidence_deg
        )
    )
)


try:
    az_spacing = par_scalar(
        RSLC_PAR,
        "azimuth_pixel_spacing",
    )

except KeyError:

    # Current Sentinel-1 geometry fallback.
    az_spacing = 13.9566


print(
    "radar grid                :",
    f"{H} x {W}",
)

print(
    "ground range spacing m    :",
    f"{ground_range_spacing:.6f}",
)

print(
    "azimuth spacing m         :",
    f"{az_spacing:.6f}",
)


# ============================================================
# FWL fold estimator
# ============================================================

t_days = np.array(
    [
        (
            np.datetime64(
                d[:4]
                + "-"
                + d[4:6]
                + "-"
                + d[6:8]
            )
            -
            np.datetime64(
                dates[0][:4]
                + "-"
                + dates[0][4:6]
                + "-"
                + dates[0][6:8]
            )
        )
        /
        np.timedelta64(
            1,
            "D",
        )

        for d in dates
    ],
    dtype=np.float64,
)


ty = (
    t_days
    /
    365.2425
)

omega = (
    2.0
    *
    np.pi
)

X0 = np.column_stack(
    (
        ty,
        np.sin(
            omega * ty
        ),
        np.cos(
            omega * ty
        ) - 1.0,
    )
)


brel = (
    b - b[0]
)


fold_a = np.arange(
    0,
    ndate,
    2,
    dtype=np.int64,
)

fold_b = np.arange(
    1,
    ndate,
    2,
    dtype=np.int64,
)


def fold_beta(indices):

    X = X0[
        indices,
        :,
    ]

    bb = brel[
        indices
    ]


    rank = int(
        np.linalg.matrix_rank(
            np.column_stack(
                (
                    X,
                    bb,
                )
            )
        )
    )

    if rank != 4:
        raise RuntimeError(
            (
                "fold rank deficient: "
                f"{rank}/4"
            )
        )


    Q, _ = np.linalg.qr(
        X,
        mode="reduced",
    )


    br = (
        bb
        -
        Q
        @
        (
            Q.T @ bb
        )
    )


    den = float(
        np.dot(
            br,
            br,
        )
    )


    if den <= 0:
        raise RuntimeError(
            "invalid fold Bperp denominator"
        )


    result = np.empty(
        npoint,
        dtype=np.float32,
    )


    chunk = 262144


    for s in range(
        0,
        npoint,
        chunk,
    ):

        e = min(
            npoint,
            s + chunk,
        )


        y = np.asarray(
            phase[
                s:e,
                :
            ][
                :,
                indices
            ],
            dtype=np.float32,
        )


        result[
            s:e
        ] = (
            y
            @
            br
            /
            den
        ).astype(
            np.float32
        )


    return result


t0 = time.perf_counter()

beta_a = fold_beta(
    fold_a
)

beta_b = fold_beta(
    fold_b
)

fold_seconds = (
    time.perf_counter()
    -
    t0
)


print(
    "fold estimation seconds   :",
    f"{fold_seconds:.6f}",
)


# ============================================================
# Sparse normalized box smoothing
# ============================================================

valid = (
    np.isfinite(
        beta_full
    )
    &
    np.isfinite(
        sigma_full
    )
)


if not np.all(valid):

    raise RuntimeError(
        "non-finite beta/sigma"
    )


# Standard inverse-variance reliability.
sig = np.asarray(
    sigma_full,
    dtype=np.float64,
)


positive_sig = sig[
    sig > 0
]


sigma_floor = float(
    np.percentile(
        positive_sig,
        10,
    )
)


weight_full = (
    1.0
    /
    (
        sig*sig
        +
        sigma_floor*sigma_floor
    )
)


# Prevent a very small number of points dominating.
w_lo, w_hi = np.percentile(
    weight_full,
    [
        2,
        98,
    ],
)


weight_full = np.clip(
    weight_full,
    w_lo,
    w_hi,
)


ones = np.ones(
    npoint,
    dtype=np.float64,
)


def smooth_points(
    values,
    weights,
    radius_m,
):

    if radius_m <= 0.0:

        return np.asarray(
            values,
            dtype=np.float32,
        ).copy(), (
            1,
            1,
        )


    hr = max(
        1,
        int(
            math.ceil(
                radius_m
                /
                az_spacing
            )
        ),
    )

    hc = max(
        1,
        int(
            math.ceil(
                radius_m
                /
                ground_range_spacing
            )
        ),
    )


    kh = (
        2*hr + 1
    )

    kw = (
        2*hc + 1
    )


    num = np.zeros(
        (
            H,
            W,
        ),
        dtype=np.float64,
    )

    den = np.zeros(
        (
            H,
            W,
        ),
        dtype=np.float64,
    )


    vv = np.asarray(
        values,
        dtype=np.float64,
    )

    ww = np.asarray(
        weights,
        dtype=np.float64,
    )


    num[
        row,
        col
    ] = (
        vv * ww
    )


    den[
        row,
        col
    ] = ww


    num_f = uniform_filter(
        num,
        size=(
            kh,
            kw,
        ),
        mode="constant",
        cval=0.0,
    )


    den_f = uniform_filter(
        den,
        size=(
            kh,
            kw,
        ),
        mode="constant",
        cval=0.0,
    )


    np.divide(
        num_f,
        den_f,
        out=num_f,
        where=(
            den_f > 0
        ),
    )


    out = num_f[
        row,
        col
    ].astype(
        np.float32
    )


    return out, (
        kh,
        kw,
    )


# ============================================================
# Scale audit
# ============================================================

raw_scale = robust_scale(
    beta_full
)


audit = []

products = {}


t1 = time.perf_counter()


for radius in RADII_M:

    # Fold comparison uses equal weights.
    sa, window = smooth_points(
        beta_a,
        ones,
        radius,
    )

    sb, _ = smooth_points(
        beta_b,
        ones,
        radius,
    )


    # Full-data product uses inverse-variance weights.
    sf, _ = smooth_points(
        beta_full,
        weight_full,
        radius,
    )


    corr = centered_corr(
        sa,
        sb,
    )


    fold_diff = (
        sa.astype(
            np.float64
        )
        -
        sb.astype(
            np.float64
        )
    )


    fold_rms = float(
        np.sqrt(
            np.mean(
                fold_diff
                *
                fold_diff
            )
        )
    )


    scale = robust_scale(
        sf
    )


    retention = (
        scale
        /
        raw_scale
        if raw_scale > 0
        else np.nan
    )


    audit.append(
        {
            "radius_m":
                float(radius),

            "window_rows":
                int(
                    window[0]
                ),

            "window_cols":
                int(
                    window[1]
                ),

            "split_half_corr":
                float(corr),

            "split_half_rms":
                fold_rms,

            "robust_scale":
                float(scale),

            "scale_retention":
                float(retention),
        }
    )


    products[
        float(radius)
    ] = sf


smooth_seconds = (
    time.perf_counter()
    -
    t1
)


# ============================================================
# Select smallest scale on stability plateau
# ============================================================

eligible = [
    x
    for x in audit
    if (
        np.isfinite(
            x["split_half_corr"]
        )
        and
        x["scale_retention"]
        >=
        MIN_SCALE_RETENTION
    )
]


if not eligible:

    raise RuntimeError(
        "no regularization scale retained enough signal"
    )


max_corr = max(
    x[
        "split_half_corr"
    ]
    for x in eligible
)


if max_corr < MIN_STABILITY_CORR:

    raise RuntimeError(
        (
            "split-half SCLA stability too weak: "
            f"max corr={max_corr:.6f}"
        )
    )


target_corr = (
    PLATEAU_FRACTION
    *
    max_corr
)


selected = min(
    (
        x
        for x in eligible
        if (
            x[
                "split_half_corr"
            ]
            >=
            target_corr
        )
    ),
    key=lambda x: x[
        "radius_m"
    ],
)


radius_sel = float(
    selected[
        "radius_m"
    ]
)


beta_reg = products[
    radius_sel
].astype(
    np.float32
)


# ============================================================
# Reference the SCLA coefficient itself
#
# Input phase is spatially referenced, therefore constant
# beta offset is not physically useful.
# ============================================================

beta_ref_med_before = float(
    np.median(
        beta_reg[
            ref_idx
        ].astype(
            np.float64
        )
    )
)


beta_reg = (
    beta_reg.astype(
        np.float64
    )
    -
    beta_ref_med_before
).astype(
    np.float32
)


beta_ref_med_after = float(
    np.median(
        beta_reg[
            ref_idx
        ].astype(
            np.float64
        )
    )
)


# ============================================================
# Spatial structure after regularization
# ============================================================

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
] = beta_reg


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


adj = np.concatenate(
    (
        dh[
            np.isfinite(dh)
        ],
        dv[
            np.isfinite(dv)
        ],
    )
)


adj_med = float(
    np.median(
        adj
    )
)


rng = np.random.default_rng(
    20260824
)


nr = min(
    500000,
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
            beta_reg[
                ia
            ].astype(
                np.float64
            )
            -
            beta_reg[
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
)


# ============================================================
# Save
# ============================================================

np.save(
    BETA_OUT,
    beta_reg
)


with CSV_OUT.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            audit[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        audit
    )


manifest = {

    "status":
        "PASS_SCLA_SPATIAL_REGULARIZATION_CANDIDATE",

    "production_phase_modified":
        False,

    "method":
        (
            "split-half temporal stability "
            "+ inverse-variance normalized box regularization"
        ),

    "points":
        int(npoint),

    "epochs":
        int(ndate),

    "folds":
        {
            "A_indices":
                fold_a.tolist(),

            "B_indices":
                fold_b.tolist(),
        },

    "pixel_spacing_m":
        {
            "ground_range":
                float(
                    ground_range_spacing
                ),

            "azimuth":
                float(
                    az_spacing
                ),
        },

    "selection":
        {
            "candidate_radii_m":
                RADII_M,

            "minimum_scale_retention":
                MIN_SCALE_RETENTION,

            "plateau_fraction":
                PLATEAU_FRACTION,

            "max_split_half_corr":
                float(
                    max_corr
                ),

            "target_corr":
                float(
                    target_corr
                ),

            "selected_radius_m":
                radius_sel,

            "selected_window_rows":
                int(
                    selected[
                        "window_rows"
                    ]
                ),

            "selected_window_cols":
                int(
                    selected[
                        "window_cols"
                    ]
                ),

            "selected_split_half_corr":
                float(
                    selected[
                        "split_half_corr"
                    ]
                ),

            "selected_scale_retention":
                float(
                    selected[
                        "scale_retention"
                    ]
                ),
        },

    "weighting":
        {
            "type":
                "inverse_variance",

            "sigma_floor":
                sigma_floor,

            "weight_clip_percentiles":
                [
                    2,
                    98,
                ],
        },

    "qa":
        {
            "beta_reference_median_before":
                beta_ref_med_before,

            "beta_reference_median_after":
                beta_ref_med_after,

            "adjacent_abs_dbeta_median":
                adj_med,

            "random_abs_dbeta_median":
                rand_med,

            "adjacent_to_random_ratio":
                spatial_ratio,

            "raw_beta_robust_scale":
                float(
                    raw_scale
                ),

            "regularized_beta_robust_scale":
                float(
                    robust_scale(
                        beta_reg
                    )
                ),
        },

    "performance":
        {
            "fold_estimation_seconds":
                fold_seconds,

            "scale_audit_seconds":
                smooth_seconds,
        },

    "output_beta":
        str(
            BETA_OUT
        ),

    "next":
        "P15-5D_SCLA_CORRECTION_APPLICATION",
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    +
    "\n"
)


# ============================================================
# Summary
# ============================================================

print("=" * 88)
print("P15-5C SCLA SPATIAL REGULARIZATION SCALE AUDIT")
print("=" * 88)

print(
    "points                    :",
    f"{npoint:,}",
)

print(
    "radar grid                :",
    f"{H} x {W}",
)

print(
    "range / az spacing m      :",
    (
        f"{ground_range_spacing:.4f} / "
        f"{az_spacing:.4f}"
    ),
)

print(
    "fold estimation seconds   :",
    f"{fold_seconds:.6f}",
)

print(
    "scale audit seconds       :",
    f"{smooth_seconds:.6f}",
)

print()

print(
    "radius(m)  window       split_corr  split_RMS   retention"
)

for x in audit:

    print(
        f"{x['radius_m']:8.1f}  "
        f"{x['window_rows']:3d}x{x['window_cols']:<3d}  "
        f"{x['split_half_corr']:10.6f}  "
        f"{x['split_half_rms']:10.6e}  "
        f"{x['scale_retention']:9.6f}"
    )


print()

print(
    "max split-half corr       :",
    f"{max_corr:.6f}",
)

print(
    "plateau target            :",
    f"{target_corr:.6f}",
)

print(
    "SELECTED radius m         :",
    f"{radius_sel:.1f}",
)

print(
    "SELECTED window           :",
    (
        f"{selected['window_rows']} x "
        f"{selected['window_cols']}"
    ),
)

print(
    "SELECTED split corr       :",
    f"{selected['split_half_corr']:.6f}",
)

print(
    "SELECTED retention        :",
    f"{selected['scale_retention']:.6f}",
)

print()

print(
    "beta ref median before    :",
    f"{beta_ref_med_before:.9e}",
)

print(
    "beta ref median after     :",
    f"{beta_ref_med_after:.9e}",
)

print(
    "adjacent/random ratio     :",
    f"{spatial_ratio:.6f}",
)

print()

print(
    "output beta               :",
    BETA_OUT,
)

print(
    "manifest                  :",
    MANIFEST,
)

print(
    "production phase modified :",
    False,
)

print("=" * 88)

print(
    "P15-5C FINAL RESULT: "
    "PASS_SCLA_SPATIAL_REGULARIZATION_CANDIDATE"
)

print("=" * 88)
