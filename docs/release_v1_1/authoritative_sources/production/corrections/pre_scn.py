from pathlib import Path
import json
import os
import re
import time

import numpy as np


# ======================================================================
# Paths
# ======================================================================

ROOT = Path("/home/ubuntu/Downloads")
PSDS = ROOT / "psds"
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
    / "stamps_scla_final_pass"
)

K_FILE = (
    SCLA
    / "K_ps_uw_rad_per_m_bperp.npy"
)

C_FILE = (
    SCLA
    / "C_ps_uw_rad.npy"
)

COEFF = (
    SCLA
    / "baseline_sm_transform_coefficients.npz"
)

C_GLS = (
    SCLA
    / "C_ps_uw_gls_contract.npz"
)

CATALOG = (
    PROC
    / "stamps_scla_baseline"
    / "complete_108_baseline_sources.json"
)

PLIST = (
    PROC
    / "gacos_geometry"
    / "strict_points.plist"
)

REF_FILE = (
    PROC
    / "gacos_fast_smoke"
    / "project_local_interp"
    / "ref_idx.npy"
)

RSLC_PAR = (
    ROOT
    / "RSLC"
    / "20151212.rslc.par"
)

OUTDIR = (
    PROC
    / "stamps_pre_scn_phase"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT = (
    OUTDIR
    / "acquisition_phase_pre_scn_rad.npy"
)

TMP = (
    OUTDIR
    / ".acquisition_phase_pre_scn_rad.tmp.npy"
)

MANIFEST = (
    OUTDIR
    / "p15_5b6_pre_scn_manifest.json"
)


CHUNK = 131072
SAMPLE_N = 4096

# Difference introduced only by eliminating pySTAMPS'
# intermediate float32 Bperp matrices.
BPERP_FLOAT32_TOL_M = 5e-5
PH_SCLA_FLOAT32_TOL_RAD = 1e-6
C_FLOAT32_TOL_RAD = 1e-6
PRE_SCN_FLOAT32_TOL_RAD = 2e-6


# ======================================================================
# Helpers
# ======================================================================

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

        m = NUM_RE.search(
            x
        )

        if m:
            return float(
                m.group(0)
            )

    raise KeyError(
        " / ".join(names)
    )


def parse_vector(
    text,
    labels,
):

    for line in text.splitlines():

        if not any(
            x in line
            for x in labels
        ):
            continue

        rhs = (
            line.split(
                ":",
                1,
            )[1]
            if ":" in line
            else line
        )

        x = NUM_RE.findall(
            rhs
        )

        if len(x) >= 3:

            return np.asarray(
                [
                    float(v)
                    for v in x[:3]
                ],
                dtype=np.float64,
            )

    return None


def parse_base(path):

    text = path.read_text(
        errors="ignore"
    )

    B = parse_vector(
        text,
        (
            "initial_baseline(TCN)",
            "initial_baseline",
        ),
    )

    Br = parse_vector(
        text,
        (
            "initial_baseline_rate",
            "baseline_rate(TCN)",
        ),
    )

    if (
        B is None
        or Br is None
    ):
        raise RuntimeError(
            f"invalid .base: {path}"
        )

    return B, Br


# ======================================================================
# Inputs
# ======================================================================

for p in (
    PHASE,
    GMAN,
    K_FILE,
    C_FILE,
    COEFF,
    C_GLS,
    CATALOG,
    PLIST,
    REF_FILE,
    RSLC_PAR,
):

    if not p.is_file():
        raise FileNotFoundError(p)


phase = np.load(
    PHASE,
    mmap_mode="r",
)

Kf32 = np.load(
    K_FILE,
    mmap_mode="r",
)

Cf32 = np.load(
    C_FILE,
    mmap_mode="r",
)

coef = np.load(
    COEFF,
    allow_pickle=False,
)

cgls = np.load(
    C_GLS,
    allow_pickle=False,
)

gman = json.loads(
    GMAN.read_text()
)

catalog = json.loads(
    CATALOG.read_text()
)


dates = list(
    gman["acquisition_dates"]
)


npoint, nimage = phase.shape


if (
    nimage != 38
    or
    len(dates) != 38
    or
    Kf32.shape != (npoint,)
    or
    Cf32.shape != (npoint,)
):

    raise RuntimeError(
        "phase/K/C contract failed"
    )


master0 = int(
    np.asarray(
        coef["master0"]
    ).reshape(-1)[0]
)


img0 = np.asarray(
    coef["img0"],
    dtype=np.int64,
).reshape(-1)


if (
    dates[master0]
    !=
    "20151212"
    or
    img0.size != 37
):

    raise RuntimeError(
        "geometric-master contract failed"
    )


Pbase = np.asarray(
    coef["Pbase"],
    dtype=np.float64,
)


C_SM = np.asarray(
    coef["C_SM"],
    dtype=np.float64,
).reshape(-1)

N_SM = np.asarray(
    coef["N_SM"],
    dtype=np.float64,
).reshape(-1)

CR_SM = np.asarray(
    coef["CR_SM"],
    dtype=np.float64,
).reshape(-1)

NR_SM = np.asarray(
    coef["NR_SM"],
    dtype=np.float64,
).reshape(-1)


phase_weights = np.asarray(
    coef["phase_weights"],
    dtype=np.float64,
).reshape(-1)


Pc = np.asarray(
    cgls["Pc"],
    dtype=np.float64,
)


if (
    Pbase.shape != (37, 108)
    or
    phase_weights.shape != (37,)
    or
    Pc.shape != (2, 37)
):

    raise RuntimeError(
        "coefficient contract failed"
    )


# ======================================================================
# Reference-area mean used by StaMPS Stage 7
# ======================================================================

ref_idx = np.load(
    REF_FILE
).astype(
    np.int64
)


if ref_idx.size != 607:

    raise RuntimeError(
        f"reference points={ref_idx.size}"
    )


ref_mean = np.nanmean(
    np.asarray(
        phase[
            ref_idx,
            :
        ],
        dtype=np.float64,
    ),
    axis=0,
)


k_ref_projection = float(
    ref_mean[
        img0
    ]
    @
    phase_weights
)


# ======================================================================
# Point coordinates
# ======================================================================

plist = np.fromfile(
    PLIST,
    dtype=">i4",
).reshape(
    -1,
    2,
)


if plist.shape[0] != npoint:

    raise RuntimeError(
        "plist point-count mismatch"
    )


col = plist[:, 0].astype(
    np.float64
)

row = plist[:, 1].astype(
    np.float64
)


# ======================================================================
# GAMMA geometry exactly as previous pointwise-Bperp stages
# ======================================================================

pars = read_par(
    RSLC_PAR
)


rslc_length = int(
    round(
        par_scalar(
            pars,
            (
                "azimuth_lines",
                "nlines",
            ),
        )
    )
)


range_spacing = par_scalar(
    pars,
    (
        "range_pixel_spacing",
    ),
)


near_range = par_scalar(
    pars,
    (
        "near_range_slc",
        "near_range",
    ),
)


sar_to_earth = par_scalar(
    pars,
    (
        "sar_to_earth_center",
    ),
)


earth_radius = par_scalar(
    pars,
    (
        "earth_radius_below_sensor",
    ),
)


prf = par_scalar(
    pars,
    (
        "prf",
    ),
)


range_looks = 4
azimuth_looks = 1


mean_azimuth = (
    rslc_length
    /
    2.0
    -
    0.5
)


def geometry_factors(
    rr,
    cc,
):

    range_original = (
        cc
        *
        range_looks
        +
        (
            range_looks
            -
            1
        )
        /
        2.0
    )


    azimuth_original = (
        rr
        *
        azimuth_looks
        +
        (
            azimuth_looks
            -
            1
        )
        /
        2.0
    )


    slant_range = (
        near_range
        +
        range_original
        *
        range_spacing
    )


    arg = (
        sar_to_earth**2
        +
        slant_range**2
        -
        earth_radius**2
    ) / (
        2.0
        *
        sar_to_earth
        *
        slant_range
    )


    look = np.arccos(
        np.clip(
            arg,
            -1.0,
            1.0,
        )
    )


    cs = np.cos(
        look
    )

    ss = np.sin(
        look
    )


    dt = (
        azimuth_original
        -
        mean_azimuth
    ) / prf


    return (
        cs,
        ss,
        dt,
    )


def fast_bsm(
    rr,
    cc,
):

    cs, ss, dt = geometry_factors(
        rr,
        cc,
    )


    return (
        cs[:, None]
        *
        C_SM[None, :]

        -
        ss[:, None]
        *
        N_SM[None, :]

        +
        (
            dt
            *
            cs
        )[:, None]
        *
        CR_SM[None, :]

        -
        (
            dt
            *
            ss
        )[:, None]
        *
        NR_SM[None, :]
    )


# ======================================================================
# Build raw 108 GAMMA coefficients for a pySTAMPS arithmetic oracle
# ======================================================================

rows_cat = catalog.get(
    "pairs",
    []
)


if len(rows_cat) != 108:

    raise RuntimeError(
        "baseline catalog != 108"
    )


BC = np.empty(
    108,
    dtype=np.float64,
)

BN = np.empty(
    108,
    dtype=np.float64,
)

BRC = np.empty(
    108,
    dtype=np.float64,
)

BRN = np.empty(
    108,
    dtype=np.float64,
)


for e, item in enumerate(
    rows_cat
):

    if int(
        item["edge"]
    ) != e + 1:

        raise RuntimeError(
            "baseline catalog ordering failed"
        )


    B, Br = parse_base(
        Path(
            item["base_file"]
        )
    )


    orientation = int(
        item.get(
            "orientation",
            1,
        )
    )


    B *= orientation
    Br *= orientation


    BC[e] = B[1]
    BN[e] = B[2]

    BRC[e] = Br[1]
    BRN[e] = Br[2]


# ======================================================================
# Precision gate against actual pySTAMPS Stage-7 arithmetic
#
# pySTAMPS:
#   bp2.bperp_mat -> float32
#   bsome = float64(bperp_ifg) @ Pbase.T
#   bperp_sm = float32(bsome)
#
# This explicitly tests the intermediate float32 quantisation that the
# earlier algebraic optimization intentionally eliminated.
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


cs, ss, dt = geometry_factors(
    row[sample_idx],
    col[sample_idx],
)


bifg64 = (
    cs[:, None]
    *
    BC[None, :]

    -
    ss[:, None]
    *
    BN[None, :]

    +
    (
        dt
        *
        cs
    )[:, None]
    *
    BRC[None, :]

    -
    (
        dt
        *
        ss
    )[:, None]
    *
    BRN[None, :]
)


# Actual bp2 input precision in pySTAMPS Stage 7.
bifg32 = bifg64.astype(
    np.float32
)


bsm_oracle64 = (
    bifg32.astype(
        np.float64
    )
    @
    Pbase.T
)


# Stored bperp_sm in pySTAMPS Stage 7.
bsm_oracle32 = (
    bsm_oracle64.astype(
        np.float32
    )
)


bsm_fast64 = fast_bsm(
    row[sample_idx],
    col[sample_idx],
)


bsm_fast32 = (
    bsm_fast64.astype(
        np.float32
    )
)


bsm_float_diff = (
    bsm_fast32.astype(
        np.float64
    )
    -
    bsm_oracle32.astype(
        np.float64
    )
)


bsm_float_max = float(
    np.max(
        np.abs(
            bsm_float_diff
        )
    )
)


bsm_float_rms = float(
    np.sqrt(
        np.mean(
            bsm_float_diff
            *
            bsm_float_diff
        )
    )
)


if (
    bsm_float_max
    >
    BPERP_FLOAT32_TOL_M
):

    raise RuntimeError(
        (
            "fast Bperp differs from "
            "pySTAMPS float32 arithmetic: "
            f"{bsm_float_max} m"
        )
    )


# ======================================================================
# Reconstruct K64 used to make ph_scla.
#
# P15-5B4 persisted K as float32 for storage, but pySTAMPS builds
# ph_scla from its unrounded K estimate before writing scla2.
#
# Reconstruct K64 and require its float32 representation to equal the
# frozen K file exactly.
# ======================================================================

sample_phase = np.asarray(
    phase[
        sample_idx,
        :
    ],
    dtype=np.float64,
)


K64_sample = (
    sample_phase[
        :,
        img0
    ]
    @
    phase_weights
    -
    k_ref_projection
)


k_storage_diff = float(
    np.max(
        np.abs(
            K64_sample.astype(
                np.float32
            ).astype(
                np.float64
            )
            -
            np.asarray(
                Kf32[
                    sample_idx
                ],
                dtype=np.float64,
            )
        )
    )
)


if k_storage_diff != 0.0:

    raise RuntimeError(
        (
            "frozen K storage does not match "
            f"reconstructed K: {k_storage_diff}"
        )
    )


# ======================================================================
# ph_scla precision oracle
# ======================================================================

phscla_oracle = (
    K64_sample[:, None]
    *
    bsm_oracle32.astype(
        np.float64
    )
).astype(
    np.float32
)


phscla_fast = (
    K64_sample[:, None]
    *
    bsm_fast64
).astype(
    np.float32
)


phscla_diff = (
    phscla_fast.astype(
        np.float64
    )
    -
    phscla_oracle.astype(
        np.float64
    )
)


phscla_max = float(
    np.max(
        np.abs(
            phscla_diff
        )
    )
)


phscla_rms = float(
    np.sqrt(
        np.mean(
            phscla_diff
            *
            phscla_diff
        )
    )
)


if (
    phscla_max
    >
    PH_SCLA_FLOAT32_TOL_RAD
):

    raise RuntimeError(
        (
            "fast ph_scla differs from "
            "pySTAMPS arithmetic: "
            f"{phscla_max} rad"
        )
    )


# ======================================================================
# C_ps precision oracle
#
# Official Stage 7 estimates C before casting it to float32 for Stage 8.
# Compare its float32 Stage-8 representation with frozen C_file.
# ======================================================================

sample_y_for_c = (
    sample_phase[
        :,
        img0
    ]
    -
    ref_mean[
        img0
    ][None, :]
    -
    K64_sample[:, None]
    *
    bsm_oracle32.astype(
        np.float64
    )
)


C_oracle64 = (
    sample_y_for_c
    @
    Pc.T
)[:, 0]


C_oracle32 = (
    C_oracle64.astype(
        np.float32
    )
)


C_frozen_sample = np.asarray(
    Cf32[
        sample_idx
    ],
    dtype=np.float32,
)


C_float_diff = (
    C_frozen_sample.astype(
        np.float64
    )
    -
    C_oracle32.astype(
        np.float64
    )
)


C_float_max = float(
    np.max(
        np.abs(
            C_float_diff
        )
    )
)


C_float_rms = float(
    np.sqrt(
        np.mean(
            C_float_diff
            *
            C_float_diff
        )
    )
)


if (
    C_float_max
    >
    C_FLOAT32_TOL_RAD
):

    raise RuntimeError(
        (
            "frozen C differs from "
            "pySTAMPS float32 representation: "
            f"{C_float_max} rad"
        )
    )


# ======================================================================
# Stage-8 arithmetic oracle on the same sample
#
# Official Stage 8:
#
#   ph_all = single(ph_uw)
#   ph_all -= single(ph_scla)
#   ph_all -= single(C_ps_uw)
# ======================================================================

# Simpler explicit full model.
phscla_oracle_full = np.zeros(
    (
        sample_n,
        38,
    ),
    dtype=np.float32,
)

phscla_oracle_full[
    :,
    img0
] = phscla_oracle


pre_oracle = (
    sample_phase.astype(
        np.float32
    )
    -
    phscla_oracle_full
    -
    C_oracle32[:, None]
)


phscla_fast_full = np.zeros(
    (
        sample_n,
        38,
    ),
    dtype=np.float32,
)

phscla_fast_full[
    :,
    img0
] = phscla_fast


pre_fast_sample = (
    sample_phase.astype(
        np.float32
    )
    -
    phscla_fast_full
    -
    C_frozen_sample[:, None]
)


pre_diff = (
    pre_fast_sample.astype(
        np.float64
    )
    -
    pre_oracle.astype(
        np.float64
    )
)


pre_float_max = float(
    np.max(
        np.abs(
            pre_diff
        )
    )
)


pre_float_rms = float(
    np.sqrt(
        np.mean(
            pre_diff
            *
            pre_diff
        )
    )
)


if (
    pre_float_max
    >
    PRE_SCN_FLOAT32_TOL_RAD
):

    raise RuntimeError(
        (
            "pre-SCN fast path differs from "
            "pySTAMPS arithmetic: "
            f"{pre_float_max} rad"
        )
    )


# ======================================================================
# Production materialization
# ======================================================================

if TMP.exists():
    TMP.unlink()


out = np.lib.format.open_memmap(
    TMP,
    mode="w+",
    dtype=np.float32,
    shape=(
        npoint,
        38,
    ),
)


max_k_storage_diff = 0.0

max_abs_correction = 0.0

correction_ss = 0.0
correction_n = 0

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


    ph32 = np.asarray(
        phase[
            start:stop,
            :
        ],
        dtype=np.float32,
    )


    # Rebuild the unrounded K used for ph_scla.
    k64 = (
        ph32[
            :,
            img0
        ].astype(
            np.float64
        )
        @
        phase_weights
        -
        k_ref_projection
    )


    # Frozen-K integrity gate.
    kd = np.max(
        np.abs(
            k64.astype(
                np.float32
            ).astype(
                np.float64
            )
            -
            np.asarray(
                Kf32[
                    start:stop
                ],
                dtype=np.float64,
            )
        )
    )


    max_k_storage_diff = max(
        max_k_storage_diff,
        float(kd),
    )


    # Fast pointwise single-master Bperp.
    bsm = fast_bsm(
        row[
            start:stop
        ],
        col[
            start:stop
        ],
    )


    ph_scla = (
        k64[:, None]
        *
        bsm
    ).astype(
        np.float32
    )


    c32 = np.asarray(
        Cf32[
            start:stop
        ],
        dtype=np.float32,
    )


    # Official Stage-8 float32 arithmetic.
    y = ph32.copy()


    y[
        :,
        img0
    ] -= ph_scla


    y -= c32[
        :,
        None
    ]


    if not np.all(
        np.isfinite(
            y
        )
    ):

        raise RuntimeError(
            (
                "non-finite pre-SCN phase "
                f"in {start}:{stop}"
            )
        )


    out[
        start:stop,
        :
    ] = y


    # QA correction magnitude:
    correction = (
        ph32.astype(
            np.float64
        )
        -
        y.astype(
            np.float64
        )
    )


    max_abs_correction = max(
        max_abs_correction,
        float(
            np.max(
                np.abs(
                    correction
                )
            )
        ),
    )


    correction_ss += float(
        np.sum(
            correction
            *
            correction
        )
    )


    correction_n += int(
        correction.size
    )


out.flush()


materialize_seconds = (
    time.perf_counter()
    -
    t0
)


if max_k_storage_diff != 0.0:

    raise RuntimeError(
        (
            "production reconstructed K "
            "does not match frozen K float32: "
            f"{max_k_storage_diff}"
        )
    )


# Atomic publish only after every gate passes.
os.replace(
    TMP,
    OUT,
)


# ======================================================================
# Output QA
# ======================================================================

pre = np.load(
    OUT,
    mmap_mode="r",
)


finite_fraction = float(
    np.mean(
        np.isfinite(
            pre
        )
    )
)


if finite_fraction != 1.0:

    raise RuntimeError(
        "pre-SCN finite fraction < 1"
    )


correction_rms = float(
    np.sqrt(
        correction_ss
        /
        correction_n
    )
)


# Stage-8 input is NOT required to retain the old temporal-reference zero.
epoch0_abs_p = np.percentile(
    np.abs(
        np.asarray(
            pre[:, 0],
            dtype=np.float64,
        )
    ),
    [
        50,
        95,
        99,
    ],
)


master_abs_p = np.percentile(
    np.abs(
        np.asarray(
            pre[:, master0],
            dtype=np.float64,
        )
    ),
    [
        50,
        95,
        99,
    ],
)


# Reference-area means after correction are diagnostic only.
reference_epoch_mean = np.mean(
    np.asarray(
        pre[
            ref_idx,
            :
        ],
        dtype=np.float64,
    ),
    axis=0,
)


# ======================================================================
# Manifest
# ======================================================================

manifest = {

    "status":
        "PASS_STAMPS_PRE_SCN_PHASE",

    "formula":
        (
            "phi_preSCN = "
            "phi_GACOS - ph_scla - C_ps_uw"
        ),

    "stage8_source_semantics":
        {
            "phase_dtype":
                "float32",

            "ph_scla_dtype":
                "float32",

            "C_dtype":
                "float32",

            "ramp_applied":
                False,
        },

    "points":
        int(
            npoint
        ),

    "images":
        38,

    "geometric_master":
        {
            "date":
                dates[
                    master0
                ],

            "index_0based":
                master0,
        },

    "hard_precision_gate":
        {
            "sample_points":
                sample_n,

            "fast_vs_pystamps_Bperp_float32_max_m":
                bsm_float_max,

            "fast_vs_pystamps_Bperp_float32_rms_m":
                bsm_float_rms,

            "Bperp_tolerance_m":
                BPERP_FLOAT32_TOL_M,

            "fast_vs_pystamps_ph_scla_max_rad":
                phscla_max,

            "fast_vs_pystamps_ph_scla_rms_rad":
                phscla_rms,

            "ph_scla_tolerance_rad":
                PH_SCLA_FLOAT32_TOL_RAD,

            "frozen_C_vs_pystamps_C_float32_max_rad":
                C_float_max,

            "frozen_C_vs_pystamps_C_float32_rms_rad":
                C_float_rms,

            "C_tolerance_rad":
                C_FLOAT32_TOL_RAD,

            "preSCN_fast_vs_pystamps_max_rad":
                pre_float_max,

            "preSCN_fast_vs_pystamps_rms_rad":
                pre_float_rms,

            "preSCN_tolerance_rad":
                PRE_SCN_FLOAT32_TOL_RAD,

            "frozen_K_float32_max_diff":
                max_k_storage_diff,
        },

    "production":
        {
            "seconds":
                materialize_seconds,

            "point_epochs_per_second":
                (
                    npoint
                    *
                    38
                    /
                    materialize_seconds
                ),

            "finite_fraction":
                finite_fraction,

            "correction_rms_rad":
                correction_rms,

            "correction_max_abs_rad":
                max_abs_correction,
        },

    "diagnostic":
        {
            "epoch0_abs_p50_p95_p99_rad":
                [
                    float(x)
                    for x in epoch0_abs_p
                ],

            "geometric_master_abs_p50_p95_p99_rad":
                [
                    float(x)
                    for x in master_abs_p
                ],

            "reference_epoch_mean_max_abs_rad":
                float(
                    np.max(
                        np.abs(
                            reference_epoch_mean
                        )
                    )
                ),
        },

    "inputs":
        {
            "phase":
                str(
                    PHASE
                ),

            "K_ps_uw":
                str(
                    K_FILE
                ),

            "C_ps_uw":
                str(
                    C_FILE
                ),

            "baseline_coefficients":
                str(
                    COEFF
                ),

            "baseline_catalog":
                str(
                    CATALOG
                ),
        },

    "outputs":
        {
            "pre_scn_phase":
                str(
                    OUT
                ),

            "ph_scla_persisted":
                False,

            "pointwise_Bperp_persisted":
                False,
        },

    "production_phase_modified":
        False,

    "next":
        (
            "P15-6 Stage-8 ps_scn_filt "
            "parity implementation"
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

print("=" * 92)
print("P15-5B6 STAMPS SCLA -> PRE-SCN PHASE")
print("=" * 92)

print(
    "points / images                 :",
    f"{npoint:,} / 38",
)

print(
    "geometric master                :",
    (
        f"{dates[master0]} "
        f"(0b={master0})"
    ),
)

print()

print(
    "Bperp float32 oracle max diff m :",
    f"{bsm_float_max:.12e}",
)

print(
    "Bperp float32 oracle RMS m      :",
    f"{bsm_float_rms:.12e}",
)

print(
    "ph_scla oracle max diff rad     :",
    f"{phscla_max:.12e}",
)

print(
    "ph_scla oracle RMS rad          :",
    f"{phscla_rms:.12e}",
)

print(
    "C oracle max diff rad           :",
    f"{C_float_max:.12e}",
)

print(
    "C oracle RMS rad                :",
    f"{C_float_rms:.12e}",
)

print(
    "pre-SCN oracle max diff rad     :",
    f"{pre_float_max:.12e}",
)

print(
    "pre-SCN oracle RMS rad          :",
    f"{pre_float_rms:.12e}",
)

print(
    "K frozen float32 max diff       :",
    f"{max_k_storage_diff:.12e}",
)

print()

print(
    "materialization seconds         :",
    f"{materialize_seconds:.6f}",
)

print(
    "throughput                      :",
    (
        f"{npoint*38/materialize_seconds:,.0f} "
        "point-epochs/s"
    ),
)

print(
    "finite fraction                 :",
    f"{finite_fraction:.12f}",
)

print(
    "SCLA+C correction RMS rad       :",
    f"{correction_rms:.6f}",
)

print(
    "SCLA+C correction max |rad|     :",
    f"{max_abs_correction:.6f}",
)

print()

print(
    "epoch0 |phase| p50/p95/p99      :",
    epoch0_abs_p,
)

print(
    "master |phase| p50/p95/p99      :",
    master_abs_p,
)

print()

print(
    "pointwise Bperp persisted        :",
    False,
)

print(
    "ph_scla matrix persisted         :",
    False,
)

print(
    "original GACOS phase modified    :",
    False,
)

print(
    "pre-SCN output                   :",
    OUT,
)

print(
    "manifest                         :",
    MANIFEST,
)

print("=" * 92)

print(
    "P15-5B6 FINAL RESULT: "
    "PASS_STAMPS_PRE_SCN_PHASE"
)

print("=" * 92)
