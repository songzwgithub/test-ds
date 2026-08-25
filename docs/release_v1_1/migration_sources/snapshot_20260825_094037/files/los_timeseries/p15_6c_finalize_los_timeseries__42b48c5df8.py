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

PRE = (
    PROC
    / "stamps_pre_scn_phase"
    / "acquisition_phase_pre_scn_rad.npy"
)

SCN = (
    PROC
    / "stamps_stage8"
    / "ph_scn_slave_rad.npy"
)

SCN_MANIFEST = (
    PROC
    / "stamps_stage8"
    / "p15_6b2_stage8_scn_manifest.json"
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

RSLC_PAR = (
    ROOT
    / "RSLC"
    / "20151212.rslc.par"
)

OUTDIR = (
    PROC
    / "final_los_timeseries"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


PHASE_OUT = (
    OUTDIR
    / "acquisition_phase_final_rad.npy"
)

LOS_M_OUT = (
    OUTDIR
    / "los_displacement_toward_satellite_m.npy"
)

LOS_MM_OUT = (
    OUTDIR
    / "los_displacement_toward_satellite_mm.npy"
)

MANIFEST = (
    OUTDIR
    / "p15_6c_final_los_timeseries_manifest.json"
)


PHASE_TMP = (
    OUTDIR
    / ".acquisition_phase_final_rad.tmp.npy"
)

LOS_M_TMP = (
    OUTDIR
    / ".los_displacement_toward_satellite_m.tmp.npy"
)

LOS_MM_TMP = (
    OUTDIR
    / ".los_displacement_toward_satellite_mm.tmp.npy"
)


# ======================================================================
# Frozen conventions
# ======================================================================

TEMPORAL_REFERENCE_DATE = "20141006"

GEOMETRIC_MASTER_DATE = "20151212"

REFERENCE_POINTS_EXPECTED = 607

C0 = 299792458.0

CHUNK = 131072


# Precision gates
REF_MEDIAN_TOL_RAD = 1.0e-7

EPOCH0_TOL_RAD = 0.0

PHASE_STORAGE_TOL_RAD = 2.0e-6

LOS_STORAGE_TOL_M = 2.0e-8


# ======================================================================
# Helpers
# ======================================================================

NUM_RE = re.compile(
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)


def read_par(path):

    result = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(
            ":",
            1,
        )

        result[
            k.strip().lower()
        ] = v.strip()

    return result


def par_scalar(
    pars,
    keys,
):

    for key in keys:

        x = pars.get(
            key.lower()
        )

        if x is None:
            continue

        m = NUM_RE.search(x)

        if m:
            return float(
                m.group(0)
            )

    raise KeyError(
        " / ".join(keys)
    )


# ======================================================================
# Input contracts
# ======================================================================

for p in (
    PRE,
    SCN,
    SCN_MANIFEST,
    GMAN,
    REF_FILE,
    RSLC_PAR,
):

    if not p.is_file():
        raise FileNotFoundError(p)


pre = np.load(
    PRE,
    mmap_mode="r",
)

scn = np.load(
    SCN,
    mmap_mode="r",
)


if pre.shape != scn.shape:

    raise RuntimeError(
        (
            "pre-SCN / SCN shape mismatch: "
            f"{pre.shape} vs {scn.shape}"
        )
    )


npoint, nepoch = pre.shape


if nepoch != 38:

    raise RuntimeError(
        f"expected 38 acquisitions, got {nepoch}"
    )


gman = json.loads(
    GMAN.read_text()
)


dates = list(
    gman[
        "acquisition_dates"
    ]
)


if len(dates) != nepoch:

    raise RuntimeError(
        "date count mismatch"
    )


if dates[0] != TEMPORAL_REFERENCE_DATE:

    raise RuntimeError(
        (
            "frozen temporal reference "
            f"changed: first date={dates[0]}"
        )
    )


tref0 = dates.index(
    TEMPORAL_REFERENCE_DATE
)


master0 = dates.index(
    GEOMETRIC_MASTER_DATE
)


scn_manifest = json.loads(
    SCN_MANIFEST.read_text()
)


if (
    scn_manifest.get(
        "status"
    )
    !=
    "PASS_STAMPS_STAGE8_SCN"
):

    raise RuntimeError(
        "P15-6B2 is not PASS"
    )


if int(
    scn_manifest[
        "scientific_contract"
    ][
        "geometric_master_index_0based"
    ]
) != master0:

    raise RuntimeError(
        "SCN geometric-master contract changed"
    )


# ======================================================================
# Frozen reference point set
# ======================================================================

ref_idx = np.load(
    REF_FILE
).astype(
    np.int64
)


if ref_idx.size != REFERENCE_POINTS_EXPECTED:

    raise RuntimeError(
        (
            "reference set changed: "
            f"{ref_idx.size}"
        )
    )


if (
    ref_idx.min() < 0
    or
    ref_idx.max() >= npoint
):

    raise RuntimeError(
        "invalid reference-point index"
    )


# ======================================================================
# LOS factor / sign
#
# Frozen P15-1 convention:
#
# phi_pypsds(t) = phase(SLC_t) - phase(SLC_ref)
#
# toward-satellite displacement:
#
# + lambda/(4*pi) * phi
# ======================================================================

pars = read_par(
    RSLC_PAR
)


radar_frequency = par_scalar(
    pars,
    (
        "radar_frequency",
    ),
)


wavelength = (
    C0
    /
    radar_frequency
)


los_factor_m_per_rad = (
    wavelength
    /
    (
        4.0
        *
        np.pi
    )
)


if not (
    0.05
    <
    wavelength
    <
    0.06
):

    raise RuntimeError(
        (
            "unexpected Sentinel-1 wavelength: "
            f"{wavelength}"
        )
    )


# ======================================================================
# First pass:
#
# Construct corrected RAW phase only for the 607 reference points.
#
# phi_raw = preSCN - SCN
#
# Then restore temporal datum:
#
# phi_t = phi_raw - phi_raw(epoch0)
#
# Spatial reference for every epoch is median of same 607 points.
# ======================================================================

ref_pre = np.asarray(
    pre[
        ref_idx,
        :
    ],
    dtype=np.float64,
)


ref_scn = np.asarray(
    scn[
        ref_idx,
        :
    ],
    dtype=np.float64,
)


ref_raw = (
    ref_pre
    -
    ref_scn
)


ref_time = (
    ref_raw
    -
    ref_raw[
        :,
        tref0
    ][
        :,
        None
    ]
)


spatial_reference_median = np.median(
    ref_time,
    axis=0,
)


# First acquisition must remain exactly zero.
if spatial_reference_median[
    tref0
] != 0.0:

    raise RuntimeError(
        (
            "reference median at temporal "
            "reference is not exactly zero: "
            f"{spatial_reference_median[tref0]}"
        )
    )


# ======================================================================
# Atomic output preparation
# ======================================================================

for p in (
    PHASE_TMP,
    LOS_M_TMP,
    LOS_MM_TMP,
):

    if p.exists():
        p.unlink()


phase_out = np.lib.format.open_memmap(
    PHASE_TMP,
    mode="w+",
    dtype=np.float32,
    shape=(
        npoint,
        nepoch,
    ),
)


los_m_out = np.lib.format.open_memmap(
    LOS_M_TMP,
    mode="w+",
    dtype=np.float32,
    shape=(
        npoint,
        nepoch,
    ),
)


los_mm_out = np.lib.format.open_memmap(
    LOS_MM_TMP,
    mode="w+",
    dtype=np.float32,
    shape=(
        npoint,
        nepoch,
    ),
)


# ======================================================================
# Production materialization
# ======================================================================

t0 = time.perf_counter()


raw_scn_correction_ss = 0.0
raw_scn_correction_n = 0
raw_scn_correction_max = 0.0


phase_abs_max = 0.0
los_abs_max_m = 0.0


# Sample precision QA accumulators
sample_phase_storage_max = 0.0
sample_los_storage_max = 0.0


for start in range(
    0,
    npoint,
    CHUNK,
):

    stop = min(
        start
        +
        CHUNK,
        npoint,
    )


    pre64 = np.asarray(
        pre[
            start:stop,
            :
        ],
        dtype=np.float64,
    )


    scn64 = np.asarray(
        scn[
            start:stop,
            :
        ],
        dtype=np.float64,
    )


    # ----------------------------------------------------------
    # StaMPS correction:
    #
    # preSCN - ph_scn_slave
    # ----------------------------------------------------------

    raw = (
        pre64
        -
        scn64
    )


    # SCN magnitude QA
    raw_scn_correction_ss += float(
        np.sum(
            scn64
            *
            scn64
        )
    )

    raw_scn_correction_n += int(
        scn64.size
    )

    raw_scn_correction_max = max(
        raw_scn_correction_max,
        float(
            np.max(
                np.abs(
                    scn64
                )
            )
        ),
    )


    # ----------------------------------------------------------
    # Restore frozen temporal datum:
    # 20141006 = 0 for every point.
    # ----------------------------------------------------------

    temporal = (
        raw
        -
        raw[
            :,
            tref0
        ][
            :,
            None
        ]
    )


    # ----------------------------------------------------------
    # Restore frozen spatial datum:
    # median of same 607-point reference = 0 each epoch.
    # ----------------------------------------------------------

    final64 = (
        temporal
        -
        spatial_reference_median[
            None,
            :
        ]
    )


    # Enforce exact temporal datum after arithmetic.
    final64[
        :,
        tref0
    ] = 0.0


    los64 = (
        final64
        *
        los_factor_m_per_rad
    )


    phase32 = final64.astype(
        np.float32
    )


    los_m32 = los64.astype(
        np.float32
    )


    los_mm32 = (
        los64
        *
        1000.0
    ).astype(
        np.float32
    )


    phase_out[
        start:stop,
        :
    ] = phase32


    los_m_out[
        start:stop,
        :
    ] = los_m32


    los_mm_out[
        start:stop,
        :
    ] = los_mm32


    # ----------------------------------------------------------
    # float32-storage precision gates
    # ----------------------------------------------------------

    pd = float(
        np.max(
            np.abs(
                phase32.astype(
                    np.float64
                )
                -
                final64
            )
        )
    )


    ld = float(
        np.max(
            np.abs(
                los_m32.astype(
                    np.float64
                )
                -
                los64
            )
        )
    )


    sample_phase_storage_max = max(
        sample_phase_storage_max,
        pd,
    )


    sample_los_storage_max = max(
        sample_los_storage_max,
        ld,
    )


    phase_abs_max = max(
        phase_abs_max,
        float(
            np.max(
                np.abs(
                    final64
                )
            )
        ),
    )


    los_abs_max_m = max(
        los_abs_max_m,
        float(
            np.max(
                np.abs(
                    los64
                )
            )
        ),
    )


    print(
        "[FINAL LOS] "
        f"{stop:,}/{npoint:,} "
        f"({100*stop/npoint:.1f}%)",
        flush=True,
    )


phase_out.flush()
los_m_out.flush()
los_mm_out.flush()


materialization_seconds = (
    time.perf_counter()
    -
    t0
)


if (
    sample_phase_storage_max
    >
    PHASE_STORAGE_TOL_RAD
):

    raise RuntimeError(
        (
            "float32 phase storage precision "
            f"failed: {sample_phase_storage_max}"
        )
    )


if (
    sample_los_storage_max
    >
    LOS_STORAGE_TOL_M
):

    raise RuntimeError(
        (
            "float32 LOS storage precision "
            f"failed: {sample_los_storage_max}"
        )
    )


del phase_out
del los_m_out
del los_mm_out


# ======================================================================
# Publish atomically
# ======================================================================

os.replace(
    PHASE_TMP,
    PHASE_OUT,
)


os.replace(
    LOS_M_TMP,
    LOS_M_OUT,
)


os.replace(
    LOS_MM_TMP,
    LOS_MM_OUT,
)


# ======================================================================
# Final hard QA
# ======================================================================

phase_final = np.load(
    PHASE_OUT,
    mmap_mode="r",
)


los_m = np.load(
    LOS_M_OUT,
    mmap_mode="r",
)


los_mm = np.load(
    LOS_MM_OUT,
    mmap_mode="r",
)


if (
    phase_final.shape
    !=
    (
        npoint,
        nepoch,
    )
):

    raise RuntimeError(
        "final phase shape failed"
    )


finite_phase = float(
    np.mean(
        np.isfinite(
            phase_final
        )
    )
)


finite_los = float(
    np.mean(
        np.isfinite(
            los_m
        )
    )
)


if (
    finite_phase != 1.0
    or
    finite_los != 1.0
):

    raise RuntimeError(
        "final output contains non-finite values"
    )


# ---------------------------------------------------------------
# Temporal datum:
# first acquisition must be bit-exact zero.
# ---------------------------------------------------------------

epoch0_max = float(
    np.max(
        np.abs(
            phase_final[
                :,
                tref0
            ]
        )
    )
)


los_epoch0_max = float(
    np.max(
        np.abs(
            los_m[
                :,
                tref0
            ]
        )
    )
)


if epoch0_max != EPOCH0_TOL_RAD:

    raise RuntimeError(
        (
            "final temporal reference failed: "
            f"{epoch0_max}"
        )
    )


if los_epoch0_max != 0.0:

    raise RuntimeError(
        (
            "LOS temporal reference failed: "
            f"{los_epoch0_max}"
        )
    )


# ---------------------------------------------------------------
# Spatial datum:
# same 607 reference points, per-epoch median = 0.
# ---------------------------------------------------------------

ref_phase_final = np.asarray(
    phase_final[
        ref_idx,
        :
    ],
    dtype=np.float64,
)


ref_median_final = np.median(
    ref_phase_final,
    axis=0,
)


ref_median_max = float(
    np.max(
        np.abs(
            ref_median_final
        )
    )
)


if (
    ref_median_max
    >
    REF_MEDIAN_TOL_RAD
):

    raise RuntimeError(
        (
            "final reference median failed: "
            f"{ref_median_max}"
        )
    )


ref_los_median = np.median(
    np.asarray(
        los_mm[
            ref_idx,
            :
        ],
        dtype=np.float64,
    ),
    axis=0,
)


ref_los_median_max_mm = float(
    np.max(
        np.abs(
            ref_los_median
        )
    )
)


# ---------------------------------------------------------------
# LOS sign/factor parity
# ---------------------------------------------------------------

rng = np.random.default_rng(
    20260824
)


nsample = min(
    100000,
    npoint,
)


sample_idx = rng.choice(
    npoint,
    size=nsample,
    replace=False,
)


p_sample = np.asarray(
    phase_final[
        sample_idx,
        :
    ],
    dtype=np.float64,
)


l_sample = np.asarray(
    los_m[
        sample_idx,
        :
    ],
    dtype=np.float64,
)


los_factor_parity = float(
    np.max(
        np.abs(
            l_sample
            -
            p_sample
            *
            los_factor_m_per_rad
        )
    )
)


# ---------------------------------------------------------------
# Product statistics
# ---------------------------------------------------------------

los_sample_mm = (
    l_sample
    *
    1000.0
)


los_abs_q_mm = np.percentile(
    np.abs(
        los_sample_mm
    ),
    [
        50,
        95,
        99,
        99.9,
    ],
)


phase_abs_q = np.percentile(
    np.abs(
        p_sample
    ),
    [
        50,
        95,
        99,
        99.9,
    ],
)


scn_rms = float(
    np.sqrt(
        raw_scn_correction_ss
        /
        raw_scn_correction_n
    )
)


# ======================================================================
# Manifest
# ======================================================================

manifest = {

    "status":
        "PASS_FINAL_REFERENCED_LOS_TIMESERIES",

    "formula":
        {
            "scn_correction":
                (
                    "phi_raw = "
                    "phi_preSCN - ph_scn_slave"
                ),

            "temporal_reference":
                (
                    "phi_t = phi_raw - "
                    "phi_raw[:,20141006]"
                ),

            "spatial_reference":
                (
                    "phi_final = phi_t - "
                    "median(phi_t[607 reference points], epoch)"
                ),

            "los":
                (
                    "d_LOS_toward = "
                    "+lambda/(4*pi) * phi_final"
                ),
        },

    "scientific_contract":
        {
            "temporal_reference_date":
                TEMPORAL_REFERENCE_DATE,

            "temporal_reference_index_0based":
                tref0,

            "spatial_reference_method":
                "median",

            "spatial_reference_points":
                int(
                    ref_idx.size
                ),

            "geometric_master_date":
                GEOMETRIC_MASTER_DATE,

            "geometric_master_index_0based":
                master0,

            "los_positive_direction":
                "toward_satellite",

            "radar_frequency_hz":
                radar_frequency,

            "wavelength_m":
                wavelength,

            "los_factor_m_per_rad":
                los_factor_m_per_rad,

            "los_factor_mm_per_rad":
                (
                    los_factor_m_per_rad
                    *
                    1000.0
                ),
        },

    "hard_qa":
        {
            "finite_phase_fraction":
                finite_phase,

            "finite_los_fraction":
                finite_los,

            "epoch0_phase_max_abs_rad":
                epoch0_max,

            "epoch0_los_max_abs_m":
                los_epoch0_max,

            "reference_phase_median_max_abs_rad":
                ref_median_max,

            "reference_los_median_max_abs_mm":
                ref_los_median_max_mm,

            "los_factor_sample_max_abs_error_m":
                los_factor_parity,

            "float32_phase_storage_max_error_rad":
                sample_phase_storage_max,

            "float32_los_storage_max_error_m":
                sample_los_storage_max,
        },

    "correction_statistics":
        {
            "scn_rms_rad":
                scn_rms,

            "scn_max_abs_rad":
                raw_scn_correction_max,

            "final_phase_max_abs_rad":
                phase_abs_max,

            "final_los_max_abs_m":
                los_abs_max_m,
        },

    "sample_statistics":
        {
            "points":
                nsample,

            "abs_phase_p50_p95_p99_p999_rad":
                [
                    float(x)
                    for x in phase_abs_q
                ],

            "abs_los_p50_p95_p99_p999_mm":
                [
                    float(x)
                    for x in los_abs_q_mm
                ],
        },

    "performance":
        {
            "materialization_seconds":
                materialization_seconds,

            "point_epochs_per_second":
                (
                    npoint
                    *
                    nepoch
                    /
                    materialization_seconds
                ),
        },

    "inputs":
        {
            "pre_scn_phase":
                str(
                    PRE
                ),

            "ph_scn_slave":
                str(
                    SCN
                ),

            "reference_indices":
                str(
                    REF_FILE
                ),
        },

    "outputs":
        {
            "final_phase_rad":
                str(
                    PHASE_OUT
                ),

            "los_toward_m":
                str(
                    LOS_M_OUT
                ),

            "los_toward_mm":
                str(
                    LOS_MM_OUT
                ),

            "dtype":
                "float32",
        },

    "upstream_modified":
        False,

    "next":
        (
            "P15-7 final LOS velocity / "
            "product geocoding and TIFF export"
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
print("P15-6C FINAL REFERENCED LOS TIMESERIES")
print("=" * 96)

print(
    "points / acquisitions           :",
    f"{npoint:,} / {nepoch}",
)

print(
    "temporal reference              :",
    (
        f"{TEMPORAL_REFERENCE_DATE} "
        f"(0b={tref0})"
    ),
)

print(
    "spatial reference               :",
    f"{ref_idx.size} points / median",
)

print(
    "geometric master                :",
    (
        f"{GEOMETRIC_MASTER_DATE} "
        f"(0b={master0})"
    ),
)

print()

print(
    "radar frequency                 :",
    f"{radar_frequency:.6f} Hz",
)

print(
    "wavelength                      :",
    f"{wavelength:.15f} m",
)

print(
    "LOS factor                      :",
    (
        f"+{los_factor_m_per_rad:.15e} "
        "m/rad"
    ),
)

print(
    "LOS positive                    :",
    "toward satellite",
)

print()

print(
    "epoch0 phase max |rad|          :",
    f"{epoch0_max:.12e}",
)

print(
    "epoch0 LOS max |m|              :",
    f"{los_epoch0_max:.12e}",
)

print(
    "reference median max |rad|      :",
    f"{ref_median_max:.12e}",
)

print(
    "reference LOS median max |mm|   :",
    f"{ref_los_median_max_mm:.12e}",
)

print(
    "LOS-factor parity max |m|       :",
    f"{los_factor_parity:.12e}",
)

print()

print(
    "phase float32 storage max err   :",
    f"{sample_phase_storage_max:.12e} rad",
)

print(
    "LOS float32 storage max err     :",
    f"{sample_los_storage_max:.12e} m",
)

print()

print(
    "SCN RMS                         :",
    f"{scn_rms:.6f} rad",
)

print(
    "SCN max |rad|                   :",
    f"{raw_scn_correction_max:.6f}",
)

print(
    "|final phase| p50/95/99/999    :",
    phase_abs_q,
)

print(
    "|final LOS| p50/95/99/999 mm   :",
    los_abs_q_mm,
)

print()

print(
    "materialization seconds         :",
    f"{materialization_seconds:.6f}",
)

print(
    "throughput                      :",
    (
        f"{npoint*nepoch/materialization_seconds:,.0f} "
        "point-epochs/s"
    ),
)

print()

print(
    "final phase                     :",
    PHASE_OUT,
)

print(
    "LOS toward satellite [m]        :",
    LOS_M_OUT,
)

print(
    "LOS toward satellite [mm]       :",
    LOS_MM_OUT,
)

print(
    "upstream modified               :",
    False,
)

print(
    "manifest                        :",
    MANIFEST,
)

print("=" * 96)

print(
    "P15-6C FINAL RESULT: "
    "PASS_FINAL_REFERENCED_LOS_TIMESERIES"
)

print("=" * 96)
