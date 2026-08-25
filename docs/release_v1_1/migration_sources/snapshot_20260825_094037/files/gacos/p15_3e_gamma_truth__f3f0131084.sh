#!/usr/bin/env bash
set -euo pipefail

DEM=/home/ubuntu/Downloads/DEM_prep
GG=/home/ubuntu/Downloads/psds/output/processing/gacos_geometry

MLI_PAR="$DEM/20151212_4_1.vv.mli.par"
DEM_PAR="$DEM/N38E023.dem_par"
DEM_SEG="$DEM/N38E023.dem"
LT="$DEM/N38E023.20151212.lt_fine"

INC_MAP="$DEM/N38E023.20151212.inc_ellipsoid_gamma"
INC_RDC="$DEM/20151212_4_1.inc_ellipsoid_gamma_rdc"

INC_PT="$GG/incidence_ellipsoid_gamma_rad.gamma_pt"

GC=/home/ubuntu/software/GAMMA_SOFTWARE/DIFF/bin/gc_map2
GEOCODE=/home/ubuntu/software/GAMMA_SOFTWARE/DIFF/bin/geocode
DATA2PT=/home/ubuntu/software/GAMMA_SOFTWARE/IPTA/bin/data2pt

echo "================================================================================"
echo "P15-3E GAMMA inc_flg=1 TRUTH VALIDATION"
echo
echo "VALIDATION ONLY"
echo "NO PHASE MODIFICATION"
echo "NO GACOS CORRECTION"
echo "================================================================================"

# ----------------------------------------------------------------------
# Geometry contract
# ----------------------------------------------------------------------

WIDTH=$(awk '$1=="width:" {print $2}' "$DEM_PAR")
NLINES=$(awk '$1=="nlines:" {print $2}' "$DEM_PAR")

echo "DEM geometry : ${NLINES} x ${WIDTH}"

if [[ "$WIDTH" != "793" || "$NLINES" != "747" ]]; then
    echo "FAIL: unexpected DEM segment geometry"
    exit 1
fi


# ----------------------------------------------------------------------
# GAMMA gc_map2 inc_flg=1
#
# Argument 26 = inc_flg = 1
#
# Important:
#   only incidence is requested;
#   no lookup, DEM segment, layover, sim_sar, etc.
# ----------------------------------------------------------------------

ARGS=(
    "$MLI_PAR"    # 1 MLI_par
    "$DEM_PAR"    # 2 DEM_par
    "$DEM_SEG"    # 3 DEM
    "-"           # 4 DEM_seg_par
    "-"           # 5 DEM_seg
    "-"           # 6 lookup_table
    "-"           # 7 lat_ovr
    "-"           # 8 lon_ovr
    "-"           # 9 ls_map
    "-"           # 10 ls_map_rdc
    "$INC_MAP"    # 11 inc
    "-"           # 12 res
    "-"           # 13 offnadir
    "-"           # 14 sim_sar
    "-"           # 15 u
    "-"           # 16 v
    "-"           # 17 psi
    "-"           # 18 pix
    "-"           # 19 r_ovr
    "-"           # 20 az_dec
    "-"           # 21 mask
    "-"           # 22 frame
    "-"           # 23 ls_scaling
    "-"           # 24 DIFF_par
    "-"           # 25 ref_flg
    "1"           # 26 inc_flg = ellipsoid incidence
)

if [[ ${#ARGS[@]} -ne 26 ]]; then
    echo "FAIL: gc_map2 argument count = ${#ARGS[@]}, expected 26"
    exit 1
fi

echo
echo "------------------------------------------------------------"
echo "GAMMA gc_map2 truth calculation"
echo "------------------------------------------------------------"

/usr/bin/time -f \
"GAMMA gc_map2 wall=%e s cpu=%P maxRSS=%M KB" \
"$GC" "${ARGS[@]}"


# ----------------------------------------------------------------------
# Validate map FLOAT
# ----------------------------------------------------------------------

EXPECTED_MAP_BYTES=$((793 * 747 * 4))
MAP_BYTES=$(stat -c%s "$INC_MAP")

echo
echo "inc map bytes : $MAP_BYTES"

if [[ "$MAP_BYTES" -ne "$EXPECTED_MAP_BYTES" ]]; then
    echo "FAIL: GAMMA incidence map size"
    exit 1
fi


# ----------------------------------------------------------------------
# Existing refined LUT:
# map geometry -> 4:1 radar geometry
# ----------------------------------------------------------------------

echo
echo "------------------------------------------------------------"
echo "geocode ellipsoid incidence -> 4:1 RDC"
echo "------------------------------------------------------------"

/usr/bin/time -f \
"geocode wall=%e s cpu=%P maxRSS=%M KB" \
"$GEOCODE" \
    "$LT" \
    "$INC_MAP" \
    793 \
    "$INC_RDC" \
    500 \
    600 \
    0 \
    0

RDC_BYTES=$(stat -c%s "$INC_RDC")

if [[ "$RDC_BYTES" -ne 1200000 ]]; then
    echo "FAIL: incidence RDC size = $RDC_BYTES"
    exit 1
fi


# ----------------------------------------------------------------------
# 4:1 RDC -> strict points
# ----------------------------------------------------------------------

echo
echo "------------------------------------------------------------"
echo "data2pt -> 881315 strict points"
echo "------------------------------------------------------------"

/usr/bin/time -f \
"data2pt wall=%e s cpu=%P maxRSS=%M KB" \
"$DATA2PT" \
    "$INC_RDC" \
    "$MLI_PAR" \
    "$GG/strict_points.plist" \
    /home/ubuntu/Downloads/RSLC/20151212.rslc.par \
    "$INC_PT" \
    1 \
    2

PT_BYTES=$(stat -c%s "$INC_PT")

if [[ "$PT_BYTES" -ne 3525260 ]]; then
    echo "FAIL: strict-point incidence size = $PT_BYTES"
    exit 1
fi


# ----------------------------------------------------------------------
# FAST vs GAMMA
# ----------------------------------------------------------------------

python - <<'PY'
from pathlib import Path
import numpy as np

ROOT = Path(
    "/home/ubuntu/Downloads/psds/output/processing/gacos_geometry"
)

fast = np.load(
    ROOT / "incidence_ellipsoid_fast_rad.npy"
).astype(np.float64)

gamma = np.fromfile(
    ROOT / "incidence_ellipsoid_gamma_rad.gamma_pt",
    dtype=">f4",
).astype(np.float64)

if fast.size != gamma.size:
    raise SystemExit(
        f"FAIL: size mismatch FAST={fast.size}, GAMMA={gamma.size}"
    )

valid = (
    np.isfinite(fast)
    & np.isfinite(gamma)
    & (fast > 0)
    & (fast < np.pi/2)
    & (gamma > 0)
    & (gamma < np.pi/2)
)

coverage = float(valid.mean())

fd = np.degrees(fast[valid])
gd = np.degrees(gamma[valid])

diff = fd - gd
adiff = np.abs(diff)

rms = float(
    np.sqrt(
        np.mean(diff * diff)
    )
)

qdiff = np.percentile(
    diff,
    [1, 5, 50, 95, 99],
)

qa = np.percentile(
    adiff,
    [50, 95, 99, 100],
)

# This is the quantity GACOS actually uses.
mf = 1.0 / np.cos(fast[valid])
mg = 1.0 / np.cos(gamma[valid])

mapping_rel = (
    (mf - mg)
    /
    mg
)

mapping_ppm = (
    mapping_rel
    *
    1.0e6
)

print()
print("=" * 88)
print("P15-3E FAST vs GAMMA inc_flg=1")
print("=" * 88)

print(
    "points                    :",
    f"{fast.size:,}"
)

print(
    "valid comparison          :",
    f"{100*coverage:.6f}%"
)

print()
print(
    "FAST incidence p01/p50/p99:",
    np.percentile(
        fd,
        [1, 50, 99]
    )
)

print(
    "GAMMA incidence p01/p50/p99:",
    np.percentile(
        gd,
        [1, 50, 99]
    )
)

print()
print(
    "FAST-GAMMA diff p01/p05/p50/p95/p99 deg:"
)

print(
    " ",
    qdiff
)

print(
    "RMS difference deg       :",
    f"{rms:.9f}"
)

print(
    "|diff| p50/p95/p99/max   :",
    " / ".join(
        f"{x:.9f}"
        for x in qa
    ),
    "deg"
)

print()
print(
    "mapping-factor error p01/p50/p99 ppm:",
    np.percentile(
        mapping_ppm,
        [1, 50, 99]
    )
)

print(
    "mapping-factor max abs ppm:",
    f"{np.max(np.abs(mapping_ppm)):.3f}"
)


# ------------------------------------------------------------
# Acceptance gates
# ------------------------------------------------------------

if coverage < 0.99999:
    raise SystemExit(
        "FAIL_FAST_GAMMA_COVERAGE"
    )

# Production-equivalence gate.
#
# 0.01 deg RMS is already negligible for ZTD->LOS,
# but retain a considerably stricter p99 gate as well.
if rms > 0.01:
    raise SystemExit(
        "FAIL_FAST_GAMMA_RMS"
    )

if qa[2] > 0.02:
    raise SystemExit(
        "FAIL_FAST_GAMMA_P99"
    )

if qa[3] > 0.05:
    raise SystemExit(
        "FAIL_FAST_GAMMA_MAX"
    )


# Strong acceptance classification
if (
    rms <= 0.002
    and
    qa[2] <= 0.005
    and
    qa[3] <= 0.02
):
    status = (
        "PASS_FAST_INCIDENCE_STRONG"
    )
else:
    status = (
        "PASS_FAST_INCIDENCE_PRODUCTION_EQUIVALENT"
    )


print()
print("=" * 88)
print(
    "P15-3E FINAL RESULT:",
    status
)
print("=" * 88)
PY
