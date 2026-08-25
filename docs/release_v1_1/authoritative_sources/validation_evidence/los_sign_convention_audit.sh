#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ubuntu/software/pyPSDS-GAMMA-v1.0
PROJECT=/home/ubuntu/Downloads/psds
OUT=$PROJECT/output
CFG=$PROJECT/production.yaml
LOGDIR=$PROJECT/production_logs

STAMP=$(date +%Y%m%d_%H%M%S)

JSON_REPORT=$LOGDIR/P15_1_los_sign_convention_${STAMP}.json
TXT_REPORT=$LOGDIR/P15_1_los_sign_convention_${STAMP}.txt

mkdir -p "$LOGDIR"

cd "$REPO"

echo "================================================================================"
echo " P15-1 LOS SIGN CONVENTION AUDIT"
echo
echo " READ ONLY"
echo " NO SOURCE MODIFICATION"
echo " NO PRODUCTION PRODUCT MODIFICATION"
echo " NO GAMMA EXECUTION"
echo " NO LOS PRODUCT CREATED"
echo "================================================================================"

python - \
    "$REPO" \
    "$PROJECT" \
    "$CFG" \
    "$JSON_REPORT" \
    "$TXT_REPORT" <<'PY'

from pathlib import Path
import json
import math
import re
import sys

import numpy as np
import yaml


REPO = Path(sys.argv[1]).resolve()
PROJECT = Path(sys.argv[2]).resolve()
CFG = Path(sys.argv[3]).resolve()

JSON_REPORT = Path(sys.argv[4]).resolve()
TXT_REPORT = Path(sys.argv[5]).resolve()


errors = []
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


def wrap(x):

    return np.arctan2(
        np.sin(x),
        np.cos(x),
    )


print()
print("=" * 96)
print("A. P15-0 ACCEPTED PHYSICAL PARAMETERS")
print("=" * 96)


# ============================================================================
# 1. Read accepted P15-0 report
# ============================================================================

reports = sorted(
    (
        PROJECT
        / "production_logs"
    ).glob(
        "P15_0_final_chain_input_audit_*.json"
    )
)


check(
    "P15-0 report found",
    len(reports) > 0,
    len(reports),
)


if not reports:

    raise SystemExit(1)


p15_0_path = reports[-1]

p15_0 = json.loads(
    p15_0_path.read_text(
        encoding="utf-8"
    )
)


check(
    "P15-0 status",
    p15_0.get("status")
    ==
    "PASS_INPUT_AUDIT",
    p15_0.get("status"),
)


wavelength = (
    p15_0
    .get(
        "sensor",
        {}
    )
    .get(
        "resolved_wavelength_m"
    )
)


check(
    "wavelength available",
    wavelength is not None,
    wavelength,
)


wavelength = float(
    wavelength
)


check(
    "Sentinel-1 C-band wavelength range",
    0.05
    <
    wavelength
    <
    0.06,
    f"{wavelength:.12f} m",
)


m_per_rad = (
    wavelength
    /
    (
        4.0
        *
        math.pi
    )
)


print()
print(
    f"wavelength               : "
    f"{wavelength:.12f} m"
)

print(
    f"lambda/(4pi)             : "
    f"{m_per_rad:.12e} m/rad"
)

print(
    f"                         : "
    f"{1000*m_per_rad:.6f} mm/rad"
)


# ============================================================================
# 2. Config geometric phase-correction sign
#
# This is NOT the LOS displacement sign.
# We record it only to ensure the accepted production configuration is known.
# ============================================================================

with CFG.open(
    encoding="utf-8"
) as f:

    cfg = yaml.safe_load(
        f
    )


pc = cfg.get(
    "phase_correction",
    {}
)


apply_sign = float(
    pc.get(
        "apply_sign",
        1.0,
    )
)


print()
print(
    "phase_correction.apply_sign:",
    apply_sign,
)


# ============================================================================
# B. Static source convention
# ============================================================================

print()
print("=" * 96)
print("B. STATIC SOURCE PHASE CONVENTION")
print("=" * 96)


stack_py = (
    REPO
    / "pypsds"
    / "points"
    / "stack.py"
)


coh_py = (
    REPO
    / "pypsds"
    / "phase_linking"
    / "coherence.py"
)


emi_py = (
    REPO
    / "pypsds"
    / "phase_linking"
    / "emi.py"
)


for p in (
    stack_py,
    coh_py,
    emi_py,
):

    check(
        f"source exists: {p.relative_to(REPO)}",
        p.is_file(),
        p,
    )


stack_text = stack_py.read_text(
    encoding="utf-8"
)

coh_text = coh_py.read_text(
    encoding="utf-8"
)

emi_text = emi_py.read_text(
    encoding="utf-8"
)


# PS:
#
# phi_ps(t) = arg(SLC_t * conj(SLC_ref))
#

ps_direct = bool(
    re.search(
        r"np\.angle\s*\("
        r"\s*x\s*\*\s*"
        r"np\.conj\s*\(\s*ref\s*\)"
        r"\s*\)",
        stack_text,
        re.S,
    )
)


check(
    "PS phase = arg(SLC_t * conj(SLC_ref))",
    ps_direct,
    "current minus reference SLC phase",
)


# DS coherence:
#
# C_ij = SLC_i * conj(SLC_j)
#

ds_direct = bool(
    re.search(
        r"zvec\s*\[\s*i\s*\]"
        r"\s*\*\s*"
        r"np\.conj\s*\("
        r"\s*zvec\s*\[\s*j\s*\]"
        r"\s*\)",
        coh_text,
        re.S,
    )
)


check(
    "DS coherence = SLC_i * conj(SLC_j)",
    ds_direct,
    "C_ij phase = phi_i - phi_j",
)


# EMI temporal-coherence prediction should use the same orientation.

emi_direct = (
    "phase[:, ii]"
    in emi_text
    and
    "np.conj(phase[:, jj])"
    in emi_text
)


check(
    "EMI predicted pair orientation",
    emi_direct,
    "phase_i * conj(phase_j)",
)


# ============================================================================
# C. Synthetic PS sign test
# ============================================================================

print()
print("=" * 96)
print("C. SYNTHETIC PS SIGN TEST")
print("=" * 96)


from pypsds.points.stack import (
    _reference_ps_phase,
)


# Deliberately asymmetric, all inside (-pi, pi)
# so interpretation is obvious.

true_phase = np.asarray(
    [
        0.37,
        0.82,
        -0.28,
        1.13,
        -0.64,
    ],
    dtype=np.float32,
)


synthetic_stack = np.exp(
    1j
    *
    true_phase
).astype(
    np.complex64
)[:, None, None]


ps_out = _reference_ps_phase(
    synthetic_stack,
    np.asarray(
        [0],
        dtype=np.int32,
    ),
    np.asarray(
        [0],
        dtype=np.int32,
    ),
    0,
)[0]


expected = wrap(
    true_phase
    -
    true_phase[0]
)


opposite = wrap(
    -(
        true_phase
        -
        true_phase[0]
    )
)


ps_direct_error = float(
    np.max(
        np.abs(
            wrap(
                ps_out
                -
                expected
            )
        )
    )
)


ps_opposite_error = float(
    np.max(
        np.abs(
            wrap(
                ps_out
                -
                opposite
            )
        )
    )
)


print(
    "direct-orientation max error : "
    f"{ps_direct_error:.3e} rad"
)

print(
    "opposite-orientation error   : "
    f"{ps_opposite_error:.3e} rad"
)


check(
    "PS dynamic orientation",
    ps_direct_error
    <
    1.0e-6
    and
    ps_opposite_error
    >
    1.0e-2,
    (
        f"direct={ps_direct_error:.3e}, "
        f"opposite={ps_opposite_error:.3e}"
    ),
)


# ============================================================================
# D. Synthetic DS / EMI sign test
# ============================================================================

print()
print("=" * 96)
print("D. SYNTHETIC DS / EMI SIGN TEST")
print("=" * 96)


from pypsds.phase_linking.emi import (
    image_pairs,
    robust_emi_batch,
)


n = true_phase.size

pairs = image_pairs(
    n
)


# Exact coherence from the code's mathematical convention:
#
# C_ij = exp(i * (phi_i - phi_j))
#

coh = np.empty(
    (
        1,
        pairs.shape[0],
    ),
    dtype=np.complex64,
)


for q, (i, j) in enumerate(
    pairs
):

    coh[
        0,
        q
    ] = np.exp(
        1j
        *
        (
            float(
                true_phase[i]
            )
            -
            float(
                true_phase[j]
            )
        )
    )


(
    phase_unit,
    estimator,
    emi_eig,
    evd_eig,
    gamma_min,
) = robust_emi_batch(
    coh,
    n_images=n,
    pairs=pairs,
    reference_idx=0,
)


ds_out = np.angle(
    phase_unit[
        0
    ]
)


ds_direct_error = float(
    np.max(
        np.abs(
            wrap(
                ds_out
                -
                expected
            )
        )
    )
)


ds_opposite_error = float(
    np.max(
        np.abs(
            wrap(
                ds_out
                -
                opposite
            )
        )
    )
)


print(
    "estimator                  :",
    int(
        estimator[0]
    ),
)

print(
    "direct-orientation max error:",
    f"{ds_direct_error:.3e} rad",
)

print(
    "opposite-orientation error  :",
    f"{ds_opposite_error:.3e} rad",
)


check(
    "DS/EMI dynamic orientation",
    ds_direct_error
    <
    1.0e-5
    and
    ds_opposite_error
    >
    1.0e-2,
    (
        f"direct={ds_direct_error:.3e}, "
        f"opposite={ds_opposite_error:.3e}"
    ),
)


check(
    "PS and DS phase orientation identical",
    float(
        np.max(
            np.abs(
                wrap(
                    ps_out
                    -
                    ds_out
                )
            )
        )
    )
    <
    1.0e-5,
)


# ============================================================================
# E. Project mathematical convention
# ============================================================================

print()
print("=" * 96)
print("E. PROJECT PHASE CONVENTION")
print("=" * 96)


if (
    ps_direct
    and
    ds_direct
    and
    ps_direct_error < 1e-6
    and
    ds_direct_error < 1e-5
):

    project_phase_convention = (
        "phi_pypsds(t) = "
        "arg(SLC_t * conj(SLC_reference))"
    )

else:

    project_phase_convention = (
        "UNRESOLVED"
    )


check(
    "project phase direction resolved",
    project_phase_convention
    !=
    "UNRESOLVED",
    project_phase_convention,
)


print()
print(
    "pyPSDS phase convention:"
)

print(
    "  phi(t) = phase(SLC_t) "
    "- phase(SLC_reference)"
)


# ============================================================================
# F. GAMMA-compatible LOS mapping
#
# External GAMMA convention frozen for pyPSDS:
#
# Standard GAMMA interferogram:
#
#   I(ref,t) = SLC_ref * conj(SLC_t)
#
# Therefore:
#
#   phi_gamma(ref,t)
#       =
#   - phi_pypsds(t)
#
# GAMMA LOS displacement convention:
#
#   positive = motion toward satellite
#   negative = motion away from satellite
#
# Hence:
#
#   d_LOS_toward
#       =
#   - lambda/(4*pi) * phi_gamma
#       =
#   + lambda/(4*pi) * phi_pypsds
#
# and slant-range change, positive away:
#
#   delta_R_away
#       =
#   - lambda/(4*pi) * phi_pypsds
#
# ============================================================================

print()
print("=" * 96)
print("F. FROZEN LOS SIGN CONVENTION")
print("=" * 96)


los_toward_factor = (
    +
    m_per_rad
)


range_away_factor = (
    -
    m_per_rad
)


print(
    "GAMMA-compatible LOS displacement:"
)

print(
    "  positive = toward satellite"
)

print(
    "  negative = away from satellite"
)

print()

print(
    "d_LOS_toward_m = "
    "+lambda/(4*pi) * phi_pypsds"
)

print(
    "factor            = "
    f"{los_toward_factor:+.12e} m/rad"
)

print()

print(
    "delta_range_away_m = "
    "-lambda/(4*pi) * phi_pypsds"
)

print(
    "factor             = "
    f"{range_away_factor:+.12e} m/rad"
)


check(
    "LOS factor sign frozen",
    los_toward_factor > 0,
    (
        f"{los_toward_factor:+.12e} "
        "m/rad"
    ),
)


# ============================================================================
# G. Sanity examples
# ============================================================================

print()
print("=" * 96)
print("G. SIGN SANITY EXAMPLES")
print("=" * 96)


for phase_example in (
    -1.0,
    +1.0,
):

    d_los_mm = (
        phase_example
        *
        los_toward_factor
        *
        1000.0
    )

    dr_mm = (
        phase_example
        *
        range_away_factor
        *
        1000.0
    )

    print(
        f"phi={phase_example:+.1f} rad  "
        f"=> LOS_toward={d_los_mm:+.6f} mm  "
        f"=> range_away={dr_mm:+.6f} mm"
    )


# ============================================================================
# H. Freeze report
# ============================================================================

print()
print("=" * 96)
print("H. FINAL DECISION")
print("=" * 96)


if errors:

    final_status = (
        "FAIL_SIGN_CONVENTION_AUDIT"
    )

    resolved = False

else:

    final_status = (
        "PASS_SIGN_CONVENTION_FROZEN"
    )

    resolved = True


report = {
    "format":
        "pyPSDS-GAMMA-P15-1-LOS-sign-convention-v1",

    "status":
        final_status,

    "source_P15_0_report":
        str(
            p15_0_path
        ),

    "sensor": {
        "wavelength_m":
            wavelength,

        "phase_to_displacement_scale_m_per_rad":
            m_per_rad,
    },

    "production_phase_correction": {
        "apply_sign":
            apply_sign,

        "note":
            (
                "Geometric/topographic phase-correction sign; "
                "not the LOS displacement sign."
            ),
    },

    "pypsds_phase_definition": {
        "formula":
            (
                "phi_pypsds(t) = "
                "arg(SLC_t * conj(SLC_reference))"
            ),

        "equivalent":
            (
                "phase(SLC_t) - "
                "phase(SLC_reference)"
            ),

        "PS_dynamic_test_max_error_rad":
            ps_direct_error,

        "DS_EMI_dynamic_test_max_error_rad":
            ds_direct_error,

        "PS_DS_same_orientation":
            True,
    },

    "gamma_interferogram_relation": {
        "gamma_pair_formula":
            (
                "phi_gamma(ref,t) = "
                "arg(SLC_ref * conj(SLC_t))"
            ),

        "relation_to_pypsds":
            (
                "phi_gamma(ref,t) = "
                "-phi_pypsds(t)"
            ),
    },

    "frozen_LOS_convention": {
        "name":
            "positive_toward_satellite",

        "positive_direction":
            "toward_satellite",

        "negative_direction":
            "away_from_satellite",

        "formula":
            (
                "d_LOS_toward_m = "
                "+lambda/(4*pi) * phi_pypsds"
            ),

        "factor_m_per_rad":
            los_toward_factor,

        "factor_mm_per_rad":
            (
                los_toward_factor
                *
                1000.0
            ),
    },

    "range_change_convention": {
        "name":
            "positive_away_from_satellite",

        "formula":
            (
                "delta_R_away_m = "
                "-lambda/(4*pi) * phi_pypsds"
            ),

        "factor_m_per_rad":
            range_away_factor,
    },

    "resolved":
        resolved,

    "checks":
        checks,

    "errors":
        errors,

    "next_step":
        (
            "P15-2_SCLA_RESIDUAL_DEM_AUDIT"
            if resolved
            else
            "STOP"
        ),
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
    "P15-1 LOS SIGN CONVENTION",
    "=" * 88,

    f"status                  : {final_status}",

    f"wavelength              : {wavelength:.12f} m",

    (
        "phase scale             : "
        f"{1000*m_per_rad:.6f} mm/rad"
    ),

    "",

    (
        "pyPSDS phase            : "
        "arg(SLC_t * conj(SLC_ref))"
    ),

    (
        "GAMMA pair phase        : "
        "arg(SLC_ref * conj(SLC_t))"
    ),

    (
        "relation                : "
        "phi_gamma = -phi_pypsds"
    ),

    "",

    (
        "FINAL LOS convention    : "
        "positive toward satellite"
    ),

    (
        "LOS formula             : "
        "d_LOS = +lambda/(4pi) * phi_pypsds"
    ),

    (
        "away-range formula      : "
        "dR = -lambda/(4pi) * phi_pypsds"
    ),

    "",

    (
        "next step               : "
        f"{report['next_step']}"
    ),

    "",

    f"JSON report             : {JSON_REPORT}",
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
        " P15-1 FINAL RESULT: FAIL"
    )
    print("=" * 96)

    raise SystemExit(1)


print()
print("=" * 96)
print(
    " P15-1 FINAL RESULT: "
    "PASS_SIGN_CONVENTION_FROZEN"
)
print("=" * 96)

PY

echo
echo "reports:"
echo "  $JSON_REPORT"
echo "  $TXT_REPORT"
