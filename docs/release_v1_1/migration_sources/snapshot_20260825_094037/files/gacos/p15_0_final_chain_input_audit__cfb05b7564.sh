#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ubuntu/software/pyPSDS-GAMMA-v1.0
PROJECT=/home/ubuntu/Downloads/psds
OUT=$PROJECT/output
PROC=$OUT/processing
CFG=$PROJECT/production.yaml
RSLC=/home/ubuntu/Downloads/RSLC
DEM=/home/ubuntu/Downloads/DEM_prep

STAMP=$(date +%Y%m%d_%H%M%S)
REPORT=$PROJECT/production_logs/P15_0_final_chain_input_audit_${STAMP}.json
TXT=$PROJECT/production_logs/P15_0_final_chain_input_audit_${STAMP}.txt

mkdir -p "$PROJECT/production_logs"

echo "================================================================================================"
echo " P15-0 FINAL DEFORMATION CHAIN INPUT AUDIT"
echo
echo " READ ONLY"
echo " NO SOURCE MODIFICATION"
echo " NO SCIENTIFIC PRODUCT MODIFICATION"
echo " NO GAMMA EXECUTION"
echo " NO SCLA / APS / GACOS APPLICATION"
echo "================================================================================================"

python - "$REPO" "$PROJECT" "$OUT" "$CFG" "$RSLC" "$DEM" "$REPORT" "$TXT" <<'PY'
from pathlib import Path
import json
import math
import os
import re
import sys
import yaml

import numpy as np

REPO    = Path(sys.argv[1]).resolve()
PROJECT = Path(sys.argv[2]).resolve()
OUT     = Path(sys.argv[3]).resolve()
CFG     = Path(sys.argv[4]).resolve()
RSLC    = Path(sys.argv[5]).resolve()
DEM     = Path(sys.argv[6]).resolve()
REPORT  = Path(sys.argv[7]).resolve()
TXT     = Path(sys.argv[8]).resolve()

PROC = OUT / "processing"

errors = []
warnings = []
checks = []
evidence = {}


def add_check(name, ok, detail=""):
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


def read_json(path):
    path = Path(path)

    if not path.is_file():
        raise RuntimeError(
            f"Missing JSON: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def gamma_par_dict(path):
    """
    Lightweight GAMMA .par parser.
    READ ONLY.
    """
    out = {}

    path = Path(path)

    if not path.is_file():
        return out

    for raw in path.read_text(
        errors="ignore"
    ).splitlines():

        raw = raw.strip()

        if (
            not raw
            or
            raw.startswith("#")
            or
            ":" not in raw
        ):
            continue

        key, val = raw.split(
            ":",
            1,
        )

        out[
            key.strip()
        ] = val.strip()

    return out


def first_number(text):
    if text is None:
        return None

    m = re.search(
        r"[-+]?"
        r"(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[Ee][-+]?\d+)?",
        str(text),
    )

    if not m:
        return None

    try:
        return float(
            m.group(0)
        )
    except Exception:
        return None


print()
print("=" * 96)
print("A. ACCEPTED UPSTREAM PRODUCT")
print("=" * 96)

p14_files = sorted(
    (
        PROJECT
        / "production_logs"
    ).glob(
        "P14_final_product_audit_*.json"
    )
)

add_check(
    "P14 report exists",
    len(p14_files) > 0,
    len(p14_files),
)

if p14_files:

    latest_p14 = p14_files[-1]

    p14 = read_json(
        latest_p14
    )

    add_check(
        "P14 accepted status",
        p14.get("status")
        ==
        "PASS_COMPUTATIONAL_PRODUCTION",
        p14.get("status"),
    )

    evidence[
        "P14_report"
    ] = str(
        latest_p14
    )


ref_manifest_path = (
    PROC
    / "referenced_timeseries"
    / "referenced_timeseries_manifest.json"
)

ref_manifest = read_json(
    ref_manifest_path
)

add_check(
    "reference product status",
    ref_manifest.get("status")
    ==
    "PRELIMINARY_REFERENCED_PHASE",
    ref_manifest.get("status"),
)

corr = ref_manifest.get(
    "corrections_applied",
    {},
)

for key in (
    "SCLA",
    "residual_DEM",
    "APS",
    "GACOS",
    "ERA5",
    "ramp",
    "SCN",
):
    add_check(
        f"{key} currently not applied",
        not bool(
            corr.get(
                key,
                False,
            )
        ),
        corr.get(
            key,
            False,
        ),
    )

add_check(
    "LOS displacement not yet created",
    not bool(
        ref_manifest.get(
            "LOS_displacement_created",
            False,
        )
    ),
    ref_manifest.get(
        "LOS_displacement_created"
    ),
)


phase_path = (
    PROC
    / "referenced_timeseries"
    / "acquisition_phase_referenced_rad.npy"
)

strict_ids_path = (
    PROC
    / "network_inversion"
    / "strict_point_ids.npy"
)

dates_path = (
    PROC
    / "network_inversion"
    / "dates.txt"
)

phase = np.load(
    phase_path,
    mmap_mode="r",
)

strict_ids = np.load(
    strict_ids_path,
    mmap_mode="r",
)

dates = [
    x.strip()
    for x in dates_path.read_text().splitlines()
    if x.strip()
]

add_check(
    "referenced phase shape",
    phase.shape
    ==
    (
        881315,
        38,
    ),
    phase.shape,
)

add_check(
    "strict point count",
    strict_ids.size
    ==
    881315,
    strict_ids.size,
)

add_check(
    "dates count",
    len(dates)
    ==
    38,
    len(dates),
)

evidence[
    "referenced_phase"
] = str(
    phase_path
)


# =============================================================================
# B. RSLC acquisition parameter files
# =============================================================================

print()
print("=" * 96)
print("B. RSLC / SENSOR PHYSICAL PARAMETERS")
print("=" * 96)

tab_candidates = [
    PROJECT
    / "prototype_outputs"
    / "v09"
    / "network"
    / "gamma_base_calc"
    / "RSLC_tab.absolute",
]

tab_path = next(
    (
        p
        for p in tab_candidates
        if p.is_file()
    ),
    None,
)

add_check(
    "RSLC tab found",
    tab_path is not None,
    tab_path,
)

rslc_entries = []

if tab_path:

    for line in tab_path.read_text(
        errors="ignore"
    ).splitlines():

        f = line.split()

        if not f:
            continue

        slc = Path(
            f[0]
        )

        par = (
            Path(f[1])
            if len(f) >= 2
            else None
        )

        rslc_entries.append(
            (
                slc,
                par,
            )
        )

add_check(
    "RSLC tab acquisition count",
    len(rslc_entries)
    ==
    38,
    len(rslc_entries),
)

par_files = [
    p
    for _, p in rslc_entries
    if (
        p is not None
        and
        p.is_file()
    )
]

add_check(
    "RSLC parameter files",
    len(par_files)
    ==
    38,
    len(par_files),
)


# -----------------------------------------------------------------
# Extract frequency / wavelength evidence from all .par files.
# -----------------------------------------------------------------

freq_values = []
wavelength_values = []
incidence_values = []
heading_values = []

interesting_keys = {}

for p in par_files:

    d = gamma_par_dict(
        p
    )

    for key, val in d.items():

        kl = key.lower()

        if (
            "frequency" in kl
            or
            "wavelength" in kl
            or
            "incidence" in kl
            or
            "heading" in kl
            or
            "look_angle" in kl
            or
            "sensor" in kl
        ):
            interesting_keys.setdefault(
                key,
                set(),
            ).add(
                val
            )

        if (
            "radar_frequency" in kl
            or
            kl == "frequency"
        ):
            x = first_number(
                val
            )

            if x is not None:
                freq_values.append(
                    x
                )

        if "wavelength" in kl:
            x = first_number(
                val
            )

            if x is not None:
                wavelength_values.append(
                    x
                )

        if "incidence" in kl:
            x = first_number(
                val
            )

            if x is not None:
                incidence_values.append(
                    x
                )

        if "heading" in kl:
            x = first_number(
                val
            )

            if x is not None:
                heading_values.append(
                    x
                )


def finite_median(values):
    if not values:
        return None

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if not x.size:
        return None

    return float(
        np.median(
            x
        )
    )


radar_frequency = finite_median(
    freq_values
)

explicit_wavelength = finite_median(
    wavelength_values
)

computed_wavelength = None

if radar_frequency:

    # Some GAMMA parameter files store GHz;
    # others may store Hz.
    f_hz = radar_frequency

    if f_hz < 1.0e6:
        f_hz *= 1.0e9

    if f_hz > 1.0e8:
        computed_wavelength = (
            299792458.0
            /
            f_hz
        )


print()
print("sensor parameter evidence:")

for key in sorted(
    interesting_keys
):
    vals = sorted(
        interesting_keys[
            key
        ]
    )

    print(
        f"  {key}: "
        +
        " | ".join(
            vals[:3]
        )
    )


print()
print(
    "radar frequency candidate :",
    radar_frequency,
)

print(
    "explicit wavelength       :",
    explicit_wavelength,
)

print(
    "computed wavelength       :",
    computed_wavelength,
)

wavelength = (
    explicit_wavelength
    if explicit_wavelength
    else
    computed_wavelength
)

wavelength_ok = (
    wavelength is not None
    and
    0.03
    <
    wavelength
    <
    0.10
)

add_check(
    "physical radar wavelength resolved",
    wavelength_ok,
    wavelength,
)

if wavelength_ok:

    print(
        f"resolved wavelength       : "
        f"{wavelength:.9f} m"
    )


# =============================================================================
# C. Geometry / look-angle evidence
# =============================================================================

print()
print("=" * 96)
print("C. RADAR GEOMETRY")
print("=" * 96)

with CFG.open() as f:
    cfg = yaml.safe_load(
        f
    )

pc = cfg.get(
    "phase_correction",
    {},
)

rh = pc.get(
    "radar_height",
    {},
)

geom_par_path = Path(
    str(
        rh.get(
            "geometry_par",
            ""
        )
    )
)

add_check(
    "production geometry .par",
    geom_par_path.is_file(),
    geom_par_path,
)

geom = gamma_par_dict(
    geom_par_path
)

geometry_candidates = {}

for key, val in geom.items():

    kl = key.lower()

    if any(
        token in kl
        for token in (
            "incidence",
            "look",
            "heading",
            "azimuth_angle",
            "range_pixel_spacing",
            "azimuth_pixel_spacing",
            "near_range",
            "center_range",
            "far_range",
        )
    ):
        geometry_candidates[
            key
        ] = val


print()
print("geometry parameter evidence:")

for key, val in geometry_candidates.items():
    print(
        f"  {key}: {val}"
    )


incidence_median = finite_median(
    incidence_values
)

print()
print(
    "incidence candidate median:",
    incidence_median,
)

if incidence_median is None:

    warnings.append(
        "No explicit incidence-angle field was resolved from "
        "the inspected RSLC parameter files. "
        "Do not perform vertical projection yet."
    )


# =============================================================================
# D. Baseline/SCLA evidence
# =============================================================================

print()
print("=" * 96)
print("D. BASELINE / SCLA INPUT AVAILABILITY")
print("=" * 96)

search_roots = [
    PROJECT
    / "prototype_outputs"
    / "v09"
    / "network",
    PROJECT
    / "output"
    / "network",
    DEM,
]

baseline_files = []

baseline_patterns = (
    "base",
    "baseline",
    "bperp",
)

for root in search_roots:

    if not root.exists():
        continue

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        name = p.name.lower()

        if any(
            token in name
            for token in baseline_patterns
        ):
            try:
                size = p.stat().st_size
            except OSError:
                continue

            # Keep the audit bounded.
            if size <= 20 * 1024 * 1024:
                baseline_files.append(
                    p
                )


baseline_files = sorted(
    set(
        baseline_files
    )
)

print(
    f"baseline-like files found : "
    f"{len(baseline_files)}"
)

for p in baseline_files[:40]:
    print(
        "  ",
        p,
    )

evidence[
    "baseline_candidate_files"
] = [
    str(p)
    for p in baseline_files
]

if not baseline_files:

    warnings.append(
        "No baseline/SCLA-ready file was identified by filename. "
        "A dedicated baseline extraction step may be required."
    )


# -----------------------------------------------------------------
# Inspect small text baseline candidates for useful key names.
# -----------------------------------------------------------------

baseline_text_hits = []

patterns = re.compile(
    r"perpendicular|"
    r"bperp|"
    r"baseline|"
    r"B_perp",
    re.I,
)

for p in baseline_files:

    try:
        if p.stat().st_size > 5_000_000:
            continue

        data = p.read_text(
            errors="ignore"
        )

    except Exception:
        continue

    hits = []

    for line in data.splitlines():

        if patterns.search(
            line
        ):
            hits.append(
                line.strip()
            )

        if len(hits) >= 10:
            break

    if hits:
        baseline_text_hits.append({
            "path": str(p),
            "hits": hits,
        })


print()
print(
    "baseline text evidence   :",
    len(
        baseline_text_hits
    ),
)

for item in baseline_text_hits[:10]:

    print(
        "  FILE:",
        item[
            "path"
        ],
    )

    for line in item[
        "hits"
    ][:5]:
        print(
            "    ",
            line,
        )


evidence[
    "baseline_text_hits"
] = baseline_text_hits


# =============================================================================
# E. Atmospheric correction evidence
# =============================================================================

print()
print("=" * 96)
print("E. ATMOSPHERIC CORRECTION INPUTS")
print("=" * 96)

atmo_roots = [
    PROJECT,
    Path(
        "/home/ubuntu/Downloads"
    ),
]

gacos_candidates = []
era_candidates = []

for root in atmo_roots:

    if not root.exists():
        continue

    # Limit to common atmospheric file extensions/names.
    for p in root.rglob("*"):

        if not p.is_file():
            continue

        low = str(
            p
        ).lower()

        if (
            "gacos" in low
            or
            p.suffix.lower()
            in (
                ".ztd",
                ".ztd.tif",
            )
        ):
            gacos_candidates.append(
                p
            )

        if (
            "era5" in low
            and
            p.suffix.lower()
            in (
                ".nc",
                ".grib",
                ".grb",
                ".tif",
                ".npy",
                ".h5",
            )
        ):
            era_candidates.append(
                p
            )


gacos_candidates = sorted(
    set(
        gacos_candidates
    )
)

era_candidates = sorted(
    set(
        era_candidates
    )
)

print(
    "GACOS-like files          :",
    len(
        gacos_candidates
    ),
)

for p in gacos_candidates[:20]:
    print(
        "  ",
        p,
    )

print()
print(
    "ERA5-like files           :",
    len(
        era_candidates
    ),
)

for p in era_candidates[:20]:
    print(
        "  ",
        p,
    )


def date_coverage(files):
    found = set()

    for p in files:

        s = str(
            p
        )

        for d in dates:

            if d in s:
                found.add(
                    d
                )

    return found


gacos_dates = date_coverage(
    gacos_candidates
)

era_dates = date_coverage(
    era_candidates
)

print()
print(
    f"GACOS date coverage       : "
    f"{len(gacos_dates)}/38"
)

print(
    f"ERA5 date coverage        : "
    f"{len(era_dates)}/38"
)

evidence[
    "atmosphere"
] = {
    "GACOS_files":
        [
            str(p)
            for p in gacos_candidates[:200]
        ],

    "GACOS_dates":
        sorted(
            gacos_dates
        ),

    "ERA5_files":
        [
            str(p)
            for p in era_candidates[:200]
        ],

    "ERA5_dates":
        sorted(
            era_dates
        ),
}


# =============================================================================
# F. Existing repository implementation
# =============================================================================

print()
print("=" * 96)
print("F. EXISTING SOURCE IMPLEMENTATION")
print("=" * 96)

source_hits = {
    "SCLA": [],
    "APS": [],
    "GACOS": [],
    "ERA5": [],
    "LOS": [],
    "residual_DEM": [],
}

source_patterns = {
    "SCLA":
        re.compile(
            r"\bSCLA\b",
            re.I,
        ),

    "APS":
        re.compile(
            r"\bAPS\b",
            re.I,
        ),

    "GACOS":
        re.compile(
            r"\bGACOS\b",
            re.I,
        ),

    "ERA5":
        re.compile(
            r"\bERA5\b",
            re.I,
        ),

    "LOS":
        re.compile(
            r"LOS_displacement|"
            r"LOS displacement|"
            r"phase.*displacement",
            re.I,
        ),

    "residual_DEM":
        re.compile(
            r"residual[_ ]DEM|"
            r"DEM[_ ]error",
            re.I,
        ),
}


for p in REPO.rglob(
    "*.py"
):

    try:
        text = p.read_text(
            errors="ignore"
        )
    except Exception:
        continue

    for name, pat in source_patterns.items():

        if pat.search(
            text
        ):
            source_hits[
                name
            ].append(
                str(
                    p.relative_to(
                        REPO
                    )
                )
            )


for name, files in source_hits.items():

    print(
        f"{name:14s}: "
        f"{len(files)} source file(s)"
    )

    for p in files[:10]:
        print(
            "  ",
            p,
        )


# Existing reference file merely declaring `False`
# is NOT considered a correction implementation.
implementation_ready = {}

for name, files in source_hits.items():

    meaningful = [
        p
        for p in files
        if not (
            p.endswith(
                "apply_reference.py"
            )
            and
            name
            in (
                "SCLA",
                "APS",
                "GACOS",
                "ERA5",
                "LOS",
                "residual_DEM",
            )
        )
    ]

    implementation_ready[
        name
    ] = bool(
        meaningful
    )


# =============================================================================
# G. LOS sign-convention evidence
# =============================================================================

print()
print("=" * 96)
print("G. LOS SIGN CONVENTION")
print("=" * 96)

sign_evidence = []

for p in REPO.rglob(
    "*.py"
):

    try:
        text = p.read_text(
            errors="ignore"
        )
    except Exception:
        continue

    for lineno, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        low = line.lower()

        if (
            "4.0 * np.pi" in low
            or
            "4*np.pi" in low
            or
            "4 * np.pi" in low
            or
            "wavelength" in low
            and
            (
                "phase" in low
                or
                "los" in low
            )
        ):
            sign_evidence.append({
                "file":
                    str(
                        p.relative_to(
                            REPO
                        )
                    ),

                "line":
                    lineno,

                "text":
                    line.strip(),
            })


print(
    "phase/LOS conversion evidence:",
    len(
        sign_evidence
    ),
)

for x in sign_evidence[:20]:
    print(
        f"  {x['file']}:{x['line']}: "
        f"{x['text']}"
    )


if not sign_evidence:

    warnings.append(
        "No explicit phase-to-LOS sign convention was found in "
        "the current repository. Do not create a final LOS product "
        "until this convention is explicitly frozen."
    )


# =============================================================================
# H. Decision matrix
# =============================================================================

print()
print("=" * 96)
print("H. P15 DECISION MATRIX")
print("=" * 96)

scla_inputs = (
    wavelength_ok
    and
    bool(
        baseline_files
    )
)

gacos_ready = (
    len(
        gacos_dates
    )
    ==
    len(
        dates
    )
)

era_ready = (
    len(
        era_dates
    )
    ==
    len(
        dates
    )
)

los_ready = (
    wavelength_ok
    and
    bool(
        sign_evidence
    )
)


matrix = {
    "SCLA_input_evidence":
        scla_inputs,

    "GACOS_complete_date_coverage":
        gacos_ready,

    "ERA5_complete_date_coverage":
        era_ready,

    "LOS_wavelength_resolved":
        wavelength_ok,

    "LOS_sign_convention_resolved":
        bool(
            sign_evidence
        ),

    "existing_SCLA_implementation":
        implementation_ready[
            "SCLA"
        ],

    "existing_APS_implementation":
        implementation_ready[
            "APS"
        ],

    "existing_GACOS_implementation":
        implementation_ready[
            "GACOS"
        ],

    "existing_ERA5_implementation":
        implementation_ready[
            "ERA5"
        ],

    "existing_LOS_implementation":
        implementation_ready[
            "LOS"
        ],
}


for key, val in matrix.items():

    print(
        f"{key:40s}: "
        f"{val}"
    )


# =============================================================================
# I. Recommended next implementation
# =============================================================================

if not wavelength_ok:

    next_step = (
        "P15-1_SENSOR_GEOMETRY_RESOLUTION"
    )

elif not bool(
    baseline_files
):

    next_step = (
        "P15-1_BASELINE_EXTRACTION"
    )

elif not bool(
    sign_evidence
):

    next_step = (
        "P15-1_LOS_SIGN_CONVENTION_AUDIT"
    )

else:

    next_step = (
        "P15-1_SCLA_DESIGN_AND_SMOKE"
    )


print()
print(
    "recommended next step    :",
    next_step,
)


report = {
    "format":
        "pyPSDS-GAMMA-P15-final-chain-input-audit-v1",

    "status":
        (
            "PASS_INPUT_AUDIT"
            if not errors
            else
            "FAIL_INPUT_AUDIT"
        ),

    "accepted_upstream": {
        "referenced_phase":
            str(
                phase_path
            ),

        "shape":
            list(
                phase.shape
            ),

        "strict_points":
            int(
                strict_ids.size
            ),

        "dates":
            dates,
    },

    "sensor": {
        "radar_frequency_candidate":
            radar_frequency,

        "explicit_wavelength_m":
            explicit_wavelength,

        "computed_wavelength_m":
            computed_wavelength,

        "resolved_wavelength_m":
            wavelength,

        "incidence_candidate_median":
            incidence_median,

        "interesting_parameter_keys": {
            k:
                sorted(
                    v
                )
            for k, v
            in interesting_keys.items()
        },
    },

    "geometry_parameter_file":
        str(
            geom_par_path
        ),

    "geometry_parameter_evidence":
        geometry_candidates,

    "baseline_evidence":
        evidence.get(
            "baseline_candidate_files",
            [],
        ),

    "baseline_text_hits":
        baseline_text_hits,

    "atmospheric_evidence":
        evidence.get(
            "atmosphere",
            {},
        ),

    "source_hits":
        source_hits,

    "implementation_ready":
        implementation_ready,

    "LOS_sign_evidence":
        sign_evidence,

    "decision_matrix":
        matrix,

    "recommended_next_step":
        next_step,

    "warnings":
        warnings,

    "errors":
        errors,

    "checks":
        checks,
}


REPORT.write_text(
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
    "P15-0 FINAL DEFORMATION CHAIN INPUT AUDIT",
    "=" * 88,
    f"status                    : {report['status']}",
    f"strict points             : {strict_ids.size:,}",
    f"acquisitions              : {len(dates)}",
    f"resolved wavelength       : {wavelength}",
    f"baseline candidates       : {len(baseline_files)}",
    f"GACOS date coverage       : {len(gacos_dates)}/{len(dates)}",
    f"ERA5 date coverage        : {len(era_dates)}/{len(dates)}",
    f"LOS sign evidence         : {len(sign_evidence)}",
    "",
    f"recommended next step     : {next_step}",
]

if warnings:

    lines.append("")
    lines.append(
        "Warnings:"
    )

    for w in warnings:
        lines.append(
            "  - " + w
        )

lines.extend([
    "",
    f"JSON report: {REPORT}",
])

TXT.write_text(
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

    raise SystemExit(1)

print()
print("=" * 96)
print(" P15-0 FINAL RESULT: PASS_INPUT_AUDIT")
print("=" * 96)
PY

echo
echo "reports:"
echo "  $REPORT"
echo "  $TXT"
