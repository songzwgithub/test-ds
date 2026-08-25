#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/ubuntu/Downloads/psds
OUT=$PROJECT/output
PROC=$OUT/processing
GACOS=/home/ubuntu/Downloads/GACOS
DEM=/home/ubuntu/Downloads/DEM_prep
CFG=$PROJECT/production.yaml
LOGDIR=$PROJECT/production_logs

STAMP=$(date +%Y%m%d_%H%M%S)

JSON_REPORT=$LOGDIR/P15_3_gacos_geometry_sign_unit_${STAMP}.json
TXT_REPORT=$LOGDIR/P15_3_gacos_geometry_sign_unit_${STAMP}.txt

mkdir -p "$LOGDIR"

echo "================================================================================================"
echo " P15-3 GACOS GEOMETRY / UNIT / SIGN AUDIT"
echo
echo " READ ONLY: scientific products"
echo " NO SOURCE MODIFICATION"
echo " NO PRODUCTION PRODUCT MODIFICATION"
echo " NO GAMMA EXECUTION"
echo " NO GACOS CORRECTION"
echo "================================================================================================"

python - \
    "$PROJECT" \
    "$OUT" \
    "$GACOS" \
    "$DEM" \
    "$CFG" \
    "$JSON_REPORT" \
    "$TXT_REPORT" <<'PY'

from pathlib import Path
import json
import math
import sys

import numpy as np
import yaml


PROJECT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
GACOS = Path(sys.argv[3]).resolve()
DEM = Path(sys.argv[4]).resolve()
CFG = Path(sys.argv[5]).resolve()

JSON_REPORT = Path(sys.argv[6]).resolve()
TXT_REPORT = Path(sys.argv[7]).resolve()

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


def parse_rsc(path):

    out = {}

    for line in Path(path).read_text(
        errors="ignore"
    ).splitlines():

        s = line.strip()

        if not s:
            continue

        parts = s.split()

        if len(parts) < 2:
            continue

        key = parts[0].upper()

        out[key] = " ".join(
            parts[1:]
        )

    return out


def as_int(d, key):

    return int(
        float(
            d[key].split()[0]
        )
    )


def as_float(d, key):

    return float(
        d[key].split()[0]
    )


def parse_gamma_par(path):

    d = {}

    for line in Path(path).read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        key, value = line.split(
            ":",
            1,
        )

        d[
            key.strip().lower()
        ] = value.strip()

    return d


def first_float(x):

    if x is None:
        return None

    try:
        return float(
            str(x).split()[0]
        )
    except Exception:
        return None


def candidate_info(path):

    path = Path(path)

    info = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }

    if path.suffix.lower() == ".npy":

        try:
            a = np.load(
                path,
                mmap_mode="r",
            )

            info[
                "shape"
            ] = list(
                a.shape
            )

            info[
                "dtype"
            ] = str(
                a.dtype
            )

        except Exception as e:

            info[
                "load_error"
            ] = str(e)

    return info


# =============================================================================
# A. Accepted upstream contracts
# =============================================================================

print()
print("=" * 96)
print("A. ACCEPTED UPSTREAM CONTRACTS")
print("=" * 96)


p152_reports = sorted(
    (
        PROJECT
        / "production_logs"
    ).glob(
        "P15_2_scla_residual_dem_audit_*.json"
    )
)

check(
    "P15-2 report found",
    len(
        p152_reports
    )
    >
    0,
    len(
        p152_reports
    ),
)

if not p152_reports:
    raise SystemExit(1)


p152_path = p152_reports[-1]

p152 = json.loads(
    p152_path.read_text(
        encoding="utf-8"
    )
)


check(
    "P15-2 accepted",
    p152.get(
        "status"
    )
    ==
    "PASS_SCLA_INPUT_IDENTIFIABILITY",
    p152.get(
        "status"
    ),
)


p151_reports = sorted(
    (
        PROJECT
        / "production_logs"
    ).glob(
        "P15_1_los_sign_convention_*.json"
    )
)

check(
    "P15-1 report found",
    len(
        p151_reports
    )
    >
    0,
    len(
        p151_reports
    ),
)

if not p151_reports:
    raise SystemExit(1)


p151_path = p151_reports[-1]

p151 = json.loads(
    p151_path.read_text(
        encoding="utf-8"
    )
)


check(
    "P15-1 sign frozen",
    p151.get(
        "status"
    )
    ==
    "PASS_SIGN_CONVENTION_FROZEN",
    p151.get(
        "status"
    ),
)


wavelength = float(
    p151[
        "sensor"
    ][
        "wavelength_m"
    ]
)


phase_per_m_los = (
    4.0
    *
    math.pi
    /
    wavelength
)


print()
print(
    f"wavelength               : "
    f"{wavelength:.12f} m"
)

print(
    f"LOS delay -> phase       : "
    f"{phase_per_m_los:.6f} rad/m"
)


# =============================================================================
# B. Dates / reference
# =============================================================================

print()
print("=" * 96)
print("B. TIME-SERIES REFERENCE CONTRACT")
print("=" * 96)


dates_path = (
    PROC
    / "network_inversion"
    / "dates.txt"
)

dates = [
    x.strip()
    for x in dates_path.read_text().splitlines()
    if x.strip()
]


check(
    "38 acquisition dates",
    len(dates)
    ==
    38,
    len(dates),
)


temporal_reference_date = dates[0]


check(
    "temporal reference date",
    temporal_reference_date
    ==
    "20141006",
    temporal_reference_date,
)


ref_indices_path = (
    PROC
    / "referenced_timeseries"
    / "reference_strict_indices.npy"
)

ref_point_ids_path = (
    PROC
    / "referenced_timeseries"
    / "reference_point_ids.npy"
)


ref_indices = np.load(
    ref_indices_path,
    mmap_mode="r",
)

ref_point_ids = np.load(
    ref_point_ids_path,
    mmap_mode="r",
)


check(
    "reference region points",
    ref_indices.size
    ==
    607,
    ref_indices.size,
)

check(
    "reference point IDs",
    ref_point_ids.size
    ==
    607,
    ref_point_ids.size,
)


# =============================================================================
# C. GACOS date coverage
# =============================================================================

print()
print("=" * 96)
print("C. GACOS DATE / FILE CONTRACT")
print("=" * 96)


check(
    "GACOS directory",
    GACOS.is_dir(),
    GACOS,
)


missing_ztd = []
missing_rsc = []

records = []


for date in dates:

    ztd = (
        GACOS
        /
        f"{date}.ztd"
    )

    rsc = (
        GACOS
        /
        f"{date}.ztd.rsc"
    )

    if not ztd.is_file():
        missing_ztd.append(
            date
        )

    if not rsc.is_file():
        missing_rsc.append(
            date
        )

    if (
        ztd.is_file()
        and
        rsc.is_file()
    ):
        records.append(
            (
                date,
                ztd,
                rsc,
            )
        )


check(
    "GACOS ZTD coverage",
    len(
        missing_ztd
    )
    ==
    0,
    (
        f"{len(records)}/38; "
        f"missing={missing_ztd}"
    ),
)


check(
    "GACOS RSC coverage",
    len(
        missing_rsc
    )
    ==
    0,
    (
        f"{38-len(missing_rsc)}/38; "
        f"missing={missing_rsc}"
    ),
)


if (
    missing_ztd
    or
    missing_rsc
):

    raise SystemExit(1)


# =============================================================================
# D. RSC geometry
# =============================================================================

print()
print("=" * 96)
print("D. GACOS GRID GEOMETRY")
print("=" * 96)


required_keys = (
    "WIDTH",
    "FILE_LENGTH",
    "X_FIRST",
    "Y_FIRST",
    "X_STEP",
    "Y_STEP",
)


rsc_geometry = []

for date, ztd, rsc in records:

    d = parse_rsc(
        rsc
    )

    missing = [
        k
        for k in required_keys
        if k not in d
    ]

    if missing:

        errors.append(
            f"{date} RSC missing keys: "
            f"{missing}"
        )

        continue


    width = as_int(
        d,
        "WIDTH",
    )

    length = as_int(
        d,
        "FILE_LENGTH",
    )

    x_first = as_float(
        d,
        "X_FIRST",
    )

    y_first = as_float(
        d,
        "Y_FIRST",
    )

    x_step = as_float(
        d,
        "X_STEP",
    )

    y_step = as_float(
        d,
        "Y_STEP",
    )


    rsc_geometry.append({
        "date":
            date,

        "width":
            width,

        "length":
            length,

        "x_first":
            x_first,

        "y_first":
            y_first,

        "x_step":
            x_step,

        "y_step":
            y_step,
    })


check(
    "all RSC geometries parsed",
    len(
        rsc_geometry
    )
    ==
    38,
    len(
        rsc_geometry
    ),
)


g0 = rsc_geometry[0]


same_geometry = True

for g in rsc_geometry[1:]:

    if (
        g[
            "width"
        ]
        !=
        g0[
            "width"
        ]
        or
        g[
            "length"
        ]
        !=
        g0[
            "length"
        ]
        or
        not np.allclose(
            [
                g[
                    "x_first"
                ],
                g[
                    "y_first"
                ],
                g[
                    "x_step"
                ],
                g[
                    "y_step"
                ],
            ],
            [
                g0[
                    "x_first"
                ],
                g0[
                    "y_first"
                ],
                g0[
                    "x_step"
                ],
                g0[
                    "y_step"
                ],
            ],
            rtol=0.0,
            atol=1.0e-12,
        )
    ):

        same_geometry = False
        break


check(
    "all 38 GACOS grids identical geometry",
    same_geometry,
)


width = g0[
    "width"
]

length = g0[
    "length"
]

x_first = g0[
    "x_first"
]

y_first = g0[
    "y_first"
]

x_step = g0[
    "x_step"
]

y_step = g0[
    "y_step"
]


lon_last = (
    x_first
    +
    x_step
    *
    (
        width
        -
        1
    )
)

lat_last = (
    y_first
    +
    y_step
    *
    (
        length
        -
        1
    )
)


lon_min = min(
    x_first,
    lon_last,
)

lon_max = max(
    x_first,
    lon_last,
)

lat_min = min(
    y_first,
    lat_last,
)

lat_max = max(
    y_first,
    lat_last,
)


print()
print(
    f"GACOS grid               : "
    f"{length} x {width}"
)

print(
    f"X_FIRST / X_STEP         : "
    f"{x_first:.10f} / "
    f"{x_step:.10f}"
)

print(
    f"Y_FIRST / Y_STEP         : "
    f"{y_first:.10f} / "
    f"{y_step:.10f}"
)

print(
    f"longitude extent         : "
    f"{lon_min:.8f} .. "
    f"{lon_max:.8f}"
)

print(
    f"latitude extent          : "
    f"{lat_min:.8f} .. "
    f"{lat_max:.8f}"
)


check(
    "GACOS north-up row convention",
    y_step < 0.0,
    y_step,
)

check(
    "GACOS longitude step positive",
    x_step > 0.0,
    x_step,
)


# =============================================================================
# E. Binary contract / units
# =============================================================================

print()
print("=" * 96)
print("E. GACOS BINARY / UNIT AUDIT")
print("=" * 96)


expected_bytes = (
    width
    *
    length
    *
    4
)


file_size_ok = True

ztd_stats = []


for date, ztd, rsc in records:

    size = ztd.stat().st_size

    if size != expected_bytes:

        file_size_ok = False

        errors.append(
            f"{date}: ZTD size "
            f"{size} != {expected_bytes}"
        )

        continue


    # Official GACOS binary contract:
    # 4-byte FLOAT, LITTLE ENDIAN.
    mm = np.memmap(
        ztd,
        dtype="<f4",
        mode="r",
        shape=(
            length,
            width,
        ),
    )


    total = (
        width
        *
        length
    )

    stride = max(
        1,
        total
        //
        200000,
    )


    sample = np.asarray(
        mm.reshape(-1)[
            ::stride
        ],
        dtype=np.float64,
    )


    sample = sample[
        :200000
    ]


    finite = np.isfinite(
        sample
    )


    finite_fraction = float(
        np.mean(
            finite
        )
    )


    if np.any(
        finite
    ):

        x = sample[
            finite
        ]

        q = np.percentile(
            x,
            [
                1,
                50,
                99,
            ],
        )

        ztd_stats.append({
            "date":
                date,

            "finite_fraction":
                finite_fraction,

            "p01_m":
                float(
                    q[0]
                ),

            "p50_m":
                float(
                    q[1]
                ),

            "p99_m":
                float(
                    q[2]
                ),

            "min_m":
                float(
                    np.min(
                        x
                    )
                ),

            "max_m":
                float(
                    np.max(
                        x
                    )
                ),
        })

    else:

        ztd_stats.append({
            "date":
                date,

            "finite_fraction":
                0.0,
        })


check(
    "all ZTD byte sizes = 4*WIDTH*FILE_LENGTH",
    file_size_ok,
    expected_bytes,
)


minimum_finite = min(
    x[
        "finite_fraction"
    ]
    for x in ztd_stats
)


check(
    "GACOS little-endian float32 readable",
    minimum_finite
    >
    0.99,
    (
        f"minimum finite fraction="
        f"{minimum_finite:.6f}"
    ),
)


medians = np.asarray(
    [
        x[
            "p50_m"
        ]
        for x in ztd_stats
        if "p50_m" in x
    ],
    dtype=np.float64,
)


print()
print(
    "ZTD epoch median range   : "
    f"{medians.min():.4f} .. "
    f"{medians.max():.4f} m"
)


plausible_ztd = bool(
    np.all(
        np.isfinite(
            medians
        )
    )
    and
    np.all(
        np.abs(
            medians
        )
        <
        10.0
    )
)


check(
    "ZTD meter-scale values plausible",
    plausible_ztd,
    (
        f"median range "
        f"{medians.min():.3f}.."
        f"{medians.max():.3f} m"
    ),
)


# =============================================================================
# F. Scene center coverage
# =============================================================================

print()
print("=" * 96)
print("F. SAR / GACOS GEOGRAPHIC COVERAGE")
print("=" * 96)


with CFG.open(
    encoding="utf-8"
) as f:

    cfg = yaml.safe_load(
        f
    )


geom_par = Path(
    cfg[
        "phase_correction"
    ][
        "radar_height"
    ][
        "geometry_par"
    ]
)


check(
    "production geometry par exists",
    geom_par.is_file(),
    geom_par,
)


gp = parse_gamma_par(
    geom_par
)


center_lat = first_float(
    gp.get(
        "center_latitude"
    )
)

center_lon = first_float(
    gp.get(
        "center_longitude"
    )
)


# Some MLI pars omit geographic center.
# Fall back to first RSLC parameter file.
if (
    center_lat is None
    or
    center_lon is None
):

    rslc_tab = (
        PROJECT
        / "prototype_outputs"
        / "v09"
        / "network"
        / "gamma_base_calc"
        / "RSLC_tab.absolute"
    )

    first_line = (
        rslc_tab
        .read_text()
        .splitlines()[0]
        .split()
    )

    if len(
        first_line
    ) >= 2:

        rp = parse_gamma_par(
            Path(
                first_line[1]
            )
        )

        center_lat = first_float(
            rp.get(
                "center_latitude"
            )
        )

        center_lon = first_float(
            rp.get(
                "center_longitude"
            )
        )


print()
print(
    "SAR center latitude      :",
    center_lat,
)

print(
    "SAR center longitude     :",
    center_lon,
)


center_resolved = (
    center_lat is not None
    and
    center_lon is not None
)


check(
    "SAR geographic center resolved",
    center_resolved,
)


if center_resolved:

    center_inside = (
        lon_min
        <=
        center_lon
        <=
        lon_max
        and
        lat_min
        <=
        center_lat
        <=
        lat_max
    )

else:

    center_inside = False


check(
    "SAR center inside GACOS grid",
    center_inside,
    (
        f"center=({center_lat},"
        f"{center_lon})"
    ),
)


if center_inside:

    lon_margin = min(
        center_lon
        -
        lon_min,
        lon_max
        -
        center_lon,
    )

    lat_margin = min(
        center_lat
        -
        lat_min,
        lat_max
        -
        center_lat,
    )

    print(
        f"GACOS center lon margin   : "
        f"{lon_margin:.5f} deg"
    )

    print(
        f"GACOS center lat margin   : "
        f"{lat_margin:.5f} deg"
    )


# =============================================================================
# G. Incidence angle
# =============================================================================

print()
print("=" * 96)
print("G. INCIDENCE-ANGLE CONTRACT")
print("=" * 96)


incidence_center = first_float(
    gp.get(
        "incidence_angle"
    )
)


check(
    "center incidence angle resolved",
    incidence_center is not None,
    incidence_center,
)


if incidence_center is not None:

    cos_center = math.cos(
        math.radians(
            incidence_center
        )
    )

    sec_center = (
        1.0
        /
        cos_center
    )


    print(
        f"center incidence          : "
        f"{incidence_center:.6f} deg"
    )

    print(
        f"1/cos(theta_center)       : "
        f"{sec_center:.9f}"
    )

else:

    cos_center = None
    sec_center = None


# Search for pointwise/radar-grid incidence assets.
search_roots = [
    DEM,
    OUT,
    PROJECT
    / "prototype_outputs"
    / "v09",
]


inc_candidates = []


for root in search_roots:

    if not root.exists():
        continue

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        low = p.name.lower()

        if (
            "incidence" not in low
            and
            "inc_angle" not in low
            and
            not low.startswith(
                "inc"
            )
            and
            "theta" not in low
        ):
            continue

        if p.suffix.lower() in (
            ".par",
            ".json",
            ".txt",
            ".log",
            ".png",
            ".jpg",
            ".jpeg",
            ".csv",
        ):
            continue

        try:
            info = candidate_info(
                p
            )
        except Exception:
            continue

        inc_candidates.append(
            info
        )


print()
print(
    f"incidence raster candidates: "
    f"{len(inc_candidates)}"
)


for x in inc_candidates[
    :20
]:

    print(
        "  ",
        x,
    )


RADAR_N = (
    600
    *
    2000
)


def recognized_radar_candidate(info):

    shape = info.get(
        "shape"
    )

    if shape in (
        [
            600,
            2000,
        ],
        [
            881315,
        ],
        [
            881516,
        ],
    ):
        return True

    if (
        info[
            "size_bytes"
        ]
        ==
        RADAR_N
        *
        4
    ):
        return True

    return False


pointwise_incidence_ready = any(
    recognized_radar_candidate(
        x
    )
    for x in inc_candidates
)


if pointwise_incidence_ready:

    print(
        "pointwise incidence       : AVAILABLE"
    )

else:

    print(
        "pointwise incidence       : NOT YET RESOLVED"
    )

    warnings.append(
        "Only center incidence angle is currently resolved. "
        "Final production GACOS correction should preferably "
        "use pointwise incidence angle."
    )


# =============================================================================
# H. Radar -> geographic coordinate assets
# =============================================================================

print()
print("=" * 96)
print("H. RADAR-POINT GEOLOCATION ASSET AUDIT")
print("=" * 96)


lon_candidates = []
lat_candidates = []


for root in search_roots:

    if not root.exists():
        continue

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        low = p.name.lower()

        if p.suffix.lower() in (
            ".par",
            ".json",
            ".txt",
            ".log",
            ".png",
            ".jpg",
            ".jpeg",
            ".csv",
        ):
            continue


        is_lon = (
            "longitude"
            in low
            or
            low.startswith(
                "lon"
            )
            or
            "_lon"
            in low
        )


        is_lat = (
            "latitude"
            in low
            or
            low.startswith(
                "lat"
            )
            or
            "_lat"
            in low
        )


        if not (
            is_lon
            or
            is_lat
        ):
            continue


        try:
            info = candidate_info(
                p
            )
        except Exception:
            continue


        if is_lon:
            lon_candidates.append(
                info
            )

        if is_lat:
            lat_candidates.append(
                info
            )


print(
    f"longitude candidates      : "
    f"{len(lon_candidates)}"
)

for x in lon_candidates[
    :20
]:
    print(
        "  ",
        x,
    )


print()
print(
    f"latitude candidates       : "
    f"{len(lat_candidates)}"
)

for x in lat_candidates[
    :20
]:
    print(
        "  ",
        x,
    )


recognized_lon = [
    x
    for x in lon_candidates
    if recognized_radar_candidate(
        x
    )
]

recognized_lat = [
    x
    for x in lat_candidates
    if recognized_radar_candidate(
        x
    )
]


radar_geolocation_ready = bool(
    recognized_lon
    and
    recognized_lat
)


print()
print(
    "radar point geolocation  : "
    +
    (
        "AVAILABLE"
        if radar_geolocation_ready
        else
        "NOT YET RESOLVED"
    )
)


if not radar_geolocation_ready:

    warnings.append(
        "No validated radar-grid/point longitude+latitude pair "
        "was identified. GACOS cannot yet be sampled at the "
        "881,315 strict points without resolving geolocation."
    )


# =============================================================================
# I. GACOS sign convention for pyPSDS
# =============================================================================

print()
print("=" * 96)
print("I. GACOS -> pyPSDS SIGN CONTRACT")
print("=" * 96)


print(
    "Frozen definitions:"
)

print(
    "  pyPSDS LOS positive       = toward satellite"
)

print(
    "  GACOS ZTD positive        = increased path delay"
)

print(
    "  temporal GACOS difference = ZTD_t - ZTD_20141006"
)

print()


print(
    "Required production mapping:"
)

print(
    "  L_t(p) = ZTD_t(p) / cos(theta(p))"
)

print(
    "  dL_t(p) = L_t(p) - L_20141006(p)"
)

print(
    "  dL_ref(t,p) = dL_t(p)"
)

print(
    "                - median_ref_region[dL_t]"
)

print()

print(
    "Atmospheric phase contained in pyPSDS:"
)

print(
    "  phi_atm = -(4*pi/lambda) * dL_ref"
)

print()

print(
    "Therefore correction MUST be:"
)

print(
    "  phi_corrected"
)

print(
    "    = phi_observed"
)

print(
    "      + (4*pi/lambda) * dL_ref"
)


correction_factor = (
    4.0
    *
    math.pi
    /
    wavelength
)


check(
    "GACOS correction phase sign frozen",
    correction_factor > 0.0,
    (
        f"+{correction_factor:.9f} "
        "rad/m LOS delay"
    ),
)


# =============================================================================
# J. First-epoch / spatial-reference invariants
# =============================================================================

print()
print("=" * 96)
print("J. REQUIRED FUTURE CORRECTION INVARIANTS")
print("=" * 96)


invariants = [
    (
        "GACOS temporal reference must be 20141006",
        True,
    ),

    (
        "first corrected atmospheric epoch must be exactly zero",
        True,
    ),

    (
        "same 607-point computational reference region must be reused",
        True,
    ),

    (
        "reference-region median atmospheric correction must be zero per epoch",
        True,
    ),

    (
        "original referenced phase file must remain immutable",
        True,
    ),
]


for name, ok in invariants:

    check(
        name,
        ok,
    )


# =============================================================================
# K. Decision
# =============================================================================

print()
print("=" * 96)
print("K. P15-3 DECISION")
print("=" * 96)


file_contract_ready = (
    len(
        errors
    )
    ==
    0
)


if not file_contract_ready:

    status = (
        "FAIL_GACOS_CONTRACT"
    )

    next_step = (
        "STOP"
    )


elif (
    radar_geolocation_ready
    and
    pointwise_incidence_ready
):

    status = (
        "PASS_GACOS_GEOMETRY_SIGN_UNIT_READY"
    )

    next_step = (
        "P15-4_GACOS_POINT_SAMPLING_SMOKE"
    )


elif not radar_geolocation_ready:

    status = (
        "PASS_GACOS_FILE_SIGN_UNIT_NEEDS_GEOLOCATION"
    )

    next_step = (
        "P15-3A_BUILD_RADAR_POINT_GEOLOCATION"
    )


else:

    status = (
        "PASS_GACOS_FILE_SIGN_UNIT_NEEDS_INCIDENCE"
    )

    next_step = (
        "P15-3A_RESOLVE_POINTWISE_INCIDENCE"
    )


print(
    f"file/unit contract        : "
    f"{file_contract_ready}"
)

print(
    f"radar geolocation ready   : "
    f"{radar_geolocation_ready}"
)

print(
    f"pointwise incidence ready : "
    f"{pointwise_incidence_ready}"
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
        "pyPSDS-GAMMA-P15-3-GACOS-geometry-sign-unit-v1",

    "status":
        status,

    "P15_1_report":
        str(
            p151_path
        ),

    "P15_2_report":
        str(
            p152_path
        ),

    "gacos": {
        "directory":
            str(
                GACOS
            ),

        "date_coverage":
            len(
                records
            ),

        "storage_contract":
            "little-endian float32",

        "physical_quantity":
            "zenith total delay",

        "unit":
            "meter",

        "grid": {
            "width":
                width,

            "length":
                length,

            "x_first":
                x_first,

            "y_first":
                y_first,

            "x_step":
                x_step,

            "y_step":
                y_step,

            "lon_min":
                lon_min,

            "lon_max":
                lon_max,

            "lat_min":
                lat_min,

            "lat_max":
                lat_max,
        },

        "ztd_stats":
            ztd_stats,
    },

    "sar_scene": {
        "center_latitude":
            center_lat,

        "center_longitude":
            center_lon,

        "center_inside_gacos":
            center_inside,

        "center_incidence_angle_deg":
            incidence_center,

        "pointwise_incidence_ready":
            pointwise_incidence_ready,

        "incidence_candidates":
            inc_candidates,
    },

    "geolocation": {
        "ready":
            radar_geolocation_ready,

        "longitude_candidates":
            lon_candidates,

        "latitude_candidates":
            lat_candidates,
    },

    "sign_contract": {
        "pypsds_LOS_positive":
            "toward_satellite",

        "ZTD_positive":
            "increased_propagation_path_delay",

        "temporal_difference":
            (
                "ZTD_t - ZTD_20141006"
            ),

        "slant_delay":
            (
                "L_t = ZTD_t / cos(theta)"
            ),

        "spatial_reference":
            (
                "subtract median over the same "
                "607 reference points"
            ),

        "atmospheric_phase_in_pypsds":
            (
                "phi_atm = "
                "-(4*pi/lambda)*dL_ref"
            ),

        "correction":
            (
                "phi_corr = phi_obs "
                "+ (4*pi/lambda)*dL_ref"
            ),

        "phase_factor_rad_per_m":
            correction_factor,
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
    "P15-3 GACOS GEOMETRY / UNIT / SIGN",
    "=" * 88,

    f"status                    : {status}",

    f"GACOS dates               : {len(records)}/38",

    (
        "GACOS grid                : "
        f"{length} x {width}"
    ),

    (
        "ZTD epoch median range    : "
        f"{medians.min():.4f} .. "
        f"{medians.max():.4f} m"
    ),

    (
        "SAR center in GACOS       : "
        f"{center_inside}"
    ),

    (
        "center incidence          : "
        f"{incidence_center}"
    ),

    (
        "pointwise incidence ready : "
        f"{pointwise_incidence_ready}"
    ),

    (
        "radar geolocation ready   : "
        f"{radar_geolocation_ready}"
    ),

    "",

    (
        "GACOS correction sign     : "
        "PLUS in pyPSDS phase space"
    ),

    (
        "formula                   : "
        "phi_corr = phi_obs + "
        "(4*pi/lambda)*dL_ref"
    ),

    "",

    f"next step                 : {next_step}",

    "",

    f"JSON report               : {JSON_REPORT}",
]


if warnings:

    lines.append("")
    lines.append(
        "Warnings:"
    )

    for w in warnings:

        lines.append(
            "  - "
            +
            w
        )


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
        " P15-3 FINAL RESULT: FAIL"
    )
    print("=" * 96)

    raise SystemExit(1)


print()
print("=" * 96)
print(
    f" P15-3 FINAL RESULT: {status}"
)
print("=" * 96)

PY

echo
echo "reports:"
echo "  $JSON_REPORT"
echo "  $TXT_REPORT"
