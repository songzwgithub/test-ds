#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/Downloads/psds
OUT=$ROOT/output
PROC=$OUT/processing
CFG=$ROOT/production.yaml
LOGDIR=$ROOT/production_logs

STAMP=$(date +%Y%m%d_%H%M%S)

JSON_REPORT=$LOGDIR/P14_final_product_audit_${STAMP}.json
TXT_REPORT=$LOGDIR/P14_final_product_audit_${STAMP}.txt

mkdir -p "$LOGDIR"

echo "================================================================================================"
echo " P14 FINAL PRODUCT AUDIT"
echo
echo " READ ONLY: scientific products"
echo " NO SOURCE MODIFICATION"
echo " NO GAMMA"
echo " NO PHASE LINKING"
echo " NO UNWRAP"
echo
echo " Writes audit report only:"
echo "   $JSON_REPORT"
echo "================================================================================================"

python - "$OUT" "$CFG" "$JSON_REPORT" "$TXT_REPORT" <<'PY'
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

OUT = Path(sys.argv[1]).resolve()
CFG = Path(sys.argv[2]).resolve()
JSON_REPORT = Path(sys.argv[3]).resolve()
TXT_REPORT = Path(sys.argv[4]).resolve()

P = OUT / "processing"

errors = []
warnings = []
checks = []
summary = {}


def check(name, condition, detail=""):
    ok = bool(condition)

    checks.append({
        "name": name,
        "pass": ok,
        "detail": str(detail),
    })

    state = "PASS" if ok else "FAIL"

    print(
        f"{state:4s}  {name}"
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


def require(path):
    path = Path(path)

    check(
        f"exists: {path.relative_to(OUT)}",
        path.is_file(),
        path,
    )

    if not path.is_file():
        raise RuntimeError(
            f"Required product missing: {path}"
        )

    return path


def load_json(path):
    path = require(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


print()
print("=" * 96)
print("A. CONFIG / PRODUCTION ROOT")
print("=" * 96)

check(
    "production config exists",
    CFG.is_file(),
    CFG,
)

check(
    "output directory",
    OUT
    ==
    Path(
        "/home/ubuntu/Downloads/psds/output"
    ).resolve(),
    OUT,
)


# =============================================================================
# B. PS / DS / PointPhaseStack
# =============================================================================

print()
print("=" * 96)
print("B. PS / DS / POINT PHASE STACK")
print("=" * 96)

ps_path = require(
    P / "final_ps_mask.npy"
)

ds_path = require(
    P / "final_ds_tc0.800_pc0.000_evd.npy"
)

pps_dir = P / "point_phase_stack"

phase_path = require(
    pps_dir / "phase_rad.npy"
)

rows_path = require(
    pps_dir / "rows.npy"
)

cols_path = require(
    pps_dir / "cols.npy"
)


ps = np.load(
    ps_path,
    mmap_mode="r",
)

ds = np.load(
    ds_path,
    mmap_mode="r",
)

phase = np.load(
    phase_path,
    mmap_mode="r",
)

rows = np.load(
    rows_path,
    mmap_mode="r",
)

cols = np.load(
    cols_path,
    mmap_mode="r",
)


check(
    "PS raster shape",
    ps.shape == (600, 2000),
    ps.shape,
)

check(
    "DS raster shape",
    ds.shape == (600, 2000),
    ds.shape,
)

nps = int(
    np.count_nonzero(ps)
)

nds = int(
    np.count_nonzero(ds)
)

check(
    "final PS count",
    nps == 17547,
    nps,
)

check(
    "final DS count",
    nds == 863969,
    nds,
)

overlap = int(
    np.count_nonzero(
        np.asarray(ps, dtype=bool)
        &
        np.asarray(ds, dtype=bool)
    )
)

check(
    "PS/DS overlap",
    overlap == 0,
    overlap,
)

npoint = (
    nps + nds
)

check(
    "PointPhaseStack count",
    npoint == 881516,
    npoint,
)

check(
    "phase stack shape",
    phase.shape == (
        881516,
        38,
    ),
    phase.shape,
)

check(
    "rows length",
    rows.size == 881516,
    rows.size,
)

check(
    "cols length",
    cols.size == 881516,
    cols.size,
)

coord_key = (
    np.asarray(
        rows,
        dtype=np.int64,
    )
    *
    2000
    +
    np.asarray(
        cols,
        dtype=np.int64,
    )
)

unique_coords = int(
    np.unique(
        coord_key
    ).size
)

check(
    "unique point coordinates",
    unique_coords == 881516,
    unique_coords,
)

ref_epoch_max = float(
    np.max(
        np.abs(
            np.asarray(
                phase[:, 0],
                dtype=np.float64,
            )
        )
    )
)

check(
    "PointPhaseStack temporal reference",
    ref_epoch_max <= 1.0e-6,
    f"{ref_epoch_max:.3e} rad",
)

summary["point_stack"] = {
    "PS": nps,
    "DS": nds,
    "points": npoint,
    "acquisitions": 38,
    "PS_DS_overlap": overlap,
}


# =============================================================================
# C. Spatial graph
# =============================================================================

print()
print("=" * 96)
print("C. PRODUCTION SPATIAL GRAPH")
print("=" * 96)

graph_dir = (
    P / "spatial_graph"
)

graph_manifest = load_json(
    graph_dir
    / "spatial_graph_manifest.json"
)

core = graph_manifest[
    "core"
]

anchors = graph_manifest[
    "residual_anchors"
]

K = int(
    core[
        "nearest_neighbors"
    ]
)

ncomp = int(
    core[
        "components_before_anchors"
    ]
)

parity = bool(
    core[
        "exact_full_R4_partition_parity"
    ]
)

final_components = int(
    graph_manifest[
        "final_components"
    ]
)

anchor_edges = int(
    anchors[
        "edges"
    ]
)

local_u = np.load(
    require(
        graph_dir / "local_u.npy"
    ),
    mmap_mode="r",
)

local_v = np.load(
    require(
        graph_dir / "local_v.npy"
    ),
    mmap_mode="r",
)

component = np.load(
    require(
        graph_dir / "local_component.npy"
    ),
    mmap_mode="r",
)

anchor_u = np.load(
    require(
        graph_dir / "anchor_u.npy"
    ),
    mmap_mode="r",
)


check(
    "audited minimum local K",
    K == 17,
    K,
)

check(
    "R4 component count",
    ncomp == 40,
    ncomp,
)

check(
    "exact R4 partition parity",
    parity,
    parity,
)

check(
    "local edge count",
    local_u.size == 7977953,
    local_u.size,
)

check(
    "local edge array parity",
    local_u.size
    ==
    local_v.size,
    f"{local_u.size}/{local_v.size}",
)

saved_ncomp = int(
    np.unique(
        component
    ).size
)

check(
    "saved local components",
    saved_ncomp == 40,
    saved_ncomp,
)

counts = np.bincount(
    np.asarray(
        component,
        dtype=np.int32,
    )
)

largest_component = int(
    counts.max()
)

check(
    "R4 main component",
    largest_component == 881099,
    largest_component,
)

check(
    "residual anchor edges",
    anchor_edges == 78,
    anchor_edges,
)

check(
    "saved anchor edge count",
    anchor_u.size == 78,
    anchor_u.size,
)

check(
    "final spatial graph connected",
    final_components == 1,
    final_components,
)

summary["spatial_graph"] = {
    "minimum_exact_K": K,
    "local_edges": int(local_u.size),
    "components_before_anchors": ncomp,
    "largest_component": largest_component,
    "anchor_edges": anchor_edges,
    "final_components": final_components,
    "exact_R4_partition_parity": parity,
}


# =============================================================================
# D. Final unwrap candidate
# =============================================================================

print()
print("=" * 96)
print("D. FINAL UNWRAP QUALITY")
print("=" * 96)

unwrap_dir = (
    P / "final_unwrap"
)

unwrap_manifest = load_json(
    unwrap_dir
    / "final_unwrap_candidate_manifest.json"
)

strict_mask = np.load(
    require(
        unwrap_dir
        / "strict_unwrap_valid_mask.npy"
    ),
    mmap_mode="r",
)

temporal_bad_mask = np.load(
    require(
        unwrap_dir
        / "temporal_integer_bad_mask.npy"
    ),
    mmap_mode="r",
)

registered_mask = np.load(
    require(
        unwrap_dir
        / "all_ifg_registered_mask.npy"
    ),
    mmap_mode="r",
)

global_delta = np.load(
    require(
        unwrap_dir
        / "global_ifg_integer_delta.npy"
    ),
    mmap_mode="r",
)

uq = unwrap_manifest[
    "quality"
]

uc = unwrap_manifest[
    "final_temporal_closure"
]

ug = unwrap_manifest[
    "global_integer_gauge"
]


check(
    "unwrap candidate status",
    unwrap_manifest["status"]
    ==
    "PASS_CANDIDATE",
    unwrap_manifest["status"],
)

check(
    "unwrap point count",
    int(
        unwrap_manifest["points"]
    )
    ==
    881516,
    unwrap_manifest["points"],
)

check(
    "unwrap IFG count",
    int(
        unwrap_manifest["ifgs"]
    )
    ==
    108,
    unwrap_manifest["ifgs"],
)

check(
    "temporal cycle count",
    int(
        unwrap_manifest["cycles"]
    )
    ==
    71,
    unwrap_manifest["cycles"],
)

strict_count = int(
    np.count_nonzero(
        strict_mask
    )
)

check(
    "STRICT valid points",
    strict_count == 881315,
    strict_count,
)

check(
    "strict count manifest parity",
    int(
        uq["strict_valid"]
    )
    ==
    strict_count,
    uq["strict_valid"],
)

check(
    "all-IFG registered count",
    int(
        np.count_nonzero(
            registered_mask
        )
    )
    ==
    881488,
    np.count_nonzero(
        registered_mask
    ),
)

temporal_bad_count = int(
    np.count_nonzero(
        temporal_bad_mask
    )
)

check(
    "total temporal bad",
    temporal_bad_count == 138,
    temporal_bad_count,
)

check(
    "SAFE conflict points",
    int(
        uq[
            "safe_conflict_points"
        ]
    )
    ==
    42,
    uq[
        "safe_conflict_points"
    ],
)

check(
    "structural incompatible",
    int(
        uq[
            "structural_incompatible"
        ]
    )
    ==
    7,
    uq[
        "structural_incompatible"
    ],
)

check(
    "final temporal closure bad points",
    int(
        uc[
            "bad_points"
        ]
    )
    ==
    0,
    uc[
        "bad_points"
    ],
)

check(
    "final temporal closure bad occurrences",
    int(
        uc[
            "bad_occurrences"
        ]
    )
    ==
    0,
    uc[
        "bad_occurrences"
    ],
)

closure_residual = float(
    uc[
        "float_residual_max_rad"
    ]
)

check(
    "final closure float residual",
    closure_residual
    <
    1.0e-4,
    f"{closure_residual:.3e} rad",
)

check(
    "global integer gauge verified",
    bool(
        ug[
            "verified"
        ]
    ),
    ug[
        "verified"
    ],
)

check(
    "global gauge nonzero IFGs",
    int(
        ug[
            "nonzero_ifgs"
        ]
    )
    ==
    21,
    ug[
        "nonzero_ifgs"
    ],
)

check(
    "global gauge maximum |delta|",
    int(
        ug[
            "max_abs_integer_delta"
        ]
    )
    ==
    2,
    ug[
        "max_abs_integer_delta"
    ],
)

check(
    "global delta array length",
    global_delta.size == 108,
    global_delta.size,
)

check(
    "global delta nonzero array parity",
    int(
        np.count_nonzero(
            global_delta
        )
    )
    ==
    int(
        ug[
            "nonzero_ifgs"
        ]
    ),
    np.count_nonzero(
        global_delta
    ),
)

summary["unwrap"] = {
    "status":
        unwrap_manifest["status"],

    "strict_valid":
        strict_count,

    "excluded_from_strict":
        881516
        -
        strict_count,

    "registered_all_ifgs":
        int(
            np.count_nonzero(
                registered_mask
            )
        ),

    "temporal_bad":
        temporal_bad_count,

    "safe_conflict_points":
        int(
            uq[
                "safe_conflict_points"
            ]
        ),

    "structural_incompatible":
        int(
            uq[
                "structural_incompatible"
            ]
        ),

    "closure_bad_points":
        int(
            uc[
                "bad_points"
            ]
        ),

    "closure_bad_occurrences":
        int(
            uc[
                "bad_occurrences"
            ]
        ),

    "closure_float_residual_max_rad":
        closure_residual,
}


# =============================================================================
# E. Time-series inversion
# =============================================================================

print()
print("=" * 96)
print("E. TEMPORAL NETWORK INVERSION")
print("=" * 96)

inv_dir = (
    P / "network_inversion"
)

inv_manifest = load_json(
    inv_dir
    / "network_inversion_parity_manifest.json"
)

strict_ids = np.load(
    require(
        inv_dir
        / "strict_point_ids.npy"
    ),
    mmap_mode="r",
)

phase_l2 = np.load(
    require(
        inv_dir
        / "acquisition_phase_l2_candidate_rad.npy"
    ),
    mmap_mode="r",
)

expected_strict_ids = np.where(
    np.asarray(
        strict_mask,
        dtype=bool,
    )
)[0].astype(
    np.int32
)

check(
    "network inversion status",
    inv_manifest["status"]
    ==
    "PASS",
    inv_manifest["status"],
)

check(
    "strict point ID count",
    strict_ids.size
    ==
    881315,
    strict_ids.size,
)

check(
    "strict IDs exactly match unwrap mask",
    np.array_equal(
        np.asarray(
            strict_ids
        ),
        expected_strict_ids,
    ),
    strict_ids.size,
)

check(
    "L2 phase shape",
    phase_l2.shape
    ==
    (
        881315,
        38,
    ),
    phase_l2.shape,
)

check(
    "design rank",
    int(
        inv_manifest[
            "rank_A"
        ]
    )
    ==
    37,
    inv_manifest[
        "rank_A"
    ],
)

check(
    "acquisition count",
    int(
        inv_manifest[
            "acquisitions"
        ]
    )
    ==
    38,
    inv_manifest[
        "acquisitions"
    ],
)

check(
    "IFG count",
    int(
        inv_manifest[
            "ifgs"
        ]
    )
    ==
    108,
    inv_manifest[
        "ifgs"
    ],
)

tree_max = float(
    inv_manifest[
        "tree_l2_parity"
    ][
        "max_rad"
    ]
)

l2_max = float(
    inv_manifest[
        "l2_network_residual"
    ][
        "max_rad"
    ]
)

wrap_max = float(
    inv_manifest[
        "acquisition_wrap_parity"
    ][
        "max_rad"
    ]
)

tol = float(
    inv_manifest[
        "numerical_tolerance_rad"
    ]
)

check(
    "tree/L2 maximum parity",
    tree_max < tol,
    f"{tree_max:.3e} < {tol:.1e}",
)

check(
    "L2 network max residual",
    l2_max < tol,
    f"{l2_max:.3e} < {tol:.1e}",
)

check(
    "wrapped acquisition parity",
    wrap_max < tol,
    f"{wrap_max:.3e} < {tol:.1e}",
)

l2_ref_max = float(
    np.max(
        np.abs(
            np.asarray(
                phase_l2[:, 0],
                dtype=np.float64,
            )
        )
    )
)

check(
    "L2 temporal reference acquisition zero",
    l2_ref_max <= 1.0e-6,
    f"{l2_ref_max:.3e} rad",
)

summary["timeseries_inversion"] = {
    "status":
        inv_manifest["status"],

    "strict_points":
        int(
            strict_ids.size
        ),

    "rank_A":
        int(
            inv_manifest[
                "rank_A"
            ]
        ),

    "condition_number_A":
        float(
            inv_manifest[
                "condition_number_A"
            ]
        ),

    "tree_L2_max_rad":
        tree_max,

    "L2_network_residual_max_rad":
        l2_max,

    "wrapped_parity_max_rad":
        wrap_max,
}


# =============================================================================
# F. Spatial reference
# =============================================================================

print()
print("=" * 96)
print("F. COMPUTATIONAL REFERENCE REGION")
print("=" * 96)

ref_dir = (
    P / "referenced_timeseries"
)

ref_manifest = load_json(
    ref_dir
    / "referenced_timeseries_manifest.json"
)

ref_phase = np.load(
    require(
        ref_dir
        / "acquisition_phase_referenced_rad.npy"
    ),
    mmap_mode="r",
)

phase_rate = np.load(
    require(
        ref_dir
        / "preliminary_phase_rate_rad_per_year.npy"
    ),
    mmap_mode="r",
)

linear_rms = np.load(
    require(
        ref_dir
        / "preliminary_linear_residual_rms_rad.npy"
    ),
    mmap_mode="r",
)

ref_mask = np.load(
    require(
        ref_dir
        / "reference_region_mask.npy"
    ),
    mmap_mode="r",
)

ref_indices = np.load(
    require(
        ref_dir
        / "reference_strict_indices.npy"
    ),
    mmap_mode="r",
)

ref_point_ids = np.load(
    require(
        ref_dir
        / "reference_point_ids.npy"
    ),
    mmap_mode="r",
)

ref_median_before = np.load(
    require(
        ref_dir
        / "reference_phase_median_rad.npy"
    ),
    mmap_mode="r",
)

ref_sigma_before = np.load(
    require(
        ref_dir
        / "reference_phase_mad_sigma_rad.npy"
    ),
    mmap_mode="r",
)

region = ref_manifest[
    "reference_region"
]

corr = ref_manifest[
    "corrections_applied"
]


check(
    "reference manifest status",
    ref_manifest[
        "status"
    ]
    ==
    "PRELIMINARY_REFERENCED_PHASE",
    ref_manifest[
        "status"
    ],
)

check(
    "reference center row",
    int(
        region[
            "center_row"
        ]
    )
    ==
    539,
    region[
        "center_row"
    ],
)

check(
    "reference center col",
    int(
        region[
            "center_col"
        ]
    )
    ==
    337,
    region[
        "center_col"
    ],
)

check(
    "reference half-row",
    int(
        region[
            "half_row"
        ]
    )
    ==
    10,
    region[
        "half_row"
    ],
)

check(
    "reference half-col",
    int(
        region[
            "half_col"
        ]
    )
    ==
    15,
    region[
        "half_col"
    ],
)

nref = int(
    region[
        "points"
    ]
)

check(
    "reference point count",
    nref == 607,
    nref,
)

check(
    "referenced phase shape",
    ref_phase.shape
    ==
    (
        881315,
        38,
    ),
    ref_phase.shape,
)

check(
    "phase-rate shape",
    phase_rate.shape
    ==
    (
        881315,
    ),
    phase_rate.shape,
)

check(
    "linear RMS shape",
    linear_rms.shape
    ==
    (
        881315,
    ),
    linear_rms.shape,
)

check(
    "reference strict-index count",
    ref_indices.size == 607,
    ref_indices.size,
)

check(
    "reference point-ID count",
    ref_point_ids.size == 607,
    ref_point_ids.size,
)

check(
    "reference full-mask count",
    int(
        np.count_nonzero(
            ref_mask
        )
    )
    ==
    607,
    np.count_nonzero(
        ref_mask
    ),
)

expected_ref_ids = np.asarray(
    strict_ids[
        np.asarray(
            ref_indices,
            dtype=np.int64,
        )
    ],
    dtype=np.int32,
)

check(
    "reference strict-index / point-ID parity",
    np.array_equal(
        expected_ref_ids,
        np.asarray(
            ref_point_ids
        ),
    ),
    ref_point_ids.size,
)

mask_ref_ids = np.where(
    np.asarray(
        ref_mask,
        dtype=bool,
    )
)[0].astype(
    np.int32
)

check(
    "reference mask / point-ID parity",
    np.array_equal(
        np.sort(
            mask_ref_ids
        ),
        np.sort(
            np.asarray(
                ref_point_ids,
                dtype=np.int32,
            )
        ),
    ),
    mask_ref_ids.size,
)

check(
    "reference epoch-median array",
    ref_median_before.shape == (38,),
    ref_median_before.shape,
)

check(
    "reference MAD array",
    ref_sigma_before.shape == (38,),
    ref_sigma_before.shape,
)

post_region = np.asarray(
    ref_phase[
        np.asarray(
            ref_indices,
            dtype=np.int64,
        ),
        :
    ],
    dtype=np.float64,
)

post_median = np.median(
    post_region,
    axis=0,
)

post_median_max = float(
    np.max(
        np.abs(
            post_median
        )
    )
)

check(
    "reference epoch median after application",
    post_median_max <= 1.0e-6,
    f"{post_median_max:.3e} rad",
)

ref_first_max = float(
    np.max(
        np.abs(
            np.asarray(
                ref_phase[:, 0],
                dtype=np.float64,
            )
        )
    )
)

check(
    "referenced first acquisition zero",
    ref_first_max <= 1.0e-6,
    f"{ref_first_max:.3e} rad",
)

check(
    "spatial reference applied",
    bool(
        corr[
            "spatial_reference"
        ]
    ),
    corr[
        "spatial_reference"
    ],
)

for name in (
    "SCLA",
    "residual_DEM",
    "APS",
    "GACOS",
    "ERA5",
    "ramp",
    "SCN",
):
    check(
        f"{name} not applied",
        not bool(
            corr[name]
        ),
        corr[name],
    )

check(
    "LOS displacement intentionally absent",
    not bool(
        ref_manifest[
            "LOS_displacement_created"
        ]
    ),
    ref_manifest[
        "LOS_displacement_created"
    ],
)

summary["reference"] = {
    "status":
        ref_manifest[
            "status"
        ],

    "center_row":
        int(
            region[
                "center_row"
            ]
        ),

    "center_col":
        int(
            region[
                "center_col"
            ]
        ),

    "window":
        [
            2
            *
            int(
                region[
                    "half_row"
                ]
            )
            +
            1,

            2
            *
            int(
                region[
                    "half_col"
                ]
            )
            +
            1,
        ],

    "reference_points":
        nref,

    "post_reference_epoch_median_max_rad":
        post_median_max,

    "LOS_displacement_created":
        False,
}


# =============================================================================
# G. Provenance
# =============================================================================

print()
print("=" * 96)
print("G. PROVENANCE")
print("=" * 96)

run_dir = (
    OUT / "logs" / "runs"
)

run_manifests = sorted(
    run_dir.glob(
        "run_*.json"
    )
)

check(
    "run manifests present",
    len(run_manifests) >= 1,
    len(run_manifests),
)

full_manifest = (
    OUT / "manifest.json"
)

if not full_manifest.exists():
    warnings.append(
        "output/manifest.json is absent because the accepted "
        "production result was completed through a partial resume."
    )

    print(
        "INFO  output/manifest.json absent "
        "(expected for accepted partial resume)"
    )
else:
    print(
        f"INFO  full manifest present: "
        f"{full_manifest}"
    )


# Important product hashes.
hash_paths = [
    ps_path,
    ds_path,
    phase_path,

    graph_dir
    / "spatial_graph_manifest.json",

    unwrap_dir
    / "final_unwrap_candidate_manifest.json",

    unwrap_dir
    / "strict_unwrap_valid_mask.npy",

    inv_dir
    / "network_inversion_parity_manifest.json",

    inv_dir
    / "strict_point_ids.npy",

    ref_dir
    / "referenced_timeseries_manifest.json",

    ref_dir
    / "reference_point_ids.npy",
]

hashes = {}

for p in hash_paths:
    p = Path(p)

    if p.is_file():
        rel = str(
            p.relative_to(
                OUT
            )
        )

        hashes[rel] = sha256(
            p
        )


# =============================================================================
# H. Final classification
# =============================================================================

print()
print("=" * 96)
print("H. FINAL CLASSIFICATION")
print("=" * 96)

if errors:

    final_status = (
        "FAIL_FINAL_PRODUCT_AUDIT"
    )

else:

    final_status = (
        "PASS_COMPUTATIONAL_PRODUCTION"
    )


report = {
    "format":
        "pyPSDS-GAMMA-P14-final-product-audit-v1",

    "status":
        final_status,

    "production_config":
        str(
            CFG
        ),

    "production_output":
        str(
            OUT
        ),

    "scientific_scope": {
        "phase_linking":
            "complete",

        "PS_DS_selection":
            "complete",

        "spatial_graph":
            "complete_adaptive_minimum_K",

        "unwrapping":
            "strict_mask_candidate_accepted",

        "network_timeseries_inversion":
            "complete",

        "computational_spatial_reference":
            "complete",

        "SCLA":
            "not_applied",

        "APS":
            "not_applied",

        "GACOS_or_ERA5":
            "not_applied",

        "LOS_displacement_sign_conversion":
            "not_applied",
    },

    "summary":
        summary,

    "checks":
        checks,

    "warnings":
        warnings,

    "errors":
        errors,

    "run_manifests":
        [
            str(x)
            for x in run_manifests
        ],

    "full_pipeline_manifest_present":
        full_manifest.is_file(),

    "important_product_sha256":
        hashes,
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


lines = []

lines.append(
    "=" * 88
)

lines.append(
    "pyPSDS-GAMMA P14 FINAL PRODUCT AUDIT"
)

lines.append(
    "=" * 88
)

lines.append(
    f"status                 : {final_status}"
)

lines.append(
    ""
)

lines.append(
    f"PS                     : {nps:,}"
)

lines.append(
    f"DS                     : {nds:,}"
)

lines.append(
    f"PointPhaseStack        : {npoint:,}"
)

lines.append(
    f"adaptive local K       : {K}"
)

lines.append(
    f"R4 components          : {ncomp}"
)

lines.append(
    f"local edges            : {local_u.size:,}"
)

lines.append(
    f"strict valid           : {strict_count:,}/{npoint:,}"
)

lines.append(
    f"strict retained        : {100*strict_count/npoint:.6f}%"
)

lines.append(
    f"closure bad points     : {int(uc['bad_points'])}"
)

lines.append(
    f"closure bad occurrences: {int(uc['bad_occurrences'])}"
)

lines.append(
    f"tree/L2 max            : {tree_max:.3e} rad"
)

lines.append(
    f"L2 residual max        : {l2_max:.3e} rad"
)

lines.append(
    f"wrapped parity max     : {wrap_max:.3e} rad"
)

lines.append(
    f"reference points       : {nref}"
)

lines.append(
    f"reference median max   : {post_median_max:.3e} rad"
)

lines.append(
    ""
)

lines.append(
    "Current product level:"
)

lines.append(
    "  PRELIMINARY REFERENCED PHASE"
)

lines.append(
    "  No SCLA / APS / atmospheric correction / LOS sign conversion."
)

lines.append(
    ""
)

lines.append(
    f"full manifest present  : {full_manifest.is_file()}"
)

lines.append(
    f"run manifests          : {len(run_manifests)}"
)

if warnings:
    lines.append("")
    lines.append("Warnings:")

    for x in warnings:
        lines.append(
            "  - " + x
        )

if errors:
    lines.append("")
    lines.append("Errors:")

    for x in errors:
        lines.append(
            "  - " + x
        )

lines.append("")
lines.append(
    f"JSON report: {JSON_REPORT}"
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

print()
print("=" * 96)

if errors:
    print(
        " P14 FINAL RESULT: FAIL"
    )

    print("=" * 96)

    raise SystemExit(1)

print(
    " P14 FINAL RESULT: PASS_COMPUTATIONAL_PRODUCTION"
)

print("=" * 96)

PY

echo
echo "Audit reports:"
echo "  $JSON_REPORT"
echo "  $TXT_REPORT"
