#!/usr/bin/env bash
set -euo pipefail

REPO="/home/ubuntu/software/pyPSDS-GAMMA-v1.0"
PROJECT="/home/ubuntu/Downloads/psds"

STAMP="$(date +%Y%m%d_%H%M%S)"

OUT="${REPO}/docs/release_v1_1/migration_sources"
SNAPSHOT="${OUT}/snapshot_${STAMP}"

mkdir -p \
    "${OUT}" \
    "${SNAPSHOT}/files"

echo "================================================================================"
echo "pyPSDS-GAMMA v1.1 MIGRATION SOURCE FREEZE"
echo "================================================================================"
echo "repo       : ${REPO}"
echo "project    : ${PROJECT}"
echo "snapshot   : ${SNAPSHOT}"
echo "mode       : READ/COPY ONLY"
echo


python - <<'PY'
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import re
import shutil


REPO = Path(
    "/home/ubuntu/software/pyPSDS-GAMMA-v1.0"
)

PROJECT = Path(
    "/home/ubuntu/Downloads/psds"
)

OUT_ROOT = (
    REPO
    / "docs"
    / "release_v1_1"
    / "migration_sources"
)

snapshots = sorted(
    p
    for p in OUT_ROOT.glob("snapshot_*")
    if p.is_dir()
)

if not snapshots:
    raise RuntimeError(
        "Snapshot directory was not created."
    )

SNAPSHOT = snapshots[-1]

FILES_OUT = (
    SNAPSHOT
    / "files"
)


# ============================================================================
# Search roots
# ============================================================================

SEARCH_ROOTS = [
    Path("/tmp"),
    PROJECT,
    Path("/home/ubuntu/Downloads"),
    REPO,
]


# ============================================================================
# Exclusions
# ============================================================================

EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "site-packages",
    "node_modules",
    "build",
    "dist",
    "pypsds_gamma.egg-info",
    "migration_sources",
}


# ============================================================================
# Strong fingerprints from the validated post-processing chain
# ============================================================================

SIGNATURES = {

    "geometry": [
        "longitude_deg.npy",
        "latitude_deg.npy",
        "incidence_gamma_compatible_fast_rad.npy",
        "strict_points.plist",
        "gacos_geometry",
    ],

    "gacos": [
        "gacos_corrected",
        "acquisition_phase_gacos_corrected_rad.npy",
        "GACOS",
        "ZTD",
        "strict_dates",
    ],

    "scla": [
        "K_ps_uw",
        "C_ps_uw",
        "sm_cov",
        "pre_scn",
        "ph_scla",
        "stamps_scla",
    ],

    "scn": [
        "ph_hpt_rad.npy",
        "ph_scn_slave_rad.npy",
        "scn_wavelength",
        "stamps_stage8",
        "400",
    ],

    "los_timeseries": [
        "los_displacement_toward_satellite",
        "acquisition_phase_final_rad.npy",
        "lambda/(4*pi)",
        "4*pi",
        "toward_satellite",
    ],

    "point_products": [
        "los_velocity_toward_satellite",
        "los_cumulative_toward_satellite",
        "velocity_slope_standard_error",
        "linear_residual_rms",
        "final_los_products",
    ],

    "geocoding": [
        "longitude_deg",
        "latitude_deg",
        "geopackage",
        "ogr2ogr",
        "GeoPackage",
        "final_point_geocoding",
    ],

    "delivery": [
        "psds_final_points",
        "phase_std_rad",
        "velocity_snr",
        "quality_flag",
        "final_delivery",
    ],
}


DEVELOPMENT_MARKERS = [
    "P15",
    "p15_",
    "PASS_STAMPS",
    "PASS_FINAL",
    "PASS_POINT",
    "prototype",
    "smoke",
]


# ============================================================================
# Helpers
# ============================================================================

def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def safe_text(path: Path) -> str:

    try:

        if path.stat().st_size > 8 * 1024 * 1024:
            return ""

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception:
        return ""


def excluded(path: Path) -> bool:

    return any(
        part in EXCLUDE_PARTS
        for part in path.parts
    )


def classify(
    path: Path,
    text: str,
):

    haystack = (
        str(path)
        + "\n"
        + text
    ).lower()

    scores = {}

    matched = {}

    for group, tokens in SIGNATURES.items():

        hits = []

        score = 0

        for token in tokens:

            n = haystack.count(
                token.lower()
            )

            if n:

                hits.append(
                    {
                        "token": token,
                        "count": n,
                    }
                )

                score += n

        if score:

            scores[group] = score
            matched[group] = hits


    if not scores:

        return (
            None,
            0,
            {},
        )


    best = max(
        scores,
        key=scores.get,
    )

    return (
        best,
        scores[best],
        matched,
    )


# ============================================================================
# Collect candidate files
# ============================================================================

candidate_paths = set()

for root in SEARCH_ROOTS:

    if not root.exists():
        continue

    try:

        iterator = root.rglob("*")

        for p in iterator:

            if not p.is_file():
                continue

            if excluded(p):
                continue

            if p.suffix.lower() not in {
                ".py",
                ".sh",
            }:
                continue

            candidate_paths.add(
                p.resolve()
            )

    except PermissionError:
        pass


print(
    f"source files scanned : {len(candidate_paths)}"
)


# ============================================================================
# Analyze
# ============================================================================

records = []

for path in sorted(
    candidate_paths
):

    text = safe_text(
        path
    )

    group, score, matched = classify(
        path,
        text,
    )

    if group is None:
        continue


    dev_hits = sorted(
        marker
        for marker in DEVELOPMENT_MARKERS
        if marker.lower()
        in (
            str(path)
            + "\n"
            + text
        ).lower()
    )


    stat = path.stat()


    record = {

        "source_path":
            str(path),

        "category":
            group,

        "score":
            int(score),

        "size_bytes":
            int(stat.st_size),

        "mtime":
            datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat(),

        "sha256":
            sha256(path),

        "development_markers":
            dev_hits,

        "matched_signatures":
            matched,
    }

    records.append(
        record
    )


# ============================================================================
# Deduplicate exact files
# ============================================================================

by_hash = {}

for rec in records:

    by_hash.setdefault(
        rec["sha256"],
        [],
    ).append(
        rec
    )


unique = []

for digest, group_records in by_hash.items():

    # Prefer /tmp development script first,
    # then project-local, then repository copy.

    def preference(r):

        p = r["source_path"]

        if p.startswith("/tmp/"):
            rank = 0

        elif p.startswith(
            "/home/ubuntu/Downloads/psds/"
        ):
            rank = 1

        elif p.startswith(
            "/home/ubuntu/Downloads/"
        ):
            rank = 2

        else:
            rank = 3

        return (
            rank,
            p,
        )


    chosen = sorted(
        group_records,
        key=preference,
    )[0]

    chosen = dict(
        chosen
    )

    chosen[
        "duplicate_locations"
    ] = sorted(
        r["source_path"]
        for r in group_records
    )

    unique.append(
        chosen
    )


unique.sort(
    key=lambda r: (
        r["category"],
        -r["score"],
        r["source_path"],
    )
)


# ============================================================================
# Copy frozen migration sources
# ============================================================================

copied = []

category_counts = {}

for i, rec in enumerate(
    unique,
    start=1,
):

    category = (
        rec["category"]
    )

    category_counts[
        category
    ] = (
        category_counts.get(
            category,
            0,
        )
        + 1
    )


    src = Path(
        rec["source_path"]
    )

    category_dir = (
        FILES_OUT
        /
        category
    )

    category_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    short_hash = (
        rec["sha256"][:10]
    )

    target = (
        category_dir
        /
        (
            f"{src.stem}"
            f"__{short_hash}"
            f"{src.suffix}"
        )
    )


    shutil.copy2(
        src,
        target,
    )


    rec["snapshot_path"] = str(
        target.relative_to(
            SNAPSHOT
        )
    )

    copied.append(
        rec
    )


# ============================================================================
# Write JSON manifest
# ============================================================================

manifest = {

    "stage":
        "v1.1_migration_source_freeze",

    "repository":
        str(REPO),

    "project":
        str(PROJECT),

    "search_roots": [
        str(x)
        for x in SEARCH_ROOTS
    ],

    "unique_candidate_count":
        len(copied),

    "category_counts":
        category_counts,

    "files":
        copied,
}


(
    SNAPSHOT
    /
    "migration_source_manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================================
# TSV
# ============================================================================

lines = [
    (
        "category\t"
        "score\t"
        "size_bytes\t"
        "mtime\t"
        "sha256\t"
        "source_path\t"
        "snapshot_path\t"
        "development_markers"
    )
]

for rec in copied:

    lines.append(
        "\t".join(
            [
                rec["category"],
                str(rec["score"]),
                str(rec["size_bytes"]),
                rec["mtime"],
                rec["sha256"],
                rec["source_path"],
                rec["snapshot_path"],
                ",".join(
                    rec[
                        "development_markers"
                    ]
                ),
            ]
        )
    )


(
    SNAPSHOT
    /
    "migration_source_inventory.tsv"
).write_text(
    "\n".join(
        lines
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================================
# Markdown summary
# ============================================================================

md = []

md.append(
    "# pyPSDS-GAMMA v1.1 migration source freeze"
)

md.append("")

md.append(
    f"- Unique candidate files: **{len(copied)}**"
)

md.append("")

md.append(
    "## Candidate counts"
)

md.append("")

md.append(
    "| Category | Files |"
)

md.append(
    "|---|---:|"
)

for key in SIGNATURES:

    md.append(
        f"| `{key}` | "
        f"{category_counts.get(key, 0)} |"
    )


md.append("")

md.append(
    "## Highest-scoring candidates"
)

md.append("")

md.append(
    "| Category | Score | Source |"
)

md.append(
    "|---|---:|---|"
)


for category in SIGNATURES:

    rows = [
        r
        for r in copied
        if r["category"]
        ==
        category
    ]

    for rec in rows[:10]:

        md.append(
            f"| `{category}` | "
            f"{rec['score']} | "
            f"`{rec['source_path']}` |"
        )


md.append("")

md.append(
    "## Migration rule"
)

md.append("")

md.append(
    "These files are frozen development sources only. "
    "They must not be imported by the production package. "
    "Each validated algorithm is migrated into a semantic "
    "`pypsds` module and regression-tested before pipeline registration."
)


(
    SNAPSHOT
    /
    "README.md"
).write_text(
    "\n".join(md)
    + "\n",
    encoding="utf-8",
)


# ============================================================================
# Console
# ============================================================================

print()
print(
    "=" * 92
)

print(
    "MIGRATION SOURCE SUMMARY"
)

print(
    "=" * 92
)

for key in SIGNATURES:

    print(
        f"{key:<20} : "
        f"{category_counts.get(key, 0)}"
    )

print()

print(
    f"unique candidates     : {len(copied)}"
)

print(
    f"snapshot              : {SNAPSHOT}"
)

print()

print(
    "Top candidates:"
)

for category in SIGNATURES:

    rows = [
        r
        for r in copied
        if r["category"]
        ==
        category
    ]

    if not rows:

        print(
            f"  {category:<18}: NONE"
        )

        continue

    best = rows[0]

    print(
        f"  {category:<18}: "
        f"{best['source_path']}"
    )

print(
    "=" * 92
)

print(
    "V1.1 MIGRATION SOURCE FREEZE: PASS"
)

print(
    "=" * 92
)
PY


echo
echo "--------------------------------------------------------------------------------"
echo "Snapshot contents"
echo "--------------------------------------------------------------------------------"

find "${SNAPSHOT}" \
    -maxdepth 3 \
    -type f \
    -printf '%P\n' \
    | sort


echo
echo "--------------------------------------------------------------------------------"
echo "Preview"
echo "--------------------------------------------------------------------------------"

sed -n '1,160p' \
    "${SNAPSHOT}/README.md"


echo
echo "================================================================================"
echo "DONE"
echo "================================================================================"
