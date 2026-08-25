#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pyPSDS-GAMMA-v1.0"

if [[ ! -d "${ROOT}" ]]; then
    echo "ERROR: repository not found: ${ROOT}" >&2
    exit 1
fi

if [[ ! -f "${ROOT}/pyproject.toml" ]]; then
    echo "ERROR: pyproject.toml not found" >&2
    exit 1
fi

if [[ ! -d "${ROOT}/pypsds" ]]; then
    echo "ERROR: pypsds package not found" >&2
    exit 1
fi

cd "${ROOT}"

OUT="${ROOT}/docs/release_v1_1"
mkdir -p "${OUT}"

echo "================================================================================"
echo "pyPSDS-GAMMA v1.1 RELEASE INVENTORY"
echo "================================================================================"
echo "Repository : ${ROOT}"
echo "Output     : ${OUT}"
echo "Mode       : READ-ONLY SOURCE AUDIT"
echo

python - <<'PY'
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import tomllib


ROOT = Path("/home/ubuntu/software/pyPSDS-GAMMA-v1.0")
OUT = ROOT / "docs" / "release_v1_1"

OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


# =============================================================================
# pyproject
# =============================================================================

with (ROOT / "pyproject.toml").open("rb") as f:
    pyproject = tomllib.load(f)

project = pyproject.get("project", {})

project_summary = {
    "name": project.get("name"),
    "version": project.get("version"),
    "requires_python": project.get("requires-python"),
    "dependencies": project.get("dependencies", []),
    "scripts": project.get("scripts", {}),
    "build_system": pyproject.get("build-system", {}),
}

(OUT / "pyproject_summary.json").write_text(
    json.dumps(project_summary, indent=2, ensure_ascii=False) + "\n"
)


# =============================================================================
# Repository inventory
# =============================================================================

skip_dirs = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "build",
    "dist",
}

source_files = []

for p in ROOT.rglob("*"):
    if not p.is_file():
        continue

    rp = p.relative_to(ROOT)

    if any(part in skip_dirs for part in rp.parts):
        continue

    if p.suffix.lower() not in TEXT_SUFFIXES:
        continue

    source_files.append(p)

source_files.sort()


# =============================================================================
# Classification
#
# This is a migration aid only. It does NOT move files.
# =============================================================================

DOMAIN_RULES = [
    (
        "preparation",
        {
            "rslc",
            "slc",
            "gamma_geometry",
            "geometry",
            "dem",
            "incidence",
            "lookup",
            "coordinate",
            "geocode",
        },
    ),
    (
        "selection",
        {
            "shp",
            "glrt",
            "candidate",
            "selection",
            "select",
            "ps",
            "ds",
        },
    ),
    (
        "phase_linking",
        {
            "emi",
            "phase_link",
            "phase_linking",
            "covariance",
            "coherence",
            "pl_",
        },
    ),
    (
        "network",
        {
            "network",
            "ifg",
            "graph",
            "edge",
            "anchor",
        },
    ),
    (
        "unwrapping",
        {
            "unwrap",
            "unwrapping",
            "closure",
            "integer",
        },
    ),
    (
        "correction",
        {
            "gacos",
            "atmos",
            "scla",
            "baseline",
            "bperp",
            "scn",
            "dem_error",
            "residual_dem",
        },
    ),
    (
        "inversion",
        {
            "timeseries",
            "time_series",
            "invert",
            "inversion",
            "velocity",
            "los",
        },
    ),
    (
        "products",
        {
            "product",
            "export",
            "parquet",
            "geopackage",
            "gpkg",
            "delivery",
        },
    ),
    (
        "quality",
        {
            "quality",
            "audit",
            "validate",
            "doctor",
            "check",
            "manifest",
            "qa",
        },
    ),
]


def classify(path: Path, text: str) -> str:
    s = (rel(path) + "\n" + text[:12000]).lower()

    scores = []

    for domain, words in DOMAIN_RULES:
        score = sum(s.count(w) for w in words)
        scores.append((score, domain))

    scores.sort(reverse=True)

    if not scores or scores[0][0] == 0:
        return "core_or_review"

    return scores[0][1]


# =============================================================================
# Development-name audit
# =============================================================================

DEV_PATTERNS = {
    "P-stage": re.compile(r"\bP(?:8|9|10|11|12|13|14|15)[-_A-Za-z0-9]*\b", re.I),
    "prototype": re.compile(r"\bprototype(?:_outputs?)?\b", re.I),
    "smoke": re.compile(r"\bsmoke\b", re.I),
    "benchmark": re.compile(r"\bbench(?:mark)?\b", re.I),
    "temporary": re.compile(r"\b(?:tmp|temporary)\b", re.I),
    "experiment": re.compile(r"\b(?:experiment|experimental|rejected_experiment)\b", re.I),
    "dated-version": re.compile(r"\bv0[0-9]\b|\bv1[0-9]\b", re.I),
}


# =============================================================================
# Hard-coded path audit
# =============================================================================

ABS_PATH = re.compile(
    r"""(?P<path>
        /(?:home|mnt|media|tmp|data|opt|usr/local)/
        [A-Za-z0-9_./+\-]+
    )""",
    re.X,
)

PROJECT_SPECIFIC = re.compile(
    r"""(?i)
    (?:
        /home/ubuntu/Downloads
        |
        /home/ubuntu/software
        |
        Downloads/psds
        |
        prototype_outputs
    )
    """
)


# =============================================================================
# Analyze
# =============================================================================

rows = []
dev_hits = []
path_hits = []
manifest_files = []

for p in source_files:
    text = read_text(p)

    try:
        nlines = len(text.splitlines())
    except Exception:
        nlines = 0

    domain = classify(p, text)

    file_dev = []

    for label, pat in DEV_PATTERNS.items():
        matches = sorted(set(m.group(0) for m in pat.finditer(text)))

        if matches:
            file_dev.extend(f"{label}:{x}" for x in matches[:20])

            for x in matches[:100]:
                dev_hits.append(
                    {
                        "file": rel(p),
                        "type": label,
                        "token": x,
                    }
                )

    abs_matches = sorted(set(m.group("path") for m in ABS_PATH.finditer(text)))

    for x in abs_matches[:100]:
        path_hits.append(
            {
                "file": rel(p),
                "path": x,
                "project_specific": bool(PROJECT_SPECIFIC.search(x)),
            }
        )

    rows.append(
        {
            "file": rel(p),
            "size_bytes": p.stat().st_size,
            "lines": nlines,
            "suggested_domain": domain,
            "development_hit_count": len(file_dev),
            "hardcoded_path_count": len(abs_matches),
        }
    )

    manifest_files.append(
        {
            "path": rel(p),
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
    )


# =============================================================================
# Root generated-artifact audit
# =============================================================================

generated_candidates = []

root_candidates = [
    ROOT / "build",
    ROOT / "dist",
    ROOT / "pypsds_gamma.egg-info",
    ROOT / "SLC2pt.log",
    ROOT / "TEST_STATUS.txt",
]

for p in root_candidates:
    if p.exists():
        generated_candidates.append(
            {
                "path": rel(p),
                "kind": "directory" if p.is_dir() else "file",
                "release_action": (
                    "exclude_from_release"
                    if p.name != "TEST_STATUS.txt"
                    else "review_or_move_to_docs"
                ),
            }
        )


# =============================================================================
# Save machine-readable results
# =============================================================================

(OUT / "source_manifest.json").write_text(
    json.dumps(
        {
            "repository": str(ROOT),
            "project": project_summary,
            "files": manifest_files,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
)

(OUT / "module_candidates.tsv").write_text(
    "file\tsize_bytes\tlines\tsuggested_domain\tdevelopment_hit_count\thardcoded_path_count\n"
    + "".join(
        (
            f"{x['file']}\t"
            f"{x['size_bytes']}\t"
            f"{x['lines']}\t"
            f"{x['suggested_domain']}\t"
            f"{x['development_hit_count']}\t"
            f"{x['hardcoded_path_count']}\n"
        )
        for x in rows
    )
)

(OUT / "development_name_hits.tsv").write_text(
    "file\ttype\ttoken\n"
    + "".join(
        f"{x['file']}\t{x['type']}\t{x['token']}\n"
        for x in dev_hits
    )
)

(OUT / "hardcoded_paths.tsv").write_text(
    "file\tpath\tproject_specific\n"
    + "".join(
        f"{x['file']}\t{x['path']}\t{x['project_specific']}\n"
        for x in path_hits
    )
)

(OUT / "generated_artifacts.json").write_text(
    json.dumps(generated_candidates, indent=2, ensure_ascii=False) + "\n"
)


# =============================================================================
# Markdown report
# =============================================================================

domain_counts = {}

for x in rows:
    domain_counts[x["suggested_domain"]] = (
        domain_counts.get(x["suggested_domain"], 0) + 1
    )

project_specific_paths = [
    x for x in path_hits if x["project_specific"]
]


md = []

md.append("# pyPSDS-GAMMA v1.1 release inventory")
md.append("")
md.append(f"- Repository: `{ROOT}`")
md.append(f"- Project name: `{project_summary.get('name')}`")
md.append(f"- Current version: `{project_summary.get('version')}`")
md.append(f"- Python requirement: `{project_summary.get('requires_python')}`")
md.append(f"- Text/source files audited: **{len(rows)}**")
md.append(f"- Development-name hits: **{len(dev_hits)}**")
md.append(f"- Absolute-path hits: **{len(path_hits)}**")
md.append(f"- Project-specific path hits: **{len(project_specific_paths)}**")
md.append("")

md.append("## Current CLI entry points")
md.append("")

scripts = project_summary.get("scripts") or {}

if scripts:
    md.append("| Command | Python entry |")
    md.append("|---|---|")
    for k, v in sorted(scripts.items()):
        md.append(f"| `{k}` | `{v}` |")
else:
    md.append("_No `[project.scripts]` entries detected._")

md.append("")
md.append("## Suggested functional domains")
md.append("")
md.append("| Domain | Files |")
md.append("|---|---:|")

for domain, count in sorted(domain_counts.items()):
    md.append(f"| `{domain}` | {count} |")

md.append("")
md.append("## Source/module inventory")
md.append("")
md.append("| File | Lines | Suggested domain | Dev hits | Hard paths |")
md.append("|---|---:|---|---:|---:|")

for x in rows:
    md.append(
        f"| `{x['file']}` | "
        f"{x['lines']} | "
        f"`{x['suggested_domain']}` | "
        f"{x['development_hit_count']} | "
        f"{x['hardcoded_path_count']} |"
    )

md.append("")
md.append("## Generated/build artifacts found at repository root")
md.append("")

if generated_candidates:
    md.append("| Path | Type | Recommended release action |")
    md.append("|---|---|---|")
    for x in generated_candidates:
        md.append(
            f"| `{x['path']}` | "
            f"{x['kind']} | "
            f"`{x['release_action']}` |"
        )
else:
    md.append("_None detected._")

md.append("")
md.append("## v1.1 naming recommendation")
md.append("")
md.append(
    "Do **not** introduce a second numeric stage hierarchy such as "
    "`01_`, `02_`, ... at the package/module level."
)
md.append("")
md.append("Recommended public functional domains:")
md.append("")
md.append("- `preparation` — input discovery, radar/DEM geometry")
md.append("- `selection` — PS/DS candidate and SHP selection")
md.append("- `phase_linking` — covariance/EMI/phase optimization")
md.append("- `network` — interferometric network and graph operations")
md.append("- `unwrapping` — phase unwrapping")
md.append("- `correction` — GACOS, SCLA/residual DEM, SCN")
md.append("- `inversion` — acquisition phase, time series, LOS/velocity")
md.append("- `products` — point products, Parquet, GeoPackage, quicklooks")
md.append("- `quality` — validation, manifests, diagnostics")
md.append("")
md.append(
    "Existing internal stage identifiers may remain where they are part of "
    "the computational dependency graph, but development identifiers such "
    "as `P15-6B2`, `smoke`, `benchmark`, and prototype-version names should "
    "not appear in the public CLI, configuration schema, or final output names."
)

md.append("")
md.append("## Next migration gate")
md.append("")
md.append(
    "Before moving any file, define a v1.1 configuration schema and a "
    "regression baseline from the current frozen production run. "
    "Every migrated module must reproduce that baseline before the old "
    "implementation is retired."
)
md.append("")

(OUT / "inventory.md").write_text(
    "\n".join(md) + "\n"
)


# =============================================================================
# Console summary
# =============================================================================

print("=" * 92)
print("V1.1 RELEASE INVENTORY SUMMARY")
print("=" * 92)

print(f"project name                    : {project_summary.get('name')}")
print(f"project version                 : {project_summary.get('version')}")
print(f"python requirement              : {project_summary.get('requires_python')}")
print(f"source/text files               : {len(rows)}")
print(f"development-name hits           : {len(dev_hits)}")
print(f"absolute-path hits              : {len(path_hits)}")
print(f"project-specific path hits      : {len(project_specific_paths)}")
print()

print("functional-domain candidates:")
for domain, count in sorted(domain_counts.items()):
    print(f"  {domain:<20} {count:>5}")

print()
print("generated/build artifacts:")
for x in generated_candidates:
    print(
        f"  {x['path']:<35} "
        f"{x['release_action']}"
    )

print()
print("reports:")
for name in (
    "inventory.md",
    "module_candidates.tsv",
    "development_name_hits.tsv",
    "hardcoded_paths.tsv",
    "source_manifest.json",
    "pyproject_summary.json",
    "generated_artifacts.json",
):
    print(f"  {OUT / name}")

print("=" * 92)
print("V1.1 INVENTORY RESULT: PASS_READ_ONLY_RELEASE_AUDIT")
print("=" * 92)
PY


echo
echo "--------------------------------------------------------------------------------"
echo "Current repository top-level"
echo "--------------------------------------------------------------------------------"

find . \
    -maxdepth 1 \
    -mindepth 1 \
    -printf '%f\n' \
    | sort


echo
echo "--------------------------------------------------------------------------------"
echo "Current VERSION"
echo "--------------------------------------------------------------------------------"

cat VERSION || true


echo
echo "--------------------------------------------------------------------------------"
echo "Git state"
echo "--------------------------------------------------------------------------------"

if [[ -d .git ]]; then
    git status --short
else
    echo "No local .git directory detected."
fi


echo
echo "================================================================================"
echo "DONE"
echo "================================================================================"
