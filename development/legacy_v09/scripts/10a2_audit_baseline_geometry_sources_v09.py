#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from pypsds.prototype import open_from_config


KEYWORDS = (
    "bperp",
    "baseline",
    "base_calc",
    "base_orbit",
    ".base",
    "bp1.mat",
    "bp2.mat",
)


def interesting(path: Path) -> bool:
    name = path.name.lower()
    return any(k.lower() in name for k in KEYWORDS)


def preview_text(path: Path, max_lines=8):
    try:
        if path.stat().st_size > 5_000_000:
            return []

        lines = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

        return lines[:max_lines]
    except Exception:
        return []


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        _,
    ) = open_from_config(
        args.config
    )

    root = Path(paths.output_dir) / "v09"

    outdir = (
        root
        / "scla_v09"
        / "baseline_source_audit"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 108)
    print(
        "Step 10a2 - Baseline / topographic-geometry source audit"
    )
    print("=" * 108)

    print(f"config                     : {config_path}")
    print(f"acquisitions               : {len(stack.dates)}")

    # ========================================================
    # 1. Confirm network baseline semantics
    # ========================================================

    pairs_path = (
        root
        / "network"
        / "pairs.csv"
    )

    with pairs_path.open() as f:
        pair_rows = list(
            csv.DictReader(f)
        )

    db = [
        float(r["delta_bperp_m"])
        for r in pair_rows
    ]

    npos = sum(x > 0 for x in db)
    nneg = sum(x < 0 for x in db)

    print()
    print("Network baseline field")
    print("-" * 108)

    print(
        "field                      : "
        "delta_bperp_m"
    )

    print(
        f"positive / negative        : "
        f"{npos} / {nneg}"
    )

    if npos == len(db) and nneg == 0:
        semantics = "ABSOLUTE_PAIR_BASELINE_MAGNITUDE"
    else:
        semantics = "SIGNED_OR_MIXED_REVIEW"

    print(
        f"interpreted semantics      : "
        f"{semantics}"
    )

    # ========================================================
    # 2. Search candidate existing products
    # ========================================================

    search_roots = []

    for p in (
        Path(paths.work_dir),
        Path(paths.data_dir),
        root,
    ):
        p = p.expanduser().resolve()

        if (
            p.exists()
            and
            p not in search_roots
        ):
            search_roots.append(p)

    found = []

    seen = set()

    print()
    print("Candidate existing baseline products")
    print("-" * 108)

    for search_root in search_roots:

        try:
            iterator = search_root.rglob("*")
        except Exception:
            continue

        for p in iterator:

            try:
                if not p.is_file():
                    continue
            except Exception:
                continue

            if not interesting(p):
                continue

            rp = str(p.resolve())

            if rp in seen:
                continue

            seen.add(rp)

            try:
                size = p.stat().st_size
            except Exception:
                size = -1

            preview = preview_text(p)

            found.append({
                "path":
                    rp,

                "size_bytes":
                    int(size),

                "preview":
                    preview,
            })

    found.sort(
        key=lambda x:
            x["path"]
    )

    for item in found[:100]:

        print()
        print(
            f"FILE: {item['path']}"
        )

        print(
            f"SIZE: {item['size_bytes']}"
        )

        for line in item["preview"]:
            print(
                "  " + line[:220]
            )

    print()
    print(
        f"candidate files found      : "
        f"{len(found)}"
    )

    # ========================================================
    # 3. GAMMA command availability
    # ========================================================

    commands = (
        "base_calc",
        "base_orbit",
        "phase_sim_orb_pt",
        "data2pt",
    )

    command_status = {}

    print()
    print("GAMMA executable availability")
    print("-" * 108)

    for cmd in commands:

        path = shutil.which(cmd)

        command_status[
            cmd
        ] = path

        print(
            f"{cmd:24s}: "
            f"{path or 'NOT FOUND'}"
        )

    # ========================================================
    # 4. RSLC/PAR availability
    # ========================================================

    print()
    print("Acquisition parameter files")
    print("-" * 108)

    missing_par = []

    for rec in stack.records:

        par = Path(rec.par)

        ok = par.is_file()

        print(
            f"{rec.date}: "
            f"{'OK' if ok else 'MISSING'} "
            f"{par}"
        )

        if not ok:
            missing_par.append(
                str(par)
            )

    # ========================================================
    # Save
    # ========================================================

    manifest = {
        "format":
            "pyPSDS-GAMMA-baseline-source-audit-v09",

        "status":
            "AUDIT_ONLY",

        "network_pair_baseline": {
            "field":
                "delta_bperp_m",

            "positive":
                int(npos),

            "negative":
                int(nneg),

            "semantics":
                semantics,

            "usable_for_signed_SCLA":
                False
                if semantics
                ==
                "ABSOLUTE_PAIR_BASELINE_MAGNITUDE"
                else None,
        },

        "candidate_existing_files":
            found,

        "gamma_commands":
            command_status,

        "missing_acquisition_par_files":
            missing_par,

        "recommended_next_source_order": [
            "existing point-wise signed bperp product if provenance is valid",
            "GAMMA orbit/geometry-derived point-wise topographic sensitivity",
            "scalar signed acquisition baseline only as diagnostic fallback",
        ],

        "phase_modified":
            False,
    }

    json_path = (
        outdir
        / "baseline_source_audit.json"
    )

    json_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n"
    )

    print()
    print(
        f"manifest                   : "
        f"{json_path}"
    )

    print()
    print(
        "STEP 10a2 STATUS: PASS / "
        "SOURCE AUDIT ONLY"
    )

    print(
        "No baseline inversion or SCLA "
        "correction has been applied."
    )


if __name__ == "__main__":
    main()
