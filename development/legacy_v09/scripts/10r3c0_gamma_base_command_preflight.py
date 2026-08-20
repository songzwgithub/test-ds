#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu/Downloads")

MISSING_JSON = (
    ROOT
    / "psds"
    / "prototype_outputs"
    / "v09"
    / "scla_v09"
    / "pystamps_bridge"
    / "r3b_grid_adapter"
    / "missing_network_bases.json"
)

MAPPING_JSON = (
    ROOT
    / "psds"
    / "prototype_outputs"
    / "v09"
    / "scla_v09"
    / "pystamps_bridge"
    / "r3b_grid_adapter"
    / "network_base_mapping.json"
)

RSLC_DIR = ROOT / "RSLC"
DIFF_DIR = ROOT / "DIFF"


def find_par(date: str) -> Path | None:

    candidates = [
        RSLC_DIR / f"{date}.rslc.par",
        RSLC_DIR / f"{date}.slc.par",
    ]

    for p in candidates:
        if p.is_file():
            return p.resolve()

    for pattern in (
        f"*{date}*.rslc.par",
        f"*{date}*.slc.par",
    ):
        found = sorted(
            RSLC_DIR.glob(pattern)
        )

        if found:
            return found[0].resolve()

    return None


def command_usage(exe: str):

    path = shutil.which(exe)

    print()
    print("=" * 108)
    print(f"{exe} local installation")
    print("=" * 108)

    print(f"resolved path              : {path}")

    if path is None:
        return None

    p = Path(path)

    try:
        data = p.read_bytes()[:4096]

        is_text = (
            b"\x00" not in data
            and
            sum(
                32 <= b <= 126
                or b in (9, 10, 13)
                for b in data
            )
            / max(1, len(data))
            > 0.85
        )

    except Exception:
        is_text = False

    print(f"appears text/script        : {is_text}")

    if is_text:

        print()
        print("--- script header / usage-relevant lines ---")

        try:
            lines = p.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()

            printed = 0

            for i, line in enumerate(
                lines[:250],
                start=1,
            ):
                low = line.lower()

                if (
                    i <= 40
                    or
                    "usage" in low
                    or
                    "base_calc" in low
                    or
                    "base_orbit" in low
                    or
                    "slc" in low
                    or
                    "par" in low
                ):
                    print(
                        f"{i:4d}: {line}"
                    )

                    printed += 1

                    if printed >= 100:
                        break

        except Exception as exc:
            print(
                f"unable to inspect script: {exc}"
            )

    print()
    print("--- execution without arguments ---")

    try:
        proc = subprocess.run(
            [path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )

        print(
            f"return code                : "
            f"{proc.returncode}"
        )

        lines = proc.stdout.splitlines()

        for line in lines[:120]:
            print(line)

        if len(lines) > 120:
            print(
                f"... {len(lines)-120} more lines omitted"
            )

    except subprocess.TimeoutExpired:
        print(
            "command timed out after 15 s"
        )

    except Exception as exc:
        print(
            f"command invocation failed  : {exc}"
        )

    return path


def print_sample_base():

    mapped = json.loads(
        MAPPING_JSON.read_text(
            encoding="utf-8"
        )
    )

    if not mapped:
        return

    sample = Path(
        mapped[0]["base_file"]
    )

    print()
    print("=" * 108)
    print("Existing .base reference format")
    print("=" * 108)

    print(f"sample                     : {sample}")

    try:
        lines = sample.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

        for line in lines[:80]:
            print(line)

    except Exception as exc:
        print(
            f"unable to read sample .base: {exc}"
        )


def main():

    if not MISSING_JSON.is_file():
        raise RuntimeError(
            f"Missing R3b output: {MISSING_JSON}"
        )

    missing = json.loads(
        MISSING_JSON.read_text(
            encoding="utf-8"
        )
    )

    print("=" * 108)
    print(
        "Step 10R3c0 - GAMMA missing-baseline "
        "generation preflight"
    )
    print("=" * 108)

    print(
        f"missing network edges      : "
        f"{len(missing)}"
    )

    print(
        f"RSLC directory             : "
        f"{RSLC_DIR}"
    )

    print(
        f"DIFF directory             : "
        f"{DIFF_DIR}"
    )

    print()
    print("=" * 108)
    print("Missing edge RSLC parameter coverage")
    print("=" * 108)

    missing_par = []

    records = []

    for item in missing:

        edge = int(
            item["edge"]
        )

        d1 = str(
            item["date_i"]
        )

        d2 = str(
            item["date_j"]
        )

        p1 = find_par(d1)
        p2 = find_par(d2)

        ok = (
            p1 is not None
            and
            p2 is not None
        )

        print(
            f"edge {edge:3d}  "
            f"{d1} -> {d2}"
        )

        print(
            f"          par1: "
            f"{p1 if p1 else 'MISSING'}"
        )

        print(
            f"          par2: "
            f"{p2 if p2 else 'MISSING'}"
        )

        if not ok:
            missing_par.append(
                edge
            )

        records.append(
            {
                "edge":
                    edge,

                "date_i":
                    d1,

                "date_j":
                    d2,

                "par_i":
                    str(p1)
                    if p1
                    else None,

                "par_j":
                    str(p2)
                    if p2
                    else None,
            }
        )

    print()
    print(
        f"edges with both RSLC par   : "
        f"{len(records)-len(missing_par)}/"
        f"{len(records)}"
    )

    print(
        f"edges missing RSLC par     : "
        f"{len(missing_par)}"
    )

    command_usage(
        "base_calc"
    )

    command_usage(
        "base_orbit"
    )

    print_sample_base()

    outdir = (
        MISSING_JSON.parent
        /
        "gamma_base_preflight"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-gamma-base-preflight-v09",

        "missing_edges":
            len(records),

        "edges_missing_rslc_par":
            len(missing_par),

        "records":
            records,

        "base_calc":
            shutil.which(
                "base_calc"
            ),

        "base_orbit":
            shutil.which(
                "base_orbit"
            ),

        "base_files_generated":
            False,
    }

    manifest_path = (
        outdir
        /
        "gamma_base_preflight_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 108)

    if (
        len(missing_par) == 0
        and
        shutil.which(
            "base_calc"
        )
        is not None
    ):
        status = (
            "PASS_READY_TO_GENERATE_16_BASES"
        )
    else:
        status = (
            "REVIEW_BASE_GENERATION_INPUTS"
        )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10R3c0 STATUS: {status}"
    )

    print(
        "No .base file was created or modified."
    )


if __name__ == "__main__":
    main()
