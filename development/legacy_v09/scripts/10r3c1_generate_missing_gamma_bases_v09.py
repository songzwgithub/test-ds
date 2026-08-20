#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/ubuntu/Downloads")

R3B = (
    ROOT
    / "psds"
    / "prototype_outputs"
    / "v09"
    / "scla_v09"
    / "pystamps_bridge"
    / "r3b_grid_adapter"
)

MAPPED_JSON = (
    R3B
    / "network_base_mapping.json"
)

PREFLIGHT_JSON = (
    R3B
    / "gamma_base_preflight"
    / "gamma_base_preflight_manifest.json"
)


def resolve_pystamps_source(explicit: str | None) -> Path:

    candidates = []

    if explicit:
        candidates.append(
            Path(explicit).expanduser()
        )

    candidates.extend([
        Path("/home/ubuntu/software/pystamps-gamma"),
        Path("/home/ubuntu/software/pystamps-gamma-main"),
        Path.home() / "software" / "pystamps-gamma",
    ])

    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            continue

        if (
            (p / "pystamps").is_dir()
            and
            (p / "pystamps" / "prep").is_dir()
        ):
            sys.path.insert(
                0,
                str(p),
            )
            return p

    raise RuntimeError(
        "Cannot locate pystamps-gamma. "
        "Use --pystamps-source."
    )


def run_base_orbit(
    exe: str,
    par1: Path,
    par2: Path,
    output: Path,
):

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.unlink(
        missing_ok=True
    )

    cmd = [
        exe,
        str(par1),
        str(par2),
        str(output),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "base_orbit failed:\n"
            +
            " ".join(cmd)
            +
            "\n"
            +
            proc.stdout
        )

    if (
        not output.is_file()
        or output.stat().st_size == 0
    ):
        raise RuntimeError(
            f"base_orbit did not create valid output: "
            f"{output}"
        )

    return proc.stdout


def model_difference(
    ref,
    test,
):

    dtcn = np.asarray(
        test.baseline_tcn,
        dtype=np.float64,
    ) - np.asarray(
        ref.baseline_tcn,
        dtype=np.float64,
    )

    drate = np.asarray(
        test.baseline_rate_tcn,
        dtype=np.float64,
    ) - np.asarray(
        ref.baseline_rate_tcn,
        dtype=np.float64,
    )

    return (
        float(
            np.max(
                np.abs(dtcn)
            )
        ),
        float(
            np.max(
                np.abs(drate)
            )
        ),
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pystamps-source",
        default=None,
    )

    ap.add_argument(
        "--keep-parity-temp",
        action="store_true",
    )

    args = ap.parse_args()

    pystamps_source = (
        resolve_pystamps_source(
            args.pystamps_source
        )
    )

    from pystamps.prep.gamma_geometry import (
        read_baseline_model,
    )

    base_orbit = shutil.which(
        "base_orbit"
    )

    if base_orbit is None:
        raise RuntimeError(
            "base_orbit not found in PATH"
        )

    if not MAPPED_JSON.is_file():
        raise RuntimeError(
            f"Missing R3b mapping: {MAPPED_JSON}"
        )

    if not PREFLIGHT_JSON.is_file():
        raise RuntimeError(
            f"Missing R3c0 manifest: {PREFLIGHT_JSON}"
        )

    mapped = json.loads(
        MAPPED_JSON.read_text(
            encoding="utf-8"
        )
    )

    preflight = json.loads(
        PREFLIGHT_JSON.read_text(
            encoding="utf-8"
        )
    )

    missing_records = (
        preflight[
            "records"
        ]
    )

    if len(mapped) != 92:
        print(
            f"WARNING: expected 92 mapped edges, "
            f"found {len(mapped)}"
        )

    if len(missing_records) != 16:
        print(
            f"WARNING: expected 16 missing edges, "
            f"found {len(missing_records)}"
        )

    outdir = (
        R3B
        /
        "generated_bases"
    )

    parity_dir = (
        outdir
        /
        "_parity_temp"
    )

    generated_dir = (
        outdir
        /
        "current_network_missing"
    )

    parity_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print(
        "Step 10R3c1 - GAMMA base_orbit parity "
        "and missing-baseline generation"
    )
    print("=" * 112)

    print(
        f"pystamps source            : "
        f"{pystamps_source}"
    )

    print(
        f"base_orbit                 : "
        f"{base_orbit}"
    )

    print(
        f"existing mapped edges      : "
        f"{len(mapped)}"
    )

    print(
        f"missing edges              : "
        f"{len(missing_records)}"
    )

    # ========================================================
    # PASS 1
    #
    # Re-generate the existing 92 baselines temporarily and
    # compare mature GAMMA baseline parameters.
    #
    # This verifies:
    #   - SLC1/SLC2 orientation
    #   - current base_orbit convention
    #   - parser compatibility
    #   - GAMMA-version consistency
    # ========================================================

    print()
    print("=" * 112)
    print(
        "PASS 1 - Existing .base parity"
    )
    print("=" * 112)

    parity_results = []

    max_tcn = 0.0
    max_rate = 0.0

    failed_parity = []

    for k, item in enumerate(
        mapped,
        start=1,
    ):

        edge = int(
            item["edge"]
        )

        d1 = str(
            item["date_i"]
        )

        d2 = str(
            item["date_j"]
        )

        old_base = Path(
            item["base_file"]
        )

        par1 = (
            ROOT
            /
            "RSLC"
            /
            f"{d1}.rslc.par"
        )

        par2 = (
            ROOT
            /
            "RSLC"
            /
            f"{d2}.rslc.par"
        )

        if not par1.is_file():
            raise RuntimeError(
                f"Missing RSLC par: {par1}"
            )

        if not par2.is_file():
            raise RuntimeError(
                f"Missing RSLC par: {par2}"
            )

        new_base = (
            parity_dir
            /
            f"{d1}_{d2}.base"
        )

        run_base_orbit(
            base_orbit,
            par1,
            par2,
            new_base,
        )

        old_model = (
            read_baseline_model(
                old_base
            )
        )

        new_model = (
            read_baseline_model(
                new_base
            )
        )

        dtcn, drate = (
            model_difference(
                old_model,
                new_model,
            )
        )

        max_tcn = max(
            max_tcn,
            dtcn,
        )

        max_rate = max(
            max_rate,
            drate,
        )

        # Text values are written at finite precision.
        # These thresholds are deliberately much tighter
        # than any scientifically relevant baseline error.
        passed = (
            dtcn <= 1.0e-4
            and
            drate <= 1.0e-6
        )

        if not passed:
            failed_parity.append(
                edge
            )

        parity_results.append(
            {
                "edge":
                    edge,

                "date_i":
                    d1,

                "date_j":
                    d2,

                "existing_base":
                    str(old_base),

                "temporary_base":
                    str(new_base),

                "max_abs_tcn_difference_m":
                    dtcn,

                "max_abs_rate_difference_m_per_s":
                    drate,

                "pass":
                    passed,
            }
        )

        if (
            k == 1
            or
            k % 10 == 0
            or
            k == len(mapped)
        ):
            print(
                f"  {k:3d}/{len(mapped)}  "
                f"max ΔTCN={max_tcn:.3e} m  "
                f"max Δrate={max_rate:.3e} m/s"
            )

    print()
    print(
        f"parity edges passed        : "
        f"{len(mapped)-len(failed_parity)}/"
        f"{len(mapped)}"
    )

    print(
        f"maximum |Δ TCN|            : "
        f"{max_tcn:.9e} m"
    )

    print(
        f"maximum |Δ baseline rate|  : "
        f"{max_rate:.9e} m/s"
    )

    if failed_parity:

        print(
            f"failed parity edges        : "
            f"{failed_parity}"
        )

        status = (
            "REVIEW_BASE_ORBIT_PARITY"
        )

        # Do not generate the missing 16 if parity fails.
        manifest = {
            "format":
                "pyPSDS-GAMMA-base-orbit-generation-v09",

            "status":
                status,

            "existing_edge_parity": {
                "tested":
                    len(mapped),

                "failed":
                    len(
                        failed_parity
                    ),

                "max_abs_tcn_difference_m":
                    max_tcn,

                "max_abs_rate_difference_m_per_s":
                    max_rate,
            },

            "missing_bases_generated":
                0,

            "production_files_modified":
                False,
        }

        manifest_path = (
            outdir
            /
            "base_generation_manifest.json"
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
        print(
            f"STEP 10R3c1 STATUS: "
            f"{status}"
        )

        print(
            "Missing baselines were NOT generated."
        )

        return

    # ========================================================
    # PASS 2
    #
    # Generate only the 16 current-network missing baselines.
    # Never modify /Downloads/DIFF.
    # ========================================================

    print()
    print("=" * 112)
    print(
        "PASS 2 - Generate missing current-network baselines"
    )
    print("=" * 112)

    generated = []

    for k, item in enumerate(
        missing_records,
        start=1,
    ):

        edge = int(
            item["edge"]
        )

        d1 = str(
            item["date_i"]
        )

        d2 = str(
            item["date_j"]
        )

        par1 = Path(
            item["par_i"]
        )

        par2 = Path(
            item["par_j"]
        )

        if not par1.is_file():
            raise RuntimeError(
                f"Missing par_i: {par1}"
            )

        if not par2.is_file():
            raise RuntimeError(
                f"Missing par_j: {par2}"
            )

        output = (
            generated_dir
            /
            f"{d1}_{d2}.base"
        )

        run_base_orbit(
            base_orbit,
            par1,
            par2,
            output,
        )

        model = (
            read_baseline_model(
                output
            )
        )

        finite = (
            np.all(
                np.isfinite(
                    model.baseline_tcn
                )
            )
            and
            np.all(
                np.isfinite(
                    model.baseline_rate_tcn
                )
            )
        )

        if not finite:
            raise RuntimeError(
                f"Generated baseline has "
                f"non-finite model: {output}"
            )

        generated.append(
            {
                "edge":
                    edge,

                "date_i":
                    d1,

                "date_j":
                    d2,

                "par_i":
                    str(par1),

                "par_j":
                    str(par2),

                "base_file":
                    str(
                        output.resolve()
                    ),

                "orientation":
                    1,

                "initial_baseline_TCN_m":
                    np.asarray(
                        model.baseline_tcn,
                        dtype=float,
                    ).tolist(),

                "initial_baseline_rate_m_per_s":
                    np.asarray(
                        model.baseline_rate_tcn,
                        dtype=float,
                    ).tolist(),
            }
        )

        print(
            f"  {k:2d}/{len(missing_records)}  "
            f"edge {edge:3d}  "
            f"{d1} -> {d2}"
        )

    # ========================================================
    # Build complete 108-edge baseline source manifest.
    #
    # Existing 92 remain untouched in DIFF.
    # Missing 16 use isolated generated copies.
    # ========================================================

    complete = {}

    for item in mapped:

        edge = int(
            item["edge"]
        )

        complete[
            edge
        ] = {
            "edge":
                edge,

            "date_i":
                str(
                    item["date_i"]
                ),

            "date_j":
                str(
                    item["date_j"]
                ),

            "base_file":
                str(
                    Path(
                        item["base_file"]
                    ).resolve()
                ),

            "source":
                "existing_DIFF",

            "orientation":
                int(
                    item.get(
                        "orientation",
                        1,
                    )
                ),
        }

    for item in generated:

        edge = int(
            item["edge"]
        )

        complete[
            edge
        ] = {
            "edge":
                edge,

            "date_i":
                item["date_i"],

            "date_j":
                item["date_j"],

            "base_file":
                item["base_file"],

            "source":
                "generated_bridge",

            "orientation":
                1,
        }

    complete_ordered = [
        complete[e]
        for e in sorted(
            complete
        )
    ]

    if len(
        complete_ordered
    ) != 108:

        raise RuntimeError(
            "Complete baseline source "
            f"contains {len(complete_ordered)} "
            "edges instead of 108"
        )

    expected_edges = list(
        range(
            1,
            109,
        )
    )

    actual_edges = [
        int(x["edge"])
        for x in complete_ordered
    ]

    if actual_edges != expected_edges:
        raise RuntimeError(
            "Complete baseline edge numbering "
            "is not exactly 1..108"
        )

    # Validate every final source through mature parser.
    final_invalid = []

    for item in complete_ordered:

        path = Path(
            item["base_file"]
        )

        try:
            model = (
                read_baseline_model(
                    path
                )
            )

            good = (
                np.all(
                    np.isfinite(
                        model.baseline_tcn
                    )
                )
                and
                np.all(
                    np.isfinite(
                        model.baseline_rate_tcn
                    )
                )
            )

        except Exception:
            good = False

        if not good:
            final_invalid.append(
                int(
                    item["edge"]
                )
            )

    if final_invalid:
        raise RuntimeError(
            "Invalid final baseline sources "
            f"for edges: {final_invalid}"
        )

    complete_path = (
        outdir
        /
        "current_network_108_baseline_sources.json"
    )

    complete_path.write_text(
        json.dumps(
            complete_ordered,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    parity_path = (
        outdir
        /
        "existing_base_orbit_parity.json"
    )

    parity_path.write_text(
        json.dumps(
            parity_results,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    generated_path = (
        outdir
        /
        "generated_missing_baselines.json"
    )

    generated_path.write_text(
        json.dumps(
            generated,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # Remove temporary 92 re-generated parity files unless requested.
    if not args.keep_parity_temp:
        shutil.rmtree(
            parity_dir,
            ignore_errors=True,
        )

    status = (
        "PASS_108_BASELINE_SOURCES_READY"
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-base-orbit-generation-v09",

        "status":
            status,

        "base_orbit":
            base_orbit,

        "existing_edge_parity": {
            "tested":
                len(mapped),

            "failed":
                0,

            "max_abs_tcn_difference_m":
                max_tcn,

            "max_abs_rate_difference_m_per_s":
                max_rate,
        },

        "baseline_sources": {
            "existing":
                len(mapped),

            "generated":
                len(generated),

            "total":
                len(
                    complete_ordered
                ),
        },

        "generated_directory":
            str(
                generated_dir
            ),

        "complete_source_manifest":
            str(
                complete_path
            ),

        "original_DIFF_modified":
            False,

        "stage7_executed":
            False,

        "phase_modified":
            False,
    }

    manifest_path = (
        outdir
        /
        "base_generation_manifest.json"
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
    print("=" * 112)
    print(
        "Final current-network baseline coverage"
    )
    print("=" * 112)

    print(
        f"existing DIFF bases         : "
        f"{len(mapped)}"
    )

    print(
        f"generated bridge bases     : "
        f"{len(generated)}"
    )

    print(
        f"total baseline sources     : "
        f"{len(complete_ordered)}/108"
    )

    print(
        f"invalid final sources      : "
        f"{len(final_invalid)}"
    )

    print(
        f"original DIFF modified     : "
        f"False"
    )

    print()
    print(
        f"complete mapping           : "
        f"{complete_path}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10R3c1 STATUS: {status}"
    )

    print(
        "No point-wise Bperp matrix "
        "has been generated yet."
    )

    print(
        "No Stage-7/Stage-8 correction "
        "has been applied."
    )


if __name__ == "__main__":
    main()
