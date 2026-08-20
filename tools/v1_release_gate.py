#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# Repository root is derived from this tool location.
ROOT = Path(__file__).resolve().parents[1]

# These are configured by main() from CLI/config.
CONFIG = ROOT / "config" / "pypsds.yaml"
OLD_ROOT = None
NEW_OUTPUT = None
NEW_ROOT = None


def configure_release_paths(
    *,
    config,
    old_root=None,
    new_output=None,
):
    """
    Resolve all release-gate paths without machine-specific
    source-code constants.

    new_output defaults to project.output_dir from config.
    """

    global CONFIG
    global OLD_ROOT
    global NEW_OUTPUT
    global NEW_ROOT

    CONFIG = Path(
        config
    ).expanduser().resolve()

    if not CONFIG.is_file():
        raise FileNotFoundError(
            CONFIG
        )

    from pypsds.config import load_config
    from pypsds.project import resolve_project_paths

    cfg, config_path = load_config(
        CONFIG
    )

    paths = resolve_project_paths(
        cfg,
        config_path,
    )

    if new_output is None:

        NEW_OUTPUT = Path(
            paths.output_dir
        ).expanduser().resolve()

    else:

        NEW_OUTPUT = Path(
            new_output
        ).expanduser().resolve()

    NEW_ROOT = (
        NEW_OUTPUT
        / "processing"
    )

    if old_root is None:

        OLD_ROOT = None

    else:

        OLD_ROOT = Path(
            old_root
        ).expanduser().resolve()


def require_old_root():

    if OLD_ROOT is None:
        raise RuntimeError(
            "--old-root is required for "
            "frozen-reference validation."
        )


def banner(s):
    print()
    print("=" * 100)
    print(s)
    print("=" * 100)


def run(
    cmd,
    *,
    check=True,
    cwd=None,
):
    print(
        "+",
        " ".join(
            map(
                str,
                cmd,
            )
        ),
    )

    return subprocess.run(
        list(
            map(
                str,
                cmd,
            )
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
        cwd=(
            str(cwd)
            if cwd is not None
            else None
        ),
    )


def normalize_relpath(rel: Path) -> Path:
    """
    Apply the same naming promotion used during v1 construction.
    """
    s = rel.as_posix()

    s = s.replace("_v09", "")
    s = s.replace("v09", "processing")
    s = s.replace("audit", "quality")
    s = s.replace("prototype", "production")

    return Path(s)


# =============================================================================
# PREFLIGHT
# =============================================================================

def check_cli():
    p = run(
        ["pypsds", "--version"]
    )
    print(p.stdout, end="")

    if "1.0.0" not in p.stdout:
        raise RuntimeError(
            "Installed pypsds is not version 1.0.0"
        )


def check_doctor():
    p = run([
        "pypsds",
        "doctor",
        "--config",
        str(CONFIG),
    ])
    print(p.stdout, end="")

    required = (
        "RSLC stack       :",
        "acquisitions     :",
        "SLC2pt",
        "data2pt",
        "phase_sim_orb_pt",
        "base_calc",
        "base_orbit",
    )

    for token in required:
        if token not in p.stdout:
            raise RuntimeError(
                f"doctor output missing: {token}"
            )


def check_reference():
    require_old_root()

    if not OLD_ROOT.is_dir():
        raise FileNotFoundError(
            f"Frozen reference root does not exist: {OLD_ROOT}"
        )

    required = [
        OLD_ROOT / "final_ps_mask.npy",
        OLD_ROOT / "final_ds_tc0.800_pc0.000_evd.npy",
        OLD_ROOT / "point_phase_stack" / "phase_rad.npy",
        OLD_ROOT / "point_phase_stack" / "rows.npy",
        OLD_ROOT / "network" / "network.itab",
        OLD_ROOT / "final_unwrap_v09" / "strict_unwrap_valid_mask.npy",
        OLD_ROOT / "network_inversion_v09"
        / "acquisition_phase_l2_candidate_rad.npy",
        OLD_ROOT / "referenced_timeseries_v09"
        / "acquisition_phase_referenced_rad.npy",
    ]

    missing = [
        p
        for p in required
        if not p.exists()
    ]

    if missing:
        print("Missing frozen reference products:")
        for p in missing:
            print(" ", p)
        raise RuntimeError(
            "Frozen v0.9 reference set is incomplete."
        )

    print("Frozen reference products: PASS")


def check_stage_interfaces():
    """
    Validate that every pipeline-generated --option is actually
    accepted by the corresponding production script.

    This catches interface errors that --dry-run alone cannot detect.
    """
    from pypsds.config import load_config
    from pypsds.project import resolve_project_paths
    from pypsds.gamma.stack import GammaStack
    from pypsds.runtime import build_runtime_plan
    from pypsds.pipeline import STAGES, _stage_args

    cfg, config_path = load_config(CONFIG)
    paths = resolve_project_paths(cfg, config_path)

    stack = GammaStack.from_rslc_tab(
        paths.rslc_tab,
        rslc_dir=paths.rslc_dir,
        dtype="auto",
        byte_order="big",
        io_workers=1,
    )

    runtime = build_runtime_plan(
        ndate=len(stack.dates),
        memory_fraction=0.85,
    )

    failures = []

    for stage in STAGES:
        script = ROOT / "scripts" / stage.script

        if not script.is_file():
            failures.append(
                f"{stage.name}: missing {script}"
            )
            continue

        p = subprocess.run(
            [sys.executable, str(script), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if p.returncode != 0:
            failures.append(
                f"{stage.name}: --help failed:\n{p.stdout}"
            )
            continue

        generated = _stage_args(
            stage,
            cfg=cfg,
            config_path=config_path,
            paths=paths,
            runtime=runtime,
            force=False,
        )

        opts = [
            x for x in generated
            if str(x).startswith("--")
        ]

        missing_opts = [
            x for x in opts
            if x not in p.stdout
        ]

        if missing_opts:
            failures.append(
                f"{stage.name}: generated options not accepted: "
                f"{missing_opts}"
            )
        else:
            print(
                f"PASS  {stage.name:32s} "
                f"({len(opts)} option(s))"
            )

    if failures:
        print()
        print("STAGE INTERFACE FAILURES:")
        for x in failures:
            print(x)
        raise RuntimeError(
            "Production stage interface preflight failed."
        )

    print("PIPELINE/STAGE ARGUMENT CONTRACT: PASS")


def check_internal_script_references():
    """
    Detect explicit Python script filenames referenced by active
    production scripts and ensure those targets exist.
    """
    import ast

    missing = []

    for src in sorted(
        (ROOT / "scripts").glob("*.py")
    ):
        tree = ast.parse(
            src.read_text(encoding="utf-8"),
            filename=str(src),
        )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue

            value = node.value

            if not isinstance(value, str):
                continue

            if not value.endswith(".py"):
                continue

            # Only step-script references matter here.
            if not value[:2].isdigit():
                continue

            p = ROOT / "scripts" / value

            if not p.exists():
                missing.append(
                    (src.name, value)
                )

    if missing:
        print("Broken internal script references:")
        for src, target in missing:
            print(
                f"  {src} -> {target}"
            )
        raise RuntimeError(
            "Internal production script dependency is broken."
        )

    print("INTERNAL SCRIPT REFERENCES: PASS")


def preflight():
    banner("V1 RELEASE PREFLIGHT")

    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)

    check_cli()
    check_doctor()
    check_reference()
    check_stage_interfaces()
    check_internal_script_references()

    banner("PREFLIGHT RESULT")
    print("V1 FULL-SCENE RUN PREFLIGHT: PASS")


# =============================================================================
# NUMERIC PARITY
# =============================================================================

def compare_npy(
    old: Path,
    new: Path,
    *,
    block_items: int = 4_000_000,
):
    a = np.load(
        old,
        mmap_mode="r",
        allow_pickle=False,
    )

    b = np.load(
        new,
        mmap_mode="r",
        allow_pickle=False,
    )

    if a.shape != b.shape:
        return {
            "status": "FAIL_SHAPE",
            "old_shape": list(a.shape),
            "new_shape": list(b.shape),
        }

    if a.dtype != b.dtype:
        # dtype change is not automatically failure if numerical parity
        # still holds, but report it.
        dtype_equal = False
    else:
        dtype_equal = True

    af = a.reshape(-1)
    bf = b.reshape(-1)

    n = af.size

    exact = True
    finite_mismatch = 0
    max_abs = 0.0
    sum_sq = 0.0
    n_num = 0

    is_float = (
        np.issubdtype(a.dtype, np.floating)
        or np.issubdtype(a.dtype, np.complexfloating)
        or np.issubdtype(b.dtype, np.floating)
        or np.issubdtype(b.dtype, np.complexfloating)
    )

    for i0 in range(0, n, block_items):
        i1 = min(
            n,
            i0 + block_items,
        )

        x = np.asarray(
            af[i0:i1]
        )

        y = np.asarray(
            bf[i0:i1]
        )

        if is_float:
            xf = np.isfinite(x)
            yf = np.isfinite(y)

            finite_mismatch += int(
                np.count_nonzero(
                    xf != yf
                )
            )

            valid = xf & yf

            if np.any(valid):
                d = (
                    x[valid].astype(np.complex128)
                    -
                    y[valid].astype(np.complex128)
                )

                ad = np.abs(d)

                if ad.size:
                    max_abs = max(
                        max_abs,
                        float(np.max(ad)),
                    )

                    sum_sq += float(
                        np.sum(
                            ad * ad,
                            dtype=np.float64,
                        )
                    )

                    n_num += int(
                        ad.size
                    )

                if np.any(ad != 0):
                    exact = False

            # NaN/inf patterns must match.
            if np.any(xf != yf):
                exact = False

            # For non-finite matching positions, compare inf signs etc.
            both_nonfinite = (~xf) & (~yf)

            if np.any(both_nonfinite):
                # equal_nan handles NaN; inf sign must match.
                if not np.array_equal(
                    x[both_nonfinite],
                    y[both_nonfinite],
                    equal_nan=True,
                ):
                    exact = False

        else:
            if not np.array_equal(x, y):
                exact = False

    rms = (
        float(
            np.sqrt(
                sum_sq / n_num
            )
        )
        if n_num
        else 0.0
    )

    # First parity run was created by promoted frozen kernels.
    # Float32 roundoff tolerance remains intentionally tight.
    if is_float:
        numerical_pass = (
            finite_mismatch == 0
            and
            max_abs <= 1.0e-5
        )
    else:
        numerical_pass = exact

    return {
        "status":
            "PASS"
            if numerical_pass
            else "FAIL_NUMERIC",

        "shape":
            list(a.shape),

        "old_dtype":
            str(a.dtype),

        "new_dtype":
            str(b.dtype),

        "dtype_equal":
            dtype_equal,

        "exact":
            bool(exact),

        "finite_mismatch":
            finite_mismatch,

        "max_abs":
            max_abs,

        "rms":
            rms,
    }


def compare_itab(old, new):
    def clean(p):
        rows = []

        for raw in p.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            f = raw.split()

            if len(f) >= 2:
                rows.append(
                    (
                        int(f[0]),
                        int(f[1]),
                    )
                )

        return rows

    a = clean(old)
    b = clean(new)

    return {
        "status":
            "PASS"
            if a == b
            else "FAIL",

        "old_edges":
            len(a),

        "new_edges":
            len(b),

        "exact":
            a == b,
    }


def parity():
    banner("V0.9 -> V1.0 NUMERICAL PARITY")

    require_old_root()

    if not OLD_ROOT.is_dir():
        raise FileNotFoundError(OLD_ROOT)

    if not NEW_ROOT.is_dir():
        raise FileNotFoundError(
            f"V1 production root missing: {NEW_ROOT}"
        )

    report = {
        "old_root": str(OLD_ROOT),
        "new_root": str(NEW_ROOT),
        "npy": {},
        "text": {},
    }

    passed = 0
    failed = 0
    missing = 0

    old_arrays = sorted(
        OLD_ROOT.rglob("*.npy")
    )

    print(
        f"Frozen NPY products discovered: "
        f"{len(old_arrays)}"
    )

    for old in old_arrays:
        rel = old.relative_to(
            OLD_ROOT
        )

        mapped = normalize_relpath(
            rel
        )

        new = NEW_ROOT / mapped

        key = rel.as_posix()

        # Historical backup created before promotion of the
        # finalized directional temporal network.
        #
        # This file is not a scientific release product:
        # its contents depend on the prior state of the output
        # directory. The formal network contract is network.itab
        # plus the finalized directional topology products.
        if key == "network/degree_pre_directional.npy":
            report["npy"][key] = {
                "status": "IGNORED_NON_RELEASE_HISTORY",
                "mapped": mapped.as_posix(),
            }
            continue

        if not new.exists():
            report["npy"][key] = {
                "status": "NEW_NOT_PRESENT",
                "mapped": mapped.as_posix(),
            }
            missing += 1
            continue

        result = compare_npy(
            old,
            new,
        )

        result["mapped"] = (
            mapped.as_posix()
        )

        report["npy"][key] = result

        if result["status"] == "PASS":
            passed += 1
            print(
                f"PASS  {key}"
                f"  max={result.get('max_abs', 0):.3e}"
            )
        else:
            failed += 1
            print(
                f"FAIL  {key}: {result}"
            )

    # Critical network topology.
    old_itab = (
        OLD_ROOT
        / "network"
        / "network.itab"
    )

    new_itab = (
        NEW_ROOT
        / "network"
        / "network.itab"
    )

    if (
        old_itab.exists()
        and
        new_itab.exists()
    ):
        r = compare_itab(
            old_itab,
            new_itab,
        )

        report["text"][
            "network/network.itab"
        ] = r

        if r["status"] == "PASS":
            passed += 1
            print(
                "PASS  network/network.itab"
            )
        else:
            failed += 1
            print(
                "FAIL  network/network.itab"
            )

    # Critical release products must all exist.
    critical_new = [
        NEW_ROOT
        / "final_ps_mask.npy",

        NEW_ROOT
        / "final_ds_tc0.800_pc0.000_evd.npy",

        NEW_ROOT
        / "point_phase_stack"
        / "phase_rad.npy",

        NEW_ROOT
        / "network"
        / "network.itab",

        NEW_ROOT
        / "final_unwrap"
        / "strict_unwrap_valid_mask.npy",

        NEW_ROOT
        / "network_inversion"
        / "acquisition_phase_l2_candidate_rad.npy",

        NEW_ROOT
        / "referenced_timeseries"
        / "acquisition_phase_referenced_rad.npy",
    ]

    critical_missing = [
        str(p)
        for p in critical_new
        if not p.exists()
    ]

    report["critical_missing"] = (
        critical_missing
    )

    report["summary"] = {
        "compared_pass":
            passed,
        "compared_fail":
            failed,
        "reference_arrays_without_v1_mapping":
            missing,
    }

    qdir = (
        NEW_OUTPUT
        / "quality"
    )

    qdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        qdir
        / "v1_parity_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("PARITY SUMMARY")
    print("=" * 100)

    print(
        f"PASS comparisons : {passed}"
    )
    print(
        f"FAIL comparisons : {failed}"
    )
    print(
        f"not mapped       : {missing}"
    )
    print(
        f"report           : {report_path}"
    )

    if critical_missing:
        print()
        print("CRITICAL PRODUCTS MISSING:")
        for x in critical_missing:
            print(" ", x)

    if failed:
        raise SystemExit(
            "V1 NUMERICAL PARITY: FAIL"
        )

    if critical_missing:
        raise SystemExit(
            "V1 RELEASE PRODUCT CONTRACT: FAIL"
        )

    print()
    print(
        "V1 NUMERICAL PARITY: PASS"
    )


# =============================================================================
# RELEASE HARDENING CHECKS
# =============================================================================

def check_compile():

    banner(
        "PYTHON COMPILE CHECK"
    )

    p = run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "pypsds",
            "scripts",
            "tools",
        ],
        check=False,
        cwd=ROOT,
    )

    if p.stdout:
        print(
            p.stdout,
            end="",
        )

    if p.returncode != 0:
        raise RuntimeError(
            "Python compile check failed."
        )

    print(
        "PYTHON COMPILE: PASS"
    )


def check_unit_tests():

    banner(
        "UNIT TESTS"
    )

    p = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests",
        ],
        check=False,
        cwd=ROOT,
    )

    print(
        p.stdout,
        end="",
    )

    if p.returncode != 0:
        raise RuntimeError(
            "Unit tests failed."
        )

    print(
        "UNIT TESTS: PASS"
    )


def check_stage_output_contract():

    # Dynamic stage-count labels must follow the current pipeline.
    from pypsds.pipeline import STAGES

    banner(
        f"{len(STAGES)}-STAGE OUTPUT CONTRACT"
    )

    tool = (
        ROOT
        / "tools"
        / "freeze_stage_output_contracts.py"
    )

    snapshot = (
        ROOT
        / "docs"
        / "release"
        / "stage_output_contract_snapshot.json"
    )

    if not tool.is_file():
        raise FileNotFoundError(
            tool
        )

    if not snapshot.is_file():
        raise FileNotFoundError(
            "Frozen stage output contract "
            f"snapshot missing: {snapshot}"
        )

    p = run(
        [
            sys.executable,
            tool,
            "--config",
            CONFIG,
            "--audit-only",
        ],
        check=False,
        cwd=ROOT,
    )

    print(
        p.stdout,
        end="",
    )

    if p.returncode != 0:
        raise RuntimeError(
            f"{len(STAGES)}-stage output contract audit failed."
        )

    token = (
        f"{len(STAGES)}-STAGE OUTPUT CONTRACT AUDIT: PASS"
    )

    if token not in p.stdout:
        raise RuntimeError(
            "Contract tool returned success but "
            "PASS token was not found."
        )

    print(
        f"{len(STAGES)}-STAGE CONTRACT: PASS"
    )


def local_tests():

    check_compile()
    check_unit_tests()
    check_stage_output_contract()

    banner(
        "LOCAL RELEASE TEST RESULT"
    )

    print(
        "LOCAL RELEASE TESTS: PASS"
    )


def all_checks():

    banner(
        "pyPSDS-GAMMA V1 RELEASE GATE"
    )

    print(
        "repository    :",
        ROOT,
    )

    print(
        "config        :",
        CONFIG,
    )

    print(
        "old reference :",
        OLD_ROOT,
    )

    print(
        "new output    :",
        NEW_OUTPUT,
    )

    # Fast engineering checks.
    check_compile()
    check_unit_tests()

    # Installation / GAMMA / stage-interface checks.
    preflight()

    # Existing-result completeness.
    check_stage_output_contract()

    # Frozen scientific baseline.
    parity()

    banner(
        "FINAL RELEASE GATE RESULT"
    )

    print(
        "CONFIG / DOCTOR          : PASS"
    )

    print(
        "PYTHON COMPILE           : PASS"
    )

    print(
        "UNIT TESTS               : PASS"
    )

    print(
        "STAGE INTERFACES         : PASS"
    )

    print(
        "31-STAGE OUTPUT CONTRACT : PASS"
    )

    print(
        "NUMERICAL PARITY         : PASS"
    )

    print()
    print(
        "V1 RELEASE GATE: PASS"
    )


def main():
    ap = argparse.ArgumentParser()

    sub = ap.add_subparsers(
        dest="command",
        required=True,
    )

    def add_common(
        parser,
    ):

        parser.add_argument(
            "--config",
            default=str(
                ROOT
                / "config"
                / "pypsds.yaml"
            ),
            help=(
                "Production configuration. "
                "Default: repository config/pypsds.yaml"
            ),
        )

        parser.add_argument(
            "--old-root",
            default=None,
            help=(
                "Frozen validated baseline root. "
                "Required for preflight/parity/all."
            ),
        )

        parser.add_argument(
            "--new-output",
            default=None,
            help=(
                "Override production output root. "
                "Default: output_dir from config."
            ),
        )

    for command in (
        "preflight",
        "parity",
        "contract",
        "tests",
        "all",
    ):

        q = sub.add_parser(
            command
        )

        add_common(
            q
        )

    args = ap.parse_args()

    configure_release_paths(
        config=args.config,
        old_root=args.old_root,
        new_output=args.new_output,
    )

    if (
        args.command
        in (
            "preflight",
            "parity",
            "all",
        )
        and
        OLD_ROOT is None
    ):

        ap.error(
            "--old-root is required for "
            f"'{args.command}'"
        )

    if args.command == "preflight":

        preflight()

    elif args.command == "parity":

        parity()

    elif args.command == "contract":

        check_stage_output_contract()

    elif args.command == "tests":

        local_tests()

    elif args.command == "all":

        all_checks()

    else:

        raise RuntimeError(
            args.command
        )


if __name__ == "__main__":
    main()
