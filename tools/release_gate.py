#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


DIST_NAME = "pypsds-gamma"


def run(
    cmd,
    *,
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

    p = subprocess.run(
        [
            str(x)
            for x in cmd
        ],
        cwd=(
            None
            if cwd is None
            else str(cwd)
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(
        p.stdout,
        end="",
    )

    if p.returncode != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{p.returncode}"
        )

    return p


def release_identity():
    # Read release identity only from authoritative source modules.
    # No version number or production-stage count is duplicated here.

    import pypsds

    from pypsds._version import (
        __version__,
    )

    from pypsds.pipeline import (
        STAGES,
    )

    if (
        pypsds.__version__
        !=
        __version__
    ):
        raise RuntimeError(
            "package/version-source mismatch: "
            f"{pypsds.__version__!r} != "
            f"{__version__!r}"
        )

    stage_names = tuple(
        stage.name
        for stage in STAGES
    )

    stage_scripts = tuple(
        stage.script
        for stage in STAGES
    )

    if not stage_names:
        raise RuntimeError(
            "production pipeline contains no stages"
        )

    if (
        len(set(stage_names))
        !=
        len(stage_names)
    ):
        raise RuntimeError(
            "duplicate production stage names"
        )

    if (
        len(set(stage_scripts))
        !=
        len(stage_scripts)
    ):
        raise RuntimeError(
            "duplicate production stage scripts"
        )

    return {
        "version":
            __version__,

        "stage_names":
            stage_names,

        "stage_scripts":
            stage_scripts,
    }


def _assert_stage_sequence(
    *,
    label,
    actual,
    expected,
):
    actual = tuple(actual)
    expected = tuple(expected)

    if actual == expected:
        return

    missing = [
        name
        for name in expected
        if name not in actual
    ]

    extra = [
        name
        for name in actual
        if name not in expected
    ]

    raise RuntimeError(
        f"{label} stage sequence mismatch\n"
        f"expected ({len(expected)}): "
        f"{list(expected)}\n"
        f"actual   ({len(actual)}): "
        f"{list(actual)}\n"
        f"missing: {missing}\n"
        f"extra  : {extra}"
    )


def _wheel_metadata_version(
    wheel: Path,
) -> str:

    with zipfile.ZipFile(
        wheel
    ) as zf:

        metadata = [
            name
            for name in zf.namelist()
            if (
                name.endswith(
                    ".dist-info/METADATA"
                )
            )
        ]

        if len(metadata) != 1:
            raise RuntimeError(
                "expected exactly one wheel "
                "METADATA file, found "
                f"{len(metadata)}"
            )

        text = (
            zf.read(
                metadata[0]
            )
            .decode(
                "utf-8"
            )
        )

    versions = [
        line.split(
            ":",
            1,
        )[1].strip()
        for line in text.splitlines()
        if line.startswith(
            "Version:"
        )
    ]

    if len(versions) != 1:
        raise RuntimeError(
            "wheel METADATA must contain "
            "exactly one Version field"
        )

    return versions[0]


def check_identity():

    identity = (
        release_identity()
    )

    print(
        "release version :",
        identity["version"],
    )

    print(
        "stage count     :",
        len(
            identity[
                "stage_names"
            ]
        ),
    )

    print(
        "first stage     :",
        identity[
            "stage_names"
        ][0],
    )

    print(
        "last stage      :",
        identity[
            "stage_names"
        ][-1],
    )

    print(
        "RELEASE IDENTITY GATE: PASS"
    )


def check_tests():

    check_identity()

    run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "pypsds",
            "tests",
            "tools",
        ],
        cwd=ROOT,
    )

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests",
        ],
        cwd=ROOT,
    )

    print(
        "SOURCE TEST GATE: PASS"
    )


def check_contract(
    config,
):
    identity = (
        release_identity()
    )

    expected_names = (
        identity[
            "stage_names"
        ]
    )

    config = (
        Path(config)
        .expanduser()
        .resolve()
    )

    if not config.is_file():
        raise FileNotFoundError(
            config
        )

    with tempfile.TemporaryDirectory(
        prefix="pypsds-contract-"
    ) as td:

        td = Path(td)

        inventory = (
            td
            / "inventory.json"
        )

        text = (
            td
            / "inventory.txt"
        )

        snapshot = (
            td
            / "snapshot.json"
        )

        run(
            [
                sys.executable,
                str(
                    ROOT
                    / "tools"
                    / "build_stage_contract_inventory.py"
                ),
                "--config",
                str(config),
                "--json",
                str(inventory),
                "--text",
                str(text),
            ],
            cwd=ROOT,
        )

        from pypsds.pipeline import (
            STAGE_CONTRACTS,
        )

        inventory_data = (
            json.loads(
                inventory.read_text(
                    encoding="utf-8"
                )
            )
        )

        inventory_stages = (
            inventory_data.get(
                "stages",
                {}
            )
        )

        _assert_stage_sequence(
            label="inventory",
            actual=(
                inventory_stages.keys()
            ),
            expected=expected_names,
        )

        enriched = []

        for (
            stage_name,
            contract,
        ) in (
            STAGE_CONTRACTS.items()
        ):

            if (
                not contract.validated
                or
                not contract.required_outputs
            ):
                continue

            stage_info = (
                inventory_stages.get(
                    stage_name
                )
            )

            if stage_info is None:
                raise RuntimeError(
                    "validated StageContract "
                    "missing from inventory: "
                    f"{stage_name}"
                )

            outputs = set(
                stage_info.get(
                    "exact_output_outputs",
                    [],
                )
            )

            outputs.update(
                str(x)
                for x in (
                    contract
                    .required_outputs
                )
            )

            stage_info[
                "exact_output_outputs"
            ] = sorted(
                outputs
            )

            enriched.append(
                (
                    stage_name,
                    len(
                        contract
                        .required_outputs
                    ),
                )
            )

        inventory.write_text(
            json.dumps(
                inventory_data,
                indent=2,
                ensure_ascii=False,
            )
            +
            "\n",
            encoding="utf-8",
        )

        print(
            "validated StageContract "
            "inventory enrichment:",
            ", ".join(
                f"{name}={count}"
                for (
                    name,
                    count,
                ) in enriched
            ),
        )

        run(
            [
                sys.executable,
                str(
                    ROOT
                    / "tools"
                    / "freeze_stage_output_contracts.py"
                ),
                "--config",
                str(config),
                "--inventory",
                str(inventory),
                "--snapshot",
                str(snapshot),
            ],
            cwd=ROOT,
        )

        run(
            [
                sys.executable,
                str(
                    ROOT
                    / "tools"
                    / "freeze_stage_output_contracts.py"
                ),
                "--config",
                str(config),
                "--inventory",
                str(inventory),
                "--snapshot",
                str(snapshot),
                "--audit-only",
            ],
            cwd=ROOT,
        )

        data = json.loads(
            snapshot.read_text(
                encoding="utf-8"
            )
        )

        if (
            data.get(
                "stage_count"
            )
            !=
            len(expected_names)
        ):
            raise RuntimeError(
                "snapshot stage-count mismatch: "
                f"{data.get('stage_count')} != "
                f"{len(expected_names)}"
            )

        _assert_stage_sequence(
            label="snapshot",
            actual=(
                data.get(
                    "stages",
                    {}
                ).keys()
            ),
            expected=expected_names,
        )

    print(
        "DYNAMIC OUTPUT CONTRACT GATE: PASS"
    )


def _venv_python(
    vdir,
):

    return (
        vdir
        /
        (
            "Scripts/python.exe"
            if sys.platform == "win32"
            else "bin/python"
        )
    )


def check_wheel():

    identity = (
        release_identity()
    )

    expected_version = (
        identity[
            "version"
        ]
    )

    expected_names = (
        identity[
            "stage_names"
        ]
    )

    expected_scripts = (
        identity[
            "stage_scripts"
        ]
    )

    with tempfile.TemporaryDirectory(
        prefix="pypsds-wheel-"
    ) as td:

        td = Path(td)

        wheel_dir = (
            td
            / "wheel"
        )

        wheel_dir.mkdir()

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                ".",
                "-w",
                str(wheel_dir),
            ],
            cwd=ROOT,
        )

        wheels = list(
            wheel_dir.glob(
                "*.whl"
            )
        )

        if len(wheels) != 1:
            raise RuntimeError(
                "expected one wheel, "
                f"found {len(wheels)}"
            )

        wheel = wheels[0]

        wheel_version = (
            _wheel_metadata_version(
                wheel
            )
        )

        if (
            wheel_version
            !=
            expected_version
        ):
            raise RuntimeError(
                "wheel/source version mismatch: "
                f"{wheel_version!r} != "
                f"{expected_version!r}"
            )

        print(
            "wheel metadata version:",
            wheel_version,
        )

        with zipfile.ZipFile(
            wheel
        ) as zf:
            names = set(
                zf.namelist()
            )

        required = {
            "pypsds/_version.py",
            "pypsds/resources/default_config.yaml",
            "pypsds/resources/ds_production_policy.json",
            "pypsds/products/point_metrics.py",
            "pypsds/runtime_backend/gacos_runtime.py",
            "pypsds/runtime_backend/scn_runtime.py",
            "pypsds/runtime_backend/final_los_runtime.py",
            "pypsds/runtime_backend/point_metrics_runtime.py",
        }

        required.update(
            "pypsds/stages/"
            + script
            for script in (
                expected_scripts
            )
        )

        missing = sorted(
            required
            -
            names
        )

        if missing:
            raise RuntimeError(
                "wheel missing production "
                "resources: "
                +
                ", ".join(
                    missing
                )
            )

        vdir = (
            td
            / "venv"
        )

        venv.EnvBuilder(
            with_pip=True,
            system_site_packages=True,
        ).create(
            vdir
        )

        py = (
            _venv_python(
                vdir
            )
        )

        run(
            [
                py,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ]
        )

        smoke = (
            td
            / "smoke"
        )

        smoke.mkdir()

        code = (
            "from pathlib import Path\n"
            "import importlib\n"
            "import importlib.metadata as md\n"
            "import sys\n"
            "import pypsds\n"
            "from pypsds.pipeline import STAGES\n"
            f"expected_version = {expected_version!r}\n"
            f"expected_names = {list(expected_names)!r}\n"
            "pkg = Path(pypsds.__file__).resolve()\n"
            "prefix = Path(sys.prefix).resolve()\n"
            "assert pkg.is_relative_to(prefix), (pkg, prefix)\n"
            "assert pypsds.__version__ == expected_version\n"
            f"assert md.version({DIST_NAME!r}) == expected_version\n"
            "actual_names = [stage.name for stage in STAGES]\n"
            "assert actual_names == expected_names, "
            "(actual_names, expected_names)\n"
            "for stage in STAGES:\n"
            "    importlib.import_module("
            "'pypsds.stages.' + Path(stage.script).stem)\n"
            "print(f'installed pypsds source: {pkg}')\n"
            "print(f'installed version: {pypsds.__version__}')\n"
            "print(f'installed stages: {len(STAGES)}')\n"
            "print('installed stage sequence: PASS')\n"
        )

        run(
            [
                py,
                "-I",
                "-c",
                code,
            ],
            cwd=smoke,
        )

        project = (
            smoke
            / "project"
        )

        run(
            [
                py,
                "-I",
                "-m",
                "pypsds.cli",
                "init",
                str(project),
            ],
            cwd=smoke,
        )

        generated = (
            project
            / "pypsds.yaml"
        )

        text = generated.read_text(
            encoding="utf-8"
        )

        forbidden = (
            "/home/",
            "/mnt/",
            "/media/",
            str(ROOT),
        )

        bad = [
            token
            for token in forbidden
            if token in text
        ]

        if bad:
            raise RuntimeError(
                "generated project config "
                "contains machine-specific "
                f"paths: {bad}"
            )

    print(
        "WHEEL / INSTALLED-PACKAGE GATE: PASS"
    )


def check_all(
    config,
):

    check_tests()

    check_wheel()

    check_contract(
        config
    )

    print(
        "FULL RELEASE GATE: PASS"
    )


def main():

    ap = argparse.ArgumentParser(
        description=(
            "pyPSDS-GAMMA release validation gate"
        )
    )

    sub = ap.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "identity"
    )

    sub.add_parser(
        "tests"
    )

    sub.add_parser(
        "wheel"
    )

    q = sub.add_parser(
        "contract"
    )

    q.add_argument(
        "--config",
        required=True,
    )

    q = sub.add_parser(
        "all"
    )

    q.add_argument(
        "--config",
        required=True,
    )

    args = ap.parse_args()

    if args.command == "identity":
        check_identity()

    elif args.command == "tests":
        check_tests()

    elif args.command == "wheel":
        check_wheel()

    elif args.command == "contract":
        check_contract(
            args.config
        )

    elif args.command == "all":
        check_all(
            args.config
        )


if __name__ == "__main__":
    main()
