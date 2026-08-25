#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pyPSDS-GAMMA-v1.0"
cd "${ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="docs/release_v1_1/backups/projectpaths_compat_${STAMP}"

mkdir -p "${BACKUP}"

cp -a \
    pypsds/project.py \
    "${BACKUP}/project.py"

echo "================================================================================"
echo "v1.1 ProjectPaths backward-compatibility patch"
echo "================================================================================"
echo "backup : ${BACKUP}"
echo


python - <<'PY'
from pathlib import Path


path = Path(
    "/home/ubuntu/software/pyPSDS-GAMMA-v1.0/pypsds/project.py"
)

text = path.read_text(
    encoding="utf-8"
)


old = '''    output_dir: Path

    dem_dir: Path | None
    gacos_dir: Path | None

    scratch_dir: Path
    products_dir: Path
'''


new = '''    output_dir: Path

    # v1.1 additions.
    #
    # Defaults preserve the v1.0 public construction API:
    #
    # ProjectPaths(
    #     work_dir=...,
    #     data_dir=...,
    #     rslc_dir=...,
    #     rslc_tab=...,
    #     output_dir=...,
    # )
    #
    # resolve_project_paths() always populates scratch_dir and
    # products_dir with concrete Paths for normal production use.
    dem_dir: Path | None = None
    gacos_dir: Path | None = None
    scratch_dir: Path | None = None
    products_dir: Path | None = None
'''


count = text.count(
    old
)


if count != 1:

    raise RuntimeError(
        "Expected exactly one ProjectPaths field block, "
        f"found {count}"
    )


path.write_text(
    text.replace(
        old,
        new,
    ),
    encoding="utf-8",
)


print(
    "PATCHED:",
    path
)
PY


echo
echo "--------------------------------------------------------------------------------"
echo "Add explicit backward-compatibility regression"
echo "--------------------------------------------------------------------------------"


cat >> tests/test_v11_project_paths.py <<'PY'


def test_v11_projectpaths_keeps_v1_constructor_compatibility(
    tmp_path,
):
    """
    Adding v1.1 auxiliary/output paths must not break code that
    directly constructs the v1.0 ProjectPaths five-field object.
    """

    from pypsds.project import ProjectPaths


    work = (
        tmp_path
        /
        "project"
    )

    data = (
        work
        /
        "data"
    )

    rslc = (
        data
        /
        "RSLC"
    )

    tab = (
        data
        /
        "RSLC_tab"
    )

    output = (
        work
        /
        "output"
    )


    paths = ProjectPaths(
        work_dir=work,
        data_dir=data,
        rslc_dir=rslc,
        rslc_tab=tab,
        output_dir=output,
    )


    assert paths.work_dir == work
    assert paths.data_dir == data
    assert paths.rslc_dir == rslc
    assert paths.rslc_tab == tab
    assert paths.output_dir == output

    assert paths.dem_dir is None
    assert paths.gacos_dir is None
    assert paths.scratch_dir is None
    assert paths.products_dir is None
PY


echo
echo "--------------------------------------------------------------------------------"
echo "Static compile"
echo "--------------------------------------------------------------------------------"

python -m py_compile \
    pypsds/project.py \
    tests/test_v11_project_paths.py

echo "PY_COMPILE PASS"


echo
echo "--------------------------------------------------------------------------------"
echo "Previously failing tests"
echo "--------------------------------------------------------------------------------"

pytest -q \
    tests/test_pipeline_sequential_dispatch.py \
    tests/test_reference_point_ids_dispatch.py

echo "PREVIOUS FAILURES: PASS"


echo
echo "--------------------------------------------------------------------------------"
echo "v1.1 config/path tests"
echo "--------------------------------------------------------------------------------"

pytest -q \
    tests/test_production_config.py \
    tests/test_v11_project_paths.py \
    tests/test_v11_public_config.py

echo "V1.1 FOCUSED TESTS: PASS"


echo
echo "--------------------------------------------------------------------------------"
echo "FULL REGRESSION SUITE"
echo "--------------------------------------------------------------------------------"

pytest -q


echo
echo "================================================================================"
echo "V1.1 CONFIG/PATH CONTRACT FINAL RESULT: PASS"
echo "================================================================================"
echo "Compatibility:"
echo "  v1.0 ProjectPaths constructor     preserved"
echo "  v1.1 resolve_project_paths        extended"
echo "  schema_version                    still 1"
echo "  numerical algorithms              unchanged"
echo "  pipeline stage graph              unchanged"
echo "================================================================================"
