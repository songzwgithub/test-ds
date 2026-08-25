#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/software/pyPSDS-GAMMA-v1.0"

cd "${ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"

BACKUP="${ROOT}/docs/release_v1_1/backups/config_path_contract_${STAMP}"

mkdir -p "${BACKUP}"

echo "================================================================================"
echo "pyPSDS-GAMMA v1.1 CONFIG / PATH CONTRACT"
echo "================================================================================"
echo "repository : ${ROOT}"
echo "backup     : ${BACKUP}"
echo


# ==============================================================================
# 0. Preconditions
# ==============================================================================

for f in \
    pypsds/project.py \
    pypsds/config.py \
    pypsds/resources/default_config.yaml \
    config/pypsds.yaml \
    config/pypsds_template.yaml
do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file missing: ${f}" >&2
        exit 1
    fi
done


# ==============================================================================
# 1. Backup
# ==============================================================================

mkdir -p \
    "${BACKUP}/pypsds/resources" \
    "${BACKUP}/config" \
    "${BACKUP}/tests"

cp -a \
    pypsds/project.py \
    "${BACKUP}/pypsds_project.py"

cp -a \
    pypsds/config.py \
    "${BACKUP}/pypsds_config.py"

cp -a \
    pypsds/resources/default_config.yaml \
    "${BACKUP}/pypsds/resources/default_config.yaml"

cp -a \
    config/pypsds.yaml \
    "${BACKUP}/config/pypsds.yaml"

cp -a \
    config/pypsds_template.yaml \
    "${BACKUP}/config/pypsds_template.yaml"

if [[ -f tests/test_production_config.py ]]; then
    cp -a \
        tests/test_production_config.py \
        "${BACKUP}/tests/test_production_config.py"
fi

echo "[1/7] Backup complete"


# ==============================================================================
# 2. Replace project.py with backward-compatible v1.1 path resolver
# ==============================================================================

cat > pypsds/project.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import cfg_get


@dataclass(
    frozen=True,
    slots=True,
)
class ProjectPaths:
    """
    Resolved filesystem contract for one pyPSDS-GAMMA project.

    Required processing inputs
    --------------------------
    work_dir
    data_dir
    rslc_dir
    rslc_tab

    Project-level auxiliary locations
    ---------------------------------
    dem_dir
    gacos_dir

    Managed output locations
    ------------------------
    output_dir
    scratch_dir
    products_dir

    Notes
    -----
    DEM and GACOS directories are optional at path-resolution
    time because the corresponding corrections may be disabled.
    Individual processing modules are responsible for requiring
    them only when their feature is enabled.
    """

    work_dir: Path
    data_dir: Path

    rslc_dir: Path
    rslc_tab: Path

    output_dir: Path

    dem_dir: Path | None
    gacos_dir: Path | None

    scratch_dir: Path
    products_dir: Path


def _resolve(
    value,
    *,
    base: Path,
) -> Path | None:

    if value in (
        None,
        "",
    ):
        return None

    p = Path(
        value
    ).expanduser()

    if not p.is_absolute():
        p = (
            base
            /
            p
        )

    return p.resolve()


def _discover_directory(
    *,
    configured,
    base: Path,
    candidates: tuple[str, ...],
) -> Path | None:
    """
    Resolve an optional project directory.

    Explicit configuration has priority.

    If the setting is null/empty, existing conventional directory
    names are discovered below ``base``.

    An explicitly configured path is returned even if it does not
    yet exist. This keeps path resolution separate from feature
    validation.
    """

    resolved = _resolve(
        configured,
        base=base,
    )

    if resolved is not None:
        return resolved

    for name in candidates:

        candidate = (
            base
            /
            name
        )

        if candidate.is_dir():
            return (
                candidate
                .resolve()
            )

    return None


def resolve_project_paths(
    cfg: dict[str, Any],
    config_path: Path,
) -> ProjectPaths:

    config_path = (
        Path(
            config_path
        )
        .expanduser()
        .resolve()
    )

    base = (
        config_path
        .parent
    )


    # ------------------------------------------------------------------
    # Project working directory
    # ------------------------------------------------------------------

    work_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.work_dir",
                None,
            ),
            base=base,
        )
        or
        base
    )


    # ------------------------------------------------------------------
    # Data root
    # ------------------------------------------------------------------

    data_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.data_dir",
                None,
            ),
            base=work_dir,
        )
        or
        work_dir
    )


    # ------------------------------------------------------------------
    # RSLC directory: required
    # ------------------------------------------------------------------

    rslc_dir = _resolve(
        cfg_get(
            cfg,
            "paths.rslc_dir",
            None,
        ),
        base=data_dir,
    )

    if rslc_dir is None:

        for name in (
            "RSLC_cropped",
            "RSLC",
        ):

            candidate = (
                data_dir
                /
                name
            )

            if candidate.is_dir():

                rslc_dir = (
                    candidate
                    .resolve()
                )

                break

    if (
        rslc_dir is None
        or
        not rslc_dir.is_dir()
    ):

        raise FileNotFoundError(
            "Unable to locate RSLC directory "
            f"below {data_dir}"
        )


    # ------------------------------------------------------------------
    # RSLC_tab: required
    # ------------------------------------------------------------------

    rslc_tab = _resolve(
        cfg_get(
            cfg,
            "paths.rslc_tab",
            None,
        ),
        base=data_dir,
    )

    if rslc_tab is None:

        direct = (
            data_dir
            /
            "RSLC_tab"
        )

        if direct.is_file():

            rslc_tab = (
                direct
                .resolve()
            )

        else:

            hits = sorted(
                data_dir.glob(
                    "*RSLC*tab*"
                )
            )

            if hits:

                rslc_tab = (
                    hits[0]
                    .resolve()
                )

    if (
        rslc_tab is None
        or
        not rslc_tab.is_file()
    ):

        raise FileNotFoundError(
            "Unable to locate RSLC_tab "
            f"below {data_dir}"
        )


    # ------------------------------------------------------------------
    # Primary output directory
    # ------------------------------------------------------------------

    output_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.output_dir",
                None,
            ),
            base=work_dir,
        )
        or
        (
            work_dir
            /
            "output"
        ).resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------------------
    # DEM / geometry directory
    #
    # Optional until a module that requires DEM geometry is enabled.
    # ------------------------------------------------------------------

    dem_dir = _discover_directory(
        configured=cfg_get(
            cfg,
            "paths.dem_dir",
            None,
        ),
        base=data_dir,
        candidates=(
            "DEM_prep",
            "DEM",
            "dem",
        ),
    )


    # ------------------------------------------------------------------
    # GACOS directory
    #
    # Optional until corrections.atmosphere.mode == gacos.
    # ------------------------------------------------------------------

    gacos_dir = _discover_directory(
        configured=cfg_get(
            cfg,
            "paths.gacos_dir",
            None,
        ),
        base=data_dir,
        candidates=(
            "GACOS",
            "gacos",
        ),
    )


    # ------------------------------------------------------------------
    # Managed scratch directory
    # ------------------------------------------------------------------

    scratch_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.scratch_dir",
                None,
            ),
            base=work_dir,
        )
        or
        (
            output_dir
            /
            ".scratch"
        ).resolve()
    )

    scratch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------------------
    # Managed final-product directory
    # ------------------------------------------------------------------

    products_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.products_dir",
                None,
            ),
            base=work_dir,
        )
        or
        (
            output_dir
            /
            "products"
        ).resolve()
    )

    products_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    return ProjectPaths(
        work_dir=work_dir,
        data_dir=data_dir,

        rslc_dir=rslc_dir,
        rslc_tab=rslc_tab,

        output_dir=output_dir,

        dem_dir=dem_dir,
        gacos_dir=gacos_dir,

        scratch_dir=scratch_dir,
        products_dir=products_dir,
    )


__all__ = [
    "ProjectPaths",
    "resolve_project_paths",
]
PY

echo "[2/7] project.py updated"


# ==============================================================================
# 3. Extend all public/config templates WITHOUT changing frozen processing values
# ==============================================================================

python - <<'PY'
from pathlib import Path
import yaml


ROOT = Path(
    "/home/ubuntu/software/pyPSDS-GAMMA-v1.0"
)


files = [
    ROOT
    / "config"
    / "pypsds.yaml",

    ROOT
    / "config"
    / "pypsds_template.yaml",

    ROOT
    / "pypsds"
    / "resources"
    / "default_config.yaml",
]


for path in files:

    cfg = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    ) or {}

    if not isinstance(
        cfg,
        dict,
    ):
        raise RuntimeError(
            f"Invalid configuration mapping: {path}"
        )


    # --------------------------------------------------------------
    # Do NOT change schema_version yet.
    #
    # v1.1 remains backward compatible with schema 1.
    # --------------------------------------------------------------

    if cfg.get(
        "schema_version"
    ) != 1:
        raise RuntimeError(
            f"Unexpected schema_version in {path}"
        )


    # --------------------------------------------------------------
    # Portable project paths
    # --------------------------------------------------------------

    paths = cfg.setdefault(
        "paths",
        {},
    )

    paths.setdefault(
        "work_dir",
        ".",
    )

    paths.setdefault(
        "data_dir",
        ".",
    )

    paths.setdefault(
        "rslc_dir",
        "RSLC",
    )

    paths.setdefault(
        "rslc_tab",
        "RSLC_tab",
    )

    paths.setdefault(
        "dem_dir",
        "DEM_prep",
    )

    paths.setdefault(
        "gacos_dir",
        "GACOS",
    )

    paths.setdefault(
        "output_dir",
        "output",
    )

    paths.setdefault(
        "scratch_dir",
        "output/.scratch",
    )

    paths.setdefault(
        "products_dir",
        "output/products",
    )


    # --------------------------------------------------------------
    # Corrections
    #
    # IMPORTANT:
    # Defaults remain disabled, preserving v1.0 behavior.
    # These keys formalize the v1.1 public contract only.
    # --------------------------------------------------------------

    corrections = cfg.setdefault(
        "corrections",
        {},
    )


    scla = corrections.setdefault(
        "scla",
        {},
    )

    scla.setdefault(
        "mode",
        "disabled",
    )

    scla.setdefault(
        "backend",
        "stamps",
    )


    atmosphere = corrections.setdefault(
        "atmosphere",
        {},
    )

    atmosphere.setdefault(
        "mode",
        "disabled",
    )

    atmosphere.setdefault(
        "backend",
        "gacos",
    )

    atmosphere.setdefault(
        "strict_dates",
        True,
    )


    scn = corrections.setdefault(
        "scn",
        {},
    )

    scn.setdefault(
        "mode",
        "disabled",
    )

    scn.setdefault(
        "backend",
        "stamps",
    )

    scn.setdefault(
        "temporal_window_days",
        365.0,
    )

    scn.setdefault(
        "wavelength_m",
        100.0,
    )

    scn.setdefault(
        "radius_factor",
        4.0,
    )


    # --------------------------------------------------------------
    # Point-first scientific product contract.
    #
    # Raster quicklooks remain explicitly non-scientific outputs.
    # --------------------------------------------------------------

    products = cfg.setdefault(
        "products",
        {},
    )


    point = products.setdefault(
        "point",
        {},
    )

    point.setdefault(
        "enabled",
        True,
    )

    point.setdefault(
        "formats",
        [
            "parquet",
            "geopackage",
            "csv",
        ],
    )

    point.setdefault(
        "crs",
        "EPSG:4326",
    )


    quicklook = products.setdefault(
        "quicklook",
        {},
    )

    quicklook.setdefault(
        "enabled",
        False,
    )

    quicklook.setdefault(
        "scientific_product",
        False,
    )


    # --------------------------------------------------------------
    # Preserve all existing development/frozen keys for now.
    #
    # Naming cleanup will be a separate regression-controlled step.
    # --------------------------------------------------------------


    text = yaml.safe_dump(
        cfg,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "UPDATED:",
        path,
    )
PY

echo "[3/7] configuration templates extended"


# ==============================================================================
# 4. Add v1.1 path/config regression tests
# ==============================================================================

cat > tests/test_v11_project_paths.py <<'PY'
from pathlib import Path

import yaml

from pypsds.project import (
    resolve_project_paths,
)


def _write_config(
    path: Path,
    payload: dict,
) -> Path:

    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return path


def test_v11_project_paths_are_relative_to_project(
    tmp_path,
):

    project = (
        tmp_path
        /
        "area_a"
    )

    project.mkdir()


    # Required stack inputs.
    (
        project
        /
        "RSLC"
    ).mkdir()

    (
        project
        /
        "RSLC_tab"
    ).write_text(
        "",
        encoding="utf-8",
    )


    # Optional project inputs.
    (
        project
        /
        "DEM_prep"
    ).mkdir()

    (
        project
        /
        "GACOS"
    ).mkdir()


    cfg = {
        "schema_version": 1,

        "paths": {
            "work_dir": ".",
            "data_dir": ".",

            "rslc_dir": "RSLC",
            "rslc_tab": "RSLC_tab",

            "dem_dir": "DEM_prep",
            "gacos_dir": "GACOS",

            "output_dir": "output",

            "scratch_dir":
                "output/.scratch",

            "products_dir":
                "output/products",
        },
    }


    config_path = _write_config(
        project
        /
        "pypsds.yaml",
        cfg,
    )


    paths = resolve_project_paths(
        cfg,
        config_path,
    )


    assert paths.work_dir == (
        project.resolve()
    )

    assert paths.data_dir == (
        project.resolve()
    )

    assert paths.rslc_dir == (
        project
        /
        "RSLC"
    ).resolve()

    assert paths.rslc_tab == (
        project
        /
        "RSLC_tab"
    ).resolve()

    assert paths.dem_dir == (
        project
        /
        "DEM_prep"
    ).resolve()

    assert paths.gacos_dir == (
        project
        /
        "GACOS"
    ).resolve()

    assert paths.output_dir == (
        project
        /
        "output"
    ).resolve()

    assert paths.scratch_dir == (
        project
        /
        "output"
        /
        ".scratch"
    ).resolve()

    assert paths.products_dir == (
        project
        /
        "output"
        /
        "products"
    ).resolve()

    assert paths.scratch_dir.is_dir()

    assert paths.products_dir.is_dir()


def test_v11_optional_dem_and_gacos_do_not_block_core(
    tmp_path,
):

    project = (
        tmp_path
        /
        "area_without_optional_corrections"
    )

    project.mkdir()

    (
        project
        /
        "RSLC"
    ).mkdir()

    (
        project
        /
        "RSLC_tab"
    ).write_text(
        "",
        encoding="utf-8",
    )


    cfg = {
        "schema_version": 1,

        "paths": {
            "work_dir": ".",
            "data_dir": ".",

            "rslc_dir": "RSLC",
            "rslc_tab": "RSLC_tab",

            "dem_dir": None,
            "gacos_dir": None,

            "output_dir": "output",
        },
    }


    config_path = _write_config(
        project
        /
        "pypsds.yaml",
        cfg,
    )


    paths = resolve_project_paths(
        cfg,
        config_path,
    )


    assert paths.rslc_dir.is_dir()

    assert paths.rslc_tab.is_file()

    assert paths.dem_dir is None

    assert paths.gacos_dir is None


def test_v11_explicit_external_auxiliary_paths(
    tmp_path,
):

    project = (
        tmp_path
        /
        "project"
    )

    data = (
        tmp_path
        /
        "stack"
    )

    auxiliaries = (
        tmp_path
        /
        "aux"
    )

    project.mkdir()
    data.mkdir()
    auxiliaries.mkdir()


    (
        data
        /
        "RSLC"
    ).mkdir()

    (
        data
        /
        "RSLC_tab"
    ).write_text(
        "",
        encoding="utf-8",
    )


    dem = (
        auxiliaries
        /
        "DEM"
    )

    gacos = (
        auxiliaries
        /
        "GACOS"
    )

    dem.mkdir()
    gacos.mkdir()


    cfg = {
        "schema_version": 1,

        "paths": {
            "work_dir":
                str(project),

            "data_dir":
                str(data),

            "rslc_dir":
                "RSLC",

            "rslc_tab":
                "RSLC_tab",

            "dem_dir":
                str(dem),

            "gacos_dir":
                str(gacos),

            "output_dir":
                "output",
        },
    }


    config_path = _write_config(
        project
        /
        "pypsds.yaml",
        cfg,
    )


    paths = resolve_project_paths(
        cfg,
        config_path,
    )


    assert paths.data_dir == (
        data.resolve()
    )

    assert paths.dem_dir == (
        dem.resolve()
    )

    assert paths.gacos_dir == (
        gacos.resolve()
    )
PY


cat > tests/test_v11_public_config.py <<'PY'
from pathlib import Path

import yaml


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


CONFIGS = (
    ROOT
    / "config"
    / "pypsds.yaml",

    ROOT
    / "config"
    / "pypsds_template.yaml",

    ROOT
    / "pypsds"
    / "resources"
    / "default_config.yaml",
)


def test_v11_public_config_contract():

    for path in CONFIGS:

        cfg = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )

        assert (
            cfg["schema_version"]
            ==
            1
        )


        paths = cfg["paths"]

        assert (
            paths["rslc_dir"]
            ==
            "RSLC"
        )

        assert (
            paths["rslc_tab"]
            ==
            "RSLC_tab"
        )

        assert (
            paths["dem_dir"]
            ==
            "DEM_prep"
        )

        assert (
            paths["gacos_dir"]
            ==
            "GACOS"
        )

        assert (
            paths["scratch_dir"]
            ==
            "output/.scratch"
        )

        assert (
            paths["products_dir"]
            ==
            "output/products"
        )


        # Public defaults must not silently enable
        # optional scientific corrections.

        assert (
            cfg["corrections"]
            ["scla"]
            ["mode"]
            ==
            "disabled"
        )

        assert (
            cfg["corrections"]
            ["atmosphere"]
            ["mode"]
            ==
            "disabled"
        )

        assert (
            cfg["corrections"]
            ["scn"]
            ["mode"]
            ==
            "disabled"
        )


        assert (
            cfg["corrections"]
            ["atmosphere"]
            ["backend"]
            ==
            "gacos"
        )

        assert (
            cfg["corrections"]
            ["scn"]
            ["temporal_window_days"]
            ==
            365.0
        )

        assert (
            cfg["corrections"]
            ["scn"]
            ["wavelength_m"]
            ==
            100.0
        )


        products = cfg["products"]

        assert (
            products["point"]["enabled"]
            is True
        )

        assert (
            products["point"]["crs"]
            ==
            "EPSG:4326"
        )

        assert (
            products["quicklook"]["enabled"]
            is False
        )

        assert (
            products["quicklook"]["scientific_product"]
            is False
        )


def test_v11_public_templates_contain_no_absolute_project_paths():

    forbidden = (
        "/home/ubuntu/",
        "/mnt/",
        "/media/",
    )

    for path in CONFIGS:

        text = path.read_text(
            encoding="utf-8"
        )

        for token in forbidden:

            assert token not in text
PY

echo "[4/7] v1.1 path/config tests created"


# ==============================================================================
# 5. Write public contract documentation
# ==============================================================================

cat > docs/release_v1_1/config_path_contract.md <<'MD'
# pyPSDS-GAMMA v1.1 project configuration contract

## Design goal

A new study area must be runnable without modifying Python source code.

The project configuration is located in the study-area directory and all
relative paths are resolved from the project/data roots.

## Required inputs

- `RSLC/`
- `RSLC_tab`

## Optional auxiliary inputs

- `DEM_prep/`
- `GACOS/`

The auxiliary directories are only required when the corresponding
processing feature is enabled.

## Managed outputs

- `output/`
- `output/.scratch/`
- `output/products/`

## Scientific correction defaults

For backward compatibility and reproducibility, v1.1 does not silently
enable new corrections.

The public defaults remain:

- SCLA: disabled
- atmospheric correction: disabled
- SCN: disabled

A production configuration must enable these explicitly.

## Point-first product policy

The primary geodetic product is the original PS/DS point product.

Supported public point formats:

- Parquet
- GeoPackage
- CSV

Raster products are optional quicklooks and are not the authoritative
scientific product.

## Stage naming

The existing computational stage names remain unchanged during the v1.1
migration.

Development identifiers are removed from the public interface only after
the corresponding tests and production policies have been migrated.
MD

echo "[5/7] contract documentation created"


# ==============================================================================
# 6. Static checks + focused regression
# ==============================================================================

echo
echo "--------------------------------------------------------------------------------"
echo "STATIC CHECK"
echo "--------------------------------------------------------------------------------"

python -m py_compile \
    pypsds/project.py \
    pypsds/config.py

echo "PY_COMPILE PASS"


echo
echo "--------------------------------------------------------------------------------"
echo "FOCUSED TESTS"
echo "--------------------------------------------------------------------------------"

pytest -q \
    tests/test_production_config.py \
    tests/test_v11_project_paths.py \
    tests/test_v11_public_config.py

echo "FOCUSED TESTS PASS"


# ==============================================================================
# 7. Full regression suite
# ==============================================================================

echo
echo "--------------------------------------------------------------------------------"
echo "FULL TEST SUITE"
echo "--------------------------------------------------------------------------------"

pytest -q

echo
echo "================================================================================"
echo "V1.1 CONFIG/PATH CONTRACT RESULT: PASS"
echo "================================================================================"
echo "backup:"
echo "${BACKUP}"
echo
echo "changed:"
echo "  pypsds/project.py"
echo "  config/pypsds.yaml"
echo "  config/pypsds_template.yaml"
echo "  pypsds/resources/default_config.yaml"
echo "  tests/test_v11_project_paths.py"
echo "  tests/test_v11_public_config.py"
echo "  docs/release_v1_1/config_path_contract.md"
echo
echo "NOT changed:"
echo "  VERSION"
echo "  pyproject.toml"
echo "  pipeline stage names"
echo "  numerical algorithms"
echo "  production defaults for optional corrections"
echo "================================================================================"
