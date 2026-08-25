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
