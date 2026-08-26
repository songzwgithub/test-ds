from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest
import yaml

from pypsds.project import resolve_project_paths


def portable_cfg():
    return {
        "schema_version": 1,
        "paths": {
            "work_dir": ".",
            "data_dir": ".",
            "rslc_dir": None,
            "rslc_tab": None,
            "output_dir": "output",
            "dem_dir": None,
            "gacos_dir": None,
            "scratch_dir": "output/.scratch",
            "products_dir": "output/products",
        },
    }


def test_packaged_default_has_no_machine_input_paths():
    text = (
        resources.files("pypsds.resources")
        .joinpath("default_config.yaml")
        .read_text(encoding="utf-8")
    )

    cfg = yaml.safe_load(text)

    assert cfg["paths"]["rslc_dir"] is None
    assert cfg["paths"]["rslc_tab"] is None
    assert cfg["paths"]["dem_dir"] is None
    assert cfg["paths"]["gacos_dir"] is None

    assert "/home/" not in text
    assert "/mnt/" not in text
    assert "/media/" not in text


def test_conventional_project_inputs_are_discovered(tmp_path):
    config = tmp_path / "pypsds.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")

    rslc = tmp_path / "RSLC_cropped"
    rslc.mkdir()

    dem = tmp_path / "DEM"
    dem.mkdir()

    gacos = tmp_path / "gacos"
    gacos.mkdir()

    tab = tmp_path / "project_RSLC_stack_tab.txt"
    tab.write_text("", encoding="utf-8")

    paths = resolve_project_paths(
        portable_cfg(),
        config,
    )

    assert paths.work_dir == tmp_path.resolve()
    assert paths.data_dir == tmp_path.resolve()

    assert paths.rslc_dir == rslc.resolve()
    assert paths.rslc_tab == tab.resolve()

    assert paths.dem_dir == dem.resolve()
    assert paths.gacos_dir == gacos.resolve()

    assert paths.output_dir == (tmp_path / "output").resolve()

    assert (
        paths.scratch_dir
        ==
        (tmp_path / "output/.scratch").resolve()
    )

    assert (
        paths.products_dir
        ==
        (tmp_path / "output/products").resolve()
    )


def test_exact_rslc_tab_has_priority(tmp_path):
    config = tmp_path / "pypsds.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")

    (tmp_path / "RSLC").mkdir()

    exact = tmp_path / "RSLC_tab"
    exact.write_text("", encoding="utf-8")

    # This must not make discovery ambiguous because the canonical
    # exact file has priority.
    other = tmp_path / "other_RSLC_stack_tab.txt"
    other.write_text("", encoding="utf-8")

    paths = resolve_project_paths(
        portable_cfg(),
        config,
    )

    assert paths.rslc_tab == exact.resolve()


def test_rslc_directory_precedence_is_deterministic(tmp_path):
    config = tmp_path / "pypsds.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")

    cropped = tmp_path / "RSLC_cropped"
    normal = tmp_path / "RSLC"

    cropped.mkdir()
    normal.mkdir()

    (tmp_path / "RSLC_tab").write_text(
        "",
        encoding="utf-8",
    )

    paths = resolve_project_paths(
        portable_cfg(),
        config,
    )

    assert paths.rslc_dir == cropped.resolve()


def test_ambiguous_noncanonical_rslc_tabs_fail(tmp_path):
    config = tmp_path / "pypsds.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")

    (tmp_path / "RSLC").mkdir()

    (tmp_path / "A_RSLC_stack_tab.txt").write_text(
        "",
        encoding="utf-8",
    )

    (tmp_path / "B_RSLC_stack_tab.txt").write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="Ambiguous RSLC_tab",
    ):
        resolve_project_paths(
            portable_cfg(),
            config,
        )


def test_explicit_project_paths_override_discovery(tmp_path):
    config = tmp_path / "pypsds.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")

    auto_rslc = tmp_path / "RSLC"
    auto_rslc.mkdir()

    explicit_rslc = tmp_path / "custom_stack"
    explicit_rslc.mkdir()

    auto_tab = tmp_path / "RSLC_tab"
    auto_tab.write_text("", encoding="utf-8")

    explicit_tab = tmp_path / "custom_stack.list"
    explicit_tab.write_text("", encoding="utf-8")

    cfg = portable_cfg()

    cfg["paths"]["rslc_dir"] = "custom_stack"
    cfg["paths"]["rslc_tab"] = "custom_stack.list"

    paths = resolve_project_paths(
        cfg,
        config,
    )

    assert paths.rslc_dir == explicit_rslc.resolve()
    assert paths.rslc_tab == explicit_tab.resolve()


def test_authoritative_config_has_no_repository_duplicates():
    root = Path(__file__).resolve().parents[1]

    authoritative = (
        root
        / "pypsds"
        / "resources"
        / "default_config.yaml"
    )

    assert authoritative.is_file()

    assert not (
        root
        / "config"
        / "pypsds.yaml"
    ).exists()

    assert not (
        root
        / "config"
        / "pypsds_template.yaml"
    ).exists()

