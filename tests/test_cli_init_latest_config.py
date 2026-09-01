from __future__ import annotations

from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import yaml

from pypsds.cli import cmd_init


def test_init_writes_current_packaged_default_byte_for_byte(tmp_path: Path):
    project = tmp_path / "project"

    cmd_init(
        SimpleNamespace(
            project_dir=str(project),
            force=False,
        )
    )

    generated = (
        project
        /
        "pypsds.yaml"
    ).read_text(
        encoding="utf-8"
    )

    packaged = resources.files(
        "pypsds.resources"
    ).joinpath(
        "default_config.yaml"
    ).read_text(
        encoding="utf-8"
    )

    assert generated == packaged


def test_init_contains_current_fast_runtime_defaults(tmp_path: Path):
    project = tmp_path / "project"

    cmd_init(
        SimpleNamespace(
            project_dir=str(project),
            force=False,
        )
    )

    cfg = yaml.safe_load(
        (
            project
            /
            "pypsds.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert int(
        cfg["runtime"]["phase_link_prefetch_tiles"]
    ) == 1

    phase = cfg["phase_correction"]

    assert int(
        phase["command_retries"]
    ) == 2

    assert float(
        phase["retry_backoff_seconds"]
    ) == 1.0

    assert (
        phase["parallel"]["spatial_workers"]
        ==
        "auto"
    )

    assert (
        phase["parallel"]["pair_workers"]
        ==
        "auto"
    )


def test_init_force_refreshes_existing_config_to_latest_template(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    target = project / "pypsds.yaml"
    target.write_text(
        "obsolete: true\n",
        encoding="utf-8",
    )

    cmd_init(
        SimpleNamespace(
            project_dir=str(project),
            force=True,
        )
    )

    generated = target.read_text(
        encoding="utf-8"
    )

    packaged = resources.files(
        "pypsds.resources"
    ).joinpath(
        "default_config.yaml"
    ).read_text(
        encoding="utf-8"
    )

    assert generated == packaged
    assert "obsolete: true" not in generated
