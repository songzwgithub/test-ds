from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile

import pytest

import pypsds
from pypsds.pipeline import STAGES


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

GATE_PATH = (
    ROOT
    / "tools"
    / "release_gate.py"
)


def _load_gate():

    spec = (
        importlib.util
        .spec_from_file_location(
            "pypsds_release_gate_test",
            GATE_PATH,
        )
    )

    assert spec is not None
    assert spec.loader is not None

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def test_release_gate_identity_uses_authoritative_sources():

    gate = _load_gate()

    identity = (
        gate.release_identity()
    )

    assert (
        identity["version"]
        ==
        pypsds.__version__
    )

    assert (
        identity["stage_names"]
        ==
        tuple(
            stage.name
            for stage in STAGES
        )
    )

    assert (
        identity["stage_scripts"]
        ==
        tuple(
            stage.script
            for stage in STAGES
        )
    )


def test_release_gate_has_no_current_release_literals():

    text = GATE_PATH.read_text(
        encoding="utf-8"
    )

    assert '"1.1.0"' not in text
    assert "'1.1.0'" not in text

    assert "== 38" not in text
    assert "!= 38" not in text

    assert (
        "release_identity()"
        in text
    )


def test_stage_sequence_guard_rejects_reordering():

    gate = _load_gate()

    with pytest.raises(
        RuntimeError,
        match="stage sequence mismatch",
    ):

        gate._assert_stage_sequence(
            label="test",
            actual=(
                "b",
                "a",
            ),
            expected=(
                "a",
                "b",
            ),
        )


def test_wheel_metadata_version_reader(
    tmp_path,
):

    gate = _load_gate()

    wheel = (
        tmp_path
        / "fake.whl"
    )

    with zipfile.ZipFile(
        wheel,
        "w",
    ) as zf:

        zf.writestr(
            (
                "pypsds_gamma-9.8.7."
                "dist-info/METADATA"
            ),
            (
                "Metadata-Version: 2.1\n"
                "Name: pypsds-gamma\n"
                "Version: 9.8.7\n"
            ),
        )

    assert (
        gate._wheel_metadata_version(
            wheel
        )
        ==
        "9.8.7"
    )
