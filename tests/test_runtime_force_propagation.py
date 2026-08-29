from pathlib import Path

import pypsds.pipeline as pipeline


def test_pipeline_propagates_force_to_nested_runtimes():
    source = Path(
        pipeline.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert (
        'env["PYPSDS_FORCE"] = "1" if force else "0"'
        in source
    )


def test_gacos_accepts_public_force_contract():
    root = (
        Path(pipeline.__file__)
        .resolve()
        .parent
    )

    source = (
        root
        / "runtime_backend"
        / "gacos_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "PYPSDS_FORCE" in source
    assert "P15_FORCE" in source
    assert (
        "pypsds run --force"
        in source
    )
