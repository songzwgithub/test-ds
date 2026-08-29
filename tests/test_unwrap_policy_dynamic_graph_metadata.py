from pathlib import Path

from pypsds.stages import finalize_unwrap_policy


def test_unwrap_policy_uses_dynamic_spatial_graph_metadata():
    source = Path(
        finalize_unwrap_policy.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert "spatial_graph_manifest.json" in source
    assert "selected_local_k" in source
    assert "core_radius" in source
    assert "expected_residual_components" in source

    assert "R4-K8 local component" not in source
    assert "historical 102" not in source
