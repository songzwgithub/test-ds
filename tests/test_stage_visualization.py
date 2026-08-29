from pathlib import Path

from pypsds.monitoring import stage_visualization
from pypsds.monitoring.visualization import (
    VISUALIZATION_PROFILE,
)


def test_stage_visualization_public_entrypoint_is_final():
    assert VISUALIZATION_PROFILE == "scientific_final_v1"
    assert len(stage_visualization.STAGES) == 39

    assert stage_visualization.STAGES[29:34] == [
        "unwrap_finalize",
        "point_geometry",
        "residual_ramp",
        "timeseries_inversion",
        "reference",
    ]


def test_stage_visualization_renderer_is_packaged():
    p = (
        Path(stage_visualization.__file__).resolve().parent
        / "visualization/renderers.py"
    )
    assert p.is_file()
