from pathlib import Path

from pypsds.stages import build_spatial_graph
from pypsds.stages import assess_local_spatial_graph


def test_spatial_graph_sources_use_canonical_reference_rslc():
    for module in (
        build_spatial_graph,
        assess_local_spatial_graph,
    ):
        source = Path(module.__file__).read_text(
            encoding="utf-8"
        )

        assert "resolve_geometry_inputs" in source
        assert "geometry.reference_rslc_par" in source

        assert (
            '"phase_correction.radar_height.geometry_par"'
            not in source
        )
