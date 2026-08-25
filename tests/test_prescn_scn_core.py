from pathlib import Path

from pypsds.corrections import pre_scn
from pypsds.corrections import scn


def test_prescn_public_core_exists():

    for name in (
        "read_par",
        "par_scalar",
        "parse_vector",
        "parse_base",
        "geometry_factors",
        "fast_bsm",
    ):

        assert callable(
            getattr(
                pre_scn,
                name,
            )
        )


def test_scn_public_core_exists():

    for name in (
        "temporal_weight_matrix",
        "build_cell_index",
        "build_safe_offsets",
        "spatial_gaussian_exact",
    ):

        assert callable(
            getattr(
                scn,
                name,
            )
        )


def test_prescn_scn_core_is_portable():

    for module in (
        pre_scn,
        scn,
    ):

        path = Path(
            module.__file__
        )

        text = path.read_text(
            encoding="utf-8"
        )

        assert "/home/ubuntu" not in text
        assert "/Downloads/psds" not in text
