from pathlib import Path

from pypsds.inversion import los_timeseries


def test_final_los_helpers_exist():

    assert callable(
        los_timeseries.read_par
    )

    assert callable(
        los_timeseries.par_scalar
    )


def test_final_los_helper_core_portable():

    path = Path(
        los_timeseries.__file__
    )

    text = path.read_text(
        encoding="utf-8"
    )

    assert "/home/ubuntu" not in text
    assert "/Downloads/psds" not in text
    assert "gacos_geometry" not in text
