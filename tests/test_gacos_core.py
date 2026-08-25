from pathlib import Path

import numpy as np

from pypsds.corrections.gacos import (
    discover_phase_dates,
    normalize_dates,
    par_scalar,
    read_rsc,
)


def test_gacos_par_scalar(tmp_path):

    path = tmp_path / "example.par"

    path.write_text(
        "radar_frequency: 5.405000000e9 Hz\n"
        "width: 241\n",
        encoding="utf-8",
    )

    assert (
        par_scalar(
            path,
            "radar_frequency",
        )
        ==
        5.405e9
    )


def test_gacos_read_rsc(tmp_path):

    path = tmp_path / "x.ztd.rsc"

    path.write_text(
        "WIDTH 241\n"
        "FILE_LENGTH 241\n"
        "X_FIRST 23.6\n"
        "Y_FIRST 38.2\n",
        encoding="utf-8",
    )

    values = read_rsc(
        path
    )

    assert values["WIDTH"] == "241"
    assert values["FILE_LENGTH"] == "241"
    assert values["X_FIRST"] == "23.6"
    assert values["Y_FIRST"] == "38.2"


def test_gacos_normalize_dates():

    values = np.array(
        [
            "20141006",
            "20141018",
            "20141030",
        ]
    )

    assert normalize_dates(values) == [
        "20141006",
        "20141018",
        "20141030",
    ]


def test_gacos_discover_phase_dates_portable(tmp_path):

    expected = [
        "20141006",
        "20141018",
        "20141030",
    ]

    np.save(
        tmp_path / "phase_dates.npy",
        np.asarray(
            expected,
            dtype="U8",
        ),
    )

    dates, sources = discover_phase_dates(
        tmp_path,
        expected,
    )

    assert dates == expected
    assert len(sources) >= 1
