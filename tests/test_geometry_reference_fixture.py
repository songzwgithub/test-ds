from pathlib import Path
import hashlib
import json

import numpy as np


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DATA = (
    ROOT
    /
    "tests"
    /
    "data"
)


NPZ = (
    DATA
    /
    "geometry_reference_v1.npz"
)


JSON = (
    DATA
    /
    "geometry_reference_v1.json"
)


def _sha256(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def test_geometry_reference_fixture_integrity():

    assert NPZ.is_file()
    assert JSON.is_file()

    meta = json.loads(
        JSON.read_text(
            encoding="utf-8"
        )
    )

    assert meta[
        "contract"
    ] == (
        "pyPSDS-GAMMA-geometry-reference-v1"
    )

    assert meta[
        "point_count"
    ] == 881315

    assert (
        _sha256(
            NPZ
        )
        ==
        meta[
            "fixture"
        ][
            "sha256"
        ]
    )

    with np.load(
        NPZ
    ) as z:

        idx = z[
            "point_index"
        ]

        lon = z[
            "longitude_deg"
        ]

        lat = z[
            "latitude_deg"
        ]

        inc = z[
            "incidence_rad"
        ]


    assert idx.ndim == 1

    assert (
        idx.size
        ==
        meta[
            "sample_count"
        ]
    )

    assert (
        lon.shape
        ==
        lat.shape
        ==
        inc.shape
        ==
        idx.shape
    )

    assert np.all(
        np.diff(
            idx
        )
        >
        0
    )

    assert int(
        idx[0]
    ) >= 0

    assert int(
        idx[-1]
    ) < 881315

    assert np.all(
        np.isfinite(
            lon
        )
    )

    assert np.all(
        np.isfinite(
            lat
        )
    )

    assert np.all(
        np.isfinite(
            inc
        )
    )

    assert np.all(
        (
            lon >= -180
        )
        &
        (
            lon <= 180
        )
    )

    assert np.all(
        (
            lat >= -90
        )
        &
        (
            lat <= 90
        )
    )

    assert np.all(
        (
            inc > 0
        )
        &
        (
            inc < np.pi / 2
        )
    )
