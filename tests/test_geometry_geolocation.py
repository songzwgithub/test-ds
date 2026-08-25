import numpy as np

from pypsds.geometry import (
    build_ipta_point_list,
    read_gamma_point_values,
)


def test_ipta_point_list_exact_binary(tmp_path):
    cols = np.array(
        [3, 40, 500],
        dtype=np.int32,
    )

    rows = np.array(
        [7, 80, 600],
        dtype=np.int32,
    )

    path = (
        tmp_path
        / "points.plist"
    )

    build_ipta_point_list(
        cols,
        rows,
        path,
    )


    expected = np.array(
        [
            [3, 7],
            [40, 80],
            [500, 600],
        ],
        dtype=">i4",
    ).tobytes()


    assert path.read_bytes() == expected

    assert (
        path.stat().st_size
        ==
        3 * 8
    )


def test_gamma_point_output_big_endian(tmp_path):
    source = np.array(
        [
            23.625,
            23.75,
            38.125,
        ],
        dtype=">f4",
    )

    path = (
        tmp_path
        / "values.gamma_pt"
    )

    source.tofile(
        path
    )


    result = read_gamma_point_values(
        path,
        expected_count=3,
    )


    assert result.dtype == np.float64

    np.testing.assert_array_equal(
        result,
        source.astype(
            np.float64
        ),
    )
