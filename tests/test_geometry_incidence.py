import numpy as np

from pypsds.geometry import (
    build_row_orbit_geometry,
    compute_incidence_rad,
    orbit_position,
)


def test_orbit_position_linear_motion():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=np.float64,
    )

    vel = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )

    result = orbit_position(
        np.array(
            [0.0, 5.0, 10.0],
            dtype=np.float64,
        ),
        sv_t0=0.0,
        sv_dt=10.0,
        position_m=pos,
        velocity_m_s=vel,
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 10.0, 15.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )


def _write_test_par(path):
    path.write_text(
        "\n".join(
            [
                "azimuth_lines: 3",
                "azimuth_line_time: 2.0",
                "center_time: 100.0",
                "number_of_state_vectors: 2",
                "time_of_first_state_vector: 98.0",
                "state_vector_interval: 4.0",
                "state_vector_position_1: 7000000.0 -1000000.0 0.0",
                "state_vector_velocity_1: 0.0 0.0 0.0",
                "state_vector_position_2: 7000000.0 -1000000.0 0.0",
                "state_vector_velocity_2: 0.0 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_row_orbit_geometry_center_time(tmp_path):
    par = tmp_path / "reference.rslc.par"

    _write_test_par(
        par
    )

    result = build_row_orbit_geometry(
        par
    )

    np.testing.assert_array_equal(
        result.row_time_s,
        np.array(
            [98.0, 100.0, 102.0],
            dtype=np.float64,
        ),
    )

    assert (
        result.satellite_xyz_m.shape
        ==
        (3, 3)
    )

    np.testing.assert_allclose(
        result.satellite_xyz_m,
        np.array(
            [
                [7000000.0, -1000000.0, 0.0],
                [7000000.0, -1000000.0, 0.0],
                [7000000.0, -1000000.0, 0.0],
            ]
        ),
        rtol=0.0,
        atol=1.0e-9,
    )


def test_compute_incidence_contract(tmp_path):
    par = tmp_path / "reference.rslc.par"

    _write_test_par(
        par
    )

    result = compute_incidence_rad(
        longitude_deg=np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        latitude_deg=np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        height_m=np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        radar_row=np.array(
            [0, 1, 2],
            dtype=np.int32,
        ),
        reference_rslc_par=par,
    )

    assert result.dtype == np.float32
    assert result.shape == (3,)

    assert np.all(
        np.isfinite(
            result
        )
    )

    assert np.all(
        result > 0.0
    )

    assert np.all(
        result < np.pi / 2.0
    )

    np.testing.assert_array_equal(
        result,
        np.full(
            3,
            result[0],
            dtype=np.float32,
        ),
    )
