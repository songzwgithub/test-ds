import numpy as np

import pypsds
from pypsds.modules import MODULES
from pypsds.pipeline import STAGES
from pypsds.monitoring.inversion import weighted_operator
from pypsds.monitoring.reference import choose_reference_region
from pypsds.monitoring.vertical import vertical_factor
from pypsds.monitoring.decompose import (
    los_geometry_eu,
    solve_east_up,
)


def test_formal_version_keeps_pipeline_shape():
    assert pypsds.__version__ == "1.3.0"
    assert len(MODULES) == 9
    assert len(STAGES) == 38


def test_uniform_wls_equals_ols():
    A = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    got = weighted_operator(A, np.ones(A.shape[0]))
    ref = np.linalg.pinv(A, rcond=1e-12)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_vertical_factor_sign():
    inc = np.deg2rad(np.asarray([30.0, 40.0]))
    up = vertical_factor(inc, "up")
    down = vertical_factor(inc, "down")
    np.testing.assert_allclose(up, 1.0 / np.cos(inc))
    np.testing.assert_allclose(down, -up)


def test_two_track_decomposition():
    inc_a = np.deg2rad(np.asarray([35.0, 35.0]))
    inc_d = np.deg2rad(np.asarray([36.0, 36.0]))
    h_a, h_d = -13.0, 193.0
    east = np.asarray([5.0, -3.0])
    up = np.asarray([-20.0, 4.0])
    ue_a, uu_a = los_geometry_eu(inc_a, h_a)
    ue_d, uu_d = los_geometry_eu(inc_d, h_d)
    los_a = ue_a * east + uu_a * up
    los_d = ue_d * east + uu_d * up
    sol = solve_east_up(
        los_a, los_d, inc_a, inc_d, h_a, h_d,
        np.ones(2), np.ones(2),
    )
    np.testing.assert_allclose(sol["east"], east, atol=1e-10)
    np.testing.assert_allclose(sol["up"], up, atol=1e-10)


def test_auto_reference_prefers_stable_cluster():
    rng = np.random.default_rng(1)
    stable = rng.normal([0.0, 0.0], [30.0, 30.0], size=(120, 2))
    moving = rng.normal([1500.0, 0.0], [30.0, 30.0], size=(120, 2))
    xy = np.vstack((stable, moving))
    rate = np.r_[np.full(120, 0.01), np.full(120, 1.0)]
    rms = np.r_[np.full(120, 0.05), np.full(120, 0.5)]
    best, _ = choose_reference_region(
        xy, rate, rms,
        radius_m=150.0,
        cell_size_m=200.0,
        min_points=80,
    )
    assert float(best["median_abs_rate"]) < 0.1
