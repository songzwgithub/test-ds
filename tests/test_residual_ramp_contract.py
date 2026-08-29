import numpy as np

from pypsds.corrections.residual_ramp import (
    build_temporal_design,
    cell_balanced_weights,
    huber_plane,
    network_project_ifg_slopes,
)
from pypsds.modules import MODULE_BY_NAME
from pypsds.pipeline import STAGE_INDEX


def test_residual_ramp_stage_order_and_module_contract():
    assert (
        STAGE_INDEX["unwrap_finalize"]
        < STAGE_INDEX["point_geometry"]
        < STAGE_INDEX["residual_ramp"]
        < STAGE_INDEX["timeseries_inversion"]
        < STAGE_INDEX["reference"]
    )

    assert MODULE_BY_NAME["timeseries"].stage_names == (
        "point_geometry",
        "residual_ramp",
        "timeseries_inversion",
    )

    assert MODULE_BY_NAME["corrections"].stage_names == (
        "reference",
        "atmosphere_correction",
        "scla",
        "scn",
    )


def test_equal_total_weight_per_cell():
    xy = np.array([
        [100.0, 100.0],
        [200.0, 100.0],
        [300.0, 100.0],
        [2100.0, 100.0],
        [2200.0, 100.0],
    ])

    w, cell, meta = cell_balanced_weights(
        xy,
        cell_size_m=2000.0,
    )

    assert meta["occupied_cells"] == 2

    totals = [
        float(np.sum(w[cell == cid]))
        for cid in np.unique(cell)
    ]

    np.testing.assert_allclose(
        totals,
        np.full(len(totals), totals[0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_huber_degree1_recovers_known_plane():
    rng = np.random.default_rng(1234)

    gx, gy = np.meshgrid(
        np.linspace(-4.0, 4.0, 41),
        np.linspace(-4.0, 4.0, 41),
    )

    X = np.column_stack((
        gx.ravel(),
        gy.ravel(),
        np.ones(gx.size),
    ))

    true = np.array([0.025, -0.017, 0.31])

    y = X @ true
    y += rng.normal(0.0, 0.003, y.size)

    out = rng.choice(
        y.size,
        size=max(5, y.size // 10),
        replace=False,
    )
    y[out] += 0.5

    beta, scale, used = huber_plane(
        X,
        y,
        np.ones(y.size),
        iterations=8,
        delta=1.345,
    )

    np.testing.assert_allclose(
        beta[:2],
        true[:2],
        atol=0.004,
    )
    assert np.isfinite(scale)
    assert used >= 1


def test_network_projection_is_integrable():
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 2),
        (1, 3),
    ]

    acq = np.array([
        [0.0, 0.0],
        [0.10, -0.02],
        [0.16, 0.03],
        [0.21, 0.08],
    ])

    A = build_temporal_design(
        edges,
        4,
        reference_idx=0,
    )

    direct = A @ acq[1:, :]
    direct = direct.copy()
    direct[3, 0] += 0.01
    direct[4, 1] -= 0.01

    projected, recovered, meta = network_project_ifg_slopes(
        edges,
        4,
        direct,
        reference_idx=0,
    )

    assert meta["design_rank"] == 3

    np.testing.assert_allclose(
        projected,
        A @ recovered[1:, :],
        atol=1e-12,
        rtol=1e-12,
    )
