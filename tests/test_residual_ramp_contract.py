from __future__ import annotations

import numpy as np

from pypsds.corrections.residual_ramp import (
    fit_epoch_planes,
    select_balanced_anchors,
)
from pypsds.modules import MODULE_BY_NAME
from pypsds.pipeline import STAGE_INDEX, STAGES


def test_residual_ramp_stage_order_and_module_contract():
    names = [x.name for x in STAGES]

    assert "residual_ramp" in names
    assert (
        STAGE_INDEX["point_geometry"]
        < STAGE_INDEX["residual_ramp"]
        < STAGE_INDEX["reference"]
    )

    corrections = MODULE_BY_NAME["corrections"].stage_names
    assert corrections == (
        "point_geometry",
        "residual_ramp",
        "reference",
        "atmosphere_correction",
        "scla",
        "scn",
    )


def test_balanced_huber_degree1_recovers_known_plane():
    rng = np.random.default_rng(1234)

    gx, gy = np.meshgrid(
        np.linspace(-4000.0, 4000.0, 41),
        np.linspace(-4000.0, 4000.0, 41),
    )
    xy = np.column_stack((gx.ravel(), gy.ravel()))
    n = xy.shape[0]

    q = rng.uniform(0.80, 1.0, n)

    anchors, ncells = select_balanced_anchors(
        xy,
        q,
        cell_size_m=2000.0,
        anchors_per_cell=8,
    )

    assert anchors.size >= 30
    assert ncells >= 16

    X = np.column_stack(
        (
            xy[anchors, 0] / 1000.0,
            xy[anchors, 1] / 1000.0,
            np.ones(anchors.size),
        )
    )

    true = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.025, -0.017, 0.31],
            [-0.011, 0.032, -0.20],
        ],
        dtype=float,
    )

    ph = X @ true.T
    ph += rng.normal(0.0, 0.003, ph.shape)

    out = rng.choice(
        anchors.size,
        size=max(5, anchors.size // 10),
        replace=False,
    )
    ph[out, 1] += 0.5
    ph[out, 2] -= 0.4

    coeff, scale, used = fit_epoch_planes(
        X,
        ph,
        q[anchors] ** 2,
        iterations=8,
        delta=1.345,
        temporal_reference_index=0,
    )

    assert np.array_equal(coeff[0], np.zeros(3))
    assert np.allclose(
        coeff[1:, :2],
        true[1:, :2],
        atol=0.004,
    )
    assert np.all(np.isfinite(scale))
    assert np.all(used[1:] >= 1)
