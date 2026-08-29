import numpy as np

from pypsds.corrections.residual_ramp import (
    cell_balanced_weights,
)


def test_all_reliable_ps_are_kept_and_cells_are_balanced():
    xy = np.array([
        [100.0, 100.0],
        [200.0, 100.0],
        [300.0, 100.0],
        [400.0, 100.0],
        [2100.0, 100.0],
        [2200.0, 100.0],
    ])

    w, cell, meta = cell_balanced_weights(
        xy,
        cell_size_m=2000.0,
    )

    assert w.size == xy.shape[0]
    assert cell.size == xy.shape[0]
    assert meta["occupied_cells"] == 2

    sums = [
        float(np.sum(w[cell == cid]))
        for cid in np.unique(cell)
    ]

    np.testing.assert_allclose(
        sums,
        [sums[0], sums[0]],
        atol=1e-12,
        rtol=1e-12,
    )
