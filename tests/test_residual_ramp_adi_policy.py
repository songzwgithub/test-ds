import numpy as np

from pypsds.corrections.residual_ramp import select_balanced_anchors


def test_lowest_adi_ps_selected_per_metric_cell():
    xy = np.array([
        [100.0, 100.0],
        [200.0, 100.0],
        [300.0, 100.0],
        [2100.0, 100.0],
        [2200.0, 100.0],
        [2300.0, 100.0],
    ])

    adi = np.array([0.20, 0.05, 0.10, 0.16, 0.04, 0.08])
    quality = 1.0 - adi

    anchors, cells = select_balanced_anchors(
        xy,
        quality,
        cell_size_m=2000.0,
        anchors_per_cell=2,
    )

    assert cells == 2
    assert np.array_equal(
        anchors,
        np.array([1, 2, 4, 5], dtype=np.int64),
    )
