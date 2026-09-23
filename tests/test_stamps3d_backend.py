from __future__ import annotations

import numpy as np

from pypsds.stamps3d_backend import (
    build_design_matrix,
    build_dense_support,
    build_representative_grid,
    synchronize_temporal_integer_cycles,
    wrap_phase,
)


def test_representative_grid_prefers_ps_over_ds():
    rows = np.asarray([0, 1, 50], dtype=np.int32)
    cols = np.asarray([0, 1, 50], dtype=np.int32)
    typ = np.asarray([2, 1, 2], dtype=np.uint8)
    tc = np.asarray([0.999, np.nan, 0.9], dtype=np.float32)

    g = build_representative_grid(
        rows,
        cols,
        typ,
        tc,
        row_spacing_m=10.0,
        col_spacing_m=10.0,
        grid_size_m=100.0,
    )
    # Points 0 and 1 share the first 100-m cell; PS point 1 wins.
    assert int(g["rep_grid"][0, 0]) == 1
    assert set(map(int, g["rep_ids"])) == {1, 2}


def test_dense_support_maps_every_cell_to_real_node():
    node_grid = np.full((5, 6), -1, dtype=np.int32)
    node_grid[0, 0] = 0
    node_grid[4, 5] = 1
    q = np.asarray([0.95, 0.85], dtype=np.float32)

    nearest, distance, corr = build_dense_support(
        node_grid,
        q,
        grid_size_m=200.0,
        gap_scale_m=600.0,
        min_corr=0.03,
    )
    assert nearest.shape == node_grid.shape
    assert np.all(nearest >= 0)
    assert np.all(np.isfinite(distance))
    assert float(corr.min()) >= 0.03
    assert float(corr.max()) <= 0.999


def test_temporal_integer_sync_recovers_integrable_network(tmp_path):
    edges = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
    A = build_design_matrix(edges, 4, 0)

    true_unknown = np.asarray(
        [
            [1, 2, 3],
            [-1, -1, 0],
            [0, 2, 1],
            [3, 1, -2],
        ],
        dtype=np.int16,
    )
    K = np.rint(true_unknown.astype(np.float64) @ A.T).astype(np.int16)

    out = synchronize_temporal_integer_cycles(
        K,
        edges,
        ndate=4,
        reference_idx=0,
        batch_size=2,
        iterations=2,
        edge_bad_threshold=0.10,
        strict_mismatch_fraction=0.0,
        blas_threads=1,
        work_dir=tmp_path,
    )

    cyc = np.load(out["cycles_path"])
    assert cyc.shape == (4, 4)
    assert np.all(cyc[:, 0] == 0)
    # A spatially common acquisition-integer gauge is not observable from
    # interferometric differences alone; it is removed later by the spatial
    # reference.  Relative node integer histories must nevertheless be exact.
    delta = cyc[:, 1:].astype(np.int32) - true_unknown.astype(np.int32)
    assert np.all(delta == delta[:1, :])
    assert np.all(out["node_valid"])
    assert np.all(out["final_edge_bad_fraction"] == 0)


def test_restoration_identity_is_wrap_congruent():
    point = np.asarray([[2.8, -2.9, 0.2]], dtype=np.float64)
    rep_wrapped = np.asarray([[-2.7, 2.95, -0.1]], dtype=np.float64)
    rep_integer = np.asarray([[3, -2, 5]], dtype=np.float64)
    rep_unwrapped = rep_wrapped + 2.0 * np.pi * rep_integer
    restored = rep_unwrapped + wrap_phase(point - rep_wrapped)
    assert np.max(np.abs(wrap_phase(restored - point))) < 1.0e-12


def test_temporal_sync_removes_common_nonintegrable_ifg_gauge(tmp_path):
    edges = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
    A = build_design_matrix(edges, 4, 0)
    true_unknown = np.asarray(
        [[1, 2, 3], [0, 2, 1], [3, 1, -2], [-2, 1, 4]],
        dtype=np.int16,
    )
    base = np.rint(true_unknown.astype(np.float64) @ A.T).astype(np.int16)
    # A deliberately non-integrable IFG-wise offset common to every node.
    common = np.asarray([1, 0, 0, 0, 0], dtype=np.int16)
    K = base + common[None, :]

    out = synchronize_temporal_integer_cycles(
        K,
        edges,
        ndate=4,
        reference_idx=0,
        batch_size=2,
        iterations=3,
        edge_bad_threshold=0.25,
        strict_mismatch_fraction=0.0,
        blas_threads=1,
        work_dir=tmp_path,
    )
    cyc = np.load(out["cycles_path"])
    delta = cyc[:, 1:].astype(np.int32) - true_unknown.astype(np.int32)
    assert np.all(delta == delta[:1, :])
    assert np.all(out["node_valid"])
