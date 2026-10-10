import numpy as np
import pytest
from scipy import ndimage

from pypsds.unwrap.spatial_topology import (
    build_occupied_spatial_edges, summarize_spatial_topology,
)
from pypsds.unwrap.statistical_cost import write_cost_file


def test_no_synthetic_bridge_across_empty_grid_cells():
    grid = np.array([[0, 1, -1, -1, 2, 3],
                     [4, 5, -1, -1, 6, 7]], dtype=np.int32)
    out = build_occupied_spatial_edges(grid)
    assert set(map(tuple, out['edge_nodes'])) == {
        (0,1),(0,4),(1,5),(4,5),
        (2,3),(2,6),(3,7),(6,7),
    }
    assert int(out['row_eid'][0,2]) == -2
    assert int(out['col_eid'][0,1]) == -2
    assert int(out['col_eid'][0,3]) == -2
    assert int(np.sum(out['edge_occurrences'])) == 8
    p = summarize_spatial_topology(grid)
    assert p['component_count'] == 2
    assert not p['global_integer_gauge_identifiable_from_occupied_edges']


def test_occupied_edges_identical_to_dense_when_every_cell_occupied():
    from pypsds.unwrap.statistical_cost import build_stamps_interp_edges
    grid = np.arange(12, dtype=np.int32).reshape(3, 4)
    a = build_occupied_spatial_edges(grid)
    b = build_stamps_interp_edges(grid)
    for key in ('edge_nodes','edge_occurrences','row_eid','row_sign','col_eid','col_sign'):
        np.testing.assert_array_equal(a[key],b[key])
    p = summarize_spatial_topology(grid, grid)
    assert p['component_count'] == 1
    assert p['unsupported_nearest_fill_interfaces'] == 0


def test_gap_interfaces_are_not_statistical_constraints(tmp_path):
    grid = np.array([[0, -1, 1],[2, -1, 3]], dtype=np.int32)
    edge = build_occupied_spatial_edges(grid)
    # Each observed column is one disconnected two-node component.
    sig = np.ones(len(edge['edge_nodes']), dtype=np.int16)
    off = np.zeros((len(sig), 1), dtype=np.int16)
    bad = np.zeros(len(sig), dtype=bool)
    path = tmp_path / 'cost.bin'
    write_cost_file(path, 0, edge['row_eid'],edge['row_sign'],
                    edge['col_eid'],edge['col_sign'],sig,off,bad)
    data = np.fromfile(path, dtype=np.int16)
    row_n = edge['row_eid'].size
    rows = data[:row_n*4].reshape(edge['row_eid'].shape+(4,))
    cols = data[row_n*4:].reshape(edge['col_eid'].shape+(4,))
    assert np.all(rows[...,3][edge['row_eid'] == -2] == 1)
    assert np.all(cols[...,3][edge['col_eid'] == -2] == 1)
    assert np.all(rows[...,3][edge['row_eid'] >= 0] == -32000)


def test_sparse_isolated_node_not_assigned_global_gauge():
    grid = np.array([[0,-1,-1],[-1,-1,-1],[-1,-1,1]],dtype=np.int32)
    p = summarize_spatial_topology(grid)
    assert p['component_count'] == 2
    assert p['nodes_outside_dominant_component'] == 1
    assert p['occupied_grid_edges'] == 0
    with pytest.raises(RuntimeError, match='No physically occupied'):
        build_occupied_spatial_edges(grid)


def test_reject_duplicate_node_ids():
    with pytest.raises(ValueError, match='unique'):
        summarize_spatial_topology(np.array([[0,0]],dtype=np.int32))


def test_nearest_filled_interfaces_detected():
    grid = np.array([[0,-1,-1,1]],dtype=np.int32)
    _, ix = ndimage.distance_transform_edt(grid<0,return_indices=True)
    nearest = grid[ix[0],ix[1]]
    p = summarize_spatial_topology(grid,nearest)
    assert p['unsupported_nearest_fill_interfaces'] >= 1
    assert p['occupied_grid_edges'] == 0
