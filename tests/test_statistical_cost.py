import numpy as np

from pypsds.unwrap.statistical_cost import (
    build_stamps_interp_edges,
    build_temporal_operator,
    build_statistical_edge_model,
    write_cost_file,
)


def test_interp_edges_count_and_orientation():
    # The same node-to-node link may occur at multiple dense-grid boundaries.
    g = np.array([[0, 0, 1], [0, 0, 1], [2, 2, 3]], dtype=np.int32)
    m = build_stamps_interp_edges(g)
    assert m['edge_nodes'].shape[1] == 2
    assert np.all(m['edge_nodes'][:, 0] < m['edge_nodes'][:, 1])
    assert int(np.sum(m['edge_occurrences'])) == int(np.sum(m['row_eid']>=0)+np.sum(m['col_eid']>=0))
    assert np.all(m['row_sign'][m['row_eid'] >= 0] != 0)


def test_statistical_model_zero_noise_and_snaphu_cost(tmp_path):
    dates=['20240101','20240113','20240125','20240206']
    temporal_edges=[(0,1),(1,2),(2,3),(0,2),(1,3)]
    _,t,_=build_temporal_operator(temporal_edges,dates)
    g=np.array([[0,1],[2,3]],dtype=np.int32)
    m=build_stamps_interp_edges(g)
    ph=np.array([[0,0.1,0.2,0.3],[0,0.2,0.4,0.6],[0,0.3,0.6,0.9],[0,0.4,0.8,1.2]],dtype=np.float32)
    paths=build_statistical_edge_model(node_phase=ph,temporal_edges=temporal_edges,temporal_operator=t,spatial_edge_nodes=m['edge_nodes'],edge_occurrences=m['edge_occurrences'],outdir=tmp_path,edge_batch=2,force=True)
    sig, off, bad=(np.load(x) for x in paths)
    assert not np.any(bad)
    assert np.all(sig>=1)
    assert np.max(np.abs(off))<=200
    cost=tmp_path/'cost.bin'
    write_cost_file(cost,0,m['row_eid'],m['row_sign'],m['col_eid'],m['col_sign'],sig,off,bad)
    expected=(m['row_eid'].size+m['col_eid'].size)*4*2
    assert cost.stat().st_size==expected


def test_temporal_operator_ignores_global_phase_and_detects_disconnected():
    dates=['20240101','20240113','20240125','20240206']
    edges=[(0,1),(1,2),(2,3),(0,2),(1,3)]
    g,t,_=build_temporal_operator(edges,dates)
    assert g.shape==(5,4)
    assert t.shape==(5,5)
    assert np.all(np.isfinite(t))
    assert np.linalg.matrix_rank(g[:, 1:]) == 3
    import pytest
    with pytest.raises(RuntimeError, match='rank deficient'):
        build_temporal_operator([(0,1),(2,3)],dates)


def test_integer_branch_spatial_qa_detects_boundary_without_calling_it_error():
    from pypsds.unwrap.statistical_cost import summarize_spatial_integer_gradients
    dates = ['20240101', '20250101', '20260101']
    grid = np.array([[0, 1], [2, 3]], dtype=np.int32)
    cycles = np.array([[0, 0, 0], [0, 1, 2], [0, 0, 0], [0, 1, 2]], dtype=np.int16)
    result = summarize_spatial_integer_gradients(grid, cycles, dates)
    assert result['edge_count'] == 4
    assert result['integer_gradient_cycles_per_year_p50_p90_p95_p99'][3] > 0.9
    assert 0 < result['fraction_exceeding_0p2_cycles_per_year'] < 1
