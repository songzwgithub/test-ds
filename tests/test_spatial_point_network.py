import csv
import numpy as np
import pytest
from pypsds.unwrap.spatial_point_network import (
    build_candidate_node_edges, build_native_edge_support,
    evaluate_native_pair_phase,
)


def make_model(grid, r, c, typ=None, q=None, **overrides):
    n = len(r)
    args = dict(node_grid=np.asarray(grid, np.int32),
                point_rows=np.asarray(r, np.int32),
                point_cols=np.asarray(c, np.int32),
                point_type=np.asarray(typ if typ is not None else [2]*n, np.uint8),
                point_quality=np.asarray(q if q is not None else [0.95]*n, np.float32),
                row_spacing_m=1., col_spacing_m=1.,
                grid_size_m=20., side_window_m=15.,
                max_point_distance_m=30., anchors_per_side=4,
                pairs_per_edge=3)
    args.update(overrides)
    return build_native_edge_support(**args)


def test_native_multiple_pairs_across_observed_boundary():
    # The rows/cols lie on either side of the 20 m grid boundary, not identical cells.
    r=[2,2,5,8,2,5,8]
    c=[0,17,18,19,20,21,22]
    out=make_model([[0,1]],r,c)
    assert out['node_pairs'].tolist()==[[0,1]]
    assert out['directly_adjacent'].tolist()==[True]
    assert out['independent_pair_count'].tolist()==[3]
    pairs=out['native_point_pairs'][0]
    assert len(set(pairs[:,0]))==3
    assert len(set(pairs[:,1]))==3
    assert np.all(out['native_pair_distances_m']<30)


def test_one_point_per_cell_does_not_fake_redundancy():
    out=make_model([[0,1]], [2,2,2],[0,18,21])
    assert out['independent_pair_count'].tolist()==[1]
    assert np.all(out['native_point_pairs'][0,1:]==-1)


def test_wide_gap_does_not_create_native_observation(tmp_path):
    g=np.array([[0,-1,1]],dtype=np.int32)
    path=tmp_path/'bridge.csv'
    with path.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['node_a_0based','node_b_0based'])
        w.writerow([0,1])
    out=make_model(g,[5,5,5,5,5,5],[1,2,3,46,47,48],
                   bridge_csv=path,max_point_distance_m=25)
    assert not out['directly_adjacent'][0]
    assert out['independent_pair_count'].tolist()==[0]


def test_wrap_consistent_multiple_observations():
    phase=np.zeros((6,7),np.float32)
    t=np.arange(7)*0.16
    phase[3:]=t+0.25
    pairs=np.array([[[0,3],[1,4],[2,5]]],np.int32)
    concentration,count=evaluate_native_pair_phase(point_phase=phase,native_point_pairs=pairs)
    assert count.tolist()==[3]
    assert concentration[0]>0.9999


def test_disagreeing_native_observations_are_not_high_confidence():
    phase=np.zeros((6,7),np.float32)
    phase[3,:]=0.0
    phase[4,:]=2*np.pi/3
    phase[5,:]=4*np.pi/3
    pairs=np.array([[[0,3],[1,4],[2,5]]],np.int32)
    concentration,_=evaluate_native_pair_phase(point_phase=phase,native_point_pairs=pairs)
    assert concentration[0]<0.001


def test_candidate_edges_reject_duplicate_grid_ids():
    with pytest.raises(ValueError, match='unique'):
        build_candidate_node_edges(np.array([[0,0]],np.int32))


def test_command_line_synthetic_project(tmp_path, monkeypatch):
    """Exercise the actual CLI with small on-disk PointPhaseStack inputs."""
    import json
    import sys
    import types
    from pypsds.unwrap.spatial_point_network import main

    root = tmp_path/'output'/'processing'
    pointdir = root/'point_phase_stack'
    nodedir = root/'stamps3d_unwrap'
    pointdir.mkdir(parents=True)
    nodedir.mkdir(parents=True)
    r = np.array([0,0,2,4,0,2,4], np.int32)
    c = np.array([0,17,18,19,20,21,22], np.int32)
    np.save(pointdir/'rows.npy',r)
    np.save(pointdir/'cols.npy',c)
    np.save(pointdir/'point_type.npy',np.full(7,2,np.uint8))
    np.save(pointdir/'temporal_coherence.npy',np.full(7,0.92,np.float32))
    phase = np.zeros((7,4), np.float32)
    phase[4:] = 0.2*np.arange(4)[None,:]
    np.save(pointdir/'phase_rad.npy',phase)
    np.save(nodedir/'coarse_node_grid.npy',np.array([[0,1]],np.int32))

    fake_context = types.ModuleType('pypsds.context')
    fake_context.open_from_config = lambda cfg: (None, cfg,
        types.SimpleNamespace(output_dir=tmp_path/'output'),
        types.SimpleNamespace(dates=['20240101','20240113','20240125','20240206']),None)
    fake_inputs = types.ModuleType('pypsds.geometry.inputs')
    fake_inputs.resolve_geometry_inputs=lambda *_: types.SimpleNamespace(reference_rslc_par='fake.par')
    fake_gamma = types.ModuleType('pypsds.gamma.geometry')
    fake_gamma.geometry_from_par=lambda *_: types.SimpleNamespace(azimuth_spacing_m=1.,ground_range_spacing_m=1.)
    for name,obj in [('pypsds.context',fake_context),
                     ('pypsds.geometry.inputs',fake_inputs),
                     ('pypsds.gamma.geometry',fake_gamma)]:
        monkeypatch.setitem(sys.modules,name,obj)
    output = tmp_path/'diagnostics'
    assert main(['--config',str(tmp_path/'fake.yaml'),
                 '--grid-size-m','20','--max-point-distance-m','30',
                 '--side-window-m','15', '--output-dir',str(output)]) == 0
    summary=json.loads((output/'summary.json').read_text())
    assert summary['phase_modified'] is False
    assert summary['relative_integer_gauge_solved'] is False
    assert summary['edges_with_phase_consistent_native_pairs']==1
    assert np.load(output/'native_point_pairs.npy').shape==(1,3,2)
