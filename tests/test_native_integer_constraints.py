import numpy as np
import pytest
from pypsds.unwrap.native_integer_constraints import (
    native_integer_edge_observations, synchronize_integer_graph, wrap_phase, node_two_core_mask,
)


def model(nnode=4, ndate=8):
    # Measurable smooth inter-node phase gradients; integer time ambiguity
    # generated from a known synthetic physical history.
    t=np.arange(ndate,dtype=float)
    true=np.array([0*t,0.42*t,0.21*t,0.63*t])
    wrapped=wrap_phase(true).copy()
    # Three separate PS points per grid node; exact repeated signals.
    point=np.repeat(wrapped,3,axis=0)
    pairs=np.array([[0,1],[0,2],[1,3],[2,3]],dtype=np.int32)
    native=np.array([[[a*3+j,b*3+j] for j in range(3)] for a,b in pairs],dtype=np.int32)
    return true,wrapped,point,pairs,native


def test_exact_integer_circulation_on_closed_synthetic_network():
    true, wrapped, point, pairs, native=model()
    idx,k,ok,dev,step=native_integer_edge_observations(
        phase=point,node_phase=wrapped,node_pairs=pairs,
        native_point_pairs=native,use_edges=np.ones(len(pairs),bool))
    assert np.all(ok)
    assert np.array_equal(idx,np.arange(4))
    out=synchronize_integer_graph(nnode=4,node_pairs=pairs,
        integer_edge_histories=k,score_primary=np.array([4,3,2,1]),
        score_secondary=np.array([1,3,2,4]))
    for name in ('cycles_primary','cycles_secondary'):
        expected=np.rint((true-wrapped)/(2*np.pi)).astype(np.int32)
        np.testing.assert_array_equal(out[name],expected)
    np.testing.assert_array_equal(out['component_root_node'],[0])
    assert not np.any(out['edge_conflict_fraction_primary'])
    assert not np.any(out['edge_conflict_fraction_secondary'])


def test_components_stay_independent():
    pairs=np.array([[0,1],[2,3]],dtype=np.int32)
    k=np.array([[0,1,1],[0,-1,-2]],dtype=np.int16)
    out=synchronize_integer_graph(nnode=5,node_pairs=pairs,integer_edge_histories=k,
        score_primary=np.ones(2),score_secondary=np.ones(2))
    assert len(out['component_root_node'])==3
    np.testing.assert_array_equal(out['cycles_primary'],[[0,0,0],[0,1,1],[0,0,0],[0,-1,-2],[0,0,0]])


def test_no_cross_component_bridge_even_when_close():
    _,wrapped,point,pairs,native=model()
    mask=np.array([1,0,0,0],dtype=bool)
    idx,k,ok,*_=native_integer_edge_observations(phase=point,node_phase=wrapped,
        node_pairs=pairs,native_point_pairs=native,use_edges=mask)
    assert idx.tolist()==[0]
    out=synchronize_integer_graph(nnode=4,node_pairs=pairs[idx],integer_edge_histories=k,
        score_primary=np.ones(1),score_secondary=np.ones(1))
    assert len(out['component_root_node'])==3


def test_reject_unstable_temporal_spatial_gradient():
    true,wrapped,point,pairs,native=model()
    # A large wrapped increment, dangerously close to pi, is not accepted.
    point[3:6,3:]+=2.94
    point=wrap_phase(point)
    idx,k,ok,*_=native_integer_edge_observations(phase=point,node_phase=wrapped,
        node_pairs=pairs,native_point_pairs=native,use_edges=np.ones(4,bool))
    assert not ok[0]


def test_exact_cycle_conflict_is_detected():
    pairs=np.array([[0,1],[0,2],[1,3],[2,3]],dtype=np.int32)
    k=np.array([[0,1],[0,0],[0,0],[0,0]],dtype=np.int16)
    out=synchronize_integer_graph(nnode=4,node_pairs=pairs,
        integer_edge_histories=k,score_primary=np.array([10,9,8,7]),
        score_secondary=np.array([7,8,9,10]))
    assert np.any(out['edge_conflict_fraction_primary']>0)
    assert np.any(out['edge_conflict_fraction_secondary']>0)


def test_invalid_temporal_gate():
    _,wrapped,point,pairs,native=model()
    with pytest.raises(ValueError,match='strictly'):
        native_integer_edge_observations(phase=point,node_phase=wrapped,
            node_pairs=pairs,native_point_pairs=native,use_edges=np.ones(4,bool),
            max_temporal_step_rad=np.pi)


def test_two_core_excludes_dangling_nodes_in_cyclic_component():
    # Four-node square plus dangling tail; only square is redundantly connected.
    pairs=np.array([[0,1],[0,2],[1,3],[2,3],[3,4],[4,5]],dtype=np.int32)
    np.testing.assert_array_equal(node_two_core_mask(6,pairs),
        [True,True,True,True,False,False])


def test_cli_synthetic_end_to_end(tmp_path, monkeypatch):
    import sys
    import types
    import json
    from pypsds.unwrap.native_integer_constraints import main
    _, wrapped, point, pairs, native = model()
    pp=tmp_path/'output/processing/point_phase_stack'
    co=tmp_path/'output/processing/stamps3d_unwrap'
    nw=tmp_path/'network'
    for d in (pp,co,nw): d.mkdir(parents=True)
    np.save(pp/'phase_rad.npy',point.astype('float32'))
    np.save(co/'coarse_complex_mean_phase_rad.npy',wrapped.astype('float32'))
    np.save(nw/'node_pairs.npy',pairs)
    np.save(nw/'native_point_pairs.npy',native)
    np.save(nw/'supported_edge_mask.npy',np.ones(len(pairs),bool))
    np.save(nw/'directly_adjacent.npy',np.ones(len(pairs),bool))
    np.save(nw/'native_pair_distances_m.npy',np.ones((len(pairs),3),np.float32)*100)
    np.save(nw/'native_pair_circular_consistency.npy',np.ones(len(pairs),np.float32))
    fake=types.ModuleType('pypsds.context')
    def opening(_):
        return {},'',types.SimpleNamespace(output_dir=tmp_path/'output'),types.SimpleNamespace(dates=[str(i) for i in range(8)]),None
    fake.open_from_config=opening
    monkeypatch.setitem(sys.modules,'pypsds.context',fake)
    dest=tmp_path/'candidates'
    assert main(['--config',str(tmp_path/'project.yaml'),'--network-dir',str(nw),'--output-dir',str(dest)])==0
    with (dest/'summary.json').open() as f: report=json.load(f)
    assert report['status']=='CANDIDATE_ONLY_NOT_PRODUCTION'
    assert report['connected_components_after_quality']==1
    assert report['nodes_with_two_core_forest_agreement']==4
    assert report['non_tree_edge_fraction_conflicting_any_epoch']==0.0
    np.testing.assert_array_equal(np.load(dest/'cycles_primary.npy'),np.load(dest/'cycles_secondary.npy'))
