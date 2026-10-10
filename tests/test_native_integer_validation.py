import json
from pathlib import Path

import numpy as np
import pytest

from pypsds.unwrap.native_integer_validation import (
    _linear_phase_rate, evaluate, load_baseline_cycles, main,
)


def fixture_data():
    # Cycle-supported 4-node ring, true difference histories defined by nodes.
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int32)
    true = np.array([[0, 0, 0, 0], [0, 1, 1, 2],
                     [0, 1, 2, 2], [0, 0, 1, 1]], np.int32)
    edges = true[pairs[:, 1]]-true[pairs[:, 0]]
    return pairs, true, edges


def evaluate_small(base=None):
    pairs, true, edges = fixture_data()
    if base is None:
        base = true.copy()
        base[2, 2:] += 1
    return evaluate(
        node_phase=np.zeros((4,4),np.float32),baseline_cycles=base,
        primary=true,secondary=true,component_id=np.zeros(4,np.int32),
        component_roots=np.array([0]),core_mask=np.ones(4,bool),
        agreement_fraction=np.ones(4),node_pairs=pairs,edge_integer=edges,
        tree_edge_primary=np.array([1,1,1,0],bool),
        tree_edge_secondary=np.array([0,1,1,1],bool),
        dates=['20240101','20240113','20240125','20240206'],
    )


def test_native_correct_baseline_wrong_and_no_output_phase_writes():
    report, aux = evaluate_small()
    assert report['accepted_native_edges']==4
    assert report['comparisons']['baseline_statcost']['all_accepted_edges']['fraction_mismatched_edge_epochs'] > 0
    assert report['comparisons']['native_primary']['all_accepted_edges']['fraction_mismatched_edge_epochs'] == 0
    assert report['relative_comparison']['fraction_edges_baseline_worse_than_native_primary'] == .5
    assert report['root_aligned_node_difference']['fraction_nodes_changed_any_acquisition'] == .25
    assert np.count_nonzero(aux['node_root_aligned_difference_fraction']) == 1
    assert report['status']=='DIAGNOSTIC_ONLY_NOT_PRODUCTION'


def test_component_gauge_offsets_are_exactly_invisible():
    _, true, _ = fixture_data()
    gauge = np.array([23,-9,17,4],np.int32)
    report, _ = evaluate_small(base=true+gauge)
    assert report['comparisons']['baseline_statcost']['all_accepted_edges']['fraction_mismatched_edge_epochs'] == 0
    assert report['root_aligned_node_difference']['fraction_nodes_changed_any_acquisition'] == 0


def test_componentwise_independent_gauges():
    n=5; nt=3
    edges=np.array([[0,1],[2,3]],np.int32)
    k=np.array([[0,1,1],[0,0,-1]],np.int32)
    primary=np.array([[0,0,0],[0,1,1],[0,0,0],[0,0,-1],[0,0,0]],np.int32)
    base=primary.copy()
    base[[0,1]]+=np.array([13,2,1],np.int32)
    base[[2,3]]+=np.array([-2,4,8],np.int32)
    result,_=evaluate(node_phase=np.zeros((n,nt),np.float32),baseline_cycles=base,
        primary=primary,secondary=primary,component_id=np.array([0,0,1,1,2]),
        component_roots=np.array([0,2,4]),core_mask=np.zeros(n,bool),
        agreement_fraction=np.ones(n),node_pairs=edges,edge_integer=k,
        tree_edge_primary=np.ones(2,bool),tree_edge_secondary=np.ones(2,bool),
        dates=['20240101','20240113','20240125'])
    assert result['root_aligned_node_difference']['fraction_nodes_changed_any_acquisition']==0
    assert result['comparisons']['baseline_statcost']['all_accepted_edges']['fraction_mismatched_edge_epochs']==0
    assert result['comparisons']['native_primary']['non_tree_edges_primary']['edges']==0


def test_baseline_phase_order_parity_check(tmp_path):
    g=np.array([[0,0.4],[-0.2,0.1]],np.float32)
    cycles=np.array([[0,1],[0,-1]],np.int32)
    b=tmp_path/'baseline';b.mkdir()
    np.save(b/'coarse_acquisition_phase_unwrapped_rad.npy',g+2*np.pi*cycles)
    actual,info=load_baseline_cycles(b,g)
    np.testing.assert_array_equal(actual,cycles)
    assert info['node_order_verification']=='wrapped_phase_parity'
    np.save(b/'coarse_acquisition_phase_unwrapped_rad.npy',g+0.4)
    with pytest.raises(ValueError,match='not wrapped-congruent'):
        load_baseline_cycles(b,g)


def test_cycle_only_requires_node_order_evidence(tmp_path):
    b=tmp_path/'baseline';b.mkdir()
    cycles=np.array([[0,1],[0,0]],np.int16)
    np.save(b/'node_acquisition_integer_cycles.npy',cycles)
    g=np.zeros((2,2),np.float32)
    with pytest.raises(ValueError,match='node ordering'):
        load_baseline_cycles(b,g)
    np.save(b/'coarse_node_grid.npy',np.array([[0,1]],np.int32))
    gp=tmp_path/'grid.npy'
    np.save(gp,np.array([[0,1]],np.int32))
    out,meta=load_baseline_cycles(b,g,node_grid_path=gp)
    np.testing.assert_array_equal(out,cycles)
    assert meta['node_order_verification']=='identical_grid'
    np.save(gp,np.array([[1,0]],np.int32))
    with pytest.raises(ValueError,match='do not match'):
        load_baseline_cycles(b,g,node_grid_path=gp)


def test_reject_cross_component_edges():
    pairs,true,edge=fixture_data()
    with pytest.raises(ValueError,match='different solution components'):
        evaluate(node_phase=np.zeros((4,4)),baseline_cycles=true,
            primary=true,secondary=true,component_id=np.array([0,0,1,1]),
            component_roots=np.array([0,2]),core_mask=np.ones(4,bool),
            agreement_fraction=np.ones(4),node_pairs=pairs,edge_integer=edge,
            tree_edge_primary=np.array([1,1,1,0],bool),
            tree_edge_secondary=np.array([0,1,1,1],bool),
            dates=['20240101','20240113','20240125','20240206'])


def test_rate_uses_wrapped_plus_integer_cycles():
    dates=['20240101','20250101','20260101']
    x=np.zeros((2,3),np.float32)
    cycles=np.array([[0,0,0],[0,1,2]],np.int32)
    slopes=_linear_phase_rate(x,cycles,dates,4.413824900882934)
    assert slopes[1]-slopes[0] == pytest.approx(2*np.pi*4.413824900882934,rel=.01)


def test_cli_end_to_end_writes_only_diagnostics(tmp_path):
    pair,true,edges=fixture_data()
    baseline=tmp_path/'baseline';baseline.mkdir()
    cand=tmp_path/'cand';cand.mkdir()
    net=tmp_path/'net';net.mkdir()
    nodephase=tmp_path/'node.npy';np.save(nodephase,np.zeros_like(true,dtype=np.float32))
    corrupted=true.copy();corrupted[2,2:]+=1
    np.save(baseline/'coarse_acquisition_phase_unwrapped_rad.npy',2*np.pi*corrupted)
    def save(directory,**entries):
        for name,x in entries.items():np.save(directory/(name+'.npy'),x)
    save(cand,cycles_primary=true,cycles_secondary=true,
        component_id=np.zeros(4,np.int32),component_root_node=np.array([0]),
        node_two_core_mask=np.ones(4,bool),node_forest_agreement_fraction=np.ones(4),
        candidate_edge_integer_histories=edges,
        tree_edge_mask_primary=np.array([1,1,1,0],bool),
        tree_edge_mask_secondary=np.array([0,1,1,1],bool),
        retained_network_edge_indices=np.arange(4))
    save(net,node_pairs=pair,directly_adjacent=np.ones(4,bool),supported_edge_mask=np.ones(4,bool))
    dates=tmp_path/'dates.txt';dates.write_text('20240101\n20240113\n20240125\n20240206\n')
    dest=tmp_path/'out'
    assert main(['--node-phase',str(nodephase),'--baseline-dir',str(baseline),
                 '--candidate-dir',str(cand),'--network-dir',str(net),
                 '--dates',str(dates),'--output-dir',str(dest)])==0
    report=json.loads((dest/'summary.json').read_text())
    assert report['relative_comparison']['fraction_edges_baseline_worse_than_native_primary']==.5
    assert (dest/'largest_components.csv').exists()
    assert (dest/'edge_baseline_mismatch_fraction.npy').is_file()
    assert (baseline/'coarse_acquisition_phase_unwrapped_rad.npy').is_file()
