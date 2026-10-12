import json
import numpy as np
import pytest
from pypsds.unwrap.point_solver_edge_audit import assess,grid_edges,main


def fixture():
    grid = np.array([[0,1,-1,2], [3,4,-1,5]],np.int32)
    phase = np.zeros((6,3),np.float32)
    labels = np.ones(6,np.uint8)
    return grid,phase,labels


def test_real_edges_not_nearest_fill():
    g,_,_=fixture()
    edge, coords = grid_edges(g)
    assert set(map(tuple,edge)) == {(0,1),(0,3),(1,4),(3,4),(2,5)}
    assert coords.shape == (5,3)


def test_zero_snaphu_labels_do_not_hide_integer_edge_disagreement():
    g,ph,lab=fixture()
    w=np.zeros(6,np.float32)
    s=np.array([0,1,0,0,0,0],np.float32)*2*np.pi
    report,arr=assess(node_phase=ph,node_grid=g,index_i=0,index_j=1,
      whirlwind_uw=w,snaphu_uw=s,whirlwind_cc=lab,snaphu_cc=lab*0)
    assert report['snaphu_labels']['labeled_nodes'] == 0
    stats=report['comparisons']['whirlwind_minus_snaphu']
    assert stats['all_directly_observed_edges']['edges'] == 5
    assert stats['all_directly_observed_edges']['fraction_integer_gradient_different'] == 2/5
    assert stats['within_both_labeled_components']['fraction_integer_gradient_different'] is None
    assert np.count_nonzero(arr['edge_gradient_difference_whirlwind_minus_snaphu']) == 2


def test_uniform_integer_gauge_per_component_cancels_on_edges():
    g,ph,lab=fixture()
    ww=np.zeros(6,np.float32)
    ss=np.array([4,4,-9,4,4,-9],np.float32)*(2*np.pi)
    report,_=assess(node_phase=ph,node_grid=g,index_i=0,index_j=1,
      whirlwind_uw=ww,snaphu_uw=ss,whirlwind_cc=lab,snaphu_cc=lab)
    assert report['comparisons']['whirlwind_minus_snaphu']['all_directly_observed_edges']['fraction_integer_gradient_different'] == 0


def test_statcost_comparison_identifies_bad_branch():
    g,ph,lab=fixture()
    ww=np.zeros(6,np.float32)
    ss=np.zeros(6,np.float32)
    base=np.zeros((6,3),np.float32)
    base[1,1]=2*np.pi
    r,_=assess(node_phase=ph,node_grid=g,index_i=0,index_j=1,
      whirlwind_uw=ww,snaphu_uw=ss,whirlwind_cc=lab,snaphu_cc=lab,
      baseline_phase=base)
    assert r['comparisons']['whirlwind_minus_statcost']['all_directly_observed_edges']['fraction_integer_gradient_different'] == 2/5


def test_wrong_input_parity_fails():
    g,ph,lab=fixture()
    with pytest.raises(ValueError,match='parity mismatch'):
      assess(node_phase=ph,node_grid=g,index_i=0,index_j=1,
       whirlwind_uw=np.full(6,0.5,np.float32),snaphu_uw=np.zeros(6,np.float32),
       whirlwind_cc=lab,snaphu_cc=lab)


def test_wrong_grid_rejected():
    g,_,_=fixture();g[0,1]=0
    with pytest.raises(ValueError,match='exactly once'):
      grid_edges(g)


def test_cli_read_only(tmp_path):
    g,ph,lab=fixture()
    node=tmp_path/'node.npy';grid=tmp_path/'grid.npy'
    np.save(node,ph);np.save(grid,g)
    d=tmp_path/'bench';d.mkdir()
    (d/'input_audit.json').write_text(json.dumps({'acquisition_index_0based':[0,1]}))
    np.save(d/'whirlwind_node_ifg_unwrapped_rad.npy',np.zeros(6,np.float32))
    np.save(d/'snaphu_node_ifg_unwrapped_rad.npy',np.array([0,1,0,0,0,0],np.float32)*2*np.pi)
    np.save(d/'whirlwind_node_conncomp.npy',lab)
    np.save(d/'snaphu_node_conncomp.npy',lab*0)
    out=tmp_path/'out'
    assert main(['--node-phase',str(node),'--node-grid',str(grid),
                 '--benchmark-dir',str(d),'--output-dir',str(out)]) == 0
    report=json.loads((out/'summary.json').read_text())
    assert report['existing_production_modified'] is False
    assert (out/'integer_disagreement_edges.csv').is_file()
    assert (out/'direct_edge_node_pairs.npy').is_file()
