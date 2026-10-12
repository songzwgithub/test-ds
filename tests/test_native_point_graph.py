import json
from pathlib import Path

import numpy as np
import pytest

from pypsds.unwrap.native_point_graph import (
    build_observed_point_edges, load_tile_points, point_edge_integer_observations,
    solve_native_tile, main, wrap,
)


def fixture():
    # Four raw radar points with 8 acquisitions. Neighbor gradients grow by
    # < pi between dates even though total gradients exceed 2pi.
    y = np.array([0,0,1,1],np.int32)
    x = np.array([0,1,0,1],np.int32)
    t = np.arange(16,dtype=float)
    signal=np.array([0*t,0.52*t,0.36*t,0.88*t])
    wrapped=wrap(signal).astype(np.float32)
    return y,x,signal,wrapped


def test_native_edge_generation_references_only_actual_points():
    rows=np.array([0,0,5,5],np.int32)
    cols=np.array([0,1,5,6],np.int32)
    e,d=build_observed_point_edges(rows=rows,cols=cols,sy=1,sx=1,
         neighbors=3,max_distance_m=1.5,workers=1)
    np.testing.assert_array_equal(e,[[0,1],[2,3]])
    np.testing.assert_allclose(d,[1,1])


def test_no_synthetic_bridge_when_distance_exceeds_limit():
    rows=np.array([0,0,20,20],np.int32)
    cols=np.array([0,1,20,21],np.int32)
    edges,_=build_observed_point_edges(rows=rows,cols=cols,sy=1,sx=1,
         max_distance_m=2,neighbors=3,workers=1)
    assert not np.any((edges[:,0]<2)&(edges[:,1]>=2))


def test_integer_histories_recover_known_gradient_without_cell_average():
    y,x,true,ph=fixture()
    edges,d=build_observed_point_edges(rows=y,cols=x,sy=1,sx=1,
        neighbors=3,max_distance_m=1.5,workers=1)
    cycles,good,step,_=point_edge_integer_observations(ph,edges)
    assert good.all()
    # The unknown phase gradients are recovered exactly (up to float roundoff)
    phase_unwrapped=ph.astype(float)
    known_cycles=np.rint((true-ph)/ (2*np.pi)).astype(np.int16)
    np.testing.assert_array_equal(cycles,known_cycles[edges[:,1]]-known_cycles[edges[:,0]])
    assert np.all(step<2.65)


def test_two_independent_components_keep_arbitrary_roots():
    t=np.arange(7)
    raw=np.array([0*t,0.4*t,0*t,0.55*t],float)
    p=wrap(raw).astype('float32')
    edges=np.array([[0,1],[2,3]],np.int32)
    res=solve_native_tile(point_phase=p,edges=edges,
        distance_m=np.ones(2,np.float32),holdout_fraction=0.0)
    assert res['report']['training_components']==2
    assert res['report']['nodes_two_core_and_forest_agreement']==0
    assert np.all(res['cycles_primary'][[0,2]]==0)


def test_holdout_closure_and_two_core_with_loop():
    y,x,true,ph=fixture()
    e,d=build_observed_point_edges(rows=y,cols=x,sy=1,sx=1,
        max_distance_m=1.5,neighbors=3,workers=1)
    out=solve_native_tile(point_phase=ph,edges=e,distance_m=d,
         holdout_fraction=.15,random_seed=3)
    report=out['report']
    assert report['native_points']==4
    assert report['temporal_accepted_edges']==len(e)
    assert report['primary_training_edge_closure']['fraction_mismatched_edge_epochs']==0
    assert report['primary_heldout_edge_closure']['fraction_mismatched_edge_epochs']==0
    assert report['empty_pixel_interpolation'] is False
    assert report['coarse_node_cycles_broadcast_to_points'] is False


def test_reject_temporal_jump_near_pi():
    p=np.zeros((2,5),np.float32)
    p[1]=[0,0.1,3.0,0.1,0.2]
    k,good,s,_=point_edge_integer_observations(p,np.array([[0,1]]),max_step_rad=2.65)
    assert not good[0]
    assert s[0]>2.65


def test_roi_input_uses_original_ids_and_types(tmp_path):
    pp=tmp_path/'pps';pp.mkdir()
    np.save(pp/'rows.npy',np.array([0,0,1,1,20,20]))
    np.save(pp/'cols.npy',np.array([0,1,0,1,20,21]))
    np.save(pp/'point_type.npy',np.array([1,2,1,2,2,2],np.uint8))
    np.save(pp/'phase_rad.npy',np.zeros((6,5),np.float32))
    ids,rows,cols,typ,ph=load_tile_points(root=pp,row0=0,col0=0,
        height=3,width=3,subset='combined')
    assert ids.tolist()==[0,1,2,3]
    assert ph.shape==(4,5)
    assert set(typ)=={1,2}
    with pytest.raises(RuntimeError,match='max-points'):
        load_tile_points(root=pp,row0=0,col0=0,height=3,width=3,max_points=3)
    ids,*_=load_tile_points(root=pp,row0=0,col0=0,height=40,width=40,subset='ps')
    assert ids.tolist()==[0,2]
    # Duplicate radar coordinate is rejected rather than merged.
    np.save(pp/'cols.npy',np.array([0,0,0,1,20,21]))
    with pytest.raises(ValueError,match='same original radar pixel'):
        load_tile_points(root=pp,row0=0,col0=0,height=3,width=3)


def test_real_cli_fixture_and_outputs_isolation(tmp_path):
    pp=tmp_path/'pps';pp.mkdir()
    y,x,_,ph=fixture()
    np.save(pp/'rows.npy',y)
    np.save(pp/'cols.npy',x)
    np.save(pp/'point_type.npy',np.array([1,2,2,2],np.uint8))
    np.save(pp/'phase_rad.npy',ph)
    dest=tmp_path/'diagnostics/native'
    assert main(['--point-phase-stack',str(pp), '--row-spacing-m','1',
           '--col-spacing-m','1', '--row0','0','--col0','0','--height','2',
           '--width','2','--max-distance-m','1.5','--neighbors','3',
           '--output-dir',str(dest)])==0
    out=json.loads((dest/'summary.json').read_text())
    assert out['status']=='LOCAL_CANDIDATE_ONLY_NO_PRODUCTION_WRITE'
    assert out['source']==str(pp)
    assert not out['production_modified']
    assert len(np.load(dest/'original_point_ids.npy'))==4
    assert (dest/'local_relative_unwrapped_phase_rad.npy').is_file()
    with pytest.raises(FileExistsError):
        main(['--point-phase-stack',str(pp),'--row-spacing-m','1',
         '--col-spacing-m','1','--row0','0','--col0','0','--height','2',
         '--width','2','--output-dir',str(dest)])
    assert np.load(pp/'phase_rad.npy').shape==(4,16)


def test_optional_baseline_gradient_and_parity():
    from pypsds.unwrap.native_point_graph import baseline_edge_comparison
    y,x,true,ph=fixture()
    edges,d=build_observed_point_edges(rows=y,cols=x,sy=1,sx=1,
        neighbors=3,max_distance_m=1.5,workers=1)
    solution=solve_native_tile(point_phase=ph,edges=edges,distance_m=d,holdout_fraction=0)
    ok=solution['candidate_point_edge_accepted']
    zeros=np.zeros(len(edges),bool)
    out=baseline_edge_comparison(wrapped_point_phase=ph,
      baseline_point_phase=true,edge_pairs=edges,
      edge_integer=solution['candidate_point_edge_cycles'],
      component_id=solution['component_id'],accepted_mask=ok,heldout_mask=zeros)
    assert out['training_edge_closure']['fraction_mismatched_edge_epochs']==0
    wrong=true.copy();wrong[2,5:]+=2*np.pi
    out=baseline_edge_comparison(wrapped_point_phase=ph,
      baseline_point_phase=wrong,edge_pairs=edges,
      edge_integer=solution['candidate_point_edge_cycles'],
      component_id=solution['component_id'],accepted_mask=ok,heldout_mask=zeros)
    assert out['training_edge_closure']['fraction_mismatched_edge_epochs']>0
    with pytest.raises(ValueError,match='parity'):
        baseline_edge_comparison(wrapped_point_phase=ph,
          baseline_point_phase=true+.5,edge_pairs=edges,
          edge_integer=solution['candidate_point_edge_cycles'],
          component_id=solution['component_id'],accepted_mask=ok,heldout_mask=zeros)
