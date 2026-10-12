"""Tests of PS/DS multi-resolution support and phase aggregation."""
import json
import sys
import types

import numpy as np
import pytest

from pypsds.unwrap.psds_resolution_audit import (
    _grid_spec, _topology, _circular_phase_audit, audit_resolutions, main,
)


def synthetic():
    rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 6, 6, 6, 6], np.int32)
    cols = np.array([0, 0, 0, 1, 0, 0, 1, 1, 6, 6, 6, 6], np.int32)
    typ  = np.array([1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2],np.uint8)
    t0   = np.array([0., 0., 0., 2., 0., 0., 1., 0., .3, 0., 1., 1.])
    t1   = np.array([0., 0., 2., 2., 1., 0., 2., 0., .4, 2., 0., 1.])
    phase=np.stack((t0,t1),axis=1).astype(np.float32)
    return rows,cols,typ,phase


def test_coarse_and_fine_cell_occupancy_and_groups():
    r,c,t,phase=synthetic()
    report=audit_resolutions(rows=r,cols=c,point_type=t,phase=phase,
        date_i=0,date_j=1,row_spacing_m=100,col_spacing_m=100,
        resolutions=(100.,200.,700.),batch=3)
    assert report['status']=='DIAGNOSTIC_ONLY_NO_UNWRAPPING'
    assert len(report['results'])==3
    for x in report['results']:
        assert x['ps_point_count']==4
        assert x['ds_point_count']==8
        assert x['source_point_count']==12
        assert x['groups']['combined']['occupied_cells'] >= x['groups']['ps_only']['occupied_cells']
        assert x['groups']['combined']['occupied_cells'] >= x['groups']['ds_only']['occupied_cells']
    assert report['results'][0]['groups']['combined']['component_count']==2
    assert report['results'][0]['groups']['combined']['occupied_cells']==5
    assert report['results'][2]['groups']['combined']['occupied_cells']==1
    assert report['results'][2]['groups']['combined']['component_count']==1


def test_circular_aggregation_is_not_commutative_for_three_or_more_points():
    # We intentionally use one cell with a spatially varying phase gradient.
    r=np.array([0,0,0]);c=np.array([0,0,0]);t=np.array([2,2,2],np.uint8)
    phase=np.array([[0,0],[0,2],[2,2]],np.float32)
    a=audit_resolutions(rows=r,cols=c,point_type=t,phase=phase,date_i=0,date_j=1,
        row_spacing_m=1,col_spacing_m=1,resolutions=(50,),minimum_points=3)
    stats=a['results'][0]['phase_aggregation_audit']['ds_only']
    assert stats['cells_tested']==1
    assert stats['aggregation_discrepancy_rad_p50_p90_p95_p99'][0] > .1
    assert stats['point_ifg_resultant_p50_p90_p95_p99'][0] < 1.0


def test_two_acquisitions_same_reference_phase_yields_zero_aggregation_difference():
    r=np.zeros(4,np.int32);c=np.zeros(4,np.int32)
    ph=np.stack((np.zeros(4),np.array([0.2,1.3,-2.,3.])),axis=1).astype(np.float32)
    d=audit_resolutions(rows=r,cols=c,point_type=np.full(4,2,np.uint8),phase=ph,
        date_i=0,date_j=1,row_spacing_m=1,col_spacing_m=1,resolutions=(50,))
    st=d['results'][0]['phase_aggregation_audit']['ds_only']
    assert st['aggregation_discrepancy_rad_p50_p90_p95_p99'][0] < 1e-5


def test_no_phase_only_counts_points_without_unwrap():
    r,c,t,_=synthetic()
    d=audit_resolutions(rows=r,cols=c,point_type=t,phase=None,date_i=0,date_j=1,
        row_spacing_m=100,col_spacing_m=100,resolutions=(50,100,200),with_phase=False)
    assert not d['phase_aggregation_checked']
    assert all('phase_aggregation_audit' not in x for x in d['results'])


def test_topology_only_4_connectivity_not_diagonal():
    m=np.array([[1,0],[0,1]],bool)
    q=_topology(m)
    assert q['component_count']==2
    assert q['singleton_components']==2
    assert q['observed_four_neighbor_edges']==0


def test_reject_non_finite_spacing_duplicate_resolution_and_invalid_point_type():
    r,c,t,ph=synthetic()
    with pytest.raises(ValueError,match='positive'):
        _grid_spec(r,c,float('nan'),10,50)
    with pytest.raises(ValueError,match='unique'):
        audit_resolutions(rows=r,cols=c,point_type=t,phase=None,date_i=0,date_j=1,
        row_spacing_m=1,col_spacing_m=1,resolutions=(50,50),with_phase=False)
    t[0]=4
    with pytest.raises(ValueError,match='Point types'):
        audit_resolutions(rows=r,cols=c,point_type=t,phase=ph,date_i=0,date_j=1,
        row_spacing_m=1,col_spacing_m=1)


def test_end_to_end_cli_uses_point_phase_stack_and_writes_only_diagnostics(tmp_path,monkeypatch):
    r,c,t,phase=synthetic()
    root=tmp_path/'output'
    pp=root/'processing'/'point_phase_stack';pp.mkdir(parents=True)
    net=root/'processing'/'network';net.mkdir()
    for name, arr in (('rows',r),('cols',c),('point_type',t),('phase_rad',phase)):
        np.save(pp/(name+'.npy'),arr)
    (net/'network.itab').write_text('1 2\n')
    par=tmp_path/'reference.rslc.par';par.write_text('azimuth_pixel_spacing: 100\nrange_pixel_spacing: 100\n')
    fake_ctx=types.ModuleType('pypsds.context')
    fake_ctx.open_from_config=lambda conf:({},None,types.SimpleNamespace(output_dir=root),types.SimpleNamespace(dates=['20240101','20240113']),None)
    fake_input=types.ModuleType('pypsds.geometry.inputs')
    fake_input.resolve_geometry_inputs=lambda cfg,paths:types.SimpleNamespace(reference_rslc_par=par)
    fake_geo=types.ModuleType('pypsds.gamma.geometry')
    fake_geo.geometry_from_par=lambda path:types.SimpleNamespace(azimuth_spacing_m=100.,ground_range_spacing_m=100.)
    monkeypatch.setitem(sys.modules,'pypsds.context',fake_ctx)
    monkeypatch.setitem(sys.modules,'pypsds.geometry.inputs',fake_input)
    monkeypatch.setitem(sys.modules,'pypsds.gamma.geometry',fake_geo)
    dest=tmp_path/'diagnostics'
    assert main(['--config',str(tmp_path/'pypsds.yaml'),'--grid-sizes-m','100','200',
        '--output-dir',str(dest),'--batch','3'])==0
    summary=json.loads((dest/'summary.json').read_text())
    assert summary['production_modified'] is False
    assert summary['phase_aggregation_checked']
    assert summary['results'][0]['groups']['combined']['occupied_cells']==5
    assert list(pp.glob('*.npy')) and not (root/'processing'/'stamps3d_unwrap').exists()


def test_existing_200m_grid_must_match_observed_points():
    r,c,t,phase=synthetic()
    ref=np.full((4,4),-1,np.int32)
    with pytest.raises(RuntimeError,match='differ from the existing'):
        audit_resolutions(rows=r,cols=c,point_type=t,phase=None,date_i=0,date_j=1,
            row_spacing_m=100,col_spacing_m=100,resolutions=(200,),with_phase=False,
            reference_grid=ref)
    ref=np.full((4,4),-1,np.int32)
    ref[0,0]=0;ref[3,3]=1
    a=audit_resolutions(rows=r,cols=c,point_type=t,phase=None,date_i=0,date_j=1,
        row_spacing_m=100,col_spacing_m=100,resolutions=(200,),with_phase=False,
        reference_grid=ref)
    assert a['results'][0]['groups']['combined']['occupied_cells']==2
