import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location('native_emcf_testmodule', BASE/'pypsds/unwrap/native_emcf_backend.py')
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def grid(n=5, gap=1):
    y,x=np.mgrid[0:n,0:n]
    return y.ravel()*gap,x.ravel()*gap


def test_uniform_grid_one_connected_island():
    r,c=grid(5)
    islands,info=mod.build_triangle_islands(r,c,1,1,2)
    assert len(islands)==1
    z=islands[0]
    assert z.npoints==25 and z.solver_eligible
    assert z.nlinks-z.npoints+1==z.ntriangles
    assert z.max_edge_m<=2
    assert info['unused_points']==0


def test_separated_regions_not_bridged():
    a,b=grid(4)
    r=np.r_[a,a+100]
    c=np.r_[b,b+100]
    islands,info=mod.build_triangle_islands(r,c,1,1,2)
    assert len(islands)==2
    assert all(z.npoints==16 and z.solver_eligible for z in islands)
    assert info['unused_points']==0
    assert info['rejected_triangle_fraction']>0


def test_threshold_rejects_all_large_triangles():
    r,c=grid(3,gap=10)
    isl,info=mod.build_triangle_islands(r,c,1,1,5)
    assert not isl and info['unused_points']==9


def test_consistent_positive_triangle_orientation():
    r,c=grid(6)
    z=mod.build_triangle_islands(r,c,1,1,1.5)[0][0]
    xy=z.points_xy_m; t=z.triangles
    v=xy[t[:,1]]-xy[t[:,0]]
    w=xy[t[:,2]]-xy[t[:,0]]
    assert np.all(v[:,0]*w[:,1]-v[:,1]*w[:,0]>0)


def test_graph_exposes_protocol_arrays():
    r,c=grid(4)
    z=mod.build_triangle_islands(r,c,1,1,2)[0][0]
    g=mod.NativePlanarGraph(z)
    assert g.points.shape==(16,2) and g.links.shape[1]==2
    assert g.cycles.shape[1]==3
    assert np.all(g.links[:,0]<g.links[:,1])


def test_output_safety(tmp_path):
    proc=tmp_path/'output/processing/point_phase_stack'
    proc.mkdir(parents=True)
    with pytest.raises(ValueError): mod._outdir_is_safe(proc/'x',proc)
    with pytest.raises(ValueError): mod._outdir_is_safe(proc.parent/'other',proc)
    new=tmp_path/'output/diagnostics/new'
    assert mod._outdir_is_safe(new,proc)==new
    new.mkdir(parents=True)
    (new/'any').write_text('existing')
    with pytest.raises(FileExistsError): mod._outdir_is_safe(new,proc)


def stub_spurt(monkeypatch, bad=False):
    # Smoke-check the *adapter contract* without claiming to test Spurt MCF.
    names=('spurt','spurt.graph','spurt.mcf','spurt.io','spurt.workflows',
           'spurt.workflows.emcf','spurt.workflows.emcf._solver',
           'spurt.workflows.emcf._settings')
    x={key:types.ModuleType(key) for key in names}
    for k,v in x.items():
        if k.count('.')<3: v.__path__=[]
        monkeypatch.setitem(sys.modules,k,v)
    class Hop3Graph:
        def __init__(self,n):
            self.links=np.column_stack((np.arange(n-1),np.arange(1,n)))
            self.cycles=np.empty((0,3),dtype=np.int32)
    class ORMCFSolver:
        def __init__(self,graph):self.graph=graph
        @property
        def edges(self):return self.graph.links
        @property
        def cycles(self):return self.graph.cycles
    class Irreg3DInput:
        def __init__(self,data,xy,time_dim):
            assert time_dim==0 and data.shape[1]==len(xy)
            self.data=data
    class SolverSettings:
        def __init__(self,**kwargs):self.kwargs=kwargs
    class EMCFSolver:
        def __init__(self,spatial,temporal,settings):self.temporal=temporal
        def unwrap_cube(self,inp):
            ph=inp.data
            pairs=self.temporal.edges
            phase=mod.wrap(ph[pairs[:,1]]-ph[pairs[:,0]])
            relative=mod.wrap(phase-phase[:,0,None])
            if bad:relative[0,1]+=0.5
            return relative
    x['spurt.graph'].Hop3Graph=Hop3Graph
    x['spurt.mcf'].ORMCFSolver=ORMCFSolver
    x['spurt.io'].Irreg3DInput=Irreg3DInput
    x['spurt.workflows.emcf._solver'].EMCFSolver=EMCFSolver
    x['spurt.workflows.emcf._settings'].SolverSettings=SolverSettings


def test_spurt_adapter_time_orientation_and_parity(monkeypatch):
    stub_spurt(monkeypatch)
    r,c=grid(3)
    z=mod.build_triangle_islands(r,c,1,1,1.5)[0][0]
    ph=np.linspace(-0.8,0.8,z.npoints)[:,None]+np.arange(5)[None,:]*0.06
    out=mod.run_spurt_emcf(ph,z)
    assert out['unwrapped_ifg_phase_rad'].shape==(4,9)
    assert out['wrapped_phase_parity_max_rad']<1e-5
    assert out['time_ifg_pairs_0based'].shape==(4,2)


def test_spurt_adapter_rejects_failed_congruence(monkeypatch):
    stub_spurt(monkeypatch,bad=True)
    r,c=grid(3)
    z=mod.build_triangle_islands(r,c,1,1,1.5)[0][0]
    ph=np.zeros((z.npoints,6))
    with pytest.raises(RuntimeError,match='not congruent'):
        mod.run_spurt_emcf(ph,z)


def test_refuse_bad_shapes_and_sizes():
    r,c=grid(4)
    z=mod.build_triangle_islands(r,c,1,1,2)[0][0]
    with pytest.raises(ValueError):mod.run_spurt_emcf(np.zeros((16,2)),z)
    with pytest.raises(ValueError):mod.run_spurt_emcf(np.zeros((15,75)),z)
    with pytest.raises(ValueError):mod.build_triangle_islands(r,c,1,1,-1)


def test_no_synthetic_edges_in_multiple_islands():
    a,b=grid(3)
    r=np.r_[a,a+1000];c=np.r_[b,b+1000]
    z,info=mod.build_triangle_islands(r,c,1,1,1.5)
    assert len(z)==2
    for isl in z:
        assert isl.max_edge_m<=1.5
        assert isl.solver_eligible
