"""Regression for bounded void faces without synthetic bridge edges.

These tests exercise the genuine Delaunay and planar topology; actual Spurt/OR
Tools integration must be run in a separate environment on the server.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE=Path(__file__).parents[1]/'pypsds/unwrap/native_emcf_backend.py'
SPEC=importlib.util.spec_from_file_location('hole_face_native_testmodule',SOURCE)
mod=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=mod
SPEC.loader.exec_module(mod)


def make_annulus(n=10, holes=((3,6,3,6),)):
    rr,cc=np.mgrid[0:n,0:n]
    r,c=rr.ravel(),cc.ravel()
    keep=np.ones(len(r),dtype=bool)
    for lo,hi,llo,hhi in holes:
        keep &= ~((r>=lo)&(r<=hi)&(c>=llo)&(c<=hhi))
    islands, report=mod.build_triangle_islands(r[keep],c[keep],1,1,1.5)
    assert len(islands)==1
    return islands[0],report


def test_real_void_is_excluded_but_can_form_bounded_face():
    isl,_=make_annulus()
    assert isl.manifold and not isl.solver_eligible
    assert isl.nlinks-isl.npoints+1-isl.ntriangles == 1
    with pytest.raises(ValueError):
        mod.NativePlanarGraph(isl)
    face=mod.build_hole_face_cycles(isl)
    assert len(face)==1
    assert len(face[0])>=4
    graph=mod.NativePlanarGraph(isl,enable_hole_faces=True)
    assert len(graph.cycles)==isl.ntriangles+1
    assert np.array_equal(graph.links,isl.links)
    assert graph.npoints==isl.npoints


def test_two_holes_are_two_independent_faces():
    isl,_=make_annulus(19,((4,6,4,7),(11,13,11,14)))
    assert isl.nlinks-isl.npoints+1-isl.ntriangles==2
    g=mod.NativePlanarGraph(isl,enable_hole_faces=True)
    assert len(g.hole_faces)==2
    assert len(g.cycles)==isl.ntriangles+2


def test_hole_faces_oriented_opposite_to_mesh_on_shared_edges():
    isl,_=make_annulus()
    g=mod.NativePlanarGraph(isl,enable_hole_faces=True)
    face_edges={}
    for face in g.cycles:
        for u,v in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(u),int(v))))
            face_edges.setdefault(key,[]).append(1 if u<v else -1)
    assert len(face_edges)==isl.nlinks
    assert all(len(signs) in (1,2) for signs in face_edges.values())
    assert all(sum(signs)==0 for signs in face_edges.values() if len(signs)==2)


def test_all_cycle_edges_are_actual_original_links():
    isl,_=make_annulus(19,((4,6,4,7),(11,13,11,14)))
    graph=mod.NativePlanarGraph(isl,enable_hole_faces=True)
    linkset={tuple(edge) for edge in isl.links}
    for cyc in graph.cycles:
        for u,v in zip(cyc,np.roll(cyc,-1)):
            assert tuple(sorted((int(u),int(v)))) in linkset


def test_mesh_with_no_hole_preserves_existing_solver_contract():
    rr,cc=np.mgrid[0:5,0:5]
    isl=mod.build_triangle_islands(rr.ravel(),cc.ravel(),1,1,1.5)[0][0]
    g=mod.NativePlanarGraph(isl,enable_hole_faces=True)
    assert isinstance(g.cycles,np.ndarray)
    assert len(g.hole_faces)==0


def test_non_manifold_flag_causes_hard_refusal():
    from dataclasses import replace
    isl,_=make_annulus()
    invalid=replace(isl,manifold=False)
    with pytest.raises(ValueError,match='Non-manifold'):
        mod.NativePlanarGraph(invalid,enable_hole_faces=True)


def test_wrong_euler_characteristic_causes_refusal():
    from dataclasses import replace
    isl,_=make_annulus()
    invalid=replace(isl,links=isl.links[:-1])
    with pytest.raises(ValueError):
        mod.build_hole_face_cycles(invalid)


def test_solver_requires_explicit_hole_opt_in():
    isl,_=make_annulus()
    with pytest.raises(ValueError,match='hole'):
        mod.run_spurt_emcf(np.zeros((isl.npoints,5),np.float32),isl)
