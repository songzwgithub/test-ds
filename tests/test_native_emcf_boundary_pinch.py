"""Half-edge regression tests for pinched native PS/DS boundaries."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE=Path(__file__).parents[1]/'pypsds/unwrap/native_emcf_backend.py'
SPEC=importlib.util.spec_from_file_location('pinch_boundary_testmodule',SOURCE)
mod=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=mod
SPEC.loader.exec_module(mod)


def make_annulus(n=10):
    rr,cc=np.mgrid[0:n,0:n]
    r,c=rr.ravel(),cc.ravel()
    keep=~((r>=3)&(r<=6)&(c>=3)&(c<=6))
    islands,_=mod.build_triangle_islands(r[keep],c[keep],1,1,1.5)
    assert len(islands)==1
    return islands[0],None

def _attach_vertex_only_triangle(island):
    """Append real triangle sharing exactly one corner of an existing mesh.

    The planar outside face becomes a pinched walk (repeated vertex), even
    though edges still belong to one/two triangles and a true void persists.
    """
    from dataclasses import replace
    xy = np.asarray(island.points_xy_m)
    anchor = np.flatnonzero(np.all(np.isclose(xy, [0., 0.]),axis=1))
    assert len(anchor)==1
    a = int(anchor[0]); v = len(xy)
    xy_new=np.concatenate((xy, np.array([[-1.,0.],[0.,-1.]],np.float64)))
    # Counterclockwise outer triangle: (0,0)->(-1,0)->(0,-1)
    tri_new=np.vstack((island.triangles,np.array([[a,v,v+1]],np.int32)))
    links,count=mod._unique_edges_with_counts(tri_new,len(xy_new))
    boundary=links[count==1]
    return replace(island, original_indices=np.arange(len(xy_new)),
                   points_xy_m=xy_new,triangles=tri_new,links=links,
                   boundary=boundary,manifold=True,simply_connected=False)


def test_pinch_on_outer_boundary_with_genuine_hole():
    island,_=make_annulus()
    pinched=_attach_vertex_only_triangle(island)
    # A standard boundary vertex->successor map fails here: degree 4.
    deg=np.bincount(pinched.boundary.ravel(),minlength=pinched.npoints)
    assert np.max(deg)==4
    assert pinched.nlinks-pinched.npoints+1-pinched.ntriangles==1
    hole=mod.build_hole_face_cycles(pinched)
    assert len(hole)==1
    g=mod.NativePlanarGraph(pinched,enable_hole_faces=True)
    assert len(g.cycles)==pinched.ntriangles+1
    assert g.npoints==pinched.npoints


def test_pinched_exterior_without_hole_has_no_added_face():
    rr,cc=np.mgrid[0:5,0:5]
    island=mod.build_triangle_islands(rr.ravel(),cc.ravel(),1,1,1.5)[0][0]
    pinched=_attach_vertex_only_triangle(island)
    # A pinch may exist even with zero Euler missing bounded faces.
    assert pinched.nlinks-pinched.npoints+1-pinched.ntriangles==0
    assert mod.build_hole_face_cycles(pinched)==[]
    g=mod.NativePlanarGraph(pinched,enable_hole_faces=True)
    assert len(g.cycles)==pinched.ntriangles


def test_pinch_hole_only_existing_edges_are_used():
    island,_=make_annulus()
    pinched=_attach_vertex_only_triangle(island)
    faces=mod.NativePlanarGraph(pinched,enable_hole_faces=True).cycles
    known={tuple(edge) for edge in pinched.links}
    for face in faces:
        for u,v in zip(face,np.roll(face,-1)):
            assert tuple(sorted((int(u),int(v)))) in known
