"""ROI-only PS/DS point phase unwrapping using Spurt's real EMCF solver.

No coarse raster, no nearest-neighbor filling, no coarse cycle broadcast.
Delaunay triangles are screened by ACTUAL observed-point edge lengths; rejected
triangles do not create graph constraints. Disconnected triangulated islands
are never merged. This is not a final referenced acquisition time series.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

TAU = 2.0 * np.pi


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


@dataclass
class Island:
    """One connected planar triangle complex with a local node index."""
    original_indices: np.ndarray
    points_xy_m: np.ndarray
    triangles: np.ndarray
    links: np.ndarray
    boundary: np.ndarray
    max_edge_m: float
    min_area_m2: float
    manifold: bool
    simply_connected: bool

    @property
    def npoints(self):
        return int(len(self.original_indices))

    @property
    def ntriangles(self):
        return int(len(self.triangles))

    @property
    def nlinks(self):
        return int(len(self.links))

    @property
    def solver_eligible(self):
        return bool(self.manifold and self.simply_connected and self.ntriangles > 0)


def _edge_triplets(triangles):
    t = np.asarray(triangles, dtype=np.int32)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError('triangles must have shape [n,3]')
    return np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))


def _unique_edges_with_counts(triangles, npoint):
    e = np.sort(_edge_triplets(triangles), axis=1)
    key = e[:, 0].astype(np.int64) * int(npoint) + e[:, 1]
    keys, counts = np.unique(key, return_counts=True)
    u = keys // int(npoint)
    v = keys % int(npoint)
    return np.column_stack((u, v)).astype(np.int32), counts


def build_triangle_islands(rows, cols, sy, sx, max_edge_m=40.0,
                           min_area_m2=1e-4):
    """Build connected islands of short-edge planar Delaunay triangles.

    Prevents long, synthetic Delaunay constraints spanning masked holes.
    Leaves smaller islands separate. No artificial edges or pixels are created.
    """
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)
    if r.ndim != 1 or c.shape != r.shape or len(r) < 3:
        raise ValueError('Need >=3 aligned native radar point coordinates')
    if not all(np.isfinite(x) and x > 0 for x in (sy, sx, max_edge_m, min_area_m2)):
        raise ValueError('Finite positive spacing, radius, and min area required')
    xy = np.column_stack(((c - c.min()) * sx, (r - r.min()) * sy))
    if not np.isfinite(xy).all():
        raise ValueError('Nonfinite point coordinates')
    if np.unique(xy, axis=0).shape[0] != len(xy):
        raise ValueError('Duplicate radar point coordinates')
    try:
        triang = Delaunay(xy)
    except QhullError as exc:
        raise ValueError('Cannot triangulate original point positions') from exc
    tri = np.asarray(triang.simplices, dtype=np.int32)
    # Ensure consistent orientation of cycles, required by ORMCFSolver.
    ab = xy[tri[:, 1]] - xy[tri[:, 0]]
    ac = xy[tri[:, 2]] - xy[tri[:, 0]]
    twice_area = ab[:, 0]*ac[:, 1] - ab[:, 1]*ac[:, 0]
    swap = twice_area < 0
    tri[swap, 1], tri[swap, 2] = tri[swap, 2].copy(), tri[swap, 1].copy()
    ea = xy[tri[:, [0, 1, 2]]]
    edge_lengths = np.stack((
        np.linalg.norm(ea[:, 1] - ea[:, 0], axis=1),
        np.linalg.norm(ea[:, 2] - ea[:, 1], axis=1),
        np.linalg.norm(ea[:, 0] - ea[:, 2], axis=1)), axis=1)
    keep = (edge_lengths.max(axis=1) <= max_edge_m + 1e-8) & (
        np.abs(twice_area) / 2 >= min_area_m2)
    short_tri = tri[keep]
    if not len(short_tri):
        return [], {'input_points': len(xy), 'delaunay_triangles': len(tri),
                    'accepted_triangles': 0, 'unused_points': len(xy),
                    'islands': 0}
    e = _edge_triplets(short_tri)
    graph = coo_matrix((np.ones(2*len(e), dtype=np.uint8),
                        (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                       shape=(len(xy),len(xy))).tocsr()
    ncomp, labels = connected_components(graph, directed=False)
    tri_labels = labels[short_tri[:, 0]]
    if not np.array_equal(tri_labels, labels[short_tri[:, 1]]) or not np.array_equal(tri_labels, labels[short_tri[:, 2]]):
        raise RuntimeError('Triangular island graph construction failed')
    used=np.unique(short_tri)
    comps, freqs = np.unique(labels[used], return_counts=True)
    order = np.lexsort((comps, -freqs))
    islands=[]
    for comp in comps[order]:
        orig = used[labels[used]==comp]
        local = np.full(len(xy), -1, dtype=np.int32)
        local[orig] = np.arange(len(orig),dtype=np.int32)
        ti = local[short_tri[tri_labels==comp]]
        links, counts = _unique_edges_with_counts(ti, len(orig))
        if len(links) == 0:
            continue
        boundary = links[counts==1]
        edge_lengths_loc=np.linalg.norm(xy[orig[links[:, 0]]] - xy[orig[links[:, 1]]], axis=1)
        a=xy[orig[ti[:, 1]]] - xy[orig[ti[:, 0]]]
        b=xy[orig[ti[:, 2]]] - xy[orig[ti[:, 0]]]
        area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
        # One planar, connected, hole-free triangulated island:
        # graph cycle rank E-V+1 equals number of bounded triangle faces.
        hole_free=(len(links)-len(orig)+1 == len(ti))
        islands.append(Island(orig,xy[orig],ti,links,boundary,
                              float(edge_lengths_loc.max()), float(area.min()),
                              bool(np.all((counts==1)|(counts==2))),bool(hole_free)))
    report={'input_points':int(len(xy)), 'delaunay_triangles':int(len(tri)),
            'accepted_triangles':int(len(short_tri)),
            'unused_points':int(len(xy)-len(used)),
            'rejected_triangle_fraction':float(1.0-len(short_tri)/len(tri)),
            'islands':len(islands),
            'solver_eligible_islands':int(sum(t.solver_eligible for t in islands)),
            'islands_first_20':[{'rank':i,'points':z.npoints,'triangles':z.ntriangles,
                                 'edges':z.nlinks,'holes_or_topology_fail':not z.solver_eligible,
                                 'solver_eligible':z.solver_eligible,
                                 'max_edge_m':z.max_edge_m} for i,z in enumerate(islands[:20])],
            'max_edge_m':float(max_edge_m),
            'note':('Only real native PS/DS vertices and triangles whose every edge is '
                    'within the maximum range. Distance is nominal radar geometry, '
                    'not a verified deformation-continuity criterion.')}
    return islands,report


def build_hole_face_cycles(island: Island):
    """Extract missing bounded planar faces using a half-edge rotation system.

    Original Delaunay triangles are consistently counterclockwise. All links
    are existing observed-point links. Each boundary half-edge, reversed from
    its incident triangle, has a missing face on its LEFT. Follow these missing
    half-edges by taking the first CLOCKWISE neighbor of the reverse ray among
    ALL real graph links. Unlike a vertex->vertex successor dictionary, this
    handles a boundary that touches itself at an original vertex (a pinch).

    A face boundary may be a closed walk with repeated *vertices*, but never
    repeat a directed boundary half-edge. We add only positive-area bounded
    faces; the unique negative-area outer face remains the MCF grounding face.
    No synthetic spatial edges or interpolated points are introduced.
    """
    if not island.manifold:
        raise ValueError('Non-manifold triangle edges: hole-face augmentation forbidden')
    tri=np.asarray(island.triangles,dtype=np.int32)
    xy=np.asarray(island.points_xy_m,dtype=np.float64)
    n=island.npoints
    edges=np.asarray(island.links,dtype=np.int32)
    if len(xy)!=n or edges.ndim!=2 or edges.shape[1]!=2:
        raise ValueError('Invalid point coordinates or graph edges')
    # Verify precisely that the input graph is the 1-skeleton of the triangles.
    incident={}
    for a,b,c in tri:
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
            key=(min(u,v),max(u,v))
            incident.setdefault(key,[]).append((u,v))
    linkset={tuple(map(int,pair)) for pair in edges}
    if set(incident)!=linkset:
        raise ValueError('Spatial graph links differ from triangle edges')
    if any(len(inc) not in (1,2) or
           (len(inc)==2 and inc[0] != (inc[1][1],inc[1][0]))
           for inc in incident.values()):
        raise ValueError('Invalid triangle orientations or edge incidences')
    # Only boundary arcs have a missing face. Orient those with missing face left.
    missing={(inc[0][1],inc[0][0]) for inc in incident.values() if len(inc)==1}
    if len(missing)!=len(island.boundary):
        raise ValueError('Boundary incidence mismatch')
    # Geometry-derived clockwise rotation order for each existing graph vertex.
    nbrs=[[] for _ in range(n)]
    for u,v in edges:
        u=int(u);v=int(v)
        nbrs[u].append(v);nbrs[v].append(u)
    clockwise_next={}
    for v, neigh in enumerate(nbrs):
        if not neigh:
            continue
        dd=xy[np.asarray(neigh)]-xy[v]
        angles=np.arctan2(dd[:,1],dd[:,0])
        order=np.argsort(angles,kind='mergesort')
        sorted_angles=angles[order]
        if len(order)>1 and (np.any(np.diff(sorted_angles)<1e-12) or
                             sorted_angles[0]+2*np.pi-sorted_angles[-1]<1e-12):
            raise ValueError('Ambiguous collinear incident edges in planar embedding')
        ordered=np.asarray(neigh,dtype=np.int32)[order]
        for k,u in enumerate(ordered):
            clockwise_next[(v,int(u))]=int(ordered[k-1])
    # A directed missing-face edge u->v is followed by v->w, where the
    # ray v->w is immediately clockwise of the reverse ray v->u.
    successor={}
    for u,v in missing:
        w=clockwise_next[(v,u)]
        next_halfedge=(v,w)
        if next_halfedge not in missing:
            raise ValueError('Boundary halfedge traversal escapes missing faces')
        successor[(u,v)]=next_halfedge
    if len(set(successor.values()))!=len(successor):
        raise ValueError('Boundary halfedge rotation is not a permutation')
    unseen=set(missing)
    walks=[]
    while unseen:
        first=min(unseen)
        curr=first
        walk=[]
        while curr in unseen:
            unseen.remove(curr)
            walk.append(curr[0])
            curr=successor[curr]
        if curr!=first or len(walk)<3:
            raise ValueError('Invalid closed boundary face walk')
        walks.append(np.asarray(walk,dtype=np.int32))
    if sum(map(len,walks))!=len(missing):
        raise ValueError('Not every boundary halfedge belongs to exactly one face')
    # On the LEFT of each reversed boundary edge is a missing face. The unique
    # unbounded face walks clockwise (negative area); genuine holes CCW.
    signed_areas=[]
    for walk in walks:
        pos=xy[walk]
        area=0.5*np.sum(pos[:,0]*np.roll(pos[:,1],-1)-
                        np.roll(pos[:,0],-1)*pos[:,1],dtype=np.float64)
        if not np.isfinite(area) or abs(area)<1e-6:
            raise ValueError('Degenerate boundary face walk')
        signed_areas.append(float(area))
    exterior=[k for k,area in enumerate(signed_areas) if area<0]
    hole_ids=[k for k,area in enumerate(signed_areas) if area>0]
    expected_holes=island.nlinks-island.npoints+1-island.ntriangles
    if len(exterior)!=1 or len(hole_ids)!=expected_holes:
        raise ValueError('Face count/orientation incompatible with planar embedding: '
                         f'exterior={len(exterior)}, bounded={len(hole_ids)}, '
                         f'Euler bounded faces={expected_holes}')
    hole_faces=[walks[k] for k in hole_ids]
    # Validate that each real link borders 1 or 2 positively oriented bounded
    # faces and never 3. Repeated pinch vertices are fine, repeated edges not.
    directions={key:[] for key in incident}
    for face in [*tri,*hole_faces]:
        for u,v in zip(face,np.roll(face,-1)):
            key=(min(int(u),int(v)),max(int(u),int(v)))
            if key not in directions:
                raise RuntimeError('A bounded face introduced a synthetic link')
            directions[key].append(1 if u<v else -1)
    if any(len(signs) not in (1,2) or
           (len(signs)==2 and sum(signs)!=0)
           for signs in directions.values()):
        raise ValueError('Invalid bounded-face incidence or orientation')
    if sum(len(signs)==1 for signs in directions.values())!=len(walks[exterior[0]]):
        raise ValueError('External grounding-face incidence failed')
    return hole_faces


class NativePlanarGraph:
    """Spurt-compatible planar graph, optionally with true void-face cycles.

    On hole-enabled graphs, `cycles` is a list of variable-length polygon faces.
    Only ORMCFSolver.compute_residues_from_gradients is supported, as used by
    Spurt EMCFSolver's spatial step. Do not call compute_residues(point_phase):
    its implementation assumes constant-length triangular cycles.
    """
    def __init__(self, island: Island, *, enable_hole_faces=False):
        if not island.manifold:
            raise ValueError('Non-manifold triangulated island')
        if not island.solver_eligible and not enable_hole_faces:
            raise ValueError('Island has holes; use --enable-hole-faces only after audit')
        self._island=island
        self.hole_faces=build_hole_face_cycles(island) if enable_hole_faces else []
        self._cycles=([*island.triangles,*self.hole_faces]
                      if self.hole_faces else island.triangles)

    @property
    def npoints(self):
        return self._island.npoints

    @property
    def points(self):
        return self._island.points_xy_m

    @property
    def cycles(self):
        return self._cycles

    @property
    def links(self):
        return self._island.links

    @property
    def boundary(self):
        return self._island.boundary


def temporal_triangle_closure(phase, links, triangles):
    """Measure 2pi time-loop inconsistencies; this is not an accuracy proof."""
    data=np.asarray(phase, dtype=np.float64)
    edges=np.asarray(links, dtype=np.int32)
    triangles=np.asarray(triangles, dtype=np.int32)
    lookup={tuple(pair):i for i,pair in enumerate(edges.tolist())}
    bad=0;total=0;maxcycle=0
    for abc in triangles:
        loop=np.zeros(data.shape[1],dtype=np.float64)
        for ii in range(3):
            a=int(abc[ii]);b=int(abc[(ii+1)%3])
            key=(min(a,b),max(a,b))
            ind=lookup[key]
            loop += (1 if a<b else -1)*data[ind]
        cyc=np.rint(loop/TAU).astype(np.int64)
        bad+=int(np.count_nonzero(cyc))
        total+=len(cyc)
        maxcycle=max(maxcycle,int(np.max(np.abs(cyc))))
    return {'tested_triangle_point_combinations':total,
            'nonzero_integer_closure_fraction':float(bad/total) if total else None,
            'max_abs_integer_closure':maxcycle}


def run_spurt_emcf(point_wrapped_phase, island, *, workers=2, links_per_batch=1024,
                   time_cost_type='distance', spatial_cost_type='distance',
                   enable_hole_faces=False):
    """Call actual Spurt EMCF; return aligned, per-IFG phase (NOT point time series).

    Standard Hop3Graph time network; no ad-hoc time branch integration.
    """
    if not island.manifold or (not island.solver_eligible and not enable_hole_faces):
        raise ValueError('Unfit spatial topology: use hole-face option only for manifold holes')
    if workers<1 or links_per_batch<1:
        raise ValueError('workers and links_per_batch must be positive')
    if enable_hole_faces and spatial_cost_type != 'distance':
        raise ValueError('Hole-aware dual faces are supported only with spatial distance costs')
    ph=np.asarray(point_wrapped_phase, dtype=np.float32)
    if ph.shape[0]!=island.npoints or ph.ndim!=2 or ph.shape[1]<4:
        raise ValueError('Point phase must be [island point,>=4 acquisitions]')
    if not np.isfinite(ph).all():
        raise ValueError('Nonfinite wrapped phase')
    # Deferred import allows geometry audit without installing Spurt/OR-Tools.
    try:
        from spurt.graph import Hop3Graph
        from spurt.mcf import ORMCFSolver
        from spurt.io import Irreg3DInput
        from spurt.workflows.emcf._solver import EMCFSolver
        from spurt.workflows.emcf._settings import SolverSettings
    except ImportError as exc:
        raise RuntimeError('Spurt is not installed. Install in a separate environment; see docs.') from exc
    space_graph=NativePlanarGraph(island,enable_hole_faces=enable_hole_faces)
    spatial=ORMCFSolver(space_graph)
    temporal=ORMCFSolver(Hop3Graph(ph.shape[1]))
    settings=SolverSettings(t_worker_count=workers,s_worker_count=int(__import__('os').environ.get('NINGBO_EMCF_SPATIAL_WORKERS', '3')),
                            links_per_batch=links_per_batch,
                            t_cost_type=time_cost_type,s_cost_type=spatial_cost_type,
                            num_parallel_tiles=1)
    solver=EMCFSolver(spatial,temporal,settings)
    phase_input=Irreg3DInput(np.ascontiguousarray(ph.T),
                             np.asarray(island.points_xy_m,np.float64),time_dim=0)
    uw=np.asarray(solver.unwrap_cube(phase_input),dtype=np.float64)
    pairs=np.asarray(temporal.edges,dtype=np.int32)
    if not np.isfinite(uw).all():
        raise RuntimeError('Spurt returned nonfinite phase')
    if uw.shape!=(len(pairs),island.npoints):
        raise RuntimeError(f'Spurt output shape mismatch: {uw.shape}')
    # Spurt outputs phase gradients anchored to first point in each IFG.
    # Add observed wrapped phase difference at that first point, as in
    # spurt/workflows/emcf/_unwrap.py's phase_offset.
    anchor_offset=wrap(ph[0,pairs[:,1]].astype(np.float64)-ph[0,pairs[:,0]].astype(np.float64))
    restored=uw+anchor_offset[:,None]
    wrapped_ifg=wrap(ph[:,pairs[:,1]].T.astype(np.float64)-ph[:,pairs[:,0]].T.astype(np.float64))
    parity=float(np.max(np.abs(wrap(restored-wrapped_ifg))))
    if parity>1e-3:
        raise RuntimeError(f'Spurt result not congruent to linked point IFGs: {parity:.7f} rad; do not export')
    loop_relative=temporal_triangle_closure(uw,pairs,temporal.cycles)
    loop_with_anchor=temporal_triangle_closure(restored,pairs,temporal.cycles)
    return {'unwrapped_ifg_phase_rad':restored.astype(np.float32),
            'time_ifg_pairs_0based':pairs,
            'wrapped_phase_parity_max_rad':parity,
            'anchor_offset_rad':anchor_offset.astype(np.float32),
            'temporal_closure_relative':loop_relative,
            'temporal_closure_with_wrapped_anchor':loop_with_anchor,
            'spatial_hole_faces':len(space_graph.hole_faces)}


def _outdir_is_safe(output, stackdir):
    o=Path(output).resolve(); s=Path(stackdir).resolve()
    if o==s or s in o.parents:
        raise ValueError('Output cannot be in PointPhaseStack')
    if s.name=='point_phase_stack' and s.parent.name=='processing':
        processing=s.parent
        if o==processing or processing in o.parents:
            raise ValueError('Diagnostic output must be outside production processing tree')
    if o.exists() and (not o.is_dir() or any(o.iterdir())):
        raise FileExistsError(f'Refusing nonempty output directory: {o}')
    return o


def main(argv=None):
    ap=argparse.ArgumentParser(description='ROI Spurt EMCF on ORIGINAL PS/DS; benchmark only')
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--row0',type=int,required=True)
    ap.add_argument('--col0',type=int,required=True)
    ap.add_argument('--height',type=int,default=1024)
    ap.add_argument('--width',type=int,default=1024)
    ap.add_argument('--subset',choices=('combined','ps','ds'),default='combined')
    ap.add_argument('--max-points',type=int,default=400000)
    ap.add_argument('--max-edge-m',type=float,default=40.0)
    ap.add_argument('--min-area-m2',type=float,default=1e-4)
    ap.add_argument('--island-rank',type=int,default=0)
    ap.add_argument('--max-run-points',type=int,default=60000)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--links-per-batch',type=int,default=1024)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--enable-hole-faces',action='store_true',
                    help='Explicitly admit validated real hole boundary faces; no filling')
    args=ap.parse_args(argv)
    if args.island_rank<0 or args.max_run_points<3:
        ap.error('Island rank must be >=0 and maximum run points >=3')
    from pypsds.context import open_from_config
    from pypsds.geometry.inputs import resolve_geometry_inputs
    from pypsds.gamma.geometry import geometry_from_par
    cfg,_,paths,stack,_=open_from_config(args.config)
    pp=Path(paths.output_dir)/'processing'/'point_phase_stack'
    output=_outdir_is_safe(args.output_dir,pp)
    geometry=geometry_from_par(resolve_geometry_inputs(cfg,paths).reference_rslc_par)
    sy=float(geometry.azimuth_spacing_m)
    sx=float(geometry.ground_range_spacing_m)
    from pypsds.unwrap.native_point_graph import load_tile_points
    point_ids,rr,cc,typ,phase=load_tile_points(root=pp,
        row0=args.row0,col0=args.col0,height=args.height,width=args.width,
        subset=args.subset,max_points=args.max_points)
    if phase.shape[1]!=len(stack.dates):
        raise RuntimeError('Linked phase acquisition count differs from stack metadata')
    islands,georeport=build_triangle_islands(rr,cc,sy,sx,args.max_edge_m,args.min_area_m2)
    info={'status':'ROI_TOPOLOGY_AUDIT' if args.audit_only else 'ROI_EMCF_BENCHMARK_ONLY',
          'source':'ORIGINAL_PHASE_LINKED_PS_DS_POINTS',
          'source_point_phase_stack':str(pp),
          'roi':[args.row0,args.col0,args.height,args.width],
          'subset':args.subset,
          'dates':int(len(stack.dates)),
          'spacings_nominal_m':[sy,sx],
          'geometry':georeport,
          'no_coarse_complex_mean':True,
          'no_nearest_raster_fill':True,
          'no_coarse_cycle_broadcast':True,
          'global_integer_gauge_determined':False,
          'production_modified':False}
    if args.audit_only:
        output.mkdir(parents=True,exist_ok=False) if not output.exists() else None
        (output/'summary.json').write_text(json.dumps(info,indent=2))
        print(json.dumps(info,indent=2));return 0
    if args.island_rank>=len(islands):
        raise RuntimeError(f'Island rank {args.island_rank} unavailable; only {len(islands)}')
    island=islands[args.island_rank]
    if not island.manifold or (not island.solver_eligible and not args.enable_hole_faces):
        raise RuntimeError('Chosen island has holes or non-manifold geometry. For validated manifold holes, explicitly pass --enable-hole-faces.')
    if island.npoints>args.max_run_points:
        raise RuntimeError(f'Island contains {island.npoints:,} points > max-run-points={args.max_run_points:,}; explicitly raise limit after topology audit')
    emcf=run_spurt_emcf(phase[island.original_indices],island,
        workers=args.workers,links_per_batch=args.links_per_batch,
        enable_hole_faces=args.enable_hole_faces)
    import spurt
    info.update({'spurt_version':str(getattr(spurt,'__version__','unknown')),
        'temporal_closure_relative':emcf['temporal_closure_relative'],
        'temporal_closure_with_wrapped_anchor':emcf['temporal_closure_with_wrapped_anchor'],
        'island_rank':args.island_rank,'selected_island':{
        'points':island.npoints,'triangles':island.ntriangles,'edges':island.nlinks,
        'boundary_edges':len(island.boundary),'max_edge_m':island.max_edge_m,
        'manifold':island.manifold,'simply_connected':island.simply_connected},
        'time_ifgs':int(len(emcf['time_ifg_pairs_0based'])),
        'hole_face_count':int(emcf['spatial_hole_faces']),
        'spatial_cycle_count':int(island.ntriangles+emcf['spatial_hole_faces']),
        'hole_faces_input_only_original_boundary_edges':True,
        'wrapped_phase_parity_max_rad':float(emcf['wrapped_phase_parity_max_rad']),
        'output_is_acquisition_timeseries':False,
        'not_validated':'No independent measurement, no cross-island gauge, no temporal inversion.',
    })
    output.mkdir(parents=True,exist_ok=False) if not output.exists() else None
    np.save(output/'unwrapped_ifg_phase_rad.npy',emcf['unwrapped_ifg_phase_rad'])
    np.save(output/'time_ifg_pairs_0based.npy',emcf['time_ifg_pairs_0based'])
    np.save(output/'anchor_offset_rad.npy',emcf['anchor_offset_rad'])
    np.save(output/'original_point_ids.npy',point_ids[island.original_indices])
    np.save(output/'original_radar_rows.npy',rr[island.original_indices])
    np.save(output/'original_radar_cols.npy',cc[island.original_indices])
    np.save(output/'original_point_type.npy',typ[island.original_indices])
    np.save(output/'space_graph_links.npy',island.links)
    np.save(output/'space_graph_cycles.npy',island.triangles)
    (output/'summary.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2));return 0


if __name__=='__main__':
    raise SystemExit(main())
