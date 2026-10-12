"""Unwrap relative integers on ORIGINAL PS/DS point positions, without raster fill.

This is a tile-sized, read-only native-point candidate / independent edge QA.
It does NOT determine a global integer gauge, bridge disconnected areas,
correct time-aliasing, or overwrite the existing production unwrapped stack.

Conceptual alignment: Dolphin/MiaplPy keep phase-linked wrapped observations
separate from the subsequent spatial unwrapping and QA stages. Here we use a
sparse graph because a sparsely masked raw-radar raster is not a connected 2-D
phase field. The scientific assumptions of point-to-point continuity remain
explicit and subject to independent validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pypsds.unwrap.native_integer_constraints import (
    synchronize_integer_graph, node_two_core_mask,
)

TAU = 2 * np.pi


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def load_tile_points(*, root, row0, col0, height, width,
                     subset='combined', max_points=400000):
    """Load only original points inside a radar-coordinate ROI, exact IDs."""
    root = Path(root)
    if min(height, width) <= 0 or min(row0, col0) < 0:
        raise ValueError('ROI dimensions must be positive and offsets nonnegative')
    if subset not in ('combined', 'ps', 'ds'):
        raise ValueError('subset must be combined, ps or ds')
    rows = np.load(root / 'rows.npy', mmap_mode='r', allow_pickle=False)
    cols = np.load(root / 'cols.npy', mmap_mode='r', allow_pickle=False)
    types = np.load(root / 'point_type.npy', mmap_mode='r', allow_pickle=False)
    phase = np.load(root / 'phase_rad.npy', mmap_mode='r', allow_pickle=False)
    if not (rows.ndim == cols.ndim == types.ndim == 1 and
            rows.shape == cols.shape == types.shape and phase.ndim == 2 and
            phase.shape[0] == len(rows)):
        raise ValueError('PointPhaseStack identity or shape mismatch')
    is_point = ((rows >= row0) & (rows < row0 + height) &
                (cols >= col0) & (cols < col0 + width))
    if subset != 'combined':
        is_point &= (types == (1 if subset == 'ps' else 2))
    ids = np.flatnonzero(is_point)
    if len(ids) > max_points:
        raise RuntimeError(f'ROI contains {len(ids):,} points > --max-points={max_points:,}; '
                           'use a smaller tile (no silent sampling)')
    if len(ids) < 2:
        raise RuntimeError(f'Only {len(ids)} original points in ROI; select a larger tile')
    rr = np.asarray(rows[ids], dtype=np.int32)
    cc = np.asarray(cols[ids], dtype=np.int32)
    tt = np.asarray(types[ids], dtype=np.uint8)
    if not np.all((tt == 1) | (tt == 2)):
        raise ValueError('Invalid original PS/DS point type')
    # Only point IDs have meaning. A duplicate radar position would produce
    # two indistinguishable observations and must be resolved upstream.
    packed = (rr.astype(np.int64) - row0) * width + (cc.astype(np.int64) - col0)
    if len(np.unique(packed)) != len(ids):
        raise ValueError('Multiple point IDs occupy the same original radar pixel')
    # Indexing preserves the original order in PointPhaseStack. No averaging.
    ph = np.asarray(phase[ids, :], dtype=np.float32)
    if not np.all(np.isfinite(ph)):
        raise ValueError('Nonfinite linked phase at a selected original point')
    return ids.astype(np.int32), rr, cc, tt, ph


def build_observed_point_edges(*, rows, cols, sy, sx, neighbors=8,
                               max_distance_m=40., workers=4):
    """Connect actual point IDs using a bounded nearest-neighbor proposal.

    Distances are NOMINAL radar-grid offsets, not verified ground/geodesic
    distances. Edges are possible observations, not guaranteed deformation
    continuity; filter and independent validation are required.
    """
    rr = np.asarray(rows); cc = np.asarray(cols)
    n = len(rr)
    if n < 2 or cc.shape != (n,):
        raise ValueError('At least two aligned coordinate vectors required')
    if not all(np.isfinite(x) and x > 0 for x in (sy, sx, max_distance_m)):
        raise ValueError('Nominal spacings and maximum radius must be positive')
    if neighbors < 1 or workers < 1:
        raise ValueError('neighbors and workers must be >=1')
    # Use a local origin so floating point positions are well-scaled.
    pos = np.empty((n, 2), np.float64)
    pos[:, 0] = (rr.astype(np.float64) - np.min(rr)) * float(sy)
    pos[:, 1] = (cc.astype(np.float64) - np.min(cc)) * float(sx)
    tree = cKDTree(pos)
    dist, dst = tree.query(pos, k=min(neighbors+1, n),
                           distance_upper_bound=float(max_distance_m),
                           workers=int(workers))
    if dist.ndim == 1:
        dist = dist[:, None]; dst = dst[:, None]
    a = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], dst.shape)
    valid = np.isfinite(dist) & (dst >= 0) & (dst < n) & (dst != a)
    u = a[valid].astype(np.int32)
    v = dst[valid].astype(np.int32)
    if not len(u):
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    u, v = np.minimum(u,v), np.maximum(u,v)
    # Unique, deterministic unordered edges; no synthetic grid positions.
    key = u.astype(np.int64) * n + v.astype(np.int64)
    key = np.unique(key)
    pairs = np.column_stack((key // n, key % n)).astype(np.int32)
    delta = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    length = np.sqrt(np.einsum('ij,ij->i',delta,delta)).astype(np.float32)
    if np.any(length < 1e-8):
        raise ValueError('Duplicate geographic/nominal coordinates in graph')
    return pairs, length


def point_edge_integer_observations(point_phase, edges, *,
                                    max_step_rad=2.65, batch=4096):
    """Recover candidate relative integer histories from wrapped edge changes.

    Major assumption: true point-to-point phase GRADIENT increments between
    consecutive acquisitions are smaller than pi. A hidden >pi jump can be
    indistinguishable from a smaller observed jump, so do not claim proof.
    """
    ph=np.asarray(point_phase)
    pairs=np.asarray(edges)
    if ph.ndim != 2 or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError('Expected phase [point,date] and edges [edge,2]')
    if len(ph)==0 or ph.shape[1]<2 or not np.all(np.isfinite(ph)):
        raise ValueError('Finite phase with >=2 acquisitions required')
    if np.any(pairs < 0) or np.any(pairs >= len(ph)) or np.any(pairs[:,0] == pairs[:,1]):
        raise ValueError('Invalid native edge IDs')
    if not 0 < max_step_rad < np.pi or batch < 1:
        raise ValueError('max_step_rad must be (0,pi) and batch>=1')
    nedge, nt = len(pairs), ph.shape[1]
    branch=np.zeros((nedge,nt),np.int16)
    valid=np.zeros(nedge, np.bool_)
    max_step=np.full(nedge,np.nan,np.float32)
    rms_step=np.full(nedge,np.nan,np.float32)
    for st in range(0,nedge,batch):
        en=min(st+batch,nedge)
        p=pairs[st:en]
        raw=(np.asarray(ph[p[:,1]],np.float64)-
             np.asarray(ph[p[:,0]],np.float64))
        wrapped=wrap(raw)
        steps=wrap(np.diff(wrapped,axis=1))
        ms=np.max(np.abs(steps),axis=1)
        rms=np.sqrt(np.mean(steps*steps,axis=1))
        uw=np.unwrap(wrapped,axis=1)
        kval=np.rint((uw-raw)/TAU)
        stable=(ms<max_step_rad)&np.all(np.isfinite(kval),axis=1)
        if np.any(np.abs(kval[stable]) > 32700):
            raise OverflowError('Observed edge integer cycles exceed int16')
        branch[st:en]=kval.astype(np.int16)
        valid[st:en]=stable
        max_step[st:en]=ms.astype(np.float32)
        rms_step[st:en]=rms.astype(np.float32)
    return branch,valid,max_step,rms_step


def solve_native_tile(*, point_phase, edges, distance_m,
                      holdout_fraction=0.20, max_step_rad=2.65,
                      random_seed=2026, batch=4096):
    """Independent heldout edges plus two training forests, all point-native."""
    ph=np.asarray(point_phase)
    edges=np.asarray(edges,dtype=np.int32)
    distance=np.asarray(distance_m,dtype=np.float32)
    if distance.shape != (len(edges),):
        raise ValueError('Native edge distances not aligned')
    if not 0<=holdout_fraction<0.5:
        raise ValueError('holdout_fraction must be [0,0.5)')
    if not len(edges):
        raise RuntimeError('No measured-point edges: cannot infer integers')
    k,okay,maxstep,rms=point_edge_integer_observations(
        ph,edges,max_step_rad=max_step_rad,batch=batch)
    if not np.any(okay):
        raise RuntimeError('No temporally accepted native point edges')
    good=np.flatnonzero(okay)
    # Disjoint validation set picked using ONLY edge identity, never integer
    # gradients. Comparison on heldout edges is independent of forest fitting,
    # but shares observations, so is NOT independent ground truth.
    gen=np.random.default_rng(random_seed)
    perm=gen.permutation(len(good))
    nh=int(round(len(good)*holdout_fraction))
    in_holdout=np.zeros(len(edges),bool)
    in_holdout[good[perm[:nh]]]=True
    in_train=okay&(~in_holdout)
    train_edges=edges[in_train]
    train_k=k[in_train]
    d=distance[in_train].astype(np.float64)
    s=maxstep[in_train].astype(np.float64)
    # Two deterministic rankings to probe alternative tree-choice fragility.
    sa=1.0/(d+1e-3)
    sb=1.0/(s+1e-3)
    sol=synchronize_integer_graph(
        nnode=ph.shape[0],node_pairs=train_edges,
        integer_edge_histories=train_k,
        score_primary=sa,score_secondary=sb,ndate=ph.shape[1])
    primary=sol['cycles_primary']
    secondary=sol['cycles_secondary']
    cc=sol['component_id']
    core=node_two_core_mask(len(ph),train_edges)
    agreement=np.mean(primary==secondary,axis=1)
    def measure(selection, cycles):
        idx=np.flatnonzero(selection)
        if len(idx)==0:
            return {'candidate_edges':0,'edges_with_same_component':0,
                    'fraction_edges_mismatched_any_epoch':None,
                    'fraction_mismatched_edge_epochs':None}
        sel=edges[idx]
        comparable=(cc[sel[:,0]]==cc[sel[:,1]])
        ii=idx[comparable]
        if len(ii)==0:
            return {'candidate_edges':len(idx),'edges_with_same_component':0,
                    'fraction_edges_mismatched_any_epoch':None,
                    'fraction_mismatched_edge_epochs':None}
        p=edges[ii]
        mismatch=(cycles[p[:,1]]-cycles[p[:,0]] != k[ii])
        return {'candidate_edges':int(len(idx)),
                'edges_with_same_component':int(len(ii)),
                'fraction_edges_mismatched_any_epoch':float(np.mean(mismatch.any(axis=1))),
                'fraction_mismatched_edge_epochs':float(np.mean(mismatch))}
    strict=(core)&(agreement>=0.99)
    ncomp=len(sol['component_root_node'])
    report={
        'status':'LOCAL_CANDIDATE_ONLY_NO_PRODUCTION_WRITE',
        'native_points':int(len(ph)),
        'dates':int(ph.shape[1]),
        'native_neighbor_edges':int(len(edges)),
        'temporal_accepted_edges':int(okay.sum()),
        'temporal_rejected_edges':int((~okay).sum()),
        'train_edges':int(in_train.sum()),
        'heldout_edges':int(in_holdout.sum()),
        'validation_note':('Heldout native edges are not used in forest fitting, '
                            'but their phase-derived integer histories and '
                            'temporal acceptance come from the same observations; '
                            'this is internal validation, not ground truth.'),
        'training_components':int(ncomp),
        'largest_training_component_points':int(np.bincount(cc,minlength=ncomp).max()),
        'nodes_two_core_and_forest_agreement':int(strict.sum()),
        'fraction_nodes_two_core_and_forest_agreement':float(strict.mean()),
        'primary_training_edge_closure':measure(in_train,primary),
        'primary_heldout_edge_closure':measure(in_holdout,primary),
        'secondary_heldout_edge_closure':measure(in_holdout,secondary),
        'baseline_assumptions':(
            'Every accepted edge still assumes locally continuous phase-gradient '
            'changes (<pi per acquisition interval); closed loops cannot prove '
            'absence of common-mode integer aliases. Component reference gauges '
            'are free and no points are interpolated.'),
        'input_representation':'ORIGINAL_POINT_PHASE_STACK_NO_GRID_AVERAGING',
        'empty_pixel_interpolation':False,
        'coarse_node_cycles_broadcast_to_points':False,
        'absolute_integer_gauge_determined':False,
    }
    return {
        'report':report, 'cycles_primary':primary,
        'cycles_secondary':secondary,
        'component_id':cc,
        'node_two_core_mask':core,
        'node_forest_agreement_fraction':agreement.astype(np.float32),
        'candidate_point_edge_ids':edges,
        'candidate_point_edge_cycles':k,
        'candidate_point_edge_distance_m':distance,
        'candidate_point_edge_max_temporal_step_rad':maxstep,
        'candidate_point_edge_rms_temporal_step_rad':rms,
        'candidate_point_edge_accepted':okay,
        'candidate_point_edge_train':in_train,
        'candidate_point_edge_heldout':in_holdout,
    }



def baseline_edge_comparison(*, wrapped_point_phase, baseline_point_phase,
                             edge_pairs, edge_integer, component_id,
                             accepted_mask, heldout_mask, tolerance_rad=1e-3):
    """Compare existing point-level unwrapped result on the SAME native edges.

    Component-wide integers cancel from edge gradients. This comparison is
    internal; it cannot identify the physically correct solution by itself.
    """
    ph=np.asarray(wrapped_point_phase,dtype=np.float64)
    base=np.asarray(baseline_point_phase,dtype=np.float64)
    edges=np.asarray(edge_pairs,dtype=np.int32)
    k=np.asarray(edge_integer,dtype=np.int16)
    components=np.asarray(component_id)
    keep=np.asarray(accepted_mask,bool)
    holdout=np.asarray(heldout_mask,bool)
    if ph.ndim != 2 or base.shape != ph.shape or k.shape != (len(edges),ph.shape[1]):
        raise ValueError('Baseline/native phase or edge shape mismatch')
    if components.shape != (len(ph),) or keep.shape != (len(edges),) or holdout.shape != keep.shape:
        raise ValueError('Native graph component/mask mismatch')
    if not np.isfinite(base).all():
        raise ValueError('Nonfinite baseline point phase')
    parity=float(np.max(np.abs(wrap(base-ph))))
    if parity > tolerance_rad:
        raise ValueError(f'Baseline point phase parity mismatch: {parity:.6g} rad')
    cyc=np.rint((base-ph)/TAU)
    if np.max(np.abs(cyc))>2**30:
        raise ValueError('Baseline integer cycles exceed int32')
    cyc=cyc.astype(np.int32)
    def measure(mask):
        ids=np.flatnonzero(mask & keep)
        if not len(ids):
            return {'accepted_edges':0,'comparable_edges':0,
                    'fraction_edges_mismatched_any_epoch':None,
                    'fraction_mismatched_edge_epochs':None}
        es=edges[ids]
        comparable=components[es[:,0]]==components[es[:,1]]
        q=ids[comparable]
        if not len(q):
            return {'accepted_edges':len(ids),'comparable_edges':0,
                    'fraction_edges_mismatched_any_epoch':None,
                    'fraction_mismatched_edge_epochs':None}
        uv=edges[q]
        diff=(cyc[uv[:,1]]-cyc[uv[:,0]]) != k[q]
        return {'accepted_edges':int(len(ids)), 'comparable_edges':int(len(q)),
                'fraction_edges_mismatched_any_epoch':float(np.mean(np.any(diff,axis=1))),
                'fraction_mismatched_edge_epochs':float(np.mean(diff))}
    return {'wrap_parity_max_rad':parity,
            'training_edge_closure':measure(~holdout),
            'heldout_edge_closure':measure(holdout),
            'note':'Same-graph closure comparison; neither source is independently verified.'}

def main(argv=None):
    p=argparse.ArgumentParser(description='Native original PS/DS integer graph on a radar ROI; no production write')
    p.add_argument('--config',type=Path,default=None,
                   help='pyPSDS YAML; required unless both radar coordinate spacings are supplied')
    p.add_argument('--point-phase-stack',type=Path,default=None,
                   help='PointPhaseStack directory; from config if omitted')
    p.add_argument('--row-spacing-m',type=float)
    p.add_argument('--col-spacing-m',type=float)
    p.add_argument('--row0',type=int,required=True)
    p.add_argument('--col0',type=int,required=True)
    p.add_argument('--height',type=int,default=1024)
    p.add_argument('--width',type=int,default=1024)
    p.add_argument('--subset',choices=('combined','ps','ds'),default='combined')
    p.add_argument('--max-points',type=int,default=400000)
    p.add_argument('--neighbors',type=int,default=8)
    p.add_argument('--max-distance-m',type=float,default=40.0)
    p.add_argument('--max-step-rad',type=float,default=2.65)
    p.add_argument('--holdout-fraction',type=float,default=0.20)
    p.add_argument('--edge-batch',type=int,default=4096)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--baseline-point-phase',type=Path,default=None,
                   help='Optional existing fully unwrapped [all original point,75] NPY')
    p.add_argument('--audit-only',action='store_true',
                   help='Read real points and build graph but do not solve integers')
    args=p.parse_args(argv)
    if args.point_phase_stack is None:
        if args.config is None:
            p.error('--config or --point-phase-stack must be supplied')
        from pypsds.context import open_from_config
        _,_,paths,stack,_=open_from_config(args.config)
        pp=Path(paths.output_dir)/'processing'/'point_phase_stack'
    else:
        pp=args.point_phase_stack
        stack=None
    if args.row_spacing_m is None or args.col_spacing_m is None:
        if args.config is None:
            p.error('Both radar spacings or --config are required')
        from pypsds.context import open_from_config
        from pypsds.geometry.inputs import resolve_geometry_inputs
        from pypsds.gamma.geometry import geometry_from_par
        cfg,_,paths,stack,_=open_from_config(args.config)
        geom=geometry_from_par(resolve_geometry_inputs(cfg,paths).reference_rslc_par)
        sy=geom.azimuth_spacing_m if args.row_spacing_m is None else args.row_spacing_m
        sx=geom.ground_range_spacing_m if args.col_spacing_m is None else args.col_spacing_m
    else:
        sy,sx=args.row_spacing_m,args.col_spacing_m
    # Restrict side effects to a fresh diagnostic output location.
    output=args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Diagnostic destination is not empty: {output}')
    processing = pp.resolve().parent
    if output == pp.resolve() or pp.resolve() in output.parents:
        raise ValueError('Diagnostic output must not be inside PointPhaseStack')
    if pp.name == 'point_phase_stack' and processing.name == 'processing':
        if output == processing or processing in output.parents:
            raise ValueError('Diagnostic output must be outside ALL production processing directories')
    point_ids,rr,cc,tt,ph=load_tile_points(
        root=pp,row0=args.row0,col0=args.col0,height=args.height,
        width=args.width,subset=args.subset,max_points=args.max_points)
    if stack is not None and len(stack.dates)!=ph.shape[1]:
        raise ValueError('Acquisition count does not match YAML stack')
    edges,dist=build_observed_point_edges(rows=rr,cols=cc,sy=sy,sx=sx,
          neighbors=args.neighbors,max_distance_m=args.max_distance_m,workers=args.workers)
    print(f'[NATIVE POINT] points={len(point_ids):,} edges={len(edges):,}; '
          f'ROI=({args.row0},{args.col0},{args.height},{args.width})',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    manifest={
        'roi':[args.row0,args.col0,args.height,args.width],
        'subset':args.subset,'spacing_nominal_m':[float(sy),float(sx)],
        'neighbors':args.neighbors,'max_distance_nominal_m':args.max_distance_m,
        'source':str(pp),'output_dir':str(output),
        'no_coarse_grid_complex_mean':True,
        'no_nearest_raster_fill':True,
        'no_coarse_cycle_restore':True,
        'production_modified':False,
    }
    if args.audit_only:
        manifest.update({'status':'NATIVE_GRAPH_TOPOLOGY_ONLY',
                         'points':len(point_ids),'candidate_edges':len(edges)})
        (output/'summary.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps(manifest,indent=2));return 0
    result=solve_native_tile(
        point_phase=ph,edges=edges,distance_m=dist,
        holdout_fraction=args.holdout_fraction,max_step_rad=args.max_step_rad,
        batch=args.edge_batch)
    for name,value in result.items():
        if name != 'report':np.save(output/(name+'.npy'),value)
    np.save(output/'original_point_ids.npy',point_ids)
    np.save(output/'original_radar_rows.npy',rr)
    np.save(output/'original_radar_cols.npy',cc)
    np.save(output/'original_point_type.npy',tt)
    # Gauge-free local-node product: NOT a reference-aligned LOS time series.
    np.save(output/'local_relative_unwrapped_phase_rad.npy',
            ph.astype(np.float64) + TAU*result['cycles_primary'].astype(np.float64))
    manifest.update(result['report'])
    if len(edges):
        u=tt[edges[:,0]];v=tt[edges[:,1]]
        acc=result['candidate_point_edge_accepted']
        eclass={'ps_ps':(u==1)&(v==1),
                'ds_ds':(u==2)&(v==2),
                'ps_ds':(u!=v)}
        manifest['edge_type_support']={key:{
            'candidate':int(mask.sum()),
            'temporal_accepted':int(np.count_nonzero(mask&acc)),
        } for key,mask in eclass.items()}
        manifest['edge_distance_nominal_m_p50_p95_p99']=[float(z) for z in np.percentile(dist,[50,95,99])]
    if args.baseline_point_phase is not None:
        base=np.load(args.baseline_point_phase,mmap_mode='r',allow_pickle=False)
        all_ph=np.load(pp/'phase_rad.npy',mmap_mode='r',allow_pickle=False)
        if base.shape != all_ph.shape:
            raise ValueError('Baseline must have the SAME full PointPhaseStack point/date shape')
        base_local=np.asarray(base[point_ids,:],dtype=np.float32)
        manifest['baseline_comparison']=baseline_edge_comparison(
            wrapped_point_phase=ph,baseline_point_phase=base_local,
            edge_pairs=edges,
            edge_integer=result['candidate_point_edge_cycles'],
            component_id=result['component_id'],
            accepted_mask=result['candidate_point_edge_accepted'],
            heldout_mask=result['candidate_point_edge_heldout'])
        manifest['baseline_point_phase']=str(args.baseline_point_phase)
    (output/'summary.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2));return 0


if __name__=='__main__':
    raise SystemExit(main())
