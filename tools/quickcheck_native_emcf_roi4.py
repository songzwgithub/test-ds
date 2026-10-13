#!/usr/bin/env python3
"""Fast READ-ONLY temporal consistency audit for saved Ningbo ROI4 EMCF.

Reads only existing pinch_v2 .npy arrays; creates only new audit CSV/JSON files.
Fundamental cycles are constructed in saved IFG row order. This basis may differ
from the backend's 145-cycle basis; exact bad counts are not basis-invariant.

A loop's common phase offset is unobservable in a relative spatial audit;
nonzero deviations from the per-loop modal integer are diagnostic, not proof
of which IFG/point is wrong. No input files are edited.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

TAU = 2 * np.pi
DEFAULT_ROOT = Path('/mnt/ningbo_process/ds/output/diagnostics')
REQUIRED = {
    'phase': 'unwrapped_ifg_phase_rad.npy',
    'pairs': 'time_ifg_pairs_0based.npy',
    'ids': 'original_point_ids.npy',
    'rows': 'original_radar_rows.npy',
    'cols': 'original_radar_cols.npy',
    'types': 'original_point_type.npy',
    'anchor': 'anchor_offset_rad.npy',
}


def write_csv(path, entries, fields):
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)


def safe_float(x):
    return float(x) if np.isfinite(x) else None


def temporal_tree(pairs, n_dates):
    root = np.arange(n_dates, dtype=np.int32)
    rank = np.zeros(n_dates, np.int8)
    def find(k):
        while root[k] != k:
            root[k] = root[root[k]]
            k = int(root[k])
        return k
    tree, chords = [], []
    for e, (x, y) in enumerate(pairs):
        a, b = find(int(x)), find(int(y))
        if a == b:
            chords.append(e)
        else:
            if rank[a] < rank[b]:
                a, b = b, a
            root[b] = a
            if rank[a] == rank[b]:
                rank[a] += 1
            tree.append(e)
    if len(tree) != n_dates - 1:
        raise ValueError(f'Time network disconnected: tree_edges={len(tree)}, expected={n_dates-1}')
    adj = [[] for _ in range(n_dates)]
    for e in tree:
        u, v = (int(z) for z in pairs[e])
        adj[u].append((v,e,1))
        adj[v].append((u,e,-1))
    parent = np.full(n_dates, -1, np.int32)
    parent_edge = np.full(n_dates, -1, np.int32)
    parent_sign = np.zeros(n_dates, np.int8)
    parent[0] = 0
    qq = deque([0])
    while qq:
        u = qq.popleft()
        for v, e, sign in adj[u]:
            if parent[v] >= 0:
                continue
            parent[v] = u
            parent_edge[v] = e
            parent_sign[v] = sign
            qq.append(v)
    assert np.all(parent >= 0)
    return chords, parent, parent_edge, parent_sign


def edges_on_cycle(chord, pairs, parent, parent_edge, parent_sign):
    u,v = (int(z) for z in pairs[chord])
    coeff = Counter({int(chord):1})
    x=u
    while x:
        coeff[int(parent_edge[x])] += int(parent_sign[x])
        x=int(parent[x])
    x=v
    while x:
        coeff[int(parent_edge[x])] -= int(parent_sign[x])
        x=int(parent[x])
    clean = [(e,int(s)) for e,s in sorted(coeff.items()) if s]
    if any(abs(s) != 1 for _,s in clean):
        raise AssertionError('Cycle contains repeated edge after cancellation')
    return clean


def geometry_audit(folder, npoint):
    links_file = folder / 'space_graph_links.npy'
    cycles_file = folder / 'space_graph_cycles.npy'
    if not (links_file.is_file() and cycles_file.is_file()):
        return {'status':'skipped','reason':'space_graph_links/cycles missing'}
    links = np.asarray(np.load(links_file,allow_pickle=False))
    cyc = np.asarray(np.load(cycles_file,allow_pickle=False))
    info={'graph_links_shape':list(links.shape),'graph_cycles_shape':list(cyc.shape),
          'interpretation':'not_assumed'}
    if links.ndim != 2 or links.shape[1] != 2:
        info['status']='unsupported_link_format';return info
    if np.any(links<0) or np.any(links>=npoint):
        info['status']='links_are_not_local_vertex_indices';return info
    unique_links={tuple(sorted(map(int,pair))) for pair in links}
    info['unique_links']=len(unique_links)
    info['graph_cycle_rank_if_connected']=len(unique_links)-npoint+1
    if cyc.ndim != 2 or cyc.shape[1] != 3:
        info['status']='unsupported_cycle_format';return info
    if np.any(cyc < 0) or np.any(cyc >= npoint):
        info['status']='cycles_not_vertex_triangles';return info
    tri_edges = Counter()
    missing=0
    for x,y,z in cyc:
        for a,b in ((x,y),(y,z),(z,x)):
            e=tuple(sorted((int(a),int(b))))
            tri_edges[e]+=1
            if e not in unique_links:
                missing+=1
    if missing:
        info['status']='cycle_format_not_triangles_or_edges_missing'
        info['triangle_edges_not_in_links']=missing
        return info
    boundary=[edge for edge,k in tri_edges.items() if k==1]
    inc=Counter()
    for a,b in boundary:
        inc[a]+=1;inc[b]+=1
    info.update({'status':'triangles_verified','boundary_edges':len(boundary),
                 'boundary_vertex_degree_counts':dict(sorted(Counter(inc.values()).items())),
                 'euler_characteristic':npoint-len(unique_links)+len(cyc),
                 'one_hole_cycle_rank_expected':len(cyc)+1})
    return info


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roi',type=Path,default=DEFAULT_ROOT/'native_emcf_roi_island4_pinch_v2')
    ap.add_argument('--out',type=Path,default=DEFAULT_ROOT/'native_emcf_roi4_quickcheck')
    ap.add_argument('--tol-rad',type=float,default=1e-3)
    args=ap.parse_args()
    roi=args.roi.resolve()
    out=args.out.resolve()
    if not roi.is_dir():
        raise SystemExit(f'Missing input: {roi}')
    if out==roi or roi in out.parents:
        raise SystemExit('Refusing to write inside original result directory')
    if args.tol_rad <= 0:
        raise SystemExit('--tol-rad must be >0')
    for f in REQUIRED.values():
        if not (roi/f).is_file():
            raise SystemExit(f'Missing required file: {roi/f}')
    dat={k:np.load(roi/f,allow_pickle=False) for k,f in REQUIRED.items()}
    uw=np.asarray(dat['phase'],dtype=np.float64)
    pairs=np.asarray(dat['pairs'])
    ids,rows,cols,types=[np.asarray(dat[k]).reshape(-1) for k in ('ids','rows','cols','types')]
    anchors=np.asarray(dat['anchor'],dtype=np.float64).reshape(-1)
    n_ifg,npoint=uw.shape
    if n_ifg != 219 or npoint != 3149:
        raise SystemExit(f'Unexpected ROI4 phase shape {uw.shape}; refusing to guess axis meaning')
    if pairs.shape != (n_ifg,2):
        raise SystemExit(f'Unexpected pairs shape: {pairs.shape}')
    if (len(ids),len(rows),len(cols),len(types),len(anchors))!=(npoint,npoint,npoint,npoint,n_ifg):
        raise SystemExit('Point-coordinate or anchor array shape mismatch')
    if not np.issubdtype(pairs.dtype,np.integer) or np.min(pairs)<0 or np.max(pairs)>=75 or np.any(pairs[:,0]==pairs[:,1]):
        raise SystemExit('Time edges are not valid zero-based pairs in 0..74')
    if not np.all(np.isfinite(uw)) or not np.all(np.isfinite(anchors)):
        raise SystemExit('Phase or anchor has NaN/Inf: needs explicit masking (no silent replacement)')
    chords,parent,parent_edge,parent_sign=temporal_tree(pairs,75)
    if len(chords)!=145:
        raise SystemExit(f'Expected 145 fundamental cycles, got {len(chords)}')

    # Unit cycle closure: C U. Using a spanning tree means one traversal for
    # all point columns, then 145 chord residual vectors.
    pot=np.zeros((75,npoint),np.float64)
    pot_anchor=np.zeros(75,np.float64)
    # Parent edges are not necessarily in date order: traverse BFS children.
    children=[[] for _ in range(75)]
    for d in range(1,75):
        children[int(parent[d])].append(d)
    queue=deque([0])
    while queue:
        u=queue.popleft()
        for v in children[u]:
            edge=int(parent_edge[v]); sign=int(parent_sign[v])
            pot[v]=pot[u]+sign*uw[edge]
            pot_anchor[v]=pot_anchor[u]+sign*anchors[edge]
            queue.append(v)

    out.mkdir(parents=True,exist_ok=True)
    anomalies=[]
    cycle_rows=[]
    cycle_specs=[]
    off_lattice=[]
    rel_bad=0; off_lattice_total=0; max_lattice_err=0.0
    loops_with_bad=0
    anchor_hypotheses={key:{'near_integer':0,'nonzero_integer_near':0,'off_integer':0}
                       for key in ('raw_unwrapped','minus_anchor_cycle','plus_anchor_cycle')}
    for ci, e in enumerate(chords):
        u,v=(int(z) for z in pairs[e])
        R=uw[e]-(pot[v]-pot[u])
        # Relative to point zero, integer relation is gauge invariant.
        difference=R-R[0]
        integers=np.rint(difference/TAU).astype(np.int64)
        error=np.abs(difference-TAU*integers)
        max_lattice_err=max(max_lattice_err,float(np.max(error)))
        valid=(error <= args.tol_rad)
        if not np.any(valid):
            raise SystemExit(f'Cycle {ci}: no modulo-consistent points; input phase semantics need inspection')
        cnt=Counter(int(x) for x in integers[valid])
        # Deterministic tie: smaller absolute integer first, then smaller value.
        mode=sorted(cnt.items(),key=lambda z:(-z[1],abs(z[0]),z[0]))[0][0]
        diff=integers-mode
        bad=np.flatnonzero(valid & (diff!=0))
        invalid=np.flatnonzero(~valid)
        rel_bad+=len(bad)
        off_lattice_total+=len(invalid)
        loops_with_bad+=bool(len(bad) or len(invalid))
        ce=edges_on_cycle(int(e),pairs,parent,parent_edge,parent_sign)
        anchor_cycle=float(anchors[e]-(pot_anchor[v]-pot_anchor[u]))
        common_phase=float(np.median(R))
        for label, candidate in (('raw_unwrapped', R),
                                 ('minus_anchor_cycle', R-anchor_cycle),
                                 ('plus_anchor_cycle', R+anchor_cycle)):
            nearest=np.rint(candidate/TAU)
            good=np.abs(candidate-TAU*nearest)<=args.tol_rad
            anchor_hypotheses[label]['near_integer']+=int(np.count_nonzero(good))
            anchor_hypotheses[label]['nonzero_integer_near']+=int(np.count_nonzero(good & (nearest!=0)))
            anchor_hypotheses[label]['off_integer']+=int(np.count_nonzero(~good))
        cycle_specs.append({'cycle_index':int(ci),'chord_ifg':int(e),
                            'oriented_ifgs':[[int(q),int(s)] for q,s in ce],
                            'edge_dates':[u,v]})
        cycle_rows.append({'cycle_index':ci,'chord_ifg':int(e),'start_date_index':u,'end_date_index':v,
                           'n_edges_in_loop':len(ce),'modal_integer_relative_to_point0':mode,
                           'nonmodal_integer_points':len(bad),'noninteger_phase_points':len(invalid),
                           'max_modulo_error_rad':float(np.max(error)),
                           'common_closure_phase_rad':common_phase,
                           'anchor_cycle_phase_rad':anchor_cycle,
                           'median_closure_minus_anchor_rad':common_phase-anchor_cycle})
        for p in bad:
            anomalies.append({'kind':'integer_deviation','cycle_index':ci,'chord_ifg':int(e),
                              'point_local_index':int(p),'original_point_id':int(ids[p]),
                              'point_type':int(types[p]),'radar_row':int(rows[p]),'radar_col':int(cols[p]),
                              'relative_integer_cycles':int(diff[p]),'phase_error_rad':float(error[p]),
                              'date_u':u,'date_v':v})
        for p in invalid:
            off_lattice.append({'kind':'noninteger_closure','cycle_index':ci,'chord_ifg':int(e),
                                'point_local_index':int(p),'original_point_id':int(ids[p]),
                                'point_type':int(types[p]),'radar_row':int(rows[p]),'radar_col':int(cols[p]),
                                'relative_integer_cycles':int(diff[p]),'phase_error_rad':float(error[p]),
                                'date_u':u,'date_v':v})

    fields=['kind','cycle_index','chord_ifg','point_local_index','original_point_id',
            'point_type','radar_row','radar_col','relative_integer_cycles',
            'phase_error_rad','date_u','date_v']
    write_csv(out/'integer_deviations.csv',anomalies,fields)
    write_csv(out/'noninteger_closures.csv',off_lattice,fields)
    write_csv(out/'cycle_summary.csv',cycle_rows,list(cycle_rows[0]))
    (out/'cycle_definitions.json').write_text(json.dumps(cycle_specs,indent=2),encoding='utf8')
    geometry=geometry_audit(roi,npoint)
    tested=npoint*len(chords)
    summary={'status':'DIAGNOSTIC_ONLY_NO_UNWRAPPING',
             'roi_directory':str(roi),'n_ifg':n_ifg,'n_dates':75,'npoints':npoint,
             'fundamental_cycles':len(chords),'tested_combinations':tested,
             'relative_nonmodal_integer_count':rel_bad,
             'relative_nonmodal_fraction':rel_bad/tested,
             'noninteger_closure_count':off_lattice_total,
             'max_point_relative_modulo_error_rad':max_lattice_err,
             'cycles_with_anomaly':loops_with_bad,
             'distinct_integer_anomaly_points':len(set(a['point_local_index'] for a in anomalies)),
             'expected_previous_report_relative_count':6,
             'matches_previous_report_count':rel_bad==6,
             'cycle_basis':'first-encountered spanning tree / chord fundamental cycles',
             'time_gauge_determined':False,
             'anchor_semantics_not_assumed':True,
             'anchor_orientation_probe_not_backend_anchor_metric':anchor_hypotheses,
             'spatial_topology':geometry,
             'warning':'Different valid cycle bases can change nonzero-count; zero modulo error alone does not prove the physical integer history.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print('Input:',roi)
    print('219 IFG × 3149 points, fundamental time cycles:',len(chords))
    print('Relative nonmodal integer tests:',rel_bad,'/',tested)
    print('Off-integer-lattice tests:',off_lattice_total,'/',tested)
    print('Max modulo residual (rad):',f'{max_lattice_err:.8g}')
    print('Distinct anomalous points:',summary['distinct_integer_anomaly_points'])
    print('Previous reported 6 reproduced:',summary['matches_previous_report_count'])
    print('Anchor convention probes (NOT the backend anchor metric):',anchor_hypotheses)
    print('Geometry:',geometry)
    print('Saved diagnostics to:',out)
    if anomalies:
        print('Top anomaly rows:')
        for a in anomalies[:20]:
            print(' ', a)
    if rel_bad!=6:
        print('NOTE: A different fundamental-cycle basis may change the count; compare back-end cycle definitions before deciding regression.')
    if off_lattice_total:
        print('WARNING: non-integer closure present. Diagnose before interpreting integer counts.')


if __name__=='__main__':
    main()
