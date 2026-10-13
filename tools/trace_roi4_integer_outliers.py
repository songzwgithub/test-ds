#!/usr/bin/env python3
"""Read-only targeted integer-cycle diagnostics for Ningbo EMCF island 4.

Requires files from the existing quickcheck and pinch_v2 runs. Tests whether
per-point relative cycle defects have a 1- or 2-IFG sparse explanation. These
are algebraic candidates, NOT guaranteed true IFG errors or automatic repairs.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

TAU = 2*np.pi


def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='', encoding='utf8') as f:
        writer=csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def signature_map(C):
    ret=defaultdict(list)
    for e in range(C.shape[1]):
        col=C[:,e]
        if not np.any(col):
            continue  # time graph bridge: not observable in cycle closure
        for sign in (-1,1):
            ret[(sign*col).astype(np.int8).tobytes()].append((int(e),int(sign)))
    return ret


def exact_explanations(v, C, sigmap, limit=200):
    """Exact one- or two-IFG support, coefficient restricted to +/-1."""
    vec=np.asarray(v,dtype=np.int8)
    matches1=sigmap.get(vec.tobytes(),[])
    if matches1:
        return 1,[[[e,s]] for e,s in matches1[:limit]],len(matches1)
    two=[];seen=set()
    for i in range(C.shape[1]):
        col=C[:,i]
        if not np.any(col):
            continue
        for s in (-1,1):
            residual=(vec.astype(np.int16)-s*col.astype(np.int16))
            if np.max(np.abs(residual))>1:
                continue
            for j,t in sigmap.get(residual.astype(np.int8).tobytes(),[]):
                if j==i:
                    continue
                explanation=tuple(sorted(((i,s),(j,t))))
                if explanation not in seen:
                    seen.add(explanation)
                    two.append([[int(x),int(y)] for x,y in explanation])
                    if len(two)>=limit:
                        return 2,two,f'>={limit}'
    return (2,two,len(two)) if two else (None,[],0)


def boundary_pinch(triangles, npoint):
    edge_occ=Counter()
    for a,b,c in triangles:
        for u,v in ((a,b),(b,c),(c,a)):
            edge_occ[tuple(sorted((int(u),int(v))))]+=1
    edges=[p for p,n in edge_occ.items() if n==1]
    deg=Counter()
    for u,v in edges:
        deg[u]+=1;deg[v]+=1
    return [int(k) for k,n in deg.items() if n>2],len(edges),dict(Counter(deg.values()))


def distances_from_sources(adj, sources):
    distances={}
    for source in sources:
        dist=np.full(len(adj),-1,dtype=np.int32)
        dist[source]=0
        q=deque([source])
        while q:
            u=q.popleft()
            for v in adj[u]:
                if dist[v]<0:
                    dist[v]=dist[u]+1
                    q.append(v)
        distances[int(source)]=dist
    return distances


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    base=Path('/mnt/ningbo_process/ds/output/diagnostics')
    ap.add_argument('--roi',type=Path,default=base/'native_emcf_roi_island4_pinch_v2')
    ap.add_argument('--quickcheck',type=Path,default=base/'native_emcf_roi4_quickcheck')
    ap.add_argument('--out',type=Path,default=base/'native_emcf_roi4_trace')
    a=ap.parse_args()
    roi=a.roi.resolve();qc=a.quickcheck.resolve();out=a.out.resolve()
    if not roi.is_dir() or not qc.is_dir():
        raise SystemExit('Missing ROI or quickcheck directory')
    if out==roi or roi in out.parents or out==qc or qc in out.parents:
        raise SystemExit('Output must be a separate folder from all inputs')
    req=['unwrapped_ifg_phase_rad.npy','time_ifg_pairs_0based.npy','original_point_ids.npy','original_radar_rows.npy','original_radar_cols.npy','original_point_type.npy','space_graph_links.npy','space_graph_cycles.npy']
    for r in req:
        if not (roi/r).is_file():
            raise SystemExit(f'Missing {roi/r}')
    cycles_json=qc/'cycle_definitions.json'
    if not cycles_json.exists():
        raise SystemExit(f'Missing {cycles_json}, run previous quickcheck first')
    U=np.load(roi/'unwrapped_ifg_phase_rad.npy',mmap_mode='r')
    pair=np.load(roi/'time_ifg_pairs_0based.npy')
    ids=np.load(roi/'original_point_ids.npy')
    rows=np.load(roi/'original_radar_rows.npy')
    cols=np.load(roi/'original_radar_cols.npy')
    types=np.load(roi/'original_point_type.npy')
    links=np.load(roi/'space_graph_links.npy')
    triangles=np.load(roi/'space_graph_cycles.npy')
    if U.shape!=(219,3149) or pair.shape!=(219,2):
        raise SystemExit(f'Unexpected shapes U={U.shape}, pairs={pair.shape}')
    cs=json.loads(cycles_json.read_text())
    if len(cs)!=145:
        raise SystemExit(f'Unexpected number cycles: {len(cs)}')
    C=np.zeros((145,219),dtype=np.int8)
    for k,entry in enumerate(cs):
        for e,sign in entry['oriented_ifgs']:
            C[k,int(e)]=int(sign)
    if np.max(np.abs(C.astype(np.int16)@pair[:,1].astype(np.int16)-C.astype(np.int16)@pair[:,0].astype(np.int16))) != 0:
        raise SystemExit('Saved cycle definitions are not valid temporal cycles')
    # Relative closure is insensitive to per-cycle common offsets.
    closure=C.astype(np.float64)@np.asarray(U,dtype=np.float64)
    centered=closure-np.median(closure,axis=1)[:,None]
    k=np.rint(centered/TAU).astype(np.int16)
    phase_error=np.max(np.abs(centered-TAU*k))
    if phase_error>1e-3:
        raise SystemExit(f'Noninteger closure residual too high: {phase_error}')
    anomaly_counts=np.count_nonzero(k,axis=0)
    badpts=np.flatnonzero(anomaly_counts)
    print(f'Nonmodal entries: {int(np.count_nonzero(k))}; anomalous points: {len(badpts)}; modulo max rad={phase_error:.3e}')
    if len(badpts)==0:
        print('No relative temporal integer defects in this cycle basis')
        return
    print('Anomalies:', [(int(p),int(ids[p]),int(anomaly_counts[p])) for p in badpts])
    adj=[[] for _ in range(len(ids))]
    for u,v in links:
        u=int(u);v=int(v)
        adj[u].append(v);adj[v].append(u)
    pinch_vertices,boundary_len,boundary_degrees=boundary_pinch(triangles,len(ids))
    dists=distances_from_sources(adj,pinch_vertices)
    sigmap=signature_map(C)
    out.mkdir(parents=True,exist_ok=True)
    reports=[];neighbors_report=[]
    for p in badpts:
        p=int(p)
        vi=k[:,p]
        count_nonzero=int(np.count_nonzero(vi))
        support,predictions,nmatches=exact_explanations(vi,C,sigmap)
        candidate_report=[]
        for explanation in predictions[:30]:
            edges=[]
            total_improvement=0
            for e,s in explanation:
                neighbor_median=float(np.median(U[e,adj[p]])) if adj[p] else float('nan')
                before=float(U[e,p]-neighbor_median)
                after=before-s*TAU
                total_improvement+=abs(before)-abs(after)
                edges.append(dict(ifg=e,dates=[int(x) for x in pair[e]],integer_cycles=s,
                                  neighbor_count=len(adj[p]),local_before_rad=round(before,6),
                                  local_after_candidate_rad=round(after,6)))
            candidate_report.append({'edges':edges,'local_abs_residual_improvement_rad':round(total_improvement,6)})
        candidate_report.sort(key=lambda z:-z['local_abs_residual_improvement_rad'])
        ranked=[]
        for e in range(219):
            for s in (-1,1):
                res=vi-s*C[:,e]
                score=int(np.count_nonzero(res))
                if score<count_nonzero:
                    ranked.append((score,int(e),int(s)))
        ranked.sort()
        info={'local_index':p,'original_point_id':int(ids[p]),'radar_row':int(rows[p]),'radar_col':int(cols[p]),
              'point_type':int(types[p]),'anomalous_time_cycles':count_nonzero,
              'anomaly_cycle_indices':np.flatnonzero(vi).astype(int).tolist(),
              'anomaly_chord_ifgs':[int(cs[z]['chord_ifg']) for z in np.flatnonzero(vi)],
              'one_or_two_edge_exact_explanation':support,
              'exact_explanations_found':nmatches,
              'best_explanations':candidate_report[:20],
              'top_single_ifg_partial_matches':[
                  {'remaining_bad_cycles':score,'ifg':e,'dates':[int(x) for x in pair[e]],'integer_cycles':s}
                  for score,e,s in ranked[:12]
              ],
              'graph_neighbors':len(adj[p]),'pinch_hops':{str(v):int(dists[v][p]) for v in pinch_vertices}}
        reports.append(info)
        for n in sorted(adj[p]):
            neighbors_report.append({'bad_local':p,'bad_id':int(ids[p]),'neighbor_local':n,
                                     'neighbor_id':int(ids[n]),'neighbor_row':int(rows[n]),'neighbor_col':int(cols[n]),
                                     'neighbor_time_cycle_count':int(anomaly_counts[n])})
        print(f'Point local={p} id={ids[p]} ({rows[p]},{cols[p]}): {count_nonzero} anomalous cycles, graph degree={len(adj[p])}')
        print('  pinch hops:',info['pinch_hops'])
        print('  exact minimal support:',support,'candidate count:',nmatches)
        for x in candidate_report[:5]:
            print('  candidate:', [(e['ifg'],e['dates'],e['integer_cycles'],e['local_before_rad'],e['local_after_candidate_rad']) for e in x['edges']], 'spatial improvement:',x['local_abs_residual_improvement_rad'])
        if not predictions:
            print('  best partial one-edge matches:',info['top_single_ifg_partial_matches'][:5])
    report={'status':'DIAGNOSTIC_ONLY_NO_EDITS',
            'relative_cycle_integer_anomalies':int(np.count_nonzero(k)),
            'outlier_points':len(badpts),
            'pinch_vertices':[{'local_index':p,'id':int(ids[p]),'row':int(rows[p]),'col':int(cols[p])} for p in pinch_vertices],
            'boundary_edges':boundary_len,'boundary_vertex_degrees':boundary_degrees,
            'points':reports,
            'interpretation':'Exact sparse-cycle signatures identify algebraic IFG candidates only; cycles cannot distinguish time-node gauge. Spatial neighborhood residuals are a heuristic, not independent ground truth.'}
    (out/'trace_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    write_csv(out/'bad_point_neighbors.csv',neighbors_report)
    print('Saved:',out/'trace_summary.json')

if __name__=='__main__':
    main()
