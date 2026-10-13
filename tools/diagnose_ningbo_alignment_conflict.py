#!/usr/bin/env python3
"""Read-only audit of Ningbo native EMCF overlap alignment conflicts.

Uses SAVED alignment_manifest.json and alignment_edges.json; does not re-run
EMCF, temporal inversion, alignment, or modify any production outputs.
Optional --duplicate-audit checks whether the mosaic would be blocked by
original point IDs assigned to different alignment components.
"""
from __future__ import annotations
import argparse
import json
from collections import deque, defaultdict
from pathlib import Path

import numpy as np


def readj(p):
    with p.open('r', encoding='utf8') as f:
        return json.load(f)


def graph_path(edges, start, goal, omit):
    graph=defaultdict(list)
    for n,e in enumerate(edges):
        if n==omit: continue
        u,v=int(e['i']),int(e['j'])
        graph[u].append((v,n,+1))
        graph[v].append((u,n,-1))
    q=deque([start]); parents={start:None}
    while q and goal not in parents:
        u=q.popleft()
        for v,n,sign in graph[u]:
            if v not in parents:
                parents[v]=(u,n,sign)
                q.append(v)
    if goal not in parents:return None
    path=[];cur=goal
    while cur!=start:
        prev,n,sign=parents[cur]
        path.append((n,sign,prev,cur))
        cur=prev
    return list(reversed(path))


def duplicate_component_audit(blocks):
    ids_all=[];owners=[];block_indices=[]
    count=0
    for idx,b in enumerate(blocks):
        base=Path(b['base'])
        raw=base/'raw'
        quality=base/'timeinv'
        if not (raw/'original_point_ids.npy').is_file():
            raise FileNotFoundError(raw/'original_point_ids.npy')
        allids=np.load(raw/'original_point_ids.npy',mmap_mode='r',allow_pickle=False)
        valid=np.load(quality/'valid_points.npy',mmap_mode='r',allow_pickle=False)
        if len(allids)!=len(valid):
            raise ValueError('Point ID/mask mismatch '+str(base))
        goodids=np.asarray(allids[np.asarray(valid,dtype=bool)],dtype=np.int64)
        if b.get('npoint') is not None and len(goodids)!=int(b['npoint']):
            raise ValueError('Point count differs from manifest '+str(base))
        if len(goodids)==0:continue
        if len(goodids)!=len(np.unique(goodids)):
            raise ValueError('Duplicate point IDs within block '+str(base))
        ids_all.append(goodids)
        owners.append(np.full(len(goodids),int(b['component']),dtype=np.int32))
        block_indices.append(np.full(len(goodids),idx,dtype=np.int32))
        count+=len(goodids)
    if not count:return {'stored_point_instances':0,'unique_source_point_ids':0,'duplicate_instances':0,'cross_component_duplicate_adjacent_pairs':0,'examples':[]}
    ids=np.concatenate(ids_all); labs=np.concatenate(owners); bid=np.concatenate(block_indices)
    del ids_all,owners,block_indices
    order=np.argsort(ids,kind='stable')
    sids=ids[order]; slab=labs[order]; sblocks=bid[order]
    repeats=(sids[1:]==sids[:-1])
    cross=repeats & (slab[1:]!=slab[:-1])
    examples=[]
    for jj in np.flatnonzero(cross)[:10]:
        examples.append({'point_id':int(sids[jj]),'component_a':int(slab[jj]),'component_b':int(slab[jj+1]),'block_a':blocks[int(sblocks[jj])]['name'],'block_b':blocks[int(sblocks[jj+1])]['name']})
    return {'stored_point_instances':int(count),'unique_source_point_ids':int(count-np.count_nonzero(repeats)),'duplicate_instances':int(np.count_nonzero(repeats)),'cross_component_duplicate_adjacent_pairs':int(np.count_nonzero(cross)),'examples':examples,
            'note':'Follows existing stitcher duplicate-component rejection rule. Does not modify offsets or assign reference gauges.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--alignment',type=Path,default=Path('/mnt/ningbo_process/ds/output/diagnostics/native_emcf_alignment_finish_v1'))
    ap.add_argument('--out',type=Path,default=None,help='New or existing JSON report path; does not change alignment files')
    ap.add_argument('--duplicate-audit',action='store_true',help='Scan all valid original point IDs to forecast mosaic duplicate-component failure (~1-3 GiB RAM)')
    args=ap.parse_args()
    root=args.alignment.resolve()
    manifest=readj(root/'alignment_manifest.json')
    edges=readj(root/'alignment_edges.json')
    blocks=manifest['blocks']
    conflicts=manifest.get('alignment_cycle_conflicts',[])
    results={'source':str(root),'manifest_status':manifest.get('status'),
             'nblocks':len(blocks),'accepted_overlap_edges':len(edges),
             'rejected_overlap_pairs':manifest.get('rejected_overlap_pairs'),
             'alignment_components':manifest.get('disconnected_alignment_components'),
             'nconflicts':len(conflicts),'conflicts':[]}
    edge_map=defaultdict(list)
    for n,e in enumerate(edges):edge_map[tuple(sorted((int(e['i']),int(e['j']))))].append(n)
    for c in conflicts:
        i,j=int(c['i']),int(c['j'])
        matches=edge_map.get(tuple(sorted((i,j))),[])
        if not matches:
            results['conflicts'].append({'manifest_conflict':c,'error':'No corresponding accepted edge'});continue
        for n in matches:
            e=edges[n]
            path=graph_path(edges,int(e['i']),int(e['j']),n)
            if path is None:
                results['conflicts'].append({'manifest_conflict':c,'edge_index':n,'error':'No alternate accepted path (inspect saved graph)'});continue
            direct=np.asarray(e['delta_rad'],dtype=np.float64)
            along=np.zeros_like(direct)
            participants=[{'edge_index':n,'name_a':blocks[int(e['i'])]['name'],'name_b':blocks[int(e['j'])]['name'],
                           'shared':int(e['shared']),'p95_rad':float(e['p95_rad']),'max_rad':float(e['max_rad']),
                           'is_conflict_edge':True}]
            for pn,sg,u,v in path:
                pe=edges[pn]
                along+=sg*np.asarray(pe['delta_rad'],dtype=np.float64)
                participants.append({'edge_index':pn,'name_a':blocks[u]['name'],'name_b':blocks[v]['name'],
                                     'shared':int(pe['shared']),'p95_rad':float(pe['p95_rad']),'max_rad':float(pe['max_rad']),
                                     'is_conflict_edge':False})
            mismatch=along-direct
            imax=np.argsort(-np.abs(mismatch))[:12]
            cases=[{'epoch_index':int(t),'cycle_mismatch_rad':float(mismatch[t]),
                    'mismatch_in_2pi_cycles':float(mismatch[t]/(2*np.pi))} for t in imax if abs(mismatch[t])>1e-6]
            residual=np.abs(mismatch-2*np.pi*np.rint(mismatch/(2*np.pi)))
            results['conflicts'].append({'manifest_conflict':c,'conflict_edge_index':n,'conflict_block_a':blocks[int(e['i'])]['name'],
                'conflict_block_b':blocks[int(e['j'])]['name'],'cycle_edge_count':len(participants),
                'max_abs_cycle_mismatch_rad':float(np.max(np.abs(mismatch))),
                'epochs_abs_mismatch_gt_0p1_rad':int(np.count_nonzero(np.abs(mismatch)>0.1)),
                'max_mod_2pi_residual_rad':float(np.max(residual)),
                'largest_epoch_mismatches':cases,
                'cycle_edges':participants,
                'caution':'Any edge of this cycle could be inconsistent. The edge flagged by BFS is NOT necessarily the faulty edge; inspect overlap strength and independent reference evidence before excluding an edge.'})
    if args.duplicate_audit:
        results['duplicate_component_audit']=duplicate_component_audit(blocks)
    dest=args.out or (root.parent/'native_emcf_alignment_conflict_diagnosis.json')
    dest=dest.resolve()
    if dest in (root/'alignment_manifest.json',root/'alignment_edges.json',root/'block_time_offset_rad.npy',root/'block_alignment_component_id.npy'):
        raise ValueError('Do not overwrite original alignment files')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
    print('ALIGNMENT_CONFLICT_AUDIT')
    print('blocks:',results['nblocks'],'accepted_edges:',results['accepted_overlap_edges'],
          'components:',results['alignment_components'],'conflicts:',results['nconflicts'])
    for c in results['conflicts']:
        print('\nConflicting cycle:',c.get('conflict_block_a'),'<->',c.get('conflict_block_b'))
        print('Cycle edges:',c.get('cycle_edge_count'),'max mismatch rad:',c.get('max_abs_cycle_mismatch_rad'),
              'epochs >0.1 rad:',c.get('epochs_abs_mismatch_gt_0p1_rad'),
              'max mod 2pi:',c.get('max_mod_2pi_residual_rad'))
        print('Highest mismatch epochs:',c.get('largest_epoch_mismatches',[])[:5])
        for e in c.get('cycle_edges',[]):
            print(('  CONFLICT' if e['is_conflict_edge'] else '  cycle   '),e['name_a'],'<->',e['name_b'],
                  'shared=',e['shared'],'p95=',round(e['p95_rad'],6),'max=',round(e['max_rad'],6))
    if args.duplicate_audit:
        d=results['duplicate_component_audit']
        print('\nPotential mosaic blocker: cross-component duplicate-adjacencies:',d['cross_component_duplicate_adjacent_pairs'])
        for x in d['examples'][:5]:print('  ',x)
    print('\nSaved:',dest)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
