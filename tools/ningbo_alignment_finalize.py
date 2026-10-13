#!/usr/bin/env python3
"""Ningbo Native EMCF: evidence-weighted alignment + masked, out-of-core mosaic.

READ ONLY original 25k island results and prior alignment edges. Does NOT
alter a phase ambiguity, invent graph edges, or claim one physical gauge.

- Maximum-support spanning forest among PREVIOUSLY ACCEPTED overlap edges.
- All non-tree edges must close. A weak integer-only cycle chord can be
  quarantined only when its support is >=10x below the weakest tree-path edge.
- Point IDs in distinct unaligned components are entirely masked.
- Duplicate observations in the SAME aligned component are compared at all
  75 dates: any discrepancy > tolerance masks that original point ID.
- Every mosaic point retains its alignment_component_id; never treats separate
  components as a shared physical reference.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

TAU = 2 * np.pi


def readj(path: Path):
    with path.open('r',encoding='utf-8') as f:
        return json.load(f)


def writej(path: Path, obj):
    p=path.with_suffix(path.suffix+'.tmp')
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(p,path)


class DSU:
    def __init__(self,n):
        self.par=list(range(n));self.size=[1]*n
    def find(self,a):
        while self.par[a]!=a:
            self.par[a]=self.par[self.par[a]];a=self.par[a]
        return a
    def join(self,a,b):
        x=self.find(a);y=self.find(b)
        if x==y:return False
        if self.size[x]<self.size[y]:x,y=y,x
        self.par[y]=x;self.size[x]+=self.size[y]
        return True


def path_edges(adj,start,goal):
    parents={start:None}
    q=deque([start])
    while q and goal not in parents:
        u=q.popleft()
        for v,edge_id in adj[u]:
            if v not in parents:
                parents[v]=(u,edge_id)
                q.append(v)
    if goal not in parents:
        raise RuntimeError('Forest path disconnected: internal error')
    ids=[];u=goal
    while u!=start:
        prev,k=parents[u];ids.append(k);u=prev
    ids.reverse()
    return ids


def weighted_align(blocks, edges, tol=0.1, min_ratio=10.0, max_weak_conflicts=1):
    n=len(blocks)
    dsu=DSU(n)
    # Tie-breaks are deterministic. Stronger support precedes any weaker link.
    ordering=sorted(range(len(edges)),key=lambda k:(-int(edges[k]['shared']),float(edges[k]['p95_rad']),float(edges[k]['max_rad']),k))
    forest=[];chords=[]
    adj=[[] for _ in range(n)]
    for k in ordering:
        e=edges[k];i=int(e['i']);j=int(e['j'])
        if i==j or not 0<=i<n or not 0<=j<n:
            raise ValueError(f'Bad overlap edge {k}')
        v=np.asarray(e['delta_rad'],dtype=np.float64)
        if v.shape!=(75,) or not np.isfinite(v).all():
            raise ValueError(f'Invalid delta array {k}')
        if dsu.join(i,j):
            forest.append(k);adj[i].append((j,k));adj[j].append((i,k))
        else:
            chords.append(k)
    labels=np.full(n,-1,dtype=np.int32)
    offsets=np.zeros((n,75),dtype=np.float64)
    comp=0
    for root in range(n):
        if labels[root]>=0:continue
        labels[root]=comp;q=deque([root])
        while q:
            i=q.popleft()
            for j,k in adj[i]:
                if labels[j]>=0:continue
                e=edges[k];i0=int(e['i']);j0=int(e['j']);d=np.asarray(e['delta_rad'],dtype=np.float64)
                if i==i0 and j==j0:offsets[j]=offsets[i]+d
                elif i==j0 and j==i0:offsets[j]=offsets[i]-d
                else:raise RuntimeError('Invalid tree edge orientation')
                labels[j]=comp;q.append(j)
        comp+=1
    quarantine=[];fatal=[];worst_ok=0.0
    for k in chords:
        e=edges[k];i=int(e['i']);j=int(e['j']);delta=np.asarray(e['delta_rad'],dtype=np.float64)
        mismatch=offsets[j]-offsets[i]-delta
        maxrad=float(np.max(np.abs(mismatch)))
        if maxrad<=tol:
            worst_ok=max(worst_ok,maxrad);continue
        modulo=mismatch-TAU*np.rint(mismatch/TAU)
        maxmod=float(np.max(np.abs(modulo)))
        pe=path_edges(adj,i,j)
        min_tree_shared=min((int(edges[t]['shared']) for t in pe),default=0)
        shared=int(e['shared'])
        rec={'edge_index':k,'name_i':blocks[i]['name'],'name_j':blocks[j]['name'],
             'shared':shared,'weakest_tree_path_shared':min_tree_shared,
             'tree_path_edge_count':len(pe),'max_mismatch_rad':maxrad,
             'max_mod_2pi_rad':maxmod,'mismatch_epoch_indices':np.flatnonzero(np.abs(mismatch)>tol).tolist(),
             'integer_cycle_values':np.rint(mismatch/TAU).astype(int).tolist()}
        if maxmod<=tol and min_tree_shared>=min_ratio*shared:
            quarantine.append(rec)
        else:
            fatal.append(rec)
    if len(quarantine)>max_weak_conflicts:
        fatal.extend(quarantine);quarantine=[]
    return labels,offsets,{'component_count':comp,'forest_edges':len(forest),
       'consistent_chords':len(chords)-len(quarantine)-len(fatal),
       'quarantined_weak_integer_links':quarantine,'fatal_inconsistencies':fatal,
       'max_consistent_cycle_residual_rad':worst_ok,'forest_edges_indices':forest}


def collect_id_partition(blocks,labels):
    """Count unique IDs and identify IDs spanning incompatible components."""
    fragments=[];owners=[]
    for i,b in enumerate(blocks):
        base=Path(b['base'])
        raw=base/'raw';timeinv=base/'timeinv'
        arr=np.load(raw/'original_point_ids.npy',mmap_mode='r',allow_pickle=False)
        valid=np.load(timeinv/'valid_points.npy',mmap_mode='r',allow_pickle=False)
        if arr.ndim!=1 or valid.shape!=arr.shape:
            raise RuntimeError('ID/valid mismatch: '+str(base))
        ids=np.asarray(arr[np.asarray(valid,dtype=bool)],dtype=np.int64)
        if len(ids)!=int(b['npoint']):
            raise RuntimeError('Manifest point count mismatch: '+str(base))
        if not len(ids):continue
        fragments.append(ids)
        owners.append(np.full(len(ids),int(labels[i]),dtype=np.int32))
        if (i+1)%5000==0:
            print(f'[PREFLIGHT] point IDs {i+1}/{len(blocks)} blocks',flush=True)
    ids=np.concatenate(fragments)
    labs=np.concatenate(owners)
    del fragments,owners
    perm=np.argsort(ids,kind='stable')
    sids=ids[perm];slabs=labs[perm]
    del ids,labs,perm
    first=np.r_[True,sids[1:]!=sids[:-1]]
    idx=np.flatnonzero(first)
    unique=sids[idx].copy()
    low=np.minimum.reduceat(slabs,idx)
    high=np.maximum.reduceat(slabs,idx)
    ambiguous=low!=high
    inst=len(sids);n=len(unique)
    del sids,slabs,low,high,idx,first
    return unique,ambiguous,{'point_instances':int(inst),'unique_point_ids':int(n),
      'duplicate_point_instances':int(inst-n),
      'cross_component_ambiguous_unique_points':int(np.count_nonzero(ambiguous))}


def assemble(blocks,labels,offsets,out,unique,ambiguous,duplicate_tol=0.05,batch=4096):
    from numpy.lib.format import open_memmap
    n=len(unique)
    out.mkdir(exist_ok=False)
    np.save(out/'original_point_ids.npy',unique,allow_pickle=False)
    phase=open_memmap(out/'acquisition_phase_by_point_rad.npy',mode='w+',dtype='float32',shape=(n,75))
    row=open_memmap(out/'radar_row.npy',mode='w+',dtype='int32',shape=(n,))
    col=open_memmap(out/'radar_col.npy',mode='w+',dtype='int32',shape=(n,))
    kind=open_memmap(out/'point_type.npy',mode='w+',dtype='uint8',shape=(n,))
    comp=open_memmap(out/'alignment_component_id.npy',mode='w+',dtype='int32',shape=(n,))
    qflag=open_memmap(out/'point_quality_flag.npy',mode='w+',dtype='uint8',shape=(n,))
    filled=np.zeros(n,dtype=bool)
    # Quality code: 0 = good; 1 = duplicated across unaligned components;
    # 2 = within-component duplicate disagrees at an epoch.
    qflag[:]=ambiguous.astype(np.uint8)
    comp[:]=-1
    for beg in range(0,n,100000):
        phase[beg:min(n,beg+100000),:]=np.nan
    duplicate_compared=0;duplicate_mismatch_instances=0
    for bi,b in enumerate(blocks):
        base=Path(b['base']);r=base/'raw';t=base/'timeinv'
        ids_all=np.load(r/'original_point_ids.npy',mmap_mode='r',allow_pickle=False)
        valid=np.load(t/'valid_points.npy',mmap_mode='r',allow_pickle=False)
        good_idx=np.flatnonzero(valid)
        ids=np.asarray(ids_all[good_idx],dtype=np.int64)
        targets=np.searchsorted(unique,ids)
        if np.any(targets>=n) or not np.array_equal(unique[targets],ids):
            raise RuntimeError('Point-ID mapping failure for '+str(base))
        u=np.load(t/'acquisition_relative_phase_rad.npy',mmap_mode='r',allow_pickle=False)
        rr=np.load(r/'original_radar_rows.npy',mmap_mode='r',allow_pickle=False)
        cc=np.load(r/'original_radar_cols.npy',mmap_mode='r',allow_pickle=False)
        typ=np.load(r/'original_point_type.npy',mmap_mode='r',allow_pickle=False)
        if u.shape!=(75,len(ids_all)) or rr.shape!=ids_all.shape or cc.shape!=ids_all.shape or typ.shape!=ids_all.shape:
            raise RuntimeError('Source dimensions mismatch '+str(base))
        off=offsets[bi]
        for start in range(0,len(ids),batch):
            end=min(len(ids),start+batch)
            ix=good_idx[start:end];dest=targets[start:end]
            new=~filled[dest]
            if np.any(new):
                to=dest[new];fromidx=ix[new]
                row[to]=rr[fromidx];col[to]=cc[fromidx];kind[to]=typ[fromidx]
                has_gauge=~ambiguous[to]
                if np.any(has_gauge):
                    tg=to[has_gauge];sg=fromidx[has_gauge]
                    z=np.asarray(u[:,sg],dtype=np.float64).T+off
                    if not np.isfinite(z).all():raise RuntimeError('Nonfinite accepted source phase '+str(base))
                    phase[tg,:]=z.astype(np.float32)
                    comp[tg]=labels[bi]
                filled[to]=True
            old=~new & ~ambiguous[dest] & (qflag[dest]==0)
            if np.any(old):
                to=dest[old];fromidx=ix[old]
                if np.any(comp[to]!=labels[bi]):
                    raise RuntimeError('Ambiguous ID unexpectedly not masked')
                z=np.asarray(u[:,fromidx],dtype=np.float64).T+off
                ref=np.asarray(phase[to,:],dtype=np.float64)
                bad=(~np.isfinite(z).all(axis=1)) | (np.max(np.abs(z-ref),axis=1)>duplicate_tol)
                duplicate_compared+=len(to)
                duplicate_mismatch_instances+=int(np.count_nonzero(bad))
                if np.any(bad):qflag[to[bad]]=2
        if (bi+1)%1000==0 or bi+1==len(blocks):
            print(f'[ASSEMBLE] blocks={bi+1}/{len(blocks)} duplicate_checked={duplicate_compared} mismatches={duplicate_mismatch_instances}',flush=True)
    if not np.all(filled):raise RuntimeError('One or more unique IDs were never assigned')
    invalid=np.asarray(qflag,dtype=np.uint8)!=0
    for beg in range(0,n,100000):
        end=min(n,beg+100000)
        bad=np.flatnonzero(invalid[beg:end])+beg
        if len(bad):phase[bad,:]=np.nan;comp[bad]=-1
    v=open_memmap(out/'valid_points.npy',mode='w+',dtype='bool',shape=(n,))
    v[:]=~invalid
    for a in (phase,row,col,kind,comp,qflag,v):a.flush()
    return {'point_instances_compared':duplicate_compared,
      'duplicate_observations_disagreeing':duplicate_mismatch_instances,
      'within_component_duplicate_ambiguous_unique_points':int(np.count_nonzero(qflag==2)),
      'cross_component_ambiguous_unique_points':int(np.count_nonzero(qflag==1)),
      'valid_unique_points':int(np.count_nonzero(~invalid)),
      'masked_unique_points':int(np.count_nonzero(invalid)),
      'mosaic_shape_point_by_epoch':[int(n),75],
      'quality_codes':{'0':'valid locally referenced point','1':'same original ID in incompatible alignment components','2':'same original ID has conflicting time series within one component'},
      'gauge_note':'Each point is relative to alignment_component_id. Different components cannot be compared as a common physical datum.'}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--alignment',type=Path,required=True,help='Existing native_emcf_alignment_finish_v1 dir')
    ap.add_argument('--out',type=Path,required=True,help='Brand-new output directory')
    ap.add_argument('--assemble',action='store_true')
    ap.add_argument('--cycle-tol-rad',type=float,default=0.1)
    ap.add_argument('--duplicate-tol-rad',type=float,default=0.05)
    ap.add_argument('--min-support-ratio',type=float,default=10)
    ap.add_argument('--max-weak-conflicts',type=int,default=1)
    a=ap.parse_args(argv)
    src=a.alignment.resolve();out=a.out.resolve()
    if out.exists():raise RuntimeError('Output path already exists. Pick a NEW directory: '+str(out))
    if src==out or src in out.parents or out in src.parents:raise RuntimeError('Source and destination must be isolated')
    if not 0<a.cycle_tol_rad<=0.5 or not 0<a.duplicate_tol_rad<=0.5:
        raise ValueError('Tolerances must be within (0,0.5] rad')
    manifest=readj(src/'alignment_manifest.json');edges=readj(src/'alignment_edges.json');blocks=manifest['blocks']
    if len(blocks)!=int(manifest['nblocks']):raise RuntimeError('Malformed manifest')
    print(f'[GRAPH] {len(blocks)} blocks, {len(edges)} existing accepted edges',flush=True)
    labels,offsets,diag=weighted_align(blocks,edges,a.cycle_tol_rad,a.min_support_ratio,a.max_weak_conflicts)
    brief={k:v for k,v in diag.items() if k!='forest_edges_indices'}
    print('[GRAPH]',json.dumps(brief,ensure_ascii=False),flush=True)
    out.mkdir(parents=True,exist_ok=False)
    np.save(out/'block_alignment_component_id.npy',labels,allow_pickle=False)
    np.save(out/'block_time_offset_rad.npy',offsets.astype(np.float32),allow_pickle=False)
    report={'status':'GRAPH_STOP_CONFLICT' if diag['fatal_inconsistencies'] else 'RELATIVE_GRAPH_QA_PASS',
      'input_alignment':str(src),'nblocks':len(blocks),'input_edges':len(edges),
      'component_count':diag['component_count'],'graph':brief,
      'zero_global_physical_gauge_verified':False,
      'phase_integer_modification_applied':False,
      'source_products_modified':False}
    writej(out/'alignment_manifest.json',report)
    if diag['fatal_inconsistencies']:
        print('STOP: conflict is not a uniquely weak integer-cycle chord. No mosaic assembled.',file=sys.stderr)
        return 2
    if not a.assemble:
        print('[FINISH] audit only; use --assemble with a NEW --out directory for actual mosaic.')
        return 0
    unique,ambiguous,counts=collect_id_partition(blocks,labels)
    print('[IDENTITY]',counts,flush=True)
    expected=len(unique)*(75*4+8+4+4+1+4+1+1)+len(unique)*2
    free=shutil.disk_usage(out).free
    if free<1.3*expected:
        report['status']='STOP_DISK_SPACE';report['disk_required_gib']=round(1.3*expected/2**30,2)
        writej(out/'alignment_manifest.json',report)
        print('STOP: insufficient free disk for out-of-core mosaic',file=sys.stderr)
        return 2
    t0=time.monotonic()
    stat=assemble(blocks,labels,offsets,out/'mosaic',unique,ambiguous,a.duplicate_tol_rad)
    report['status']='RELATIVE_MOSAIC_QA_FILTERED_WITH_COMPONENT_IDS'
    report['point_identity']=counts;report['mosaic']=stat
    report['assembly_seconds']=round(time.monotonic()-t0,2)
    writej(out/'alignment_manifest.json',report)
    print('[FINISH]',json.dumps({'status':report['status'],**counts,**stat,'alignment_components':diag['component_count']},ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        print('STOP:',repr(e),file=sys.stderr)
        sys.exit(2)
