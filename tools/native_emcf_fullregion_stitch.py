#!/usr/bin/env python3
"""Overlap-only alignment and optional out-of-core mosaic of Native EMCF islands.

Connects ONLY blocks with identical ORIGINAL PS/DS point IDs in geometric
ROI overlap; never interpolates or bridges disconnected components. An accepted
edge constrains per-epoch RELATIVE phase offset, NOT absolute physical motion.
"""
from __future__ import annotations
import argparse,json,os,shutil,sys
from collections import defaultdict,deque
from pathlib import Path
import numpy as np

def write_json(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')

def find_blocks(root,require_complete):
    cfg=json.loads((root/'pipeline_config.json').read_text())
    def count_tiles(size):
        tile=cfg['tile'];ov=cfg['overlap']
        if size<=tile:return 1
        starts=list(range(0,size-tile+1,tile-ov))
        return len(starts)+(starts[-1] != size-tile)
    nr=count_tiles(cfg['rows'])
    nc=count_tiles(cfg['cols'])
    done=[];other=[];blocks=[]
    for t in sorted((root/'tiles').glob('r*_c*')):
        s=t/'run_status.json'
        if not s.exists():other.append(t.name+':missing-run-status');continue
        status=json.loads(s.read_text())
        if status.get('status')!='DONE':other.append(t.name+':'+status.get('status','unknown'))
        else:done.append(t.name)
        for q in sorted(t.glob('island_*/timeinv/qa_summary.json')):
            meta=json.loads(q.read_text())
            if meta.get('status')!='GO_FULLRES_ISLAND_RELATIVE':continue
            base=q.parent.parent
            if not (base/'raw/original_point_ids.npy').is_file() or not (q.parent/'acquisition_relative_phase_rad.npy').is_file():continue
            raw=base/'raw'
            ids=np.load(raw/'original_point_ids.npy',mmap_mode='r')
            valid=np.load(q.parent/'valid_points.npy',mmap_mode='r')
            if len(ids)!=len(valid):raise RuntimeError('Bad point alignment '+str(base))
            coords_r=np.load(raw/'original_radar_rows.npy',mmap_mode='r')
            coords_c=np.load(raw/'original_radar_cols.npy',mmap_mode='r')
            xx=np.flatnonzero(valid)
            if not len(xx):continue
            bound=(int(coords_r[xx].min()),int(coords_r[xx].max()),int(coords_c[xx].min()),int(coords_c[xx].max()))
            blocks.append({'name':t.name+'/'+base.name,'tile':t.name,'base':base,'npoint':int(len(xx)),
                           'bounds':bound,'ids':np.asarray(ids[xx],dtype=np.int64),'local_index':xx.astype(np.int32)})
    if require_complete and (len(done)!=nr*nc or other):
        raise RuntimeError(f'Incomplete region: {len(done)}/{nr*nc} tiles DONE; {len(other)} non-DONE. Run all tiles or pass --allow-partial for audit only.')
    return cfg,blocks,done,other,nr*nc

def bbox_touch(a,b):
    ra,rb,ca,cb=a;sa,sb,da,db=b
    return ra<=sb and sa<=rb and ca<=db and da<=cb

def time_series(b):
    return np.load(b['base']/'timeinv/acquisition_relative_phase_rad.npy',mmap_mode='r')

def make_edges(blocks,min_shared,max_p95):
    tiles=defaultdict(list)
    for i,b in enumerate(blocks):tiles[b['tile']].append(i)
    names=sorted(tiles)
    edges=[];reject=[];npairs=0
    # Each tile is max 1024x1024. Candidate only if tile names' rectangle overlaps.
    def xy(name):
        q=name.split('_');return int(q[0][1:]),int(q[1][1:])
    # Avoid all-tile O(n^2): group candidate tile keys by offsets computed from names.
    xy_to_name={xy(t):t for t in names}
    rr=sorted(set(r for r,c in xy_to_name));cc=sorted(set(c for r,c in xy_to_name))
    for t in names:
        r,c=xy(t)
        for r2 in rr:
            if r2<r or r2>r+2048:continue
            for c2 in cc:
                if abs(c2-c)>2048:continue
                u=xy_to_name.get((r2,c2))
                if u is None or u<=t:continue
                for i in tiles[t]:
                    for j in tiles[u]:
                        b=blocks[i];d=blocks[j]
                        if not bbox_touch(b['bounds'],d['bounds']):continue
                        shared,ii,jj=np.intersect1d(b['ids'],d['ids'],assume_unique=True,return_indices=True)
                        if len(shared)<min_shared:continue
                        npairs+=1
                        x=b['local_index'][ii];y=d['local_index'][jj]
                        p=time_series(b)[:,x].astype(np.float64)
                        q=time_series(d)[:,y].astype(np.float64)
                        diff=p-q
                        delta=np.median(diff,axis=1)
                        resid=np.abs(diff-delta[:,None]);p95=float(np.quantile(resid,.95));worst=float(np.max(resid))
                        record={'i':i,'j':j,'shared':int(len(shared)),'p95_rad':p95,'max_rad':worst,
                                'delta_rad':delta.tolist()}
                        if p95<=max_p95:edges.append(record)
                        else:reject.append({k:v for k,v in record.items() if k!='delta_rad'})
    return edges,reject,npairs

def align(blocks,edges,cycle_tol):
    neighbors=defaultdict(list)
    for e in edges:
        off=np.asarray(e['delta_rad'])
        neighbors[e['i']].append((e['j'],off))
        neighbors[e['j']].append((e['i'],-off))
    labels=np.full(len(blocks),-1,np.int32)
    offsets=np.zeros((len(blocks),75),dtype=np.float64)
    component=0
    for root in range(len(blocks)):
        if labels[root]>=0:continue
        labels[root]=component
        q=deque([root])
        while q:
            i=q.popleft()
            for j,d in neighbors[i]:
                if labels[j]>=0:continue
                offsets[j]=offsets[i]+d
                labels[j]=component
                q.append(j)
        component+=1
    conflicts=[]
    for e in edges:
        expected=offsets[e['j']]-offsets[e['i']]
        error=float(np.max(np.abs(expected-np.asarray(e['delta_rad']))))
        if error>cycle_tol:conflicts.append({'i':e['i'],'j':e['j'],'max_cycle_mismatch_rad':error})
    return labels,offsets,conflicts

def make_mosaic(a,blocks,labels,offsets,destination):
    # Not a single shared gauge when labels contains multiple values!
    if destination.exists() and any(destination.iterdir()):raise RuntimeError('Refuse to overwrite mosaic directory')
    allids=np.concatenate([b['ids'] for b in blocks])
    if not len(allids):raise RuntimeError('No accepted points')
    owner=np.concatenate([np.full(len(b['ids']),int(labels[i]),dtype=np.int32) for i,b in enumerate(blocks)])
    order=np.argsort(allids,kind='stable')
    adjacent=(allids[order][1:]==allids[order][:-1])
    mismatches=int(np.count_nonzero(adjacent & (owner[order][1:]!=owner[order][:-1])))
    if mismatches:
        raise RuntimeError(f'{mismatches} duplicate source IDs belong to UNALIGNED components; refuse ambiguous mosaic. Check rejected overlap edges.')
    del owner,order,adjacent
    unique=np.unique(allids)
    n=len(unique)
    estimated=int(n*(75*4+8+4+4+1+4)+n)  # phase + IDs + radar coord/type/component
    free=shutil.disk_usage(destination.parent).free
    if free < estimated*1.2:
        raise RuntimeError(f'Insufficient disk free: need ~{estimated*1.2/2**30:.1f} GiB, have {free/2**30:.1f} GiB')
    destination.mkdir(parents=True,exist_ok=True)
    np.save(destination/'original_point_ids.npy',unique,allow_pickle=False)
    from numpy.lib.format import open_memmap
    phase=open_memmap(destination/'acquisition_phase_by_point_rad.npy',mode='w+',dtype='float32',shape=(n,75))
    radar_row=open_memmap(destination/'radar_row.npy',mode='w+',dtype='int32',shape=(n,))
    radar_col=open_memmap(destination/'radar_col.npy',mode='w+',dtype='int32',shape=(n,))
    typ=open_memmap(destination/'point_type.npy',mode='w+',dtype='uint8',shape=(n,))
    group=open_memmap(destination/'alignment_component_id.npy',mode='w+',dtype='int32',shape=(n,))
    filled=np.zeros(n,dtype=bool);duplicate=0;unlinked_duplicates=0
    for i,b in enumerate(blocks):
        ids=b['ids'];dst=np.searchsorted(unique,ids)
        if not np.array_equal(unique[dst],ids):raise RuntimeError('Point-ID merge corruption')
        exists=filled[dst]
        if np.any(exists):
            duplicate+=int(exists.sum())
            bad=group[dst[exists]]!=labels[i]
            unlinked_duplicates+=int(bad.sum())
        select=~exists
        if not np.any(select):continue
        source=b['local_index'][select]
        target=dst[select]
        u=time_series(b)[:,source].T.astype(np.float64)
        phase[target,:]=(u+offsets[i]).astype(np.float32)
        raw=b['base']/'raw'
        radar_row[target]=np.load(raw/'original_radar_rows.npy',mmap_mode='r')[source]
        radar_col[target]=np.load(raw/'original_radar_cols.npy',mmap_mode='r')[source]
        typ[target]=np.load(raw/'original_point_type.npy',mmap_mode='r')[source]
        group[target]=labels[i];filled[target]=True
    phase.flush();radar_row.flush();radar_col.flush();typ.flush();group.flush()
    if not filled.all():raise RuntimeError('Unfilled mosaic entries')
    return {'unique_points':int(n),'duplicates_collapsed':duplicate,
            'duplicates_from_unconnected_components':unlinked_duplicates,
            'aligned_spatial_components':int(len(np.unique(labels))),
            'mosaic_phase_shape':[int(n),75],
            'global_absolute_integer_gauge_determined':False,
            'intercomponent_physical_reference_determined':False}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,default=Path('/mnt/ningbo_process/ds/output/diagnostics/native_emcf_fullregion_engineering'))
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--min-shared',type=int,default=8)
    ap.add_argument('--max-overlap-p95-rad',type=float,default=0.05)
    ap.add_argument('--max-alignment-cycle-rad',type=float,default=0.1)
    ap.add_argument('--allow-partial',action='store_true')
    ap.add_argument('--assemble',action='store_true',help='Create out-of-core point x date .npy mosaic; time/space expensive')
    a=ap.parse_args();src=a.run.resolve();dest=(a.out or src/'alignment').resolve()
    if dest==src or src in dest.parents and dest!=src/'alignment':
        # Alignment is allowed as an isolated child; other children are not expected.
        raise RuntimeError('Choose default alignment output or separate directory')
    if dest.exists() and any(dest.iterdir()):raise RuntimeError('Refusing nonempty alignment output')
    cfg,blocks,done,other,total=find_blocks(src,not a.allow_partial)
    if not blocks:raise RuntimeError('No completed island inversions')
    edges,reject,npairs=make_edges(blocks,a.min_shared,a.max_overlap_p95_rad)
    labels,offsets,conflicts=align(blocks,edges,a.max_alignment_cycle_rad)
    dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/'block_time_offset_rad.npy',offsets.astype(np.float32))
    np.save(dest/'block_alignment_component_id.npy',labels)
    manifest={'status':'STOP_ALIGNMENT_CYCLE' if conflicts else 'RELATIVE_ALIGNMENT_GRAPH_AUDITED',
              'blocks':[{'name':b['name'],'base':str(b['base']),'npoint':b['npoint'],'component':int(labels[i])} for i,b in enumerate(blocks)],
              'nblocks':len(blocks),'successful_tiles':len(done),'expected_tiles':total,'noncomplete_tiles':other,
              'candidate_overlap_pairs':npairs,'accepted_overlap_edges':len(edges),'rejected_overlap_pairs':len(reject),
              'disconnected_alignment_components':int(len(set(labels.tolist()))),
              'alignment_cycle_conflicts':conflicts,'rejected_edges':reject,
              'cross_component_physical_gauge_determined':False,'production_modified':False}
    write_json(dest/'alignment_edges.json',edges)
    write_json(dest/'alignment_manifest.json',manifest)
    if conflicts:
        print(json.dumps({k:v for k,v in manifest.items() if k not in ('blocks','rejected_edges')},indent=2))
        raise RuntimeError('Conflicting overlap cycles; mosaic blocked')
    if a.assemble:
        mosaic=make_mosaic(a,blocks,labels,offsets,dest/'mosaic')
        manifest['mosaic']=mosaic
        manifest['status']='RELATIVE_MOSAIC_WITH_COMPONENT_IDS_NOT_PHYSICAL_REFERENCE'
        write_json(dest/'alignment_manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('blocks','rejected_edges')},indent=2))
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        print('STOP:',e,file=sys.stderr);sys.exit(2)
