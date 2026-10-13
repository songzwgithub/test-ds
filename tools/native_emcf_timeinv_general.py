#!/usr/bin/env python3
"""Streaming temporal inversion of one Native EMCF island (engineering / relative gauge).

Uses ONLY the saved Spurt island IFGs. Subtracts per-IFG wrapped first-point
anchor before solving, so acquisitions are spatially relative to the island's
first saved point. Never recovers physical/global integer gauge. No input edits.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

TAU=2*np.pi

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roi',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--tol-rad',type=float,default=1e-3)
    p.add_argument('--max-bad-point-fraction',type=float,default=0.001)
    p.add_argument('--chunk-points',type=int,default=8192)
    a=p.parse_args(); src=a.roi.resolve(); dest=a.out.resolve()
    if not src.is_dir(): p.error(f'Missing island {src}')
    if dest==src or dest in src.parents or src in dest.parents:p.error('Input/output directories must be disjoint')
    if dest.exists() and any(dest.iterdir()):p.error(f'Refusing nonempty output {dest}')
    if not (0<a.tol_rad<=0.1 and 0<=a.max_bad_point_fraction<=0.1 and a.chunk_points>0):p.error('Invalid tolerance/chunk')
    def arr(name):
        path=src/name
        if not path.is_file():p.error(f'Missing {path}')
        return np.load(path,mmap_mode='r',allow_pickle=False)
    U=arr('unwrapped_ifg_phase_rad.npy'); pairs=np.asarray(arr('time_ifg_pairs_0based.npy'),dtype=np.int64)
    anchor=np.asarray(arr('anchor_offset_rad.npy'),dtype=np.float64)
    ids=arr('original_point_ids.npy')
    if U.ndim!=2 or U.shape[0]!=219:p.error(f'Unexpected IFG matrix shape {U.shape}')
    m,n=U.shape
    if pairs.shape!=(m,2) or anchor.shape!=(m,) or ids.shape!=(n,):p.error('Metadata shape mismatch')
    if not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids))!=n:p.error('Point IDs duplicated or invalid')
    if np.any(pairs<0) or np.any(pairs>=75) or np.any(pairs[:,0]==pairs[:,1]):p.error('Invalid time network')
    if not np.isfinite(anchor).all():p.error('Invalid anchor')
    provenance_file=src/'summary.json'
    if not provenance_file.is_file():p.error('Missing provenance summary')
    provenance=json.loads(provenance_file.read_text())
    if provenance.get('status')!='ROI_EMCF_BENCHMARK_ONLY':p.error('Unexpected solver source status')
    A=np.zeros((m,74),dtype=np.float64)
    for e,(u,v) in enumerate(pairs):
        if u:A[e,u-1]-=1.0
        if v:A[e,v-1]+=1.0
    rank=int(np.linalg.matrix_rank(A))
    if rank!=74:p.error(f'Time graph disconnected: rank={rank}')
    first=np.asarray(U[:,0],dtype=np.float64)
    anchor_error=float(np.max(np.abs(first-anchor)))
    spurt_parity=provenance.get('wrapped_phase_parity_max_rad')
    relfrac=provenance.get('temporal_closure_relative',{}).get('nonzero_integer_closure_fraction')
    G=cho_factor(A.T@A,check_finite=False)
    dest.mkdir(parents=True,exist_ok=True)
    temp=dest/'acquisition_relative_phase_rad.npy.tmp'
    cube=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(75,n))
    cube[0,:]=0.0
    valid=np.zeros(n,dtype=bool)
    sumsq=0.0; good_count=0; bad_examples=[]; max_good_res=0.0; worst_res=0.0
    try:
        for left in range(0,n,a.chunk_points):
            right=min(n,left+a.chunk_points)
            raw=np.asarray(U[:,left:right],dtype=np.float64)
            finite=np.isfinite(raw).all(axis=0)
            block=np.zeros_like(raw)
            block[:,finite]=raw[:,finite]-anchor[:,None]
            coef=cho_solve(G,A.T@block,check_finite=False)
            res=A@coef-block
            perpoint=np.max(np.abs(res),axis=0)
            good=finite & (perpoint<=a.tol_rad)
            valid[left:right]=good
            cube[1:,left:right]=np.where(good[None,:],coef,np.nan).astype(np.float32)
            if not good.all() and len(bad_examples)<30:
                for j in np.flatnonzero(~good)[:30-len(bad_examples)]:
                    bad_examples.append({'local_index':int(left+j),'original_point_id':int(ids[left+j]),'max_residual_rad':float(perpoint[j]) if finite[j] else None})
            if good.any():
                sumsq+=float(np.sum(res[:,good]**2))
                good_count+=int(good.sum())*m
                max_good_res=max(max_good_res,float(np.max(perpoint[good])))
            worst_res=max(worst_res,float(np.max(perpoint[finite],initial=0)))
            del raw,block,coef,res
        cube.flush();del cube
        nbad=int((~valid).sum());frac=nbad/n
        reasons=[]
        if anchor_error>a.tol_rad:reasons.append(f'First point does not match wrapped anchor ({anchor_error})')
        if spurt_parity is None or spurt_parity>a.tol_rad:reasons.append('Source wrapped phase parity missing/over threshold')
        if relfrac is None or relfrac>a.max_bad_point_fraction:reasons.append('Source relative temporal closure over threshold')
        if frac>a.max_bad_point_fraction:reasons.append(f'Too many bad point time series {nbad}/{n} ({frac:.6%})')
        if not valid[0]:reasons.append('Island first reference point failed')
        status='GO_FULLRES_ISLAND_RELATIVE' if not reasons else 'STOP_ISLAND_TIMEINV'
        qa={'status':status,'source':str(src),'output':str(dest),'n_dates':75,'n_ifg':m,'npoint':n,
            'time_graph_rank':rank,'accepted_points':int(valid.sum()),'bad_points':nbad,
            'bad_point_fraction':frac,'bad_examples':bad_examples,'first_point_anchor_residual_rad':anchor_error,
            'valid_ifg_reconstruction_max_rad':max_good_res,'valid_ifg_reconstruction_rms_rad':float(np.sqrt(sumsq/good_count)) if good_count else None,
            'worst_raw_point_residual_rad':worst_res,'source_wrapped_parity_max_rad':spurt_parity,
            'source_relative_closure_fraction':relfrac,'failed_conditions':reasons,
            'spatial_gauge':'island first saved point per acquisition',
            'temporal_gauge':'acquisition index 0 equals zero',
            'global_integer_gauge_determined':False,'cross_island_gauge_determined':False,
            'absolute_physical_deformation_validated':False,'production_modified':False}
        (dest/'qa_summary.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
        np.save(dest/'valid_points.npy',valid,allow_pickle=False)
        if reasons:
            temp.unlink(missing_ok=True)
        else:
            temp.replace(dest/'acquisition_relative_phase_rad.npy')
        print(json.dumps({k:v for k,v in qa.items() if k!='bad_examples'},indent=2))
        return 0 if not reasons else 2
    except Exception:
        if 'cube' in locals(): del cube
        temp.unlink(missing_ok=True)
        raise

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,np.linalg.LinAlgError) as e:
        print(f'STOP: {e}',file=sys.stderr);sys.exit(2)
