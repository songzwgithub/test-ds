#!/usr/bin/env python3
"""Finish outstanding Ningbo Native EMCF islands without reprocessing successes.

  - Remaining original raw islands: cached Spurt EMCF, <= configured 400k points.
  - Any raw-valid/timeinv-not-valid island: 75-acquisition modal-medoid inversion.
  - Bad points are masked, NEVER integer-cycle-corrected or interpolated.
  - Stage outputs in isolated directories, publish only after QA; backup old bad timeinv.
  - Preserve all GO original island products; recompute tile status from the actual files.

This is ENGINEERING-RELATIVE temporal phase only, not geodetically stable reference.
Do not run alongside another writer to this fullregion directory.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np

RUN = Path('/mnt/ningbo_process/ds/output/diagnostics/native_emcf_fullregion_engineering')


def readj(p):
    try: return json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError): return None


def writej(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def good_raw(p):
    q = readj(p/'summary.json') or {}
    return q.get('status') == 'ROI_EMCF_BENCHMARK_ONLY' and all((p/x).is_file() for x in (
        'unwrapped_ifg_phase_rad.npy', 'time_ifg_pairs_0based.npy',
        'anchor_offset_rad.npy', 'original_point_ids.npy'))


def good_inv(p):
    q = readj(p/'qa_summary.json') or {}
    return q.get('status') == 'GO_FULLRES_ISLAND_RELATIVE' and all((p/x).is_file() for x in (
        'acquisition_relative_phase_rad.npy', 'valid_points.npy'))


def validate_raw(p, npoint):
    if not good_raw(p): return False, 'RAW_QA_MISSING'
    try:
        z = readj(p/'summary.json') or {}
        ph = np.load(p/'unwrapped_ifg_phase_rad.npy', mmap_mode='r', allow_pickle=False)
        ids = np.load(p/'original_point_ids.npy', mmap_mode='r', allow_pickle=False)
        pairs = np.load(p/'time_ifg_pairs_0based.npy', mmap_mode='r', allow_pickle=False)
        if ph.shape != (219, npoint) or ids.shape != (npoint,) or pairs.shape != (219, 2):
            return False, 'RAW_SHAPE'
        if z.get('selected_island',{}).get('points') != npoint: return False, 'RAW_POINT_COUNT'
        if z.get('wrapped_phase_parity_max_rad', 1) > 1e-3: return False, 'RAW_PARITY'
        return True, 'OK'
    except (OSError,ValueError,KeyError) as e:
        return False, 'RAW_READ_'+repr(e)


def validate_inv(p, npoint):
    if not good_inv(p): return False, 'INVERSION_QA_MISSING'
    try:
        q = readj(p/'qa_summary.json') or {}
        v = np.load(p/'valid_points.npy', mmap_mode='r', allow_pickle=False)
        u = np.load(p/'acquisition_relative_phase_rad.npy', mmap_mode='r', allow_pickle=False)
        if v.shape != (npoint,) or v.dtype != np.bool_ or u.shape != (75,npoint):
            return False, 'INVERSION_SHAPE'
        ng = int(v.sum())
        if q.get('npoint') != npoint or ng != q.get('accepted_points') or ng < 3:
            return False, 'INVERSION_COUNT'
        if q.get('valid_ifg_reconstruction_max_rad', float('inf')) > 1e-3:
            return False, 'INVERSION_RESIDUAL'
        if q.get('time_graph_rank') != 74 or q.get('global_integer_gauge_determined') is not False:
            return False, 'INVERSION_GAUGE'
        # Verify all masked 75 epochs; strict checks in chunks limit RAM use.
        for start in range(0,npoint,8192):
            end=min(start+8192,npoint)
            mask=np.asarray(v[start:end]); arr=np.asarray(u[:,start:end])
            if mask.any():
                if not np.isfinite(arr[:,mask]).all() or np.max(np.abs(arr[0,mask]))>1e-5:
                    return False, 'GOOD_POINT_NOT_FINITE_OR_EPOCH0'
            if (~mask).any() and not np.isnan(arr[:,~mask]).all():
                return False, 'BAD_POINT_NOT_FULLY_MASKED'
        return True, 'OK'
    except (OSError, ValueError, TypeError) as e:
        return False, 'INVERSION_READ_'+repr(e)


def tile_record(run, td, a):
    plan = readj(td/'plan.json')
    if plan is not None:return plan
    # Recover tile metadata after an interrupted run (only two such tiles known).
    meta = readj(td/'roi_cache'/'cache_meta.json') or {}
    if 'tile' in meta:
        tile=meta['tile']
    else:
        config=readj(run/'pipeline_config.json') or {}
        rr,cc=(int(td.name.split('_')[0][1:]),int(td.name.split('_')[1][1:]))
        tile={'id':td.name,'row0':rr,'col0':cc,
              'height':min(int(config.get('tile',1024)),int(config.get('rows',12225))-rr),
              'width':min(int(config.get('tile',1024)),int(config.get('cols',44842))-cc)}
    print(f'[PLAN] Missing plan for {td.name}, rebuilding only this tile (no EMCF)',flush=True)
    sys.path.insert(0,str(a.repo/'tools'))
    import native_emcf_fullregion_run as old
    from types import SimpleNamespace
    args=SimpleNamespace(root=a.root,max_points=a.max_points,max_edge_m=40.0)
    di=old.discover(args,tile,td)
    plan={'tile':tile,'input_points':di['input_points'],'unused_points':di['unused_points'],
          'valid_triangle_points':di['valid_triangle_points'],
          'islands':di['islands'],'min_points':32,'max_edge_m':40.0,'max_points':a.max_points}
    writej(td/'plan.json',plan)
    return plan


def log_subprocess(cmd, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    t0=time.monotonic()
    with log.open('w',encoding='utf-8') as f:
        f.write('COMMAND: '+' '.join(map(str,cmd))+'\n');f.flush()
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
        f.write(f'\nEXIT CODE: {r.returncode}\n')
    return r.returncode,round(time.monotonic()-t0,2)


def reserve_stage(path):
    if path.is_dir():
        return not any(path.iterdir())
    return not path.exists()


def archive_name(loc,prefix):
    for k in range(1000):
        d=loc/f'{prefix}_backup_{datetime.now():%Y%m%d_%H%M%S}_{k:03d}'
        if not d.exists():return d
    raise RuntimeError('Too many backup folders')


def publish_stage(stage, dest, known_good):
    if known_good(dest):return 'ALREADY_GOOD'
    if not known_good(stage):return 'STAGE_NOT_VERIFIED'
    if dest.exists():
        archive=archive_name(dest.parent,dest.name)
        os.replace(dest,archive)
    os.replace(stage,dest)
    qa=dest/'qa_summary.json'
    q=readj(qa)
    if q is not None and 'output' in q:
        q['output']=str(dest)
        writej(qa,q)
    return 'PROMOTED'


def compute_raw(task,a):
    td,rank,n,plan=task
    loc=td/f'island_{rank:05d}';raw=loc/'raw'
    if good_raw(raw):return (task,True,'RAW_REUSED',0)
    stage=loc/'recovery_raw_max400k_v1'
    if good_raw(stage):
        valid,why=validate_raw(stage,n)
        if valid:return(task,publish_stage(stage,raw,good_raw)=='PROMOTED','RAW_PUBLISHED_FROM_STAGE',0)
        return (task,False,why,0)
    if not reserve_stage(stage):return(task,False,'PARTIAL_RAW_STAGE_EXISTS_CHECK_LOG',0)
    tile=plan['tile']
    cmd=[sys.executable,str(a.repo/'tools'/'native_emcf_cached_worker.py'),
         '--cache',str(td/'roi_cache'),'--',
         '--config',str(a.root/'pypsds.yaml'),
         '--row0',str(tile['row0']),'--col0',str(tile['col0']),
         '--height',str(tile['height']),'--width',str(tile['width']),
         '--subset','combined','--max-edge-m','40','--max-points',str(a.max_points),
         '--island-rank',str(rank),'--max-run-points',str(a.max_run_points),
         '--enable-hole-faces','--workers',str(a.workers),'--links-per-batch',str(a.links_per_batch),
         '--output-dir',str(stage)]
    env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    rc,dt=log_subprocess(cmd,loc/'logs'/'recovery_emcf.log',env)
    valid,why=validate_raw(stage,n)
    if rc or not valid:return(task,False,f'EMCF_EXIT_{rc}_{why}',dt)
    result=publish_stage(stage,raw,good_raw)
    return(task,result in ('PROMOTED','ALREADY_GOOD'),result,dt)


def compute_inv(task,a):
    td,rank,n,plan=task
    loc=td/f'island_{rank:05d}';raw=loc/'raw';dest=loc/'timeinv'
    if good_inv(dest):return(task,True,'TIMEINV_REUSED',0)
    if not good_raw(raw):return(task,False,'RAW_NOT_GOOD',0)
    stage=loc/'recovery_optref_v1'
    if stage.exists() and any(stage.iterdir()):
        valid,why=validate_inv(stage,n)
        if not valid:return(task,False,f'OPTREF_STAGE_NOT_GOOD_{why}',0)
    else:
        cmd=[sys.executable,str(a.repo/'tools'/'native_emcf_timeinv_optref.py'),
             '--roi',str(raw),'--out',str(stage)]
        env=os.environ.copy()
        for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
        rc,dt=log_subprocess(cmd,loc/'logs'/'recovery_optref.log',env)
        valid,why=validate_inv(stage,n)
        if rc or not valid:return(task,False,f'OPTREF_EXIT_{rc}_{why}',dt)
    result=publish_stage(stage,dest,good_inv)
    return(task,result in ('PROMOTED','ALREADY_GOOD'),result,0)


def refresh_status(run,plans):
    tiles=Counter();types=Counter();completed=0;remaining=[]
    for td,plan in plans:
        chosen=[x for x in plan['islands'] if x['chosen']]
        done=0;fails=[]
        for x in chosen:
            rank=int(x['rank']);n=int(x['points']);loc=td/f'island_{rank:05d}'
            if good_raw(loc/'raw') and good_inv(loc/'timeinv'):
                done+=1
            else:
                stage='raw_missing_or_failed' if not good_raw(loc/'raw') else 'timeinv_missing_or_failed'
                fails.append({'rank':rank,'points':n,'stage':stage});types[stage]+=1
                if len(remaining)<30:remaining.append(f'{td.name} island_{rank:05d}: {stage}')
        status='DONE' if not fails else 'PARTIAL_FAILURE'
        old=readj(td/'run_status.json') or {}
        summary={'tile':td.name,'status':status,'points':plan.get('input_points',0),
                 'available_triangular_points':plan.get('valid_triangle_points',0),
                 'untriangulated_points':plan.get('unused_points',0),
                 'islands':len(plan['islands']),'selected_islands':len(chosen),
                 'skipped_small_points':sum(int(x['points']) for x in plan['islands'] if not x['chosen']),
                 'completed_islands':done,'failures':fails,
                 'previous_status':old.get('status'),'recovered_by':'finish_ningbo_fullres_v1'}
        writej(td/'run_status.json',summary)
        completed+=done;tiles[status]+=1
    return {'tile_status_counts':dict(tiles),'completed_islands':completed,
            'remaining_by_stage':dict(types),'remaining_samples':remaining}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--root',type=Path,default=Path('/mnt/ningbo_process/ds'))
    p.add_argument('--repo',type=Path,default=Path('/mnt/ningbo_process/ds/test-ds-main'))
    p.add_argument('--jobs-time',type=int,default=8)
    p.add_argument('--jobs-spatial',type=int,default=2)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-run-points',type=int,default=400000)
    p.add_argument('--max-points',type=int,default=400000)
    p.add_argument('--links-per-batch',type=int,default=1024)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--stitch-after',action='store_true',help='After all 700 tiles DONE, run overlap QA and out-of-core mosaic if conflict-free')
    a=p.parse_args()
    a.run=a.run.resolve();a.root=a.root.resolve();a.repo=a.repo.resolve()
    if not (1<=a.jobs_time<=16 and 1<=a.jobs_spatial<=4 and 1<=a.workers<=4):
        p.error('Require jobs-time 1..16, jobs-spatial 1..4, workers 1..4')
    if a.max_run_points < 306000 or a.max_points < a.max_run_points:
        p.error('Point caps must cover largest known 305840-point island; max-points>=max-run-points')
    for name in ('native_emcf_cached_worker.py','native_emcf_timeinv_optref.py','native_emcf_fullregion_run.py'):
        if not (a.repo/'tools'/name).is_file():p.error('Missing '+name)
    if not (a.run/'tiles').is_dir():p.error('Missing run tiles')
    if a.stitch_after and not (a.repo/'tools'/'native_emcf_fullregion_stitch.py').is_file():
        p.error('Missing tools/native_emcf_fullregion_stitch.py')
    lock=a.run/'.fullregion_finish.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another finisher is using this run')
        dirs=sorted(t for t in (a.run/'tiles').iterdir() if t.is_dir() and t.name.startswith('r') and '_c' in t.name)
        plans=[];tasks=[];plan_errors=[]
        for td in dirs:
            try:
                plan=tile_record(a.run,td,a) if a.apply else readj(td/'plan.json')
            except Exception as exc:
                plan_errors.append({'tile':td.name,'reason':repr(exc)})
                print(f'[PLAN_FAILURE] {td.name}: {exc!r}',flush=True)
                continue
            if plan is None:
                print(f'[PLAN_MISSING] {td.name} (will generate with --apply)',flush=True)
                continue
            plans.append((td,plan))
            for x in plan['islands']:
                if not x['chosen']:continue
                r=int(x['rank']);n=int(x['points']);loc=td/f'island_{r:05d}'
                if good_raw(loc/'raw') and good_inv(loc/'timeinv'):continue
                tasks.append((td,r,n,plan))
        raw_tasks=[x for x in tasks if not good_raw(x[0]/f'island_{x[1]:05d}'/'raw')]
        only_inv=[x for x in tasks if good_raw(x[0]/f'island_{x[1]:05d}'/'raw')]
        print(json.dumps({'mode':'APPLY' if a.apply else 'PLAN_ONLY','tile_plans':len(plans),
           'unfinished_tasks':len(tasks),'plan_failures':plan_errors,'need_spatial_emcf':len(raw_tasks),
           'reuse_raw_timeinv_only':len(only_inv),'jobs_time':a.jobs_time,
           'jobs_spatial':a.jobs_spatial,'max_run_points':a.max_run_points},ensure_ascii=False,indent=2),flush=True)
        if not a.apply:return 0
        reports=[];counts=Counter();newly_raw=[]
        # Start with existing raw successes. Expensive spatial work separately capped.
        with futures.ThreadPoolExecutor(max_workers=a.jobs_time) as timepool, futures.ThreadPoolExecutor(max_workers=a.jobs_spatial) as rawpool:
            # Launch spatial and time-inversion pools concurrently. Futures are explicitly awaited.
            # Submitting direct work avoids nesting pools and deadlocks.
            sf={rawpool.submit(compute_raw,x,a):x for x in sorted(raw_tasks,key=lambda z:-z[2])}
            tf={timepool.submit(compute_inv,x,a):x for x in only_inv}
            for ix,fut in enumerate(futures.as_completed(tf),1):
                task=tf[fut]
                try:_,ok,reason,dt=fut.result()
                except BaseException as exc:ok=False;reason=repr(exc);dt=0
                counts['TIMEINV_'+('SUCCESS' if ok else 'FAILED')]+=1
                reports.append({'tile':task[0].name,'rank':task[1],'npoint':task[2],
                                'stage':'TIMEINV','ok':ok,'reason':reason,'seconds':dt})
                if ix%25==0 or ix==len(tf):print(f'[TIMEINV] {ix}/{len(tf)} good={counts["TIMEINV_SUCCESS"]} failed={counts["TIMEINV_FAILED"]}',flush=True)
            for ix,fut in enumerate(futures.as_completed(sf),1):
                task=sf[fut]
                try:_,ok,reason,dt=fut.result()
                except BaseException as exc:ok=False;reason=repr(exc);dt=0
                counts['SPATIAL_'+('SUCCESS' if ok else 'FAILED')]+=1
                reports.append({'tile':task[0].name,'rank':task[1],'npoint':task[2],
                                'stage':'SPATIAL','ok':ok,'reason':reason,'seconds':dt})
                if ok:
                    # As soon as a large raw island finishes, queue inversion onto time pool.
                    tf2=timepool.submit(compute_inv,task,a)
                    newly_raw.append((task,tf2))
                if ix%5==0 or ix==len(sf):print(f'[SPATIAL] {ix}/{len(sf)} good={counts["SPATIAL_SUCCESS"]} failed={counts["SPATIAL_FAILED"]}',flush=True)
            for ix,(task,fut) in enumerate(newly_raw,1):
                try:_,ok,reason,dt=fut.result()
                except BaseException as exc:ok=False;reason=repr(exc);dt=0
                counts['LARGE_TIMEINV_'+('SUCCESS' if ok else 'FAILED')]+=1
                reports.append({'tile':task[0].name,'rank':task[1],'npoint':task[2],
                                'stage':'LARGE_TIMEINV','ok':ok,'reason':reason,'seconds':dt})
                if ix%5==0 or ix==len(newly_raw):print(f'[LARGE_TIMEINV] {ix}/{len(newly_raw)} good={counts["LARGE_TIMEINV_SUCCESS"]} failed={counts["LARGE_TIMEINV_FAILED"]}',flush=True)
        final=refresh_status(a.run,plans)
        report={'status':'ALL_SELECTED_TASKS_DONE' if final['tile_status_counts'].get('PARTIAL_FAILURE',0)==0 and len(plans)==700 else 'PARTIAL_CHECK_FAILURES',
                'run':str(a.run),'known_tiles':len(plans),'plan_failures':plan_errors,'execution_counts':dict(counts),**final,
                'note':'Computational island-relative phase only. No cross-tile integer-gauge/physical-reference validation.'}
        out=a.run/'fullregion_finish_report.json';writej(out,report)
        detail=a.run/'fullregion_finish_attempts.json';writej(detail,reports)
        if a.stitch_after and report['status']=='ALL_SELECTED_TASKS_DONE':
            dest=a.run.parent/'native_emcf_alignment_finish_v1'
            if dest.exists() and any(dest.iterdir()):
                report['alignment_stage']='STOP_OUTPUT_ALREADY_EXISTS';report['alignment_directory']=str(dest)
            else:
                cmd=[sys.executable,str(a.repo/'tools'/'native_emcf_fullregion_stitch.py'),
                     '--run',str(a.run),'--out',str(dest),'--assemble']
                env=os.environ.copy()
                for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
                rc,dt=log_subprocess(cmd,a.run/'fullregion_finish_stitch.log',env)
                report['alignment_stage']='COMPLETED' if rc==0 else f'FAILED_RC_{rc}'
                report['alignment_directory']=str(dest)
                report['alignment_seconds']=dt
                manifest=readj(dest/'alignment_manifest.json') or {}
                report['alignment_manifest_status']=manifest.get('status')
                report['alignment_components']=manifest.get('disconnected_alignment_components')
                report['alignment_cycle_conflicts']=len(manifest.get('alignment_cycle_conflicts') or [])
            writej(out,report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        print('SAVED',out,detail,flush=True)
        return 0 if report['status']=='ALL_SELECTED_TASKS_DONE' else 2


if __name__ == '__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:print('INTERRUPTED; existing successful checkpoint outputs preserved',file=sys.stderr);sys.exit(130)
