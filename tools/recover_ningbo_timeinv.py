#!/usr/bin/env python3
"""Conservative checkpoint-preserving recovery of failed Ningbo EMCF island temporal inversion.

DEFAULT MODE IS READ-ONLY. Requires explicitly --apply to compute new sidecar output,
--promote to back up old failed timeinv and atomically publish verified recovery.
Never redoes spatial EMCF, never edits original PointPhaseStack or TP-LA products.
Bad points are NaN-masked, not silently corrected. Large islands are NOT touched.

Uses already installed tools/native_emcf_timeinv_general.py so mathematical test
stays identical; only the whole-island bad-fraction policy may change.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np

DEFAULT_RUN = Path('/mnt/ningbo_process/ds/output/diagnostics/native_emcf_fullregion_engineering')
STAGE = 'timeinv_recovery_v1'
MODULE = None


def loadj(p):
    try:
        return json.loads(p.read_text(encoding='utf8')) if p.is_file() else None
    except (OSError, ValueError):
        return None


def atomic_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')
    os.replace(t, p)


def files_ok(raw):
    z = loadj(raw/'summary.json') or {}
    return z.get('status') == 'ROI_EMCF_BENCHMARK_ONLY' and all((raw/x).is_file() for x in
        ('unwrapped_ifg_phase_rad.npy','time_ifg_pairs_0based.npy','anchor_offset_rad.npy',
         'original_point_ids.npy'))


def validate(stage, raw, threshold, tolerance):
    qa = loadj(stage/'qa_summary.json') or {}
    if qa.get('status') != 'GO_FULLRES_ISLAND_RELATIVE':
        return False, 'QA_NOT_GO'
    n = int(qa.get('npoint') or 0)
    if n < 1 or not stage.joinpath('valid_points.npy').is_file() or not stage.joinpath('acquisition_relative_phase_rad.npy').is_file():
        return False, 'MISSING_OUTPUT'
    if qa.get('bad_point_fraction', 1) > threshold + 1e-12:
        return False, 'BAD_FRACTION_GT_LIMIT'
    for key in ('valid_ifg_reconstruction_max_rad', 'first_point_anchor_residual_rad', 'source_wrapped_parity_max_rad'):
        x = qa.get(key)
        if x is None or not np.isfinite(float(x)) or float(x) > tolerance:
            return False, key
    if qa.get('source_relative_closure_fraction') is None or qa['source_relative_closure_fraction'] > threshold + 1e-12:
        return False, 'SOURCE_RELATIVE_CLOSURE'
    if qa.get('time_graph_rank') != 74 or qa.get('global_integer_gauge_determined') is not False:
        return False, 'GAUGE_OR_TIME_RANK'
    ids = np.load(raw/'original_point_ids.npy', mmap_mode='r', allow_pickle=False)
    v = np.load(stage/'valid_points.npy', mmap_mode='r', allow_pickle=False)
    u = np.load(stage/'acquisition_relative_phase_rad.npy', mmap_mode='r', allow_pickle=False)
    if ids.shape != (n,) or v.shape != (n,) or v.dtype != np.bool_ or u.shape != (75, n):
        return False, 'SHAPE_OR_DTYPE'
    if qa.get('accepted_points') != int(np.count_nonzero(v)) or qa.get('bad_points') != int(np.count_nonzero(~v)):
        return False, 'VALID_MASK_COUNT_MISMATCH'
    if not v[0]:
        return False, 'REFERENCE_POINT_INVALID'
    for start in range(0,n,8192):
        end = min(n,start+8192)
        block = np.asarray(u[:,start:end]); good = np.asarray(v[start:end])
        if not np.isfinite(block[:,good]).all():
            return False, 'INVALID_PHASE_IN_MASKED_GOOD_POINT'
        if np.any(good) and np.max(np.abs(block[0,good])) > 1e-6:
            return False, 'NONZERO_TEMPORAL_EPOCH_0'
        if np.any(~good) and not np.isnan(block[1:,~good]).all():
            return False, 'UNMASKED_INVALID_POINT'
        if np.any(~good) and not np.all(block[0,~good] == 0):
            return False, 'REFERENCE_EPOCH_OF_BAD_POINT_NOT_ZERO'
    return True, 'VERIFIED'


def inspect(run, threshold):
    tasks = []
    reasons = Counter()
    npts = Counter()
    hist = Counter()
    for p in sorted((run/'tiles').glob('r*_c*/run_status.json')):
        rs = loadj(p)
        if not rs:continue
        tile = p.parent
        for f in rs.get('failures', []):
            rank = f.get('rank')
            if not isinstance(rank,int):continue
            loc = tile/f'island_{rank:05d}'
            raw = loc/'raw'; old = loc/'timeinv'; staging = loc/STAGE
            oldqa = loadj(old/'qa_summary.json') or {}
            if f.get('stage') == 'island_too_large_split_tile':
                reason = 'OVERSIZE_SEPARATE'
            elif not files_ok(raw):
                reason = 'RAW_NOT_VALIDATED'
            elif oldqa.get('status') != 'STOP_ISLAND_TIMEINV':
                reason = 'NOT_FAILED_TIMEINV'
            else:
                n = int(oldqa.get('npoint') or f.get('points') or 0)
                fraction = oldqa.get('bad_point_fraction')
                source_rel = oldqa.get('source_relative_closure_fraction')
                if not isinstance(fraction,(int,float)) or not isinstance(source_rel,(int,float)):
                    reason = 'MISSING_OLD_QA_FRACTIONS'
                elif not np.isfinite(fraction) or not np.isfinite(source_rel):
                    reason = 'NONFINITE_QA'
                elif any('reference point failed' in str(x).lower() for x in oldqa.get('failed_conditions', [])):
                    reason = 'REFERENCE_POINT_REQUIRES_REVIEW'
                elif fraction > threshold or source_rel > threshold:
                    reason = 'DEFER_HIGH_BAD_FRACTION'
                elif n == 0:
                    reason = 'MISSING_POINT_COUNT'
                else:
                    reason = 'ELIGIBLE'
                    hist['bad_0to0.1pct' if fraction <= .001 else 'bad_0.1to1pct' if fraction <= .01 else 'bad_above1pct'] += 1
                    tasks.append({'tile':tile.name,'rank':rank,'npoint':n,
                                  'raw':str(raw),'old':str(old),'staging':str(staging),
                                  'bad_fraction':fraction,'source_closure_fraction':source_rel})
            reasons[reason]+=1
            npts[reason]+=int(f.get('points') or 0)
    return tasks, reasons, npts, hist


def work(task, module_path, threshold, tolerance):
    global MODULE
    import importlib.util
    from pathlib import Path
    if MODULE is None:
        spec=importlib.util.spec_from_file_location('native_emcf_timeinv_general',module_path)
        if spec is None or spec.loader is None:
            return task, False, 'MISSING_MODULE'
        MODULE=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MODULE)
    raw=Path(task['raw']);stage=Path(task['staging']);log=stage.parent/'timeinv_recovery_v1.log'
    if stage.exists() and any(stage.iterdir()):
        ok,why=validate(stage,raw,threshold,tolerance)
        return task,ok,'REUSED_'+why if ok else 'STAGING_ALREADY_EXISTS_'+why
    argv=['native_emcf_timeinv_general.py','--roi',str(raw),'--out',str(stage),
          '--tol-rad',str(tolerance),'--max-bad-point-fraction',str(threshold)]
    oldargs=sys.argv
    try:
        with log.open('w',encoding='utf8') as output,contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            sys.argv=argv
            try:rc=int(MODULE.main() or 0)
            except BaseException as exc:
                traceback.print_exc()
                rc=2
            print('RECOVERY_EXIT_CODE:',rc)
    finally:
        sys.argv=oldargs
    if rc:return task,False,'TIMEINV_EXIT_'+str(rc)
    ok,why=validate(stage,raw,threshold,tolerance)
    return task,ok,why


def promote(task, threshold, tolerance, stamp):
    loc=Path(task['raw']).parent
    old=loc/'timeinv';stage=loc/STAGE;raw=loc/'raw'
    good,why=validate(stage,raw,threshold,tolerance)
    if not good: return False, 'PROMOTION_VERIFY_FAILED_'+why
    oldqa=loadj(old/'qa_summary.json') or {}
    if oldqa.get('status')!='STOP_ISLAND_TIMEINV':
        return False,'OLD_TIMEINV_STATUS_CHANGED'
    dest=loc/('timeinv_failed_backup_'+stamp)
    if dest.exists():return False,'BACKUP_ALREADY_EXISTS'
    old.rename(dest)
    try:stage.rename(old)
    except BaseException:
        dest.rename(old)
        raise
    newqa = loadj(old/'qa_summary.json') or {}
    newqa['output'] = str(old)
    newqa['recovery_provenance'] = {
        'old_failed_timeinv_backup': str(dest),
        'source_spatial_emcf_unchanged': True,
        'point_residual_tolerance_rad': tolerance,
        'max_bad_point_fraction_applied': threshold,
        'publish_scope': 'ISLAND_RELATIVE_WITH_POINT_MASK_ONLY',
    }
    atomic_json(old/'qa_summary.json', newqa)
    return True,str(dest)


def update_status_for_tiles(run, promoted_tiles, stamp):
    stats=Counter()
    for tid in sorted(promoted_tiles):
        path=run/'tiles'/tid/'run_status.json'
        rs=loadj(path)
        if not rs: stats['STATUS_MISSING']+=1;continue
        oldf=rs.get('failures',[])
        remaining=[];done_add=0
        for f in oldf:
            r=f.get('rank')
            loc=path.parent/f'island_{r:05d}' if isinstance(r,int) else None
            qa=loadj(loc/'timeinv'/'qa_summary.json') if loc else None
            if qa and qa.get('status')=='GO_FULLRES_ISLAND_RELATIVE' and (loc/'timeinv'/'acquisition_relative_phase_rad.npy').is_file() and files_ok(loc/'raw'):
                done_add+=1
            else:remaining.append(f)
        if not done_add: stats['NO_CHANGE']+=1;continue
        bak=path.with_name('run_status_before_timeinv_recovery_'+stamp+'.json')
        if bak.exists():stats['BACKUP_COLLISION']+=1;continue
        bak.write_bytes(path.read_bytes())
        rs['failures']=remaining
        rs['completed_islands']=int(rs.get('completed_islands',0))+done_add
        rs['status']='PARTIAL_FAILURE' if remaining else 'DONE'
        rs['timeinv_recovery']={'timestamp':stamp,'recovered_islands':done_add,'baseline_status_backup':bak.name}
        atomic_json(path,rs)
        stats[rs['status']]+=1
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--module',type=Path,default=Path('/mnt/ningbo_process/ds/test-ds-main/tools/native_emcf_timeinv_general.py'))
    p.add_argument('--max-bad-fraction',type=float,default=.01,
                   help='1%% island bad points allowed; every valid point still must pass strict IFG residual')
    p.add_argument('--tol-rad',type=float,default=1e-3)
    p.add_argument('--jobs',type=int,default=6)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--promote',action='store_true',help='After successful sidecar QA, BACK UP old failure folder and replace it; update run_status')
    a=p.parse_args()
    if a.promote and not a.apply:p.error('--promote requires --apply')
    if not(0 < a.max_bad_fraction <= .05):p.error('max-bad-fraction must be (0,.05]; >=5% failures require separate review')
    if not(0 < a.tol_rad <= .001):p.error('tol-rad must be (0,.001]')
    if not(1 <= a.jobs <= 16):p.error('jobs must be 1..16')
    run=a.run.resolve();out=(a.out or run.parent/'native_emcf_timeinv_recovery_report').resolve()
    if not (run/'tiles').is_dir():p.error('Missing run tiles directory')
    if out==run or run in out.parents:p.error('--out cannot be within run directory')
    if a.apply and not a.module.is_file():p.error('Cannot find native_emcf_timeinv_general.py')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    tasks,reasons,npts,hist=inspect(run,a.max_bad_fraction)
    out.mkdir(parents=True,exist_ok=True)
    summary={'status':'PLAN_ONLY' if not a.apply else 'STARTED',
             'run':str(run),'max_bad_point_fraction':a.max_bad_fraction,'tol_rad':a.tol_rad,
             'eligibility_counts':dict(reasons),'tile_overlap_points':dict(npts),
             'eligible_fraction_histogram':dict(hist),
             'note':'Original per-point residual tolerance is unchanged. No spatial EMCF recomputation. Ineligible high-error islands are deliberately deferred.'}
    print('=== NINGBO TIMEINV RECOVERY ===',flush=True)
    print('Eligibility:',dict(reasons),flush=True)
    print('Eligible points (tile-overlap counted):',npts.get('ELIGIBLE',0),flush=True)
    if not a.apply:
        atomic_json(out/'plan_summary.json',summary)
        print('PLAN_ONLY. Use --apply to recompute sidecars, optionally --promote to back up & publish.',flush=True)
        return 0
    lock=run/'.fullregion_fast.lock'
    successes=[];failed=[]
    with lock.open('a+') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another FAST fullregion runner holds its lock; STOP, do not run concurrently')
        with ProcessPoolExecutor(max_workers=a.jobs,mp_context=get_context('spawn')) as pool:
            futures=[pool.submit(work,t,str(a.module.resolve()),a.max_bad_fraction,a.tol_rad) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                try:t,ok,why=f.result()
                except BaseException as ex:
                    failed.append({'stage':'EXCEPTION','error':repr(ex)})
                    continue
                if ok:successes.append(t)
                else:failed.append({'tile':t['tile'],'rank':t['rank'],'why':why})
                if i % 100 == 0 or i==len(futures):
                    print(f'[TIMEINV] {i}/{len(futures)} verified={len(successes)} failed={len(failed)}',flush=True)
        promoted=[];promote_failed=[]
        if a.promote:
            for t in successes:
                try:ok,detail=promote(t,a.max_bad_fraction,a.tol_rad,stamp)
                except BaseException as ex:ok=False;detail=repr(ex)
                if ok:promoted.append(t)
                else:promote_failed.append({'tile':t['tile'],'rank':t['rank'],'why':detail})
            statuses=update_status_for_tiles(run,set(t['tile'] for t in promoted),stamp)
        else:statuses={}
    summary.update({'status':'RECOVERY_REPORTED','sidecar_verified':len(successes),'sidecar_failed':len(failed),
                    'promoted':len(promoted),'promotion_failed':promote_failed[:50],
                    'status_updates':dict(statuses),'sample_sidecar_failures':failed[:50],
                    'missing_status_tiles_still_unhandled':True,'oversize_islands_still_unhandled':True})
    atomic_json(out/('recovery_summary_'+stamp+'.json'),summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sample_sidecar_failures','promotion_failed')},indent=2,ensure_ascii=False),flush=True)
    return 0 if not failed and not promote_failed else 1

if __name__=='__main__':
    sys.exit(main())
