#!/usr/bin/env python3
"""Fast drop-in runner: persistent per-tile island batches, existing QA/checkpoints.

Requires original native_emcf_fullregion_run.py and native_emcf_timeinv_general.py
in same directory, plus native_emcf_tile_batch_worker.py. Never run simultaneously
with an old fullregion runner writing the same output root.
"""
from __future__ import annotations
import fcntl, json, os, subprocess, sys, traceback
from pathlib import Path
from types import SimpleNamespace
import native_emcf_fullregion_run as original


def safe_preflight(a):
    # Preserve original validations except artificial concurrency ceiling.
    saved=a.jobs
    if not (1<=saved<=16):raise RuntimeError('jobs must be 1..16')
    a.jobs=min(saved,4)
    try:original_preflight(a)
    finally:a.jobs=saved
    if not Path(__file__).with_name('native_emcf_tile_batch_worker.py').exists():
        raise RuntimeError('Missing tile batch worker')

original_preflight=original.preflight

def process_tile(a,tile):
    tile_dir=a.out/'tiles'/tile['id']
    tile_dir.mkdir(parents=True,exist_ok=True)
    planfile=tile_dir/'plan.json'
    plan=original.read_json(planfile)
    if plan is None:
        try:di=original.discover(a,tile,tile_dir)
        except Exception as exc:
            result={'status':'PLAN_FAILURE','tile':tile,'error':repr(exc),'trace':traceback.format_exc()}
            original.atomic_json(tile_dir/'failure.json',result)
            return result
        plan={'tile':tile,'input_points':di['input_points'],'unused_points':di['unused_points'],
              'valid_triangle_points':di['valid_triangle_points'],'islands':di['islands'],
              'min_points':a.min_points,'max_edge_m':a.max_edge_m,'max_points':a.max_points}
        original.atomic_json(planfile,plan)
    else:
        if plan['tile']!=tile or plan['min_points']!=a.min_points or plan['max_edge_m']!=a.max_edge_m:
            return {'tile':tile,'status':'PARAMETER_MISMATCH','error':'Plan parameters changed'}
    if plan['islands'] and not (tile_dir/'roi_cache'/'cache_meta.json').is_file():
        return {'tile':tile['id'],'status':'CACHE_MISSING','error':'Existing plan lacks ROI cache'}
    if a.dry_run:
        return {'tile':tile['id'],'status':'PLANNED','points':plan['input_points'],
                'islands':len(plan['islands']),'selected':sum(x['chosen'] for x in plan['islands'])}
    chosen=[x for x in plan['islands'] if x['chosen']]
    if a.max_islands:chosen=chosen[:a.max_islands]
    remaining=[]; failed=[];completed=0
    for x in chosen:
        r=x['rank'];loc=tile_dir/f'island_{r:05d}';raw=loc/'raw';inv=loc/'timeinv'
        if original.qa_good(inv) and original.raw_good(raw):completed+=1;continue
        if (raw.exists() and any(raw.iterdir()) and not original.raw_good(raw)):
            failed.append({'rank':r,'points':x['points'],'stage':'raw_partial_unsafe_to_overwrite'});continue
        if x['points']>a.max_run_points:
            failed.append({'rank':r,'points':x['points'],'stage':'island_too_large_split_tile'});continue
        if inv.exists() and any(inv.iterdir()) and not original.qa_good(inv):
            failed.append({'rank':r,'points':x['points'],'stage':'timeinv_partial_or_failed_unsafe_to_overwrite'});continue
        remaining.append(x)
    env=dict(os.environ)
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[k]=str(a.blas_threads)
    chunk=a.batch_size
    for start in range(0,len(remaining),chunk):
        batch=remaining[start:start+chunk]
        cmd=[sys.executable,str(Path(__file__).with_name('native_emcf_tile_batch_worker.py')),
             '--cache',str(tile_dir/'roi_cache'),'--tile-dir',str(tile_dir),
             '--root',str(a.root),'--ranks',','.join(str(x['rank']) for x in batch),
             '--max-run-points',str(a.max_run_points),'--max-points',str(a.max_points),
             '--max-edge-m',str(a.max_edge_m),'--workers',str(a.workers),
             '--links-per-batch',str(a.links_per_batch),'--max-bad-fraction',str(a.max_bad_fraction)]
        rc=original.subprocess_logged(cmd,tile_dir/'batch_logs'/f'batch_{start//chunk:04d}.log',a.repo,env)
        for x in batch:
            r=x['rank'];loc=tile_dir/f'island_{r:05d}'
            if original.qa_good(loc/'timeinv') and original.raw_good(loc/'raw'):
                completed+=1
            else:
                failed.append({'rank':r,'points':x['points'],'stage':'batch_worker_failure','exit_code':rc})
    status='PARTIAL_FAILURE' if failed else ('SMOKE_LIMIT' if a.max_islands and len(chosen)<sum(x['chosen'] for x in plan['islands']) else 'DONE')
    result={'tile':tile['id'],'status':status,'points':plan['input_points'],
            'available_triangular_points':plan['valid_triangle_points'],
            'untriangulated_points':plan['unused_points'],'islands':len(plan['islands']),
            'selected_islands':len(chosen),'skipped_small_points':int(sum(x['points'] for x in plan['islands'] if not x['chosen'])),
            'completed_islands':completed,'failures':failed}
    original.atomic_json(tile_dir/'run_status.json',result)
    return result

# No per-runner conflicts: lock until execution ends. Existing legacy runner
# does not acquire this lock; user MUST stop legacy runner before starting.
def main():
    import argparse
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--jobs',type=int,default=8)
    opts, remaining=parser.parse_known_args()
    if not(1<=opts.jobs<=16 and 1<=opts.batch_size<=256):
        raise SystemExit('--jobs 1..16, --batch-size 1..256')
    # The original parser supplies a.jobs; pass through --jobs and inject
    # batch size via the patched preflight safely.
    sys.argv=[str(Path(__file__)),*remaining,'--jobs',str(opts.jobs)]
    def preflight_with_batch(a):
        a.batch_size=opts.batch_size
        return safe_preflight(a)
    original.preflight=preflight_with_batch
    original.process_tile=process_tile
    # The default run root is fixed by the original runner; allow custom --out.
    out=Path('/mnt/ningbo_process/ds/output/diagnostics/native_emcf_fullregion_engineering')
    for i,v in enumerate(remaining):
        if v=='--out' and i+1<len(remaining):out=Path(remaining[i+1])
    out.mkdir(parents=True,exist_ok=True)
    lock=out/'.fullregion_fast.lock'
    with lock.open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another FAST runner has this output lock')
        return original.main()

if __name__=='__main__':sys.exit(main())
