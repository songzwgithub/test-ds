#!/usr/bin/env python3
"""Read-only failure classifier for Ningbo Native EMCF fullregion fast runner.

Only reads run_status.json, plan.json, island raw/summary.json,
 timeinv/qa_summary.json and a bounded amount of available text logs.
Does not load phase matrices, run a solver, modify checkpoints, or decide
scientific acceptability of a time inversion.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_RUN = Path('/mnt/ningbo_process/ds/output/diagnostics/native_emcf_fullregion_engineering')
RE_RAW = ('unwrapped_ifg_phase_rad.npy', 'time_ifg_pairs_0based.npy', 'original_point_ids.npy')
RE_INV = ('acquisition_relative_phase_rad.npy', 'valid_points.npy')
PAT = re.compile(r'(Traceback|Error|Exception|^STOP|FATAL|No such file|not converge|infeasible|out of memory|SIGKILL|exit code|fatal)', re.I)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    except (ValueError, OSError) as ex:
        return {'_read_error': str(ex)}


def tail(path, count=10, max_bytes=48000):
    if not path.is_file():
        return []
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            size=f.tell()
            f.seek(max(0, size-max_bytes))
            txt=f.read().decode('utf-8', 'replace')
        lines=txt.splitlines()
        hits=[x.strip()[:320] for x in lines if PAT.search(x)]
        return (hits[-count:] or [x.strip()[:320] for x in lines[-min(4,count):]])
    except OSError as ex:
        return [f'LOG_UNREADABLE: {ex}']


def hasfiles(p, required):
    return all((p/x).is_file() for x in required)


def classify(tilepath, f):
    rank=f.get('rank')
    if not isinstance(rank, int) or rank < 0:
        return 'INVALID_FAILURE_RECORD', {}, {}, [], 'Bad/missing rank'
    loc=tilepath/f'island_{rank:05d}'
    raw=loc/'raw'; inv=loc/'timeinv'
    rawqa=read_json(raw/'summary.json') or {}
    invqa=read_json(inv/'qa_summary.json') or {}
    rg = rawqa.get('status')=='ROI_EMCF_BENCHMARK_ONLY' and hasfiles(raw, RE_RAW)
    ig = invqa.get('status')=='GO_FULLRES_ISLAND_RELATIVE' and hasfiles(inv, RE_INV)
    s=str(f.get('stage', 'UNKNOWN'))
    why=''
    if rg and ig:
        cat='ACTUALLY_PASSED_RECHECK'
    elif s == 'island_too_large_split_tile':
        cat='OVERSIZE_ISLAND'
    elif rg and invqa.get('status') == 'STOP_ISLAND_TIMEINV':
        cat='TIMEINV_QA_REJECTED'
        why='; '.join(map(str, invqa.get('failed_conditions', [])))
    elif rg and invqa.get('_read_error'):
        cat='TIMEINV_QA_CORRUPTED'; why=invqa['_read_error']
    elif rg and inv.is_dir() and any(inv.iterdir()):
        cat='TIMEINV_PARTIAL_OR_MISSING_OUTPUT'
        why=f'qa_status={invqa.get("status")} missing={[x for x in RE_INV if not (inv/x).is_file()]}'
    elif rg:
        cat='TIMEINV_NOT_STARTED'
    elif rawqa.get('_read_error'):
        cat='RAW_QA_CORRUPTED'; why=rawqa['_read_error']
    elif raw.is_dir() and any(raw.iterdir()):
        cat='RAW_PARTIAL_OR_BAD_OUTPUT'
        why=f'raw_status={rawqa.get("status")} missing={[x for x in RE_RAW if not (raw/x).is_file()]}'
    else:
        cat='EMCF_NOT_SAVED_OR_WORKER_ABORTED'
    logs={}
    for key, file in [('emcf',loc/'logs'/'emcf.log'),('timeinv',loc/'logs'/'timeinv.log')]:
        lines=tail(file, 3)
        if lines: logs[key]=lines
    return cat, rawqa, invqa, logs, why


def get_expected_ids(run, config):
    if not config:
        return None
    try:
        rows=int(config['rows']);cols=int(config['cols'])
        t=int(config['tile']);ov=int(config['overlap'])
        def pos(size):
            if size<=t:return [0]
            starts=list(range(0,size-t+1,t-ov))
            if starts[-1]!=size-t: starts.append(size-t)
            return starts
        return {f'r{r:05d}_c{c:05d}' for r in pos(rows) for c in pos(cols)}
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--out', type=Path, help='Separate audit output folder; defaults to run sibling')
    p.add_argument('--samples', type=int, default=4)
    a=p.parse_args()
    run=a.run.resolve();out=(a.out or run.parent/(run.name+'_failure_audit')).resolve()
    if not (run/'tiles').is_dir(): p.error('No tiles directory: '+str(run/'tiles'))
    if out==run or run in out.parents:p.error('Refusing audit output inside live run')
    out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(run/'pipeline_config.json') or {}
    manifest=read_json(run/'last_run_summary.json') or {}
    expected=get_expected_ids(run,cfg)
    dirs={x.name:x for x in (run/'tiles').iterdir() if x.is_dir() and re.fullmatch(r'r\d{5}_c\d{5}', x.name)}
    statusids={t for t,d in dirs.items() if (d/'run_status.json').is_file()}
    missing=sorted(expected-statusids) if expected is not None else sorted(set(dirs)-statusids)
    stages=Counter(); cats=Counter(); npoints=Counter(); tiles=defaultdict(set)
    bads=[]; bad_by_cat=defaultdict(list); all_tile_status=Counter()
    for tid in sorted(dirs):
        tp=dirs[tid]
        ts=read_json(tp/'run_status.json')
        if not ts:
            all_tile_status['NO_STATUS']+=1
            continue
        all_tile_status[str(ts.get('status','UNKNOWN'))]+=1
        for f in ts.get('failures',[]):
            if not isinstance(f, dict):continue
            stage=str(f.get('stage','UNKNOWN'))
            cat, rawqa, invqa, logs, why=classify(tp,f)
            row={'tile':tid,'rank':f.get('rank'),'points':f.get('points'),
                 'original_stage':stage,'exit_code':f.get('exit_code',''),
                 'diagnosed_class':cat,'reason':why,
                 'raw_status':rawqa.get('status',''), 'timeinv_status':invqa.get('status',''),
                 'timeinv_bad_points':invqa.get('bad_points',''),
                 'timeinv_bad_fraction':invqa.get('bad_point_fraction',''),
                 'timeinv_relative_closure':invqa.get('source_relative_closure_fraction',''),
                 'log_excerpt':json.dumps(logs,ensure_ascii=False)}
            stages[stage]+=1;cats[cat]+=1
            npoints[cat]+=int(f.get('points') or 0)
            tiles[cat].add(tid)
            bads.append(row);bad_by_cat[cat].append(row)
    if missing:
        with (out/'missing_tiles.txt').open('w',encoding='utf-8') as fh:
            for tid in missing:
                tp=dirs.get(tid)
                pl=read_json(tp/'plan.json') if tp else None
                fl=read_json(tp/'failure.json') if tp else None
                fh.write(f'{tid}\tplan={bool(pl)}\tfailure={json.dumps(fl,ensure_ascii=False)[:900]}\n')
    with (out/'failures.csv').open('w',newline='',encoding='utf-8') as fh:
        cols=['tile','rank','points','original_stage','exit_code','diagnosed_class','reason',
              'raw_status','timeinv_status','timeinv_bad_points','timeinv_bad_fraction',
              'timeinv_relative_closure','log_excerpt']
        wr=csv.DictWriter(fh,fieldnames=cols);wr.writeheader();wr.writerows(bads)
    summary={'run':str(run),'tile_directories':len(dirs),'tile_status_counts':dict(all_tile_status),
             'expected_tile_count':len(expected) if expected is not None else None,
             'missing_tile_status_count':len(missing),'missing_tile_ids':missing,
             'original_failure_stages':dict(stages),
             'classified_failure_counts':dict(cats),
             'classified_failure_tile_counts':{k:len(v) for k,v in tiles.items()},
             'classified_failure_points_with_tile_overlaps':dict(npoints),
             'total_failed_records':len(bads),
             'last_run_status':manifest.get('status'),
             'note':'Per-tile sums double-count overlapping PS/DS point IDs. Classification does not establish physical correctness or safe auto-retry.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('======= NINGBO FULLREGION FAILURE AUDIT (READ ONLY) =======')
    print('Run:', run)
    print('Tile statuses:',dict(all_tile_status))
    print('Missing status tiles:',len(missing),missing[:15])
    print('Original failure stages:',dict(stages))
    print('Classified failure counts:')
    for key, val in cats.most_common():
        print(f'  {key:<36} {val:>6} islands  {npoints[key]:>10} tile-overlap points  {len(tiles[key]):>4} tiles')
    print('Samples by diagnosed class:')
    for k,_ in cats.most_common():
        for x in bad_by_cat[k][:max(0,a.samples)]:
            s=f" {x['tile']} island={x['rank']} points={x['points']} orig={x['original_stage']}"
            if x['reason']:s+=' reason='+x['reason'][:260]
            print(' ',k,s)
            if k not in ('OVERSIZE_ISLAND','ACTUALLY_PASSED_RECHECK') and x['log_excerpt']!='{}':
                print('    log:',x['log_excerpt'][:500])
    print('Saved:',out/'summary.json',out/'failures.csv')

if __name__=='__main__':main()
