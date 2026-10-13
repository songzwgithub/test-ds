#!/usr/bin/env python3
"""Read-only post-recovery triage of remaining Ningbo EMCF fullregion gaps.

Inspect the *current* run_status files after timeinv recovery. No raw phase loads,
solver invocations, checkpoint edits, or global gauge claims.
"""
from __future__ import annotations
import argparse, collections, csv, json, math
from pathlib import Path

DEF=Path('/mnt/ningbo_process/ds/output/diagnostics/native_emcf_fullregion_engineering')

def readj(f):
    try:
        return json.loads(f.read_text()) if f.is_file() else None
    except (OSError,ValueError):
        return None

def fnum(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def bucket(v):
    if v is None:return 'UNKNOWN'
    cuts=((.001,'0-0.1%'),(.01,'0.1-1%'),(.03,'1-3%'),(.05,'3-5%'),(.10,'5-10%'),(.20,'10-20%'),(.50,'20-50%'),(1.000001,'50-100%'))
    for top,name in cuts:
        if v<=top:return name
    return '>100%_INVALID'

def sizegroup(n):
    if n<=80000:return '(60k,80k]'
    if n<=120000:return '(80k,120k]'
    if n<=200000:return '(120k,200k]'
    return '>200k'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=DEF)
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args()
    run=a.run.resolve()
    out=(a.out or run.parent/'native_emcf_remaining_triage').resolve()
    if not (run/'tiles').is_dir():p.error(f'Run missing tiles: {run}')
    if out==run or run in out.parents:p.error('Refusing to write within production run')
    tile_dirs=sorted(x for x in (run/'tiles').iterdir() if x.is_dir() and x.name.startswith('r') and '_c' in x.name)
    status_counts=collections.Counter(); kinds=collections.Counter(); rows=[]
    bad_hist=collections.Counter(); rel_hist=collections.Counter(); size_hist=collections.Counter()
    reason_hist=collections.Counter(); pts=collections.Counter(); missing=[]; issue_samples={}
    for tile in tile_dirs:
        stat=readj(tile/'run_status.json')
        if not stat:
            status_counts['NO_STATUS']+=1;missing.append(tile.name)
            continue
        status_counts[str(stat.get('status','UNKNOWN'))]+=1
        for fail in stat.get('failures',[]):
            rank=fail.get('rank'); n=int(fail.get('points') or 0)
            if not isinstance(rank,int):continue
            loc=tile/f'island_{rank:05d}'
            qa=readj(loc/'timeinv'/'qa_summary.json') or {}
            raw=readj(loc/'raw'/'summary.json') or {}
            fr=fnum(qa.get('bad_point_fraction')); rel=fnum(qa.get('source_relative_closure_fraction'))
            stage=fail.get('stage','UNKNOWN')
            if stage=='island_too_large_split_tile':kind='OVERSIZE'
            elif qa.get('status')=='STOP_ISLAND_TIMEINV':kind='TIMEINV_REJECTED'
            else:kind='NEEDS_INSPECTION'
            reasons=qa.get('failed_conditions') or []
            if not isinstance(reasons,list):reasons=[str(reasons)]
            reason=' | '.join(str(x) for x in reasons)
            row={'tile':tile.name,'rank':rank,'points_with_tile_overlap':n,
                 'kind':kind,'stage':stage,'bad_fraction':fr,'source_relative_closure':rel,
                 'bad_points':qa.get('bad_points'), 'accepted_points':qa.get('accepted_points'),
                 'first_point_anchor_residual_rad':qa.get('first_point_anchor_residual_rad'),
                 'worst_raw_point_residual_rad':qa.get('worst_raw_point_residual_rad'),
                 'raw_status':raw.get('status'), 'timeinv_status':qa.get('status'),
                 'failed_conditions':reason}
            rows.append(row);kinds[kind]+=1;pts[kind]+=n
            if kind=='TIMEINV_REJECTED':
                bad_hist[bucket(fr)]+=1;rel_hist[bucket(rel)]+=1
                if fr is None:reason_hist['MISSING_BAD_FRACTION']+=1
                if rel is None:reason_hist['MISSING_CLOSURE_FRACTION']+=1
                if fr is not None and fr>.01:reason_hist['BAD_FRACTION_GT_1PCT']+=1
                if rel is not None and rel>.01:reason_hist['SOURCE_CLOSURE_GT_1PCT']+=1
                if any('reference' in str(x).lower() for x in reasons):reason_hist['REFERENCE_REJECTED']+=1
                if fr is not None and fr<=.05 and rel is not None and rel<=.01 and not any('reference' in str(x).lower() for x in reasons):
                    reason_hist['POTENTIAL_MASK_ONLY_1_TO_5PCT']+=1
            elif kind=='OVERSIZE':size_hist[sizegroup(n)]+=1
            if kind not in issue_samples:issue_samples[kind]=row
    out.mkdir(parents=True,exist_ok=True)
    csvname=out/'remaining_islands.csv'
    with csvname.open('w',newline='',encoding='utf8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [])
        if rows:wr.writeheader();wr.writerows(rows)
    js={'tile_status_counts':dict(status_counts),'tile_directories':len(tile_dirs),
        'missing_status_tiles':missing, 'remaining_islands_by_kind':dict(kinds),
        'points_with_tile_overlap_by_kind':dict(pts), 'timeinv_bad_fraction_histogram':dict(bad_hist),
        'timeinv_source_closure_histogram':dict(rel_hist),'timeinv_failure_flags':dict(reason_hist),
        'oversize_size_distribution':dict(size_hist),'examples':issue_samples,
        'caveats':['Counts are AFTER recovery, not the original 1656 failures.',
                   'Point totals count tile overlaps; not unique source coverage.',
                   'Passing an IFG-fit threshold does not establish correct absolute integer histories.',
                   'This command is read-only toward the run and makes no repair decisions.']}
    (out/'summary.json').write_text(json.dumps(js,ensure_ascii=False,indent=2),encoding='utf8')
    print('=== POST-RECOVERY NINGBO TRIAGE ===')
    for name in ['tile_status_counts','remaining_islands_by_kind','points_with_tile_overlap_by_kind','timeinv_bad_fraction_histogram','timeinv_source_closure_histogram','timeinv_failure_flags','oversize_size_distribution']:
        print(name+':',json.dumps(js[name],ensure_ascii=False,sort_keys=True))
    print('Missing statuses:',missing)
    print('Saved:',out/'summary.json',csvname)
if __name__=='__main__':main()
