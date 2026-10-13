#!/usr/bin/env python3
"""One Python process for several islands of ONE cached native PS/DS tile.

Uses local patched native_emcf_backend.main unchanged. Reuses the same PointPhaseStack
ROI cache and the expensive Delaunay/island construction across selected ranks.
Isolated output folders and the existing temporal inverter QA are preserved.
"""
from __future__ import annotations
import argparse, contextlib, importlib.util, json, logging, sys, traceback
from pathlib import Path
import numpy as np


def good_raw(p: Path) -> bool:
    s=p/'summary.json'
    try:
        x=json.loads(s.read_text())
        return x.get('status')=='ROI_EMCF_BENCHMARK_ONLY' and all((p/f).is_file() for f in
                    ('unwrapped_ifg_phase_rad.npy','time_ifg_pairs_0based.npy','original_point_ids.npy','anchor_offset_rad.npy'))
    except (OSError,ValueError):
        return False


def good_inv(p: Path) -> bool:
    s=p/'qa_summary.json'
    try:
        x=json.loads(s.read_text())
        return x.get('status')=='GO_FULLRES_ISLAND_RELATIVE' and all((p/f).is_file() for f in
                    ('acquisition_relative_phase_rad.npy','valid_points.npy'))
    except (OSError,ValueError):
        return False


def log_call(log_path: Path, func, *args):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w',encoding='utf8') as out:
        # Rebind third-party log handlers that captured a previous island log
        # stream and would otherwise write into a closed file.
        loggers=[logging.getLogger()] + [q for q in logging.Logger.manager.loggerDict.values() if isinstance(q,logging.Logger)]
        for logger in loggers:
            for handler in logger.handlers:
                if isinstance(handler,logging.StreamHandler) and not isinstance(handler,logging.FileHandler):
                    if getattr(handler.stream,'closed',False):
                        handler.stream=out
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                rc=func(*args)
        except BaseException as exc:
            out.write('EXCEPTION: '+repr(exc)+'\n')
            traceback.print_exc(file=out)
            rc=int(exc.code) if isinstance(exc,SystemExit) and isinstance(exc.code,int) else 2
        out.write('\nEXIT CODE: '+str(rc)+'\n')
    return int(rc or 0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--tile-dir',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--ranks',required=True,help='Comma-separated ranks, same tile; 0-based')
    p.add_argument('--max-run-points',type=int,default=60000)
    p.add_argument('--max-points',type=int,default=400000)
    p.add_argument('--max-edge-m',type=float,default=40.0)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--links-per-batch',type=int,default=1024)
    p.add_argument('--max-bad-fraction',type=float,default=.001)
    args=p.parse_args()
    cache=args.cache.resolve(); tile_dir=args.tile_dir.resolve()
    meta=json.loads((cache/'cache_meta.json').read_text())
    tile=meta['tile']
    plan=json.loads((tile_dir/'plan.json').read_text())
    valid_ranks={int(x['rank']):x for x in plan['islands']}
    ranks=[int(x) for x in args.ranks.split(',') if x]
    if len(set(ranks))!=len(ranks) or any(r not in valid_ranks for r in ranks):
        raise RuntimeError('Duplicate/invalid island ranks')
    if any(not valid_ranks[r]['chosen'] for r in ranks):
        raise RuntimeError('Attempting to run skipped island')
    arrays={k:np.load(cache/(k+'.npy'),mmap_mode='r',allow_pickle=False)
            for k in ('original_point_ids','original_radar_rows','original_radar_cols','original_point_type','phase_rad')}
    from pypsds.unwrap import native_point_graph, native_emcf_backend
    keys=('original_point_ids','original_radar_rows','original_radar_cols','original_point_type','phase_rad')
    n=len(arrays[keys[0]])
    if arrays['phase_rad'].shape!=(n,75):raise RuntimeError('Cached phase shape does not match')
    cached_islands=None
    build_calls=0
    original_build=native_emcf_backend.build_triangle_islands

    def cached_load(*,root,row0,col0,height,width,subset='combined',max_points=400000):
        if (row0,col0,height,width)!=(tile['row0'],tile['col0'],tile['height'],tile['width']):
            raise RuntimeError('Cached tile geometry mismatch')
        if subset!='combined' or n>max_points:raise RuntimeError('Unsupported cached ROI request')
        if Path(root).resolve()!=Path(meta['source']).resolve():raise RuntimeError('Cached phase source mismatch')
        return tuple(arrays[k] for k in keys)

    def cached_build(rows,cols,sy,sx,max_edge_m=40.0,min_area_m2=1e-4):
        nonlocal cached_islands,build_calls
        if cached_islands is None:
            if float(max_edge_m)!=float(args.max_edge_m):raise RuntimeError('max-edge mismatch')
            cached_islands=original_build(rows,cols,sy,sx,max_edge_m,min_area_m2)
            build_calls+=1
            if len(cached_islands[0])!=len(plan['islands']):
                raise RuntimeError('Island count differs from cached plan!')
            if any(z.npoints!=int(plan['islands'][i]['points']) for i,z in enumerate(cached_islands[0])):
                raise RuntimeError('Island rank/point counts differ from cached plan!')
        return cached_islands

    native_point_graph.load_tile_points=cached_load
    native_emcf_backend.build_triangle_islands=cached_build
    spec=importlib.util.spec_from_file_location('native_emcf_timeinv_general',Path(__file__).with_name('native_emcf_timeinv_general.py'))
    if spec is None or spec.loader is None:raise RuntimeError('Cannot import temporal inverter')
    inverter=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inverter)
    passed=0;failed=[]
    for rank in ranks:
        x=valid_ranks[rank];loc=tile_dir/f'island_{rank:05d}'
        raw=loc/'raw'; inv=loc/'timeinv'
        if good_raw(raw) and good_inv(inv):
            passed+=1;continue
        if not good_raw(raw):
            if raw.exists() and any(raw.iterdir()):
                failed.append((rank,'raw_partial_exists'));continue
            if x['points']>args.max_run_points:
                failed.append((rank,'island_too_large'));continue
            cmd=['--config',str(args.root/'pypsds.yaml'),'--row0',str(tile['row0']),
                 '--col0',str(tile['col0']),'--height',str(tile['height']),
                 '--width',str(tile['width']),'--subset','combined',
                 '--max-edge-m',str(args.max_edge_m),'--max-points',str(args.max_points),
                 '--island-rank',str(rank),'--max-run-points',str(args.max_run_points),
                 '--enable-hole-faces','--workers',str(args.workers),
                 '--links-per-batch',str(args.links_per_batch),'--output-dir',str(raw)]
            rc=log_call(loc/'logs'/'emcf.log',native_emcf_backend.main,cmd)
            if rc or not good_raw(raw):
                failed.append((rank,'spatial_emcf:'+str(rc)));continue
        if not good_inv(inv):
            if inv.exists() and any(inv.iterdir()):
                failed.append((rank,'timeinv_partial_exists'));continue
            argv=['--roi',str(raw),'--out',str(inv),'--max-bad-point-fraction',str(args.max_bad_fraction)]
            old=sys.argv
            try:
                sys.argv=['native_emcf_timeinv_general.py',*argv]
                rc=log_call(loc/'logs'/'timeinv.log',inverter.main)
            finally:
                sys.argv=old
            if rc or not good_inv(inv):
                failed.append((rank,'temporal_inversion:'+str(rc)));continue
        passed+=1
    print(json.dumps({'tile':tile['id'],'batch_ranks':len(ranks),'passed':passed,
                      'failed':failed,'geometry_build_calls':build_calls},ensure_ascii=False),flush=True)
    return 0 if not failed else 2

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print('FATAL BATCH ERROR:',repr(exc),file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
