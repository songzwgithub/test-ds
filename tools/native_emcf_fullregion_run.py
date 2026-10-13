#!/usr/bin/env python3
"""Full-resolution Native EMCF engineering runner with per-island checkpoints.

NOT a geographically referenced displacement product. Uses original PS/DS only.
ROI overlap carries repeated ORIGINAL point IDs for a later gauge-alignment pass.
Does not interpolate, synthesize points, delete files, or write in production trees.

Run from test-ds-main on the server. See FULLREGION_README.md.
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

DEFAULT_ROOT=Path('/mnt/ningbo_process/ds')

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)

def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None

def positions(size,tile,overlap):
    if size<3 or tile<=overlap or overlap<1:raise ValueError('Invalid tile/overlap')
    # Avoid tiny terminal ROI slivers, with a possibly larger last overlap.
    if size<=tile:return [0]
    starts=list(range(0,size-tile+1,tile-overlap))
    last=size-tile
    if starts[-1]!=last:starts.append(last)
    return starts

def tile_records(a):
    for row in positions(a.rows,a.tile,a.overlap):
        for col in positions(a.cols,a.tile,a.overlap):
            height=min(a.tile,a.rows-row);width=min(a.tile,a.cols-col)
            if height<3 or width<3:continue
            yield dict(id=f'r{row:05d}_c{col:05d}',row0=row,col0=col,height=height,width=width)

def preflight(a):
    if a.out.resolve()==a.root.resolve() or a.out.resolve()==(a.root/'output'/'processing').resolve():
        raise RuntimeError('Refusing unsafe output')
    processing=(a.root/'output'/'processing').resolve()
    if processing in a.out.resolve().parents:raise RuntimeError('Output inside production processing')
    if a.out.resolve() in (a.root/'output'/'products_tp_la').resolve().parents:raise RuntimeError('Output inside TP-LA product')
    help_run=subprocess.run([sys.executable,'-m','pypsds.unwrap.native_emcf_backend','--help'],
                            capture_output=True,text=True,cwd=a.repo)
    if help_run.returncode:raise RuntimeError('Backend not runnable: '+help_run.stderr[-1200:])
    if '--enable-hole-faces' not in help_run.stdout:
        raise RuntimeError('LOCAL backend does not offer --enable-hole-faces. Do not use older remote version.')
    timeinv=Path(__file__).with_name('native_emcf_timeinv_general.py')
    if not timeinv.exists():raise RuntimeError(f'Missing generalized time inverter: {timeinv}')
    if not Path(__file__).with_name('native_emcf_cached_worker.py').is_file():raise RuntimeError('Missing cached backend worker')
    if not (a.root/'pypsds.yaml').is_file():raise RuntimeError('Missing pypsds.yaml')
    if not (a.root/'output'/'processing'/'point_phase_stack').is_dir():raise RuntimeError('Missing PointPhaseStack')
    if a.out.resolve()==(a.root/'output'/'diagnostics'/'native_emcf_roi_island0_fullres').resolve():
        raise RuntimeError('Refusing to overwrite Island0 validation')
    if a.jobs<1 or a.jobs>4:raise RuntimeError('Jobs must be 1..4 (64GiB machine)')
    if not(32<=a.tile<=2048 and 1<=a.overlap<a.tile and a.max_run_points>=3 and a.min_points>=3):
        raise RuntimeError('Invalid tiling or island constraints')
    if a.limit_tiles<0 or a.max_islands<0:raise RuntimeError('Invalid limit')

def discover(a,tile,task_dir):
    """Use the ACTUAL LOCAL backend geometry rather than guessing island counts from first20 report."""
    from pypsds.context import open_from_config
    from pypsds.geometry.inputs import resolve_geometry_inputs
    from pypsds.gamma.geometry import geometry_from_par
    from pypsds.unwrap.native_point_graph import load_tile_points
    from pypsds.unwrap.native_emcf_backend import build_triangle_islands
    config=a.root/'pypsds.yaml'
    cfg,_,paths,stack,_=open_from_config(config)
    gg=geometry_from_par(resolve_geometry_inputs(cfg,paths).reference_rslc_par)
    source=Path(paths.output_dir)/'processing'/'point_phase_stack'
    try:
        ids, rr, cc, typ, ph=load_tile_points(
            root=source,
            row0=tile['row0'],col0=tile['col0'],height=tile['height'],width=tile['width'],
            subset='combined',max_points=a.max_points)
    except RuntimeError as exc:
        if str(exc).startswith('Only ') and ' original points in ROI' in str(exc):
            return {'input_points':int(str(exc).split()[1]),'islands':[],'unused_points':int(str(exc).split()[1]),'valid_triangle_points':0}
        raise
    if ph.shape[1]!=75 or len(stack.dates)!=75:raise RuntimeError('75-date assumption violated')
    nobs=len(ids)
    cache=task_dir/'roi_cache'
    cache.mkdir(parents=True,exist_ok=True)
    for name,data in (('original_point_ids',ids),('original_radar_rows',rr),('original_radar_cols',cc),('original_point_type',typ),('phase_rad',ph)):
        np.save(cache/(name+'.npy'),data,allow_pickle=False)
    atomic_json(cache/'cache_meta.json',{'tile':tile,'source':str(source.resolve()),'npoint':nobs,'dates':75})
    del ph
    if nobs<3:
        return {'input_points':nobs,'islands':[],'unused_points':nobs,'valid_triangle_points':0,'point_ids_unique':True}
    if len(np.unique(ids))!=nobs:raise RuntimeError('Duplicate source IDs in ROI')
    islands,geo=build_triangle_islands(rr,cc,float(gg.azimuth_spacing_m),
        float(gg.ground_range_spacing_m),max_edge_m=a.max_edge_m)
    item=[]
    for i,z in enumerate(islands):
        item.append({'rank':i,'points':z.npoints,'triangles':z.ntriangles,
                     'boundary_edges':int(len(z.boundary)),'original_solver_eligible':z.solver_eligible,
                     'chosen':z.npoints>=a.min_points})
    return {'input_points':nobs,'islands':item,
            'unused_points':int(geo.get('unused_points',0)),
            'valid_triangle_points':int(sum(z.npoints for z in islands)),
            'point_ids_unique':True}

def subprocess_logged(command,logpath,cwd,env):
    logpath.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open('w',encoding='utf8') as fh:
        fh.write('COMMAND: '+ ' '.join(map(str,command))+'\n');fh.flush()
        p=subprocess.run(command,cwd=cwd,env=env,stdout=fh,stderr=subprocess.STDOUT)
        fh.write('\nEXIT CODE: '+str(p.returncode)+'\n')
    return p.returncode

def qa_good(p):
    q=read_json(p/'qa_summary.json')
    return bool(q and q.get('status')=='GO_FULLRES_ISLAND_RELATIVE'
                and (p/'acquisition_relative_phase_rad.npy').is_file()
                and (p/'valid_points.npy').is_file())

def raw_good(p):
    s=read_json(p/'summary.json')
    return bool(s and s.get('status')=='ROI_EMCF_BENCHMARK_ONLY'
                and (p/'unwrapped_ifg_phase_rad.npy').is_file()
                and (p/'original_point_ids.npy').is_file()
                and (p/'time_ifg_pairs_0based.npy').is_file())

def process_tile(a,tile):
    task_dir=a.out/'tiles'/tile['id'];task_dir.mkdir(parents=True,exist_ok=True)
    planfile=task_dir/'plan.json'
    prev=read_json(planfile)
    if prev is None:
        try:
            di=discover(a,tile,task_dir)
        except Exception as ex:
            err={'status':'PLAN_FAILURE','tile':tile,'error':repr(ex),'trace':traceback.format_exc()}
            atomic_json(task_dir/'failure.json',err)
            return err
        prev={'tile':tile,'input_points':di['input_points'],'unused_points':di['unused_points'],
              'valid_triangle_points':di['valid_triangle_points'],'islands':di['islands'],
              'min_points':a.min_points,'max_edge_m':a.max_edge_m,'max_points':a.max_points}
        atomic_json(planfile,prev)
    else:
        if prev['tile']!=tile or prev['min_points']!=a.min_points or prev['max_edge_m']!=a.max_edge_m:
            return {'tile':tile,'status':'PARAMETER_MISMATCH','error':'Existing plan differs; use a new output root'}
    if prev['islands'] and not (task_dir/'roi_cache'/'cache_meta.json').is_file():
        # A completed source scan must not be treated as a runnable cached ROI.
        return {'tile':tile['id'],'status':'CACHE_MISSING','error':'ROI cache absent; use fresh output root'}
    if a.dry_run:
        return {'tile':tile['id'],'status':'PLANNED','points':prev['input_points'],
                'islands':len(prev['islands']),'selected':sum(x['chosen'] for x in prev['islands'])}
    chosen=[x for x in prev['islands'] if x['chosen']]
    if a.max_islands:chosen=chosen[:a.max_islands]
    env=dict(os.environ)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]=str(a.blas_threads)
    completed=0;failed=[]
    for x in chosen:
        rank=x['rank']; loc=task_dir/f'island_{rank:05d}';raw=loc/'raw';inv=loc/'timeinv'
        if qa_good(inv) and raw_good(raw):
            completed+=1;continue
        if not raw_good(raw):
            if raw.exists() and any(raw.iterdir()):
                failed.append({'rank':rank,'points':x['points'],'stage':'raw_partial_unsafe_to_overwrite'});continue
            if x['points']>a.max_run_points:
                failed.append({'rank':rank,'points':x['points'],'stage':'island_too_large_split_tile'});continue
            command=[sys.executable,str(Path(__file__).with_name('native_emcf_cached_worker.py')),
                     '--cache',str(task_dir/'roi_cache'),'--',
                     '--config',str(a.root/'pypsds.yaml'),
                     '--row0',str(tile['row0']),'--col0',str(tile['col0']),
                     '--height',str(tile['height']),'--width',str(tile['width']),
                     '--subset','combined','--max-edge-m',str(a.max_edge_m),
                     '--max-points',str(a.max_points),'--island-rank',str(rank),
                     '--max-run-points',str(a.max_run_points),
                     '--enable-hole-faces','--workers',str(a.workers),
                     '--links-per-batch',str(a.links_per_batch),
                     '--output-dir',str(raw)]
            ret=subprocess_logged(command,loc/'logs'/'emcf.log',a.repo,env)
            if ret or not raw_good(raw):
                failed.append({'rank':rank,'points':x['points'],'stage':'spatial_emcf','exit_code':ret});continue
        if not qa_good(inv):
            if inv.exists() and any(inv.iterdir()):
                failed.append({'rank':rank,'points':x['points'],'stage':'timeinv_partial_or_failed_unsafe_to_overwrite'});continue
            command=[sys.executable,str(Path(__file__).with_name('native_emcf_timeinv_general.py')),
                     '--roi',str(raw),'--out',str(inv),'--max-bad-point-fraction',str(a.max_bad_fraction)]
            ret=subprocess_logged(command,loc/'logs'/'timeinv.log',a.repo,env)
            if ret or not qa_good(inv):
                failed.append({'rank':rank,'points':x['points'],'stage':'temporal_inversion','exit_code':ret});continue
        completed+=1
    result={'tile':tile['id'],'status':('PARTIAL_FAILURE' if failed else ('SMOKE_LIMIT' if a.max_islands and len(chosen)<sum(x['chosen'] for x in prev['islands']) else 'DONE')),
            'points':prev['input_points'],'available_triangular_points':prev['valid_triangle_points'],
            'untriangulated_points':prev['unused_points'],
            'islands':len(prev['islands']),'selected_islands':len(chosen),
            'skipped_small_points':int(sum(x['points'] for x in prev['islands'] if not x['chosen'])),
            'completed_islands':completed,'failures':failed}
    atomic_json(task_dir/'run_status.json',result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--repo',type=Path,default=DEFAULT_ROOT/'test-ds-main')
    ap.add_argument('--out',type=Path,default=DEFAULT_ROOT/'output/diagnostics/native_emcf_fullregion_engineering')
    ap.add_argument('--rows',type=int,default=12225,help='RSLC radar rows: VERIFY YOUR geometry before use')
    ap.add_argument('--cols',type=int,default=44842,help='RSLC radar columns: VERIFY YOUR geometry before use')
    ap.add_argument('--tile',type=int,default=1024)
    ap.add_argument('--overlap',type=int,default=128)
    ap.add_argument('--max-edge-m',type=float,default=40)
    ap.add_argument('--max-points',type=int,default=400000)
    ap.add_argument('--max-run-points',type=int,default=60000)
    ap.add_argument('--min-points',type=int,default=32,help='Smaller islands NOT silently included; report omitted point counts')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--links-per-batch',type=int,default=1024)
    ap.add_argument('--jobs',type=int,default=1,help='Concurrent ROI tiles (1 safest, 2 on 64 GiB)')
    ap.add_argument('--blas-threads',type=int,default=2)
    ap.add_argument('--max-bad-fraction',type=float,default=0.001)
    ap.add_argument('--limit-tiles',type=int,default=0,help='First N tiles for smoke test; 0=all')
    ap.add_argument('--only-tile',type=str,default='',help='e.g. r00000_c00000')
    ap.add_argument('--max-islands',type=int,default=0,help='Only first K islands per tile, for smoke test; 0=all')
    ap.add_argument('--dry-run',action='store_true',help='Build exact geometry plans but no Spurt unwrapping')
    a=ap.parse_args();a.root=a.root.resolve();a.repo=a.repo.resolve();a.out=a.out.resolve()
    preflight(a)
    tiles=list(tile_records(a))
    if a.only_tile:tiles=[x for x in tiles if x['id']==a.only_tile]
    if a.limit_tiles:tiles=tiles[:a.limit_tiles]
    if not tiles:raise SystemExit('No tiles selected')
    a.out.mkdir(parents=True,exist_ok=True)
    config={'rows':a.rows,'cols':a.cols,'tile':a.tile,'overlap':a.overlap,'max_edge_m':a.max_edge_m,
            'min_points':a.min_points,'max_run_points':a.max_run_points,
            'reference':'Each island relative to its first point; cross-tile gauge NOT aligned',
            'raw_phase_source':str(a.root/'output/processing/point_phase_stack'),
            'production_modified':False}
    fp=a.out/'pipeline_config.json';old=read_json(fp)
    if old is not None and old!=config:raise SystemExit('Configuration changed! Use a new output folder or same settings.')
    if old is None:atomic_json(fp,config)
    print('Tiles selected:',len(tiles),'output:',a.out,'dry_run:',a.dry_run,flush=True)
    started=time.time();results=[]
    if a.jobs==1:
        for i,t in enumerate(tiles,1):
            try:r=process_tile(a,t)
            except Exception as ex:r={'tile':t['id'],'status':'UNCAUGHT_ERROR','error':repr(ex),'trace':traceback.format_exc()}
            results.append(r)
            print(f'[{i}/{len(tiles)}] {t["id"]} {r["status"]} completed={r.get("completed_islands",0)} failures={len(r.get("failures",[]))}',flush=True)
    else:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures={pool.submit(process_tile,a,t):t for t in tiles}
            for i,fut in enumerate(as_completed(futures),1):
                t=futures[fut]
                try:r=fut.result()
                except Exception as ex:r={'tile':t['id'],'status':'UNCAUGHT_ERROR','error':repr(ex),'trace':traceback.format_exc()}
                results.append(r)
                print(f'[{i}/{len(tiles)}] {t["id"]} {r["status"]} completed={r.get("completed_islands",0)} failures={len(r.get("failures",[]))}',flush=True)
    limited=(a.limit_tiles>0 or bool(a.only_tile) or a.max_islands>0)
    overall={'status':'INCOMPLETE' if any(x['status'] in ('PARTIAL_FAILURE','PARAMETER_MISMATCH','PLAN_FAILURE','UNCAUGHT_ERROR','CACHE_MISSING') for x in results) else ('DRY_RUN' if a.dry_run else ('LIMITED_PILOT' if limited else 'ENGINEERING_RAW_AND_RELATIVE_TIMEINV_DONE')),
             'tiles_selected':len(tiles),'tiles_done':sum(x['status']=='DONE' for x in results),
             'islands_done':sum(x.get('completed_islands',0) for x in results),
             'failures':sum(len(x.get('failures',[])) for x in results),'elapsed_seconds':round(time.time()-started,2),
             'results':results,'cross_tile_gauge_determined':False,'production_modified':False}
    atomic_json(a.out/'last_run_summary.json',overall)
    print(json.dumps({k:v for k,v in overall.items() if k!='results'},indent=2),flush=True)
    return 0 if overall['status']!='INCOMPLETE' else 2

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print('STOP:',e,file=sys.stderr);sys.exit(2)
