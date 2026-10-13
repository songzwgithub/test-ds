#!/usr/bin/env python3
"""Read-only: locate earlier coarse-only native heldout errors by EMCF island.

Requires the existing native ROI, boundary audit CSV, and identical EMCF ROI.
This is a *routing* report, not physical truth or a new validation metric.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pypsds.unwrap.native_emcf_backend import build_triangle_islands


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args(argv)
    root=a.root.resolve()
    native=root/'output/diagnostics/native_point_graph_roi_candidate'
    audit=root/'output/diagnostics/native_emcf_roi_topology'
    errors=root/'output/diagnostics/native_vs_coarse_boundary_audit/old_error_edges.csv'
    if not audit.joinpath('summary.json').is_file():
        parser.error(f'Missing first successful topology audit: {audit}')
    summary=json.loads((audit/'summary.json').read_text())
    ids=np.load(native/'original_point_ids.npy',allow_pickle=False).astype(np.int64)
    rows=np.load(native/'original_radar_rows.npy',allow_pickle=False)
    cols=np.load(native/'original_radar_cols.npy',allow_pickle=False)
    if not (len(ids)==len(rows)==len(cols) and np.all(np.diff(ids)>0)):
        raise RuntimeError('Native point ID / geometry mismatch')
    sy,sx=summary['spacings_nominal_m']
    islands,info=build_triangle_islands(rows,cols,sy,sx,summary['geometry']['max_edge_m'])
    for k in ('input_points','accepted_triangles','islands','unused_points'):
        if info[k]!=summary['geometry'][k]:
            raise RuntimeError(f'ROI island identity changed: {k}: {info[k]} != {summary["geometry"][k]}')
    rank=np.full(len(ids),-1,dtype=np.int32)
    for i,z in enumerate(islands):
        if np.any(rank[z.original_indices]>=0):
            raise RuntimeError('A native point belongs to multiple islands')
        rank[z.original_indices]=i
    stats=defaultdict(Counter)
    bad=Counter()
    with errors.open(newline='') as inp:
        for item in csv.DictReader(inp):
            pa,pb=int(item['point_a']),int(item['point_b'])
            idx=np.searchsorted(ids,[pa,pb])
            if np.any(idx>=len(ids)) or ids[idx[0]]!=pa or ids[idx[1]]!=pb:
                raise RuntimeError('Audit original point ID absent from candidate ROI')
            u,v=rank[idx]
            oldbad=int(item['old_bad_epochs'])
            oldonly=(oldbad>0 and int(item['primary_bad_epochs'])==0
                         and int(item['secondary_bad_epochs'])==0)
            if u==v and u>=0:
                b=stats[int(u)]
                b['old_bad_edges']+=1
                b['old_bad_epochs']+=oldbad
                if oldonly:
                    b['old_only_edges']+=1
                    b['old_only_bad_epochs']+=oldbad
                b['ps_ds_bad_edges']+=(item['type_a']!=item['type_b'])
            else:
                bad['cross_island_or_unused_old_bad_edges']+=1
                bad['cross_island_or_unused_old_bad_epochs']+=oldbad
                if oldonly:bad['cross_island_or_unused_old_only_edges']+=1
            bad['total_old_bad_edges']+=1
            if oldonly:bad['total_old_only_edges']+=1
    print('ROI topo: points',len(ids),'islands',len(islands))
    print('Old bad edge total:',bad['total_old_bad_edges'])
    print('Old-only (both native candidates clean):',bad['total_old_only_edges'])
    print('Old-only cross islands or unused:',bad['cross_island_or_unused_old_only_edges'])
    print('Ranked same-island targets (sorted by old-only heldout edge count):')
    print('rank   points  holes  old_bad  old_only  old_only_epochs  ps_ds_bad')
    ranked=sorted(stats,key=lambda i:(-stats[i]['old_only_edges'], -stats[i]['old_bad_edges'],i))
    for i in ranked[:30]:
        z=islands[i];s=stats[i]
        holes=z.nlinks-z.npoints+1-z.ntriangles
        print(f'{i:4d} {z.npoints:8d} {holes:6d} {s["old_bad_edges"]:8d} '
              f'{s["old_only_edges"]:9d} {s["old_only_bad_epochs"]:16d} '
              f'{s["ps_ds_bad_edges"]:10d}')
    print('Note: errors across distinct islands need later cross-component '
          'constraints, not single-island EMCF benchmarking.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
