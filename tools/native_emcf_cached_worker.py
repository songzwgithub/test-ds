#!/usr/bin/env python3
"""Invoke LOCAL native_emcf_backend using exact cached ROI PS/DS arrays.

Avoid O(total 26m points) rescanning for each island within the same ROI.
Cache is made from the ORIGINAL point_phase_stack, without modification.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    a,rest=p.parse_known_args()
    if rest and rest[0]=='--':rest=rest[1:]
    c=a.cache.resolve();meta=json.loads((c/'cache_meta.json').read_text())
    arrays={name:np.load(c/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            for name in ('original_point_ids','original_radar_rows','original_radar_cols','original_point_type','phase_rad')}
    n=len(arrays['original_point_ids'])
    if arrays['phase_rad'].shape!=(n,75) or any(arrays[k].shape!=(n,) for k in ('original_radar_rows','original_radar_cols','original_point_type')):
        raise RuntimeError('Cache shape mismatch')
    from pypsds.unwrap import native_point_graph
    from pypsds.unwrap import native_emcf_backend
    def patched_load(*,root,row0,col0,height,width,subset='combined',max_points=400000):
        t=meta['tile']
        if (row0,col0,height,width)!=(t['row0'],t['col0'],t['height'],t['width']):
            raise RuntimeError('ROI coordinates mismatch cached arrays')
        if subset!='combined':raise RuntimeError('Only combined ROI cache supported')
        if n>max_points:raise RuntimeError('max_points smaller than ROI cache')
        src=Path(root).resolve()
        if src!=Path(meta['source']).resolve():raise RuntimeError('Cache source does not match backend PointPhaseStack')
        return tuple(arrays[k] for k in ('original_point_ids','original_radar_rows','original_radar_cols','original_point_type','phase_rad'))
    native_point_graph.load_tile_points=patched_load
    return native_emcf_backend.main(rest)

if __name__=='__main__':
    sys.exit(main())
