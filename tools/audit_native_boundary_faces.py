#!/usr/bin/env python3
"""Read-only original PS/DS boundary-pinch and missing-face audit."""
import argparse
import json
import numpy as np
from pathlib import Path
from pypsds.context import open_from_config
from pypsds.geometry.inputs import resolve_geometry_inputs
from pypsds.gamma.geometry import geometry_from_par
from pypsds.unwrap.native_point_graph import load_tile_points
from pypsds.unwrap.native_emcf_backend import build_triangle_islands, build_hole_face_cycles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--ranks',default='4,5,0,2,14')
    p.add_argument('--row0',type=int,default=7155)
    p.add_argument('--col0',type=int,default=14125)
    p.add_argument('--height',type=int,default=1024)
    p.add_argument('--width',type=int,default=1024)
    p.add_argument('--max-edge-m',type=float,default=40.0)
    args=p.parse_args()
    cfg,_,paths,_,_=open_from_config(args.config)
    pp=Path(paths.output_dir)/'processing'/'point_phase_stack'
    geometry=geometry_from_par(resolve_geometry_inputs(cfg,paths).reference_rslc_par)
    ids,rr,cc,tt,ph=load_tile_points(root=pp,row0=args.row0,col0=args.col0,
        height=args.height,width=args.width,subset='combined',max_points=400000)
    islands,_=build_triangle_islands(rr,cc,float(geometry.azimuth_spacing_m),
           float(geometry.ground_range_spacing_m),args.max_edge_m)
    out={'status':'READ_ONLY_BOUNDARY_FACE_AUDIT', 'islands':len(islands),
         'results':[]}
    for rk in map(int,args.ranks.split(',')):
        if rk<0 or rk>=len(islands):
            raise ValueError(f'Unknown island rank {rk}')
        isl=islands[rk]
        degree=np.bincount(isl.boundary.ravel(),minlength=isl.npoints)
        vals, cnt=np.unique(degree[degree>0],return_counts=True)
        status='OK'; msg=''; nf=None
        try:
            faces=build_hole_face_cycles(isl)
            nf=len(faces)
        except ValueError as exc:
            status='REFUSED';msg=str(exc)
        out['results'].append({
            'rank':rk,'points':isl.npoints,'euler_bounded_missing_faces':
              isl.nlinks-isl.npoints+1-isl.ntriangles,
            'boundary_degree_histogram':{str(int(x)):int(y) for x,y in zip(vals,cnt)},
            'pinched_boundary_vertices_degree_gt2':int(np.sum(degree>2)),
            'faces_traced':nf,'status':status,'reason':msg})
    print(json.dumps(out,indent=2))
    if any(x['status']!='OK' for x in out['results']):
        return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
