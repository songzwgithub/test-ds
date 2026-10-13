#!/usr/bin/env python3
"""Read-only audit of a Native EMCF ROI island.

The code NEVER edits EMCF inputs or results. Its temporal audit requires an
IFG-level per-point integer correction q[e,p] and an oriented temporal edge
list (master,slave). When direct q is unavailable it can derive q from
UNWRAPPED and WRAPPED IFG phase with a verified modulo-2pi check.

Two closure statistics:
 * anchored: fundamental-cycle sum of q, using its given integer gauge
 * relative: the sum after subtracting the modal common integer per cycle
These are integer bookkeeping checks, NOT physical validation of deformation.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
import numpy as np

TWO_PI = 2.0 * np.pi
N_POINT = 3149
N_DATE = 75
N_IFG = 219


def candidates(folder: Path, limit_depth: int = 3):
    if not folder.is_dir():
        return []
    files = []
    for p in folder.rglob('*'):
        if p.is_file() and len(p.relative_to(folder).parts) <= limit_depth and p.suffix.lower() in ('.npy', '.npz', '.json', '.csv'):
            files.append(p)
    return sorted(files)


def inventory(folder: Path):
    result=[]
    for p in candidates(folder):
        desc = {'path':str(p), 'size_bytes':p.stat().st_size}
        if p.suffix == '.npy':
            try:
                a=np.load(p, mmap_mode='r', allow_pickle=False)
                desc.update(shape=list(a.shape),dtype=str(a.dtype))
            except Exception as ex:
                desc['error']=str(ex)
        elif p.suffix == '.npz':
            try:
                with np.load(p,allow_pickle=False) as z:
                    desc['arrays']={k:list(z[k].shape) for k in z.files}
            except Exception as ex:
                desc['error']=str(ex)
        result.append(desc)
    return result


def read_array(path: str | None, key: str | None = None):
    if path is None:
        return None
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix == '.npy':
        return np.load(p,allow_pickle=False)
    if p.suffix == '.npz':
        with np.load(p,allow_pickle=False) as z:
            if key is None:
                if len(z.files)!=1:
                    raise ValueError(f'{p} contains {z.files}; specify --npz-key')
                key=z.files[0]
            return z[key].copy()
    if p.suffix == '.csv':
        return np.loadtxt(p,delimiter=',',dtype=float)
    raise ValueError(f'Unsupported extension {p.suffix}: {p}')


def normalize_matrix(arr, nrows, ncols, name):
    if arr is None:
        raise ValueError(f'missing {name}')
    a=np.asarray(arr)
    if a.shape == (nrows,ncols):
        return a
    if a.shape == (ncols,nrows):
        return a.T
    raise ValueError(f'{name} shape={a.shape}; expected {(nrows,ncols)} or {(ncols,nrows)}')


def orient_pairs(a, n_dates):
    if a is None:
        raise ValueError('missing IFG date-pairs (N_IFG x 2, zero-based)')
    a=np.asarray(a)
    if a.shape == (2,N_IFG):
        a=a.T
    if a.shape != (N_IFG,2):
        raise ValueError(f'pairs shape={a.shape}, expected ({N_IFG},2)')
    if not np.all(np.isfinite(a)) or not np.allclose(a,np.rint(a),atol=0,rtol=0):
        raise ValueError('temporal date pairs must be integer date indices')
    a=np.rint(a).astype(np.int64)
    if np.min(a)<0 or np.max(a)>=n_dates or np.any(a[:,0]==a[:,1]):
        raise ValueError('pairs must use distinct zero-based node indices in 0..74; do not silently shift one-based indices')
    return a


def integers_from_phase(uw, wr):
    uw=normalize_matrix(uw,N_IFG,N_POINT,'unwrapped IFG phase')
    wr=normalize_matrix(wr,N_IFG,N_POINT,'wrapped IFG phase')
    if not np.all(np.isfinite(uw)) or not np.all(np.isfinite(wr)):
        raise ValueError('input IFG phase contains NaN/Inf; explicit valid-point masking is required')
    q=np.rint((uw-wr)/TWO_PI)
    err=np.angle(np.exp(1j*(uw-wr)))
    max_err=float(np.max(np.abs(err)))
    if max_err>1e-3:
        raise ValueError(f'Unwrapped-minus-wrapped parity fails: max={max_err:.4g} rad. Refusing to infer integer corrections.')
    return q.astype(np.int64),max_err


def root_potentials(pairs, q):
    # Choose a deterministic spanning forest in IFG input order.
    par=np.arange(N_DATE,dtype=np.int64)
    rank=np.zeros(N_DATE,dtype=np.int8)
    def find(x):
        while par[x]!=x:
            par[x]=par[par[x]]
            x=par[x]
        return x
    tree=[]
    chords=[]
    for i,(u,v) in enumerate(pairs):
        ru,rv=find(int(u)),find(int(v))
        if ru!=rv:
            if rank[ru]<rank[rv]: ru,rv=rv,ru
            par[rv]=ru
            if rank[ru]==rank[rv]:rank[ru]+=1
            tree.append(i)
        else:
            chords.append(i)
    if len(tree)!=N_DATE-1:
        raise ValueError(f'Temporal graph disconnected: {len(tree)} tree edges but {N_DATE-1} required')
    if len(chords)!=N_IFG-N_DATE+1:
        raise AssertionError('Invalid fundamental-cycle count')
    neighbors=[[] for _ in range(N_DATE)]
    for i in tree:
        u,v=map(int,pairs[i]); neighbors[u].append((v,i,1));neighbors[v].append((u,i,-1))
    pot=np.zeros((N_DATE,N_POINT),dtype=np.int64)
    parent=np.full(N_DATE,-1,dtype=np.int64)
    parent_edge=np.full(N_DATE,-1,dtype=np.int64)
    parent_sign=np.zeros(N_DATE,dtype=np.int8)
    parent[0]=0; queue=deque([0])
    while queue:
        node=queue.popleft()
        for child,edge,sgn in neighbors[node]:
            if parent[child]>=0:continue
            parent[child]=node;parent_edge[child]=edge;parent_sign[child]=sgn
            pot[child]=pot[node]+sgn*q[edge]
            queue.append(child)
    if np.any(parent<0):
        raise AssertionError('Graph traversal missed nodes')
    return chords,pot,parent,parent_edge,parent_sign


def cycle_edge_list(edge,pairs,parent,parent_edge,parent_sign):
    # Closed cycle is chord(u->v) plus tree path(v->u).
    u,v=map(int,pairs[edge])
    def ascend(x):
        out=[]
        while x!=0:
            out.append((int(parent_edge[x]),int(parent_sign[x])))
            x=int(parent[x])
        return out
    p_u=ascend(u);p_v=ascend(v)
    # Subtract tree(root->v) and add tree(root->u).
    return [(int(edge),1)]+[(e,s) for e,s in p_u]+[(e,-s) for e,s in p_v]


def write_csv(path, rows, fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)


def temporal_audit(q,pairs,rows,cols,out,hole_vertices=None):
    q=normalize_matrix(q,N_IFG,N_POINT,'integer IFG corrections')
    if not np.all(np.isfinite(q)) or not np.allclose(q,np.rint(q),atol=1e-5,rtol=0):
        raise ValueError('Integer corrections must be finite integers; refuse phase floats')
    q=np.rint(q).astype(np.int64)
    chords,pot,parent,parent_edge,parent_sign=root_potentials(pairs,q)
    rel_records=[]; raw_nonzero=0; relative_nonzero=0; ambiguous=[]; raw_abs_max=0
    nearest_hole=None
    if hole_vertices is not None and rows is not None and cols is not None:
        boundary=np.asarray(hole_vertices,dtype=np.int64)
        if len(boundary):
            # Minimum distance to an actual hole-boundary vertex (nominal radar meters).
            # Shortest point-to-segment distance would be smaller; do not label this as such.
            dr=(rows[:,None].astype(float)-rows[boundary][None,:])*13.9812174
            dc=(cols[:,None].astype(float)-cols[boundary][None,:])*3.8728370193856696
            nearest_hole=np.sqrt(np.min(dr*dr+dc*dc,axis=1))
    cycle_details=[]
    for ci,edge in enumerate(chords):
        u,v=map(int,pairs[edge])
        residue=q[edge]-(pot[v]-pot[u])
        vals,counts=np.unique(residue,return_counts=True)
        max_count=int(np.max(counts))
        modal_candidates=vals[counts==max_count]
        modal=int(modal_candidates[0]); tie=len(modal_candidates)>1
        if tie: ambiguous.append(ci)
        relative=residue-modal
        bad=np.flatnonzero(relative!=0)
        raw_nonzero+=int(np.count_nonzero(residue))
        relative_nonzero+=len(bad)
        raw_abs_max=max(raw_abs_max,int(np.max(np.abs(residue))))
        ce=cycle_edge_list(edge,pairs,parent,parent_edge,parent_sign)
        cycle_details.append({'cycle_index':ci,'chord_ifg':int(edge),'date_u':u,'date_v':v,
                              'modal_integer_closure':modal,'mode_support':max_count,
                              'nonmodal_points':int(len(bad)),'loop_ifgs':[[int(e),int(s)] for e,s in ce]})
        for p in bad:
            rel_records.append({'point_local_id':int(p),
                                'row': '' if rows is None else int(rows[p]),
                                'col': '' if cols is None else int(cols[p]),
                                'cycle_index':ci,'chord_ifg':int(edge),'date_u':u,'date_v':v,
                                'nearest_hole_vertex_m': '' if nearest_hole is None else round(float(nearest_hole[p]),3),
                                'integer_closure_raw':int(residue[p]),
                                'cycle_modal_closure':modal,'integer_closure_relative':int(relative[p])})
    write_csv(out/'relative_temporal_conflicts.csv',rel_records,
              ['point_local_id','row','col','nearest_hole_vertex_m','cycle_index','chord_ifg','date_u','date_v',
               'integer_closure_raw','cycle_modal_closure','integer_closure_relative'])
    (out/'temporal_cycle_definitions.json').write_text(json.dumps(cycle_details,indent=2),encoding='utf-8')
    tested=N_POINT*len(chords)
    result={'tested':tested,'fundamental_cycles':len(chords),
            'anchored_nonzero':raw_nonzero,'anchored_fraction':raw_nonzero/tested,
            'relative_nonzero':relative_nonzero,'relative_fraction':relative_nonzero/tested,
            'raw_max_abs_integer_closure':raw_abs_max,'unique_relative_bad_points':len({r['point_local_id'] for r in rel_records}),
            'ambiguous_mode_cycle_ids':ambiguous,
            'expected_from_previous_benchmark':{'tested':456605,'relative_nonzero':6,'anchored_fraction':0.2551855542536766},
            'matches_previous_relative_count':relative_nonzero==6}
    return result


def topology_audit(triangles,rows,cols):
    if triangles is None:return {'status':'SKIPPED','reason':'triangles file not supplied'}
    tri=np.asarray(triangles)
    if tri.shape!=(6129,3) or not np.all(np.isfinite(tri)) or not np.allclose(tri,np.rint(tri)):
        return {'status':'BLOCKED','reason':f'Expected local index triangles [6129,3], got {tri.shape}'}
    tri=np.rint(tri).astype(np.int64)
    if np.min(tri)<0 or np.max(tri)>=N_POINT:
        return {'status':'BLOCKED','reason':'Triangle vertices are not local indices in 0..3148'}
    edges=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]],axis=0)
    edges.sort(axis=1)
    ue,count=np.unique(edges,axis=0,return_counts=True)
    boundary=ue[count==1]
    deg=Counter(boundary.reshape(-1).tolist())
    max_degree=max(deg.values()) if deg else 0
    # Every boundary node of a clean 2-manifold with boundary must have degree 2.
    valid_boundary=max_degree==2 and min(deg.values())==2 if deg else False
    components=[]
    if valid_boundary:
        nbr=defaultdict(list)
        for u,v in boundary:
            nbr[int(u)].append(int(v));nbr[int(v)].append(int(u))
        seen=set()
        for start in nbr:
            if start in seen:continue
            loop=[]; prev=-1;cur=start
            while cur not in seen:
                seen.add(cur);loop.append(cur)
                a,b=nbr[cur];next_node=a if a!=prev else b
                prev,cur=cur,next_node
            components.append(loop)
    loop_stats=[]
    for loop in components:
        detail={'vertices':len(loop),'local_point_ids':[int(v) for v in loop]}
        if rows is not None and cols is not None:
            # Radar geometry for area comparison (nominal only).
            x=cols[loop].astype(float)*3.8728370193856696
            y=rows[loop].astype(float)*13.9812174
            detail['signed_nominal_area_m2']=float(0.5*np.sum(x*np.roll(y,-1)-np.roll(x,-1)*y))
        loop_stats.append(detail)
    loop_stats.sort(key=lambda d:-abs(d.get('signed_nominal_area_m2',d['vertices'])))
    e=int(len(ue));v=N_POINT;f=len(tri)
    result={'vertices':v,'triangles':f,'edges':e,'boundary_edges':int(len(boundary)),
            'edge_incidence_max':int(np.max(count)),'boundary_all_degree_2':valid_boundary,
            'boundary_loop_count':len(components),'boundary_loops':loop_stats,
            'euler_v_minus_e_plus_f':v-e+f,'cycle_rank':e-v+1,
            'topology_pass':e==9278 and len(boundary)==169 and len(components)==2 and v-e+f==0 and e-v+1==6130 and np.max(count)<=2}
    result['note']='Combinatorial geometry only; actual spatial integer-cycle residuals require saved edge corrections / constraints. Distance is to hole boundary VERTICES, nominal radar geometry.'
    return result


def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--roi-dir',type=Path,required=True,help='Existing ROI4 Native EMCF benchmark output directory (read-only)')
    p.add_argument('--out',type=Path,required=True,help='NEW separate audit output folder')
    p.add_argument('--inventory-only',action='store_true',help='Discover actual array paths/shapes before full audit')
    p.add_argument('--integers',help='IFG x point integer corrections q, .npy or .npz')
    p.add_argument('--unwrapped',help='Optional unwrapped IFG phase array, used WITH --wrapped to obtain q')
    p.add_argument('--wrapped',help='Optional wrapped IFG phase array, used WITH --unwrapped to obtain q')
    p.add_argument('--pairs',help='219x2 IFG master/slave date indices (0-based)')
    p.add_argument('--rows',help='3149 local point radar rows')
    p.add_argument('--cols',help='3149 local point radar cols')
    p.add_argument('--triangles',help='6129x3 LOCAL triangle vertex indices')
    p.add_argument('--npz-key',help='Array key for any single npz input, if needed')
    return p.parse_args()


def main():
    a=parse()
    if not a.roi_dir.is_dir():raise SystemExit(f'Missing ROI directory: {a.roi_dir}')
    if a.out.resolve()==a.roi_dir.resolve() or a.roi_dir.resolve() in a.out.resolve().parents:
        raise SystemExit('Output directory cannot be the benchmark directory or inside it. Keep originals untouched.')
    a.out.mkdir(parents=True,exist_ok=True)
    inv=inventory(a.roi_dir)
    (a.out/'roi_file_inventory.json').write_text(json.dumps(inv,indent=2),encoding='utf-8')
    print('ROI inventory written:',a.out/'roi_file_inventory.json')
    for f in inv:
        print(f['path'],f.get('shape',f.get('arrays','')),f.get('dtype',''))
    if a.inventory_only:
        print('Inventory only; no closure statistics computed.')
        return
    key=a.npz_key
    rows=read_array(a.rows,key)
    cols=read_array(a.cols,key)
    if (rows is None)!=(cols is None):raise SystemExit('Provide both --rows and --cols, or neither')
    if rows is not None:
        rows=np.asarray(rows).reshape(-1);cols=np.asarray(cols).reshape(-1)
        if len(rows)!=N_POINT or len(cols)!=N_POINT:
            raise SystemExit('Rows/cols must each have 3149 local island points in the exact same order as q')
    summary={'roi':[7155,14125,1024,1024],'island_rank':4,'read_only':True,
             'topology':topology_audit(read_array(a.triangles,key),rows,cols)}
    try:
        pairs=orient_pairs(read_array(a.pairs,key),N_DATE)
        if a.integers:
            q=read_array(a.integers,key)
            summary['integer_source']='supplied q array'
        elif a.unwrapped and a.wrapped:
            q,err=integers_from_phase(read_array(a.unwrapped,key),read_array(a.wrapped,key))
            summary['integer_source']='inferred from unwrapped/wrapped IFG arrays'
            summary['phase_parity_max_rad']=err
        else:
            raise ValueError('Provide --pairs and either --integers OR both --unwrapped and --wrapped; do not infer from summary.json')
        top=summary['topology']
        hole_vertices=top['boundary_loops'][1]['local_point_ids'] if top.get('topology_pass') and len(top['boundary_loops'])==2 else None
        summary['temporal']=temporal_audit(q,pairs,rows,cols,a.out,hole_vertices=hole_vertices)
        print('\nTEMPORAL:',json.dumps(summary['temporal'],indent=2))
        if not summary['temporal']['matches_previous_relative_count']:
            print('WARNING: relative nonzero count differs from 6; verify q definition, IFG ordering and temporal gauge. Do NOT repair yet.')
    except (ValueError,FileNotFoundError) as ex:
        summary['temporal']={'status':'BLOCKED','reason':str(ex)}
        print('\nTEMPORAL AUDIT BLOCKED:',ex)
        print('Supply explicit matrix paths based on the inventory. No conflict locations were inferred.')
    (a.out/'roi4_audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('TOPOLOGY:',json.dumps(summary['topology'],indent=2))
    print('SUMMARY:',a.out/'roi4_audit_summary.json')


if __name__=='__main__':
    main()
