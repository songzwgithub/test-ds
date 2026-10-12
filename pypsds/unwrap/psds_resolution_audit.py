"""Read-only native PS/DS spatial-support audit before any integer unwrapping.

This uses PointPhaseStack's wrapped, phase-linked acquisitions. It compares
PS-only, DS-only, and combined sampling on identical nominal radar grids at
multiple resolutions, and audits the noncommutativity of spatial averaging
and acquisition differencing. It never unwraps, fills gaps, or modifies
production data. Radar-grid spacings are approximate metric coordinates, not
independently validated geodesic distances or calibrated coherence statistics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy import ndimage

TAU = 2.0 * np.pi
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def _grid_spec(rows, cols, spacing_y, spacing_x, cell_m):
    if not all(np.isfinite(x) and x > 0 for x in (spacing_y, spacing_x, cell_m)):
        raise ValueError('Grid sizes and radar spacings must be finite and positive')
    y0, x0 = int(np.min(rows)), int(np.min(cols))
    nr = int(math.floor((int(np.max(rows)) - y0) * spacing_y / cell_m)) + 1
    nc = int(math.floor((int(np.max(cols)) - x0) * spacing_x / cell_m)) + 1
    if nr < 1 or nc < 1 or nr * nc > 100_000_000:
        raise ValueError('Invalid or excessively large requested grid')
    return y0, x0, nr, nc


@njit(cache=True)
def _count_native(rows, cols, types, y0, x0, sy, sx, cell_m, nr, nc):
    ps = np.zeros(nr * nc, np.int32)
    ds = np.zeros(nr * nc, np.int32)
    for p in range(len(rows)):
        y = int(math.floor((int(rows[p]) - y0) * sy / cell_m))
        x = int(math.floor((int(cols[p]) - x0) * sx / cell_m))
        if y < 0 or y >= nr or x < 0 or x >= nc:
            raise ValueError('Point outside grid')
        ix = y * nc + x
        if types[p] == 1:
            ps[ix] += 1
        elif types[p] == 2:
            ds[ix] += 1
        else:
            raise ValueError('Point type must be PS=1 or DS=2')
    return ps, ds


@njit(cache=True)
def _accumulate_native(rows, cols, types, p0, p1, cellmap, sums,
                       y0, x0, sy, sx, cell_m, nc):
    for p in range(len(rows)):
        y = int(math.floor((int(rows[p]) - y0) * sy / cell_m))
        x = int(math.floor((int(cols[p]) - x0) * sx / cell_m))
        ix = int(cellmap[y * nc + x])
        if ix < 0:
            raise RuntimeError('Observed native point mapped to empty grid')
        typ = 0 if types[p] == 1 else 1
        a, b = float(p0[p]), float(p1[p])
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError('Non-finite point phase in selected dates')
        sums[typ, 0, 0, ix] += math.cos(a)
        sums[typ, 0, 1, ix] += math.sin(a)
        sums[typ, 1, 0, ix] += math.cos(b)
        sums[typ, 1, 1, ix] += math.sin(b)
        sums[typ, 2, 0, ix] += math.cos(b-a)
        sums[typ, 2, 1, ix] += math.sin(b-a)


def _topology(mask):
    # Limit all topology computation to observed cells. An empty cell NEVER
    # connects or contributes a spatial observation.
    m = np.asarray(mask, bool)
    labeled, ncomp = ndimage.label(m, structure=CROSS)
    count = np.bincount(labeled[m], minlength=ncomp + 1)[1:]
    h = int(np.count_nonzero(m[:, :-1] & m[:, 1:]))
    v = int(np.count_nonzero(m[:-1, :] & m[1:, :]))
    n = int(np.count_nonzero(m))
    return {
        'occupied_cells': n,
        'occupied_fraction_of_footprint': float(n / m.size),
        'component_count': int(ncomp),
        'largest_component_cells': int(count.max()) if count.size else 0,
        'largest_component_fraction_of_occupied': float(count.max()/n) if count.size else 0.,
        'singleton_components': int(np.count_nonzero(count == 1)),
        'component_sizes_top5': sorted(map(int, count), reverse=True)[:5],
        'observed_four_neighbor_edges': h + v,
    }


def _percentile(values):
    z = np.asarray(values)
    if not z.size:
        return None
    return [float(x) for x in np.percentile(z, [50, 90, 95, 99])]


def _circular_phase_audit(sums, count, minimum_points):
    # Entry [0]: acquisition i; [1]: acquisition j; [2]: direct point IFG.
    mask = count >= minimum_points
    n = int(np.count_nonzero(mask))
    if n == 0:
        return {'cells_tested': 0, 'reason': 'No cell meets point-count threshold'}
    z = sums[:, :, mask]
    a = np.arctan2(z[0, 1], z[0, 0]); b = np.arctan2(z[1, 1], z[1, 0])
    direct = np.arctan2(z[2, 1], z[2, 0])
    discrepancy = np.abs(np.arctan2(np.sin(b-a-direct), np.cos(b-a-direct)))
    resultant = np.hypot(z[2, 0], z[2, 1]) / count[mask]
    return {
        'cells_tested': n,
        'minimum_points_per_cell': int(minimum_points),
        'point_ifg_resultant_p50_p90_p95_p99': _percentile(resultant),
        'fraction_point_ifg_resultant_below_0p5': float(np.mean(resultant < .5)),
        'aggregation_discrepancy_rad_p50_p90_p95_p99': _percentile(discrepancy),
        'fraction_aggregation_discrepancy_above_pi_over_4': float(np.mean(discrepancy > np.pi/4)),
        'fraction_aggregation_discrepancy_above_pi_over_2': float(np.mean(discrepancy > np.pi/2)),
        'definition': 'wrapped(angle(sum phasors_j) - angle(sum phasors_i) - angle(sum native IFG phasors))',
        'warning': 'Circular resultant is NOT interferometric sample coherence; discrepancy is a change of estimator, not proof of unwrapping error',
    }


def audit_resolutions(*, rows, cols, point_type, phase, date_i, date_j,
                      row_spacing_m, col_spacing_m, resolutions=(50., 100., 200.),
                      batch=131072, minimum_points=3, with_phase=True,
                      reference_grid=None, reference_resolution_m=200.):
    """Analyze all grid resolutions, reading each phase-stack point batch once."""
    rows, cols, typ = (np.asarray(x) for x in (rows, cols, point_type))
    if any(x.ndim != 1 or x.shape != rows.shape for x in (rows, cols, typ)):
        raise ValueError('Rows, columns and point types must be aligned vectors')
    if rows.size == 0:
        raise ValueError('No points')
    if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(cols.dtype, np.integer):
        raise ValueError('Radar rows/columns must be integers')
    if not np.all((typ == 1) | (typ == 2)):
        raise ValueError('Point types must be 1=PS and 2=DS')
    if batch < 1 or minimum_points < 2:
        raise ValueError('Batch must be positive and minimum points >= 2')
    if with_phase:
        if phase is None or phase.ndim != 2 or phase.shape[0] != len(rows):
            raise ValueError('PointPhaseStack phase shape differs from point metadata')
        if not (0 <= date_i < phase.shape[1] and 0 <= date_j < phase.shape[1] and date_i != date_j):
            raise ValueError('Invalid pair indices')
    sizes = [float(x) for x in resolutions]
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError('Resolutions must be unique and non-empty')
    results = []
    intermediates = []
    for cell_m in sizes:
        y0, x0, nr, nc = _grid_spec(rows,cols,row_spacing_m,col_spacing_m,cell_m)
        ps, ds = _count_native(rows,cols,typ,y0,x0,row_spacing_m,col_spacing_m,cell_m,nr,nc)
        total=ps+ds
        if int(np.sum(total, dtype=np.int64)) != len(rows):
            raise RuntimeError('Point counts were not conserved')
        occupancy=(total>0)
        if reference_grid is not None and abs(cell_m - reference_resolution_m) < 1e-9:
            ref=np.asarray(reference_grid)
            if ref.shape != (nr,nc) or not np.array_equal(ref >= 0, occupancy.reshape(nr,nc)):
                raise RuntimeError('Grid footprint or occupied nodes differ from the existing production 200m grid; stop before interpreting resolution comparisons')
        compact=np.full(len(total),-1,np.int32)
        compact[occupancy]=np.arange(int(occupancy.sum()),dtype=np.int32)
        groups={
            'ps_only':_topology((ps>0).reshape(nr,nc)),
            'ds_only':_topology((ds>0).reshape(nr,nc)),
            'combined':_topology(occupancy.reshape(nr,nc)),
        }
        count_ps=ps[occupancy]
        count_ds=ds[occupancy]
        record={
            'nominal_grid_size_m':float(cell_m),
            'grid_shape':[nr,nc],
            'grid_cells':int(nr*nc),
            'source_point_count':int(len(rows)),
            'ps_point_count':int(np.count_nonzero(typ==1)),
            'ds_point_count':int(np.count_nonzero(typ==2)),
            'cells_with_both_ps_and_ds':int(np.count_nonzero((ps>0)&(ds>0))),
            'cells_added_by_ds_over_ps_only':int(np.count_nonzero((ps==0)&(ds>0))),
            'cells_added_by_ps_over_ds_only':int(np.count_nonzero((ds==0)&(ps>0))),
            'groups':groups,
        }
        if with_phase:
            sums=np.zeros((2,3,2,len(count_ps)),np.float64)
            intermediates.append((cell_m,y0,x0,nc,compact,sums,count_ps,count_ds))
        results.append(record)
        print(f'[SUPPORT] grid={cell_m:g}m nodes={len(count_ps):,} components={groups["combined"]["component_count"]:,}',flush=True)
    if with_phase:
        for start in range(0,len(rows),batch):
            stop=min(len(rows),start+batch)
            # Read a contiguous block ONCE, so access to mmap phase[point,75]
            # is sequential on disk, not one 8-GiB traversal per resolution.
            ph=np.asarray(phase[start:stop,:],dtype=np.float32)
            a=ph[:,date_i].copy();b=ph[:,date_j].copy()
            for cell_m,y0,x0,nc,compact,sums,_,_ in intermediates:
                _accumulate_native(rows[start:stop],cols[start:stop],typ[start:stop],
                                   a,b,compact,sums,y0,x0,row_spacing_m,col_spacing_m,cell_m,nc)
            if stop==len(rows) or stop % (batch*20)==0:
                print(f'[PHASOR] {stop:,}/{len(rows):,}',flush=True)
        for record,item in zip(results,intermediates):
            _,_,_,_,_,sums,ps,ds=item
            record['phase_aggregation_audit']={
                'ps_only':_circular_phase_audit(sums[0],ps,minimum_points),
                'ds_only':_circular_phase_audit(sums[1],ds,minimum_points),
                'combined':_circular_phase_audit(sums[0]+sums[1],ps+ds,minimum_points),
            }
    return {'status':'DIAGNOSTIC_ONLY_NO_UNWRAPPING',
            'source':'PointPhaseStack phase_rad.npy (wrapped phase-linked PS/DS)',
            'acquisition_indices_0based':[int(date_i),int(date_j)] if with_phase else None,
            'phase_aggregation_checked':bool(with_phase),
            'distance_caveat':'Nominal radar spacing from GAMMA parameter file, not geocoded ground distances',
            'integer_gauge_solved':False,
            'results':results}


def _read_network(path: Path, ndate: int):
    edges = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 2:
            raise ValueError(f"Invalid temporal network row {lineno}")
        a, b = int(fields[0])-1, int(fields[1])-1
        if not (0 <= a < ndate and 0 <= b < ndate and a != b):
            raise ValueError(f"Invalid 1-based temporal network edge {lineno}")
        edges.append((a,b))
    if not edges:
        raise ValueError("No temporal network edges")
    return edges


def main(argv=None):
    p=argparse.ArgumentParser(description='Read-only high-resolution PS/DS spatial-support and phase aggregation audit')
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--grid-sizes-m',type=float,nargs='+',default=[50.,100.,200.])
    p.add_argument('--pair-index',type=int,default=None,help='0-based network.itab pair; default selects earliest pair with both acquisition indices > 0')
    p.add_argument('--batch',type=int,default=131072)
    p.add_argument('--minimum-points',type=int,default=3)
    p.add_argument('--no-phase',action='store_true')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    from pypsds.context import open_from_config
    from pypsds.geometry.inputs import resolve_geometry_inputs
    from pypsds.gamma.geometry import geometry_from_par
    cfg,_,paths,stack,_=open_from_config(args.config)
    geom=geometry_from_par(resolve_geometry_inputs(cfg,paths).reference_rslc_par)
    pp=Path(paths.output_dir)/'processing'/'point_phase_stack'
    rows=np.load(pp/'rows.npy',mmap_mode='r',allow_pickle=False)
    cols=np.load(pp/'cols.npy',mmap_mode='r',allow_pickle=False)
    typ=np.load(pp/'point_type.npy',mmap_mode='r',allow_pickle=False)
    pairs=_read_network(Path(paths.output_dir)/'processing'/'network'/'network.itab',len(stack.dates))
    if args.pair_index is None:
        idx=next((i for i,(a,b) in enumerate(pairs) if a!=0 and b!=0),0)
    else:
        idx=args.pair_index
    if not 0<=idx<len(pairs):
        p.error('pair index out of range')
    a,b=pairs[idx]
    phase=None if args.no_phase else np.load(pp/'phase_rad.npy',mmap_mode='r',allow_pickle=False)
    production_grid=Path(paths.output_dir)/'processing'/'stamps3d_unwrap'/'coarse_node_grid.npy'
    reference_grid=np.load(production_grid,mmap_mode='r',allow_pickle=False) if production_grid.exists() else None
    report=audit_resolutions(rows=rows,cols=cols,point_type=typ,phase=phase,
        date_i=a,date_j=b,row_spacing_m=geom.azimuth_spacing_m,
        col_spacing_m=geom.ground_range_spacing_m,
        resolutions=args.grid_sizes_m,batch=args.batch,
        minimum_points=args.minimum_points,with_phase=not args.no_phase,
        reference_grid=reference_grid) 
    report.update({'dates':[str(stack.dates[a]),str(stack.dates[b])],
                   'temporal_pair_index_0based':int(idx),
                   'ps_point_type':1,'ds_point_type':2,
                   'production_modified':False,
                   'reference_200m_grid_verified':bool(reference_grid is not None and 200.0 in args.grid_sizes_m)})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    out=args.output_dir/'summary.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
