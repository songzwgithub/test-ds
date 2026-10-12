"""Read-only, gauge-invariant integer-gradient audit for PS/DS node IFG A/B.

Requires saved whirlwind_point_ab outputs. Compares only real occupied 4-neighbor
edges and audits connected-component labels separately. Agreement/disagreement
is NOT a proof of geophysical correctness, especially on edges adjacent to
independently gauged solver components. No product/source phase array is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TAU = 2.0 * np.pi


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def grid_edges(grid):
    """Node IDs and grid positions for truly occupied horizontal/vertical edges."""
    g = np.asarray(grid)
    if g.ndim != 2 or not np.issubdtype(g.dtype, np.integer) or np.any(g < -1):
        raise ValueError('Expected 2-D integer node grid with -1 meaning empty')
    occupied = g[g >= 0]
    nnode = int(occupied.size)
    if nnode < 1 or not np.array_equal(np.sort(occupied), np.arange(nnode)):
        raise ValueError('Occupied node identifiers must occur exactly once and be contiguous')
    a, b = g[:, :-1], g[:, 1:]
    hor = (a >= 0) & (b >= 0)
    rh, ch = np.nonzero(hor)
    u_h, v_h = a[hor], b[hor]
    a, b = g[:-1, :], g[1:, :]
    ver = (a >= 0) & (b >= 0)
    rv, cv = np.nonzero(ver)
    u_v, v_v = a[ver], b[ver]
    edges = np.stack((np.r_[u_h, u_v], np.r_[v_h, v_v]), axis=1).astype(np.int32)
    coords = np.stack((np.r_[rh, rv], np.r_[ch, cv],
                       np.r_[np.zeros(len(rh), np.int32), np.ones(len(rv), np.int32)]), axis=1).astype(np.int32)
    # orientation=0: horizontal edge from (row,col) to (row,col+1)
    # orientation=1: vertical edge from (row,col) to (row+1,col)
    return edges, coords


def _labels_report(labels):
    lab = np.asarray(labels)
    if lab.ndim != 1 or not np.issubdtype(lab.dtype, np.integer) or np.any(lab < 0):
        raise ValueError('Invalid nonnegative integer connected-component labels')
    positive = lab[lab > 0]
    uniques, sizes = np.unique(positive, return_counts=True)
    return {
        'labeled_nodes': int(positive.size),
        'unlabeled_nodes': int(lab.size - positive.size),
        'labeled_fraction': float(positive.size / max(lab.size, 1)),
        'positive_component_count': int(len(uniques)),
        'largest_labeled_component_nodes': int(np.max(sizes)) if len(sizes) else 0,
        'zero_labels_do_not_imply_no_unwrapped_phase': True,
    }


def _stats(difference, subset):
    d = np.asarray(difference)
    sub = np.asarray(subset, bool)
    count = int(np.count_nonzero(sub))
    if count == 0:
        return {'edges':0, 'fraction_integer_gradient_different':None,
                'fraction_abs_difference_gt_1_cycle':None, 'max_abs_integer_difference':None}
    v = np.abs(d[sub])
    return {'edges':count,
            'fraction_integer_gradient_different':float(np.mean(v != 0)),
            'fraction_abs_difference_gt_1_cycle':float(np.mean(v > 1)),
            'max_abs_integer_difference':int(np.max(v))}


def assess(*, node_phase, node_grid, index_i, index_j, whirlwind_uw,
           snaphu_uw, whirlwind_cc, snaphu_cc, baseline_phase=None,
           tolerance_rad=1e-3):
    grid = np.asarray(node_grid)
    phase = np.asarray(node_phase)
    if phase.ndim != 2 or not (0 <= index_i < phase.shape[1] and
                              0 <= index_j < phase.shape[1] and index_i != index_j):
        raise ValueError('Invalid [node, acquisition] source or IFG epoch indices')
    edge, coords = grid_edges(grid)
    nnode = int(np.count_nonzero(grid >= 0))
    if phase.shape[0] != nnode:
        raise ValueError('Node grid does not match phase point count')
    raw = wrap(np.asarray(phase[:, index_j], np.float64) -
               np.asarray(phase[:, index_i], np.float64))
    if not np.all(np.isfinite(raw)):
        raise ValueError('Source node phase contains nonfinite values')
    output = {}
    cycles = {}
    for name, arr in (('whirlwind', whirlwind_uw), ('snaphu', snaphu_uw)):
        v = np.asarray(arr)
        if v.shape != (nnode,) or not np.all(np.isfinite(v)):
            raise ValueError(f'{name} node IFG phase missing or nonfinite')
        parity = float(np.max(np.abs(wrap(np.asarray(v, np.float64) - raw))))
        if parity > tolerance_rad:
            raise ValueError(f'{name} input phase parity mismatch: {parity:.6g} rad')
        cycles[name] = np.rint((np.asarray(v, np.float64) - raw)/TAU).astype(np.int32)
        output[name+'_wrapped_parity_max_rad'] = parity

    for name, lab in (('whirlwind', whirlwind_cc), ('snaphu', snaphu_cc)):
        if np.asarray(lab).shape != (nnode,):
            raise ValueError(f'{name} node labels shape mismatch')
        output[name+'_labels'] = _labels_report(lab)

    if baseline_phase is not None:
        base = np.asarray(baseline_phase)
        if base.shape != phase.shape:
            raise ValueError('Statcost baseline and linked node phase shapes differ')
        b0 = np.asarray(base[:, index_i], np.float64)
        b1 = np.asarray(base[:, index_j], np.float64)
        max_err = max(float(np.max(np.abs(wrap(b0-phase[:,index_i])))),
                      float(np.max(np.abs(wrap(b1-phase[:,index_j])))))
        if max_err > tolerance_rad:
            raise ValueError(f'Statcost source-node parity mismatch: {max_err:.6g} rad')
        cycles['statcost'] = np.rint((b1 - b0 - raw)/TAU).astype(np.int32)
        output['statcost_source_parity_max_rad'] = max_err
    u, v = edge[:,0], edge[:,1]
    gradients = {name: k[v]-k[u] for name, k in cycles.items()}
    labw = np.asarray(whirlwind_cc)
    labs = np.asarray(snaphu_cc)
    w_inside = (labw[u] > 0) & (labw[u] == labw[v])
    s_inside = (labs[u] > 0) & (labs[u] == labs[v])
    all_edges = np.ones(len(edge), bool)
    sets = {'all_directly_observed_edges':all_edges,
            'within_whirlwind_labeled_components':w_inside,
            'within_snaphu_labeled_components':s_inside,
            'within_both_labeled_components':w_inside & s_inside}
    comparisons = {}
    differences = {}
    for a, b in [('whirlwind','snaphu'),('whirlwind','statcost'),('snaphu','statcost')]:
        if a not in gradients or b not in gradients:
            continue
        name = a+'_minus_'+b
        differences[name] = gradients[a] - gradients[b]
        comparisons[name] = {key:_stats(differences[name], selection)
                             for key, selection in sets.items()}

    output.update({
      'status':'DIAGNOSTIC_ONLY_NOT_PRODUCTION',
      'input_origin':'PHASE_LINKED_PS_DS_COARSE_NODES',
      'node_count':nnode,
      'accepted_observed_4_neighbor_edges':int(len(edge)),
      'epoch_indices_0based':[int(index_i),int(index_j)],
      'comparisons':comparisons,
      'scientific_note':('Direct-edge gradient differences cancel component-wise integer offsets only '
                         'within each genuinely connected support. All-edge comparisons are diagnostic: '
                         'a missing/zero solver label is not evidence of correctness. No result is truth.'),
      'existing_production_modified':False,
    })
    auxiliary = {'direct_edge_node_pairs':edge, 'direct_edge_grid_positions':coords,
                 'edge_inside_whirlwind_component':w_inside,
                 'edge_inside_snaphu_component':s_inside,
                 **{'edge_gradient_difference_'+key:val for key,val in differences.items()}}
    return output, auxiliary


def main(argv=None):
    p = argparse.ArgumentParser(description='Compare saved Whirlwind/SNAPHU integer jumps on observed PS/DS edges')
    p.add_argument('--node-phase',required=True,type=Path)
    p.add_argument('--node-grid',required=True,type=Path)
    p.add_argument('--benchmark-dir',required=True,type=Path)
    p.add_argument('--baseline-dir',type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    args = p.parse_args(argv)
    d = args.benchmark_dir
    audit = json.loads((d/'input_audit.json').read_text(encoding='utf-8'))
    ij = audit.get('acquisition_index_0based')
    if not isinstance(ij, list) or len(ij)!=2:
        raise ValueError('Benchmark input audit does not contain acquisition pair indices')
    ph = np.load(args.node_phase, mmap_mode='r', allow_pickle=False)
    g = np.load(args.node_grid, mmap_mode='r', allow_pickle=False)
    baseline = None
    if args.baseline_dir is not None:
        path = args.baseline_dir/'coarse_acquisition_phase_unwrapped_rad.npy'
        baseline = np.load(path, mmap_mode='r', allow_pickle=False)
    def load(name):
        return np.load(d/(name+'.npy'), mmap_mode='r',allow_pickle=False)
    summary, arrays = assess(node_phase=ph,node_grid=g,index_i=ij[0],index_j=ij[1],
          whirlwind_uw=load('whirlwind_node_ifg_unwrapped_rad'),
          snaphu_uw=load('snaphu_node_ifg_unwrapped_rad'),
          whirlwind_cc=load('whirlwind_node_conncomp'),
          snaphu_cc=load('snaphu_node_conncomp'),baseline_phase=baseline)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for key, val in arrays.items():
        np.save(out/(key+'.npy'),val)
    # A small, inspectable CSV for mismatching direct edges, never a phase write.
    primary = arrays['edge_gradient_difference_whirlwind_minus_snaphu']
    idx = np.flatnonzero(primary != 0)[:2000]
    with (out/'integer_disagreement_edges.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['edge_index_0based','node_a','node_b','row','col','orientation_0h_1v',
                    'whirlwind_minus_snaphu_integer_gradient'])
        for n in idx:
            e=arrays['direct_edge_node_pairs'][n]
            pos=arrays['direct_edge_grid_positions'][n]
            w.writerow([int(n),int(e[0]),int(e[1]),int(pos[0]),int(pos[1]),int(pos[2]),int(primary[n])])
    summary['benchmark_dir']=str(d)
    summary['baseline_dir']=str(args.baseline_dir) if args.baseline_dir is not None else None
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
