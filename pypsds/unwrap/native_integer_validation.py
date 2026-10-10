"""Read-only, gauge-invariant native-edge integer comparison against StaMPS/SNAPHU.

Relative component integer gauges are not observable from the wrapped phase.
This diagnostic compares gradients on exactly the same accepted native edges,
so component-wide integer gauge offsets cancel. It does NOT prove a solution is
physically correct and it never writes back to production products.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

TWOPI = 2.0 * np.pi


def _wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def _load_npy(directory, name, *, mmap=True):
    path = Path(directory) / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Required array missing: {path}")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def load_baseline_cycles(baseline_dir, node_phase, *, node_grid_path=None,
                         allow_unverified_cycle_only=False, tolerance_rad=1e-3):
    """Require node-order evidence: phase congruence or identical explicit grid."""
    directory = Path(baseline_dir)
    unwrapped_path = directory / 'coarse_acquisition_phase_unwrapped_rad.npy'
    cycle_path = directory / 'node_acquisition_integer_cycles.npy'
    shape = tuple(node_phase.shape)
    if unwrapped_path.is_file():
        reference = np.load(unwrapped_path, mmap_mode='r', allow_pickle=False)
        if reference.shape != shape:
            raise ValueError(f'Baseline unwrapped shape {reference.shape} != node phase {shape}')
        out = np.empty(shape, np.int32)
        worst = 0.0
        for start in range(0, shape[0], 4096):
            end = min(shape[0], start + 4096)
            ph = np.asarray(node_phase[start:end], np.float64)
            uw = np.asarray(reference[start:end], np.float64)
            if not (np.isfinite(ph).all() and np.isfinite(uw).all()):
                raise ValueError('Baseline or node phase contains NaN/Inf')
            worst = max(worst, float(np.max(np.abs(_wrap(uw-ph)))))
            out[start:end] = np.rint((uw-ph)/TWOPI).astype(np.int32)
        if worst > tolerance_rad:
            raise ValueError(
                f'Baseline unwrapped phase is not wrapped-congruent with the '
                f'candidate node phase (max error {worst:.5g} rad). '
                'Do not compare nodes with incompatible identities/phases.'
            )
        return out, {'source':str(unwrapped_path), 'node_order_verification':'wrapped_phase_parity',
                     'maximum_wrap_difference_rad':worst}
    if not cycle_path.is_file():
        raise FileNotFoundError(
            f'No baseline coarse unwrapped phase or node cycles in {directory}. '
            'Provide the genuine statcost baseline directory, not final point outputs.'
        )
    baseline = np.load(cycle_path, mmap_mode='r', allow_pickle=False)
    if baseline.shape != shape:
        raise ValueError(f'Baseline integer cycles shape {baseline.shape} != {shape}')
    reference_grid = directory / 'coarse_node_grid.npy'
    verified = False
    if reference_grid.is_file() and node_grid_path is not None:
        a = np.load(reference_grid, mmap_mode='r', allow_pickle=False)
        b = np.load(node_grid_path, mmap_mode='r', allow_pickle=False)
        verified = a.shape == b.shape and bool(np.array_equal(a, b))
        if not verified:
            raise ValueError('Baseline and candidate coarse node grids do not match')
    if not verified and not allow_unverified_cycle_only:
        raise ValueError('Cycles-only baseline lacks a proven node ordering. '
                         'Provide a baseline unwrapped phase or a matching '
                         'coarse_node_grid.npy. Use --allow-unverified-cycle-only '
                         'only after independent node-ID verification.')
    if not np.issubdtype(baseline.dtype, np.integer):
        raise ValueError('Baseline cycles must have integer dtype')
    return np.asarray(baseline, np.int32), {
        'source':str(cycle_path),
        'node_order_verification':'identical_grid' if verified else 'UNVERIFIED_EXPLICIT_OVERRIDE',
        'maximum_wrap_difference_rad':None,
    }


def _quantiles(x):
    if len(x) == 0:
        return None
    return [float(v) for v in np.percentile(x, [50, 90, 95, 99])]


def _summary_edge_residual(values, select):
    arr = np.asarray(values)[np.asarray(select, bool)]
    if arr.size == 0:
        return {'edges':0, 'fraction_edges_with_any_epoch_mismatch':None,
                'fraction_mismatched_edge_epochs':None,
                'per_edge_mismatch_fraction_p50_p90_p95_p99':None}
    return {
        'edges':int(arr.size),
        'fraction_edges_with_any_epoch_mismatch':float(np.mean(arr > 0)),
        'fraction_mismatched_edge_epochs':float(np.mean(arr)),
        'per_edge_mismatch_fraction_p50_p90_p95_p99':_quantiles(arr),
    }


def _linear_phase_rate(node_phase, cycles, dates, mm_per_rad):
    if len(dates) < 2:
        raise ValueError('At least 2 dates needed')
    start = datetime.strptime(str(dates[0]), '%Y%m%d')
    x = np.array([(datetime.strptime(str(d), '%Y%m%d')-start).days
                  for d in dates], np.float64)/365.2425
    x = x-x.mean()
    denominator = float(x@x)
    if denominator <= 0:
        raise ValueError('No temporal span')
    weights = x/denominator
    return (np.asarray(node_phase, np.float64) @ weights +
            TWOPI*(np.asarray(cycles, np.float64) @ weights)) * float(mm_per_rad)


def evaluate(*, node_phase, baseline_cycles, primary, secondary, component_id,
             component_roots, core_mask, agreement_fraction, node_pairs,
             edge_integer, tree_edge_primary, tree_edge_secondary, dates,
             mm_per_rad=4.413824900882934, agreement_threshold=.99,
             batch=4096):
    """Compare same native observed edges; align node cycles only for reporting.

    No graph-wide or component-wide phase is altered. All edge residuals are
    gauge invariant; the root-aligned node difference is a diagnostic only.
    """
    phase = np.asarray(node_phase)
    if phase.ndim != 2:
        raise ValueError('Node phase must be [node, acquisition]')
    nnode, nt = phase.shape
    cyc = [np.asarray(z) for z in (baseline_cycles, primary, secondary)]
    if any(z.shape != (nnode, nt) for z in cyc):
        raise ValueError('Integer cycle arrays and node phase have different dimensions')
    labels = np.asarray(component_id, np.int32)
    roots = np.asarray(component_roots, np.int32)
    core = np.asarray(core_mask, bool)
    agree = np.asarray(agreement_fraction, np.float64)
    if any(x.shape != (nnode,) for x in (labels, core, agree)):
        raise ValueError('Node metadata length mismatch')
    if roots.ndim != 1 or (labels < 0).any() or (labels >= len(roots)).any():
        raise ValueError('Invalid component labels')
    if (roots < 0).any() or (roots >= nnode).any() or not np.array_equal(labels[roots], np.arange(len(roots))):
        raise ValueError('Component roots do not match component IDs')
    links = np.asarray(node_pairs, np.int32)
    k = np.asarray(edge_integer)
    if links.shape != (len(k), 2) or k.ndim != 2 or k.shape[1] != nt:
        raise ValueError('Native integer edge histories do not match node pairs')
    if np.any(links < 0) or np.any(links >= nnode) or np.any(links[:, 0] == links[:, 1]):
        raise ValueError('Invalid edge endpoints')
    if np.any(labels[links[:, 0]] != labels[links[:, 1]]):
        raise ValueError('Accepted native edge connects different solution components')
    ta, tb = np.asarray(tree_edge_primary, bool), np.asarray(tree_edge_secondary, bool)
    if ta.shape != (len(links),) or tb.shape != (len(links),):
        raise ValueError('Spanning-tree mask has wrong shape')
    if len(dates) != nt:
        raise ValueError('Date count differs from acquisition count')
    ecount = len(links)
    residual = np.zeros((3, ecount), np.float32)
    mismatch_epoch = np.zeros((3, nt), np.int64)
    for start in range(0, ecount, batch):
        end = min(start+batch, ecount)
        a, b = links[start:end, 0], links[start:end, 1]
        edge_obs = np.asarray(k[start:end], np.int32)
        for j, z in enumerate(cyc):
            err = np.asarray(z[b], np.int32) - np.asarray(z[a], np.int32) - edge_obs
            failed = err != 0
            residual[j, start:end] = np.mean(failed, axis=1).astype(np.float32)
            mismatch_epoch[j] += failed.sum(axis=0)
    strict_nodes = core & (agree >= agreement_threshold)
    both_strict = strict_nodes[links[:, 0]] & strict_nodes[links[:, 1]]
    non_tree = ~ta
    result = {
        'status':'DIAGNOSTIC_ONLY_NOT_PRODUCTION',
        'points_are_coarse_nodes':int(nnode),
        'epochs':int(nt),
        'accepted_native_edges':int(ecount),
        'candidate_components':int(len(roots)),
        'node_fraction_two_core_and_forest_agreement':float(np.mean(strict_nodes)),
        'edge_fraction_both_endpoints_strict':float(np.mean(both_strict)) if ecount else None,
        'comparisons':{},
        'scientific_note':(
            'Native-edge integer closure is internal consistency, not proof of '
            'the true spatial gradient. No component-wise integer gauge is '
            'determined independently; baseline alignment is only used to '
            'make diagnostic node differences comparable.'),
    }
    names = ('baseline_statcost','native_primary','native_secondary')
    for j, name in enumerate(names):
        result['comparisons'][name] = {
            'all_accepted_edges':_summary_edge_residual(residual[j], np.ones(ecount, bool)),
            'non_tree_edges_primary':_summary_edge_residual(residual[j], non_tree),
            'edges_with_both_strict_endpoints':_summary_edge_residual(residual[j], both_strict),
            'mismatched_edges_by_acquisition_count':[int(x) for x in mismatch_epoch[j]],
        }
    diff_vs_baseline = {
        'fraction_edges_baseline_worse_than_native_primary':float(np.mean(residual[0] > residual[1])) if ecount else None,
        'fraction_edges_baseline_better_than_native_primary':float(np.mean(residual[0] < residual[1])) if ecount else None,
        'fraction_edges_equal':float(np.mean(residual[0] == residual[1])) if ecount else None,
        'fraction_non_tree_edges_baseline_worse':float(np.mean(residual[0,non_tree] > residual[1,non_tree])) if np.any(non_tree) else None,
    }
    result['relative_comparison'] = diff_vs_baseline
    b, pa, pb = cyc
    # The two forests are rooted at the same arbitrary node per component.
    # Gauge-aligned integer differences are meaningful only within components.
    rr = roots[labels]
    b_rel = np.asarray(b, np.int32) - np.asarray(b[rr], np.int32)
    p_rel = np.asarray(pa, np.int32) - np.asarray(pa[rr], np.int32)
    s_rel = np.asarray(pb, np.int32) - np.asarray(pb[rr], np.int32)
    node_change_fraction = np.mean(b_rel != p_rel, axis=1).astype(np.float32)
    result['root_aligned_node_difference'] = {
        'fraction_nodes_changed_any_acquisition':float(np.mean(node_change_fraction > 0)),
        'fraction_strict_nodes_changed_any_acquisition':float(np.mean((node_change_fraction > 0)[strict_nodes])) if np.any(strict_nodes) else None,
        'node_difference_fraction_p50_p90_p95_p99':_quantiles(node_change_fraction),
        'note':'Only a root-aligned comparison against baseline, not a new absolute gauge.',
    }
    slopes = [_linear_phase_rate(phase, z, dates, mm_per_rad) for z in cyc]
    for j, name in enumerate(names):
        edge_rate = np.abs(slopes[j][links[:, 1]] - slopes[j][links[:, 0]])
        result['comparisons'][name]['native_edge_absolute_los_rate_gradient_mm_per_year_p50_p90_p95_p99'] = _quantiles(edge_rate)
        if j == 0:
            baseline_gradient = edge_rate.astype(np.float32)
        elif j == 1:
            primary_gradient = edge_rate.astype(np.float32)
        else:
            secondary_gradient = edge_rate.astype(np.float32)
    # Component-level aggregate for locating *where* disagreement is concentrated.
    component_edge = labels[links[:, 0]]
    component_size = np.bincount(labels, minlength=len(roots))
    comp_edge_count = np.bincount(component_edge, minlength=len(roots))
    component_baseline_mean = np.bincount(component_edge, weights=residual[0], minlength=len(roots)) / np.maximum(comp_edge_count, 1)
    component_primary_mean = np.bincount(component_edge, weights=residual[1], minlength=len(roots)) / np.maximum(comp_edge_count, 1)
    comp_node_change = np.bincount(labels, weights=node_change_fraction > 0, minlength=len(roots)) / np.maximum(component_size, 1)
    auxiliary = {
        'edge_baseline_mismatch_fraction':residual[0],
        'edge_native_primary_mismatch_fraction':residual[1],
        'edge_native_secondary_mismatch_fraction':residual[2],
        'edge_both_strict_endpoints':both_strict,
        'node_root_aligned_difference_fraction':node_change_fraction,
        'native_baseline_edge_rate_gradient_mm_per_year':baseline_gradient,
        'native_primary_edge_rate_gradient_mm_per_year':primary_gradient,
        'native_secondary_edge_rate_gradient_mm_per_year':secondary_gradient,
        'component_size_nodes':component_size,
        'component_accepted_edge_count':comp_edge_count,
        'component_baseline_native_mismatch_mean':component_baseline_mean,
        'component_primary_native_mismatch_mean':component_primary_mean,
        'component_node_difference_fraction':comp_node_change,
    }
    return result, auxiliary


def main(argv=None):
    p = argparse.ArgumentParser(description='Read-only, native edge vs statcost integer comparison')
    p.add_argument('--node-phase', type=Path, required=True)
    p.add_argument('--node-grid', type=Path)
    p.add_argument('--baseline-dir', type=Path, required=True)
    p.add_argument('--candidate-dir', type=Path, required=True)
    p.add_argument('--network-dir', type=Path, required=True)
    p.add_argument('--dates', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--mm-per-rad', type=float, default=4.413824900882934)
    p.add_argument('--allow-unverified-cycle-only', action='store_true')
    args = p.parse_args(argv)
    phase = np.load(args.node_phase, mmap_mode='r', allow_pickle=False)
    b, evidence = load_baseline_cycles(
        args.baseline_dir, phase, node_grid_path=args.node_grid,
        allow_unverified_cycle_only=args.allow_unverified_cycle_only)
    c = args.candidate_dir
    n = args.network_dir
    ids = _load_npy(c, 'retained_network_edge_indices')
    network_edges = _load_npy(n, 'node_pairs')
    observed = _load_npy(n, 'directly_adjacent')
    native_support = _load_npy(n, 'supported_edge_mask')
    if ids.ndim != 1 or (ids < 0).any() or (ids >= len(network_edges)).any() or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid candidate edge index mapping')
    if not (np.all(observed[ids]) and np.all(native_support[ids])):
        raise ValueError('Candidate edges contain unsupported or non-observed links')
    dates = args.dates.read_text(encoding='utf8').split()
    print('Validating statcost and native candidate on identical observed edges...', flush=True)
    summary, data = evaluate(
        node_phase=phase, baseline_cycles=b,
        primary=_load_npy(c, 'cycles_primary'),
        secondary=_load_npy(c, 'cycles_secondary'),
        component_id=_load_npy(c, 'component_id'),
        component_roots=_load_npy(c, 'component_root_node'),
        core_mask=_load_npy(c, 'node_two_core_mask'),
        agreement_fraction=_load_npy(c, 'node_forest_agreement_fraction'),
        node_pairs=network_edges[ids],
        edge_integer=_load_npy(c, 'candidate_edge_integer_histories'),
        tree_edge_primary=_load_npy(c, 'tree_edge_mask_primary'),
        tree_edge_secondary=_load_npy(c, 'tree_edge_mask_secondary'),
        dates=dates, mm_per_rad=args.mm_per_rad,
    )
    summary['baseline_evidence'] = evidence
    summary['source_files'] = {key:str(value) for key,value in {
        'node_phase':args.node_phase, 'baseline':args.baseline_dir,
        'candidate':c, 'network':n, 'dates':args.dates}.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in data.items():
        np.save(args.output_dir / (key+'.npy'), value)
    comp = np.arange(len(data['component_size_nodes']), dtype=np.int32)
    rank = np.argsort(-data['component_size_nodes'])[:100]
    with (args.output_dir/'largest_components.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['component_id','nodes','accepted_edges','baseline_mismatch_mean',
                    'native_primary_mismatch_mean','fraction_nodes_changed_against_baseline'])
        for j in rank:
            w.writerow([int(comp[j]), int(data['component_size_nodes'][j]),
                        int(data['component_accepted_edge_count'][j]),
                        float(data['component_baseline_native_mismatch_mean'][j]),
                        float(data['component_primary_native_mismatch_mean'][j]),
                        float(data['component_node_difference_fraction'][j])])
    path = args.output_dir/'summary.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    # concise console summary, with all details in JSON
    view = {'status':summary['status'], 'baseline_evidence':evidence,
            'accepted_native_edges':summary['accepted_native_edges'],
            'candidate_components':summary['candidate_components'],
            'baseline':summary['comparisons']['baseline_statcost']['all_accepted_edges'],
            'native_primary':summary['comparisons']['native_primary']['all_accepted_edges'],
            'native_secondary':summary['comparisons']['native_secondary']['all_accepted_edges'],
            'relative_comparison':summary['relative_comparison'],
            'root_aligned_node_difference':summary['root_aligned_node_difference'],
            'output_summary':str(path)}
    print(json.dumps(view, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
