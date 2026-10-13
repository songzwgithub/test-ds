#!/usr/bin/env python3
"""Ningbo Native EMCF - single-step Island 0 temporal integration and release gate.

Reads saved full-resolution island results. Never edits the inputs. The output is a
SPATIALLY RELATIVE acquisition series: each date is referenced to the island's
first saved point, and acquisition index zero has phase zero. It DOES NOT
resolve absolute physical displacement or cross-island/tile gauges.

Run: python native_emcf_step1_timeinv.py --roi PATH --out NEW_EMPTY_PATH
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

TAU = 2.0 * np.pi
DEFAULT_ROOT = Path('/mnt/ningbo_process/ds/output/diagnostics')


def read(path: Path):
    if not path.is_file():
        raise ValueError(f'Missing required file: {path}')
    return np.load(path, mmap_mode='r', allow_pickle=False)


def check_finite(arr: np.ndarray, name: str):
    if not np.isfinite(arr).all():
        raise ValueError(f'Nonfinite values in {name}')


def time_cycles(pairs: np.ndarray, ndate: int):
    """Fundamental cycles of edge graph with deterministic ordered Kruskal tree."""
    root = list(range(ndate))
    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x
    adj = [[] for _ in range(ndate)]
    chords = []
    for i, (ua, vb) in enumerate(pairs):
        u, v = int(ua), int(vb)
        a, b = find(u), find(v)
        if a == b:
            chords.append(i)
        else:
            root[b] = a
            adj[u].append((v, i, 1))
            adj[v].append((u, i, -1))
    if len(set(find(x) for x in range(ndate))) != 1:
        raise ValueError('Temporal interferogram graph is disconnected')
    parent = [-1]*ndate
    pedge = [-1]*ndate
    pdir = [0]*ndate
    parent[0] = 0
    queue = [0]
    for u in queue:
        for v, e, s in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                pedge[v] = e
                pdir[v] = s
                queue.append(v)
    if len(queue) != ndate:
        raise ValueError('Temporal tree traversal failed')
    return chords, parent, pedge, pdir, queue


def anchor_cycle_diagnostics(pairs, anchor, chords, parent, pedge, pdir, queue):
    n = len(parent)
    potential = np.zeros(n, dtype=np.float64)
    for v in queue[1:]:
        potential[v] = potential[parent[v]] + pdir[v] * anchor[pedge[v]]
    c = np.asarray([anchor[e] - (potential[pairs[e, 1]] - potential[pairs[e, 0]]) for e in chords])
    k = np.rint(c/TAU)
    mod = np.abs(c - k*TAU)
    return {
        'cycle_basis': 'input-order spanning-tree fundamental cycles',
        'fundamental_cycles': len(chords),
        'nonzero_wrapped_anchor_integer_cycles': int(np.count_nonzero(k)),
        'max_abs_wrapped_anchor_integer': int(np.max(np.abs(k), initial=0)),
        'max_wrapped_anchor_modulo_residual_rad': float(np.max(mod, initial=0.0)),
        'note': 'Anchor cycles are a common IFG gauge ambiguity, not evidence that this percentage of points is wrong.'
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roi', type=Path, default=DEFAULT_ROOT/'native_emcf_roi_island0_fullres')
    ap.add_argument('--out', type=Path, default=DEFAULT_ROOT/'native_emcf_roi_island0_timeinv_gate')
    ap.add_argument('--tol-rad', type=float, default=1e-3,
                    help='Maximum permissible IFG reconstruction and anchor residual in radians')
    ap.add_argument('--expected-points', type=int, default=25391)
    a = ap.parse_args()
    roi = a.roi.resolve()
    out = a.out.resolve()
    if not roi.is_dir():
        ap.error(f'No ROI source directory: {roi}')
    if out == roi or roi in out.parents or out in roi.parents:
        ap.error('Output must be separate from and outside the source ROI')
    if out.exists() and any(out.iterdir()):
        ap.error(f'Refusing to overwrite nonempty output: {out}')
    if not (0 < a.tol_rad <= 0.1):
        ap.error('tol-rad must be (0,0.1]')
    u = read(roi/'unwrapped_ifg_phase_rad.npy')
    pairs = np.asarray(read(roi/'time_ifg_pairs_0based.npy'), dtype=np.int64)
    anchor = np.asarray(read(roi/'anchor_offset_rad.npy'), dtype=np.float64).reshape(-1)
    point_ids = read(roi/'original_point_ids.npy')
    if u.ndim != 2 or u.shape[0] != 219 or u.shape[1] != a.expected_points:
        ap.error(f'Unexpected IFG x point matrix: {u.shape}; expected (219,{a.expected_points})')
    nifg, npoint = u.shape
    if pairs.shape != (nifg,2) or len(anchor) != nifg or len(point_ids)!=npoint:
        ap.error('IFG pair, anchor, or point ID dimension mismatch')
    if np.any(pairs < 0) or np.any(pairs >= 75) or np.any(pairs[:,0]==pairs[:,1]):
        ap.error('Invalid zero-based IFG date indices')
    if len(np.unique(np.asarray(point_ids))) != npoint:
        ap.error('Duplicate original point IDs')
    check_finite(u, 'unwrapped IFGs')
    check_finite(anchor,'IFG anchor')
    summaryfile = roi/'summary.json'
    provenance = json.loads(summaryfile.read_text(encoding='utf-8')) if summaryfile.exists() else {}
    if provenance.get('island_rank') not in (None,0):
        ap.error('Source is not Island 0')
    if provenance.get('global_integer_gauge_determined') is True:
        ap.error('Unexpected source gauge metadata; inspect before using this script')

    # Spurt's uw is relative to the first point of the island for EACH IFG;
    # backend saves uw + wrapped_phase_difference_at_first_point.
    # The latter is the saved anchor_offset_rad and must be removed for an
    # internally compatible, spatially referenced time series.
    first_point_residual = np.asarray(u[:,0], dtype=np.float64) - anchor
    anchor_parity = float(np.max(np.abs(first_point_residual)))

    # A x = b, with each IFG (date u -> date v) = phase[v]-phase[u].
    # Date 0 is fixed to zero. Uses graph Laplacian's positive definite
    # normal equations to handle all 25,391 point columns together.
    ndate = 75
    A = np.zeros((nifg, ndate-1), dtype=np.float64)
    for i, (d0, d1) in enumerate(pairs):
        if d0 != 0:
            A[i, d0-1] -= 1.0
        if d1 != 0:
            A[i, d1-1] += 1.0
    rank = int(np.linalg.matrix_rank(A))
    if rank != ndate-1:
        ap.error(f'Temporal graph rank insufficient: {rank}/{ndate-1}')

    # Extract relative phase before solving (no nearest neighbor, averaging,
    # clipping or integer-rounding). Work in float64 for numerical integrity.
    B = np.asarray(u, dtype=np.float64) - anchor[:,None]
    G = A.T @ A
    coeff = cho_solve(cho_factor(G, check_finite=False), A.T @ B, check_finite=False)
    fit = A @ coeff
    err = fit - B
    abs_err = np.abs(err)
    reconstruction_max = float(abs_err.max())
    reconstruction_p99 = float(np.quantile(abs_err, .99))
    reconstruction_rms = float(np.sqrt(np.mean(err**2)))
    per_point_rms = np.sqrt(np.mean(err**2, axis=0))
    per_ifg_max = abs_err.max(axis=1)
    acquisition = np.zeros((ndate, npoint),dtype=np.float32)
    acquisition[1:] = coeff.astype(np.float32)
    epoch0_max = float(np.max(np.abs(acquisition[0])))
    spatial_anchor_max = float(np.max(np.abs(acquisition[:,0])))
    spurt_parity = provenance.get('wrapped_phase_parity_max_rad')
    rel_status = provenance.get('temporal_closure_relative', {})
    closure_nonzero = rel_status.get('nonzero_integer_closure_fraction')
    spurt_pass = spurt_parity is not None and spurt_parity <= a.tol_rad
    rel_pass = closure_nonzero == 0.0

    chords, parent, pedge, pdir, queue = time_cycles(pairs, ndate)
    anchordiag = anchor_cycle_diagnostics(pairs,anchor,chords,parent,pedge,pdir,queue)
    anchor_mod_pass = anchordiag['max_wrapped_anchor_modulo_residual_rad'] <= a.tol_rad
    reasons = []
    if anchor_parity > a.tol_rad: reasons.append(f'first-point IFG anchor mismatch: {anchor_parity:.3g}')
    if reconstruction_max > a.tol_rad: reasons.append(f'IFG reconstruction max residual: {reconstruction_max:.3g}')
    if not spurt_pass: reasons.append('Spurt wrapped-phase parity missing or over tolerance in source summary')
    if not rel_pass: reasons.append('Spurt relative cycle closure not zero in source summary')
    if not anchor_mod_pass: reasons.append('Wrapped anchor cycles not 2pi-integral within tolerance')
    if len(chords) != 145: reasons.append(f'Time cycle count {len(chords)}, expected 145')
    if epoch0_max != 0 or spatial_anchor_max > a.tol_rad:
        reasons.append('Reference-date/point gauge failed')

    success = not reasons
    meta = {
        'status': 'GO_FULLRES_ENGINEERING_PILOT' if success else 'STOP_CHECK_TEMPORAL_INPUTS',
        'source_directory': str(roi), 'output_directory':str(out),
        'n_dates':ndate,'n_ifg':nifg,'npoint':npoint,
        'time_graph_rank':rank,'time_graph_cycles':len(chords),
        'max_first_point_anchor_mismatch_rad':anchor_parity,
        'ifg_reconstruction_max_abs_rad':reconstruction_max,
        'ifg_reconstruction_p99_abs_rad':reconstruction_p99,
        'ifg_reconstruction_rms_rad':reconstruction_rms,
        'per_point_residual_rms_p99_rad':float(np.quantile(per_point_rms, .99)),
        'per_ifg_max_abs_rad':per_ifg_max.tolist(),
        'source_wrapped_phase_parity_max_rad':spurt_parity,
        'source_relative_temporal_cycle_bad_fraction':closure_nonzero,
        'temporal_anchor':anchordiag,
        'gauge': {
            'spatial_reference':'FIRST_SAVED_POINT in the island, at EACH epoch',
            'temporal_reference':'acquisition index 0 fixed to zero',
            'global_integer_gauge_determined':False,
            'cross_island_tile_integer_gauge_determined':False,
            'absolute_physical_deformation_validated':False,
            'meaning':'Relative phase only. No absolute motion, no cross-component offsets.'
        },
        'release_scope':'Engineering full-resolution spatial-unwrapping pilot; NOT fully referenced displacement production',
        'failed_conditions':reasons,
        'production_modified':False,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out/'qa_summary.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    if success:
        # Atomic publication after all QA, rather than creating a misleading
        # valid-looking time series on a failed run.
        tmp = out/'acquisition_relative_phase_rad.npy.tmp'
        with tmp.open('wb') as fh:
            np.save(fh, acquisition, allow_pickle=False)
        tmp.replace(out/'acquisition_relative_phase_rad.npy')
    print(json.dumps({k:v for k,v in meta.items() if k!='per_ifg_max_abs_rad'}, indent=2, ensure_ascii=False))
    print(f'Saved QA: {out / "qa_summary.json"}')
    if success:
        print(f'Saved acquisition phase: {out / "acquisition_relative_phase_rad.npy"} (75,{npoint})')
    return 0 if success else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        sys.exit(2)
