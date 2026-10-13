#!/usr/bin/env python3
"""Ningbo native EMCF: modal temporal-cycle re-referencing and point-only masking.

Input: saved per-island 219 IFGs from full-resolution Spurt backend.
No new spatial unwrapping, no alteration of integer phase, no interpolation.
Choose a temporal medoid point in largest identical integer-cycle-signature
class as a relative computational reference, then invert IFGs and NaN-mask only points
that fail the STRICT per-point IFG reconstruction test.

Outputs to new empty dir; requires later independent overlapping-tiles QA.
Does not prove physical correctness of systematic/integrable integer errors.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

TAU = 2.0 * np.pi


def read(src, name):
    p = src / name
    if not p.is_file():
        raise ValueError(f'Missing input: {p}')
    return np.load(p, mmap_mode='r', allow_pickle=False)


def make_time_graph(pairs, ndate):
    nedge = len(pairs)
    parent = list(range(ndate))
    adj = [[] for _ in range(ndate)]
    chords = []
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for e, (u0, v0) in enumerate(pairs):
        u, v = int(u0), int(v0)
        a, b = find(u), find(v)
        if a == b:
            chords.append(e)
        else:
            parent[b] = a
            adj[u].append((v, e, 1))
            adj[v].append((u, e, -1))
    if len({find(j) for j in range(ndate)}) != 1:
        raise ValueError('Time network disconnected')
    parents = np.full(ndate, -1, np.int32)
    pedge = np.full(ndate, -1, np.int32)
    direction = np.zeros(ndate, np.int8)
    parents[0] = 0
    bfs = [0]
    for u in bfs:
        for v, e, s in adj[u]:
            if parents[v] < 0:
                parents[v], pedge[v], direction[v] = u, e, s
                bfs.append(v)
    if len(bfs) != ndate or len(chords) != nedge - ndate + 1:
        raise ValueError('Time graph cycle basis failure')
    a = np.zeros((nedge, ndate - 1), dtype=np.float64)
    for e, (u, v) in enumerate(pairs):
        if u: a[e, u-1] -= 1
        if v: a[e, v-1] += 1
    if np.linalg.matrix_rank(a) != ndate-1:
        raise ValueError('Time graph rank < 74')
    return np.asarray(chords, np.intp), parents, pedge, direction, bfs, a


def cycle_integer_fingerprints(raw, pairs, chords, parents, pedge, direction, bfs):
    """One integer-cycle vector per native point; independent of reference.

    Phase for each chord minus the potential implied by the FIRST SPANNING TREE,
    rounded to integral multiples of 2pi. Never use this rounding to repair data:
    it is ONLY a discrete classification key, and full IFG QA follows.
    """
    ndate = len(parents)
    npt = raw.shape[1]
    phi = np.zeros((ndate, npt), dtype=np.float64)
    for v in bfs[1:]:
        phi[v] = phi[parents[v]] + direction[v] * raw[pedge[v]]
    gap = raw[chords] - (phi[pairs[chords, 1]] - phi[pairs[chords, 0]])
    rounded = np.rint(gap/TAU)
    residual = np.abs(gap - rounded*TAU).max(axis=0)
    int16_ok = np.all(np.abs(rounded) < np.iinfo(np.int16).max, axis=0)
    rounded = np.clip(rounded, -32766, 32766).astype(np.int16).T
    return rounded, residual, int16_ok


def choose_modal_medoid(U, good_mode, A, point_type, graph_degree, *, max_candidates=2048):
    """Find a representative REAL point across the whole temporal-cycle mode.

    Target: temporal median of an evenly stratified deterministic subsample.
    Search: ALL points in the modal class in bounded-size chunks; choose the
    point whose 75-date relative time series best follows that target.
    Prefer PS / high graph degree only among candidates within 0.01 rad RMS.
    This is a computational medoid, NOT proof of geodetic stability.
    """
    members = np.asarray(good_mode, dtype=np.int64)
    if len(members) == 0 or max_candidates < 1:
        raise ValueError('Cannot select reference from empty class / bad sample size')
    sample = members
    if len(sample) > max_candidates:
        sample = sample[np.linspace(0, len(sample)-1, max_candidates, dtype=np.int64)]
    base = np.asarray(U[:, int(sample[0])], np.float64).reshape(-1, 1)
    factor = cho_factor(A.T @ A, check_finite=False)
    b = np.asarray(U[:, sample], dtype=np.float64) - base
    if not np.isfinite(b).all():
        raise ValueError('Nonfinite reference candidate IFGs')
    x = cho_solve(factor, A.T @ b, check_finite=False)
    # Acquisition 0 is exactly zero across all candidates. Target is therefore
    # a valid 75-epoch spatially relative signal in the provisional gauge.
    med = np.median(x, axis=1)
    del x, b

    scores = np.empty(len(members), np.float64)
    for left in range(0, len(members), 4096):
        right = min(len(members), left + 4096)
        b = np.asarray(U[:, members[left:right]], dtype=np.float64) - base
        x = cho_solve(factor, A.T @ b, check_finite=False)
        # The zero reference acquisition contributes exactly zero to RMS.
        scores[left:right] = np.sqrt(np.sum((x-med[:, None])**2, axis=0)/75.0)

    best = float(np.min(scores))
    near = np.flatnonzero(scores <= best + 0.01)
    is_ps = (point_type[members[near]] == 1) if point_type is not None else np.zeros(len(near), bool)
    deg = graph_degree[members[near]] if graph_degree is not None else np.zeros(len(near), np.int32)
    rank = np.lexsort((members[near], scores[near], -deg, -is_ps.astype(np.int8)))
    chosen_pos = int(near[rank[0]])
    ref = int(members[chosen_pos])
    return ref, {
        'method': 'MODAL_CYCLE_CLASS_TEMPORAL_MEDOID_ALL_CLASS_POINTS',
        'modal_class_points_searched': int(len(members)),
        'temporal_median_estimation_sample_points': int(len(sample)),
        'temporal_median_sampling': 'ALL' if len(members) <= max_candidates else 'DETERMINISTIC_INDEX_STRATIFIED',
        'selected_medoid_rms_to_modal_ensemble_rad': float(scores[chosen_pos]),
        'minimum_medoid_rms_to_modal_ensemble_rad': best,
        'selected_ps': bool(point_type[ref] == 1) if point_type is not None else None,
        'selected_graph_degree': int(graph_degree[ref]) if graph_degree is not None else None,
        'physical_stability_proven': False,
        'note': 'Computational medoid only. The temporal median may move with real regional subsidence. A physically stable reference needs independent evidence.',
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roi', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tol-rad', type=float, default=1e-3)
    ap.add_argument('--cycle-mod-tol-rad', type=float, default=1e-3)
    ap.add_argument('--chunk-points', type=int, default=4096)
    ap.add_argument('--min-valid-points', type=int, default=3)
    ap.add_argument('--ref-max-candidates', type=int, default=2048,
                    help='Maximum sampled candidates used to ESTIMATE group median; ALL modal points are scored')
    args = ap.parse_args(argv)
    src, out = args.roi.resolve(), args.out.resolve()
    if not src.is_dir(): ap.error('Input directory missing')
    if out == src or out in src.parents or src in out.parents:
        ap.error('Input and output may not contain one another')
    if out.exists() and any(out.iterdir()): ap.error('Refuse nonempty output directory')
    if not(0 < args.tol_rad <= .001 and 0 < args.cycle_mod_tol_rad <= .001):
        ap.error('Strict tolerance ranges (0,0.001]')
    if args.chunk_points < 1 or args.min_valid_points < 1 or args.ref_max_candidates < 1:
        ap.error('Positive chunk/min-valid/ref-max-candidates required')

    U = read(src, 'unwrapped_ifg_phase_rad.npy')
    pairs = np.asarray(read(src, 'time_ifg_pairs_0based.npy'), dtype=np.intp)
    anchors = np.asarray(read(src, 'anchor_offset_rad.npy'), dtype=np.float64)
    ids = read(src, 'original_point_ids.npy')
    # Optional original-point evidence. Absent arrays simply disable
    # PS/topology tie-breakers; they are never invented or interpolated.
    point_type = np.asarray(read(src, 'original_point_type.npy')) if (src/'original_point_type.npy').exists() else None
    graph_degree = None
    if (src/'space_graph_links.npy').exists():
        links = np.asarray(read(src, 'space_graph_links.npy'), dtype=np.int64)
        if links.ndim != 2 or links.shape[1] != 2 or np.any(links < 0) or np.any(links >= len(ids)):
            raise ValueError('Invalid original spatial graph links')
        graph_degree = np.bincount(links.reshape(-1), minlength=len(ids))
    if point_type is not None and point_type.shape != (len(ids),):
        raise ValueError('Original point types have incorrect shape')
    if U.ndim != 2 or U.shape[0] != 219 or pairs.shape != (219, 2) or anchors.shape != (219,):
        raise ValueError('Expected 219 IFG data plus 219x2 time graph and 219 anchors')
    nifg, npoint = U.shape
    if ids.shape != (npoint,) or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != npoint:
        raise ValueError('Invalid or duplicate point IDs')
    if np.any(pairs < 0) or np.any(pairs >= 75) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError('Bad time graph date pairs')
    if not np.isfinite(anchors).all(): raise ValueError('Non-finite source anchors')
    meta = json.loads((src/'summary.json').read_text(encoding='utf8'))
    if meta.get('status') != 'ROI_EMCF_BENCHMARK_ONLY' or meta.get('time_ifgs') != 219:
        raise ValueError('Source spatial EMCF QA/status mismatch')
    parity = meta.get('wrapped_phase_parity_max_rad')
    if parity is None or not np.isfinite(parity) or parity > args.tol_rad:
        raise ValueError('Source wrapped-phase parity failed')
    first = np.asarray(U[:, 0], np.float64)
    if not np.isfinite(first).all() or np.max(np.abs(first - anchors)) > args.tol_rad:
        raise ValueError('Source first-point IFG anchor mismatch')
    chords, parents, pedge, direction, bfs, A = make_time_graph(pairs, 75)
    if len(chords) != 145: raise ValueError('Time cycle count mismatch')

    # Classification for all source points with bounded-size reads.
    cycles = np.empty((npoint, len(chords)), dtype=np.int16)
    cycle_mod = np.full(npoint, np.inf, dtype=np.float32)
    classifiable = np.zeros(npoint, bool)
    for left in range(0, npoint, args.chunk_points):
        right = min(npoint, left+args.chunk_points)
        raw = np.asarray(U[:, left:right], dtype=np.float64)
        finite = np.isfinite(raw).all(axis=0)
        if not finite.all():raw[:, ~finite] = 0.0
        fp, mod, intok = cycle_integer_fingerprints(raw, pairs, chords, parents, pedge, direction, bfs)
        cycles[left:right] = fp
        cycle_mod[left:right] = np.where(finite, mod, np.inf).astype(np.float32)
        classifiable[left:right] = finite & intok & (mod <= args.cycle_mod_tol_rad)

    eligible = np.flatnonzero(classifiable)
    ref = None
    ref_diagnostics = None
    group = np.zeros(npoint, bool)
    nclasses = 0
    max_group = 0
    if len(eligible):
        # Group by EXACT integer vector. No phase modifications take place.
        signatures, inv, counts = np.unique(cycles[eligible], axis=0, return_inverse=True, return_counts=True)
        nclasses = len(counts)
        mode = int(np.argmax(counts))
        good_mode = eligible[inv == mode]
        max_group = len(good_mode)
        group[good_mode] = True
        # Choose a representative REAL point within the largest consistent
        # class; never blindly use the first point of that class.
        ref, ref_diagnostics = choose_modal_medoid(
            U, good_mode, A, point_type, graph_degree,
            max_candidates=args.ref_max_candidates)
        del signatures, inv, counts

    out.mkdir(parents=True, exist_ok=True)
    valid = np.zeros(npoint, dtype=bool)
    acq_tmp = out/'acquisition_relative_phase_rad.npy.tmp'
    cube = np.lib.format.open_memmap(acq_tmp,mode='w+',dtype='float32',shape=(75,npoint))
    cube[0, :] = 0
    accepted_max = 0.0
    accepted_ssq = 0.0
    accepted_observations = 0
    examples = []
    if ref is not None:
        ref_ifg = np.asarray(U[:,ref], dtype=np.float64)
        factor = cho_factor(A.T@A, check_finite=False)
        for left in range(0,npoint,args.chunk_points):
            right=min(npoint,left+args.chunk_points)
            raw=np.asarray(U[:,left:right],dtype=np.float64)
            eligible_block=group[left:right].copy() & np.isfinite(raw).all(axis=0)
            raw[:,~eligible_block]=ref_ifg[:,None]
            b=raw-ref_ifg[:,None]
            coeff=cho_solve(factor,A.T@b,check_finite=False)
            residual=A@coeff-b
            err=np.max(np.abs(residual),axis=0)
            ok=eligible_block & np.isfinite(err) & (err <= args.tol_rad)
            valid[left:right]=ok
            cube[1:,left:right]=np.where(ok[None,:],coeff,np.nan).astype('float32')
            if ok.any():
                accepted_max=max(accepted_max,float(np.max(err[ok])))
                accepted_ssq+=float(np.sum(residual[:,ok]**2))
                accepted_observations+=int(np.count_nonzero(ok)*nifg)
            if len(examples)<20:
                for j in np.flatnonzero(~ok)[:20-len(examples)]:
                    examples.append({'point_id':int(ids[left+j]),'point_index':left+int(j),
                                     'cycle_signature_in_modal_class':bool(group[left+j]),
                                     'ifg_reconstruction_max_rad':float(err[j]) if eligible_block[j] else None})
    else:
        cube[1:]=np.nan
    # Explicitly mask all 75 acquisitions, including reference epoch, for bad points.
    cube[0, ~valid] = np.nan
    cube.flush();del cube
    if ref is not None and not valid[ref]: raise ValueError('Consensus reference unexpectedly failed')
    valid_count = int(np.count_nonzero(valid))
    bad_count = npoint-valid_count
    ok = valid_count >= args.min_valid_points
    # Do not label an empty or trivial island as successful even if no fraction cutoff.
    status = 'GO_FULLRES_ISLAND_RELATIVE' if ok else 'STOP_NO_RELIABLE_REFERENCE_OR_VALID_POINTS'
    qa={
        'status':status, 'source':str(src),'output':str(out),'n_dates':75,'n_ifg':219,'npoint':npoint,
        'time_graph_rank':74,'time_graph_cycles':145,
        'accepted_points':valid_count,'bad_points':bad_count,'bad_point_fraction':float(bad_count/npoint),
        'reference_point_local_index':ref,
        'reference_point_original_id':int(ids[ref]) if ref is not None else None,
        'spatial_gauge':'modal temporal-cycle class original point per acquisition',
        'reference_selection':'MODAL_TEMPORAL_MEDOID_NO_POINT_INTEGER_CORRECTION',
        'reference_selection_diagnostics':ref_diagnostics,
        'reference_class_points':max_group,'reference_class_fraction':max_group/npoint,
        'distinct_temporal_cycle_signature_classes':nclasses,
        'cycle_modulo_tolerance_rad':args.cycle_mod_tol_rad,
        'points_failing_cycle_modulo':int(np.count_nonzero(~classifiable)),
        'valid_ifg_reconstruction_max_rad':accepted_max if valid_count else None,
        'valid_ifg_reconstruction_rms_rad':float(math.sqrt(accepted_ssq/accepted_observations)) if accepted_observations else None,
        'source_wrapped_parity_max_rad':parity,
        'source_relative_closure_fraction':meta.get('temporal_closure_relative',{}).get('nonzero_integer_closure_fraction'),
        'source_original_first_point_anchor_mismatch_rad':float(np.max(np.abs(first-anchors))),
        'failed_conditions':[] if ok else ['No valid reference or insufficient strictly valid points'],
        'quality_flags':([ 'LOW_VALID_COVERAGE' ] if valid_count/npoint<.1 else []) +
            (['LOW_MODAL_CLASS_SUPPORT_REVIEW_REQUIRED'] if max_group/npoint<.5 else []) +
            (['REFERENCE_SHIFT_FROM_SOURCE_FIRST_POINT'] if ref not in (None,0) else []),
        'per_point_bad_fraction_has_no_release_limit':True,
        'all_75_epochs_nan_for_masked_points':True,
        'time_series_is_reference_relative':True,
        'global_integer_gauge_determined':False,'cross_island_gauge_determined':False,
        'absolute_physical_deformation_validated':False,'production_modified':False,
        'note':'Cycle-integer vectors are used ONLY to choose a reference class, not to repair phases. '
               'Reference medoid is statistically representative but NOT a physically validated stable benchmark. '
               'No forced 2pi corrections. Consistency is not proof of physical correctness.',
        'bad_examples':examples,
    }
    np.save(out/'valid_points.npy',valid,allow_pickle=False)
    np.save(out/'reference_cycle_signature.npy',cycles[ref] if ref is not None else np.zeros(145,np.int16),allow_pickle=False)
    (out/'qa_summary.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf8')
    if ok:
        acq_tmp.replace(out/'acquisition_relative_phase_rad.npy')
    else:
        acq_tmp.unlink(missing_ok=True)
    print(json.dumps({k:v for k,v in qa.items() if k!='bad_examples'},indent=2,ensure_ascii=False))
    return 0 if ok else 2


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,np.linalg.LinAlgError) as ex:
        print('STOP:',ex,file=sys.stderr)
        sys.exit(2)
