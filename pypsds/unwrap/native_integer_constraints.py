"""Native PS/DS spatial-edge integer candidates with independent forest QA.

The solution is *relative to an arbitrary root in every connected component*.
Neither wrapped observations nor spanning-tree agreement establishes the true
cross-component or absolute spatial integer gauge. This module never writes to
production phase arrays.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

TWOPI = 2.0 * np.pi


def wrap_phase(x):
    return np.arctan2(np.sin(x), np.cos(x))


def native_integer_edge_observations(*, phase, node_phase, node_pairs,
                                     native_point_pairs, use_edges,
                                     max_temporal_step_rad=0.90 * np.pi,
                                     max_node_native_difference_rad=2.5,
                                     batch=1024):
    """Temporally unwrap replicated native spatial gradients, not each node.

    Assumption: between successive acquisitions the *spatial phase gradient*
    evolves by less than pi. This cannot be proven from wrapped data. Edges
    with observed steps near pi are rejected; undetectable >pi aliases remain
    a scientific limitation. The returned integer difference k obeys
    k[e,t] = n[node_b,t] - n[node_a,t] at candidate-supported edges.
    """
    if phase.ndim != 2 or node_phase.ndim != 2:
        raise ValueError('Point/node phases require [location, date]')
    pairs = np.asarray(node_pairs, dtype=np.int32)
    native = np.asarray(native_point_pairs, dtype=np.int64)
    use = np.asarray(use_edges, dtype=bool)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or native.shape[:1] != pairs.shape[:1] or native.ndim != 3 or native.shape[2] != 2 or use.shape != (len(pairs),):
        raise ValueError('Invalid edge/native-pair dimensions')
    if node_phase.shape[1] != phase.shape[1] or np.any(pairs < 0) or np.any(pairs >= node_phase.shape[0]):
        raise ValueError('Node identity/acquisition mismatch')
    if not 0 < max_temporal_step_rad < np.pi:
        raise ValueError('max_temporal_step_rad must lie strictly between zero and pi')
    ndate = phase.shape[1]
    candidate_ids = np.flatnonzero(use).astype(np.int32)
    k = np.zeros((len(candidate_ids), ndate), dtype=np.int16)
    stable = np.zeros(len(candidate_ids), dtype=bool)
    native_node_disagreement = np.full(len(candidate_ids), np.nan, dtype=np.float32)
    max_step = np.full(len(candidate_ids), np.nan, dtype=np.float32)
    for s in range(0, len(candidate_ids), batch):
        e = min(len(candidate_ids), s + batch)
        subset = candidate_ids[s:e]
        uv = pairs[subset]
        ids = native[subset]
        valid = (ids[..., 0] >= 0) & (ids[..., 1] >= 0)
        if not np.all(valid.sum(axis=1) >= 2):
            raise RuntimeError('Selected edge lacks two native point pairs')
        a = ids[..., 0].clip(min=0).reshape(-1)
        b = ids[..., 1].clip(min=0).reshape(-1)
        if a.size and (np.max(a) >= phase.shape[0] or np.max(b) >= phase.shape[0]):
            raise ValueError('Native point IDs exceed PointPhaseStack')
        z = np.exp(1j * (np.asarray(phase[b, :], dtype=np.float64) -
                          np.asarray(phase[a, :], dtype=np.float64))).reshape(e-s, ids.shape[1], ndate)
        z[~valid, :] = 0
        native_wrapped = np.angle(z.sum(axis=1))
        raw = (np.asarray(node_phase[uv[:, 1], :], dtype=np.float64) -
               np.asarray(node_phase[uv[:, 0], :], dtype=np.float64))
        # Native pair means and grid complex means need not coincide.
        delta = wrap_phase(native_wrapped - wrap_phase(raw))
        disagreement = np.median(np.abs(delta), axis=1)
        increments = np.abs(wrap_phase(np.diff(native_wrapped, axis=1)))
        step = increments.max(axis=1) if ndate > 1 else np.zeros(e-s)
        # A time-constant large disagreement could reflect strong legitimate
        # gradients, so this is a conservative confidence gate, not a diagnosis.
        okay = (np.max(np.abs(delta), axis=1) < max_node_native_difference_rad) & (step < max_temporal_step_rad)
        unwrapped = np.unwrap(native_wrapped, axis=1)
        inferred = np.rint((unwrapped - delta - raw) / TWOPI)
        if np.any(np.abs(inferred[okay]) > 32700):
            raise OverflowError('Integer branch exceeds int16')
        k[s:e] = inferred.astype(np.int16)
        stable[s:e] = okay & np.all(np.isfinite(inferred), axis=1)
        native_node_disagreement[s:e] = disagreement.astype(np.float32)
        max_step[s:e] = step.astype(np.float32)
    return candidate_ids, k, stable, native_node_disagreement, max_step


def _component_labels(nnode, a, b):
    if len(a):
        mat = csr_matrix((np.ones(len(a)*2, dtype=np.uint8),
                          (np.r_[a,b], np.r_[b,a])), shape=(nnode,nnode))
        ncomp, labels = connected_components(mat, directed=False)
    else:
        ncomp, labels = nnode, np.arange(nnode, dtype=np.int32)
    root = np.full(ncomp, nnode, dtype=np.int32)
    np.minimum.at(root, labels, np.arange(nnode, dtype=np.int32))
    return labels, root


def _forest_solution(nnode, node_pairs, k, score, labels, roots):
    """Kruskal forest, assigning exactly-integrable integer node potentials."""
    nedge, ndate = k.shape
    parent = np.arange(nnode, dtype=np.int32)
    size = np.ones(nnode, dtype=np.int32)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    chosen = np.zeros(nedge, dtype=bool)
    for e in np.argsort(-score, kind='stable'):
        a, b = map(int, node_pairs[e])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        chosen[e] = True
    pick = np.flatnonzero(chosen)
    u, v = node_pairs[pick, 0], node_pairs[pick, 1]
    # CSR adjacency indexed by parent node, carrying edge index and direction.
    src = np.r_[u, v]
    dst = np.r_[v, u]
    eid = np.r_[pick, pick]
    sign = np.r_[np.ones(len(pick), dtype=np.int8), -np.ones(len(pick), dtype=np.int8)]
    order = np.argsort(src, kind='stable')
    src, dst, eid, sign = src[order], dst[order], eid[order], sign[order]
    indptr = np.r_[0, np.cumsum(np.bincount(src, minlength=nnode))]
    result = np.zeros((nnode, ndate), dtype=np.int32)
    visited = np.zeros(nnode, dtype=bool)
    for root in roots:
        stack = [int(root)]; visited[int(root)] = True
        while stack:
            u0 = stack.pop()
            for j in range(indptr[u0], indptr[u0+1]):
                v0 = int(dst[j])
                if visited[v0]:
                    continue
                visited[v0] = True
                result[v0] = result[u0] + int(sign[j]) * k[eid[j]].astype(np.int32)
                stack.append(v0)
    if not np.all(visited):
        raise RuntimeError('Spanning forest failed to reach all components')
    return result, chosen


def synchronize_integer_graph(*, nnode, node_pairs, integer_edge_histories,
                              score_primary, score_secondary, ndate=None):
    """Two forests + independent loop closure; NEVER global-gauge solver."""
    pairs = np.asarray(node_pairs, dtype=np.int32)
    k = np.asarray(integer_edge_histories, dtype=np.int16)
    if k.ndim != 2 or pairs.shape != (len(k),2):
        raise ValueError('Node pairs and integer histories differ')
    if ndate is not None and k.shape[1] != ndate:
        raise ValueError('Acquisition count differs')
    if np.any(pairs < 0) or np.any(pairs >= nnode) or np.any(pairs[:,0] == pairs[:,1]):
        raise ValueError('Invalid node pair')
    labels, roots = _component_labels(nnode,pairs[:,0],pairs[:,1])
    primary, tree_a = _forest_solution(nnode,pairs,k,np.asarray(score_primary),labels,roots)
    secondary, tree_b = _forest_solution(nnode,pairs,k,np.asarray(score_secondary),labels,roots)
    residual_a = np.abs(primary[pairs[:,1]]-primary[pairs[:,0]]-k.astype(np.int32))
    residual_b = np.abs(secondary[pairs[:,1]]-secondary[pairs[:,0]]-k.astype(np.int32))
    agreement = np.mean(primary==secondary,axis=1)
    # Component integer gauge is local: two roots are not tied merely because
    # their native wrapped time series match.
    return {
        'cycles_primary':primary, 'cycles_secondary':secondary,
        'component_id':labels.astype(np.int32),'component_root_node':roots,
        'tree_edge_mask_primary':tree_a,'tree_edge_mask_secondary':tree_b,
        'edge_conflict_fraction_primary':np.mean(residual_a!=0,axis=1).astype(np.float32),
        'edge_conflict_fraction_secondary':np.mean(residual_b!=0,axis=1).astype(np.float32),
        'node_forest_agreement_fraction':agreement.astype(np.float32),
    }



def node_two_core_mask(nnode, node_pairs):
    """Conservative redundant-support gate: peel all degree-0/1 nodes.

    Membership in the 2-core is necessary, not sufficient, for an
    independently checked relative integer branch (bridges can remain).
    """
    p = np.asarray(node_pairs, dtype=np.int32)
    if p.shape != (len(p), 2):
        raise ValueError('Expected [edge,2] node endpoints')
    if len(p)==0:
        return np.zeros(nnode,dtype=bool)
    a,b=p[:,0],p[:,1]
    adj=csr_matrix((np.ones(len(p)*2,dtype=np.uint8),
                     (np.r_[a,b], np.r_[b,a])),shape=(nnode,nnode))
    adj.sum_duplicates()
    deg=np.diff(adj.indptr).astype(np.int32)
    active=np.ones(nnode,dtype=bool)
    todo=deque(map(int,np.flatnonzero(deg < 2)))
    while todo:
        u=todo.popleft()
        if not active[u]:
            continue
        active[u]=False
        for v in adj.indices[adj.indptr[u]:adj.indptr[u+1]]:
            if active[v]:
                deg[v]-=1
                if deg[v] == 1:
                    todo.append(int(v))
    return active

def _quantiles(values):
    x = np.asarray(values)
    return [float(v) for v in np.percentile(x,[50,90,95,99])] if x.size else None


def main(argv=None):
    ap = argparse.ArgumentParser(description='Native-PS/DS graph integer candidates (isolated, never production write)')
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--network-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--max-temporal-step-rad',type=float,default=float(0.9*np.pi))
    ap.add_argument('--max-node-native-difference-rad',type=float,default=2.5)
    ap.add_argument('--batch',type=int,default=1024)
    ap.add_argument('--require-fraction-two-forests',type=float,default=0.99)
    args = ap.parse_args(argv)
    from pypsds.context import open_from_config
    _, _, paths, stack, _=open_from_config(args.config)
    pp=Path(paths.output_dir)/'processing'/'point_phase_stack'
    co=Path(paths.output_dir)/'processing'/'stamps3d_unwrap'
    point_phase=np.load(pp/'phase_rad.npy',mmap_mode='r')
    node_phase=np.load(co/'coarse_complex_mean_phase_rad.npy',mmap_mode='r')
    node_pairs=np.load(args.network_dir/'node_pairs.npy',mmap_mode='r')
    native_pairs=np.load(args.network_dir/'native_point_pairs.npy',mmap_mode='r')
    supported=np.load(args.network_dir/'supported_edge_mask.npy',mmap_mode='r')
    direct=np.load(args.network_dir/'directly_adjacent.npy',mmap_mode='r')
    distances=np.load(args.network_dir/'native_pair_distances_m.npy',mmap_mode='r')
    coherence=np.load(args.network_dir/'native_pair_circular_consistency.npy',mmap_mode='r')
    if node_phase.shape[1] != len(stack.dates) or point_phase.shape[1]!=len(stack.dates):
        raise ValueError('Acquisition dates disagree with phase stack')
    if len(node_pairs)!=len(supported) or len(direct)!=len(supported):
        raise ValueError('Network metadata misaligned')
    # Only observed four-neighbor relationships are eligible for integer
    # synchronization. Candidate bridges remain explicitly diagnostic.
    eligible=np.asarray(supported,dtype=bool)&np.asarray(direct,dtype=bool)
    print(f'[INTEGER] Native-supported occupied edges: {int(eligible.sum()):,}',flush=True)
    indices,obs,stable,disagreement,maxstep=native_integer_edge_observations(
        phase=point_phase,node_phase=node_phase,node_pairs=node_pairs,
        native_point_pairs=native_pairs,use_edges=eligible,
        max_temporal_step_rad=args.max_temporal_step_rad,
        max_node_native_difference_rad=args.max_node_native_difference_rad,batch=args.batch)
    good=np.flatnonzero(stable)
    used=indices[good]
    retained=node_pairs[used].astype(np.int32)
    k=obs[good]
    if k.shape[0]==0:
        raise RuntimeError('All native integer edges were rejected; no candidate created')
    score_a=np.asarray(coherence[used],dtype=np.float64)
    d=np.nanmean(np.asarray(distances[used],dtype=np.float64),axis=1)
    score_b=1.0/(1.0+np.maximum(0,d))
    print(f'[INTEGER] Temporal and node-native supported edges: {len(used):,}',flush=True)
    result=synchronize_integer_graph(nnode=node_phase.shape[0],node_pairs=retained,
        integer_edge_histories=k,score_primary=score_a,score_secondary=score_b,ndate=node_phase.shape[1])
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    for name,val in result.items():
        np.save(output/(name+'.npy'),val)
    np.save(output/'retained_network_edge_indices.npy',used)
    np.save(output/'candidate_edge_integer_histories.npy',k)
    np.save(output/'candidate_edge_max_temporal_step_rad.npy',maxstep[good])
    np.save(output/'candidate_edge_native_node_disagreement_rad.npy',disagreement[good])
    agree=result['node_forest_agreement_fraction']
    root=result['component_root_node']
    cc=result['component_id']
    ncomp=len(root)
    # Only report agreement inside components with closed cycles.
    tree=np.asarray(result['tree_edge_mask_primary'])
    cycleedge=~tree
    cycles_by_comp=np.bincount(cc[retained[cycleedge,0]],minlength=ncomp)
    component_has_loops=cycles_by_comp>0
    two_core=node_two_core_mask(node_phase.shape[0],retained)
    np.save(output/"node_two_core_mask.npy",two_core)
    strict=(agree>=args.require_fraction_two_forests)&two_core
    conflict=np.asarray(result['edge_conflict_fraction_primary'])
    frac_bad=float(np.mean(conflict[cycleedge]>0)) if np.any(cycleedge) else None
    summary={
        'status':'CANDIDATE_ONLY_NOT_PRODUCTION',
        'input_points':int(point_phase.shape[0]),
        'node_count':int(node_phase.shape[0]),
        'acquisitions':int(node_phase.shape[1]),
        'network_observed_supported_edges':int(np.sum(eligible)),
        'temporal_accepted_edges':int(len(used)),
        'temporal_rejected_edges':int(len(indices)-len(used)),
        'connected_components_after_quality':int(ncomp),
        'nodes_with_two_core_forest_agreement':int(strict.sum()),
        'fraction_nodes_with_two_core_forest_agreement':float(strict.mean()),
        'nodes_in_two_core':int(two_core.sum()),
        'components_with_at_least_one_cycle':int(np.count_nonzero(component_has_loops)),
        'non_tree_edge_fraction_conflicting_any_epoch':frac_bad,
        'edge_conflict_fraction_p50_p90_p95_p99':_quantiles(conflict[cycleedge]),
        'node_forest_agreement_fraction_p50_p90_p95_p99':_quantiles(agree),
        'temporal_step_rad_p50_p90_p95_p99':_quantiles(maxstep[good]),
        'original_wrapped_phases_modified':False,
        'global_integer_gauge_solved':False,
        'physical_gradient_alias_excluded':False,
        'two_core_is_sufficient_proof':False,
        'caveat':'Temporal unwrap of wrapped spatial gradients assumes no unobserved >pi gradient change between acquisitions. A component-wise integer gauge remains unobservable; tree agreement is not external validation.',
    }
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
