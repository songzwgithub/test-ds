from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import os

import numpy as np
from numba import njit
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class ComponentBridgeForest:
    """Deterministic sparse forest over local spatial components."""

    global_root: int
    parent: np.ndarray
    depth: np.ndarray
    forest_root: np.ndarray
    edge_group_index: np.ndarray
    selected_weak: np.ndarray
    selected_edge_count: int
    forest_count: int


@njit(cache=True)
def build_component_csr(labels, ncomp):
    """O(N) counting-sort CSR: component -> point ids."""
    n = labels.size
    counts = np.zeros(ncomp, dtype=np.int64)
    for i in range(n):
        counts[labels[i]] += 1
    offsets = np.zeros(ncomp + 1, dtype=np.int64)
    for c in range(ncomp):
        offsets[c + 1] = offsets[c] + counts[c]
    cursor = offsets[:-1].copy()
    order = np.empty(n, dtype=np.int32)
    for i in range(n):
        c = labels[i]
        p = cursor[c]
        order[p] = i
        cursor[c] += 1
    return offsets, order


@njit(cache=True)
def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _kruskal_forest(ncomp, edge_order, edge_a, edge_b):
    parent = np.arange(ncomp, dtype=np.int32)
    size = np.ones(ncomp, dtype=np.int32)
    selected = np.empty(max(0, ncomp - 1), dtype=np.int32)
    nsel = 0
    for k in range(edge_order.size):
        eid = edge_order[k]
        a = _uf_find(parent, edge_a[eid])
        b = _uf_find(parent, edge_b[eid])
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        selected[nsel] = eid
        nsel += 1
        if nsel == ncomp - 1:
            break
    roots = np.empty(ncomp, dtype=np.int32)
    for i in range(ncomp):
        roots[i] = _uf_find(parent, i)
    return selected[:nsel], roots


def _reduce_candidates(
    comp_a,
    comp_b,
    point_a,
    point_b,
    rows,
    cols,
    row_spacing,
    col_spacing,
    ncomp,
    keep_per_pair,
):
    comp_a = np.asarray(comp_a, dtype=np.int32)
    if comp_a.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    comp_b = np.asarray(comp_b, dtype=np.int32)
    point_a = np.asarray(point_a, dtype=np.int32)
    point_b = np.asarray(point_b, dtype=np.int32)

    swap = comp_a > comp_b
    if np.any(swap):
        a0 = comp_a.copy()
        b0 = comp_b.copy()
        p0 = point_a.copy()
        q0 = point_b.copy()
        comp_a[swap] = b0[swap]
        comp_b[swap] = a0[swap]
        point_a[swap] = q0[swap]
        point_b[swap] = p0[swap]

    dr = np.abs(rows[point_a].astype(np.int64) - rows[point_b].astype(np.int64))
    dc = np.abs(cols[point_a].astype(np.int64) - cols[point_b].astype(np.int64))
    radius = np.maximum(dr, dc).astype(np.int32)
    distance = np.hypot(
        dr * float(row_spacing),
        dc * float(col_spacing),
    ).astype(np.float32)

    edge_key = comp_a.astype(np.int64) * np.int64(ncomp) + comp_b.astype(np.int64)
    point_key = (point_a.astype(np.uint64) << np.uint64(32)) | point_b.astype(np.uint64)

    # Primary key: component pair. Within a pair: smallest Chebyshev radius,
    # then physical distance, then deterministic point ids.
    order = np.lexsort((point_key, distance, radius, edge_key))
    edge_key = edge_key[order]
    comp_a = comp_a[order]
    comp_b = comp_b[order]
    point_a = point_a[order]
    point_b = point_b[order]
    radius = radius[order]
    distance = distance[order]
    point_key = point_key[order]

    unique_pair = np.ones(edge_key.size, dtype=bool)
    if edge_key.size > 1:
        unique_pair[1:] = (
            (edge_key[1:] != edge_key[:-1])
            | (point_key[1:] != point_key[:-1])
        )

    edge_key = edge_key[unique_pair]
    comp_a = comp_a[unique_pair]
    comp_b = comp_b[unique_pair]
    point_a = point_a[unique_pair]
    point_b = point_b[unique_pair]
    radius = radius[unique_pair]
    distance = distance[unique_pair]

    first = np.ones(edge_key.size, dtype=bool)
    if edge_key.size > 1:
        first[1:] = edge_key[1:] != edge_key[:-1]
    starts = np.maximum.accumulate(
        np.where(first, np.arange(edge_key.size, dtype=np.int64), 0)
    )
    rank = np.arange(edge_key.size, dtype=np.int64) - starts
    keep = rank < int(keep_per_pair)

    return (
        comp_a[keep],
        comp_b[keep],
        point_a[keep],
        point_b[keep],
        radius[keep],
        distance[keep],
    )


def _extract_strip(
    r0,
    r1,
    nearest_indices,
    index_grid,
    labels,
    rows,
    cols,
    row_spacing,
    col_spacing,
    ncomp,
    keep_per_pair,
):
    H, W = index_grid.shape
    halo1 = min(H, r1 + 1)
    ir = nearest_indices[0, r0:halo1]
    ic = nearest_indices[1, r0:halo1]
    nearest_id = index_grid[ir, ic]
    nearest_label = labels[nearest_id]

    parts = [[] for _ in range(6)]
    nsrc = r1 - r0

    def add(a, b, la, lb):
        m = la != lb
        if not np.any(m):
            return
        reduced = _reduce_candidates(
            la[m],
            lb[m],
            a[m],
            b[m],
            rows,
            cols,
            row_spacing,
            col_spacing,
            ncomp,
            keep_per_pair,
        )
        for dst, src in zip(parts, reduced):
            dst.append(src)

    # Four half-neighbour directions are sufficient to enumerate every
    # 8-neighbour boundary of the discrete Voronoi tessellation exactly once.
    if W > 1 and nsrc > 0:
        add(
            nearest_id[:nsrc, :-1],
            nearest_id[:nsrc, 1:],
            nearest_label[:nsrc, :-1],
            nearest_label[:nsrc, 1:],
        )

    nd = min(nsrc, H - 1 - r0)
    if nd > 0:
        add(
            nearest_id[:nd, :],
            nearest_id[1 : nd + 1, :],
            nearest_label[:nd, :],
            nearest_label[1 : nd + 1, :],
        )
        if W > 1:
            add(
                nearest_id[:nd, :-1],
                nearest_id[1 : nd + 1, 1:],
                nearest_label[:nd, :-1],
                nearest_label[1 : nd + 1, 1:],
            )
            add(
                nearest_id[:nd, 1:],
                nearest_id[1 : nd + 1, :-1],
                nearest_label[:nd, 1:],
                nearest_label[1 : nd + 1, :-1],
            )

    merged = []
    dtypes = (
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.float32,
    )
    for seq, dtype in zip(parts, dtypes):
        merged.append(np.concatenate(seq) if seq else np.empty(0, dtype=dtype))
    if merged[0].size == 0:
        return tuple(merged)
    return _reduce_candidates(
        merged[0],
        merged[1],
        merged[2],
        merged[3],
        rows,
        cols,
        row_spacing,
        col_spacing,
        ncomp,
        keep_per_pair,
    )


def build_voronoi_component_candidates(
    rows,
    cols,
    labels,
    *,
    row_spacing=1.0,
    col_spacing=1.0,
    block_rows=256,
    workers=None,
    keep_per_pair=8,
):
    """
    Build a sparse component-adjacency graph without O(Npoint*Ncomponent) scans.

    A global chessboard Voronoi transform is O(H*W) in compiled SciPy code.
    Only boundaries between Voronoi cells are examined.  For every component
    pair, at most ``keep_per_pair`` shortest point-pair witnesses are retained.
    Threads share the large nearest-index array; no multi-process duplication.
    """
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    if rows.size == 0 or cols.size != rows.size or labels.size != rows.size:
        raise ValueError("rows/cols/labels must be non-empty and equal length")
    ncomp = int(labels.max()) + 1
    if ncomp < 1 or labels.min() < 0:
        raise ValueError("labels must be contiguous non-negative component ids")
    H = int(rows.max()) + 1
    W = int(cols.max()) + 1

    index_grid = np.full((H, W), -1, dtype=np.int32)
    index_grid[rows, cols] = np.arange(rows.size, dtype=np.int32)
    occupied = index_grid >= 0
    nearest_indices = ndimage.distance_transform_cdt(
        ~occupied,
        metric="chessboard",
        return_distances=False,
        return_indices=True,
    )
    del occupied

    block_rows = max(1, int(block_rows))
    spans = [(r0, min(H, r0 + block_rows)) for r0 in range(0, H, block_rows)]
    if workers is None:
        workers = min(16, os.cpu_count() or 1)
    workers = max(1, min(int(workers), len(spans)))

    kwargs = dict(
        nearest_indices=nearest_indices,
        index_grid=index_grid,
        labels=labels,
        rows=rows,
        cols=cols,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        ncomp=ncomp,
        keep_per_pair=keep_per_pair,
    )

    results = [None] * len(spans)
    if workers == 1:
        for i, (r0, r1) in enumerate(spans):
            results[i] = _extract_strip(r0, r1, **kwargs)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_extract_strip, r0, r1, **kwargs): i
                for i, (r0, r1) in enumerate(spans)
            }
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

    del nearest_indices

    arrays = []
    for j, dtype in enumerate(
        (np.int32, np.int32, np.int32, np.int32, np.int32, np.float32)
    ):
        seq = [x[j] for x in results if x is not None and x[j].size]
        arrays.append(np.concatenate(seq) if seq else np.empty(0, dtype=dtype))

    reduced = _reduce_candidates(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        rows,
        cols,
        row_spacing,
        col_spacing,
        ncomp,
        keep_per_pair,
    )
    return reduced, index_grid


def candidate_groups(comp_a, comp_b, radius, distance, ncomp):
    if comp_a.size == 0:
        raise RuntimeError("component candidate graph is empty")
    edge_key = comp_a.astype(np.int64) * np.int64(ncomp) + comp_b.astype(np.int64)
    first = np.ones(edge_key.size, dtype=bool)
    first[1:] = edge_key[1:] != edge_key[:-1]
    starts = np.flatnonzero(first).astype(np.int64)
    stops = np.r_[starts[1:], edge_key.size].astype(np.int64)
    edge_a = comp_a[starts].astype(np.int32, copy=False)
    edge_b = comp_b[starts].astype(np.int32, copy=False)
    edge_radius = radius[starts].astype(np.int32, copy=False)
    edge_distance = distance[starts].astype(np.float32, copy=False)
    support = (stops - starts).astype(np.int32)
    return starts, stops, edge_a, edge_b, edge_radius, edge_distance, support


def build_component_forest(
    comp_a,
    comp_b,
    radius,
    distance,
    ncomp,
    component_sizes,
    global_root,
    *,
    max_radius=30,
    max_distance_m=None,
):
    """
    Build a locality-constrained minimum spanning forest.

    Crucially, bridges longer than the scientific locality threshold are NOT
    invented simply to make the graph connected.  If physical gaps remain,
    they become detached forests and are excluded from the global-gauge strict
    solution downstream rather than receiving an unsupported 2pi offset.
    """
    starts, stops, edge_a, edge_b, edge_radius, edge_distance, support = (
        candidate_groups(comp_a, comp_b, radius, distance, ncomp)
    )

    # Scientific locality applies to every point-pair witness, not merely
    # to the shortest representative of a component pair.  This matters for
    # two-anchor registration: a pair may have one valid local witness and a
    # second much longer witness that must not silently bypass max_radius.
    witness_allowed = radius <= int(max_radius)
    if max_distance_m is not None:
        witness_allowed &= distance <= float(max_distance_m)

    effective_support = np.add.reduceat(
        witness_allowed.astype(np.int32, copy=False),
        starts,
    )

    # Find the first valid witness in every component-pair group. Candidates
    # are already sorted by radius, physical distance and point ids.
    valid_idx = np.flatnonzero(witness_allowed).astype(np.int64)
    first_valid = np.full(starts.size, -1, dtype=np.int64)
    if valid_idx.size:
        valid_group = np.searchsorted(
            starts,
            valid_idx,
            side="right",
        ) - 1
        first = np.ones(valid_idx.size, dtype=bool)
        if valid_idx.size > 1:
            first[1:] = valid_group[1:] != valid_group[:-1]
        first_valid[valid_group[first]] = valid_idx[first]

    allowed = first_valid >= 0
    allowed_ids = np.flatnonzero(allowed).astype(np.int64)

    effective_edge_radius = np.full(
        starts.size,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )
    effective_edge_distance = np.full(
        starts.size,
        np.inf,
        dtype=np.float32,
    )
    effective_edge_radius[allowed] = radius[first_valid[allowed]]
    effective_edge_distance[allowed] = distance[first_valid[allowed]]

    # "weak" means fewer than two independent witnesses that both satisfy the
    # production locality limits, not merely fewer than two raw candidates.
    weak = effective_support < 2
    both_singleton = (
        (component_sizes[edge_a] == 1)
        & (component_sizes[edge_b] == 1)
    )

    # Primary: bridge radius. Then prefer independent support and avoid
    # singleton/singleton bridges. Physical distance and ids give deterministic
    # tie breaks. All selected edges are already <= max_radius.
    if allowed_ids.size:
        order_local = np.lexsort(
            (
                edge_b[allowed_ids],
                edge_a[allowed_ids],
                effective_edge_distance[allowed_ids],
                both_singleton[allowed_ids].astype(np.int8),
                weak[allowed_ids].astype(np.int8),
                effective_edge_radius[allowed_ids],
            )
        )
        edge_order = allowed_ids[order_local]
    else:
        edge_order = np.empty(0, dtype=np.int64)

    selected, roots_uf = _kruskal_forest(
        ncomp,
        edge_order,
        edge_a,
        edge_b,
    )

    # Build selected adjacency.
    adj = [[] for _ in range(ncomp)]
    for eid0 in selected:
        eid = int(eid0)
        a = int(edge_a[eid])
        b = int(edge_b[eid])
        adj[a].append((b, eid))
        adj[b].append((a, eid))

    # Determine a deterministic root for each forest: largest point component,
    # with smallest component id as tie break.  The forest containing the global
    # largest R4 component is rooted at that exact component.
    # Pick one deterministic orientation root per union-find forest in
    # O(Ncomp log Ncomp), rather than repeatedly scanning all components for
    # every detached forest. Sort by UF root, then descending component size,
    # then ascending component id.
    component_ids = np.arange(ncomp, dtype=np.int32)
    root_order = np.lexsort(
        (
            component_ids,
            -np.asarray(component_sizes, dtype=np.int64),
            roots_uf,
        )
    )
    sorted_uf = roots_uf[root_order]
    first_in_forest = np.ones(ncomp, dtype=bool)
    if ncomp > 1:
        first_in_forest[1:] = sorted_uf[1:] != sorted_uf[:-1]
    forest_roots = root_order[first_in_forest].astype(np.int32, copy=False)

    # The global forest must be rooted at the exact global R4 component even
    # if another member were larger under a future alternative size metric.
    global_uf = int(roots_uf[global_root])
    root_uf = roots_uf[forest_roots]
    hit = np.flatnonzero(root_uf == global_uf)
    if hit.size != 1:
        raise RuntimeError("global component forest-root lookup failed")
    forest_roots = forest_roots.copy()
    forest_roots[int(hit[0])] = int(global_root)

    parent = np.full(ncomp, -2, dtype=np.int32)
    depth = np.full(ncomp, -1, dtype=np.int32)
    forest_root = np.full(ncomp, -1, dtype=np.int32)
    edge_group = np.full(ncomp, -1, dtype=np.int32)

    for root in np.sort(forest_roots):
        parent[root] = -1
        depth[root] = 0
        forest_root[root] = root
        queue = [root]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            # Deterministic traversal independent of Python dict/list history.
            for v, eid in sorted(adj[u], key=lambda x: (x[0], x[1])):
                if depth[v] >= 0:
                    continue
                parent[v] = u
                depth[v] = depth[u] + 1
                forest_root[v] = root
                edge_group[v] = eid
                queue.append(v)

    if np.any(depth < 0) or np.any(parent == -2):
        raise RuntimeError("failed to orient all component-forest nodes")

    selected_weak = np.zeros(ncomp, dtype=bool)
    m = edge_group >= 0
    selected_weak[m] = weak[edge_group[m]]

    return (
        ComponentBridgeForest(
            global_root=int(global_root),
            parent=parent,
            depth=depth,
            forest_root=forest_root,
            edge_group_index=edge_group,
            selected_weak=selected_weak,
            selected_edge_count=int(selected.size),
            forest_count=int(forest_roots.size),
        ),
        (starts, stops, edge_a, edge_b, edge_radius, edge_distance, support),
    )


def find_same_component_neighbor(
    point_id,
    component,
    index_grid,
    rows,
    cols,
    labels,
    max_radius=4,
):
    r0 = int(rows[point_id])
    c0 = int(cols[point_id])
    H, W = index_grid.shape
    for rad in range(1, max_radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                rr = r0 + dr
                cc = c0 + dc
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                q = int(index_grid[rr, cc])
                if q >= 0 and q != point_id and int(labels[q]) == int(component):
                    return q
    return -1


def select_forest_anchors(
    forest,
    groups,
    comp_a,
    comp_b,
    point_a,
    point_b,
    radius,
    distance,
    component_sizes,
    index_grid,
    rows,
    cols,
    labels,
    *,
    row_spacing=1.0,
    col_spacing=1.0,
    core_radius=4,
    max_anchor_radius=None,
    max_anchor_distance_m=None,
):
    """Select two deterministic witnesses for every selected forest edge."""
    starts, stops, edge_a, edge_b, edge_radius, edge_distance, support = groups
    result = []
    duplicate_count = 0
    synthetic_count = 0

    for child in range(forest.parent.size):
        parent = int(forest.parent[child])
        if parent < 0:
            continue
        eid = int(forest.edge_group_index[child])
        s = int(starts[eid])
        e = int(stops[eid])
        anchors = []
        for k in range(s, e):
            a = int(comp_a[k])
            b = int(comp_b[k])
            p = int(point_a[k])
            q = int(point_b[k])
            rad = int(radius[k])
            dist_m = float(distance[k])

            # Apply the production locality contract to EVERY witness.
            # Previously only the component-pair representative was capped,
            # allowing anchor #2 to exceed Rmax.
            if max_anchor_radius is not None and rad > int(max_anchor_radius):
                continue
            if (
                max_anchor_distance_m is not None
                and dist_m > float(max_anchor_distance_m)
            ):
                continue

            if a == child and b == parent:
                anchors.append((p, q, rad, dist_m, False))
            elif a == parent and b == child:
                anchors.append((q, p, rad, dist_m, False))

        uniq = []
        seen = set()
        for x in anchors:
            key = (x[0], x[1])
            if key not in seen:
                seen.add(key)
                uniq.append(x)
            if len(uniq) >= 2:
                break
        anchors = uniq
        if not anchors:
            raise RuntimeError(f"forest edge {child}->{parent} has no point anchor")

        if len(anchors) == 1:
            cp, pp, _, _, _ = anchors[0]
            alts = []
            c2 = find_same_component_neighbor(
                cp,
                child,
                index_grid,
                rows,
                cols,
                labels,
                core_radius,
            )
            if c2 >= 0:
                dr = abs(int(rows[c2]) - int(rows[pp]))
                dc = abs(int(cols[c2]) - int(cols[pp]))
                rad2 = max(dr, dc)
                dist2 = float(
                    math.hypot(
                        dr * float(row_spacing),
                        dc * float(col_spacing),
                    )
                )
                radius_ok = (
                    max_anchor_radius is None
                    or rad2 <= int(max_anchor_radius)
                )
                distance_ok = (
                    max_anchor_distance_m is None
                    or dist2 <= float(max_anchor_distance_m)
                )
                if radius_ok and distance_ok:
                    alts.append((c2, pp, rad2, dist2, True))
            p2 = find_same_component_neighbor(
                pp,
                parent,
                index_grid,
                rows,
                cols,
                labels,
                core_radius,
            )
            if p2 >= 0:
                dr = abs(int(rows[cp]) - int(rows[p2]))
                dc = abs(int(cols[cp]) - int(cols[p2]))
                rad2 = max(dr, dc)
                dist2 = float(
                    math.hypot(
                        dr * float(row_spacing),
                        dc * float(col_spacing),
                    )
                )
                radius_ok = (
                    max_anchor_radius is None
                    or rad2 <= int(max_anchor_radius)
                )
                distance_ok = (
                    max_anchor_distance_m is None
                    or dist2 <= float(max_anchor_distance_m)
                )
                if radius_ok and distance_ok:
                    alts.append((cp, p2, rad2, dist2, True))
            if alts:
                alts.sort(key=lambda x: (x[2], x[3], x[0], x[1]))
                anchors.append(alts[0])
                synthetic_count += 1
            else:
                # Both components are effectively singleton at local R4 scale.
                # Duplicate the witness only so downstream structures remain
                # rectangular; policy forces this component to long/low confidence.
                anchors.append((cp, pp, anchors[0][2], anchors[0][3], True))
                duplicate_count += 1

        result.append(
            (
                child,
                parent,
                int(forest.depth[child]),
                anchors[:2],
                bool(forest.selected_weak[child]),
            )
        )

    return result, synthetic_count, duplicate_count
