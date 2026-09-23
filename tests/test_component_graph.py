import numpy as np

from pypsds.component_graph import (
    build_component_csr,
    build_component_forest,
    build_voronoi_component_candidates,
    select_forest_anchors,
)


def _scene():
    points = []
    labels = []

    def add(label, coords):
        for rc in coords:
            points.append(rc)
            labels.append(label)

    add(0, [(2, 2), (2, 3), (3, 2), (3, 3)])
    add(1, [(2, 10), (2, 11), (3, 10), (3, 11)])
    add(2, [(10, 3), (10, 4), (11, 3), (11, 4)])
    add(3, [(12, 13), (12, 14), (13, 13), (13, 14)])

    rows = np.asarray([p[0] for p in points], dtype=np.int32)
    cols = np.asarray([p[1] for p in points], dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    return rows, cols, labels


def test_component_csr_exact_partition():
    rows, cols, labels = _scene()
    offsets, order = build_component_csr(labels, 4)
    assert offsets.tolist() == [0, 4, 8, 12, 16]
    for comp in range(4):
        ids = order[offsets[comp] : offsets[comp + 1]]
        assert np.all(labels[ids] == comp)


def test_voronoi_component_forest_is_hierarchical_when_local_bridges_exist():
    rows, cols, labels = _scene()
    (ca, cb, pa, pb, radius, distance), index_grid = (
        build_voronoi_component_candidates(
            rows,
            cols,
            labels,
            row_spacing=2.0,
            col_spacing=1.0,
            block_rows=5,
            workers=2,
            keep_per_pair=8,
        )
    )
    sizes = np.bincount(labels, minlength=4)
    forest, groups = build_component_forest(
        ca,
        cb,
        radius,
        distance,
        4,
        sizes,
        global_root=0,
        max_radius=30,
    )
    assert forest.parent[0] == -1
    assert forest.forest_count == 1
    assert np.all(forest.depth >= 0)
    assert np.count_nonzero(forest.parent >= 0) == 3
    # Component 3 reaches the global root through an intermediate component;
    # it is not forced to take a direct long edge to component 0.
    assert forest.parent[3] != 0

    anchors, synthetic, duplicate = select_forest_anchors(
        forest,
        groups,
        ca,
        cb,
        pa,
        pb,
        radius,
        distance,
        sizes,
        index_grid,
        rows,
        cols,
        labels,
        row_spacing=2.0,
        col_spacing=1.0,
    )
    assert len(anchors) == 3
    assert duplicate == 0
    assert all(len(row[3]) == 2 for row in anchors)


def test_bridge_limit_creates_forest_instead_of_unsupported_long_edge():
    rows, cols, labels = _scene()
    (ca, cb, pa, pb, radius, distance), _ = build_voronoi_component_candidates(
        rows,
        cols,
        labels,
        block_rows=4,
        workers=1,
        keep_per_pair=4,
    )
    sizes = np.bincount(labels, minlength=4)
    forest, _ = build_component_forest(
        ca,
        cb,
        radius,
        distance,
        4,
        sizes,
        global_root=0,
        max_radius=5,
    )
    assert forest.forest_count > 1
    assert np.count_nonzero(forest.parent < 0) == forest.forest_count
    assert forest.parent[0] == -1
