import numpy as np

from pypsds.phase_linking.shp_policy import (
    resolve_shp_policy,
    split_fallback_by_rank,
    window_capacity,
)


def cfg(strategy="sequential"):
    return {
        "selection": {
            "shp": {
                "half_row": 5,
                "half_col": 11,
                "min_count": 48,
                "policy": "solver_aware",
                "rank_guard": True,
                "adaptive_window": {
                    "enabled": True,
                },
            },
        },
        "phase_linking": {
            "temporal_reference_index": 0,
            "temporal": {
                "strategy": strategy,
                "ministack_size": 19,
                "max_num_compressed": 5,
                "state_min_shp": 24,
                "full_scm_fallback": True,
            },
        },
    }


def dates(n):
    return tuple(f"D{i:04d}" for i in range(n))


def test_reference_n38_is_scientifically_neutral():
    p = resolve_shp_policy(cfg(), dates(38))
    assert p.max_solver_size == 20
    assert p.state_min_shp == 24
    assert p.formal_min_shp == 48
    assert p.full_scm_rank_min_shp == 48
    assert (p.half_row, p.half_col) == (5, 11)
    assert p.window_capacity == 252
    assert not p.window_adapted


def test_sequential_n83_uses_solver_dimension_for_state():
    p = resolve_shp_policy(cfg(), dates(83))
    assert p.effective_strategy == "sequential"
    assert p.max_solver_size <= 24
    assert p.state_min_shp == 24
    assert p.formal_min_shp == 48
    assert p.full_scm_rank_min_shp == 83


def test_full_scm_n83_requires_k83():
    p = resolve_shp_policy(cfg("full_scm"), dates(83))
    assert p.effective_strategy == "full_scm"
    assert p.max_solver_size == 83
    assert p.state_min_shp == 83
    assert p.formal_min_shp == 83
    assert p.full_scm_rank_min_shp == 83


def test_n300_expands_window_capacity():
    p = resolve_shp_policy(cfg(), dates(300))
    assert p.full_scm_rank_min_shp == 300
    assert p.window_adapted
    assert window_capacity(p.half_row, p.half_col) >= 300


def test_fallback_rank_split_is_conservative():
    fallback = np.array([[True, True], [True, False]], dtype=bool)
    K = np.array([[48, 83], [100, -1]], dtype=np.int16)

    supported, under = split_fallback_by_rank(
        fallback,
        K,
        full_scm_min_shp=83,
        rank_guard=True,
    )

    assert np.array_equal(
        supported,
        np.array([[False, True], [True, False]], dtype=bool),
    )
    assert np.array_equal(supported | under, fallback)
    assert not np.any(supported & under)
