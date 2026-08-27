from __future__ import annotations

from pypsds.phase_linking.phase_source import (
    _plan_fullspan_cache_cells,
)


def test_current_scene_is_40_cells():

    plan = _plan_fullspan_cache_cells(
        H=600,
        W=2000,
        canonical_rows=128,
        canonical_cols=256,
        ndate=38,
        available_bytes=52 * 1024**3,
        memory_fraction=0.10,
    )

    assert plan["scene_rows"] == 5
    assert plan["scene_cols"] == 8
    assert plan["scene_cells"] == 40
    assert plan["target_cells"] == 40


def test_huge_scene_is_memory_bounded():

    plan = _plan_fullspan_cache_cells(
        H=50000,
        W=50000,
        canonical_rows=128,
        canonical_cols=256,
        ndate=38,
        available_bytes=2 * 1024**3,
        memory_fraction=0.10,
    )

    assert (
        plan["target_cells"]
        <=
        plan["memory_cells"]
    )

    assert (
        plan["target_cells"]
        <
        plan["scene_cells"]
    )


def test_memory_fraction_is_capped():

    plan = _plan_fullspan_cache_cells(
        H=1000,
        W=1000,
        canonical_rows=128,
        canonical_cols=256,
        ndate=38,
        available_bytes=1024**3,
        memory_fraction=0.90,
    )

    assert (
        plan["memory_budget_bytes"]
        <=
        int(
            0.25
            *
            1024**3
        )
    )
