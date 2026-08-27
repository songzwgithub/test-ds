from __future__ import annotations

import ast
import inspect
from pathlib import Path

from pypsds.phase_linking import (
    fullspan_quality,
    sequential_production,
)


ROOT = Path(__file__).resolve().parents[1]


def _call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_fullspan_api_exposes_support_coordinates():
    sig = inspect.signature(
        fullspan_quality.evaluate_fullspan_quality_points
    )

    assert "support_rows" in sig.parameters
    assert "support_cols" in sig.parameters


def test_fullspan_cache_lookup_uses_support_coordinates():
    src = inspect.getsource(
        fullspan_quality.evaluate_fullspan_quality_points
    )

    assert "support_rr = rr" in src
    assert "support_cc = cc" in src
    assert "support_rr[" in src
    assert "support_cc[" in src


def test_fused_helper_receives_static_support_cache():
    sig = inspect.signature(
        sequential_production._run_gamma_post_phase_fused
    )

    assert "static_support_cache" in sig.parameters


def test_fused_quality_call_has_global_cache_bridge():
    src = inspect.getsource(
        sequential_production._run_gamma_post_phase_fused
    )

    tree = ast.parse(src)

    calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and
            _call_name(node)
            ==
            "evaluate_fullspan_quality_points"
        )
    ]

    assert len(calls) == 1

    keys = {
        kw.arg
        for kw in calls[0].keywords
        if kw.arg is not None
    }

    assert {
        "static_support_cache",
        "support_rows",
        "support_cols",
    }.issubset(keys)


def test_production_passes_cache_to_fused_helper():
    src = (
        ROOT
        /
        "pypsds"
        /
        "phase_linking"
        /
        "sequential_production.py"
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(src)

    calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and
            _call_name(node)
            ==
            "_run_gamma_post_phase_fused"
        )
    ]

    assert len(calls) == 1

    keys = {
        kw.arg
        for kw in calls[0].keywords
        if kw.arg is not None
    }

    assert "static_support_cache" in keys
