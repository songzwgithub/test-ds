from __future__ import annotations

import ast
import inspect

from pypsds.phase_linking import (
    fullspan_quality,
)


def _call_name(call):

    if isinstance(
        call.func,
        ast.Name,
    ):
        return call.func.id

    if isinstance(
        call.func,
        ast.Attribute,
    ):
        return call.func.attr

    return None


def test_production_fullspan_uses_two_step_reference_chain():

    src = inspect.getsource(
        fullspan_quality
        .evaluate_fullspan_quality_points
    )

    tree = ast.parse(
        src
    )

    called = {
        _call_name(
            node
        )
        for node in ast.walk(
            tree
        )
        if isinstance(
            node,
            ast.Call,
        )
    }

    assert (
        (
            "compressed_coherence"
            in called
        )
        or
        (
            "compressed_coherence_all_pairs"
            in called
        )
    )

    assert (
        "temporal_quality_streaming"
        in called
    )



def test_split_timing_is_emitted():

    src = inspect.getsource(
        fullspan_quality
        .evaluate_fullspan_quality_points
    )

    assert (
        "fullspan_coherence_seconds"
        in src
    )

    assert (
        "fullspan_temporal_seconds"
        in src
    )

    assert (
        "fullspan split timing"
        in src
    )
