from __future__ import annotations

from collections import OrderedDict

import numpy as np

from pypsds.phase_linking.phase_source import (
    GammaStreamingPhaseSource,
    _CanonicalCell,
    _plan_temporal_piece_cache_cells,
)


def _cell(values, *, geom=True):
    yxt = np.asarray(
        values,
        dtype=np.complex64,
    ).reshape(
        1,
        1,
        -1,
    )

    geometry = np.asarray(
        [[geom]],
        dtype=np.bool_,
    )

    return _CanonicalCell(
        row0=0,
        row1=1,
        col0=0,
        col1=1,
        yxt=yxt,
        geometry_valid=geometry,
        phase_min=-2.0,
        phase_max=3.0,
    )


def test_current_scene_retains_two_temporal_parts():
    plan = _plan_temporal_piece_cache_cells(
        H=600,
        W=2000,
        canonical_rows=128,
        canonical_cols=256,
        ndate=38,
        temporal_parts=2,
        available_bytes=52 * 1024**3,
        memory_fraction=0.10,
    )

    assert plan["scene_cells"] == 40
    assert plan["desired_entries"] == 80
    assert plan["target_entries"] == 80


def test_temporal_pieces_compose_exactly():
    src = object.__new__(
        GammaStreamingPhaseSource
    )
    src._cache = OrderedDict()

    src._cache[
        (0, 0, (0, 1))
    ] = _cell(
        [1 + 1j, 2 + 2j]
    )

    src._cache[
        (0, 0, (2, 3))
    ] = _cell(
        [3 + 3j, 4 + 4j]
    )

    out = src._cache_compose_temporal_cell(
        spatial_key=(0, 0),
        date_indices=(0, 1, 2, 3),
    )

    assert out is not None

    np.testing.assert_array_equal(
        out.yxt[0, 0, :],
        np.asarray(
            [
                1 + 1j,
                2 + 2j,
                3 + 3j,
                4 + 4j,
            ],
            dtype=np.complex64,
        ),
    )


def test_geometry_mismatch_forces_fallback():
    src = object.__new__(
        GammaStreamingPhaseSource
    )
    src._cache = OrderedDict()

    src._cache[
        (0, 0, (0, 1))
    ] = _cell(
        [1 + 0j, 2 + 0j],
        geom=True,
    )

    src._cache[
        (0, 0, (2, 3))
    ] = _cell(
        [3 + 0j, 4 + 0j],
        geom=False,
    )

    out = src._cache_compose_temporal_cell(
        spatial_key=(0, 0),
        date_indices=(0, 1, 2, 3),
    )

    assert out is None
