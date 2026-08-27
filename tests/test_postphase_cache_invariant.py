from __future__ import annotations

from collections import OrderedDict

from pypsds.phase_linking.phase_source import (
    GammaStreamingPhaseSource,
)


def test_postphase_configuration_never_shrinks_retained_temporal_cache():

    src = object.__new__(
        GammaStreamingPhaseSource
    )

    src.canonical_rows = 128
    src.canonical_cols = 256
    src.ndate = 38

    src.cache_max_cells = 80
    src._cache = OrderedDict(
        (
            (
                i,
                0,
                (0, 1),
            ),
            object(),
        )
        for i in range(80)
    )

    plan = (
        src
        .configure_postphase_fullspan_cache(
            local_H=600,
            local_W=2000,
            memory_fraction=0.10,
            clear_stage_cache=False,
        )
    )

    assert src.cache_max_cells >= 80
    assert len(src._cache) == 80
    assert plan["cache_max_cells"] >= 80
