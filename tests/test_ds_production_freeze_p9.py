from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import yaml

from pypsds.pipeline import STAGE_INDEX


def _load_cfg():
    return yaml.safe_load(
        Path(
            "config/pypsds.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )


def test_p9_default_ds_policy_is_frozen():
    cfg = _load_cfg()

    shp = cfg[
        "selection"
    ][
        "shp"
    ]

    assert shp[
        "method"
    ] == "rayleigh_glrt"

    assert float(
        shp[
            "alpha"
        ]
    ) == 0.005

    assert int(
        shp[
            "half_row"
        ]
    ) == 5

    assert int(
        shp[
            "half_col"
        ]
    ) == 11

    assert int(
        shp[
            "min_count"
        ]
    ) == 48

    assert shp[
        "policy"
    ] == "solver_aware"

    assert shp[
        "rank_guard"
    ] is True

    assert shp[
        "adaptive_window"
    ][
        "enabled"
    ] is True

    pl = cfg[
        "phase_linking"
    ]

    assert pl[
        "method"
    ] == "robust_emi"

    assert float(
        pl[
            "beta"
        ]
    ) == 0.0

    assert float(
        pl[
            "target_eigenvalue"
        ]
    ) == 0.99

    temporal = pl[
        "temporal"
    ]

    assert temporal[
        "strategy"
    ] == "sequential"

    assert int(
        temporal[
            "ministack_size"
        ]
    ) == 19

    assert int(
        temporal[
            "max_num_compressed"
        ]
    ) == 5

    assert int(
        temporal[
            "state_min_shp"
        ]
    ) == 24

    ds = cfg[
        "selection"
    ][
        "ds"
    ]

    assert float(
        ds[
            "temporal_coherence_min"
        ]
    ) == 0.80

    assert float(
        ds[
            "pair_coherence_min"
        ]
    ) == 0.0

    assert ds[
        "accept_evd"
    ] is True


def test_adaptive_filter_is_disabled_and_position_is_explicit():
    cfg = _load_cfg()

    af = cfg[
        "adaptive_filter"
    ]

    assert af[
        "enabled"
    ] is False

    assert af[
        "domain"
    ] == "wrapped_virtual_interferogram"

    assert af[
        "method"
    ] == "goldstein_werner_adaptive"

    assert af[
        "alpha"
    ] is None

    assert af[
        "preserve_unfiltered"
    ] is True

    assert af[
        "benchmark_required"
    ] is True

    assert af[
        "planned_position"
    ] == (
        "after_unwrap_policy_before_unwrap"
    )

    assert (
        STAGE_INDEX[
            "network_finalize"
        ]
        <
        STAGE_INDEX[
            "virtual_ifg_quality"
        ]
    )


def test_packaged_policy_matches_default_profile():
    text = (
        resources.files(
            "pypsds.resources"
        )
        .joinpath(
            "ds_production_policy_v1.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    policy = json.loads(
        text
    )

    p = policy[
        "default_profile"
    ]

    assert p[
        "phase_linking"
    ][
        "temporal"
    ][
        "ministack_size"
    ] == 19

    assert p[
        "phase_linking"
    ][
        "temporal"
    ][
        "max_num_compressed"
    ] == 5

    assert p[
        "phase_linking"
    ][
        "pair_policy"
    ][
        "zero_correlation_threshold"
    ] == 0.0

    assert p[
        "phase_linking"
    ][
        "pair_policy"
    ][
        "baseline_lag"
    ] is None

    assert policy[
        "adaptive_interferogram_filter"
    ][
        "default_enabled"
    ] is False
