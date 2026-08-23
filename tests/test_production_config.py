from pathlib import Path

from pypsds.config import (
    load_config,
)


def test_production_config_schema():

    p = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "config"
        / "pypsds.yaml"
    )

    cfg, _ = load_config(p)

    assert (
        cfg["schema_version"]
        == 1
    )

    assert (
        cfg["selection"]
        ["ps"]
        ["amplitude_dispersion_max"]
        == 0.25
    )

    assert (
        cfg["corrections"]
        ["scla"]
        ["mode"]
        == "disabled"
    )


def test_production_sequential_production_config():

    from pypsds.phase_linking.temporal_plan import (
        build_temporal_plan,
    )

    p = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "config"
        / "pypsds.yaml"
    )

    cfg, _ = load_config(p)

    # Frozen EMI semantics.
    assert cfg["phase_linking"]["beta"] == 0.0
    assert (
        cfg["phase_linking"]["target_eigenvalue"]
        == 0.99
    )

    # Formal DS definition remains K >= 48.
    assert (
        cfg["selection"]["shp"]["min_count"]
        == 48
    )

    # Formal DS center domain.
    assert (
        cfg["selection"]["ds"]["center_mode"]
        == "all"
    )

    # Moraine KS is diagnostic only.
    assert (
        cfg["selection"]["center_prior"]["enabled"]
        is False
    )

    # Production DS quality gate.
    assert (
        cfg["selection"]["ds"]["temporal_coherence_min"]
        == 0.80
    )

    t = cfg["phase_linking"]["temporal"]

    assert t["strategy"] == "sequential"
    assert t["ministack_size"] == 19
    assert t["max_num_compressed"] == 5

    # This is compressed-state continuity only,
    # NOT the formal DS SHP threshold.
    assert t["state_min_shp"] == 24

    assert t["full_scm_fallback"] is True

    # Freeze the validated N=38 / M=19 planner contract.
    dates = tuple(
        f"D{i:02d}"
        for i in range(38)
    )

    plan = build_temporal_plan(
        dates,
        strategy=t["strategy"],
        ministack_size=t["ministack_size"],
        max_num_compressed=t["max_num_compressed"],
        reference_index=(
            cfg["phase_linking"]
            ["temporal_reference_index"]
        ),
    )

    assert plan.execution_ready
    assert plan.effective_strategy == "sequential"
    assert len(plan.stages) == 2

    assert [
        s.real_count
        for s in plan.stages
    ] == [
        19,
        19,
    ]

    assert [
        s.compressed_count
        for s in plan.stages
    ] == [
        0,
        1,
    ]

    assert [
        s.solver_size
        for s in plan.stages
    ] == [
        19,
        20,
    ]
