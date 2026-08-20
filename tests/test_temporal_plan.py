from pypsds.phase_linking.temporal_plan import (
    build_temporal_plan,
)


def dates(n):
    return [
        f"D{i:04d}"
        for i in range(n)
    ]


def test_full_scm_one_stage():

    p = build_temporal_plan(
        dates(38),
        strategy="full_scm",
        ministack_size=30,
    )

    assert (
        p.effective_strategy
        ==
        "full_scm"
    )

    assert len(
        p.stages
    ) == 1

    assert (
        p.stages[0]
        .solver_size
        ==
        38
    )

    assert (
        p.stages[0]
        .real_indices
        ==
        tuple(
            range(38)
        )
    )


def test_sequential_exact_collapse():

    p = build_temporal_plan(
        dates(38),
        strategy="sequential",
        ministack_size=38,
    )

    assert p.exact_collapse
    assert p.execution_ready

    assert (
        p.effective_strategy
        ==
        "full_scm"
    )

    assert len(
        p.stages
    ) == 1

    assert (
        p.stages[0]
        .compressed_count
        ==
        0
    )

    assert (
        p.stages[0]
        .real_indices
        ==
        tuple(
            range(38)
        )
    )


def test_sequential_38_m15():

    p = build_temporal_plan(
        dates(38),
        strategy="sequential",
        ministack_size=15,
        max_num_compressed=5,
    )

    assert (
        p.effective_strategy
        ==
        "sequential"
    )

    assert p.execution_ready

    assert [
        x.real_count
        for x in p.stages
    ] == [
        15,
        15,
        8,
    ]

    assert [
        x.compressed_count
        for x in p.stages
    ] == [
        0,
        1,
        2,
    ]

    assert [
        x.solver_size
        for x in p.stages
    ] == [
        15,
        16,
        10,
    ]


def test_compressed_history_cap():

    p = build_temporal_plan(
        dates(100),
        strategy="sequential",
        ministack_size=10,
        max_num_compressed=2,
    )

    assert (
        p.max_compressed_inputs
        ==
        2
    )

    assert all(
        x.compressed_count <= 2
        for x in p.stages
    )


def test_auto_is_not_arbitrarily_resolved():

    p = build_temporal_plan(
        dates(500),
        strategy="auto",
        ministack_size=30,
    )

    assert (
        p.effective_strategy
        ==
        "unresolved"
    )

    assert not p.execution_ready

    assert len(
        p.stages
    ) == 0


def test_sequential_38_m19_production_plan():

    p = build_temporal_plan(
        dates(38),
        strategy="sequential",
        ministack_size=19,
        max_num_compressed=5,
        reference_index=0,
    )

    assert p.effective_strategy == "sequential"
    assert p.execution_ready
    assert not p.exact_collapse

    assert len(p.stages) == 2

    assert [
        x.real_count
        for x in p.stages
    ] == [
        19,
        19,
    ]

    assert [
        x.compressed_count
        for x in p.stages
    ] == [
        0,
        1,
    ]

    assert [
        x.solver_size
        for x in p.stages
    ] == [
        19,
        20,
    ]

    assert p.stages[0].real_indices == tuple(
        range(0, 19)
    )

    assert p.stages[1].real_indices == tuple(
        range(19, 38)
    )

    assert (
        p.stages[0].output_reference
        ==
        "real:0"
    )

    assert (
        p.stages[1].output_reference
        ==
        "compressed:c0000"
    )

    assert p.max_solver_size == 20
    assert p.max_compressed_inputs == 1
