from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pypsds.phase_linking.sequential_plan_executor as spe

from pypsds.phase_linking.sequential_phase_writer import (
    SequentialPhaseWriter,
)
from pypsds.phase_linking.temporal_plan import (
    build_temporal_plan,
)


def test_two_stage_plan_orchestration(
    tmp_path,
    monkeypatch,
):

    H = 4
    W = 5
    N = 6

    dates = tuple(
        f"D{i}"
        for i in range(N)
    )

    plan = build_temporal_plan(
        dates,
        strategy="sequential",
        ministack_size=3,
        max_num_compressed=5,
        reference_index=0,
    )

    assert plan.execution_ready
    assert len(plan.stages) == 2

    yxt = np.ones(
        (H, W, N),
        dtype=np.complex64,
    )

    scale2 = np.ones(
        (H, W),
        dtype=np.float32,
    )

    valid = np.ones(
        (H, W),
        dtype=np.bool_,
    )

    ps = np.zeros(
        (H, W),
        dtype=np.bool_,
    )

    state_core = np.zeros(
        (H, W),
        dtype=np.bool_,
    )

    state_core[
        1,
        2,
    ] = True

    expected_k = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    expected_k[
        1,
        2,
    ] = 24

    writer = SequentialPhaseWriter(
        tmp_path
        /
        "linked_phase.npy",
        ndate=N,
        rows=H,
        cols=W,
        overwrite=True,
    )

    calls = []

    def fake_stage(
        **kw,
    ):

        stage_index = int(
            kw["stage_index"]
        )

        input_ids = tuple(
            kw[
                "compressed_input_ids"
            ]
        )

        compressed_inputs = tuple(
            kw[
                "compressed_inputs"
            ]
        )

        real_indices = tuple(
            kw[
                "real_indices"
            ]
        )

        calls.append(
            {
                "stage_index":
                    stage_index,

                "input_ids":
                    input_ids,

                "ncomp":
                    len(
                        compressed_inputs
                    ),

                "inputs_complete":
                    kw[
                        "inputs_complete"
                    ],
            }
        )

        # Stage 1 must receive the ACTUAL stage-0
        # compressed product through the registry.
        if stage_index == 0:

            assert input_ids == ()
            assert len(
                compressed_inputs
            ) == 0

        elif stage_index == 1:

            assert input_ids == (
                "c0000",
            )

            assert len(
                compressed_inputs
            ) == 1

            np.testing.assert_allclose(
                compressed_inputs[
                    0
                ][
                    1,
                    2,
                ],
                np.complex64(
                    10.0
                    +
                    1.0j
                ),
                rtol=0,
                atol=0,
            )

        else:
            raise AssertionError(
                stage_index
            )

        # Emit deterministic REAL acquisition phase
        # through the production sink contract.
        rr = np.array(
            [1],
            dtype=np.int32,
        )

        cc = np.array(
            [2],
            dtype=np.int32,
        )

        angle = (
            0.1
            *
            np.asarray(
                real_indices,
                dtype=np.float32,
            )
        )

        ph = np.exp(
            1j
            *
            angle[
                None,
                :
            ]
        ).astype(
            np.complex64
        )

        kw[
            "phase_sink"
        ](
            stage_index=(
                stage_index
            ),
            real_indices=(
                real_indices
            ),
            rows=rr,
            cols=cc,
            phase=ph,
        )

        outdir = Path(
            kw["output_dir"]
        )

        outdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        comp_path = (
            outdir
            /
            f"fake_stage"
            f"{stage_index:04d}"
            "_compressed.npy"
        )

        valid_path = (
            outdir
            /
            f"fake_stage"
            f"{stage_index:04d}"
            "_valid.npy"
        )

        comp = np.full(
            (H, W),
            np.complex64(
                np.nan
                +
                1j * np.nan
            ),
            dtype=np.complex64,
        )

        comp[
            1,
            2,
        ] = np.complex64(
            10.0
            +
            (stage_index + 1)
            * 1.0j
        )

        comp_valid = np.zeros(
            (H, W),
            dtype=np.bool_,
        )

        comp_valid[
            1,
            2,
        ] = True

        np.save(
            comp_path,
            comp,
        )

        np.save(
            valid_path,
            comp_valid,
        )

        return SimpleNamespace(
            compressed_path=(
                comp_path
            ),
            valid_path=(
                valid_path
            ),
            elapsed_seconds=(
                1.25
            ),
        )

    monkeypatch.setattr(
        spe,
        "run_sequential_stage",
        fake_stage,
    )

    result = spe.run_sequential_plan(
        plan=plan,

        yxt=yxt,
        scale2=scale2,
        valid=valid,
        ps=ps,

        state_core=(
            state_core
        ),

        expected_effective_k=(
            expected_k
        ),

        output_dir=(
            tmp_path
            /
            "sequential"
        ),

        phase_sink=writer,

        full_glrt_nslc=N,
        state_min_shp=24,

        half_row=1,
        half_col=1,

        beta=0.0,

        tile_rows=4,
        tile_cols=5,

        center_batch=16,
        support_block=16,

        pl_workers=1,
        pl_chunk_size=16,

        formula_audit_points=0,
    )

    assert result.stage_count == 2

    assert (
        result.total_stage_seconds
        ==
        2.5
    )

    assert set(
        result.compressed_paths
    ) == {
        "c0000",
        "c0001",
    }

    assert set(
        result.valid_paths
    ) == {
        "c0000",
        "c0001",
    }

    assert calls == [
        {
            "stage_index": 0,
            "input_ids": (),
            "ncomp": 0,
            "inputs_complete": True,
        },
        {
            "stage_index": 1,
            "input_ids": (
                "c0000",
            ),
            "ncomp": 1,
            "inputs_complete": True,
        },
    ]

    writer.flush()

    cube = np.load(
        writer.path,
        mmap_mode="r",
    )

    center = cube[
        :,
        1,
        2,
    ]

    expected = np.exp(
        1j
        *
        (
            0.1
            *
            np.arange(
                N,
                dtype=np.float32,
            )
        )
    ).astype(
        np.complex64
    )

    np.testing.assert_allclose(
        center,
        expected,
        rtol=0,
        atol=2e-7,
    )

    np.testing.assert_array_equal(
        writer.finite_counts(),
        np.ones(
            N,
            dtype=np.int64,
        ),
    )

    writer.close()


def test_reject_full_scm_plan(
    tmp_path,
):

    plan = build_temporal_plan(
        ("D0", "D1", "D2"),
        strategy="full_scm",
        reference_index=0,
    )

    x = np.ones(
        (2, 2, 3),
        dtype=np.complex64,
    )

    m = np.ones(
        (2, 2),
        dtype=np.bool_,
    )

    k = np.ones(
        (2, 2),
        dtype=np.int16,
    )

    try:

        spe.run_sequential_plan(
            plan=plan,
            yxt=x,
            scale2=np.ones(
                (2, 2),
                dtype=np.float32,
            ),
            valid=m,
            ps=np.zeros_like(
                m
            ),
            state_core=m,
            expected_effective_k=k,
            output_dir=tmp_path,
            full_glrt_nslc=3,
            state_min_shp=1,
        )

    except ValueError as e:

        assert "sequential" in str(
            e
        )

    else:

        raise AssertionError(
            "full-SCM plan was not rejected"
        )
