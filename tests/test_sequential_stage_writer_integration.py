import numpy as np

import pypsds.phase_linking.sequential_multistage as sm

from pypsds.phase_linking.sequential_phase_writer import (
    SequentialPhaseWriter,
)


def _fake_glrt(
    ctx,
    rows,
    cols,
    *,
    alpha,
    nslc,
    block_size,
):
    """
    Deterministic 3x3 support.

    State-core masking inside run_sequential_stage()
    reduces this to the single active state pixel.
    """

    del ctx, cols, alpha, nslc, block_size

    b = len(rows)

    support = np.ones(
        (b, 3, 3),
        dtype=np.bool_,
    )

    K = np.full(
        b,
        9,
        dtype=np.int16,
    )

    return support, K


def _fake_coherence(
    stage_tile,
    rows,
    cols,
    support,
    pi,
    pj,
):
    del (
        stage_tile,
        cols,
        support,
        pj,
    )

    return np.ones(
        (
            len(rows),
            len(pi),
        ),
        dtype=np.complex64,
    )


def _fake_emi(
    coh,
    *,
    n_images,
    pairs,
    beta,
    gamma_jitter,
    emi_mu,
    reference_idx,
    workers,
    chunk_size,
):
    del (
        pairs,
        beta,
        gamma_jitter,
        emi_mu,
        workers,
        chunk_size,
    )

    b = coh.shape[0]

    # Deterministic phase history.
    angle = (
        0.07
        *
        np.arange(
            n_images,
            dtype=np.float32,
        )
    )

    ph = np.exp(
        1j * angle
    ).astype(
        np.complex64
    )

    ph = np.broadcast_to(
        ph,
        (b, n_images),
    ).copy()

    # Match real EMI reference semantics:
    # requested reference becomes phase origin.
    ph *= np.conj(
        ph[
            :,
            reference_idx,
        ][
            :,
            None
        ]
    )

    est = np.zeros(
        b,
        dtype=np.uint8,
    )

    dummy = np.zeros(
        b,
        dtype=np.float32,
    )

    return (
        ph,
        est,
        dummy.copy(),
        dummy.copy(),
        dummy.copy(),
    )


def _fake_tc(
    coh,
    ph,
    pairs,
):
    del ph, pairs

    return np.ones(
        coh.shape[0],
        dtype=np.float32,
    )


def test_two_stage_executor_to_linked_phase(
    tmp_path,
    monkeypatch,
):

    # --------------------------------------------------------
    # Patch only numerical kernels.
    #
    # Everything else remains production code:
    #   run_sequential_stage
    #   state routing
    #   immediate compression
    #   phase_sink
    #   SequentialPhaseWriter
    # --------------------------------------------------------

    monkeypatch.setattr(
        sm,
        "glrt_support_vectorized_exact",
        _fake_glrt,
    )

    monkeypatch.setattr(
        sm,
        "compressed_coherence",
        _fake_coherence,
    )

    monkeypatch.setattr(
        sm,
        "robust_emi_threaded",
        _fake_emi,
    )

    monkeypatch.setattr(
        sm,
        "temporal_coherence",
        _fake_tc,
    )

    # --------------------------------------------------------
    # Synthetic scene.
    #
    # N=6 miniature analogue of production:
    #
    #   stage0: real 0:3
    #   stage1: compressed c0000 + real 3:6
    # --------------------------------------------------------

    H = 5
    W = 5
    N = 6

    rng = np.random.default_rng(
        20260819
    )

    yxt = (
        rng.normal(
            size=(H, W, N)
        )
        +
        1j
        *
        rng.normal(
            size=(H, W, N)
        )
    ).astype(
        np.complex64
    )

    # Avoid pathological zero amplitude.
    yxt += np.complex64(
        2.0 + 0.5j
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

    # One interior compressed-state pixel.
    state_core = np.zeros(
        (H, W),
        dtype=np.bool_,
    )

    state_core[
        2,
        2,
    ] = True

    # After:
    #
    # GLRT support
    #   ∩ K-state core
    #   ∩ sequential availability
    #
    # effective K = 1 at the test center.
    expected_k = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    expected_k[
        2,
        2,
    ] = 1

    outdir = (
        tmp_path
        /
        "sequential"
    )

    phase_path = (
        tmp_path
        /
        "linked_phase.npy"
    )

    writer = SequentialPhaseWriter(
        phase_path,
        ndate=N,
        rows=H,
        cols=W,
        overwrite=True,
    )

    # ========================================================
    # Stage 0
    # ========================================================

    r0 = sm.run_sequential_stage(
        stage_index=0,

        compressed_input_ids=(),
        compressed_inputs=(),

        yxt=yxt,
        real_indices=(0, 1, 2),

        scale2=scale2,
        valid=valid,
        ps=ps,

        state_core=state_core,
        expected_effective_k=expected_k,

        output_dir=outdir,

        full_glrt_nslc=N,
        state_min_shp=1,

        inputs_complete=True,

        half_row=1,
        half_col=1,
        alpha=0.005,

        beta=0.0,
        gamma_jitter=1e-6,
        emi_mu=0.99,

        tile_rows=5,
        tile_cols=5,

        center_batch=16,
        support_block=16,

        pl_workers=1,
        pl_chunk_size=16,

        formula_audit_points=0,

        phase_sink=writer,
    )

    assert r0.solver_size == 3
    assert r0.first_real_idx == 0
    assert r0.reference_idx == 0

    assert r0.state_pixels == 1
    assert r0.state_valid == 1

    assert r0.low_k == 0
    assert r0.pl_invalid == 0
    assert r0.compression_invalid == 0

    assert r0.static_k_mismatch == 0
    assert r0.static_k_excess == 0

    comp0 = np.load(
        r0.compressed_path,
        mmap_mode="r",
    )

    assert np.isfinite(
        comp0[
            2,
            2,
        ].real
    )

    assert np.isfinite(
        comp0[
            2,
            2,
        ].imag
    )

    # ========================================================
    # Stage 1
    #
    # Feed REAL production compressed output from stage 0.
    # ========================================================

    r1 = sm.run_sequential_stage(
        stage_index=1,

        compressed_input_ids=(
            "c0000",
        ),

        compressed_inputs=(
            comp0,
        ),

        yxt=yxt,
        real_indices=(3, 4, 5),

        scale2=scale2,
        valid=valid,
        ps=ps,

        state_core=state_core,
        expected_effective_k=expected_k,

        output_dir=outdir,

        full_glrt_nslc=N,
        state_min_shp=1,

        inputs_complete=True,

        half_row=1,
        half_col=1,
        alpha=0.005,

        beta=0.0,
        gamma_jitter=1e-6,
        emi_mu=0.99,

        tile_rows=5,
        tile_cols=5,

        center_batch=16,
        support_block=16,

        pl_workers=1,
        pl_chunk_size=16,

        formula_audit_points=0,

        phase_sink=writer,
    )

    assert r1.solver_size == 4
    assert r1.first_real_idx == 1
    assert r1.reference_idx == 0

    assert r1.state_pixels == 1
    assert r1.state_valid == 1

    assert r1.low_k == 0
    assert r1.pl_invalid == 0
    assert r1.compression_invalid == 0

    assert r1.static_k_mismatch == 0
    assert r1.static_k_excess == 0

    writer.flush()

    # ========================================================
    # Validate final production-format phase cube.
    # ========================================================

    cube = np.load(
        phase_path,
        mmap_mode="r",
    )

    assert cube.shape == (
        N,
        H,
        W,
    )

    assert (
        cube.dtype
        ==
        np.dtype(
            np.complex64
        )
    )

    # Exactly one routed center is finite at every date.
    counts = writer.finite_counts()

    np.testing.assert_array_equal(
        counts,
        np.ones(
            N,
            dtype=np.int64,
        ),
    )

    # No missing dates at the sequential route center.
    center_phase = cube[
        :,
        2,
        2,
    ]

    assert np.all(
        np.isfinite(
            center_phase.real
        )
        &
        np.isfinite(
            center_phase.imag
        )
    )

    # Everything outside the route remains sparse zero.
    np.testing.assert_array_equal(
        cube[
            :,
            0,
            0,
        ],
        np.zeros(
            N,
            dtype=np.complex64,
        ),
    )

    # --------------------------------------------------------
    # Expected deterministic fake-EMI phases.
    #
    # Stage0:
    # solver [real0 real1 real2]
    # => [0, .07, .14]
    #
    # Stage1:
    # solver [c0000 real3 real4 real5]
    # => real [.07, .14, .21]
    # --------------------------------------------------------

    expected_angle = np.array(
        [
            0.00,
            0.07,
            0.14,
            0.07,
            0.14,
            0.21,
        ],
        dtype=np.float32,
    )

    expected_phase = np.exp(
        1j
        *
        expected_angle
    ).astype(
        np.complex64
    )

    np.testing.assert_allclose(
        center_phase,
        expected_phase,
        rtol=0,
        atol=3e-7,
    )

    # Every real acquisition written exactly once.
    np.testing.assert_array_equal(
        writer.written_counts,
        np.ones(
            N,
            dtype=np.int64,
        ),
    )

    writer.close()
