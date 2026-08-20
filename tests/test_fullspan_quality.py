import numpy as np

import pypsds.phase_linking.fullspan_quality as fq

from pypsds.phase_linking.emi import (
    ESTIMATOR_EVD,
    ESTIMATOR_EMI,
    ESTIMATOR_INVALID,
)


def _fake_prepare(
    scale2,
    valid,
    ps,
    *,
    half_row,
    half_col,
):
    del (
        scale2,
        valid,
        ps,
        half_row,
        half_col,
    )

    return None


def _fake_glrt(
    ctx,
    rows,
    cols,
    *,
    alpha,
    nslc,
    block_size,
):
    del (
        ctx,
        cols,
        alpha,
        nslc,
        block_size,
    )

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
    yxt,
    rows,
    cols,
    support,
    pi,
    pj,
):
    del (
        yxt,
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


def test_fullspan_quality_complete_phase(
    monkeypatch,
):

    monkeypatch.setattr(
        fq,
        "prepare_glrt_window_context",
        _fake_prepare,
    )

    monkeypatch.setattr(
        fq,
        "glrt_support_vectorized_exact",
        _fake_glrt,
    )

    monkeypatch.setattr(
        fq,
        "compressed_coherence",
        _fake_coherence,
    )

    H = 5
    W = 5
    N = 4

    yxt = np.ones(
        (H, W, N),
        dtype=np.complex64,
    )

    rr = np.array(
        [2, 3],
        dtype=np.int32,
    )

    cc = np.array(
        [2, 2],
        dtype=np.int32,
    )

    phase = np.ones(
        (2, N),
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

    core = np.ones(
        (H, W),
        dtype=np.bool_,
    )

    expected_k = np.full(
        (H, W),
        9,
        dtype=np.int16,
    )

    out = (
        fq.evaluate_fullspan_quality_points(
            yxt=yxt,
            phase_points=phase,

            rows=rr,
            cols=cc,

            scale2=scale2,
            valid=valid,
            ps=ps,

            state_core=core,

            expected_effective_k=(
                expected_k
            ),

            half_row=1,
            half_col=1,

            batch=1,
            support_block=16,
        )
    )

    np.testing.assert_array_equal(
        out.effective_k,
        np.array(
            [9, 9],
            dtype=np.int16,
        ),
    )

    np.testing.assert_allclose(
        out.temporal_coherence,
        np.ones(
            2,
            dtype=np.float32,
        ),
        rtol=0,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        out.median_pair_coherence,
        np.ones(
            2,
            dtype=np.float32,
        ),
        rtol=0,
        atol=1e-7,
    )

    assert np.all(
        out.phase_complete
    )


def test_incomplete_phase_becomes_invalid(
    monkeypatch,
):

    monkeypatch.setattr(
        fq,
        "prepare_glrt_window_context",
        _fake_prepare,
    )

    monkeypatch.setattr(
        fq,
        "glrt_support_vectorized_exact",
        _fake_glrt,
    )

    monkeypatch.setattr(
        fq,
        "compressed_coherence",
        _fake_coherence,
    )

    H = 5
    W = 5
    N = 4

    phase = np.ones(
        (1, N),
        dtype=np.complex64,
    )

    phase[
        0,
        2,
    ] = np.complex64(
        np.nan
        +
        1j * np.nan
    )

    out = (
        fq.evaluate_fullspan_quality_points(
            yxt=np.ones(
                (H, W, N),
                dtype=np.complex64,
            ),

            phase_points=phase,

            rows=np.array(
                [2],
                dtype=np.int32,
            ),

            cols=np.array(
                [2],
                dtype=np.int32,
            ),

            scale2=np.ones(
                (H, W),
                dtype=np.float32,
            ),

            valid=np.ones(
                (H, W),
                dtype=np.bool_,
            ),

            ps=np.zeros(
                (H, W),
                dtype=np.bool_,
            ),

            state_core=np.ones(
                (H, W),
                dtype=np.bool_,
            ),

            expected_effective_k=np.full(
                (H, W),
                9,
                dtype=np.int16,
            ),

            half_row=1,
            half_col=1,
        )
    )

    assert not out.phase_complete[0]

    assert np.isnan(
        out.temporal_coherence[0]
    )

    assert np.isnan(
        out.median_pair_coherence[0]
    )


def test_stage_estimator_aggregation():

    s0 = np.array(
        [
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    s1 = np.array(
        [
            [1, 0, 255],
        ],
        dtype=np.uint8,
    )

    rr = np.array(
        [0, 0, 0],
        dtype=np.int32,
    )

    cc = np.array(
        [0, 1, 2],
        dtype=np.int32,
    )

    got = fq.aggregate_stage_estimators(
        (
            s0,
            s1,
        ),
        rows=rr,
        cols=cc,
    )

    np.testing.assert_array_equal(
        got,
        np.array(
            [
                ESTIMATOR_EMI,
                ESTIMATOR_EVD,
                ESTIMATOR_INVALID,
            ],
            dtype=np.uint8,
        ),
    )
