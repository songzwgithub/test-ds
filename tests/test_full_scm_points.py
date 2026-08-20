import numpy as np
import pytest

import pypsds.phase_linking.full_scm_points as fs


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

    return (
        np.ones(
            (
                b,
                3,
                3,
            ),
            dtype=np.bool_,
        ),
        np.full(
            b,
            9,
            dtype=np.int16,
        ),
    )


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

    angle = (
        0.1
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
        (
            b,
            n_images,
        ),
    ).copy()

    ph *= np.conj(
        ph[
            :,
            reference_idx,
        ][
            :,
            None
        ]
    )

    est = np.ones(
        b,
        dtype=np.uint8,
    )

    x = np.ones(
        b,
        dtype=np.float32,
    )

    return (
        ph,
        est,
        x.copy(),
        x.copy(),
        x.copy(),
    )


def test_sparse_full_scm_executor(
    monkeypatch,
):

    monkeypatch.setattr(
        fs,
        "prepare_glrt_window_context",
        _fake_prepare,
    )

    monkeypatch.setattr(
        fs,
        "glrt_support_vectorized_exact",
        _fake_glrt,
    )

    monkeypatch.setattr(
        fs,
        "compressed_coherence",
        _fake_coherence,
    )

    monkeypatch.setattr(
        fs,
        "robust_emi_threaded",
        _fake_emi,
    )

    H = 5
    W = 5
    N = 4

    rr = np.array(
        [2, 3],
        dtype=np.int32,
    )

    cc = np.array(
        [2, 2],
        dtype=np.int32,
    )

    expected_k = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    expected_k[
        rr,
        cc,
    ] = 9

    calls = []

    def sink(**kw):

        calls.append(
            kw
        )

    out = fs.run_full_scm_points(
        yxt=np.ones(
            (
                H,
                W,
                N,
            ),
            dtype=np.complex64,
        ),

        rows=rr,
        cols=cc,

        scale2=np.ones(
            (
                H,
                W,
            ),
            dtype=np.float32,
        ),

        valid=np.ones(
            (
                H,
                W,
            ),
            dtype=np.bool_,
        ),

        ps=np.zeros(
            (
                H,
                W,
            ),
            dtype=np.bool_,
        ),

        expected_original_k=(
            expected_k
        ),

        phase_sink=sink,

        half_row=1,
        half_col=1,

        min_shp=9,

        beta=0.0,

        batch=1,
        support_block=16,

        pl_workers=1,
        pl_chunk_size=16,
    )

    assert out.size == 2
    assert out.valid_count == 2

    np.testing.assert_array_equal(
        out.shp_count,
        np.array(
            [9, 9],
            dtype=np.int16,
        ),
    )

    assert out.phase.shape == (
        2,
        N,
    )

    assert np.all(
        np.isfinite(
            out.temporal_coherence
        )
    )

    assert np.all(
        np.isfinite(
            out.median_pair_coherence
        )
    )

    assert len(calls) == 2

    assert calls[0][
        "stage_index"
    ] == -1

    assert calls[0][
        "real_indices"
    ] == (
        0,
        1,
        2,
        3,
    )


def test_original_k_parity_failure(
    monkeypatch,
):

    monkeypatch.setattr(
        fs,
        "prepare_glrt_window_context",
        _fake_prepare,
    )

    monkeypatch.setattr(
        fs,
        "glrt_support_vectorized_exact",
        _fake_glrt,
    )

    H = 5
    W = 5
    N = 4

    expected = np.full(
        (
            H,
            W,
        ),
        -1,
        dtype=np.int16,
    )

    expected[
        2,
        2,
    ] = 8

    with pytest.raises(
        RuntimeError,
        match="original-K parity",
    ):

        fs.run_full_scm_points(
            yxt=np.ones(
                (
                    H,
                    W,
                    N,
                ),
                dtype=np.complex64,
            ),

            rows=np.array(
                [2],
                dtype=np.int32,
            ),

            cols=np.array(
                [2],
                dtype=np.int32,
            ),

            scale2=np.ones(
                (
                    H,
                    W,
                ),
                dtype=np.float32,
            ),

            valid=np.ones(
                (
                    H,
                    W,
                ),
                dtype=np.bool_,
            ),

            ps=np.zeros(
                (
                    H,
                    W,
                ),
                dtype=np.bool_,
            ),

            expected_original_k=(
                expected
            ),

            half_row=1,
            half_col=1,

            min_shp=8,

            batch=1,
        )
