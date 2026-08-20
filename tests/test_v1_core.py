import numpy as np

from pypsds.phase_linking import (
    compressed_coherence,
    image_pairs,
    robust_emi_batch,
    temporal_coherence,
)

from pypsds.selection import (
    glrt_statistic,
    glrt_threshold,
)

from pypsds.runtime import (
    build_runtime_plan,
)


def test_glrt_equal_scale_zero():

    stat = glrt_statistic(
        np.array([2.0]),
        np.array([2.0]),
        nslc=38,
    )

    assert np.allclose(
        stat,
        0.0,
    )

    assert (
        glrt_threshold(0.005)
        > 0
    )


def test_compressed_coherence_perfect():

    H, W, N = 5, 5, 4

    phase = np.exp(
        1j
        *
        np.array(
            [0, 0.2, -0.4, 0.7],
            dtype=np.float32,
        )
    )

    x = np.empty(
        (H, W, N),
        np.complex64,
    )

    for r in range(H):
        for c in range(W):

            x[r, c] = (
                phase
                *
                np.complex64(
                    2
                    + 0.1*r
                    + 0.05*c
                )
            )

    rows = np.array(
        [2],
        np.int32,
    )

    cols = np.array(
        [2],
        np.int32,
    )

    support = np.ones(
        (1, 3, 3),
        bool,
    )

    pairs = image_pairs(N)

    coh = compressed_coherence(
        x,
        rows,
        cols,
        support,
        pairs[:, 0],
        pairs[:, 1],
    )

    assert np.allclose(
        np.abs(coh),
        1.0,
        atol=1e-5,
    )


def test_robust_emi_perfect():

    N = 6

    pairs = image_pairs(N)

    true_phase = np.exp(
        1j
        *
        np.linspace(
            0,
            1,
            N,
        )
    ).astype(
        np.complex64
    )

    coh = (
        true_phase[
            pairs[:, 0]
        ]
        *
        np.conj(
            true_phase[
                pairs[:, 1]
            ]
        )
    )[None, :]

    (
        phase,
        est,
        emi,
        evd,
        gmin,
    ) = robust_emi_batch(
        coh,
        n_images=N,
        pairs=pairs,
        beta=0.05,
        gamma_jitter=1e-6,
        emi_mu=0.99,
        reference_idx=0,
    )

    tc = temporal_coherence(
        coh,
        phase,
        pairs,
    )

    assert est[0] in (0, 1)

    assert np.isfinite(
        tc[0]
    )

    assert tc[0] > 0.999


def test_runtime_plan():

    plan = build_runtime_plan(
        ndate=38,
    )

    assert plan.cpu_count >= 1

    assert (
        plan.phase_link_workers
        >= 1
    )

    assert (
        plan.phase_link_chunk_size
        >= 64
    )

    assert (
        plan.phase_link_batch_size
        >=
        plan.phase_link_chunk_size
    )
