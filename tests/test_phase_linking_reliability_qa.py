import numpy as np

from pypsds.phase_linking.emi import image_pairs
from pypsds.phase_linking.reliability_qa import (
    connected_support_count,
    crlb_median_std_from_compressed,
    nearest_triplet_closure_metrics,
    sampled_phase_similarity,
)


def test_connected_support_excludes_island():
    support = np.zeros(
        (
            1,
            5,
            5,
        ),
        dtype=bool,
    )

    support[
        0,
        2,
        1,
    ] = True

    support[
        0,
        1,
        1,
    ] = True

    support[
        0,
        4,
        4,
    ] = True

    out = connected_support_count(
        support
    )

    assert int(
        out[0]
    ) == 2


def test_consistent_phase_has_zero_nearest_triplet_closure():
    n = 5
    pairs = image_pairs(
        n
    )

    theta = np.array(
        [
            0.0,
            0.2,
            0.7,
            1.1,
            1.4,
        ],
        dtype=np.float64,
    )

    coh = np.empty(
        (
            1,
            pairs.shape[0],
        ),
        dtype=np.complex64,
    )

    for q, (i, j) in enumerate(
        pairs
    ):
        coh[
            0,
            q,
        ] = np.exp(
            1j
            *
            (
                theta[i]
                -
                theta[j]
            )
        )

    rms, med, mx = nearest_triplet_closure_metrics(
        coh,
        pairs,
        n,
    )

    assert float(
        rms[0]
    ) < 1.0e-6

    assert float(
        med[0]
    ) < 1.0e-6

    assert float(
        mx[0]
    ) < 1.0e-6


def test_crlb_finite_for_ar1_coherence():
    n = 5
    pairs = image_pairs(
        n
    )

    G = np.fromfunction(
        lambda i, j:
            0.8
            **
            np.abs(
                i
                -
                j
            ),
        (
            n,
            n,
        ),
        dtype=float,
    )

    coh = np.asarray(
        [
            [
                G[i, j]
                for i, j in pairs
            ]
        ],
        dtype=np.complex64,
    )

    out = crlb_median_std_from_compressed(
        coh,
        pairs,
        n,
        num_looks=5.0,
    )

    assert np.isfinite(
        out[0]
    )

    assert float(
        out[0]
    ) > 0.0


def test_identical_phase_field_similarity_is_one():
    phase = np.ones(
        (
            6,
            9,
            9,
        ),
        dtype=np.complex64,
    )

    mask = np.ones(
        (
            9,
            9,
        ),
        dtype=bool,
    )

    (
        rows,
        cols,
        med,
        mx,
    ) = sampled_phase_similarity(
        phase,
        mask,
        max_points=20,
        search_radius=3,
        nearest_n=2,
        batch_size=8,
    )

    assert rows.size == 20
    assert cols.size == 20

    assert np.allclose(
        med,
        1.0,
        atol=1.0e-7,
    )

    assert np.allclose(
        mx,
        1.0,
        atol=1.0e-7,
    )
