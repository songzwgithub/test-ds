import numpy as np

from pypsds.phase_linking.emi import (
    image_pairs,
    robust_emi_batch,
)

from pypsds.phase_linking.emi_threshold import (
    robust_emi_threshold_batch,
)


def _physical_coherence(
    *,
    seed,
    batch,
    n_images,
    samples,
):

    rng = np.random.default_rng(
        seed
    )

    z = (
        rng.normal(
            size=(
                batch,
                samples,
                n_images,
            )
        )
        +
        1j
        *
        rng.normal(
            size=(
                batch,
                samples,
                n_images,
            )
        )
    ).astype(
        np.complex64
    )

    C = np.einsum(
        "bki,bkj->bij",
        np.conj(z),
        z,
        optimize=True,
    )

    p = np.real(
        np.diagonal(
            C,
            axis1=1,
            axis2=2,
        )
    )

    den = np.sqrt(
        p[
            :,
            :,
            None,
        ]
        *
        p[
            :,
            None,
            :,
        ]
    )

    C = (
        C
        /
        den
    ).astype(
        np.complex64
    )

    pairs = image_pairs(
        n_images
    )

    coh = C[
        :,
        pairs[:, 0],
        pairs[:, 1],
    ]

    return (
        coh,
        pairs,
    )


def test_threshold_cholesky_matches_current_emi():

    coh, pairs = _physical_coherence(
        seed=20260820,
        batch=128,
        n_images=19,
        samples=80,
    )

    ref = robust_emi_batch(
        coh,
        n_images=19,
        pairs=pairs,
        beta=0.0,
        gamma_jitter=1e-6,
        emi_mu=0.99,
        reference_idx=0,
        min_gamma_eig=1e-7,
    )

    fast = robust_emi_threshold_batch(
        coh,
        n_images=19,
        pairs=pairs,
        beta=0.0,
        gamma_jitter=1e-6,
        emi_mu=0.99,
        reference_idx=0,
        min_gamma_eig=1e-7,
    )

    np.testing.assert_array_equal(
        ref[1],
        fast[1],
    )

    ph0 = ref[0]
    ph1 = fast[0]

    finite = (
        np.isfinite(
            ph0.real
        )
        &
        np.isfinite(
            ph0.imag
        )
        &
        np.isfinite(
            ph1.real
        )
        &
        np.isfinite(
            ph1.imag
        )
    )

    assert np.all(
        finite
    )

    delta = np.angle(
        ph1
        *
        np.conj(
            ph0
        )
    )

    assert float(
        np.max(
            np.abs(
                delta
            )
        )
    ) <= 5e-6
