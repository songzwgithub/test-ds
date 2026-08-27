from __future__ import annotations

import numpy as np

from pypsds.phase_linking.coherence import (
    compressed_coherence,
    compressed_coherence_all_pairs,
)

from pypsds.phase_linking.emi import (
    image_pairs,
)


def _case(
    seed,
    *,
    H,
    W,
    N,
    B,
    hr,
    hc,
):

    rng = np.random.default_rng(
        seed
    )

    yxt = (
        rng.normal(
            size=(
                H,
                W,
                N,
            )
        )
        +
        1j
        *
        rng.normal(
            size=(
                H,
                W,
                N,
            )
        )
    ).astype(
        np.complex64
    )

    rows = rng.integers(
        hr,
        H - hr,
        size=B,
        dtype=np.int32,
    )

    cols = rng.integers(
        hc,
        W - hc,
        size=B,
        dtype=np.int32,
    )

    support = (
        rng.random(
            size=(
                B,
                2 * hr + 1,
                2 * hc + 1,
            )
        )
        <
        0.27
    )

    support[
        :,
        hr,
        hc,
    ] = True

    return (
        yxt,
        rows,
        cols,
        support,
    )


def _compare(case):

    (
        yxt,
        rows,
        cols,
        support,
    ) = case

    pairs = image_pairs(
        yxt.shape[2]
    )

    ref = compressed_coherence(
        yxt,
        rows,
        cols,
        support,
        pairs[
            :,
            0
        ],
        pairs[
            :,
            1
        ],
    )

    got = (
        compressed_coherence_all_pairs(
            yxt,
            rows,
            cols,
            support,
        )
    )

    np.testing.assert_allclose(
        got,
        ref,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_all_pairs_matches_general_small():

    _compare(
        _case(
            123,
            H=28,
            W=40,
            N=8,
            B=128,
            hr=2,
            hc=4,
        )
    )


def test_all_pairs_matches_general_38_dates():

    _compare(
        _case(
            456,
            H=32,
            W=56,
            N=38,
            B=96,
            hr=3,
            hc=6,
        )
    )
