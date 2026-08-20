import numpy as np

from pypsds.phase_linking.support_cache import (
    pack_support_bool,
    popcount_support_bits,
    unpack_support_bits,
)


def test_exact_support_pack_roundtrip():

    rng = np.random.default_rng(
        20260820
    )

    x = (
        rng.random(
            (
                257,
                11,
                23,
            )
        )
        <
        0.35
    )

    # Formal center position excluded.
    x[
        :,
        5,
        11,
    ] = False

    bits = pack_support_bool(
        x
    )

    assert bits.shape == (
        257,
        4,
    )

    y = unpack_support_bits(
        bits,
        half_row=5,
        half_col=11,
    )

    np.testing.assert_array_equal(
        x,
        y,
    )

    k0 = np.sum(
        x,
        axis=(1, 2),
        dtype=np.int32,
    ).astype(
        np.int16
    )

    k1 = popcount_support_bits(
        bits
    )

    np.testing.assert_array_equal(
        k0,
        k1,
    )
