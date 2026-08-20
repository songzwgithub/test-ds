import numpy as np

from pypsds.phase_linking.compression import (
    compress_real_slcs,
    compress_stage_slcs,
)


def test_compress_real_slcs_direct_formula():

    rng = np.random.default_rng(
        1234
    )

    z = (
        rng.normal(
            size=(50, 10)
        )
        +
        1j
        *
        rng.normal(
            size=(50, 10)
        )
    ).astype(
        np.complex64
    )

    p = np.exp(
        1j
        *
        rng.uniform(
            -np.pi,
            np.pi,
            size=(50, 10),
        )
    ).astype(
        np.complex64
    )

    got = compress_real_slcs(
        z,
        p,
        reference_idx=3,
    )

    pref = (
        p
        *
        np.conj(
            p[
                :,
                3
            ][
                :,
                None
            ]
        )
    )

    projected = np.sum(
        z
        *
        np.conj(
            pref
        ),
        axis=1,
    )

    amp = np.mean(
        np.abs(
            z
        ),
        axis=1,
    )

    expected = (
        amp
        *
        np.exp(
            1j
            *
            np.angle(
                projected
            )
        )
    ).astype(
        np.complex64
    )

    np.testing.assert_allclose(
        got,
        expected,
        rtol=0,
        atol=2e-6,
    )


def test_stage_reference_before_slice():

    rng = np.random.default_rng(
        9
    )

    # 2 previous compressed + 8 real.
    z = (
        rng.normal(
            size=(100, 10)
        )
        +
        1j
        *
        rng.normal(
            size=(100, 10)
        )
    ).astype(
        np.complex64
    )

    phase = np.exp(
        1j
        *
        rng.uniform(
            -np.pi,
            np.pi,
            size=(100, 10),
        )
    ).astype(
        np.complex64
    )

    got = compress_stage_slcs(
        z,
        phase,
        first_real_idx=2,
        compressed_reference_idx=1,
    )

    referenced = (
        phase
        *
        np.conj(
            phase[
                :,
                1
            ][
                :,
                None
            ]
        )
    )

    real_z = z[
        :,
        2:
    ]

    real_phase = referenced[
        :,
        2:
    ]

    projected = np.sum(
        real_z
        *
        np.conj(
            real_phase
        ),
        axis=1,
    )

    amp = np.mean(
        np.abs(
            real_z
        ),
        axis=1,
    )

    expected = (
        amp
        *
        np.exp(
            1j
            *
            np.angle(
                projected
            )
        )
    ).astype(
        np.complex64
    )

    np.testing.assert_allclose(
        got,
        expected,
        rtol=0,
        atol=2e-6,
    )
