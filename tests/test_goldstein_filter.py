import numpy as np

from pypsds.filtering.goldstein import (
    goldstein_filter,
)


def test_goldstein_preserves_shape_dtype_and_zero_mask():
    rng = np.random.default_rng(
        42
    )

    phase = rng.normal(
        size=(
            48,
            64,
        )
    ).astype(
        np.float32
    )

    z = np.exp(
        1j
        *
        phase
    ).astype(
        np.complex64
    )

    z[
        3:8,
        11:17,
    ] = 0

    out = goldstein_filter(
        z,
        alpha=0.5,
        patch_size=32,
    )

    assert out.shape == z.shape
    assert out.dtype == np.complex64

    assert np.all(
        out[
            3:8,
            11:17,
        ]
        ==
        0
    )

    assert np.all(
        np.isfinite(
            out.real
        )
    )

    assert np.all(
        np.isfinite(
            out.imag
        )
    )


def test_alpha_zero_is_nearly_identity_on_valid_field():
    rng = np.random.default_rng(
        7
    )

    phase = rng.uniform(
        -np.pi,
        np.pi,
        size=(
            64,
            80,
        ),
    ).astype(
        np.float32
    )

    z = np.exp(
        1j
        *
        phase
    ).astype(
        np.complex64
    )

    out = goldstein_filter(
        z,
        alpha=0.0,
        patch_size=32,
    )

    d = np.angle(
        out
        *
        np.conj(
            z
        )
    )

    assert float(
        np.max(
            np.abs(
                d
            )
        )
    ) < 1.0e-5


def test_stronger_alpha_reduces_phase_roughness_on_noise():
    rng = np.random.default_rng(
        11
    )

    yy, xx = np.mgrid[
        0:96,
        0:96,
    ]

    signal = (
        0.015
        *
        xx
        +
        0.01
        *
        yy
    )

    noise = rng.normal(
        scale=0.9,
        size=signal.shape,
    )

    z = np.exp(
        1j
        *
        (
            signal
            +
            noise
        )
    ).astype(
        np.complex64
    )

    out = goldstein_filter(
        z,
        alpha=0.7,
        patch_size=32,
    )

    raw_phase = np.angle(
        z
    )

    filt_phase = np.angle(
        out
    )

    raw_dx = np.angle(
        np.exp(
            1j
            *
            (
                raw_phase[
                    :,
                    1:
                ]
                -
                raw_phase[
                    :,
                    :-1
                ]
            )
        )
    )

    filt_dx = np.angle(
        np.exp(
            1j
            *
            (
                filt_phase[
                    :,
                    1:
                ]
                -
                filt_phase[
                    :,
                    :-1
                ]
            )
        )
    )

    assert float(
        np.median(
            np.abs(
                filt_dx
            )
        )
    ) < float(
        np.median(
            np.abs(
                raw_dx
            )
        )
    )
