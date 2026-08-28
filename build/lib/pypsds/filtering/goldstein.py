from __future__ import annotations

import numpy as np


def _overlap_weight(
    patch_size: int,
) -> np.ndarray:
    """
    Symmetric overlap-add weight used by the Dolphin Goldstein implementation.
    """
    p = int(
        patch_size
    )

    if (
        p < 4
        or
        p % 2
    ):
        raise ValueError(
            "patch_size must be an even integer >= 4"
        )

    half = p // 2

    if half == 1:
        raise ValueError(
            "patch_size is too small"
        )

    w = (
        1.0
        -
        np.abs(
            np.arange(
                half,
                dtype=np.float64,
            )
            -
            (
                p / 2.0
                -
                1.0
            )
        )
        /
        (
            p / 2.0
            -
            1.0
        )
    )

    q = np.outer(
        w,
        w,
    )

    return np.block(
        [
            [
                q,
                np.flip(
                    q,
                    axis=1,
                ),
            ],
            [
                np.flip(
                    q,
                    axis=0,
                ),
                np.flip(
                    np.flip(
                        q,
                        axis=0,
                    ),
                    axis=1,
                ),
            ],
        ]
    )


def goldstein_filter(
    data: np.ndarray,
    *,
    alpha: float,
    patch_size: int = 32,
) -> np.ndarray:
    """
    Goldstein-Werner adaptive interferogram filter.

    Parameters
    ----------
    data
        2-D complex wrapped interferogram.  Zero magnitude marks pixels that
        are absent from filtering support.
    alpha
        Spectral exponent in [0, 1].
    patch_size
        Even FFT patch size. Dolphin currently uses 32 by default.

    Notes
    -----
    This implementation follows the same overlapping-patch spectral weighting
    convention as Dolphin:
        FFT -> |FFT|**alpha * FFT -> inverse FFT -> overlap/add.

    Invalid/non-finite input is converted to zero support before FFT and is
    restored to zero in the output.  This is required because pyPSDS has an
    explicit sparse phase-validity domain.
    """
    a = float(
        alpha
    )

    if not (
        0.0
        <=
        a
        <=
        1.0
    ):
        raise ValueError(
            "alpha must be within [0, 1]"
        )

    p = int(
        patch_size
    )

    if (
        p < 4
        or
        p % 2
    ):
        raise ValueError(
            "patch_size must be an even integer >= 4"
        )

    x = np.asarray(
        data,
    )

    if x.ndim != 2:
        raise ValueError(
            "data must be 2-D"
        )

    if np.iscomplexobj(
        x
    ):
        z = np.asarray(
            x,
            dtype=np.complex64,
        ).copy()
    else:
        phase = np.asarray(
            x,
            dtype=np.float32,
        )

        z = np.exp(
            1j
            *
            phase
        ).astype(
            np.complex64,
        )

    empty = (
        ~np.isfinite(
            z.real
        )
        |
        ~np.isfinite(
            z.imag
        )
        |
        (
            np.abs(
                z
            )
            ==
            0
        )
    )

    z[
        empty
    ] = np.complex64(
        0.0
        +
        0.0j
    )

    if np.all(
        empty
    ):
        return z

    nrows, ncols = z.shape
    step = p // 2

    pad_top = step
    pad_left = step

    pad_bottom = (
        step
        +
        (
            step
            -
            (
                nrows
                %
                step
            )
        )
        %
        step
    )

    pad_right = (
        step
        +
        (
            step
            -
            (
                ncols
                %
                step
            )
        )
        %
        step
    )

    zp = np.pad(
        z,
        (
            (
                pad_top,
                pad_bottom,
            ),
            (
                pad_left,
                pad_right,
            ),
        ),
        mode="reflect",
    )

    out = np.zeros(
        zp.shape,
        dtype=np.complex64,
    )

    weight_sum = np.zeros(
        zp.shape,
        dtype=np.float64,
    )

    ow = _overlap_weight(
        p
    )

    nr, nc = zp.shape

    for r0 in range(
        0,
        nr - p + 1,
        step,
    ):
        for c0 in range(
            0,
            nc - p + 1,
            step,
        ):
            block = zp[
                r0:
                r0 + p,
                c0:
                c0 + p,
            ]

            spec = np.fft.fft2(
                block,
                s=(
                    p,
                    p,
                ),
            )

            spectral_weight = np.power(
                np.abs(
                    spec
                ),
                a,
            )

            filtered = np.fft.ifft2(
                spec
                *
                spectral_weight,
                s=(
                    p,
                    p,
                ),
            )

            out[
                r0:
                r0 + p,
                c0:
                c0 + p,
            ] += (
                ow
                *
                filtered
            ).astype(
                np.complex64,
                copy=False,
            )

            weight_sum[
                r0:
                r0 + p,
                c0:
                c0 + p,
            ] += ow

    valid = (
        weight_sum
        >
        0
    )

    out[
        valid
    ] /= weight_sum[
        valid
    ]

    out = out[
        pad_top:
        pad_top + nrows,
        pad_left:
        pad_left + ncols,
    ]

    out[
        empty
    ] = np.complex64(
        0.0
        +
        0.0j
    )

    return np.asarray(
        out,
        dtype=np.complex64,
    )


__all__ = [
    "goldstein_filter",
]
