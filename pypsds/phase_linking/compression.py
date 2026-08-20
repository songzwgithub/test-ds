from __future__ import annotations

import numpy as np


def compress_real_slcs(
    real_slcs: np.ndarray,
    linked_phase: np.ndarray,
    *,
    reference_idx: int = 0,
    mean_amplitude: np.ndarray | None = None,
) -> np.ndarray:
    """
    Create one compressed complex SLC from a real-SLC ministack.

    Parameters
    ----------
    real_slcs
        Complex SLC samples. Last axis is acquisition:
            (..., N)

    linked_phase
        Unit-magnitude phase estimates with the same shape:
            (..., N)

    reference_idx
        Phase estimate is first re-referenced to this acquisition.

    mean_amplitude
        Optional output magnitude (...).
        If omitted:
            mean(abs(real_slcs), axis=-1)

    Mathematical definition
    -----------------------
    phi_ref[t] =
        phi[t] * conj(phi[reference_idx])

    projected =
        sum_t SLC[t] * conj(phi_ref[t])

    compressed =
        mean_amplitude * exp(i * angle(projected))

    This is an independent pyPSDS implementation of the
    compressed-SLC projection used in sequential PL.
    """

    z = np.asarray(
        real_slcs,
        dtype=np.complex64,
    )

    ph = np.asarray(
        linked_phase,
        dtype=np.complex64,
    )

    if z.shape != ph.shape:
        raise ValueError(
            f"real_slcs shape {z.shape} != "
            f"linked_phase shape {ph.shape}"
        )

    if z.ndim < 1:
        raise ValueError(
            "Input must have an acquisition axis"
        )

    n = z.shape[-1]

    if not (
        0 <= reference_idx < n
    ):
        raise ValueError(
            "reference_idx outside ministack"
        )

    # --------------------------------------------------------
    # Re-reference first.
    # --------------------------------------------------------

    ph_ref = (
        ph
        *
        np.conj(
            ph[
                ...,
                reference_idx
            ][
                ...,
                None
            ]
        )
    ).astype(
        np.complex64,
        copy=False,
    )

    # --------------------------------------------------------
    # Pixel-wise complex projection.
    # --------------------------------------------------------

    with np.errstate(
        invalid="ignore",
    ):

        projected = np.nansum(
            z
            *
            np.conj(
                ph_ref
            ),
            axis=-1,
        )

    phase = np.angle(
        projected
    )

    if mean_amplitude is None:

        with np.errstate(
            invalid="ignore",
        ):

            amp = np.nanmean(
                np.abs(
                    z
                ),
                axis=-1,
                dtype=np.float64,
            ).astype(
                np.float32
            )

    else:

        amp = np.asarray(
            mean_amplitude,
            dtype=np.float32,
        )

        if (
            amp.shape
            !=
            z.shape[:-1]
        ):

            raise ValueError(
                f"mean_amplitude shape "
                f"{amp.shape} != "
                f"{z.shape[:-1]}"
            )

    out = (
        amp
        *
        np.exp(
            1j
            *
            phase
        )
    ).astype(
        np.complex64
    )

    # Preserve invalid projected pixels.
    invalid = (
        ~np.isfinite(
            projected.real
        )
        |
        ~np.isfinite(
            projected.imag
        )
        |
        ~np.isfinite(
            amp
        )
        |
        (
            np.abs(
                projected
            )
            ==
            0
        )
    )

    out[
        invalid
    ] = np.complex64(
        np.nan
        +
        1j
        *
        np.nan
    )

    return out


def compress_stage_slcs(
    stage_stack: np.ndarray,
    stage_phase: np.ndarray,
    *,
    first_real_idx: int,
    compressed_reference_idx: int,
    mean_amplitude: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sequential-stage wrapper.

    `stage_stack` may begin with previous compressed SLCs.

    Phase referencing is performed using the complete stage phase
    vector BEFORE previous compressed layers are removed.

    Only real SLCs are then used in the compression projection.
    """

    z = np.asarray(
        stage_stack,
        dtype=np.complex64,
    )

    ph = np.asarray(
        stage_phase,
        dtype=np.complex64,
    )

    if z.shape != ph.shape:
        raise ValueError(
            "stage_stack / stage_phase shape mismatch"
        )

    n = z.shape[-1]

    if not (
        0 <= first_real_idx < n
    ):
        raise ValueError(
            "first_real_idx outside stage"
        )

    if not (
        0
        <=
        compressed_reference_idx
        <
        n
    ):
        raise ValueError(
            "compressed_reference_idx outside stage"
        )

    # Re-reference using the complete ministack phase vector.
    ref = ph[
        ...,
        compressed_reference_idx
    ][
        ...,
        None
    ]

    ph_ref = (
        ph
        *
        np.conj(
            ref
        )
    ).astype(
        np.complex64,
        copy=False,
    )

    # Then discard previous compressed layers.
    real_z = z[
        ...,
        first_real_idx:
    ]

    real_ph = ph_ref[
        ...,
        first_real_idx:
    ]

    with np.errstate(
        invalid="ignore",
    ):

        projected = np.nansum(
            real_z
            *
            np.conj(
                real_ph
            ),
            axis=-1,
        )

    phase = np.angle(
        projected
    )

    if mean_amplitude is None:

        with np.errstate(
            invalid="ignore",
        ):

            amp = np.nanmean(
                np.abs(
                    real_z
                ),
                axis=-1,
                dtype=np.float64,
            ).astype(
                np.float32
            )

    else:

        amp = np.asarray(
            mean_amplitude,
            dtype=np.float32,
        )

    out = (
        amp
        *
        np.exp(
            1j
            *
            phase
        )
    ).astype(
        np.complex64
    )

    invalid = (
        ~np.isfinite(
            projected.real
        )
        |
        ~np.isfinite(
            projected.imag
        )
        |
        ~np.isfinite(
            amp
        )
        |
        (
            np.abs(
                projected
            )
            ==
            0
        )
    )

    out[
        invalid
    ] = np.complex64(
        np.nan
        +
        1j
        *
        np.nan
    )

    return out


__all__ = [
    "compress_real_slcs",
    "compress_stage_slcs",
]
