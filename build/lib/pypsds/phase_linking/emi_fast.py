from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import nullcontext

import numpy as np

try:
    from threadpoolctl import (
        threadpool_limits,
    )
except Exception:
    threadpool_limits = None

from .emi import (
    ESTIMATOR_EVD,
    ESTIMATOR_EMI,
    ESTIMATOR_INVALID,
    _reference_unit_phase,
    _take_eigvec,
    uncompress_coherence,
)


def _recursive_cholesky_inverse(
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Gamma^-1 using batched Cholesky.

    NumPy batched cholesky raises if any matrix in the
    batch is not positive definite. When this happens,
    recursively split the batch until the failing matrices
    are isolated.

    This makes the common path fast while retaining a
    deterministic EVD fallback for rejected matrices.
    """

    gamma = np.asarray(
        gamma,
        dtype=np.float64,
    )

    b, n, _ = gamma.shape

    inv = np.zeros_like(
        gamma,
        dtype=np.float64,
    )

    ok = np.zeros(
        b,
        dtype=np.bool_,
    )

    eye = np.eye(
        n,
        dtype=np.float64,
    )

    def solve_range(
        start: int,
        stop: int,
    ) -> None:

        if start >= stop:
            return

        g = gamma[
            start:stop
        ]

        try:
            L = np.linalg.cholesky(
                g
            )

            rhs = np.broadcast_to(
                eye,
                (
                    stop - start,
                    n,
                    n,
                ),
            )

            y = np.linalg.solve(
                L,
                rhs,
            )

            cur_inv = np.linalg.solve(
                np.swapaxes(
                    L,
                    -1,
                    -2,
                ),
                y,
            )

            finite = np.all(
                np.isfinite(
                    cur_inv
                ),
                axis=(1, 2),
            )

            ids = np.flatnonzero(
                finite
            )

            if ids.size:
                inv[
                    start + ids
                ] = cur_inv[
                    ids
                ]

                ok[
                    start + ids
                ] = True

        except np.linalg.LinAlgError:

            if stop - start == 1:
                return

            middle = (
                start
                +
                stop
            ) // 2

            solve_range(
                start,
                middle,
            )

            solve_range(
                middle,
                stop,
            )

    solve_range(
        0,
        b,
    )

    return (
        inv,
        ok,
    )


def robust_emi_cholesky_batch(
    coh: np.ndarray,
    *,
    n_images: int,
    pairs: np.ndarray,
    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,
    reference_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Memory-bounded fast robust EMI.

    Mathematical estimator is unchanged from the validated
    pyPSDS-GAMMA EMI implementation except for the numerical
    route used to obtain Gamma^-1:

        C         = coherence matrix
        Gamma     = |C|
        Gamma_reg = (1-beta) Gamma + beta I + jitter I
        A         = Gamma_reg^-1 ⊙ C

    EMI:
        eigenvalue of A nearest emi_mu

    fallback:
        largest eigenpair of C ⊙ |C|

    Cholesky is used instead of a full eigendecomposition
    of Gamma. Matrices for which Cholesky fails use the
    existing EVD fallback definition.
    """

    coh = np.asarray(
        coh,
        dtype=np.complex64,
    )

    C = uncompress_coherence(
        coh,
        n_images,
        pairs,
    ).astype(
        np.complex128,
        copy=False,
    )

    b = C.shape[0]

    eye = np.eye(
        n_images,
        dtype=np.float64,
    )

    Gamma = np.abs(
        C
    ).real

    if beta > 0:
        Gamma = (
            (1.0 - beta)
            *
            Gamma
            +
            beta
            *
            eye[
                None,
                :,
                :,
            ]
        )

    Gamma = (
        Gamma
        +
        gamma_jitter
        *
        eye[
            None,
            :,
            :,
        ]
    )

    Gamma = (
        0.5
        *
        (
            Gamma
            +
            np.swapaxes(
                Gamma,
                -1,
                -2,
            )
        )
    )

    (
        Gamma_inv,
        gamma_ok,
    ) = _recursive_cholesky_inverse(
        Gamma
    )

    phase = np.full(
        (
            b,
            n_images,
        ),
        np.nan
        +
        1j * np.nan,
        dtype=np.complex64,
    )

    estimator = np.full(
        b,
        ESTIMATOR_INVALID,
        dtype=np.uint8,
    )

    emi_ok = np.zeros(
        b,
        dtype=np.bool_,
    )

    # --------------------------------------------------------
    # EMI only for Cholesky-valid matrices.
    # --------------------------------------------------------

    ids = np.flatnonzero(
        gamma_ok
    )

    if ids.size:

        A = (
            Gamma_inv[
                ids
            ]
            *
            C[
                ids
            ]
        )

        A = (
            0.5
            *
            (
                A
                +
                np.swapaxes(
                    A.conj(),
                    -1,
                    -2,
                )
            )
        )

        ew, ev = np.linalg.eigh(
            A
        )

        emi_idx = np.argmin(
            np.abs(
                ew.real
                -
                emi_mu
            ),
            axis=1,
        )

        emi_vec = _take_eigvec(
            ev,
            emi_idx,
        )

        emi_val = ew[
            np.arange(
                ids.size
            ),
            emi_idx,
        ].real

        finite = (
            np.isfinite(
                emi_val
            )
            &
            np.all(
                np.isfinite(
                    emi_vec.real
                )
                &
                np.isfinite(
                    emi_vec.imag
                ),
                axis=1,
            )
        )

        good_ids = ids[
            finite
        ]

        if good_ids.size:

            v = emi_vec[
                finite
            ]

            norm = np.linalg.norm(
                v,
                axis=1,
                keepdims=True,
            )

            norm = np.where(
                norm > 0,
                norm,
                1.0,
            )

            # Keep the validated Dolphin/pyPSDS
            # normalization convention.
            v = (
                np.sqrt(
                    n_images
                )
                *
                v
                /
                norm
            )

            phase[
                good_ids
            ] = _reference_unit_phase(
                v,
                reference_idx,
            )

            estimator[
                good_ids
            ] = ESTIMATOR_EMI

            emi_ok[
                good_ids
            ] = True

    # --------------------------------------------------------
    # Exact existing lazy-EVD fallback definition.
    # --------------------------------------------------------

    bad = ~emi_ok

    if np.any(
        bad
    ):

        Cb = C[
            bad
        ]

        E = (
            Cb
            *
            np.abs(
                Cb
            )
        )

        E = (
            0.5
            *
            (
                E
                +
                np.swapaxes(
                    E.conj(),
                    -1,
                    -2,
                )
            )
        )

        bw, bv = np.linalg.eigh(
            E
        )

        evd_idx = np.argmax(
            bw.real,
            axis=1,
        )

        evd_vec = _take_eigvec(
            bv,
            evd_idx,
        )

        evd_val = bw[
            np.arange(
                bw.shape[0]
            ),
            evd_idx,
        ].real

        evd_ok = (
            np.isfinite(
                evd_val
            )
            &
            np.all(
                np.isfinite(
                    evd_vec.real
                )
                &
                np.isfinite(
                    evd_vec.imag
                ),
                axis=1,
            )
        )

        bad_ids = np.flatnonzero(
            bad
        )

        good_bad_ids = bad_ids[
            evd_ok
        ]

        if good_bad_ids.size:

            phase[
                good_bad_ids
            ] = _reference_unit_phase(
                evd_vec[
                    evd_ok
                ],
                reference_idx,
            )

            estimator[
                good_bad_ids
            ] = ESTIMATOR_EVD

    return (
        phase,
        estimator,
    )


def robust_emi_cholesky_threaded(
    coh: np.ndarray,
    *,
    n_images: int,
    pairs: np.ndarray,
    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,
    reference_idx: int = 0,
    workers: int = 16,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Threaded fast-Cholesky EMI.

    BLAS/LAPACK threads must remain at one.
    Parallelism is across independent DS point chunks.
    """

    coh = np.asarray(
        coh,
        dtype=np.complex64,
    )

    b = coh.shape[0]

    if b == 0:
        return (
            np.empty(
                (
                    0,
                    n_images,
                ),
                dtype=np.complex64,
            ),
            np.empty(
                0,
                dtype=np.uint8,
            ),
        )

    workers = max(
        1,
        int(
            workers
        ),
    )

    chunk_size = max(
        1,
        int(
            chunk_size
        ),
    )

    if (
        workers == 1
        or
        b <= chunk_size
    ):
        return robust_emi_cholesky_batch(
            coh,
            n_images=n_images,
            pairs=pairs,
            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,
            reference_idx=reference_idx,
        )

    phase = np.empty(
        (
            b,
            n_images,
        ),
        dtype=np.complex64,
    )

    estimator = np.empty(
        b,
        dtype=np.uint8,
    )

    ranges = [
        (
            start,
            min(
                b,
                start
                +
                chunk_size,
            ),
        )
        for start in range(
            0,
            b,
            chunk_size,
        )
    ]

    def work(
        start: int,
        stop: int,
    ):
        result = (
            robust_emi_cholesky_batch(
                coh[
                    start:stop
                ],
                n_images=n_images,
                pairs=pairs,
                beta=beta,
                gamma_jitter=gamma_jitter,
                emi_mu=emi_mu,
                reference_idx=reference_idx,
            )
        )

        return (
            start,
            stop,
            result,
        )

    ctx = (
        threadpool_limits(
            limits=1
        )
        if threadpool_limits
        is not None
        else nullcontext()
    )

    with ctx:

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=(
                "pypsds-pl-cholesky"
            ),
        ) as ex:

            futures = [
                ex.submit(
                    work,
                    start,
                    stop,
                )
                for start, stop
                in ranges
            ]

            for fut in as_completed(
                futures
            ):

                (
                    start,
                    stop,
                    result,
                ) = fut.result()

                ph, est = result

                phase[
                    start:stop
                ] = ph

                estimator[
                    start:stop
                ] = est

    return (
        phase,
        estimator,
    )


__all__ = [
    "robust_emi_cholesky_batch",
    "robust_emi_cholesky_threaded",
]
