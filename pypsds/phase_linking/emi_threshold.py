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
    robust_emi_batch,
    uncompress_coherence,
)


def _threshold_cholesky_inverse(
    gamma: np.ndarray,
    *,
    min_gamma_eig: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast production Gamma acceptance + inverse.

    The validated current implementation accepts EMI only when

        lambda_min(Gamma) > min_gamma_eig.

    For Hermitian Gamma,

        Gamma - tau*I positive definite

    is mathematically equivalent to

        lambda_min(Gamma) > tau.

    Therefore Cholesky(Gamma - tau I) implements the same
    mathematical threshold without computing Gamma eigenvectors.

    Only matrices which pass that gate are inverted here.

    Failed/borderline matrices are NOT classified as EVD here.
    The caller sends them through the original validated
    robust_emi_batch() implementation.
    """

    gamma = np.asarray(
        gamma,
        dtype=np.float64,
    )

    B, N, _ = gamma.shape

    inv = np.zeros_like(
        gamma,
        dtype=np.float64,
    )

    ok = np.zeros(
        B,
        dtype=np.bool_,
    )

    eye = np.eye(
        N,
        dtype=np.float64,
    )

    shifted = (
        gamma
        -
        float(min_gamma_eig)
        *
        eye[
            None,
            :,
            :,
        ]
    )

    def solve_range(
        start: int,
        stop: int,
    ) -> None:

        if start >= stop:
            return

        # ----------------------------------------------------
        # Exact mathematical threshold gate.
        #
        # One bad matrix makes NumPy's batched Cholesky fail,
        # therefore recursively isolate failing matrices.
        # ----------------------------------------------------

        try:

            np.linalg.cholesky(
                shifted[
                    start:stop
                ]
            )

        except np.linalg.LinAlgError:

            if stop - start == 1:
                return

            mid = (
                start
                +
                stop
            ) // 2

            solve_range(
                start,
                mid,
            )

            solve_range(
                mid,
                stop,
            )

            return

        # ----------------------------------------------------
        # All matrices in this sub-range satisfy the threshold.
        #
        # Invert ORIGINAL Gamma, not shifted Gamma.
        # ----------------------------------------------------

        try:

            G = gamma[
                start:stop
            ]

            L = np.linalg.cholesky(
                G
            )

            rhs = np.broadcast_to(
                eye,
                (
                    stop - start,
                    N,
                    N,
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

            # Extremely conservative handling:
            # fall back to smaller ranges, and ultimately
            # to the current exact implementation in caller.

            if stop - start == 1:
                return

            mid = (
                start
                +
                stop
            ) // 2

            solve_range(
                start,
                mid,
            )

            solve_range(
                mid,
                stop,
            )

    solve_range(
        0,
        B,
    )

    return (
        inv,
        ok,
    )


def robust_emi_threshold_batch(
    coh: np.ndarray,
    *,
    n_images: int,
    pairs: np.ndarray,
    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,
    reference_idx: int = 0,
    min_gamma_eig: float = 1e-7,
):
    """
    Production-fast robust EMI.

    Fast path changes ONLY how Gamma^-1 is obtained.

    Preserved:
      C
      Gamma = |C|
      beta regularization
      gamma jitter
      min_gamma_eig criterion
      A = Gamma^-1 ⊙ C
      EMI eigenvalue nearest mu
      reference normalization
      lazy EVD fallback

    Any matrix not confidently solved by the threshold-Cholesky
    route is evaluated by the existing robust_emi_batch().
    """

    coh = np.asarray(
        coh,
        dtype=np.complex64,
    )

    B = coh.shape[0]

    if B == 0:

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

            np.empty(
                0,
                dtype=np.float32,
            ),

            np.empty(
                0,
                dtype=np.float32,
            ),

            np.empty(
                0,
                dtype=np.float32,
            ),
        )


    C = uncompress_coherence(
        coh,
        n_images,
        pairs,
    ).astype(
        np.complex128,
        copy=False,
    )


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
        threshold_ok,
    ) = _threshold_cholesky_inverse(
        Gamma,
        min_gamma_eig=(
            min_gamma_eig
        ),
    )


    phase = np.full(
        (
            B,
            n_images,
        ),
        np.nan
        +
        1j * np.nan,
        dtype=np.complex64,
    )


    estimator = np.full(
        B,
        ESTIMATOR_INVALID,
        dtype=np.uint8,
    )


    emi_eigenvalue = np.full(
        B,
        np.nan,
        dtype=np.float32,
    )


    evd_eigenvalue = np.full(
        B,
        np.nan,
        dtype=np.float32,
    )


    # Gamma minimum eigenvalue is intentionally unknown on the
    # Cholesky fast path. Sequential production does not expose
    # this diagnostic. Slow-path values remain exact.
    gamma_min = np.full(
        B,
        np.nan,
        dtype=np.float32,
    )


    # ====================================================================
    # Fast threshold-approved EMI
    # ====================================================================

    fast_ids = np.flatnonzero(
        threshold_ok
    )


    fast_good = np.zeros(
        B,
        dtype=np.bool_,
    )


    if fast_ids.size:

        A = (
            Gamma_inv[
                fast_ids
            ]
            *
            C[
                fast_ids
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
                fast_ids.size
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


        good_ids = fast_ids[
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


            emi_eigenvalue[
                good_ids
            ] = emi_val[
                finite
            ].astype(
                np.float32
            )


            fast_good[
                good_ids
            ] = True


    # ====================================================================
    # Conservative exact slow path
    #
    # Includes:
    #   threshold failures
    #   Cholesky numerical failures
    #   non-finite fast EMI outputs
    #
    # These points use the ORIGINAL validated implementation.
    # ====================================================================

    slow = (
        ~fast_good
    )


    if np.any(
        slow
    ):

        (
            slow_phase,
            slow_est,
            slow_emi,
            slow_evd,
            slow_gamma,
        ) = robust_emi_batch(
            coh[
                slow
            ],

            n_images=n_images,
            pairs=pairs,

            beta=beta,

            gamma_jitter=(
                gamma_jitter
            ),

            emi_mu=emi_mu,

            reference_idx=(
                reference_idx
            ),

            min_gamma_eig=(
                min_gamma_eig
            ),
        )


        phase[
            slow
        ] = slow_phase


        estimator[
            slow
        ] = slow_est


        emi_eigenvalue[
            slow
        ] = slow_emi


        evd_eigenvalue[
            slow
        ] = slow_evd


        gamma_min[
            slow
        ] = slow_gamma


    return (
        phase,
        estimator,
        emi_eigenvalue,
        evd_eigenvalue,
        gamma_min,
    )


def robust_emi_threshold_threaded(
    coh: np.ndarray,
    *,
    n_images: int,
    pairs: np.ndarray,
    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,
    reference_idx: int = 0,
    min_gamma_eig: float = 1e-7,
    workers: int = 8,
    chunk_size: int = 1024,
):
    """
    Threaded production threshold-Cholesky EMI.

    BLAS/LAPACK remains one thread.
    Parallelism is across independent DS point chunks.
    """

    coh = np.asarray(
        coh,
        dtype=np.complex64,
    )


    B = coh.shape[0]


    if B == 0:

        return robust_emi_threshold_batch(
            coh,

            n_images=n_images,
            pairs=pairs,

            beta=beta,

            gamma_jitter=(
                gamma_jitter
            ),

            emi_mu=emi_mu,

            reference_idx=(
                reference_idx
            ),

            min_gamma_eig=(
                min_gamma_eig
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
        B <= chunk_size
    ):

        return robust_emi_threshold_batch(
            coh,

            n_images=n_images,
            pairs=pairs,

            beta=beta,

            gamma_jitter=(
                gamma_jitter
            ),

            emi_mu=emi_mu,

            reference_idx=(
                reference_idx
            ),

            min_gamma_eig=(
                min_gamma_eig
            ),
        )


    phase = np.empty(
        (
            B,
            n_images,
        ),
        dtype=np.complex64,
    )


    estimator = np.empty(
        B,
        dtype=np.uint8,
    )


    emi_eig = np.empty(
        B,
        dtype=np.float32,
    )


    evd_eig = np.empty(
        B,
        dtype=np.float32,
    )


    gamma_min = np.empty(
        B,
        dtype=np.float32,
    )


    ranges = [
        (
            start,

            min(
                B,
                start
                +
                chunk_size,
            ),
        )

        for start
        in range(
            0,
            B,
            chunk_size,
        )
    ]


    def work(
        start: int,
        stop: int,
    ):

        result = (
            robust_emi_threshold_batch(
                coh[
                    start:stop
                ],

                n_images=n_images,
                pairs=pairs,

                beta=beta,

                gamma_jitter=(
                    gamma_jitter
                ),

                emi_mu=emi_mu,

                reference_idx=(
                    reference_idx
                ),

                min_gamma_eig=(
                    min_gamma_eig
                ),
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

        if
        threadpool_limits
        is not None

        else
        nullcontext()
    )


    with ctx:

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=(
                "pypsds-pl-threshold-chol"
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


                (
                    ph,
                    est,
                    ee,
                    ve,
                    gm,
                ) = result


                phase[
                    start:stop
                ] = ph


                estimator[
                    start:stop
                ] = est


                emi_eig[
                    start:stop
                ] = ee


                evd_eig[
                    start:stop
                ] = ve


                gamma_min[
                    start:stop
                ] = gm


    return (
        phase,
        estimator,
        emi_eig,
        evd_eig,
        gamma_min,
    )


__all__ = [
    "robust_emi_threshold_batch",
    "robust_emi_threshold_threaded",
]
