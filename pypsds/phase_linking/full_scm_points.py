from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..progress import ProgressReporter

from .coherence import (
    compressed_coherence,
)
from .emi import (
    ESTIMATOR_INVALID,
    image_pairs,
    robust_emi_threaded,
)
from .shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)
from .streaming_quality import (
    temporal_quality_streaming,
)


@dataclass(slots=True)
class FullSCMPointResult:
    rows: np.ndarray
    cols: np.ndarray

    shp_count: np.ndarray

    phase: np.ndarray

    temporal_coherence: np.ndarray
    median_pair_coherence: np.ndarray

    estimator: np.ndarray

    emi_eigenvalue: np.ndarray
    evd_eigenvalue: np.ndarray
    gamma_min_eigenvalue: np.ndarray

    pl_valid: np.ndarray

    @property
    def size(self) -> int:
        return int(
            self.rows.size
        )

    @property
    def valid_count(self) -> int:
        return int(
            np.count_nonzero(
                self.pl_valid
            )
        )


def run_full_scm_points(
    *,
    yxt,

    rows,
    cols,

    scale2,
    valid,
    ps,

    expected_original_k=None,

    phase_sink=None,

    half_row: int = 5,
    half_col: int = 11,
    alpha: float = 0.005,

    min_shp: int = 48,

    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,

    batch: int = 2048,
    support_block: int = 1024,

    pl_workers: int = 16,
    pl_chunk_size: int = 512,
) -> FullSCMPointResult:
    """
    Full-stack SCM phase linking for an explicit sparse set
    of formal DS centers.

    Intended production use:
        sequential fallback centers.

    Important
    ---------
    Support is the ORIGINAL full GLRT support.

    No K-state / compressed-state mask is applied here.
    """

    yxt = np.asarray(
        yxt
    )

    if yxt.ndim != 3:
        raise ValueError(
            "yxt must have shape [H,W,N]"
        )

    H, W, ndate = yxt.shape

    rr = np.asarray(
        rows,
        dtype=np.int32,
    )

    cc = np.asarray(
        cols,
        dtype=np.int32,
    )

    if (
        rr.ndim != 1
        or
        cc.ndim != 1
        or
        rr.size != cc.size
    ):
        raise ValueError(
            "rows/cols must be equal-length 1-D arrays"
        )

    if rr.size:
        if (
            np.any(rr < 0)
            or
            np.any(rr >= H)
            or
            np.any(cc < 0)
            or
            np.any(cc >= W)
        ):
            raise ValueError(
                "rows/cols outside scene"
            )

    scale2 = np.asarray(
        scale2
    )

    valid = np.asarray(
        valid,
        dtype=np.bool_,
    )

    ps = np.asarray(
        ps,
        dtype=np.bool_,
    )

    for name, arr in (
        ("scale2", scale2),
        ("valid", valid),
        ("ps", ps),
    ):
        if arr.shape != (
            H,
            W,
        ):
            raise ValueError(
                f"{name} shape={arr.shape}, "
                f"expected={(H, W)}"
            )

    if expected_original_k is not None:

        expected_original_k = np.asarray(
            expected_original_k
        )

        if expected_original_k.shape != (
            H,
            W,
        ):
            raise ValueError(
                "expected_original_k shape mismatch"
            )

    if phase_sink is not None and not callable(
        phase_sink
    ):
        raise TypeError(
            "phase_sink must be callable"
        )

    if min_shp < 1:
        raise ValueError(
            "min_shp must be >= 1"
        )

    batch = max(
        1,
        int(batch),
    )

    support_block = max(
        1,
        int(support_block),
    )

    ctx = (
        prepare_glrt_window_context(
            scale2,
            valid,
            ps,
            half_row=half_row,
            half_col=half_col,
        )
    )

    pairs = image_pairs(
        ndate
    )

    pi = np.asarray(
        pairs[:, 0],
        dtype=np.int32,
    )

    pj = np.asarray(
        pairs[:, 1],
        dtype=np.int32,
    )

    n = rr.size

    progress = ProgressReporter(
        label="full-SCM-fallback",
        total=n,
        unit="center",
        min_interval=5.0,
    )

    shp_count = np.full(
        n,
        -1,
        dtype=np.int16,
    )

    phase = np.full(
        (
            n,
            ndate,
        ),
        np.complex64(
            np.nan
            +
            1j * np.nan
        ),
        dtype=np.complex64,
    )

    tc = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    pair_coh = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    estimator = np.full(
        n,
        ESTIMATOR_INVALID,
        dtype=np.uint8,
    )

    emi_eig = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    evd_eig = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    gamma_min = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    pl_valid = np.zeros(
        n,
        dtype=np.bool_,
    )

    for b0 in range(
        0,
        n,
        batch,
    ):

        b1 = min(
            n,
            b0 + batch,
        )

        br = rr[
            b0:b1
        ]

        bc = cc[
            b0:b1
        ]

        # ----------------------------------------------------
        # ORIGINAL full GLRT support.
        #
        # Deliberately no K24/state-core intersection.
        # ----------------------------------------------------

        support, _ = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=alpha,
                nslc=ndate,
                block_size=support_block,
            )
        )

        K = np.sum(
            support,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

        shp_count[
            b0:b1
        ] = K

        if expected_original_k is not None:

            expected = np.asarray(
                expected_original_k[
                    br,
                    bc,
                ],
                dtype=np.int16,
            )

            mismatch = (
                K != expected
            )

            if np.any(
                mismatch
            ):

                k = int(
                    np.flatnonzero(
                        mismatch
                    )[0]
                )

                raise RuntimeError(
                    "original-K parity failure at "
                    f"({int(br[k])},{int(bc[k])}): "
                    f"computed={int(K[k])}, "
                    f"expected={int(expected[k])}"
                )

        below = (
            K < min_shp
        )

        if np.any(
            below
        ):

            k = int(
                np.flatnonzero(
                    below
                )[0]
            )

            raise RuntimeError(
                "fallback input is not a formal DS center at "
                f"({int(br[k])},{int(bc[k])}): "
                f"K={int(K[k])} < {min_shp}"
            )

        # ----------------------------------------------------
        # Full N-date covariance.
        # ----------------------------------------------------

        coh = compressed_coherence(
            yxt,
            br,
            bc,
            support,
            pi,
            pj,
        )

        # ----------------------------------------------------
        # Full-SCM EMI / EVD fallback.
        # ----------------------------------------------------

        (
            ph,
            est,
            ee,
            ve,
            gm,
        ) = robust_emi_threaded(
            coh,
            n_images=ndate,
            pairs=pairs,

            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,

            reference_idx=0,

            workers=pl_workers,
            chunk_size=pl_chunk_size,
        )

        tc_b, pair_b = (
            temporal_quality_streaming(
                coh,
                ph,
                pi,
                pj,
            )
        )

        ph = np.asarray(
            ph,
            dtype=np.complex64,
        )

        tc_b = np.asarray(
            tc_b,
            dtype=np.float32,
        )

        pair_b = np.asarray(
            pair_b,
            dtype=np.float32,
        )

        phase_complete = np.all(
            np.isfinite(
                ph.real
            )
            &
            np.isfinite(
                ph.imag
            ),
            axis=1,
        )

        ok = (
            (est != ESTIMATOR_INVALID)
            &
            np.isfinite(
                tc_b
            )
            &
            phase_complete
        )

        phase[
            b0:b1
        ] = ph

        tc[
            b0:b1
        ] = tc_b

        pair_coh[
            b0:b1
        ] = pair_b

        estimator[
            b0:b1
        ] = est

        emi_eig[
            b0:b1
        ] = ee

        evd_eig[
            b0:b1
        ] = ve

        gamma_min[
            b0:b1
        ] = gm

        pl_valid[
            b0:b1
        ] = ok

        # ----------------------------------------------------
        # Optional production linked_phase writer.
        #
        # Only valid PL outputs are written, matching the
        # existing Step04 linked_phase semantics.
        # ----------------------------------------------------

        progress.update(
            b1,
            detail=(
                f"valid={int(np.count_nonzero(pl_valid[:b1])):,}"
            ),
        )

        if (
            phase_sink is not None
            and
            np.any(ok)
        ):

            phase_sink(
                stage_index=-1,

                real_indices=tuple(
                    range(
                        ndate
                    )
                ),

                rows=br[
                    ok
                ],

                cols=bc[
                    ok
                ],

                phase=ph[
                    ok
                ],
            )

    progress.finish(
        n,
        detail=(
            f"valid={int(np.count_nonzero(pl_valid)):,}"
        ),
    )

    if (
        phase_sink is not None
        and
        hasattr(
            phase_sink,
            "flush",
        )
    ):
        phase_sink.flush()

    return FullSCMPointResult(
        rows=rr,
        cols=cc,

        shp_count=shp_count,

        phase=phase,

        temporal_coherence=tc,

        median_pair_coherence=(
            pair_coh
        ),

        estimator=estimator,

        emi_eigenvalue=(
            emi_eig
        ),

        evd_eigenvalue=(
            evd_eig
        ),

        gamma_min_eigenvalue=(
            gamma_min
        ),

        pl_valid=pl_valid,
    )


__all__ = [
    "FullSCMPointResult",
    "run_full_scm_points",
]
