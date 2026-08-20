from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..progress import ProgressReporter

from .coherence import (
    compressed_coherence,
)
from .emi import (
    ESTIMATOR_EVD,
    ESTIMATOR_EMI,
    ESTIMATOR_INVALID,
    image_pairs,
)
from .shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)
from .state_domain import (
    make_windows,
)
from .streaming_quality import (
    temporal_quality_streaming,
)


@dataclass(slots=True)
class FullspanQualityResult:
    rows: np.ndarray
    cols: np.ndarray

    effective_k: np.ndarray

    temporal_coherence: np.ndarray
    median_pair_coherence: np.ndarray

    phase_complete: np.ndarray

    @property
    def size(self) -> int:
        return int(
            self.rows.size
        )


def aggregate_stage_estimators(
    stage_estimators,
    *,
    rows,
    cols,
):
    """
    Collapse per-stage sequential estimator status into one
    production estimator code.

    Semantics
    ---------
    1   : every required stage used EMI
    0   : no stage invalid, but >=1 stage used EVD fallback
    255 : >=1 required stage invalid

    This preserves the existing production estimator coding:
        EVD=0, EMI=1, INVALID=255.
    """

    rr = np.asarray(
        rows,
        dtype=np.int32,
    )

    cc = np.asarray(
        cols,
        dtype=np.int32,
    )

    maps = tuple(
        np.asarray(x)
        for x
        in stage_estimators
    )

    if not maps:
        raise ValueError(
            "no stage estimator maps"
        )

    shape = maps[0].shape

    for arr in maps:
        if arr.shape != shape:
            raise ValueError(
                "stage estimator shape mismatch"
            )

    vals = np.stack(
        [
            arr[
                rr,
                cc,
            ]
            for arr
            in maps
        ],
        axis=0,
    ).astype(
        np.uint8,
        copy=False,
    )

    allowed = (
        (vals == ESTIMATOR_EMI)
        |
        (vals == ESTIMATOR_EVD)
        |
        (vals == ESTIMATOR_INVALID)
    )

    if not np.all(
        allowed
    ):
        raise ValueError(
            "unknown estimator code "
            "in sequential stage output"
        )

    out = np.full(
        rr.size,
        ESTIMATOR_EMI,
        dtype=np.uint8,
    )

    any_invalid = np.any(
        vals == ESTIMATOR_INVALID,
        axis=0,
    )

    any_evd = np.any(
        vals == ESTIMATOR_EVD,
        axis=0,
    )

    out[
        any_evd
        &
        ~any_invalid
    ] = ESTIMATOR_EVD

    out[
        any_invalid
    ] = ESTIMATOR_INVALID

    return out


def evaluate_fullspan_quality_points(
    *,
    yxt,
    phase_points=None,
    phase_cube=None,

    rows,
    cols,

    scale2,
    valid,
    ps,

    state_core,
    expected_effective_k,

    half_row: int = 5,
    half_col: int = 11,
    alpha: float = 0.005,

    batch: int = 4096,
    support_block: int = 1024,
    static_support_cache=None,
) -> FullspanQualityResult:
    """
    Re-evaluate final sequential phase histories against the
    full temporal stack using the frozen K-state support.

    This is the production form of the validated full-span
    sequential quality audit.

    Important
    ---------
    This function DOES NOT phase-link again.

    It only performs:

        frozen full-stack GLRT support
            intersect K-state core
        -> full-span coherence
        -> TC(final sequential phase)
        -> median pair coherence

    Therefore quality is evaluated over the final N-date phase
    history rather than by averaging ministack TC values.
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

    # --------------------------------------------------------
    # Phase source
    #
    # phase_points:
    #     legacy/in-memory [point,date] array
    #
    # phase_cube:
    #     production mmap [date,row,col]
    #
    # Exactly one source is required.
    # --------------------------------------------------------

    if (
        phase_points is None
        and
        phase_cube is None
    ):
        raise ValueError(
            "one of phase_points or phase_cube is required"
        )

    if (
        phase_points is not None
        and
        phase_cube is not None
    ):
        raise ValueError(
            "phase_points and phase_cube are mutually exclusive"
        )

    phase = None
    cube = None

    if phase_points is not None:

        phase = np.asarray(
            phase_points,
            dtype=np.complex64,
        )

        if phase.shape != (
            rr.size,
            ndate,
        ):
            raise ValueError(
                f"phase_points shape={phase.shape}, "
                f"expected={(rr.size, ndate)}"
            )

    else:

        cube = np.asarray(
            phase_cube
        )

        if cube.shape != (
            ndate,
            H,
            W,
        ):
            raise ValueError(
                f"phase_cube shape={cube.shape}, "
                f"expected={(ndate, H, W)}"
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

    core = np.asarray(
        state_core,
        dtype=np.bool_,
    )

    expected_k_map = np.asarray(
        expected_effective_k
    )

    for name, arr in (
        ("scale2", scale2),
        ("valid", valid),
        ("ps", ps),
        ("state_core", core),
        (
            "expected_effective_k",
            expected_k_map,
        ),
    ):
        if arr.shape != (
            H,
            W,
        ):
            raise ValueError(
                f"{name} shape={arr.shape}, "
                f"expected={(H, W)}"
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

    core_windows = make_windows(
        core,
        half_row=half_row,
        half_col=half_col,
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
        label="fullspan-quality",
        total=n,
        unit="center",
        min_interval=10.0,
    )

    effective_k = np.full(
        n,
        -1,
        dtype=np.int16,
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

    phase_complete = np.zeros(
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

        if static_support_cache is None:

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

        else:

            # Exact GLRT support was generated once by
            # glrt_support_vectorized_exact() and then
            # losslessly packed into the static cache.
            support = (
                static_support_cache.support(
                    br,
                    bc,
                )
            )

        support &= np.asarray(
            core_windows[
                br,
                bc,
            ],
            dtype=np.bool_,
        )

        K = np.sum(
            support,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

        effective_k[
            b0:b1
        ] = K

        expected = expected_k_map[
            br,
            bc,
        ].astype(
            np.int16,
            copy=False,
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
                "full-span K parity failure at "
                f"({int(br[k])},{int(bc[k])}): "
                f"computed={int(K[k])}, "
                f"expected={int(expected[k])}"
            )

        coh = compressed_coherence(
            yxt,
            br,
            bc,
            support,
            pi,
            pj,
        )

        # ----------------------------------------------------
        # Read only this batch of phase histories.
        #
        # For large scenes this avoids materializing the full
        # [Npoint,Ndate] sequential phase matrix.
        # ----------------------------------------------------

        if phase is not None:

            phase_b = np.asarray(
                phase[
                    b0:b1
                ],
                dtype=np.complex64,
            )

        else:

            phase_b = np.asarray(
                cube[
                    :,
                    br,
                    bc,
                ].T,
                dtype=np.complex64,
            )

        complete = np.all(
            np.isfinite(
                phase_b.real
            )
            &
            np.isfinite(
                phase_b.imag
            ),
            axis=1,
        )

        phase_complete[
            b0:b1
        ] = complete

        tc_b, pair_b = (
            temporal_quality_streaming(
                coh,
                phase_b,
                pi,
                pj,
            )
        )

        tc_b = np.asarray(
            tc_b,
            dtype=np.float32,
        )

        pair_b = np.asarray(
            pair_b,
            dtype=np.float32,
        )

        # A production sequential phase history must contain
        # every real acquisition. Do not silently derive TC
        # from an incomplete phase history.
        tc_b[
            ~complete
        ] = np.nan

        pair_b[
            ~complete
        ] = np.nan

        tc[
            b0:b1
        ] = tc_b

        pair_coh[
            b0:b1
        ] = pair_b

        progress.update(
            b1
        )

    progress.finish(
        n
    )

    return FullspanQualityResult(
        rows=rr,
        cols=cc,

        effective_k=(
            effective_k
        ),

        temporal_coherence=(
            tc
        ),

        median_pair_coherence=(
            pair_coh
        ),

        phase_complete=(
            phase_complete
        ),
    )


__all__ = [
    "FullspanQualityResult",
    "aggregate_stage_estimators",
    "evaluate_fullspan_quality_points",
]
