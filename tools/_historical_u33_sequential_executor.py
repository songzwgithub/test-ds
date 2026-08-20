from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .coherence import compressed_coherence
from .compression import compress_stage_slcs
from .emi import (
    ESTIMATOR_INVALID,
    image_pairs,
    robust_emi_threaded,
    temporal_coherence,
)
from .shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


STATE_OUTSIDE = np.uint8(0)
STATE_VALID = np.uint8(1)
STATE_LOW_K = np.uint8(2)
STATE_PL_INVALID = np.uint8(3)
STATE_COMPRESSION_INVALID = np.uint8(4)
STATE_CENTER_INPUT_INVALID = np.uint8(5)


@dataclass(slots=True)
class FirstMiniStackResult:
    state_pixels: int
    state_valid: int

    low_k: int
    pl_invalid: int
    compression_invalid: int
    center_input_invalid: int

    k_parity_mismatch: int
    compression_formula_max_abs_diff: float

    elapsed_seconds: float

    support_seconds: float
    covariance_seconds: float
    phase_linking_seconds: float
    compression_seconds: float

    compressed_path: Path
    valid_path: Path
    state_code_path: Path
    shp_count_path: Path
    temporal_coherence_path: Path
    estimator_path: Path


def _new_memmap(
    path: Path,
    *,
    shape,
    dtype,
    fill,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arr = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )

    arr[...] = fill

    return arr


def _bool_windows(
    arr: np.ndarray,
    *,
    half_row: int,
    half_col: int,
):
    arr = np.asarray(
        arr,
        dtype=np.bool_,
    )

    padded = np.pad(
        arr,
        (
            (half_row, half_row),
            (half_col, half_col),
        ),
        mode="constant",
        constant_values=False,
    )

    return (
        np.lib.stride_tricks
        .sliding_window_view(
            padded,
            (
                2 * half_row + 1,
                2 * half_col + 1,
            ),
        )
    )


def _iter_tiles(
    H: int,
    W: int,
    tile_rows: int,
    tile_cols: int,
):
    for r0 in range(
        0,
        H,
        tile_rows,
    ):
        r1 = min(
            H,
            r0 + tile_rows,
        )

        for c0 in range(
            0,
            W,
            tile_cols,
        ):
            c1 = min(
                W,
                c0 + tile_cols,
            )

            yield (
                r0,
                r1,
                c0,
                c1,
            )


def _manual_compression(
    real_z: np.ndarray,
    phase: np.ndarray,
):
    """
    Independent reference formula for audit only.

    Same mathematical definition as sequential compression,
    but intentionally does not call compress_stage_slcs().
    """

    real_z = np.asarray(
        real_z,
        dtype=np.complex64,
    )

    phase = np.asarray(
        phase,
        dtype=np.complex64,
    )

    ref = phase[
        :,
        0,
    ][
        :,
        None,
    ]

    ph_ref = (
        phase
        *
        np.conj(
            ref
        )
    ).astype(
        np.complex64,
        copy=False,
    )

    with np.errstate(
        invalid="ignore",
    ):
        projected = np.nansum(
            real_z
            *
            np.conj(
                ph_ref
            ),
            axis=1,
        )

        amp = np.nanmean(
            np.abs(
                real_z
            ),
            axis=1,
            dtype=np.float64,
        ).astype(
            np.float32
        )

    out = (
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

    bad = (
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
        bad
    ] = np.complex64(
        np.nan
        +
        1j
        *
        np.nan
    )

    return out


def run_first_ministack_fused(
    *,
    yxt: np.ndarray,
    scale2: np.ndarray,
    valid: np.ndarray,
    ps: np.ndarray,
    state_core: np.ndarray,
    expected_effective_k: np.ndarray,
    real_indices: tuple[int, ...],
    output_dir: Path,
    full_glrt_nslc: int,
    state_min_shp: int = 24,
    half_row: int = 5,
    half_col: int = 11,
    alpha: float = 0.005,
    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,
    tile_rows: int = 256,
    tile_cols: int = 512,
    center_batch: int = 16000,
    support_block: int = 1024,
    pl_workers: int = 16,
    pl_chunk_size: int = 512,
    formula_audit_points: int = 5000,
) -> FirstMiniStackResult:
    """
    U3.3a:
    fused first-ministack phase linking -> compressed SLC.

    Critical constraints
    --------------------
    * First sequential stage only.
    * No previous compressed inputs.
    * solver reference index = 0.
    * compressed reference index = 0.
    * GLRT uses the frozen full-stack nslc.
    * Spatial SHP is restricted to the frozen K24 state core.
    * Linked phase is never materialized as a scene-wide cube.
    """

    yxt = np.asarray(
        yxt,
    )

    H, W, ndate = yxt.shape

    if scale2.shape != (
        H,
        W,
    ):
        raise ValueError(
            "scale2 shape mismatch"
        )

    if valid.shape != (
        H,
        W,
    ):
        raise ValueError(
            "valid shape mismatch"
        )

    if ps.shape != (
        H,
        W,
    ):
        raise ValueError(
            "ps shape mismatch"
        )

    if state_core.shape != (
        H,
        W,
    ):
        raise ValueError(
            "state_core shape mismatch"
        )

    if expected_effective_k.shape != (
        H,
        W,
    ):
        raise ValueError(
            "expected_effective_k shape mismatch"
        )

    real_indices = tuple(
        int(i)
        for i
        in real_indices
    )

    if not real_indices:
        raise ValueError(
            "real_indices is empty"
        )

    # U3.3a intentionally supports the FIRST stage only.
    if real_indices != tuple(
        range(
            len(
                real_indices
            )
        )
    ):
        raise ValueError(
            "U3.3a requires first-stage "
            "real_indices = 0..M-1"
        )

    if (
        real_indices[-1]
        >=
        ndate
    ):
        raise ValueError(
            "real index outside YXT stack"
        )

    if (
        full_glrt_nslc
        !=
        ndate
    ):
        raise ValueError(
            "U3.3a expects frozen GLRT "
            "nslc to equal full input stack size"
        )

    if state_min_shp < 1:
        raise ValueError(
            "state_min_shp must be positive"
        )

    valid = np.asarray(
        valid,
        dtype=np.bool_,
    )

    ps = np.asarray(
        ps,
        dtype=np.bool_,
    )

    state_core = np.asarray(
        state_core,
        dtype=np.bool_,
    )

    expected_effective_k = (
        np.asarray(
            expected_effective_k,
            dtype=np.int16,
        )
    )

    if np.any(
        state_core
        &
        ~valid
    ):
        raise RuntimeError(
            "state core contains invalid pixels"
        )

    if np.any(
        state_core
        &
        ps
    ):
        raise RuntimeError(
            "state core unexpectedly contains PS"
        )

    expected_core_k = (
        expected_effective_k[
            state_core
        ]
    )

    if np.any(
        expected_core_k
        <
        state_min_shp
    ):
        raise RuntimeError(
            "state core is not a valid "
            "fixed-point K>=state_min_shp core"
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    compressed_path = (
        output_dir
        /
        "u33a_stage0000_compressed.npy"
    )

    valid_path = (
        output_dir
        /
        "u33a_stage0000_state_valid.npy"
    )

    code_path = (
        output_dir
        /
        "u33a_stage0000_state_code.npy"
    )

    k_path = (
        output_dir
        /
        "u33a_stage0000_effective_shp_count.npy"
    )

    tc_path = (
        output_dir
        /
        "u33a_stage0000_temporal_coherence.npy"
    )

    est_path = (
        output_dir
        /
        "u33a_stage0000_estimator.npy"
    )

    compressed_out = _new_memmap(
        compressed_path,
        shape=(H, W),
        dtype=np.complex64,
        fill=(
            np.nan
            +
            1j
            *
            np.nan
        ),
    )

    state_valid_out = _new_memmap(
        valid_path,
        shape=(H, W),
        dtype=np.bool_,
        fill=False,
    )

    state_code_out = _new_memmap(
        code_path,
        shape=(H, W),
        dtype=np.uint8,
        fill=STATE_OUTSIDE,
    )

    k_out = _new_memmap(
        k_path,
        shape=(H, W),
        dtype=np.int16,
        fill=-1,
    )

    tc_out = _new_memmap(
        tc_path,
        shape=(H, W),
        dtype=np.float32,
        fill=np.nan,
    )

    est_out = _new_memmap(
        est_path,
        shape=(H, W),
        dtype=np.uint8,
        fill=ESTIMATOR_INVALID,
    )

    stage_n = len(
        real_indices
    )

    pairs = image_pairs(
        stage_n
    )

    pi = np.asarray(
        pairs[
            :,
            0
        ],
        dtype=np.int32,
    )

    pj = np.asarray(
        pairs[
            :,
            1
        ],
        dtype=np.int32,
    )

    tiles = list(
        _iter_tiles(
            H,
            W,
            tile_rows,
            tile_cols,
        )
    )

    total_state = int(
        np.count_nonzero(
            state_core
        )
    )

    total_done = 0
    total_valid = 0

    n_low_k = 0
    n_pl_invalid = 0
    n_comp_invalid = 0
    n_center_input_invalid = 0

    k_parity_mismatch = 0

    formula_checked = 0
    formula_max_abs_diff = 0.0

    support_seconds = 0.0
    covariance_seconds = 0.0
    phase_link_seconds = 0.0
    compression_seconds = 0.0

    t_all = perf_counter()

    for tile_index, (
        r0,
        r1,
        c0,
        c1,
    ) in enumerate(
        tiles,
        start=1,
    ):

        # ----------------------------------------------------
        # Core state centers for this tile.
        # ----------------------------------------------------

        core_sub = state_core[
            r0:r1,
            c0:c1,
        ]

        sr, sc = np.where(
            core_sub
        )

        if sr.size == 0:
            continue

        gr = (
            sr
            +
            r0
        ).astype(
            np.int32,
            copy=False,
        )

        gc = (
            sc
            +
            c0
        ).astype(
            np.int32,
            copy=False,
        )

        # ----------------------------------------------------
        # Read core + exact SHP halo.
        # ----------------------------------------------------

        ir0 = max(
            0,
            r0
            -
            half_row,
        )

        ir1 = min(
            H,
            r1
            +
            half_row,
        )

        ic0 = max(
            0,
            c0
            -
            half_col,
        )

        ic1 = min(
            W,
            c1
            +
            half_col,
        )

        # First stage real indices are contiguous 0..M-1.
        m = stage_n

        stage_tile = (
            np.ascontiguousarray(
                yxt[
                    ir0:ir1,
                    ic0:ic1,
                    0:m,
                ],
                dtype=np.complex64,
            )
        )

        scale_tile = np.ascontiguousarray(
            scale2[
                ir0:ir1,
                ic0:ic1,
            ],
            dtype=np.float32,
        )

        valid_tile = np.ascontiguousarray(
            valid[
                ir0:ir1,
                ic0:ic1,
            ],
            dtype=np.bool_,
        )

        ps_tile = np.ascontiguousarray(
            ps[
                ir0:ir1,
                ic0:ic1,
            ],
            dtype=np.bool_,
        )

        state_tile = np.ascontiguousarray(
            state_core[
                ir0:ir1,
                ic0:ic1,
            ],
            dtype=np.bool_,
        )

        # A spatial sample may enter stage covariance only if
        # every stage image is finite there.
        stage_sample_valid = np.all(
            np.isfinite(
                stage_tile.real
            )
            &
            np.isfinite(
                stage_tile.imag
            ),
            axis=2,
        )

        ctx = (
            prepare_glrt_window_context(
                scale_tile,
                valid_tile,
                ps_tile,
                half_row=half_row,
                half_col=half_col,
            )
        )

        state_windows = _bool_windows(
            state_tile,
            half_row=half_row,
            half_col=half_col,
        )

        stage_valid_windows = _bool_windows(
            stage_sample_valid,
            half_row=half_row,
            half_col=half_col,
        )

        lr = (
            gr
            -
            ir0
        ).astype(
            np.int32,
            copy=False,
        )

        lc = (
            gc
            -
            ic0
        ).astype(
            np.int32,
            copy=False,
        )

        for b0 in range(
            0,
            gr.size,
            center_batch,
        ):

            b1 = min(
                gr.size,
                b0
                +
                center_batch,
            )

            br = lr[
                b0:b1
            ]

            bc = lc[
                b0:b1
            ]

            bgr = gr[
                b0:b1
            ]

            bgc = gc[
                b0:b1
            ]

            # ------------------------------------------------
            # Exact frozen GLRT.
            # ------------------------------------------------

            ts = perf_counter()

            support, _ = (
                glrt_support_vectorized_exact(
                    ctx,
                    br,
                    bc,
                    alpha=alpha,
                    nslc=full_glrt_nslc,
                    block_size=support_block,
                )
            )

            # Production state support:
            #
            # original exact SHP
            #   ∩ K24 fixed-point state
            #   ∩ input state validity
            support &= np.asarray(
                state_windows[
                    br,
                    bc,
                ],
                dtype=np.bool_,
            )

            support &= np.asarray(
                stage_valid_windows[
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

            support_seconds += (
                perf_counter()
                -
                ts
            )

            k_out[
                bgr,
                bgc,
            ] = K

            # ------------------------------------------------
            # HARD tile/halo parity gate against U3.2c6.
            # ------------------------------------------------

            expected = (
                expected_effective_k[
                    bgr,
                    bgc,
                ]
            )

            mismatch = (
                K
                !=
                expected
            )

            if np.any(
                mismatch
            ):

                k_parity_mismatch += int(
                    np.count_nonzero(
                        mismatch
                    )
                )

                first = int(
                    np.flatnonzero(
                        mismatch
                    )[0]
                )

                raise RuntimeError(
                    "U3.3a K parity failure at "
                    f"global pixel "
                    f"({int(bgr[first])},"
                    f"{int(bgc[first])}): "
                    f"tile K={int(K[first])}, "
                    f"reference K="
                    f"{int(expected[first])}"
                )

            center_input_ok = (
                stage_sample_valid[
                    br,
                    bc,
                ]
            )

            bad_center_input = (
                ~center_input_ok
            )

            if np.any(
                bad_center_input
            ):
                state_code_out[
                    bgr[
                        bad_center_input
                    ],
                    bgc[
                        bad_center_input
                    ],
                ] = (
                    STATE_CENTER_INPUT_INVALID
                )

                n_center_input_invalid += int(
                    np.count_nonzero(
                        bad_center_input
                    )
                )

            good_k = (
                center_input_ok
                &
                (
                    K
                    >=
                    state_min_shp
                )
            )

            low_k = (
                center_input_ok
                &
                ~good_k
            )

            if np.any(
                low_k
            ):
                state_code_out[
                    bgr[
                        low_k
                    ],
                    bgc[
                        low_k
                    ],
                ] = (
                    STATE_LOW_K
                )

                n_low_k += int(
                    np.count_nonzero(
                        low_k
                    )
                )

            if not np.any(
                good_k
            ):
                total_done += int(
                    bgr.size
                )

                continue

            gr2 = bgr[
                good_k
            ]

            gc2 = bgc[
                good_k
            ]

            lr2 = br[
                good_k
            ]

            lc2 = bc[
                good_k
            ]

            support2 = support[
                good_k
            ]

            # ------------------------------------------------
            # Covariance/coherence.
            # ------------------------------------------------

            ts = perf_counter()

            coh = compressed_coherence(
                stage_tile,
                lr2,
                lc2,
                support2,
                pi,
                pj,
            )

            covariance_seconds += (
                perf_counter()
                -
                ts
            )

            # ------------------------------------------------
            # Phase linking.
            #
            # Stage 0:
            # solver reference index = 0.
            # ------------------------------------------------

            ts = perf_counter()

            (
                ph,
                est,
                _,
                _,
                _,
            ) = robust_emi_threaded(
                coh,
                n_images=stage_n,
                pairs=pairs,
                beta=beta,
                gamma_jitter=gamma_jitter,
                emi_mu=emi_mu,
                reference_idx=0,
                workers=pl_workers,
                chunk_size=pl_chunk_size,
            )

            tc = temporal_coherence(
                coh,
                ph,
                pairs,
            )

            phase_link_seconds += (
                perf_counter()
                -
                ts
            )

            est_out[
                gr2,
                gc2,
            ] = est

            tc_out[
                gr2,
                gc2,
            ] = tc

            ph_finite = np.all(
                np.isfinite(
                    ph.real
                )
                &
                np.isfinite(
                    ph.imag
                ),
                axis=1,
            )

            pl_ok = (
                (est != ESTIMATOR_INVALID)
                &
                np.isfinite(
                    tc
                )
                &
                ph_finite
            )

            if np.any(
                ~pl_ok
            ):

                bad_r = gr2[
                    ~pl_ok
                ]

                bad_c = gc2[
                    ~pl_ok
                ]

                state_code_out[
                    bad_r,
                    bad_c,
                ] = (
                    STATE_PL_INVALID
                )

                n_pl_invalid += int(
                    bad_r.size
                )

            if not np.any(
                pl_ok
            ):

                total_done += int(
                    bgr.size
                )

                continue

            gr3 = gr2[
                pl_ok
            ]

            gc3 = gc2[
                pl_ok
            ]

            lr3 = lr2[
                pl_ok
            ]

            lc3 = lc2[
                pl_ok
            ]

            ph3 = ph[
                pl_ok
            ]

            # ------------------------------------------------
            # FUSED compression.
            #
            # No phase image is written.
            # ------------------------------------------------

            z3 = stage_tile[
                lr3,
                lc3,
                :,
            ]

            ts = perf_counter()

            comp = compress_stage_slcs(
                z3,
                ph3,
                first_real_idx=0,
                compressed_reference_idx=0,
                mean_amplitude=None,
            )

            compression_seconds += (
                perf_counter()
                -
                ts
            )

            comp_ok = (
                np.isfinite(
                    comp.real
                )
                &
                np.isfinite(
                    comp.imag
                )
            )

            if np.any(
                ~comp_ok
            ):

                bad_r = gr3[
                    ~comp_ok
                ]

                bad_c = gc3[
                    ~comp_ok
                ]

                state_code_out[
                    bad_r,
                    bad_c,
                ] = (
                    STATE_COMPRESSION_INVALID
                )

                n_comp_invalid += int(
                    bad_r.size
                )

            if np.any(
                comp_ok
            ):

                wr = gr3[
                    comp_ok
                ]

                wc = gc3[
                    comp_ok
                ]

                compressed_out[
                    wr,
                    wc,
                ] = comp[
                    comp_ok
                ]

                state_valid_out[
                    wr,
                    wc,
                ] = True

                state_code_out[
                    wr,
                    wc,
                ] = (
                    STATE_VALID
                )

                total_valid += int(
                    wr.size
                )

            # ------------------------------------------------
            # Independent compression-formula audit.
            # ------------------------------------------------

            remaining = (
                formula_audit_points
                -
                formula_checked
            )

            if (
                remaining > 0
                and
                np.any(
                    comp_ok
                )
            ):

                ids = np.flatnonzero(
                    comp_ok
                )[
                    :remaining
                ]

                manual = _manual_compression(
                    z3[
                        ids
                    ],
                    ph3[
                        ids
                    ],
                )

                delta = np.abs(
                    comp[
                        ids
                    ]
                    -
                    manual
                )

                finite_delta = delta[
                    np.isfinite(
                        delta
                    )
                ]

                if finite_delta.size:

                    formula_max_abs_diff = max(
                        formula_max_abs_diff,
                        float(
                            np.max(
                                finite_delta
                            )
                        ),
                    )

                formula_checked += int(
                    ids.size
                )

            total_done += int(
                bgr.size
            )

        elapsed = (
            perf_counter()
            -
            t_all
        )

        rate = (
            total_done
            /
            elapsed
            if elapsed > 0
            else 0.0
        )

        print(
            f"tile "
            f"{tile_index:3d}/"
            f"{len(tiles):3d} "
            f"core="
            f"r{r0}:{r1},"
            f"c{c0}:{c1} "
            f"state="
            f"{total_done:,}/"
            f"{total_state:,} "
            f"valid="
            f"{total_valid:,} "
            f"rate="
            f"{rate:,.0f} center/s"
        )

    # --------------------------------------------------------
    # Final flush only.
    #
    # U3.3a intentionally does NOT flush after every batch.
    # --------------------------------------------------------

    for arr in (
        compressed_out,
        state_valid_out,
        state_code_out,
        k_out,
        tc_out,
        est_out,
    ):
        arr.flush()

    elapsed = (
        perf_counter()
        -
        t_all
    )

    return FirstMiniStackResult(
        state_pixels=total_state,
        state_valid=total_valid,

        low_k=n_low_k,
        pl_invalid=n_pl_invalid,
        compression_invalid=n_comp_invalid,
        center_input_invalid=(
            n_center_input_invalid
        ),

        k_parity_mismatch=(
            k_parity_mismatch
        ),

        compression_formula_max_abs_diff=(
            formula_max_abs_diff
        ),

        elapsed_seconds=elapsed,

        support_seconds=support_seconds,
        covariance_seconds=(
            covariance_seconds
        ),
        phase_linking_seconds=(
            phase_link_seconds
        ),
        compression_seconds=(
            compression_seconds
        ),

        compressed_path=compressed_path,
        valid_path=valid_path,
        state_code_path=code_path,
        shp_count_path=k_path,
        temporal_coherence_path=tc_path,
        estimator_path=est_path,
    )


__all__ = [
    "STATE_OUTSIDE",
    "STATE_VALID",
    "STATE_LOW_K",
    "STATE_PL_INVALID",
    "STATE_COMPRESSION_INVALID",
    "STATE_CENTER_INPUT_INVALID",
    "FirstMiniStackResult",
    "run_first_ministack_fused",
]
