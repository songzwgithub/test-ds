from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pypsds.selection.shp import glrt_threshold


@dataclass(slots=True)
class GlrtWindowContext:
    scale_windows: np.ndarray
    valid_windows: np.ndarray
    ps_windows: np.ndarray
    scale2: np.ndarray

    half_row: int
    half_col: int

    @property
    def wh(self) -> int:
        return 2 * self.half_row + 1

    @property
    def ww(self) -> int:
        return 2 * self.half_col + 1


def prepare_glrt_window_context(
    scale2,
    valid,
    ps,
    *,
    half_row: int,
    half_col: int,
) -> GlrtWindowContext:
    """
    Prepare zero-copy sliding-window views once per tile.

    Padding is only 2-D and therefore bounded by the tile size,
    not by scene size or acquisition count.
    """

    scale2 = np.ascontiguousarray(
        scale2,
        dtype=np.float32,
    )

    valid = np.ascontiguousarray(
        valid,
        dtype=np.bool_,
    )

    ps = np.ascontiguousarray(
        ps,
        dtype=np.bool_,
    )

    pad = (
        (half_row, half_row),
        (half_col, half_col),
    )

    scale_pad = np.pad(
        scale2,
        pad,
        mode="constant",
        constant_values=np.nan,
    )

    valid_pad = np.pad(
        valid,
        pad,
        mode="constant",
        constant_values=False,
    )

    # Outside the scene must never become an SHP.
    ps_pad = np.pad(
        ps,
        pad,
        mode="constant",
        constant_values=True,
    )

    wh = 2 * half_row + 1
    ww = 2 * half_col + 1

    scale_windows = (
        np.lib.stride_tricks.sliding_window_view(
            scale_pad,
            (wh, ww),
        )
    )

    valid_windows = (
        np.lib.stride_tricks.sliding_window_view(
            valid_pad,
            (wh, ww),
        )
    )

    ps_windows = (
        np.lib.stride_tricks.sliding_window_view(
            ps_pad,
            (wh, ww),
        )
    )

    # Expected first two dimensions equal the original tile.
    if scale_windows.shape[:2] != scale2.shape:
        raise RuntimeError(
            f"window context shape mismatch: "
            f"{scale_windows.shape[:2]} != {scale2.shape}"
        )

    return GlrtWindowContext(
        scale_windows=scale_windows,
        valid_windows=valid_windows,
        ps_windows=ps_windows,
        scale2=scale2,
        half_row=half_row,
        half_col=half_col,
    )


def _vectorized_exact_block(
    ctx: GlrtWindowContext,
    rows,
    cols,
    *,
    threshold: float,
    nslc: int,
):
    """
    Exact NumPy arithmetic corresponding to:

      N * (
          2*log((s0+s1)/2)
          - log(s0)
          - log(s1)
      )

    Important:
      subtraction order deliberately matches the existing
      glrt_statistic() expression:
          (2*log(pooled) - log(center)) - log(neighbor)

    No scalar math.log / Numba approximation is used.
    """

    rows = np.asarray(
        rows,
        dtype=np.int32,
    )

    cols = np.asarray(
        cols,
        dtype=np.int32,
    )

    # Copies B x wh x ww float32 -> float64.
    # B is bounded by the PL batch/block size.
    neighbor = np.asarray(
        ctx.scale_windows[
            rows,
            cols,
        ],
        dtype=np.float64,
    )

    good = np.asarray(
        ctx.valid_windows[
            rows,
            cols,
        ],
        dtype=np.bool_,
    )

    good &= ~np.asarray(
        ctx.ps_windows[
            rows,
            cols,
        ],
        dtype=np.bool_,
    )

    center = np.asarray(
        ctx.scale2[
            rows,
            cols,
        ],
        dtype=np.float64,
    )[:, None, None]

    # Preserve log(neighbor) before reusing neighbor storage.
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        log_neighbor = np.log(
            neighbor
        )

        # neighbor becomes pooled.
        neighbor += center
        neighbor *= 0.5

        # neighbor becomes log(pooled).
        np.log(
            neighbor,
            out=neighbor,
        )

        # neighbor becomes 2*log(pooled).
        neighbor *= 2.0

        # EXACT expression order:
        # 2log(pooled) - log(center) - log(neighbor_original)
        neighbor -= np.log(
            center
        )

        neighbor -= log_neighbor

        neighbor *= float(
            nslc
        )

    support = (
        good
        &
        np.isfinite(
            neighbor
        )
        &
        (
            neighbor
            <
            threshold
        )
    )

    # Frozen implementation excludes center.
    support[
        :,
        ctx.half_row,
        ctx.half_col,
    ] = False

    K = np.sum(
        support,
        axis=(1, 2),
        dtype=np.int32,
    ).astype(
        np.int16,
    )

    return (
        support,
        K,
    )


def glrt_support_vectorized_exact(
    ctx: GlrtWindowContext,
    rows,
    cols,
    *,
    alpha: float,
    nslc: int,
    block_size: int = 0,
):
    """
    Exact NumPy GLRT with center-blocking for bounded memory.

    block_size=0 means process the whole supplied center batch.

    Output remains bool [B,wh,ww] so the already validated
    compressed_coherence() can be reused unchanged.
    """

    rows = np.asarray(
        rows,
        dtype=np.int32,
    )

    cols = np.asarray(
        cols,
        dtype=np.int32,
    )

    B = rows.size

    wh = ctx.wh
    ww = ctx.ww

    support = np.zeros(
        (B, wh, ww),
        dtype=np.bool_,
    )

    K = np.zeros(
        B,
        dtype=np.int16,
    )

    if B == 0:
        return support, K

    threshold = glrt_threshold(
        alpha
    )

    if block_size <= 0:
        block_size = B

    block_size = max(
        1,
        int(block_size),
    )

    for start in range(
        0,
        B,
        block_size,
    ):
        stop = min(
            B,
            start + block_size,
        )

        s, k = _vectorized_exact_block(
            ctx,
            rows[start:stop],
            cols[start:stop],
            threshold=threshold,
            nslc=nslc,
        )

        support[
            start:stop
        ] = s

        K[
            start:stop
        ] = k

    return (
        support,
        K,
    )


__all__ = [
    "GlrtWindowContext",
    "prepare_glrt_window_context",
    "glrt_support_vectorized_exact",
]
