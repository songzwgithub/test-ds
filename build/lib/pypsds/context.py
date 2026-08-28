from __future__ import annotations

from pathlib import Path

from .config import cfg_get, load_config
from .gamma.stack import GammaStack
from .project import resolve_project_paths
from .runtime import logical_cpu_count


def _roi_int(
    value,
    default: int,
) -> int:
    return (
        default
        if value in (None, "")
        else int(value)
    )


def _io_workers(cfg) -> int:

    raw = cfg_get(
        cfg,
        "runtime.io_workers",
        "auto",
    )

    if raw in (
        None,
        "",
        "auto",
        0,
        "0",
    ):
        return min(
            8,
            max(
                1,
                logical_cpu_count() // 4,
            ),
        )

    return max(
        1,
        int(raw),
    )


def open_from_config(
    config: str | Path | None = None,
):

    cfg, config_path = load_config(
        config
    )

    paths = resolve_project_paths(
        cfg,
        config_path,
    )

    stack = GammaStack.from_rslc_tab(
        paths.rslc_tab,
        rslc_dir=paths.rslc_dir,
        dtype=str(
            cfg_get(
                cfg,
                "gamma.rslc_dtype",
                "auto",
            )
        ),
        byte_order=str(
            cfg_get(
                cfg,
                "gamma.byte_order",
                "big",
            )
        ),
        io_workers=_io_workers(cfg),
    )

    roi = (
        cfg_get(
            cfg,
            "processing.roi",
            {},
        )
        or {}
    )

    row0 = _roi_int(
        roi.get("row0"),
        0,
    )

    col0 = _roi_int(
        roi.get("col0"),
        0,
    )

    max_rows = (
        stack.shape[1]
        - row0
    )

    max_cols = (
        stack.shape[2]
        - col0
    )

    rows = _roi_int(
        roi.get("rows"),
        max_rows,
    )

    cols = _roi_int(
        roi.get("cols"),
        max_cols,
    )

    if (
        row0 < 0
        or col0 < 0
        or rows <= 0
        or cols <= 0
    ):
        raise ValueError(
            "ROI must have non-negative "
            "origin and positive dimensions."
        )

    if (
        row0 + rows > stack.shape[1]
        or
        col0 + cols > stack.shape[2]
    ):
        raise ValueError(
            "ROI exceeds RSLC stack: "
            f"stack={stack.shape}, "
            f"row0={row0}, "
            f"col0={col0}, "
            f"rows={rows}, "
            f"cols={cols}"
        )

    return (
        cfg,
        config_path,
        paths,
        stack,
        (
            row0,
            col0,
            rows,
            cols,
        ),
    )


__all__ = [
    "open_from_config",
]
