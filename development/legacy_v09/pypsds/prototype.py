from __future__ import annotations

from pathlib import Path

from .config import cfg_get, load_config
from .project_paths import resolve_project_paths
from .gamma.stack import GammaStack


def _roi_int(value, default: int) -> int:
    return default if value in (None, "") else int(value)


def open_from_config(config: str | Path | None = None):
    cfg, config_path = load_config(config)
    paths = resolve_project_paths(cfg, config_path)
    stack = GammaStack.from_rslc_tab(
        paths.rslc_tab,
        rslc_dir=paths.rslc_dir,
        dtype=str(cfg_get(cfg, "gamma.rslc_dtype", "auto")),
        byte_order=str(cfg_get(cfg, "gamma.byte_order", "big")),
        io_workers=int(cfg_get(cfg, "runtime.io_workers", 4)),
    )

    roi = cfg.get("prototype", {}).get("roi", {}) or {}
    row0 = _roi_int(roi.get("row0"), 0)
    col0 = _roi_int(roi.get("col0"), 0)
    max_rows = stack.shape[1] - row0
    max_cols = stack.shape[2] - col0
    rows = _roi_int(roi.get("rows"), max_rows)
    cols = _roi_int(roi.get("cols"), max_cols)

    if row0 < 0 or col0 < 0 or rows <= 0 or cols <= 0:
        raise ValueError("ROI values must be non-negative with positive rows/cols")
    if row0 + rows > stack.shape[1] or col0 + cols > stack.shape[2]:
        raise ValueError(
            f"ROI exceeds stack shape {stack.shape}: "
            f"row0={row0}, col0={col0}, rows={rows}, cols={cols}"
        )
    return cfg, config_path, paths, stack, (row0, col0, rows, cols)
