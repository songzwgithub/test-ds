from __future__ import annotations

import numpy as np

from _common import parser
from pypsds.config import cfg_get
from pypsds.progress import ProgressTracker, StepTimer
from pypsds.prototype import open_from_config
from pypsds.qa.plots import plot_step01


def _tiles(rows, cols, tr, tc):
    for r0 in range(0, rows, tr):
        r1 = min(rows, r0 + tr)
        for c0 in range(0, cols, tc):
            c1 = min(cols, c0 + tc)
            yield r0, r1, c0, c1


def main() -> None:
    args = parser("Step 01 - validate GAMMA RSLC stack reader").parse_args()
    cfg, config_path, paths, stack, (row0, col0, rows, cols) = open_from_config(args.config)
    max_dates = cfg_get(cfg, "prototype.max_dates", None)
    date_indices = None if max_dates in (None, "null") else list(range(min(int(max_dates), stack.shape[0])))
    dates = stack.dates if date_indices is None else [stack.dates[i] for i in date_indices]
    ndate = len(dates)
    tr = int(cfg_get(cfg, "runtime.stats_tile_rows", cfg_get(cfg, "dense.tile_rows", 128)))
    tc = int(cfg_get(cfg, "runtime.stats_tile_cols", cfg_get(cfg, "dense.tile_cols", 128)))
    tiles = list(_tiles(rows, cols, tr, tc))

    print("=== pyPSDS step 01 ===")
    print(f"config      : {config_path}")
    print(f"RSLC_tab    : {paths.rslc_tab}")
    print(f"RSLC_dir    : {paths.rslc_dir}")
    print(f"stack shape : {stack.shape}")
    print(f"ROI         : row={row0}:{row0+rows}, col={col0}:{col0+cols}")
    print(f"stats tiles : {tr} x {tc} ({len(tiles)} tiles)")
    print(f"dates used  : {ndate}")

    sums = np.zeros(ndate, np.float64)
    sums2 = np.zeros(ndate, np.float64)
    counts = np.zeros(ndate, np.int64)
    amp_min = np.inf
    amp_max = -np.inf
    sample = None
    tracker = ProgressTracker("Step01 amplitude scan", len(tiles))

    with StepTimer("Step01 tiled reader/statistics"):
        for i, (r0, r1, c0, c1) in enumerate(tiles, 1):
            block = stack.read_window(
                row0=row0 + r0,
                col0=col0 + c0,
                rows=r1 - r0,
                cols=c1 - c0,
                date_indices=date_indices,
            )
            amp = np.abs(block).astype(np.float32, copy=False)
            finite = np.isfinite(amp)
            safe = np.where(finite, amp, 0.0).astype(np.float64, copy=False)
            sums += safe.sum(axis=(1, 2))
            sums2 += (safe * safe).sum(axis=(1, 2))
            counts += finite.sum(axis=(1, 2))
            if np.any(finite):
                amp_min = min(amp_min, float(np.nanmin(amp)))
                amp_max = max(amp_max, float(np.nanmax(amp)))
            if sample is None:
                sample = block[:, : min(32, block.shape[1]), : min(32, block.shape[2])].copy()
            tracker.update(i, detail=f"tile={r0}:{r1},{c0}:{c1}")

    mean_amp = (sums / np.maximum(counts, 1)).astype(np.float32)
    var = sums2 / np.maximum(counts, 1) - mean_amp.astype(np.float64) ** 2
    std_amp = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    if sample is None:
        sample = np.empty((ndate, 0, 0), np.complex64)

    out = paths.output_dir / "01_stack_check.npz"
    np.savez_compressed(
        out,
        dates=np.asarray(dates),
        roi=np.asarray([row0, col0, rows, cols], dtype=np.int32),
        mean_amplitude=mean_amp,
        std_amplitude=std_amp,
        sample=sample,
    )
    print(f"amp range   : {amp_min:.6g} .. {amp_max:.6g}")
    print(f"saved       : {out}")
    if bool(cfg_get(cfg, "qa.enabled", True)):
        for p in plot_step01(
            paths.output_dir / "figures",
            dates,
            mean_amp,
            sample,
            int(cfg_get(cfg, "qa.dpi", 140)),
        ):
            print(f"figure      : {p}")


if __name__ == "__main__":
    main()
