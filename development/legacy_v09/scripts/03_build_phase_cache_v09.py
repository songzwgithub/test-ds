from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config
from pypsds.gamma.phase_correction import GammaPointPhaseCorrectionProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tile-rows", type=int, default=128)
    ap.add_argument("--tile-cols", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg, config_path, paths, stack, (base_row0, base_col0, H, W) = open_from_config(args.config)
    N = len(stack.dates)
    cache_dir = Path(paths.output_dir) / "v09" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yxt_path = cache_dir / "phase_corrected_yxt.npy"
    geom_path = cache_dir / "phase_geometry_valid.npy"
    manifest_path = cache_dir / "phase_cache_manifest.json"

    if yxt_path.exists() and geom_path.exists() and not args.overwrite:
        yxt = np.load(yxt_path, mmap_mode="r")
        g = np.load(geom_path, mmap_mode="r")
        if yxt.shape == (H, W, N) and yxt.dtype == np.complex64 and g.shape == (H, W):
            print(f"v0.9 phase cache already valid: {yxt_path}")
            return
        raise RuntimeError("Existing cache shape/dtype mismatch; use --overwrite")

    print("=" * 80)
    print("v0.9 - parallel GAMMA phase correction -> C-contiguous Y-X-Time cache")
    print("=" * 80)
    print(f"config   : {config_path}")
    print(f"scene    : {H} x {W} x {N}")
    print(f"tile     : {args.tile_rows} x {args.tile_cols}")
    print(f"workers  : {args.workers}")
    print(f"target   : {yxt_path}")

    raw = stack.read_window(row0=base_row0, col0=base_col0, rows=H, cols=W).astype(np.complex64, copy=False)
    provider = GammaPointPhaseCorrectionProvider(cfg, paths, stack)
    assets = provider.prepare()

    yxt = np.lib.format.open_memmap(yxt_path, mode="w+", dtype=np.complex64, shape=(H, W, N))
    geom = np.zeros((H, W), dtype=bool)
    tiles = []
    tid = 0
    for tr0 in range(0, H, args.tile_rows):
        tr1 = min(H, tr0 + args.tile_rows)
        for tc0 in range(0, W, args.tile_cols):
            tc1 = min(W, tc0 + args.tile_cols)
            tiles.append((tid, tr0, tr1, tc0, tc1))
            tid += 1

    def worker(tile):
        tid, tr0, tr1, tc0, tc1 = tile
        block = np.ascontiguousarray(raw[:, tr0:tr1, tc0:tc1])
        corrected, valid, stats = provider.correct_block(
            block,
            global_row0=base_row0 + tr0,
            global_col0=base_col0 + tc0,
            tile_label=f"v09cache_{tid:04d}_r{tr0}_{tr1}_c{tc0}_{tc1}",
        )
        # Materialize real C-contiguous Y-X-Time layout here; no v0.6 transpose cache needed.
        corrected_yxt = np.ascontiguousarray(np.moveaxis(corrected, 0, -1))
        return tid, tr0, tr1, tc0, tc1, corrected_yxt, valid, stats

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(worker, t) for t in tiles]
        for fut in as_completed(futs):
            tid, tr0, tr1, tc0, tc1, block_yxt, valid, stats = fut.result()
            yxt[tr0:tr1, tc0:tc1, :] = block_yxt
            geom[tr0:tr1, tc0:tc1] = valid
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0.0
            eta = (len(tiles)-done)/rate if rate else 0.0
            print(
                f"tile {done:3d}/{len(tiles):3d} ({100*done/len(tiles):6.2f}%) "
                f"phase={stats.simulation_seconds:5.1f}s elapsed={elapsed:7.1f}s ETA={eta:6.1f}s"
            )
            yxt.flush()

    np.save(geom_path, geom)
    yxt.flush()
    del yxt
    chk = np.load(yxt_path, mmap_mode="r")
    manifest = {
        "version": "0.9.0",
        "config": str(config_path),
        "dates": list(stack.dates),
        "shape_yxt": list(chk.shape),
        "dtype": str(chk.dtype),
        "c_contiguous": bool(chk.flags.c_contiguous),
        "strides": list(chk.strides),
        "geometric_reference_date": assets.reference_date,
        "apply_sign": float(provider.sign),
        "geometry_valid": int(geom.sum()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"geometry valid: {geom.sum()}/{geom.size} ({100*geom.mean():.3f}%)")
    print(f"C contiguous  : {chk.flags.c_contiguous}")
    print(f"strides       : {chk.strides}")
    print(f"elapsed       : {time.time()-t0:.1f}s")
    print(f"saved         : {yxt_path}")
    print(f"saved         : {geom_path}")


if __name__ == "__main__":
    main()
