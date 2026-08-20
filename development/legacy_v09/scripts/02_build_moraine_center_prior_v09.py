from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypsds.prototype import open_from_config
from pypsds.ds.moraine_ks import exact_moraine_shp_count, strict_valid_mask


def save_map(arr, path: Path, title: str, label: str, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(arr, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--half-row", type=int, default=5)
    ap.add_argument("--half-col", type=int, default=5)
    ap.add_argument("--p-max", type=float, default=0.05)
    ap.add_argument("--min-shp", type=int, default=50)
    args = ap.parse_args()

    cfg, config_path, paths, stack, (row0, col0, H, W) = open_from_config(args.config)
    outdir = Path(paths.output_dir) / "v09"
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("v0.9 - exact Moraine KS center prior")
    print("=" * 80)
    print(f"config      : {config_path}")
    print(f"scene       : {H} x {W}")
    print(f"dates       : {len(stack.dates)}")
    print(f"window      : {2*args.half_row+1} x {2*args.half_col+1}")
    print(f"rule        : p < {args.p_max}")
    print(f"candidate   : selected count >= {args.min_shp}")

    t0 = time.time()
    rslc = stack.read_window(row0=row0, col0=col0, rows=H, cols=W).astype(np.complex64, copy=False)
    valid = strict_valid_mask(rslc)
    print(f"strict valid: {valid.sum()}/{valid.size} ({100*valid.mean():.3f}%)")

    count, available, valid2 = exact_moraine_shp_count(
        rslc,
        half_row=args.half_row,
        half_col=args.half_col,
        p_max=args.p_max,
        valid=valid,
    )
    candidate = valid2 & (count >= args.min_shp)
    fraction = np.divide(
        count,
        available,
        out=np.full((H, W), np.nan, np.float32),
        where=available > 0,
    )

    outpath = outdir / "moraine_center_prior.npz"
    np.savez_compressed(
        outpath,
        candidate_mask=candidate,
        shp_count=count,
        available_count=available,
        selected_fraction=fraction,
        valid_pixel=valid2,
        p_max=np.float32(args.p_max),
        min_shp=np.int32(args.min_shp),
    )

    save_map(
        np.where(valid2, count.astype(np.float32), np.nan),
        figdir / "v09_moraine_shp_count.png",
        f"v0.9 - exact Moraine selected-neighbor count\n{2*args.half_row+1}x{2*args.half_col+1}, p<{args.p_max}",
        "Selected neighbor count",
        0,
        (2*args.half_row+1)*(2*args.half_col+1)-1,
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[candidate, 0] = 1.0
    ax.imshow(rgb, origin="upper", aspect="auto")
    ax.set_title(
        f"v0.9 - Moraine DS center prior; N={candidate.sum():,}, density={100*candidate.mean():.2f}%"
    )
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.tight_layout()
    fig.savefig(figdir / "v09_moraine_center_prior.png", dpi=160)
    plt.close(fig)

    print(f"median count : {np.median(count[valid2]):.1f}")
    print(f"candidate    : {candidate.sum()} ({100*candidate.mean():.3f}%)")
    print(f"saved        : {outpath}")
    print(f"elapsed      : {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
