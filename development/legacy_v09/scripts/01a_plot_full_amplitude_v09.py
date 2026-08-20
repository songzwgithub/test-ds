#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypsds.prototype import open_from_config


def main():
    ap = argparse.ArgumentParser(
        description="Plot full-scene amplitude of the first acquisition."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--pmin",
        type=float,
        default=1.0,
        help="Display lower percentile.",
    )
    ap.add_argument(
        "--pmax",
        type=float,
        default=99.5,
        help="Display upper percentile.",
    )
    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (row0, col0, H, W),
    ) = open_from_config(args.config)

    print("=" * 80)
    print("Step 01a - Full-scene first-acquisition amplitude")
    print("=" * 80)
    print(f"config       : {config_path}")
    print(f"scene        : {H} x {W}")
    print(f"dates        : {len(stack.dates)}")
    print(f"first date   : {stack.dates[0]}")

    # read_window returns [date, row, col]
    z = stack.read_window(
        row0=row0,
        col0=col0,
        rows=H,
        cols=W,
    )

    first = z[0]

    valid = (
        np.isfinite(first.real)
        & np.isfinite(first.imag)
        & ~(
            (first.real == 0)
            & (first.imag == 0)
        )
    )

    amp = np.full(
        (H, W),
        np.nan,
        dtype=np.float32,
    )

    amp[valid] = np.abs(
        first[valid]
    ).astype(np.float32)

    vals = amp[np.isfinite(amp)]

    if vals.size == 0:
        raise RuntimeError(
            "No valid pixels in first acquisition."
        )

    vmin = float(
        np.percentile(vals, args.pmin)
    )
    vmax = float(
        np.percentile(vals, args.pmax)
    )

    print(
        f"valid pixels : {vals.size}/{H*W} "
        f"({100*vals.size/(H*W):.3f}%)"
    )
    print(f"amplitude min: {np.nanmin(amp):.6f}")
    print(f"amplitude p01: {np.nanpercentile(amp,1):.6f}")
    print(f"amplitude med: {np.nanmedian(amp):.6f}")
    print(f"amplitude p99: {np.nanpercentile(amp,99):.6f}")
    print(f"amplitude max: {np.nanmax(amp):.6f}")
    print(
        f"display      : "
        f"p{args.pmin:g}={vmin:.3f}, "
        f"p{args.pmax:g}={vmax:.3f}"
    )

    outdir = Path(paths.output_dir) / "v09" / "figures"
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = outdir / "01_amplitude_sample.png"
    npy = outdir / "01_first_acquisition_amplitude.npy"

    np.save(
        npy,
        amp,
    )

    fig, ax = plt.subplots(
        figsize=(16, 6),
    )

    im = ax.imshow(
        amp,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(
        "Step 01 - First acquisition amplitude, full scene\n"
        f"{stack.dates[0]} | {H} × {W} pixels"
    )
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")

    cb = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
    )
    cb.set_label("Amplitude")

    fig.tight_layout()
    fig.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print()
    print(f"PNG          : {png}")
    print(f"NPY          : {npy}")
    print("Step 01a complete.")


if __name__ == "__main__":
    main()
