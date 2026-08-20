#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypsds.prototype import open_from_config


def percentile_report(name, x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    print()
    print(f"=== {name} (N={len(x)}) ===")

    for p in [
        0, 1, 5, 10, 25,
        50, 75, 90, 95,
        99, 100,
    ]:
        print(
            f"p{p:03d}: "
            f"{np.percentile(x,p):.6f}"
        )


def save_map(
    arr,
    path,
    title,
    label,
    *,
    vmin=None,
    vmax=None,
    cmap=None,
):
    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    im = ax.imshow(
        arr,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_xlabel(
        "Range column"
    )
    ax.set_ylabel(
        "Azimuth row"
    )
    ax.set_title(title)

    cb = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
    )
    cb.set_label(label)

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Audit raw ADI PS versus "
            "per-acquisition amplitude-normalized ADI PS."
        )
    )

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--adi-max",
        type=float,
        default=0.25,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (row0, col0, H, W),
    ) = open_from_config(
        args.config
    )

    dates = list(
        stack.dates
    )
    T = len(dates)

    print("=" * 80)
    print(
        "Step 01b - PS amplitude-normalization stability audit"
    )
    print("=" * 80)

    print(
        f"config          : {config_path}"
    )
    print(
        f"scene           : {H} x {W}"
    )
    print(
        f"dates           : {T}"
    )
    print(
        f"ADI threshold   : {args.adi_max}"
    )

    # ========================================================
    # 1. Read complete RSLC stack
    # ========================================================

    print()
    print(
        "[01b:1/6] Reading full RSLC stack..."
    )

    z = stack.read_window(
        row0=row0,
        col0=col0,
        rows=H,
        cols=W,
    ).astype(
        np.complex64,
        copy=False,
    )

    if z.shape != (
        T,
        H,
        W,
    ):
        raise RuntimeError(
            f"Unexpected stack shape: {z.shape}"
        )

    finite = (
        np.isfinite(z.real)
        & np.isfinite(z.imag)
    )

    nonzero = ~(
        (z.real == 0)
        & (z.imag == 0)
    )

    # Same strict validity rule used by v0.9:
    # every acquisition must be finite and non-zero.
    valid = np.all(
        finite & nonzero,
        axis=0,
    )

    Nvalid = int(
        valid.sum()
    )

    print(
        f"strict valid    : "
        f"{Nvalid}/{H*W} "
        f"({100*Nvalid/(H*W):.3f}%)"
    )

    # ========================================================
    # 2. Amplitudes and acquisition scales
    # ========================================================

    print()
    print(
        "[01b:2/6] Computing acquisition amplitude scales..."
    )

    amp = np.abs(z).astype(
        np.float32
    )

    del z

    scale_median = np.full(
        T,
        np.nan,
        dtype=np.float64,
    )

    scale_mean = np.full(
        T,
        np.nan,
        dtype=np.float64,
    )

    for t in range(T):
        a = amp[t][valid]

        scale_median[t] = np.median(
            a
        )

        scale_mean[t] = np.mean(
            a,
            dtype=np.float64,
        )

        print(
            f"  {t:02d} "
            f"{dates[t]} "
            f"median={scale_median[t]:10.4f} "
            f"mean={scale_mean[t]:10.4f}"
        )

    reference_scale = float(
        np.median(
            scale_median
        )
    )

    correction = (
        reference_scale
        / scale_median
    )

    print()
    print(
        f"reference median scale : "
        f"{reference_scale:.6f}"
    )

    print(
        f"correction min/max     : "
        f"{correction.min():.6f} / "
        f"{correction.max():.6f}"
    )

    # ========================================================
    # 3. Raw ADI
    # ========================================================

    print()
    print(
        "[01b:3/6] Computing raw ADI..."
    )

    raw_mean = np.mean(
        amp,
        axis=0,
        dtype=np.float64,
    )

    raw_std = np.std(
        amp,
        axis=0,
        dtype=np.float64,
        ddof=0,
    )

    adi_raw = np.full(
        (H,W),
        np.nan,
        dtype=np.float32,
    )

    ok = (
        valid
        & np.isfinite(raw_mean)
        & (raw_mean > 0)
    )

    adi_raw[ok] = (
        raw_std[ok]
        / raw_mean[ok]
    ).astype(
        np.float32
    )

    ps_raw = (
        ok
        & (
            adi_raw
            <= args.adi_max
        )
    )

    del raw_mean
    del raw_std

    # ========================================================
    # 4. Per-acquisition normalized ADI
    # ========================================================

    print()
    print(
        "[01b:4/6] Computing acquisition-normalized ADI..."
    )

    # Scale each date in place.
    #
    # This only removes one global multiplicative amplitude
    # scale per acquisition.
    for t in range(T):
        amp[t] *= np.float32(
            correction[t]
        )

    norm_mean = np.mean(
        amp,
        axis=0,
        dtype=np.float64,
    )

    norm_std = np.std(
        amp,
        axis=0,
        dtype=np.float64,
        ddof=0,
    )

    adi_norm = np.full(
        (H,W),
        np.nan,
        dtype=np.float32,
    )

    ok_norm = (
        valid
        & np.isfinite(norm_mean)
        & (norm_mean > 0)
    )

    adi_norm[ok_norm] = (
        norm_std[ok_norm]
        / norm_mean[ok_norm]
    ).astype(
        np.float32
    )

    ps_norm = (
        ok_norm
        & (
            adi_norm
            <= args.adi_max
        )
    )

    del amp
    del norm_mean
    del norm_std

    # ========================================================
    # 5. Compare
    # ========================================================

    print()
    print(
        "[01b:5/6] Comparing PS masks..."
    )

    intersection = (
        ps_raw & ps_norm
    )

    union = (
        ps_raw | ps_norm
    )

    raw_only = (
        ps_raw & ~ps_norm
    )

    norm_only = (
        ps_norm & ~ps_raw
    )

    changed = (
        raw_only | norm_only
    )

    n_raw = int(
        ps_raw.sum()
    )

    n_norm = int(
        ps_norm.sum()
    )

    n_inter = int(
        intersection.sum()
    )

    n_union = int(
        union.sum()
    )

    n_raw_only = int(
        raw_only.sum()
    )

    n_norm_only = int(
        norm_only.sum()
    )

    n_changed = int(
        changed.sum()
    )

    jaccard = (
        n_inter / n_union
        if n_union > 0
        else np.nan
    )

    overlap_raw = (
        n_inter / n_raw
        if n_raw > 0
        else np.nan
    )

    overlap_norm = (
        n_inter / n_norm
        if n_norm > 0
        else np.nan
    )

    print()
    print("=" * 80)
    print("PS stability result")
    print("=" * 80)

    print(
        f"raw PS          : "
        f"{n_raw:8d} "
        f"({100*n_raw/(H*W):7.3f}% scene)"
    )

    print(
        f"normalized PS   : "
        f"{n_norm:8d} "
        f"({100*n_norm/(H*W):7.3f}% scene)"
    )

    print(
        f"intersection    : "
        f"{n_inter:8d}"
    )

    print(
        f"union           : "
        f"{n_union:8d}"
    )

    print(
        f"raw only        : "
        f"{n_raw_only:8d}"
    )

    print(
        f"normalized only : "
        f"{n_norm_only:8d}"
    )

    print(
        f"changed         : "
        f"{n_changed:8d}"
    )

    print(
        f"Jaccard         : "
        f"{jaccard:.6f}"
    )

    print(
        f"raw retained    : "
        f"{100*overlap_raw:.3f}%"
    )

    print(
        f"norm retained   : "
        f"{100*overlap_norm:.3f}%"
    )

    delta_adi = (
        adi_norm
        - adi_raw
    )

    percentile_report(
        "raw ADI",
        adi_raw[valid],
    )

    percentile_report(
        "normalized ADI",
        adi_norm[valid],
    )

    percentile_report(
        "normalized minus raw ADI",
        delta_adi[valid],
    )

    # ========================================================
    # 6. Outputs
    # ========================================================

    print()
    print(
        "[01b:6/6] Saving outputs..."
    )

    outdir = (
        Path(paths.output_dir)
        / "v09"
        / "step01b_ps_normalization_audit"
    )

    figdir = (
        outdir
        / "figures"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        outdir / "adi_raw.npy",
        adi_raw,
    )

    np.save(
        outdir / "adi_normalized.npy",
        adi_norm,
    )

    np.save(
        outdir / "ps_raw.npy",
        ps_raw,
    )

    np.save(
        outdir / "ps_normalized.npy",
        ps_norm,
    )

    np.save(
        outdir / "ps_changed.npy",
        changed,
    )

    np.save(
        outdir / "acquisition_median_scale.npy",
        scale_median,
    )

    # CSV acquisition statistics
    with open(
        outdir / "acquisition_amplitude_scale.csv",
        "w",
        newline="",
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "index",
            "date",
            "median_amplitude",
            "mean_amplitude",
            "normalization_factor",
        ])

        for t in range(T):
            w.writerow([
                t,
                dates[t],
                f"{scale_median[t]:.10f}",
                f"{scale_mean[t]:.10f}",
                f"{correction[t]:.10f}",
            ])

    # --------------------------------------------
    # Acquisition scale plot
    # --------------------------------------------

    fig, ax = plt.subplots(
        figsize=(13,5)
    )

    ax.plot(
        np.arange(T),
        scale_median,
        marker="o",
        label="Scene median amplitude",
    )

    ax.plot(
        np.arange(T),
        scale_mean,
        marker="o",
        label="Scene mean amplitude",
    )

    ax.axhline(
        reference_scale,
        linestyle="--",
        linewidth=1,
        label="Median reference scale",
    )

    ax.set_xlabel(
        "Acquisition index"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_title(
        "Step 01b - Per-acquisition amplitude scale"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figdir
        / "01b_acquisition_amplitude_scale.png",
        dpi=180,
    )

    plt.close(fig)

    # --------------------------------------------
    # ADI maps
    # --------------------------------------------

    adi_vmax = float(
        np.nanpercentile(
            np.concatenate([
                adi_raw[valid],
                adi_norm[valid],
            ]),
            99,
        )
    )

    save_map(
        adi_raw,
        figdir / "01b_adi_raw.png",
        "Step 01b - Raw amplitude dispersion index",
        "ADI",
        vmin=0,
        vmax=adi_vmax,
    )

    save_map(
        adi_norm,
        figdir / "01b_adi_normalized.png",
        (
            "Step 01b - ADI after "
            "per-acquisition amplitude normalization"
        ),
        "ADI",
        vmin=0,
        vmax=adi_vmax,
    )

    dlim = float(
        np.nanpercentile(
            np.abs(
                delta_adi[valid]
            ),
            99,
        )
    )

    if not np.isfinite(dlim) or dlim <= 0:
        dlim = 0.01

    save_map(
        delta_adi,
        figdir / "01b_delta_adi_norm_minus_raw.png",
        (
            "Step 01b - Normalized ADI minus raw ADI"
        ),
        "ΔADI",
        vmin=-dlim,
        vmax=dlim,
        cmap="coolwarm",
    )

    # --------------------------------------------
    # Difference-class map
    #
    # 0 = neither
    # 1 = both
    # 2 = raw only
    # 3 = normalized only
    # --------------------------------------------

    cls = np.zeros(
        (H,W),
        dtype=np.uint8,
    )

    cls[
        intersection
    ] = 1

    cls[
        raw_only
    ] = 2

    cls[
        norm_only
    ] = 3

    save_map(
        cls,
        figdir / "01b_ps_mask_comparison.png",
        (
            "Step 01b - PS mask comparison\n"
            "0=neither, 1=both, "
            "2=raw only, 3=normalized only"
        ),
        "Class",
        vmin=0,
        vmax=3,
    )

    # --------------------------------------------
    # Scatter
    # --------------------------------------------

    ids = np.flatnonzero(
        valid.ravel()
    )

    max_scatter = 100000

    if len(ids) > max_scatter:
        pick = np.linspace(
            0,
            len(ids)-1,
            max_scatter,
            dtype=np.int64,
        )
        ids = ids[pick]

    x = adi_raw.ravel()[ids]
    y = adi_norm.ravel()[ids]

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    fig, ax = plt.subplots(
        figsize=(7,7)
    )

    ax.scatter(
        x[good],
        y[good],
        s=2,
        alpha=0.12,
    )

    maxxy = float(
        np.nanpercentile(
            np.concatenate([
                x[good],
                y[good],
            ]),
            99,
        )
    )

    ax.plot(
        [0,maxxy],
        [0,maxxy],
        "--",
        linewidth=1,
    )

    ax.axvline(
        args.adi_max,
        linestyle=":",
        linewidth=1,
    )

    ax.axhline(
        args.adi_max,
        linestyle=":",
        linewidth=1,
    )

    ax.set_xlim(
        0,
        maxxy,
    )

    ax.set_ylim(
        0,
        maxxy,
    )

    ax.set_xlabel(
        "Raw ADI"
    )

    ax.set_ylabel(
        "Normalized ADI"
    )

    ax.set_title(
        "Step 01b - Raw vs normalized ADI"
    )

    fig.tight_layout()

    fig.savefig(
        figdir
        / "01b_raw_vs_normalized_adi.png",
        dpi=180,
    )

    plt.close(fig)

    # text summary
    with open(
        outdir / "summary.txt",
        "w",
    ) as f:
        f.write(
            f"strict_valid={Nvalid}\n"
        )
        f.write(
            f"adi_threshold={args.adi_max}\n"
        )
        f.write(
            f"raw_ps={n_raw}\n"
        )
        f.write(
            f"normalized_ps={n_norm}\n"
        )
        f.write(
            f"intersection={n_inter}\n"
        )
        f.write(
            f"union={n_union}\n"
        )
        f.write(
            f"raw_only={n_raw_only}\n"
        )
        f.write(
            f"normalized_only={n_norm_only}\n"
        )
        f.write(
            f"changed={n_changed}\n"
        )
        f.write(
            f"jaccard={jaccard:.10f}\n"
        )
        f.write(
            f"raw_retained={overlap_raw:.10f}\n"
        )
        f.write(
            f"norm_retained={overlap_norm:.10f}\n"
        )
        f.write(
            f"scale_min={scale_median.min():.10f}\n"
        )
        f.write(
            f"scale_max={scale_median.max():.10f}\n"
        )
        f.write(
            f"correction_min={correction.min():.10f}\n"
        )
        f.write(
            f"correction_max={correction.max():.10f}\n"
        )

    print()
    print(
        f"outputs         : {outdir}"
    )

    print()
    print(
        "STOP HERE."
    )

    print(
        "This is an audit only. "
        "Do not replace the production PS mask yet."
    )


if __name__ == "__main__":
    main()
