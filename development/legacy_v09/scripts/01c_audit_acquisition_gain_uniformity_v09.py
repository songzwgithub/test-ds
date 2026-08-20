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


def robust_mad(x, axis=None):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(
        np.abs(x - med),
        axis=axis,
    )
    return 1.4826 * mad


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Audit whether per-acquisition amplitude changes "
            "behave like a spatially uniform multiplicative gain."
        )
    )

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--tile-rows",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--tile-cols",
        type=int,
        default=200,
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

    dates = list(stack.dates)
    T = len(dates)

    print("=" * 80)
    print("Step 01c - Acquisition gain spatial-uniformity audit")
    print("=" * 80)

    print(f"config       : {config_path}")
    print(f"scene        : {H} x {W}")
    print(f"dates        : {T}")
    print(
        f"tiles        : "
        f"{args.tile_rows} x {args.tile_cols}"
    )

    # ==========================================================
    # 1. Read RSLC
    # ==========================================================

    print()
    print("[01c:1/7] Reading full RSLC stack...")

    z = stack.read_window(
        row0=row0,
        col0=col0,
        rows=H,
        cols=W,
    ).astype(
        np.complex64,
        copy=False,
    )

    finite = (
        np.isfinite(z.real)
        & np.isfinite(z.imag)
    )

    nonzero = ~(
        (z.real == 0)
        & (z.imag == 0)
    )

    strict_valid = np.all(
        finite & nonzero,
        axis=0,
    )

    Nvalid = int(strict_valid.sum())

    print(
        f"strict valid : {Nvalid}/{H*W} "
        f"({100*Nvalid/(H*W):.3f}%)"
    )

    amp = np.abs(z).astype(
        np.float32
    )

    del z

    # ==========================================================
    # 2. Pixel-centered log amplitudes
    # ==========================================================

    print()
    print(
        "[01c:2/7] Computing pixel-centered log amplitudes..."
    )

    # Only strict-valid pixels.
    logamp = np.log(
        np.maximum(
            amp,
            np.float32(1e-12),
        )
    ).astype(
        np.float32
    )

    pixel_temporal_med = np.median(
        logamp,
        axis=0,
    )

    residual = (
        logamp
        - pixel_temporal_med[None, :, :]
    )

    # Invalid pixels must never enter medians.
    residual[
        :,
        ~strict_valid
    ] = np.nan

    # ==========================================================
    # 3. Global two-way median date effect
    # ==========================================================

    print()
    print(
        "[01c:3/7] Estimating global acquisition effect..."
    )

    global_gain_log = np.nanmedian(
        residual.reshape(T, -1),
        axis=1,
    )

    # Remove arbitrary constant.
    global_gain_log -= np.median(
        global_gain_log
    )

    global_scale = np.exp(
        global_gain_log
    )

    global_correction = np.exp(
        -global_gain_log
    )

    print()
    print("Global multiplicative component:")

    for t in range(T):
        print(
            f"  {t:02d} {dates[t]} "
            f"log_gain={global_gain_log[t]:+9.5f} "
            f"scale={global_scale[t]:8.5f} "
            f"corr={global_correction[t]:8.5f}"
        )

    # ==========================================================
    # 4. Spatial-tile date effects
    # ==========================================================

    print()
    print(
        "[01c:4/7] Estimating acquisition effect by tile..."
    )

    tile_series = []
    tile_labels = []
    tile_centers = []
    tile_counts = []

    tile_id = 0

    for r0 in range(
        0,
        H,
        args.tile_rows,
    ):
        r1 = min(
            H,
            r0 + args.tile_rows,
        )

        for c0 in range(
            0,
            W,
            args.tile_cols,
        ):
            c1 = min(
                W,
                c0 + args.tile_cols,
            )

            vm = strict_valid[
                r0:r1,
                c0:c1,
            ]

            n = int(vm.sum())

            # Reject nearly empty tiles.
            if n < 1000:
                continue

            x = residual[
                :,
                r0:r1,
                c0:c1,
            ].reshape(
                T,
                -1,
            )

            gt = np.nanmedian(
                x,
                axis=1,
            )

            # Independent arbitrary constant per tile.
            gt -= np.median(gt)

            tile_series.append(
                gt.astype(np.float32)
            )

            tile_labels.append(
                f"T{tile_id:02d}"
            )

            tile_centers.append(
                (
                    (r0+r1-1)/2,
                    (c0+c1-1)/2,
                )
            )

            tile_counts.append(n)

            tile_id += 1

    tile_gain = np.stack(
        tile_series,
        axis=0,
    )
    # [Ntile, T]

    Ntile = tile_gain.shape[0]

    print(
        f"usable tiles : {Ntile}"
    )

    # ==========================================================
    # 5. Uniformity metrics
    # ==========================================================

    print()
    print(
        "[01c:5/7] Computing spatial-uniformity metrics..."
    )

    tile_median_by_date = np.median(
        tile_gain,
        axis=0,
    )

    tile_spread_by_date = robust_mad(
        tile_gain,
        axis=0,
    )

    p10_by_date = np.percentile(
        tile_gain,
        10,
        axis=0,
    )

    p90_by_date = np.percentile(
        tile_gain,
        90,
        axis=0,
    )

    # Correlation of every tile's temporal pattern
    # with the global temporal pattern.
    tile_corr = np.full(
        Ntile,
        np.nan,
        dtype=np.float64,
    )

    for k in range(Ntile):
        a = tile_gain[k]
        b = global_gain_log

        if (
            np.std(a) > 0
            and np.std(b) > 0
        ):
            tile_corr[k] = np.corrcoef(
                a,
                b,
            )[0,1]

    global_temporal_sigma = float(
        np.std(
            global_gain_log,
            ddof=0,
        )
    )

    median_spatial_sigma = float(
        np.median(
            tile_spread_by_date
        )
    )

    uniformity_ratio = (
        median_spatial_sigma
        / global_temporal_sigma
        if global_temporal_sigma > 0
        else np.nan
    )

    # Dates whose global shift is large enough
    # to make sign agreement meaningful.
    strong = (
        np.abs(global_gain_log)
        >= 0.03
    )

    sign_agreement = np.full(
        T,
        np.nan,
        dtype=np.float64,
    )

    for t in range(T):

        if not strong[t]:
            continue

        target_sign = np.sign(
            global_gain_log[t]
        )

        sign_agreement[t] = np.mean(
            np.sign(
                tile_gain[:, t]
            )
            == target_sign
        )

    print()
    print("=" * 80)
    print("Spatial-uniformity summary")
    print("=" * 80)

    print(
        f"global temporal sigma       : "
        f"{global_temporal_sigma:.6f}"
    )

    print(
        f"median spatial tile MAD     : "
        f"{median_spatial_sigma:.6f}"
    )

    print(
        f"spatial/global ratio        : "
        f"{uniformity_ratio:.6f}"
    )

    print(
        f"tile correlation median     : "
        f"{np.nanmedian(tile_corr):.6f}"
    )

    print(
        f"tile correlation p10        : "
        f"{np.nanpercentile(tile_corr,10):.6f}"
    )

    print(
        f"tile correlation minimum    : "
        f"{np.nanmin(tile_corr):.6f}"
    )

    if np.any(strong):

        print(
            f"strong-shift dates          : "
            f"{strong.sum()}/{T}"
        )

        print(
            f"sign agreement median       : "
            f"{np.nanmedian(sign_agreement[strong]):.6f}"
        )

        print(
            f"sign agreement minimum      : "
            f"{np.nanmin(sign_agreement[strong]):.6f}"
        )

    # ==========================================================
    # 6. Recompute PS using two-way gain correction
    # ==========================================================

    print()
    print(
        "[01c:6/7] Testing PS with two-way gain correction..."
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
    )

    adi_raw = np.full(
        (H,W),
        np.nan,
        dtype=np.float32,
    )

    ok = (
        strict_valid
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
            adi_raw <= args.adi_max
        )
    )

    # Apply one multiplicative correction per acquisition.
    for t in range(T):
        amp[t] *= np.float32(
            global_correction[t]
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
    )

    adi_gain = np.full(
        (H,W),
        np.nan,
        dtype=np.float32,
    )

    ok2 = (
        strict_valid
        & (norm_mean > 0)
    )

    adi_gain[ok2] = (
        norm_std[ok2]
        / norm_mean[ok2]
    ).astype(
        np.float32
    )

    ps_gain = (
        ok2
        & (
            adi_gain <= args.adi_max
        )
    )

    inter = (
        ps_raw & ps_gain
    )

    union = (
        ps_raw | ps_gain
    )

    raw_only = (
        ps_raw & ~ps_gain
    )

    gain_only = (
        ps_gain & ~ps_raw
    )

    n_raw = int(ps_raw.sum())
    n_gain = int(ps_gain.sum())
    n_inter = int(inter.sum())
    n_union = int(union.sum())

    jaccard = (
        n_inter/n_union
        if n_union
        else np.nan
    )

    print()
    print("Two-way correction PS comparison:")
    print(
        f"  raw PS          : {n_raw}"
    )
    print(
        f"  gain-corrected  : {n_gain}"
    )
    print(
        f"  intersection    : {n_inter}"
    )
    print(
        f"  raw only        : {raw_only.sum()}"
    )
    print(
        f"  corrected only  : {gain_only.sum()}"
    )
    print(
        f"  Jaccard         : {jaccard:.6f}"
    )
    print(
        f"  raw retained    : "
        f"{100*n_inter/n_raw:.3f}%"
    )

    # ==========================================================
    # 7. Save
    # ==========================================================

    print()
    print(
        "[01c:7/7] Saving outputs..."
    )

    outdir = (
        Path(paths.output_dir)
        / "v09"
        / "step01c_gain_uniformity"
    )

    figdir = outdir / "figures"

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        outdir / "global_gain_log.npy",
        global_gain_log,
    )

    np.save(
        outdir / "global_correction.npy",
        global_correction,
    )

    np.save(
        outdir / "tile_gain_log.npy",
        tile_gain,
    )

    np.save(
        outdir / "tile_correlation.npy",
        tile_corr,
    )

    np.save(
        outdir / "adi_gain_corrected.npy",
        adi_gain,
    )

    np.save(
        outdir / "ps_gain_corrected.npy",
        ps_gain,
    )

    # --------------------------------------------
    # CSV
    # --------------------------------------------

    with open(
        outdir / "date_gain_uniformity.csv",
        "w",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "index",
            "date",
            "global_log_gain",
            "global_scale",
            "correction",
            "tile_median_log_gain",
            "tile_mad",
            "tile_p10",
            "tile_p90",
            "sign_agreement",
        ])

        for t in range(T):
            w.writerow([
                t,
                dates[t],
                global_gain_log[t],
                global_scale[t],
                global_correction[t],
                tile_median_by_date[t],
                tile_spread_by_date[t],
                p10_by_date[t],
                p90_by_date[t],
                sign_agreement[t],
            ])

    # --------------------------------------------
    # Figure 1: all tile trajectories
    # --------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    x = np.arange(T)

    for k in range(Ntile):
        ax.plot(
            x,
            tile_gain[k],
            linewidth=0.6,
            alpha=0.28,
        )

    ax.plot(
        x,
        global_gain_log,
        linewidth=2.5,
        label="Global two-way median",
    )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.set_xlabel(
        "Acquisition index"
    )

    ax.set_ylabel(
        "Centered log-amplitude effect"
    )

    ax.set_title(
        "Step 01c - Acquisition effect across spatial tiles"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figdir / "01c_tile_gain_trajectories.png",
        dpi=180,
    )

    plt.close(fig)

    # --------------------------------------------
    # Figure 2: global + p10/p90
    # --------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    ax.fill_between(
        x,
        p10_by_date,
        p90_by_date,
        alpha=0.25,
        label="Tile p10-p90",
    )

    ax.plot(
        x,
        global_gain_log,
        marker="o",
        linewidth=2,
        label="Global effect",
    )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.set_xlabel(
        "Acquisition index"
    )

    ax.set_ylabel(
        "Centered log-amplitude effect"
    )

    ax.set_title(
        "Step 01c - Spatial consistency of acquisition-scale effect"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figdir / "01c_global_gain_spatial_envelope.png",
        dpi=180,
    )

    plt.close(fig)

    # --------------------------------------------
    # Figure 3: tile x date heatmap
    # --------------------------------------------

    vmax = float(
        np.percentile(
            np.abs(tile_gain),
            99,
        )
    )

    fig, ax = plt.subplots(
        figsize=(14,8)
    )

    im = ax.imshow(
        tile_gain,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xlabel(
        "Acquisition index"
    )

    ax.set_ylabel(
        "Spatial tile"
    )

    ax.set_title(
        "Step 01c - Tile-wise acquisition log-gain"
    )

    cb = fig.colorbar(
        im,
        ax=ax,
    )

    cb.set_label(
        "Centered log-amplitude effect"
    )

    fig.tight_layout()

    fig.savefig(
        figdir / "01c_tile_gain_heatmap.png",
        dpi=180,
    )

    plt.close(fig)

    # --------------------------------------------
    # Figure 4: tile correlation map
    # --------------------------------------------

    corr_map = np.full(
        (H,W),
        np.nan,
        dtype=np.float32,
    )

    k = 0

    for r0 in range(
        0,
        H,
        args.tile_rows,
    ):
        r1 = min(
            H,
            r0+args.tile_rows,
        )

        for c0 in range(
            0,
            W,
            args.tile_cols,
        ):
            c1 = min(
                W,
                c0+args.tile_cols,
            )

            vm = strict_valid[
                r0:r1,
                c0:c1,
            ]

            if int(vm.sum()) < 1000:
                continue

            corr_map[
                r0:r1,
                c0:c1
            ] = tile_corr[k]

            k += 1

    fig, ax = plt.subplots(
        figsize=(16,6)
    )

    im = ax.imshow(
        corr_map,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    ax.set_xlabel(
        "Range column"
    )

    ax.set_ylabel(
        "Azimuth row"
    )

    ax.set_title(
        "Step 01c - Tile correlation with global acquisition effect"
    )

    cb = fig.colorbar(
        im,
        ax=ax,
    )

    cb.set_label(
        "Temporal correlation"
    )

    fig.tight_layout()

    fig.savefig(
        figdir / "01c_tile_correlation_map.png",
        dpi=180,
    )

    plt.close(fig)

    with open(
        outdir / "summary.txt",
        "w",
    ) as f:

        f.write(
            f"global_temporal_sigma="
            f"{global_temporal_sigma:.10f}\n"
        )

        f.write(
            f"median_spatial_mad="
            f"{median_spatial_sigma:.10f}\n"
        )

        f.write(
            f"uniformity_ratio="
            f"{uniformity_ratio:.10f}\n"
        )

        f.write(
            f"tile_corr_median="
            f"{np.nanmedian(tile_corr):.10f}\n"
        )

        f.write(
            f"tile_corr_p10="
            f"{np.nanpercentile(tile_corr,10):.10f}\n"
        )

        f.write(
            f"tile_corr_min="
            f"{np.nanmin(tile_corr):.10f}\n"
        )

        if np.any(strong):
            f.write(
                f"sign_agreement_median="
                f"{np.nanmedian(sign_agreement[strong]):.10f}\n"
            )

            f.write(
                f"sign_agreement_min="
                f"{np.nanmin(sign_agreement[strong]):.10f}\n"
            )

        f.write(
            f"raw_ps={n_raw}\n"
        )

        f.write(
            f"gain_corrected_ps={n_gain}\n"
        )

        f.write(
            f"intersection={n_inter}\n"
        )

        f.write(
            f"raw_only={raw_only.sum()}\n"
        )

        f.write(
            f"corrected_only={gain_only.sum()}\n"
        )

        f.write(
            f"jaccard={jaccard:.10f}\n"
        )

    print()
    print(
        f"outputs      : {outdir}"
    )

    print()
    print("STOP HERE.")
    print(
        "Do not change production PS selection "
        "until the spatial-uniformity result is reviewed."
    )


if __name__ == "__main__":
    main()
