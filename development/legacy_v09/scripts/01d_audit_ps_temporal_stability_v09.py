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


def strict_valid_mask(z):
    finite = (
        np.isfinite(z.real)
        & np.isfinite(z.imag)
    )
    nonzero = ~(
        (z.real == 0)
        & (z.imag == 0)
    )
    return np.all(
        finite & nonzero,
        axis=0,
    )


def raw_adi(amp, valid):
    mean = np.mean(
        amp,
        axis=0,
        dtype=np.float64,
    )
    std = np.std(
        amp,
        axis=0,
        dtype=np.float64,
        ddof=0,
    )

    out = np.full(
        mean.shape,
        np.nan,
        dtype=np.float32,
    )

    ok = (
        valid
        & np.isfinite(mean)
        & (mean > 0)
    )

    out[ok] = (
        std[ok] / mean[ok]
    ).astype(np.float32)

    return out


def estimate_two_way_gain(amp, valid):
    """
    Estimate one multiplicative common-mode gain per acquisition.

    log(A_tp) = pixel_effect_p + date_effect_t + residual_tp

    Robust approximation:
      pixel_effect = median_t(log A)
      date_effect  = median_p(log A - pixel_effect)
    """
    logamp = np.log(
        np.maximum(
            amp,
            np.float32(1e-12),
        )
    ).astype(np.float32)

    pix_med = np.median(
        logamp,
        axis=0,
    )

    residual = (
        logamp
        - pix_med[None, :, :]
    )

    residual[:, ~valid] = np.nan

    gain_log = np.nanmedian(
        residual.reshape(
            residual.shape[0],
            -1,
        ),
        axis=1,
    )

    # Arbitrary constant has no effect on ADI.
    gain_log -= np.median(
        gain_log
    )

    correction = np.exp(
        -gain_log
    ).astype(np.float32)

    return gain_log, correction


def corrected_adi(
    amp,
    valid,
):
    gain_log, correction = (
        estimate_two_way_gain(
            amp,
            valid,
        )
    )

    # Avoid copying the entire T,H,W array.
    #
    # Compute moments after scale correction:
    #
    # mean(A*c)
    # mean((A*c)^2)
    T = amp.shape[0]

    s1 = np.zeros(
        valid.shape,
        dtype=np.float64,
    )

    s2 = np.zeros(
        valid.shape,
        dtype=np.float64,
    )

    for t in range(T):
        a = (
            amp[t].astype(
                np.float64,
                copy=False,
            )
            * float(
                correction[t]
            )
        )

        s1 += a
        s2 += a * a

    mean = s1 / T

    var = (
        s2 / T
        - mean * mean
    )

    var = np.maximum(
        var,
        0.0,
    )

    std = np.sqrt(var)

    adi = np.full(
        valid.shape,
        np.nan,
        dtype=np.float32,
    )

    ok = (
        valid
        & np.isfinite(mean)
        & (mean > 0)
    )

    adi[ok] = (
        std[ok]
        / mean[ok]
    ).astype(np.float32)

    return (
        adi,
        gain_log,
        correction,
    )


def mask_metrics(
    reference,
    test,
):
    inter = reference & test
    union = reference | test

    nr = int(reference.sum())
    nt = int(test.sum())
    ni = int(inter.sum())
    nu = int(union.sum())

    jaccard = (
        ni / nu
        if nu
        else np.nan
    )

    reference_retained = (
        ni / nr
        if nr
        else np.nan
    )

    test_precision = (
        ni / nt
        if nt
        else np.nan
    )

    return {
        "reference": nr,
        "test": nt,
        "intersection": ni,
        "union": nu,
        "jaccard": jaccard,
        "reference_retained": reference_retained,
        "test_precision": test_precision,
    }


def evaluate_subset(
    amp_full,
    full_valid,
    ids,
    adi_max,
):
    amp = amp_full[
        ids
    ]

    # Use the SAME spatial support as the 38-date
    # strict-valid mask. This prevents changing nodata
    # coverage from contaminating the comparison.
    valid = full_valid

    ar = raw_adi(
        amp,
        valid,
    )

    ag, gain_log, correction = (
        corrected_adi(
            amp,
            valid,
        )
    )

    ps_r = (
        valid
        & np.isfinite(ar)
        & (ar <= adi_max)
    )

    ps_g = (
        valid
        & np.isfinite(ag)
        & (ag <= adi_max)
    )

    return (
        ps_r,
        ps_g,
        ar,
        ag,
        gain_log,
        correction,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--adi-max",
        type=float,
        default=0.25,
    )

    ap.add_argument(
        "--random-tests",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--fraction",
        type=float,
        default=0.75,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260816,
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
    print(
        "Step 01d - PS temporal-subset stability audit"
    )
    print("=" * 80)

    print(f"config          : {config_path}")
    print(f"scene           : {H} x {W}")
    print(f"dates           : {T}")
    print(f"ADI threshold   : {args.adi_max}")
    print(
        f"random tests    : "
        f"{args.random_tests}"
    )
    print(
        f"subset fraction : "
        f"{args.fraction:.3f}"
    )

    # =========================================================
    # 1. Load full stack
    # =========================================================

    print()
    print(
        "[01d:1/5] Reading complete RSLC stack..."
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

    valid = strict_valid_mask(z)

    print(
        f"strict valid     : "
        f"{valid.sum()}/{H*W} "
        f"({100*valid.mean():.3f}%)"
    )

    amp = np.abs(z).astype(
        np.float32
    )

    del z

    # =========================================================
    # 2. Full-reference masks
    # =========================================================

    print()
    print(
        "[01d:2/5] Computing full-stack PS references..."
    )

    adi_raw_full = raw_adi(
        amp,
        valid,
    )

    (
        adi_gain_full,
        full_gain_log,
        full_correction,
    ) = corrected_adi(
        amp,
        valid,
    )

    ps_raw_full = (
        valid
        & (
            adi_raw_full
            <= args.adi_max
        )
    )

    ps_gain_full = (
        valid
        & (
            adi_gain_full
            <= args.adi_max
        )
    )

    print(
        f"full raw PS      : "
        f"{ps_raw_full.sum()}"
    )

    print(
        f"full gain PS     : "
        f"{ps_gain_full.sum()}"
    )

    # =========================================================
    # 3. Build temporal subsets
    # =========================================================

    print()
    print(
        "[01d:3/5] Building temporal subsets..."
    )

    subsets = []

    subsets.append(
        (
            "odd",
            np.arange(
                0,
                T,
                2,
                dtype=np.int64,
            ),
        )
    )

    subsets.append(
        (
            "even",
            np.arange(
                1,
                T,
                2,
                dtype=np.int64,
            ),
        )
    )

    subsets.append(
        (
            "first_half",
            np.arange(
                0,
                T // 2,
                dtype=np.int64,
            ),
        )
    )

    subsets.append(
        (
            "last_half",
            np.arange(
                T // 2,
                T,
                dtype=np.int64,
            ),
        )
    )

    rng = np.random.default_rng(
        args.seed
    )

    nsub = int(
        round(
            args.fraction
            * T
        )
    )

    nsub = max(
        3,
        min(
            T - 1,
            nsub,
        ),
    )

    for i in range(
        args.random_tests
    ):
        ids = np.sort(
            rng.choice(
                T,
                size=nsub,
                replace=False,
            )
        )

        subsets.append(
            (
                f"random_{i+1:02d}",
                ids,
            )
        )

    print(
        f"subset count     : "
        f"{len(subsets)}"
    )

    # =========================================================
    # 4. Evaluate
    # =========================================================

    print()
    print(
        "[01d:4/5] Running temporal stability tests..."
    )

    records = []

    for k, (
        name,
        ids,
    ) in enumerate(
        subsets,
        start=1,
    ):

        (
            ps_r,
            ps_g,
            ar,
            ag,
            gain_log,
            correction,
        ) = evaluate_subset(
            amp,
            valid,
            ids,
            args.adi_max,
        )

        mr = mask_metrics(
            ps_raw_full,
            ps_r,
        )

        mg = mask_metrics(
            ps_gain_full,
            ps_g,
        )

        records.append({
            "name": name,
            "n_dates": len(ids),
            "raw_ps": int(ps_r.sum()),
            "gain_ps": int(ps_g.sum()),
            "raw_jaccard": mr["jaccard"],
            "gain_jaccard": mg["jaccard"],
            "delta_jaccard": (
                mg["jaccard"]
                - mr["jaccard"]
            ),
            "raw_retained": (
                mr["reference_retained"]
            ),
            "gain_retained": (
                mg["reference_retained"]
            ),
            "raw_precision": (
                mr["test_precision"]
            ),
            "gain_precision": (
                mg["test_precision"]
            ),
            "indices": ",".join(
                str(int(x))
                for x in ids
            ),
        })

        print(
            f"{k:02d}/{len(subsets):02d} "
            f"{name:12s} "
            f"Ndate={len(ids):2d} | "
            f"raw J={mr['jaccard']:.4f} "
            f"gain J={mg['jaccard']:.4f} "
            f"Δ={mg['jaccard']-mr['jaccard']:+.4f}"
        )

    raw_j = np.array(
        [
            r["raw_jaccard"]
            for r in records
        ]
    )

    gain_j = np.array(
        [
            r["gain_jaccard"]
            for r in records
        ]
    )

    delta_j = (
        gain_j
        - raw_j
    )

    raw_ret = np.array(
        [
            r["raw_retained"]
            for r in records
        ]
    )

    gain_ret = np.array(
        [
            r["gain_retained"]
            for r in records
        ]
    )

    # =========================================================
    # 5. Report/save
    # =========================================================

    print()
    print(
        "[01d:5/5] Final stability report"
    )

    print()
    print("=" * 80)
    print(
        "Temporal stability summary"
    )
    print("=" * 80)

    print(
        f"raw Jaccard median       : "
        f"{np.median(raw_j):.6f}"
    )

    print(
        f"gain Jaccard median      : "
        f"{np.median(gain_j):.6f}"
    )

    print(
        f"median ΔJ               : "
        f"{np.median(delta_j):+.6f}"
    )

    print(
        f"mean ΔJ                 : "
        f"{np.mean(delta_j):+.6f}"
    )

    print(
        f"gain better tests       : "
        f"{np.sum(delta_j > 0)}/{len(delta_j)}"
    )

    print(
        f"raw better tests        : "
        f"{np.sum(delta_j < 0)}/{len(delta_j)}"
    )

    print(
        f"equal tests             : "
        f"{np.sum(delta_j == 0)}/{len(delta_j)}"
    )

    print()
    print(
        f"raw retention median    : "
        f"{np.median(raw_ret):.6f}"
    )

    print(
        f"gain retention median   : "
        f"{np.median(gain_ret):.6f}"
    )

    # Random tests separately
    random_mask = np.array([
        r["name"].startswith(
            "random_"
        )
        for r in records
    ])

    if np.any(
        random_mask
    ):

        print()
        print(
            "Random 75% subsets:"
        )

        print(
            f"  raw J median          : "
            f"{np.median(raw_j[random_mask]):.6f}"
        )

        print(
            f"  gain J median         : "
            f"{np.median(gain_j[random_mask]):.6f}"
        )

        print(
            f"  median ΔJ             : "
            f"{np.median(delta_j[random_mask]):+.6f}"
        )

        print(
            f"  gain better           : "
            f"{np.sum(delta_j[random_mask] > 0)}"
            f"/{random_mask.sum()}"
        )

    outdir = (
        Path(paths.output_dir)
        / "v09"
        / "step01d_ps_temporal_stability"
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

    with open(
        outdir / "temporal_stability.csv",
        "w",
        newline="",
    ) as f:

        fields = list(
            records[0].keys()
        )

        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()
        w.writerows(records)

    np.save(
        outdir / "ps_raw_full.npy",
        ps_raw_full,
    )

    np.save(
        outdir / "ps_gain_full.npy",
        ps_gain_full,
    )

    np.save(
        outdir / "full_gain_log.npy",
        full_gain_log,
    )

    # ---------------------------------------------------------
    # Figure 1 - per-test Jaccard
    # ---------------------------------------------------------

    x = np.arange(
        len(records)
    )

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    ax.plot(
        x,
        raw_j,
        marker="o",
        label="Raw ADI",
    )

    ax.plot(
        x,
        gain_j,
        marker="o",
        label="Gain-corrected ADI",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            r["name"]
            for r in records
        ],
        rotation=60,
        ha="right",
    )

    ax.set_ylabel(
        "Jaccard vs full-stack PS"
    )

    ax.set_title(
        "Step 01d - Temporal subset PS stability"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figdir
        / "01d_jaccard_by_temporal_subset.png",
        dpi=180,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 2 - delta Jaccard
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    ax.bar(
        x,
        delta_j,
    )

    ax.axhline(
        0,
        linewidth=1,
        linestyle="--",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            r["name"]
            for r in records
        ],
        rotation=60,
        ha="right",
    )

    ax.set_ylabel(
        "Gain Jaccard - Raw Jaccard"
    )

    ax.set_title(
        "Step 01d - Stability improvement from common-gain correction"
    )

    fig.tight_layout()

    fig.savefig(
        figdir
        / "01d_delta_jaccard.png",
        dpi=180,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 3 - raw vs gain J
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7,7)
    )

    ax.scatter(
        raw_j,
        gain_j,
        s=45,
    )

    lo = float(
        min(
            raw_j.min(),
            gain_j.min(),
        )
    )

    hi = float(
        max(
            raw_j.max(),
            gain_j.max(),
        )
    )

    pad = 0.02

    ax.plot(
        [lo-pad, hi+pad],
        [lo-pad, hi+pad],
        "--",
        linewidth=1,
    )

    ax.set_xlim(
        lo-pad,
        hi+pad,
    )

    ax.set_ylim(
        lo-pad,
        hi+pad,
    )

    ax.set_xlabel(
        "Raw ADI PS Jaccard"
    )

    ax.set_ylabel(
        "Gain-corrected ADI PS Jaccard"
    )

    ax.set_title(
        "Step 01d - Raw vs gain-corrected PS stability"
    )

    fig.tight_layout()

    fig.savefig(
        figdir
        / "01d_raw_vs_gain_jaccard.png",
        dpi=180,
    )

    plt.close(fig)

    with open(
        outdir / "summary.txt",
        "w",
    ) as f:

        f.write(
            f"raw_full_ps={ps_raw_full.sum()}\n"
        )

        f.write(
            f"gain_full_ps={ps_gain_full.sum()}\n"
        )

        f.write(
            f"raw_jaccard_median={np.median(raw_j):.10f}\n"
        )

        f.write(
            f"gain_jaccard_median={np.median(gain_j):.10f}\n"
        )

        f.write(
            f"median_delta_jaccard={np.median(delta_j):.10f}\n"
        )

        f.write(
            f"mean_delta_jaccard={np.mean(delta_j):.10f}\n"
        )

        f.write(
            f"gain_better_tests={np.sum(delta_j>0)}\n"
        )

        f.write(
            f"raw_better_tests={np.sum(delta_j<0)}\n"
        )

        f.write(
            f"total_tests={len(delta_j)}\n"
        )

        f.write(
            f"raw_retention_median={np.median(raw_ret):.10f}\n"
        )

        f.write(
            f"gain_retention_median={np.median(gain_ret):.10f}\n"
        )

    print()
    print(
        f"outputs                 : {outdir}"
    )

    print()
    print("STOP HERE.")
    print(
        "Use this test to decide whether the production "
        "PS detector should use raw or common-gain-corrected ADI."
    )


if __name__ == "__main__":
    main()
