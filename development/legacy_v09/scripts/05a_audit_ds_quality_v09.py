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


EST_EVD = 0
EST_EMI = 1


TC_LEVELS = np.array(
    [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    dtype=float,
)

PAIR_LEVELS = np.array(
    [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    dtype=float,
)


def qreport(name, x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    print()
    print(f"=== {name} (N={len(x)}) ===")

    if len(x) == 0:
        return

    for p in (
        0, 1, 5, 10, 25,
        50, 75, 90, 95,
        99, 100
    ):
        print(
            f"p{p:03d}: "
            f"{np.percentile(x, p):.6f}"
        )


def make_hexbin(
    x,
    y,
    path,
    *,
    xlabel,
    ylabel,
    title,
    xlim=None,
    ylim=None,
    gridsize=100,
):
    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = np.asarray(x)[good]
    y = np.asarray(y)[good]

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    hb = ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        bins="log",
        mincnt=1,
    )

    cb = fig.colorbar(
        hb,
        ax=ax,
    )

    cb.set_label(
        "log10(count)"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(
        alpha=0.15
    )

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
            "Audit full-scene DS quality relationships "
            "before choosing final thresholds."
        )
    )

    ap.add_argument(
        "--config",
        required=True,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (_, _, H, W),
    ) = open_from_config(
        args.config
    )

    d = (
        Path(paths.output_dir)
        / "v09"
    )

    outdir = (
        d
        / "step05a_ds_quality_audit"
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

    print("=" * 80)
    print(
        "Step 05a - Full-scene DS quality audit"
    )
    print("=" * 80)

    print(
        f"config       : {config_path}"
    )

    print(
        f"scene        : {H} x {W}"
    )

    print(
        f"v09 outputs  : {d}"
    )

    # ==========================================================
    # 1. Load quality layers
    # ==========================================================

    print()
    print(
        "[05a:1/6] Loading v0.9 quality layers..."
    )

    ps = np.load(
        d / "ps_mask.npy",
        mmap_mode="r",
    )

    pl = np.load(
        d / "pl_valid.npy",
        mmap_mode="r",
    )

    tc = np.load(
        d / "temporal_coherence.npy",
        mmap_mode="r",
    )

    pc = np.load(
        d / "median_pair_coherence.npy",
        mmap_mode="r",
    )

    K = np.load(
        d / "shp_count.npy",
        mmap_mode="r",
    )

    est = np.load(
        d / "estimator_code.npy",
        mmap_mode="r",
    )

    emi_eig = np.load(
        d / "emi_eigenvalue.npy",
        mmap_mode="r",
    )

    evd_eig = np.load(
        d / "evd_eigenvalue.npy",
        mmap_mode="r",
    )

    gamma_min = np.load(
        d / "gamma_min_eigenvalue.npy",
        mmap_mode="r",
    )

    ds = (
        np.asarray(pl)
        & ~np.asarray(ps)
        & np.isfinite(
            np.asarray(tc)
        )
        & np.isfinite(
            np.asarray(pc)
        )
        & (
            np.asarray(K) >= 0
        )
    )

    rr, cc = np.where(ds)

    tcv = np.asarray(tc)[ds].astype(
        np.float64
    )

    pcv = np.asarray(pc)[ds].astype(
        np.float64
    )

    kv = np.asarray(K)[ds].astype(
        np.float64
    )

    estv = np.asarray(est)[ds]

    emiv = np.asarray(emi_eig)[ds].astype(
        np.float64
    )

    evdv = np.asarray(evd_eig)[ds].astype(
        np.float64
    )

    gminv = np.asarray(gamma_min)[ds].astype(
        np.float64
    )

    N = len(tcv)

    emi = (
        estv == EST_EMI
    )

    evd = (
        estv == EST_EVD
    )

    print(
        f"linked DS     : {N}"
    )

    print(
        f"EMI           : {emi.sum()} "
        f"({100*emi.mean():.3f}%)"
    )

    print(
        f"EVD           : {evd.sum()} "
        f"({100*evd.mean():.3f}%)"
    )

    # ==========================================================
    # 2. Marginal distributions
    # ==========================================================

    print()
    print(
        "[05a:2/6] Marginal distributions..."
    )

    qreport(
        "Temporal coherence",
        tcv,
    )

    qreport(
        "Median pair coherence",
        pcv,
    )

    qreport(
        "GLRT support K",
        kv,
    )

    qreport(
        "EMI eigenvalue",
        emiv[emi],
    )

    qreport(
        "EVD eigenvalue",
        evdv[evd],
    )

    qreport(
        "Gamma minimum eigenvalue",
        gminv,
    )

    # ==========================================================
    # 3. Correlations
    # ==========================================================

    print()
    print(
        "[05a:3/6] Quality correlations..."
    )

    # Pearson is useful here only as a descriptive number.
    # We also compute rank correlations with ranks generated by
    # stable argsort to avoid requiring pandas.
    def rankdata_simple(x):
        order = np.argsort(
            x,
            kind="mergesort",
        )

        ranks = np.empty(
            len(x),
            dtype=np.float64,
        )

        ranks[order] = np.arange(
            len(x),
            dtype=np.float64,
        )

        return ranks

    def corr(a, b):
        m = (
            np.isfinite(a)
            & np.isfinite(b)
        )

        if m.sum() < 2:
            return np.nan

        return float(
            np.corrcoef(
                a[m],
                b[m],
            )[0, 1]
        )

    pear_tc_pc = corr(
        tcv,
        pcv,
    )

    pear_tc_k = corr(
        tcv,
        kv,
    )

    pear_pc_k = corr(
        pcv,
        kv,
    )

    rtc = rankdata_simple(tcv)
    rpc = rankdata_simple(pcv)
    rk = rankdata_simple(kv)

    spear_tc_pc = corr(
        rtc,
        rpc,
    )

    spear_tc_k = corr(
        rtc,
        rk,
    )

    spear_pc_k = corr(
        rpc,
        rk,
    )

    print(
        f"Pearson TC vs pair     : "
        f"{pear_tc_pc:+.6f}"
    )

    print(
        f"Pearson TC vs K        : "
        f"{pear_tc_k:+.6f}"
    )

    print(
        f"Pearson pair vs K      : "
        f"{pear_pc_k:+.6f}"
    )

    print(
        f"Rank corr TC vs pair   : "
        f"{spear_tc_pc:+.6f}"
    )

    print(
        f"Rank corr TC vs K      : "
        f"{spear_tc_k:+.6f}"
    )

    print(
        f"Rank corr pair vs K    : "
        f"{spear_pc_k:+.6f}"
    )

    # ==========================================================
    # 4. Joint threshold grid
    # ==========================================================

    print()
    print(
        "[05a:4/6] TC x pair-coherence threshold grid..."
    )

    rows = []

    count_grid = np.zeros(
        (
            len(TC_LEVELS),
            len(PAIR_LEVELS),
        ),
        dtype=np.int64,
    )

    evd_grid = np.zeros_like(
        count_grid,
    )

    median_k_grid = np.full(
        count_grid.shape,
        np.nan,
        dtype=np.float64,
    )

    for i, tmin in enumerate(
        TC_LEVELS
    ):

        for j, pmin in enumerate(
            PAIR_LEVELS
        ):

            m = (
                (tcv >= tmin)
                & (pcv >= pmin)
            )

            n = int(m.sum())

            nevd = int(
                np.sum(
                    m & evd
                )
            )

            nemi = int(
                np.sum(
                    m & emi
                )
            )

            medk = (
                float(
                    np.median(
                        kv[m]
                    )
                )
                if n
                else np.nan
            )

            count_grid[i, j] = n
            evd_grid[i, j] = nevd
            median_k_grid[i, j] = medk

            rows.append({
                "tc_min": tmin,
                "pair_min": pmin,
                "count": n,
                "linked_fraction": (
                    n / N
                    if N
                    else np.nan
                ),
                "scene_fraction": (
                    n / (H * W)
                ),
                "emi_count": nemi,
                "evd_count": nevd,
                "evd_fraction": (
                    nevd / n
                    if n
                    else np.nan
                ),
                "median_K": medk,
            })

    with open(
        outdir
        / "tc_pair_threshold_grid.csv",
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(rows)

    print()
    print(
        "Selected DS counts:"
    )

    header = (
        "TC\\pair "
        + " ".join(
            f"{p:>9.2f}"
            for p in PAIR_LEVELS
        )
    )

    print(header)

    for i, t in enumerate(
        TC_LEVELS
    ):

        vals = " ".join(
            f"{count_grid[i,j]:9d}"
            for j in range(
                len(PAIR_LEVELS)
            )
        )

        print(
            f"{t:7.2f} {vals}"
        )

    # ==========================================================
    # 5. Estimator-conditioned quality
    # ==========================================================

    print()
    print(
        "[05a:5/6] EMI/EVD-conditioned statistics..."
    )

    for name, m in (
        ("EMI", emi),
        ("EVD", evd),
    ):

        print()
        print(
            f"--- {name} N={m.sum()} ---"
        )

        print(
            f"TC median       : "
            f"{np.median(tcv[m]):.6f}"
        )

        print(
            f"pair median     : "
            f"{np.median(pcv[m]):.6f}"
        )

        print(
            f"K median        : "
            f"{np.median(kv[m]):.1f}"
        )

        for th in (
            0.5,
            0.7,
            0.8,
            0.9,
        ):

            print(
                f"TC >= {th:.1f}       : "
                f"{np.sum(m & (tcv >= th))}"
            )

    # ==========================================================
    # 6. Figures
    # ==========================================================

    print()
    print(
        "[05a:6/6] Saving figures..."
    )

    make_hexbin(
        pcv,
        tcv,
        figdir
        / "05a_tc_vs_pair_coherence.png",
        xlabel="Median pair coherence",
        ylabel="Temporal coherence",
        title=(
            "Step 05a - DS temporal coherence "
            "vs median pair coherence"
        ),
        xlim=(0, 1),
        ylim=(0, 1),
    )

    make_hexbin(
        kv,
        tcv,
        figdir
        / "05a_tc_vs_glrt_K.png",
        xlabel="GLRT covariance support K",
        ylabel="Temporal coherence",
        title=(
            "Step 05a - DS temporal coherence "
            "vs GLRT support"
        ),
        ylim=(0, 1),
    )

    make_hexbin(
        kv,
        pcv,
        figdir
        / "05a_pair_coherence_vs_glrt_K.png",
        xlabel="GLRT covariance support K",
        ylabel="Median pair coherence",
        title=(
            "Step 05a - Median pair coherence "
            "vs GLRT support"
        ),
        ylim=(0, 1),
    )

    # ----------------------------------------------------------
    # Threshold count heatmap
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    pct = (
        100.0
        * count_grid
        / N
    )

    im = ax.imshow(
        pct,
        origin="lower",
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(PAIR_LEVELS)
        )
    )

    ax.set_xticklabels(
        [
            f"{x:.2f}"
            for x in PAIR_LEVELS
        ]
    )

    ax.set_yticks(
        np.arange(
            len(TC_LEVELS)
        )
    )

    ax.set_yticklabels(
        [
            f"{x:.2f}"
            for x in TC_LEVELS
        ]
    )

    ax.set_xlabel(
        "Minimum median pair coherence"
    )

    ax.set_ylabel(
        "Minimum temporal coherence"
    )

    ax.set_title(
        "Step 05a - Retained phase-linked DS (%)"
    )

    cb = fig.colorbar(
        im,
        ax=ax,
    )

    cb.set_label(
        "% of phase-linked DS"
    )

    for i in range(
        len(TC_LEVELS)
    ):

        for j in range(
            len(PAIR_LEVELS)
        ):

            ax.text(
                j,
                i,
                f"{pct[i,j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        figdir
        / "05a_threshold_retention_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ----------------------------------------------------------
    # TC hist EMI vs EVD
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bins = np.linspace(
        0,
        1,
        101,
    )

    ax.hist(
        tcv[emi],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.7,
        label=f"EMI (N={emi.sum()})",
    )

    ax.hist(
        tcv[evd],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.7,
        label=f"EVD fallback (N={evd.sum()})",
    )

    ax.set_xlabel(
        "Temporal coherence"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Step 05a - TC distribution by PL estimator"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figdir
        / "05a_tc_emi_vs_evd.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ----------------------------------------------------------
    # EVD fallback spatial map
    # ----------------------------------------------------------

    evd_map = np.full(
        (H, W),
        np.nan,
        dtype=np.float32,
    )

    evd_map[
        rr,
        cc
    ] = 0.0

    evd_map[
        rr[evd],
        cc[evd]
    ] = 1.0

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    im = ax.imshow(
        evd_map,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel(
        "Range column"
    )

    ax.set_ylabel(
        "Azimuth row"
    )

    ax.set_title(
        "Step 05a - PL estimator on linked DS "
        "(0=EMI, 1=EVD fallback)"
    )

    cb = fig.colorbar(
        im,
        ax=ax,
    )

    cb.set_label(
        "EVD fallback"
    )

    fig.tight_layout()

    fig.savefig(
        figdir
        / "05a_evd_fallback_map.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    with open(
        outdir
        / "summary.txt",
        "w",
    ) as f:

        f.write(
            f"linked_ds={N}\n"
        )

        f.write(
            f"emi={emi.sum()}\n"
        )

        f.write(
            f"evd={evd.sum()}\n"
        )

        f.write(
            f"tc_median={np.median(tcv):.10f}\n"
        )

        f.write(
            f"pair_median={np.median(pcv):.10f}\n"
        )

        f.write(
            f"K_median={np.median(kv):.10f}\n"
        )

        f.write(
            f"pearson_tc_pair={pear_tc_pc:.10f}\n"
        )

        f.write(
            f"pearson_tc_K={pear_tc_k:.10f}\n"
        )

        f.write(
            f"pearson_pair_K={pear_pc_k:.10f}\n"
        )

        f.write(
            f"rank_tc_pair={spear_tc_pc:.10f}\n"
        )

        f.write(
            f"rank_tc_K={spear_tc_k:.10f}\n"
        )

        f.write(
            f"rank_pair_K={spear_pc_k:.10f}\n"
        )

    print()
    print(
        f"outputs       : {outdir}"
    )

    print()
    print("STOP HERE.")

    print(
        "Do not run 05_select_ds_v09.py yet. "
        "Review the joint quality space first."
    )


if __name__ == "__main__":
    main()
