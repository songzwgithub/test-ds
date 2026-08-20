#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pypsds.phase_linking.coherence import compressed_coherence

from pypsds.phase_linking.emi import (
    ESTIMATOR_INVALID,
    image_pairs,
    robust_emi_threaded,
    temporal_coherence,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    v = np.percentile(
        x,
        [5, 25, 50, 75, 95],
    )

    return {
        "p05": float(v[0]),
        "p25": float(v[1]),
        "median": float(v[2]),
        "p75": float(v[3]),
        "p95": float(v[4]),
    }


def phase_agreement(a, b):
    good = (
        np.all(
            np.isfinite(a.real)
            & np.isfinite(a.imag),
            axis=1,
        )
        &
        np.all(
            np.isfinite(b.real)
            & np.isfinite(b.imag),
            axis=1,
        )
    )

    sim = np.full(
        a.shape[0],
        np.nan,
        dtype=np.float32,
    )

    diff = np.full(
        a.shape[0],
        np.nan,
        dtype=np.float32,
    )

    ids = np.flatnonzero(good)

    if ids.size:

        # Reference epoch 0 is identically aligned,
        # so assess only non-reference dates.
        delta = (
            a[ids, 1:]
            *
            np.conj(
                b[ids, 1:]
            )
        )

        sim[ids] = np.abs(
            np.mean(
                delta,
                axis=1,
            )
        ).astype(
            np.float32
        )

        diff[ids] = (
            np.median(
                np.abs(
                    np.angle(
                        delta
                    )
                ),
                axis=1,
            )
            *
            180.0
            /
            np.pi
        ).astype(
            np.float32
        )

    return sim, diff


def deterministic_sample(
    rows,
    cols,
    max_count,
):
    n = rows.size

    if (
        max_count <= 0
        or
        n <= max_count
    ):
        return rows, cols

    ids = np.linspace(
        0,
        n - 1,
        max_count,
        dtype=np.int64,
    )

    return (
        rows[ids],
        cols[ids],
    )


def solve_phase(
    coh,
    *,
    ndate,
    pairs,
    beta,
    gamma_jitter,
    emi_mu,
    workers,
    chunk,
):

    (
        phase,
        estimator,
        _,
        _,
        _,
    ) = robust_emi_threaded(
        coh,
        n_images=ndate,
        pairs=pairs,
        beta=beta,
        gamma_jitter=gamma_jitter,
        emi_mu=emi_mu,
        reference_idx=0,
        workers=workers,
        chunk_size=chunk,
    )

    return (
        phase,
        estimator,
    )


def audit_group(
    *,
    name,
    rows,
    cols,
    ctx,
    core_windows,
    yxt,
    pi,
    pj,
    pairs,
    ndate,
    alpha,
    batch,
    support_block,
    beta,
    gamma_jitter,
    emi_mu,
    workers,
    chunk,
):

    n = rows.size

    metrics = {
        "K_original":
            np.full(n, -1, np.int16),

        "K_core":
            np.full(n, -1, np.int16),

        "full_core_similarity":
            np.full(n, np.nan, np.float32),

        "full_core_difference_deg":
            np.full(n, np.nan, np.float32),

        "core_tc":
            np.full(n, np.nan, np.float32),

        "split_cross_tc":
            np.full(n, np.nan, np.float32),

        "split_similarity":
            np.full(n, np.nan, np.float32),

        "split_difference_deg":
            np.full(n, np.nan, np.float32),

        "valid":
            np.zeros(n, np.bool_),
    }

    print()
    print("-" * 100)
    print(
        f"{name}: {n:,} centers"
    )
    print("-" * 100)

    t0 = time.perf_counter()

    for start in range(
        0,
        n,
        batch,
    ):

        stop = min(
            n,
            start + batch,
        )

        br = rows[start:stop]
        bc = cols[start:stop]

        support, K0 = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=alpha,
                nslc=ndate,
                block_size=support_block,
            )
        )

        core_local = np.asarray(
            core_windows[
                br,
                bc,
            ],
            dtype=np.bool_,
        )

        support_core = (
            support
            &
            core_local
        )

        Kc = np.sum(
            support_core,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

        # ----------------------------------------------------
        # Original-support phase
        # ----------------------------------------------------

        coh_full = compressed_coherence(
            yxt,
            br,
            bc,
            support,
            pi,
            pj,
        )

        ph_full, est_full = solve_phase(
            coh_full,
            ndate=ndate,
            pairs=pairs,
            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,
            workers=workers,
            chunk=chunk,
        )

        # ----------------------------------------------------
        # Production K24-core phase
        # ----------------------------------------------------

        coh_core = compressed_coherence(
            yxt,
            br,
            bc,
            support_core,
            pi,
            pj,
        )

        ph_core, est_core = solve_phase(
            coh_core,
            ndate=ndate,
            pairs=pairs,
            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,
            workers=workers,
            chunk=chunk,
        )

        sim_fc, deg_fc = phase_agreement(
            ph_full,
            ph_core,
        )

        tc_core = temporal_coherence(
            coh_core,
            ph_core,
            pairs,
        )

        # ----------------------------------------------------
        # Independent stability WITHIN filtered core support.
        # ----------------------------------------------------

        B = stop - start

        flat = support_core.reshape(
            B,
            -1,
        )

        rank = np.cumsum(
            flat,
            axis=1,
            dtype=np.int16,
        )

        A = (
            flat
            &
            (
                rank % 2
                ==
                1
            )
        ).reshape(
            support_core.shape
        )

        Bmask = (
            flat
            &
            (
                rank % 2
                ==
                0
            )
        ).reshape(
            support_core.shape
        )

        cohA = compressed_coherence(
            yxt,
            br,
            bc,
            A,
            pi,
            pj,
        )

        cohB = compressed_coherence(
            yxt,
            br,
            bc,
            Bmask,
            pi,
            pj,
        )

        phA, estA = solve_phase(
            cohA,
            ndate=ndate,
            pairs=pairs,
            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,
            workers=workers,
            chunk=chunk,
        )

        phB, estB = solve_phase(
            cohB,
            ndate=ndate,
            pairs=pairs,
            beta=beta,
            gamma_jitter=gamma_jitter,
            emi_mu=emi_mu,
            workers=workers,
            chunk=chunk,
        )

        tcAB = temporal_coherence(
            cohB,
            phA,
            pairs,
        )

        tcBA = temporal_coherence(
            cohA,
            phB,
            pairs,
        )

        cross_tc = np.minimum(
            tcAB,
            tcBA,
        )

        sim_split, deg_split = (
            phase_agreement(
                phA,
                phB,
            )
        )

        ok = (
            (est_full != ESTIMATOR_INVALID)
            &
            (est_core != ESTIMATOR_INVALID)
            &
            (estA != ESTIMATOR_INVALID)
            &
            (estB != ESTIMATOR_INVALID)
            &
            np.isfinite(sim_fc)
            &
            np.isfinite(sim_split)
            &
            np.isfinite(cross_tc)
        )

        metrics[
            "K_original"
        ][start:stop] = K0

        metrics[
            "K_core"
        ][start:stop] = Kc

        metrics[
            "full_core_similarity"
        ][start:stop] = sim_fc

        metrics[
            "full_core_difference_deg"
        ][start:stop] = deg_fc

        metrics[
            "core_tc"
        ][start:stop] = tc_core

        metrics[
            "split_cross_tc"
        ][start:stop] = cross_tc

        metrics[
            "split_similarity"
        ][start:stop] = sim_split

        metrics[
            "split_difference_deg"
        ][start:stop] = deg_split

        metrics[
            "valid"
        ][start:stop] = ok

        del support
        del support_core
        del core_local
        del coh_full
        del coh_core
        del cohA
        del cohB
        del ph_full
        del ph_core
        del phA
        del phB

        if (
            stop == n
            or
            stop % (
                batch * 10
            ) == 0
        ):

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            rate = (
                stop / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"{name:<10s} "
                f"{stop:,}/{n:,} "
                f"({100*stop/n:6.2f}%) "
                f"rate={rate:,.0f} center/s"
            )

    return metrics


def summarize(data):

    good = data["valid"]

    n = good.size
    ng = int(
        np.count_nonzero(
            good
        )
    )

    return {
        "n":
            int(n),

        "valid":
            ng,

        "valid_fraction":
            (
                ng / n
                if n
                else 0.0
            ),

        "K_original":
            quantiles(
                data[
                    "K_original"
                ][good]
            ),

        "K_core":
            quantiles(
                data[
                    "K_core"
                ][good]
            ),

        "full_core_similarity":
            quantiles(
                data[
                    "full_core_similarity"
                ][good]
            ),

        "full_core_difference_deg":
            quantiles(
                data[
                    "full_core_difference_deg"
                ][good]
            ),

        "core_tc":
            quantiles(
                data[
                    "core_tc"
                ][good]
            ),

        "split_cross_tc":
            quantiles(
                data[
                    "split_cross_tc"
                ][good]
            ),

        "split_similarity":
            quantiles(
                data[
                    "split_similarity"
                ][good]
            ),

        "split_difference_deg":
            quantiles(
                data[
                    "split_difference_deg"
                ][good]
            ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--processing-dir",
        required=True,
    )

    ap.add_argument(
        "--sample-per-group",
        type=int,
        default=10000,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--half-row",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--half-col",
        type=int,
        default=11,
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=0.005,
    )

    ap.add_argument(
        "--beta",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--gamma-jitter",
        type=float,
        default=1e-6,
    )

    ap.add_argument(
        "--emi-mu",
        type=float,
        default=0.99,
    )

    ap.add_argument(
        "--pl-workers",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--pl-chunk",
        type=int,
        default=256,
    )

    args = ap.parse_args()

    processing = Path(
        args.processing_dir
    )

    seqdir = (
        processing
        /
        "sequential"
    )

    stats = (
        processing
        /
        "ds_statistics"
    )

    core_path = (
        seqdir
        /
        "compression_state_core_K24.npy"
    )

    scale_path = (
        stats
        /
        "rayleigh_scale2.npy"
    )

    raw_valid_path = (
        stats
        /
        "raw_valid.npy"
    )

    geom_path = (
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy"
    )

    ps_path = (
        processing
        /
        "ps_mask.npy"
    )

    yxt_path = (
        processing
        /
        "cache"
        /
        "phase_corrected_yxt.npy"
    )

    for p in (
        core_path,
        scale_path,
        raw_valid_path,
        geom_path,
        ps_path,
        yxt_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    core = np.load(
        core_path,
        mmap_mode="r",
    )

    scale2 = np.load(
        scale_path,
        mmap_mode="r",
    )

    raw_valid = np.load(
        raw_valid_path,
        mmap_mode="r",
    )

    geom = np.load(
        geom_path,
        mmap_mode="r",
    )

    ps = np.load(
        ps_path,
        mmap_mode="r",
    )

    yxt = np.load(
        yxt_path,
        mmap_mode="r",
    )

    H, W = core.shape

    if (
        yxt.ndim != 3
        or
        yxt.shape[:2] != (H, W)
    ):
        raise RuntimeError(
            f"phase cache shape mismatch: "
            f"{yxt.shape}"
        )

    ndate = int(
        yxt.shape[2]
    )

    valid = np.ascontiguousarray(
        np.asarray(
            raw_valid,
            dtype=np.bool_,
        )
        &
        np.asarray(
            geom,
            dtype=np.bool_,
        )
    )

    ps_bool = np.ascontiguousarray(
        np.asarray(
            ps,
            dtype=np.bool_,
        )
        &
        valid
    )

    core_bool = np.asarray(
        core,
        dtype=np.bool_,
    )

    # --------------------------------------------------------
    # Exact GLRT context
    # --------------------------------------------------------

    ctx = (
        prepare_glrt_window_context(
            scale2,
            valid,
            ps_bool,
            half_row=args.half_row,
            half_col=args.half_col,
        )
    )

    pad = (
        (args.half_row, args.half_row),
        (args.half_col, args.half_col),
    )

    core_pad = np.pad(
        core_bool,
        pad,
        mode="constant",
        constant_values=False,
    )

    core_windows = (
        np.lib.stride_tricks
        .sliding_window_view(
            core_pad,
            (
                2 * args.half_row + 1,
                2 * args.half_col + 1,
            ),
        )
    )

    # --------------------------------------------------------
    # First pass:
    # compute EFFECTIVE K under final K24 core.
    # --------------------------------------------------------

    rr_all, cc_all = np.where(
        core_bool
    )

    rr_all = rr_all.astype(
        np.int32,
        copy=False,
    )

    cc_all = cc_all.astype(
        np.int32,
        copy=False,
    )

    K_eff_map = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    print("=" * 100)
    print(
        "U3.2c6 K24 core-filtered phase stability audit"
    )
    print("=" * 100)

    print(
        "scene                  :",
        f"{H} x {W}",
    )

    print(
        "dates                  :",
        ndate,
    )

    print(
        "K24 core pixels        :",
        f"{rr_all.size:,}",
    )

    print()

    t0 = time.perf_counter()

    count_batch = 16000

    for start in range(
        0,
        rr_all.size,
        count_batch,
    ):

        stop = min(
            rr_all.size,
            start + count_batch,
        )

        br = rr_all[start:stop]
        bc = cc_all[start:stop]

        support, _ = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=args.alpha,
                nslc=ndate,
                block_size=1024,
            )
        )

        local_core = np.asarray(
            core_windows[
                br,
                bc,
            ],
            dtype=np.bool_,
        )

        Kc = np.sum(
            support
            &
            local_core,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

        K_eff_map[
            br,
            bc,
        ] = Kc

        del support
        del local_core

    count_elapsed = (
        time.perf_counter()
        -
        t0
    )

    if np.any(
        K_eff_map[
            core_bool
        ]
        <
        24
    ):
        raise RuntimeError(
            "K24 fixed-point parity failure: "
            "core contains K_eff < 24"
        )

    kmap_path = (
        seqdir
        /
        "compression_state_core_K24_effective_shp_count.npy"
    )

    np.save(
        kmap_path,
        K_eff_map,
    )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    group_masks = {
        "EDGE24_31":
            core_bool
            &
            (K_eff_map >= 24)
            &
            (K_eff_map <= 31),

        "MID32_47":
            core_bool
            &
            (K_eff_map >= 32)
            &
            (K_eff_map <= 47),

        "CONTROL48_63":
            core_bool
            &
            (K_eff_map >= 48)
            &
            (K_eff_map <= 63),
    }

    group_coords = {}

    print(
        "effective-K counting    :",
        f"{count_elapsed:.3f} s",
    )

    print()

    for name, mask in group_masks.items():

        rr, cc = np.where(
            mask
        )

        rr = rr.astype(
            np.int32,
            copy=False,
        )

        cc = cc.astype(
            np.int32,
            copy=False,
        )

        total = int(
            rr.size
        )

        rr, cc = deterministic_sample(
            rr,
            cc,
            args.sample_per_group,
        )

        group_coords[
            name
        ] = (
            rr,
            cc,
            total,
        )

        print(
            f"{name:<16s}: "
            f"population={total:,}, "
            f"audit={rr.size:,}"
        )

    # --------------------------------------------------------
    # Solver setup
    # --------------------------------------------------------

    pairs = image_pairs(
        ndate
    )

    pi = np.asarray(
        pairs[:, 0],
        dtype=np.int32,
    )

    pj = np.asarray(
        pairs[:, 1],
        dtype=np.int32,
    )

    summaries = {}
    raw = {}

    for name, (
        rr,
        cc,
        population,
    ) in group_coords.items():

        data = audit_group(
            name=name,
            rows=rr,
            cols=cc,
            ctx=ctx,
            core_windows=core_windows,
            yxt=yxt,
            pi=pi,
            pj=pj,
            pairs=pairs,
            ndate=ndate,
            alpha=args.alpha,
            batch=args.batch,
            support_block=args.support_block,
            beta=args.beta,
            gamma_jitter=args.gamma_jitter,
            emi_mu=args.emi_mu,
            workers=args.pl_workers,
            chunk=args.pl_chunk,
        )

        item = summarize(
            data
        )

        item[
            "population"
        ] = population

        summaries[
            name
        ] = item

        raw[
            name
        ] = data

    # --------------------------------------------------------
    # Save compact raw metrics
    # --------------------------------------------------------

    npz_dict = {}

    for name, data in raw.items():

        prefix = name.lower()

        for key, arr in data.items():

            npz_dict[
                f"{prefix}_{key}"
            ] = arr

    npz_path = (
        seqdir
        /
        "compression_k24_core_phase_stability_metrics.npz"
    )

    np.savez_compressed(
        npz_path,
        **npz_dict,
    )

    report = {
        "format":
            "pyPSDS-GAMMA-K24-core-phase-stability-v1",

        "shape":
            [H, W],

        "ndate":
            ndate,

        "state_min_shp":
            24,

        "core_pixels":
            int(
                np.count_nonzero(
                    core_bool
                )
            ),

        "groups":
            summaries,

        "decision":
            (
                "audit_only_no_automatic_production_approval"
            ),
    }

    json_path = (
        seqdir
        /
        "compression_k24_core_phase_stability.json"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 132)
    print(
        "U3.2c6 K24 CORE PHASE STABILITY SUMMARY"
    )
    print("=" * 132)

    print(
        "group          n    valid%   "
        "Keff50   full-core Sim50  full-core Diff50  "
        "split CrossTC50  split Sim50  split Diff50"
    )

    print("-" * 132)

    for name in (
        "EDGE24_31",
        "MID32_47",
        "CONTROL48_63",
    ):

        x = summaries[
            name
        ]

        print(
            f"{name:<14s} "
            f"{x['n']:6,d} "
            f"{100*x['valid_fraction']:8.3f} "
            f"{x['K_core']['median']:8.1f} "
            f"{x['full_core_similarity']['median']:16.4f} "
            f"{x['full_core_difference_deg']['median']:17.3f} "
            f"{x['split_cross_tc']['median']:16.4f} "
            f"{x['split_similarity']['median']:11.4f} "
            f"{x['split_difference_deg']['median']:12.3f}"
        )

    print()

    print(
        "effective K map:",
        kmap_path,
    )

    print(
        "metrics        :",
        npz_path,
    )

    print(
        "json           :",
        json_path,
    )

    print()

    print(
        "U3.2c6 K24 CORE-FILTERED PHASE STABILITY AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
