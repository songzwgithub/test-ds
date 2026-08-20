#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config

from pypsds.phase_linking.coherence import (
    compressed_coherence,
)

from pypsds.phase_linking.emi import (
    image_pairs,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)

from pypsds.phase_linking.streaming_quality import (
    temporal_quality_streaming,
)


def bool_windows(mask, hr, hc):
    x = np.asarray(
        mask,
        dtype=np.bool_,
    )

    p = np.pad(
        x,
        ((hr, hr), (hc, hc)),
        mode="constant",
        constant_values=False,
    )

    return np.lib.stride_tricks.sliding_window_view(
        p,
        (2 * hr + 1, 2 * hc + 1),
    )


def q(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return {}

    v = np.percentile(
        x,
        [5, 25, 50, 75, 95, 99],
    )

    return {
        "p05": float(v[0]),
        "p25": float(v[1]),
        "median": float(v[2]),
        "p75": float(v[3]),
        "p95": float(v[4]),
        "p99": float(v[5]),
    }


def quality_report(
    name,
    mask,
    *,
    similarity,
    mederr,
    p95err,
    full_tc,
    seq_tc,
):
    n = int(
        np.count_nonzero(mask)
    )

    print()
    print(f"## {name}")
    print()

    print(
        "n                    :",
        f"{n:,}",
    )

    if n == 0:
        return

    print(
        "phase similarity     :",
        q(similarity[mask]),
    )

    print(
        "median error deg     :",
        q(mederr[mask]),
    )

    print(
        "p95 error deg        :",
        q(p95err[mask]),
    )

    print(
        "full38 TC            :",
        q(full_tc[mask]),
    )

    print(
        "seq fullspan TC      :",
        q(seq_tc[mask]),
    )

    print(
        "median err >30 deg   :",
        f"{100*np.mean(mederr[mask] > 30):.6f}%",
    )

    print(
        "median err >60 deg   :",
        f"{100*np.mean(mederr[mask] > 60):.6f}%",
    )

    print(
        "p95 err >30 deg      :",
        f"{100*np.mean(p95err[mask] > 30):.6f}%",
    )

    print(
        "p95 err >60 deg      :",
        f"{100*np.mean(p95err[mask] > 60):.6f}%",
    )

    print(
        "p95 err >90 deg      :",
        f"{100*np.mean(p95err[mask] > 90):.6f}%",
    )

    print(
        "p95 err >120 deg     :",
        f"{100*np.mean(p95err[mask] > 120):.6f}%",
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=16000,
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=1024,
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

    processing = (
        Path(paths.output_dir)
        / "processing"
    )

    seqdir = (
        processing
        / "sequential"
    )

    metric_path = (
        seqdir
        / "u34m_beta0_phase_parity_metrics.npz"
    )

    seq_phase_path = (
        seqdir
        / "u34m_beta0_sequential_phase_points.npy"
    )

    core_path = (
        seqdir
        / "compression_state_core_K24.npy"
    )

    yxt_path = (
        processing
        / "cache"
        / "phase_corrected_yxt.npy"
    )

    geom_path = (
        processing
        / "cache"
        / "phase_geometry_valid.npy"
    )

    scale_path = (
        processing
        / "ds_statistics"
        / "rayleigh_scale2.npy"
    )

    raw_valid_path = (
        processing
        / "ds_statistics"
        / "raw_valid.npy"
    )

    ps_path = (
        processing
        / "ps_mask.npy"
    )

    required = (
        metric_path,
        seq_phase_path,
        core_path,
        yxt_path,
        geom_path,
        scale_path,
        raw_valid_path,
        ps_path,
    )

    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)

    z = np.load(
        metric_path
    )

    required_keys = (
        "rows",
        "cols",
        "effective_K",
        "full_temporal_coherence",
        "phase_similarity",
        "median_abs_error_deg",
        "p95_abs_error_deg",
    )

    for key in required_keys:
        if key not in z.files:
            raise KeyError(
                f"missing {key!r}; "
                f"available={z.files}"
            )

    rr = np.asarray(
        z["rows"],
        dtype=np.int32,
    )

    cc = np.asarray(
        z["cols"],
        dtype=np.int32,
    )

    K_ref = np.asarray(
        z["effective_K"],
        dtype=np.int16,
    )

    full_tc = np.asarray(
        z["full_temporal_coherence"],
        dtype=np.float32,
    )

    similarity = np.asarray(
        z["phase_similarity"],
        dtype=np.float32,
    )

    mederr = np.asarray(
        z["median_abs_error_deg"],
        dtype=np.float32,
    )

    p95err = np.asarray(
        z["p95_abs_error_deg"],
        dtype=np.float32,
    )

    n = rr.size
    ndate = len(stack.dates)

    seq_phase = np.load(
        seq_phase_path,
        mmap_mode="r",
    )

    if seq_phase.shape != (
        n,
        ndate,
    ):
        raise RuntimeError(
            f"sequential phase shape "
            f"{seq_phase.shape} != {(n, ndate)}"
        )

    yxt = np.load(
        yxt_path,
        mmap_mode="r",
    )

    geom = np.load(
        geom_path,
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

    ps = np.load(
        ps_path,
        mmap_mode="r",
    )

    state_core = np.load(
        core_path,
        mmap_mode="r",
    )

    if yxt.shape != (
        H,
        W,
        ndate,
    ):
        raise RuntimeError(
            f"YXT shape={yxt.shape}"
        )

    valid = (
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

    ps_bool = (
        np.asarray(
            ps,
            dtype=np.bool_,
        )
        &
        valid
    )

    state_core = np.asarray(
        state_core,
        dtype=np.bool_,
    )

    print(
        "=" * 108
    )

    print(
        "U3.5b M19 FULL-SPAN "
        "SEQUENTIAL TEMPORAL COHERENCE"
    )

    print(
        "=" * 108
    )

    print(
        "config                 :",
        config_path,
    )

    print(
        "scene                  :",
        f"{H} x {W}",
    )

    print(
        "dates                  :",
        ndate,
    )

    print(
        "centers                :",
        f"{n:,}",
    )

    print(
        "batch                  :",
        args.batch,
    )

    print(
        "support block          :",
        args.support_block,
    )

    print()

    # ---------------------------------------------------------
    # Frozen exact GLRT context.
    # ---------------------------------------------------------

    ctx = prepare_glrt_window_context(
        scale2,
        valid,
        ps_bool,
        half_row=5,
        half_col=11,
    )

    core_windows = bool_windows(
        state_core,
        5,
        11,
    )

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

    # Warm Numba quality kernel outside timing.
    dummy_coh = np.ones(
        (1, pairs.shape[0]),
        dtype=np.complex64,
    )

    dummy_phase = np.ones(
        (1, ndate),
        dtype=np.complex64,
    )

    _ = temporal_quality_streaming(
        dummy_coh,
        dummy_phase,
        pi,
        pj,
    )

    del dummy_coh
    del dummy_phase

    out_path = (
        seqdir
        / "u35b_m19_fullspan_sequential_tc.npy"
    )

    seq_tc = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n,),
    )

    seq_tc[:] = np.nan

    K_mismatch = 0
    K_min = None

    t0 = time.perf_counter()

    for b0 in range(
        0,
        n,
        args.batch,
    ):

        b1 = min(
            n,
            b0 + args.batch,
        )

        br = rr[b0:b1]
        bc = cc[b0:b1]

        support, _ = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=0.005,
                nslc=ndate,
                block_size=args.support_block,
            )
        )

        support &= np.asarray(
            core_windows[
                br,
                bc,
            ],
            dtype=np.bool_,
        )

        K = np.sum(
            support,
            axis=(1, 2),
            dtype=np.int32,
        ).astype(
            np.int16
        )

        K_mismatch += int(
            np.count_nonzero(
                K
                !=
                K_ref[b0:b1]
            )
        )

        if K.size:
            cur_min = int(
                K.min()
            )

            K_min = (
                cur_min
                if K_min is None
                else min(
                    K_min,
                    cur_min,
                )
            )

        coh = compressed_coherence(
            yxt,
            br,
            bc,
            support,
            pi,
            pj,
        )

        tc, _ = temporal_quality_streaming(
            coh,
            np.asarray(
                seq_phase[b0:b1],
                dtype=np.complex64,
            ),
            pi,
            pj,
        )

        seq_tc[
            b0:b1
        ] = tc

        del support
        del coh
        del tc

        if (
            b1 == n
            or
            b1 % (
                args.batch * 5
            ) == 0
        ):
            elapsed = (
                time.perf_counter()
                -
                t0
            )

            rate = (
                b1 / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"fullspan TC "
                f"{b1:,}/{n:,} "
                f"({100*b1/n:6.2f}%) "
                f"rate={rate:,.0f} center/s "
                f"Kmis={K_mismatch}"
            )

    seq_tc.flush()

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    seq_tc_arr = np.asarray(
        seq_tc,
        dtype=np.float32,
    )

    finite = (
        np.isfinite(
            seq_tc_arr
        )
        &
        np.isfinite(
            full_tc
        )
        &
        np.isfinite(
            similarity
        )
        &
        np.isfinite(
            mederr
        )
        &
        np.isfinite(
            p95err
        )
    )

    full_accept = (
        finite
        &
        (
            full_tc >= 0.80
        )
    )

    seq_accept = (
        finite
        &
        (
            seq_tc_arr >= 0.80
        )
    )

    both = (
        full_accept
        &
        seq_accept
    )

    seq_only = (
        seq_accept
        &
        ~full_accept
    )

    full_only = (
        full_accept
        &
        ~seq_accept
    )

    neither = (
        finite
        &
        ~full_accept
        &
        ~seq_accept
    )

    tp = int(
        np.count_nonzero(
            both
        )
    )

    fp = int(
        np.count_nonzero(
            seq_only
        )
    )

    fn = int(
        np.count_nonzero(
            full_only
        )
    )

    tn = int(
        np.count_nonzero(
            neither
        )
    )

    precision = (
        tp
        /
        (tp + fp)
        if tp + fp
        else np.nan
    )

    recall = (
        tp
        /
        (tp + fn)
        if tp + fn
        else np.nan
    )

    jaccard = (
        tp
        /
        (tp + fp + fn)
        if tp + fp + fn
        else np.nan
    )

    agreement = (
        (tp + tn)
        /
        (tp + fp + fn + tn)
    )

    print()
    print(
        "=" * 108
    )

    print(
        "U3.5b RESULTS"
    )

    print(
        "=" * 108
    )

    print(
        "elapsed                :",
        f"{elapsed:.3f} s",
    )

    print(
        "K minimum              :",
        K_min,
    )

    print(
        "K parity mismatch      :",
        K_mismatch,
    )

    print(
        "finite TC              :",
        f"{np.count_nonzero(finite):,}",
    )

    print()

    print(
        "full38 TC              :",
        q(full_tc[finite]),
    )

    print(
        "seq fullspan TC        :",
        q(seq_tc_arr[finite]),
    )

    print(
        "seqTC-fullTC           :",
        q(
            seq_tc_arr[finite]
            -
            full_tc[finite]
        ),
    )

    print()

    print(
        "threshold              : 0.80"
    )

    print(
        "full38 accepted        :",
        f"{np.count_nonzero(full_accept):,}",
    )

    print(
        "seq-fullspan accepted  :",
        f"{np.count_nonzero(seq_accept):,}",
    )

    print(
        "both / TP              :",
        f"{tp:,}",
    )

    print(
        "seq-only / FP          :",
        f"{fp:,}",
    )

    print(
        "full-only / FN         :",
        f"{fn:,}",
    )

    print(
        "both rejected / TN     :",
        f"{tn:,}",
    )

    print()

    print(
        "mask agreement         :",
        f"{100*agreement:.6f}%",
    )

    print(
        "precision vs full38    :",
        f"{100*precision:.6f}%",
    )

    print(
        "recall vs full38       :",
        f"{100*recall:.6f}%",
    )

    print(
        "Jaccard                :",
        f"{100*jaccard:.6f}%",
    )

    quality_report(
        "SEQ FULLSPAN TC >=0.80",
        seq_accept,
        similarity=similarity,
        mederr=mederr,
        p95err=p95err,
        full_tc=full_tc,
        seq_tc=seq_tc_arr,
    )

    quality_report(
        "BOTH ACCEPTED",
        both,
        similarity=similarity,
        mederr=mederr,
        p95err=p95err,
        full_tc=full_tc,
        seq_tc=seq_tc_arr,
    )

    quality_report(
        "SEQ-ONLY",
        seq_only,
        similarity=similarity,
        mederr=mederr,
        p95err=p95err,
        full_tc=full_tc,
        seq_tc=seq_tc_arr,
    )

    quality_report(
        "FULL38-ONLY",
        full_only,
        similarity=similarity,
        mederr=mederr,
        p95err=p95err,
        full_tc=full_tc,
        seq_tc=seq_tc_arr,
    )

    metrics_path = (
        seqdir
        /
        "u35b_m19_fullspan_tc_metrics.npz"
    )

    np.savez_compressed(
        metrics_path,
        rows=rr,
        cols=cc,
        effective_K=K_ref,
        full_temporal_coherence=full_tc,
        sequential_fullspan_temporal_coherence=seq_tc_arr,
        full_accept=full_accept,
        sequential_accept=seq_accept,
        phase_similarity=similarity,
        median_abs_error_deg=mederr,
        p95_abs_error_deg=p95err,
    )

    print()
    print(
        "TC output              :",
        out_path,
    )

    print(
        "metrics                :",
        metrics_path,
    )

    print()
    print(
        "U3.5b COMPUTATIONAL INTEGRITY:",
        (
            "PASS"
            if (
                K_mismatch == 0
                and
                np.count_nonzero(finite)
                == n
            )
            else
            "FAIL"
        ),
    )

    print(
        "U3.5b SCIENTIFIC DECISION: "
        "PENDING OBSERVED METRICS"
    )


if __name__ == "__main__":
    main()
