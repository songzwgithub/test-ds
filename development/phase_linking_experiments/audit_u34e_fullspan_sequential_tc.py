#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.context import open_from_config
from pypsds.phase_linking.coherence import compressed_coherence
from pypsds.phase_linking.emi import (
    image_pairs,
    temporal_coherence,
)
from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


def bool_windows(x, hr, hc):
    x = np.asarray(x, dtype=np.bool_)

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


def qs(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    p = np.percentile(
        x,
        [5, 25, 50, 75, 95, 99],
    )

    return {
        "p05": float(p[0]),
        "p25": float(p[1]),
        "median": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
        "p99": float(p[5]),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--tc-min", type=float, default=0.80)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--support-block", type=int, default=1024)

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (row0, col0, H, W),
    ) = open_from_config(args.config)

    processing = Path(paths.output_dir) / "processing"
    seqdir = processing / "sequential"

    # --------------------------------------------------------
    # Verify current U3.4a is M16.
    # --------------------------------------------------------

    phase_report = json.loads(
        (
            seqdir
            / "u34a_phase_parity_report.json"
        ).read_text()
    )

    if phase_report["ministack_size"] != 16:
        raise RuntimeError(
            "Current U3.4a phase stack is not M16. "
            f"Found M={phase_report['ministack_size']}"
        )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    yxt = np.load(
        processing
        / "cache"
        / "phase_corrected_yxt.npy",
        mmap_mode="r",
    )

    scale2 = np.load(
        processing
        / "ds_statistics"
        / "rayleigh_scale2.npy",
        mmap_mode="r",
    )

    raw_valid = np.load(
        processing
        / "ds_statistics"
        / "raw_valid.npy",
        mmap_mode="r",
    )

    geom = np.load(
        processing
        / "cache"
        / "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    ps = np.load(
        processing
        / "ps_mask.npy",
        mmap_mode="r",
    )

    core = np.load(
        seqdir
        / "compression_state_core_K24.npy",
        mmap_mode="r",
    )

    full_tc = np.load(
        processing
        / "temporal_coherence.npy",
        mmap_mode="r",
    )

    phase = np.load(
        seqdir
        / "u34a_sequential_phase_points.npy",
        mmap_mode="r",
    )

    rr = np.load(
        seqdir
        / "u34a_rows.npy"
    )

    cc = np.load(
        seqdir
        / "u34a_cols.npy"
    )

    K_ref = np.load(
        seqdir
        / "u34a_effective_K.npy"
    )

    metrics = np.load(
        seqdir
        / "u34a_phase_parity_metrics.npz"
    )

    mederr = metrics[
        "median_abs_error_deg"
    ]

    p95err = metrics[
        "p95_abs_error_deg"
    ]

    similarity = metrics[
        "phase_similarity"
    ]

    ndate = yxt.shape[2]

    if ndate != 38:
        raise RuntimeError(
            f"Expected 38 dates, got {ndate}"
        )

    if phase.shape != (
        rr.size,
        ndate,
    ):
        raise RuntimeError(
            f"phase shape={phase.shape}"
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

    ps = np.asarray(
        ps,
        dtype=np.bool_,
    )

    core = np.asarray(
        core,
        dtype=np.bool_,
    )

    # --------------------------------------------------------
    # Frozen exact 38-date SHP support.
    # --------------------------------------------------------

    ctx = prepare_glrt_window_context(
        scale2,
        valid,
        ps,
        half_row=5,
        half_col=11,
    )

    core_windows = bool_windows(
        core,
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

    n = rr.size

    seq_tc = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    t0 = perf_counter()

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
                nslc=38,
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

        if not np.array_equal(
            K,
            K_ref[b0:b1],
        ):
            bad = int(
                np.flatnonzero(
                    K != K_ref[b0:b1]
                )[0]
            )

            raise RuntimeError(
                "K parity failure at "
                f"({int(br[bad])},{int(bc[bad])})"
            )

        coh = compressed_coherence(
            yxt,
            br,
            bc,
            support,
            pi,
            pj,
        )

        ph = np.asarray(
            phase[b0:b1],
            dtype=np.complex64,
        )

        seq_tc[
            b0:b1
        ] = temporal_coherence(
            coh,
            ph,
            pairs,
        )

        if (
            b1 == n
            or
            b1 % 50000 < args.batch
        ):
            elapsed = perf_counter() - t0

            print(
                f"{b1:8,d}/{n:8,d} "
                f"({100*b1/n:6.2f}%) "
                f"rate={b1/elapsed:,.0f} center/s"
            )

    elapsed = perf_counter() - t0

    # --------------------------------------------------------
    # Reference full-SCM TC.
    # --------------------------------------------------------

    ref = np.asarray(
        full_tc[
            rr,
            cc,
        ],
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(seq_tc)
        &
        np.isfinite(ref)
    ):
        raise RuntimeError(
            "non-finite TC values"
        )

    tc_abs = np.abs(
        seq_tc - ref
    )

    corr = float(
        np.corrcoef(
            seq_tc,
            ref,
        )[0, 1]
    )

    ref_accept = (
        ref >= args.tc_min
    )

    seq_accept = (
        seq_tc >= args.tc_min
    )

    tp = int(
        np.count_nonzero(
            ref_accept
            &
            seq_accept
        )
    )

    fp = int(
        np.count_nonzero(
            ~ref_accept
            &
            seq_accept
        )
    )

    fn = int(
        np.count_nonzero(
            ref_accept
            &
            ~seq_accept
        )
    )

    tn = int(
        np.count_nonzero(
            ~ref_accept
            &
            ~seq_accept
        )
    )

    agreement = (
        tp + tn
    ) / n

    false_accept = (
        fp
        /
        np.count_nonzero(
            ~ref_accept
        )
    )

    false_reject = (
        fn
        /
        np.count_nonzero(
            ref_accept
        )
    )

    # --------------------------------------------------------
    # Phase-error tail after sequential TC gate.
    # --------------------------------------------------------

    accepted_med = mederr[
        seq_accept
    ]

    accepted_p95 = p95err[
        seq_accept
    ]

    accepted_sim = similarity[
        seq_accept
    ]

    fp_p95 = p95err[
        seq_accept
        &
        ~ref_accept
    ]

    report = {
        "format":
            "pyPSDS-GAMMA-U3.4e-fullspan-sequential-TC-v1",

        "ministack_size":
            16,

        "n":
            int(n),

        "threshold":
            args.tc_min,

        "tc_correlation":
            corr,

        "tc_abs_error":
            qs(tc_abs),

        "reference_accepted":
            int(ref_accept.sum()),

        "sequential_accepted":
            int(seq_accept.sum()),

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "agreement_fraction":
            float(agreement),

        "false_accept_fraction":
            float(false_accept),

        "false_reject_fraction":
            float(false_reject),

        "accepted_phase": {
            "similarity":
                qs(
                    accepted_sim
                ),

            "median_error_deg":
                qs(
                    accepted_med
                ),

            "p95_error_deg":
                qs(
                    accepted_p95
                ),

            "median_gt30_fraction":
                float(
                    np.mean(
                        accepted_med > 30
                    )
                ),

            "p95_gt60_fraction":
                float(
                    np.mean(
                        accepted_p95 > 60
                    )
                ),

            "p95_gt90_fraction":
                float(
                    np.mean(
                        accepted_p95 > 90
                    )
                ),
        },

        "false_accept_phase": {
            "n":
                int(fp_p95.size),

            "p95_error_deg":
                qs(
                    fp_p95
                ),

            "p95_gt60_fraction":
                float(
                    np.mean(
                        fp_p95 > 60
                    )
                )
                if fp_p95.size
                else None,

            "p95_gt90_fraction":
                float(
                    np.mean(
                        fp_p95 > 90
                    )
                )
                if fp_p95.size
                else None,
        },

        "elapsed_seconds":
            elapsed,

        "decision":
            "pending_metrics",
    }

    out_json = (
        seqdir
        / "u34e_fullspan_sequential_tc.json"
    )

    out_npz = (
        seqdir
        / "u34e_fullspan_sequential_tc.npz"
    )

    np.savez_compressed(
        out_npz,

        rows=rr,
        cols=cc,

        full_scm_tc=ref,
        sequential_fullspan_tc=seq_tc,

        full_accept=ref_accept,
        sequential_accept=seq_accept,
    )

    out_json.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("U3.4e FULL-SPAN SEQUENTIAL TC")
    print("=" * 120)

    print(
        "centers               :",
        f"{n:,}",
    )

    print(
        "TC correlation        :",
        f"{corr:.6f}",
    )

    print(
        "TC |difference|       :",
        qs(tc_abs),
    )

    print(
        "reference accepted    :",
        f"{ref_accept.sum():,}",
    )

    print(
        "sequential accepted   :",
        f"{seq_accept.sum():,}",
    )

    print(
        "agreement             :",
        f"{100*agreement:.6f}%",
    )

    print(
        "false accept          :",
        f"{fp:,}",
        f"({100*false_accept:.6f}%)",
    )

    print(
        "false reject          :",
        f"{fn:,}",
        f"({100*false_reject:.6f}%)",
    )

    print()

    print("Accepted sequential phase quality")
    print(
        "similarity            :",
        qs(accepted_sim),
    )

    print(
        "median error deg      :",
        qs(accepted_med),
    )

    print(
        "p95 error deg         :",
        qs(accepted_p95),
    )

    print(
        "p95 > 60 deg         :",
        f"{100*np.mean(accepted_p95 > 60):.6f}%",
    )

    print(
        "p95 > 90 deg         :",
        f"{100*np.mean(accepted_p95 > 90):.6f}%",
    )

    print()

    print("False-accept phase tail")

    print(
        "FP count              :",
        f"{fp_p95.size:,}",
    )

    if fp_p95.size:

        print(
            "FP p95 error         :",
            qs(fp_p95),
        )

        print(
            "FP p95 >60           :",
            f"{100*np.mean(fp_p95 > 60):.6f}%",
        )

        print(
            "FP p95 >90           :",
            f"{100*np.mean(fp_p95 > 90):.6f}%",
        )

    print()

    print(
        "elapsed               :",
        f"{elapsed:.3f} s",
    )

    print(
        "json                  :",
        out_json,
    )

    print(
        "npz                   :",
        out_npz,
    )

    print()

    print(
        "U3.4e COMPUTATIONAL AUDIT: PASS"
    )

    print(
        "U3.4e TC DECISION: PENDING METRICS"
    )


if __name__ == "__main__":
    main()
