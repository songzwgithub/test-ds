#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def qs(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    p = np.percentile(x, [5, 25, 50, 75, 95, 99])

    return {
        "p05": float(p[0]),
        "p25": float(p[1]),
        "median": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
        "p99": float(p[5]),
    }


def safe_frac(n, d):
    return float(n / d) if d else float("nan")


def phase_summary(mask, mederr, p95err, similarity):
    mask = np.asarray(mask, dtype=bool)

    if not np.any(mask):
        return {"n": 0}

    m = mederr[mask]
    p = p95err[mask]
    s = similarity[mask]

    return {
        "n": int(mask.sum()),
        "similarity": qs(s),
        "median_error_deg": qs(m),
        "p95_error_deg": qs(p),

        "median_gt30_fraction":
            float(np.mean(m > 30.0)),

        "p95_gt60_fraction":
            float(np.mean(p > 60.0)),

        "p95_gt90_fraction":
            float(np.mean(p > 90.0)),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--tc-min", type=float, default=0.80)

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
    # Verify current sequential outputs really are M=16.
    # --------------------------------------------------------

    u33_report_path = seqdir / "u33b_multistage_report.json"

    d = json.loads(
        u33_report_path.read_text(encoding="utf-8")
    )

    if d["ministack_size"] != 16:
        raise RuntimeError(
            f"Current U3.3b is M={d['ministack_size']}, expected M=16"
        )

    if d["stage_count"] != 3:
        raise RuntimeError(
            f"M16 stage_count={d['stage_count']}, expected 3"
        )

    real_counts = [
        len(x["real_indices"])
        for x in d["stages"]
    ]

    if real_counts != [16, 16, 6]:
        raise RuntimeError(
            f"Unexpected M16 real counts: {real_counts}"
        )

    tc_stage = []

    for i in range(3):

        p = (
            seqdir
            / f"u33b_stage{i:04d}_temporal_coherence.npy"
        )

        if not p.is_file():
            raise FileNotFoundError(p)

        a = np.load(
            p,
            mmap_mode="r",
        )

        if a.shape != (H, W):
            raise RuntimeError(
                f"{p.name} shape={a.shape}"
            )

        tc_stage.append(a)

    tc0, tc1, tc2 = tc_stage

    # --------------------------------------------------------
    # Existing full-SCM reference products.
    # --------------------------------------------------------

    full_tc = np.load(
        processing / "temporal_coherence.npy",
        mmap_mode="r",
    )

    pl_valid = np.load(
        processing / "pl_valid.npy",
        mmap_mode="r",
    ).astype(bool, copy=False)

    prior = np.load(
        processing / "center_prior.npy",
        mmap_mode="r",
    ).astype(bool, copy=False)

    ps = np.load(
        processing / "ps_mask.npy",
        mmap_mode="r",
    ).astype(bool, copy=False)

    keff = np.load(
        seqdir
        / "compression_state_core_K24_effective_shp_count.npy",
        mmap_mode="r",
    )

    formal_ds = (
        pl_valid
        & prior
        & ~ps
    )

    seq_route = (
        formal_ds
        & (keff >= 48)
    )

    fallback_route = (
        formal_ds
        & (keff < 48)
    )

    # --------------------------------------------------------
    # Sequential TC candidates.
    # --------------------------------------------------------

    mean_tc = (
        (
            np.asarray(tc0, dtype=np.float32)
            + np.asarray(tc1, dtype=np.float32)
            + np.asarray(tc2, dtype=np.float32)
        )
        / np.float32(3.0)
    )

    weighted_tc = (
        (
            np.float32(16.0) * np.asarray(tc0, dtype=np.float32)
            + np.float32(16.0) * np.asarray(tc1, dtype=np.float32)
            + np.float32(6.0) * np.asarray(tc2, dtype=np.float32)
        )
        / np.float32(38.0)
    )

    min_tc = np.minimum(
        np.minimum(
            np.asarray(tc0, dtype=np.float32),
            np.asarray(tc1, dtype=np.float32),
        ),
        np.asarray(tc2, dtype=np.float32),
    )

    rules = {
        "mean_stage_TC": mean_tc,
        "weighted_real_dates_TC": weighted_tc,
        "min_stage_TC": min_tc,
    }

    # --------------------------------------------------------
    # Reference acceptance classification on sequential route.
    # --------------------------------------------------------

    rr, cc = np.where(seq_route)

    ref_tc = np.asarray(
        full_tc[rr, cc],
        dtype=np.float32,
    )

    ref_accept = (
        np.isfinite(ref_tc)
        & (ref_tc >= args.tc_min)
    )

    ref_reject = ~ref_accept

    nseq = rr.size
    nref_accept = int(ref_accept.sum())
    nref_reject = int(ref_reject.sum())

    # Full-SCM fallback keeps full-SCM TC semantics.
    frr, fcc = np.where(fallback_route)

    fallback_accept = int(
        np.count_nonzero(
            np.isfinite(full_tc[frr, fcc])
            & (full_tc[frr, fcc] >= args.tc_min)
        )
    )

    # --------------------------------------------------------
    # Load M16 phase-parity snapshot, NOT mutable current U3.4a.
    # --------------------------------------------------------

    phase_metrics_path = (
        seqdir
        / "u34c_M16_phase_metrics.npz"
    )

    if not phase_metrics_path.is_file():
        raise FileNotFoundError(
            phase_metrics_path
        )

    z = np.load(phase_metrics_path)

    mrr = z["rows"]
    mcc = z["cols"]

    similarity = z["phase_similarity"]
    mederr = z["median_abs_error_deg"]
    p95err = z["p95_abs_error_deg"]

    if mrr.size != nseq:
        raise RuntimeError(
            f"M16 phase population={mrr.size:,}, "
            f"seq route={nseq:,}"
        )

    if not (
        np.array_equal(mrr, rr)
        and np.array_equal(mcc, cc)
    ):
        raise RuntimeError(
            "M16 phase metrics point ordering does not match "
            "current sequential route"
        )

    # --------------------------------------------------------
    # Audit each rule.
    # --------------------------------------------------------

    results = {}

    print("=" * 145)
    print("U3.4d M16 SEQUENTIAL TEMPORAL-COHERENCE GATE")
    print("=" * 145)

    print("formal DS                   :", f"{formal_ds.sum():,}")
    print("sequential route            :", f"{nseq:,}")
    print("full-SCM accepted on route  :", f"{nref_accept:,}")
    print("full-SCM rejected on route  :", f"{nref_reject:,}")
    print("fallback route              :", f"{fallback_route.sum():,}")
    print("fallback accepted           :", f"{fallback_accept:,}")
    print()

    print(
        "rule                    accept    agreement   "
        "falseAccept falseReject  TC_MAE50  TC_MAE95  "
        "medErr50  medErr95  p95Err50 p95Err95  p95>60%"
    )

    print("-" * 145)

    for name, tcmap in rules.items():

        pred_tc = np.asarray(
            tcmap[rr, cc],
            dtype=np.float32,
        )

        finite = (
            np.isfinite(pred_tc)
            & np.isfinite(ref_tc)
        )

        if not np.all(finite):
            raise RuntimeError(
                f"{name}: non-finite TC on "
                f"{np.count_nonzero(~finite):,} routed pixels"
            )

        pred_accept = (
            pred_tc >= args.tc_min
        )

        tp = int(
            np.count_nonzero(
                pred_accept & ref_accept
            )
        )

        fp = int(
            np.count_nonzero(
                pred_accept & ref_reject
            )
        )

        fn = int(
            np.count_nonzero(
                ~pred_accept & ref_accept
            )
        )

        tn = int(
            np.count_nonzero(
                ~pred_accept & ref_reject
            )
        )

        agreement = safe_frac(
            tp + tn,
            nseq,
        )

        false_accept = safe_frac(
            fp,
            nref_reject,
        )

        false_reject = safe_frac(
            fn,
            nref_accept,
        )

        tc_abs = np.abs(
            pred_tc - ref_tc
        )

        corr = float(
            np.corrcoef(
                pred_tc,
                ref_tc,
            )[0, 1]
        )

        phase_all = phase_summary(
            pred_accept,
            mederr,
            p95err,
            similarity,
        )

        false_accept_phase = phase_summary(
            pred_accept & ref_reject,
            mederr,
            p95err,
            similarity,
        )

        results[name] = {
            "accepted": int(pred_accept.sum()),

            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,

            "agreement_fraction":
                agreement,

            "false_accept_fraction":
                false_accept,

            "false_reject_fraction":
                false_reject,

            "tc_correlation":
                corr,

            "tc_abs_error":
                qs(tc_abs),

            "accepted_phase":
                phase_all,

            "false_accept_phase":
                false_accept_phase,

            "hybrid_total_accepted":
                int(
                    pred_accept.sum()
                    + fallback_accept
                ),
        }

        print(
            f"{name:<23s} "
            f"{pred_accept.sum():8,d} "
            f"{100*agreement:9.3f}% "
            f"{100*false_accept:10.3f}% "
            f"{100*false_reject:10.3f}% "
            f"{np.median(tc_abs):8.4f} "
            f"{np.percentile(tc_abs,95):8.4f} "
            f"{phase_all['median_error_deg']['median']:9.3f} "
            f"{phase_all['median_error_deg']['p95']:9.3f} "
            f"{phase_all['p95_error_deg']['median']:9.3f} "
            f"{phase_all['p95_error_deg']['p95']:9.3f} "
            f"{100*phase_all['p95_gt60_fraction']:8.3f}%"
        )

    print()

    print("=" * 145)
    print("FALSE-ACCEPT PHASE TAIL")
    print("=" * 145)

    print(
        "rule                    n_FP      medErr50 "
        "p95Err50 p95Err95 p95>60% p95>90%"
    )

    print("-" * 145)

    for name, x in results.items():

        p = x["false_accept_phase"]

        if p["n"] == 0:

            print(
                f"{name:<23s} "
                f"{0:8d} "
                "       -        -        -        -        -"
            )

            continue

        print(
            f"{name:<23s} "
            f"{p['n']:8,d} "
            f"{p['median_error_deg']['median']:9.3f} "
            f"{p['p95_error_deg']['median']:8.3f} "
            f"{p['p95_error_deg']['p95']:8.3f} "
            f"{100*p['p95_gt60_fraction']:7.3f}% "
            f"{100*p['p95_gt90_fraction']:7.3f}%"
        )

    report = {
        "format":
            "pyPSDS-GAMMA-U3.4d-M16-sequential-TC-gate-v1",

        "ministack_size": 16,

        "stage_real_counts":
            real_counts,

        "tc_threshold":
            args.tc_min,

        "formal_DS":
            int(formal_ds.sum()),

        "sequential_route":
            int(nseq),

        "reference_accepted":
            nref_accept,

        "reference_rejected":
            nref_reject,

        "fallback_route":
            int(fallback_route.sum()),

        "fallback_accepted":
            fallback_accept,

        "rules":
            results,

        "decision":
            "pending_observed_gate_metrics",
    }

    out = (
        seqdir
        / "u34d_sequential_tc_gate.json"
    )

    out.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("report:", out)
    print()
    print("U3.4d COMPUTATIONAL AUDIT: PASS")
    print("U3.4d TC RULE DECISION: PENDING METRICS")


if __name__ == "__main__":
    main()
