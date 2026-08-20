#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np


SEQDIR = Path(
    "/home/ubuntu/Downloads/psds/output/processing/sequential"
)

U35A = SEQDIR / "u35a_m19_dolphin_tc_metrics.npz"
U35B = SEQDIR / "u35b_m19_fullspan_tc_metrics.npz"


def q(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    v = np.percentile(
        x,
        [0, 5, 25, 50, 75, 95, 99, 100],
    )

    names = (
        "min",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
        "max",
    )

    return {
        k: float(vv)
        for k, vv in zip(names, v)
    }


def report(
    name,
    mask,
    *,
    tc0,
    tc1,
    avg_tc,
    fullspan_tc,
    full_tc,
    similarity,
    mederr,
    p95err,
):
    n = int(np.count_nonzero(mask))

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print("n                       :", f"{n:,}")

    if n == 0:
        return

    min_stage = np.minimum(tc0, tc1)
    max_stage = np.maximum(tc0, tc1)
    stage_diff = np.abs(tc0 - tc1)

    avg_minus_fullspan = (
        avg_tc - fullspan_tc
    )

    fullspan_minus_full = (
        fullspan_tc - full_tc
    )

    print()
    print("stage0 TC               :", q(tc0[mask]))
    print("stage1 TC               :", q(tc1[mask]))
    print("min(stage TC)           :", q(min_stage[mask]))
    print("max(stage TC)           :", q(max_stage[mask]))
    print("|stage0-stage1|         :", q(stage_diff[mask]))

    print()
    print("avg stage TC             :", q(avg_tc[mask]))
    print("seq fullspan TC          :", q(fullspan_tc[mask]))
    print("full38 TC                :", q(full_tc[mask]))

    print()
    print("avgTC-fullspanTC         :", q(avg_minus_fullspan[mask]))
    print("fullspanTC-full38TC      :", q(fullspan_minus_full[mask]))

    print()
    print("phase similarity         :", q(similarity[mask]))
    print("median error deg         :", q(mederr[mask]))
    print("p95 error deg            :", q(p95err[mask]))

    print()
    print(
        "stage0 < 0.80           :",
        f"{100*np.mean(tc0[mask] < 0.80):.6f}%"
    )

    print(
        "stage1 < 0.80           :",
        f"{100*np.mean(tc1[mask] < 0.80):.6f}%"
    )

    print(
        "either stage <0.80      :",
        f"{100*np.mean(min_stage[mask] < 0.80):.6f}%"
    )

    print(
        "|stage difference| >.05 :",
        f"{100*np.mean(stage_diff[mask] > 0.05):.6f}%"
    )

    print(
        "|stage difference| >.10 :",
        f"{100*np.mean(stage_diff[mask] > 0.10):.6f}%"
    )

    print()
    print(
        "p95 error >30 deg       :",
        f"{100*np.mean(p95err[mask] > 30.0):.6f}%"
    )

    print(
        "p95 error >60 deg       :",
        f"{100*np.mean(p95err[mask] > 60.0):.6f}%"
    )

    print(
        "p95 error >90 deg       :",
        f"{100*np.mean(p95err[mask] > 90.0):.6f}%"
    )


def main():

    for p in (U35A, U35B):
        if not p.is_file():
            raise FileNotFoundError(p)

    a = np.load(U35A)
    b = np.load(U35B)

    print("# U3.5c M19 SEQ-ONLY DIAGNOSTICS")
    print()
    print("U3.5a keys:", a.files)
    print("U3.5b keys:", b.files)

    # ---------------------------------------------------------
    # Population alignment
    # ---------------------------------------------------------

    if not np.array_equal(
        a["rows"],
        b["rows"],
    ):
        raise RuntimeError("row population mismatch")

    if not np.array_equal(
        a["cols"],
        b["cols"],
    ):
        raise RuntimeError("column population mismatch")

    rr = b["rows"]
    n = rr.size

    print()
    print("population              :", f"{n:,}")
    print("row/col alignment       : PASS")

    # ---------------------------------------------------------
    # Existing production-computable diagnostics
    # ---------------------------------------------------------

    tc0 = np.asarray(
        a["stage0_temporal_coherence"],
        dtype=np.float32,
    )

    tc1 = np.asarray(
        a["stage1_temporal_coherence"],
        dtype=np.float32,
    )

    avg_tc = np.asarray(
        a["sequential_average_temporal_coherence"],
        dtype=np.float32,
    )

    fullspan_tc = np.asarray(
        b["sequential_fullspan_temporal_coherence"],
        dtype=np.float32,
    )

    full_tc = np.asarray(
        b["full_temporal_coherence"],
        dtype=np.float32,
    )

    similarity = np.asarray(
        b["phase_similarity"],
        dtype=np.float32,
    )

    mederr = np.asarray(
        b["median_abs_error_deg"],
        dtype=np.float32,
    )

    p95err = np.asarray(
        b["p95_abs_error_deg"],
        dtype=np.float32,
    )

    full_accept = (
        np.asarray(
            b["full_accept"],
            dtype=bool,
        )
    )

    seq_accept = (
        np.asarray(
            b["sequential_accept"],
            dtype=bool,
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
        ~full_accept
        &
        ~seq_accept
    )

    # ---------------------------------------------------------
    # Reports
    # ---------------------------------------------------------

    report(
        "BOTH ACCEPTED",
        both,
        tc0=tc0,
        tc1=tc1,
        avg_tc=avg_tc,
        fullspan_tc=fullspan_tc,
        full_tc=full_tc,
        similarity=similarity,
        mederr=mederr,
        p95err=p95err,
    )

    report(
        "SEQ-ONLY / PROBLEM POPULATION",
        seq_only,
        tc0=tc0,
        tc1=tc1,
        avg_tc=avg_tc,
        fullspan_tc=fullspan_tc,
        full_tc=full_tc,
        similarity=similarity,
        mederr=mederr,
        p95err=p95err,
    )

    report(
        "FULL38-ONLY",
        full_only,
        tc0=tc0,
        tc1=tc1,
        avg_tc=avg_tc,
        fullspan_tc=fullspan_tc,
        full_tc=full_tc,
        similarity=similarity,
        mederr=mederr,
        p95err=p95err,
    )

    report(
        "BOTH REJECTED",
        neither,
        tc0=tc0,
        tc1=tc1,
        avg_tc=avg_tc,
        fullspan_tc=fullspan_tc,
        full_tc=full_tc,
        similarity=similarity,
        mederr=mederr,
        p95err=p95err,
    )

    print()
    print("U3.5c COMPUTATIONAL INTEGRITY: PASS")
    print(
        "U3.5c SCIENTIFIC DECISION: "
        "PENDING DIAGNOSTIC SEPARATION"
    )


if __name__ == "__main__":
    main()
