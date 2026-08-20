#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np


SEQDIR = Path(
    "/home/ubuntu/Downloads/psds/output/processing/sequential"
)

METRIC_PATH = (
    SEQDIR
    / "u34m_beta0_phase_parity_metrics.npz"
)

STAGE0_TC_PATH = (
    SEQDIR
    / "u33b_stage0000_temporal_coherence.npy"
)

STAGE1_TC_PATH = (
    SEQDIR
    / "u33b_stage0001_temporal_coherence.npy"
)


def quantiles(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return {}

    q = np.percentile(
        x,
        [
            5,
            25,
            50,
            75,
            95,
            99,
        ],
    )

    return {
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
    }


def report_phase_quality(
    name,
    mask,
    *,
    similarity,
    median_error,
    p95_error,
    full_tc,
    seq_tc,
):
    n = int(
        np.count_nonzero(
            mask
        )
    )

    print()
    print(
        f"## {name}"
    )
    print()

    print(
        "n                    :",
        f"{n:,}",
    )

    if n == 0:
        return

    print(
        "phase similarity     :",
        quantiles(
            similarity[
                mask
            ]
        ),
    )

    print(
        "median error deg     :",
        quantiles(
            median_error[
                mask
            ]
        ),
    )

    print(
        "p95 error deg        :",
        quantiles(
            p95_error[
                mask
            ]
        ),
    )

    print(
        "full38 TC            :",
        quantiles(
            full_tc[
                mask
            ]
        ),
    )

    print(
        "sequential avg TC    :",
        quantiles(
            seq_tc[
                mask
            ]
        ),
    )

    print(
        "median err >30 deg   :",
        f"{100*np.mean(median_error[mask] > 30.0):.6f}%",
    )

    print(
        "median err >60 deg   :",
        f"{100*np.mean(median_error[mask] > 60.0):.6f}%",
    )

    print(
        "p95 err >30 deg      :",
        f"{100*np.mean(p95_error[mask] > 30.0):.6f}%",
    )

    print(
        "p95 err >60 deg      :",
        f"{100*np.mean(p95_error[mask] > 60.0):.6f}%",
    )

    print(
        "p95 err >90 deg      :",
        f"{100*np.mean(p95_error[mask] > 90.0):.6f}%",
    )

    print(
        "p95 err >120 deg     :",
        f"{100*np.mean(p95_error[mask] > 120.0):.6f}%",
    )


def main():

    for p in (
        METRIC_PATH,
        STAGE0_TC_PATH,
        STAGE1_TC_PATH,
    ):
        if not p.is_file():
            raise FileNotFoundError(
                p
            )

    z = np.load(
        METRIC_PATH
    )

    required = (
        "rows",
        "cols",
        "full_temporal_coherence",
        "phase_similarity",
        "median_abs_error_deg",
        "p95_abs_error_deg",
    )

    for k in required:
        if k not in z.files:
            raise KeyError(
                f"Missing {k!r}; "
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

    full_tc = np.asarray(
        z[
            "full_temporal_coherence"
        ],
        dtype=np.float32,
    )

    similarity = np.asarray(
        z[
            "phase_similarity"
        ],
        dtype=np.float32,
    )

    median_error = np.asarray(
        z[
            "median_abs_error_deg"
        ],
        dtype=np.float32,
    )

    p95_error = np.asarray(
        z[
            "p95_abs_error_deg"
        ],
        dtype=np.float32,
    )

    tc0_map = np.load(
        STAGE0_TC_PATH,
        mmap_mode="r",
    )

    tc1_map = np.load(
        STAGE1_TC_PATH,
        mmap_mode="r",
    )

    if tc0_map.shape != tc1_map.shape:
        raise RuntimeError(
            "stage TC raster shape mismatch"
        )

    if (
        rr.size != cc.size
        or
        rr.size != full_tc.size
        or
        rr.size != similarity.size
        or
        rr.size != median_error.size
        or
        rr.size != p95_error.size
    ):
        raise RuntimeError(
            "U3.4m metric population mismatch"
        )

    tc0 = np.asarray(
        tc0_map[
            rr,
            cc,
        ],
        dtype=np.float32,
    )

    tc1 = np.asarray(
        tc1_map[
            rr,
            cc,
        ],
        dtype=np.float32,
    )

    finite_stage = (
        np.isfinite(
            tc0
        )
        &
        np.isfinite(
            tc1
        )
    )

    seq_tc = np.full(
        rr.size,
        np.nan,
        dtype=np.float32,
    )

    seq_tc[
        finite_stage
    ] = (
        (
            tc0[
                finite_stage
            ].astype(
                np.float64
            )
            +
            tc1[
                finite_stage
            ].astype(
                np.float64
            )
        )
        /
        2.0
    ).astype(
        np.float32
    )

    finite = (
        finite_stage
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
            median_error
        )
        &
        np.isfinite(
            p95_error
        )
    )

    full_accept = (
        finite
        &
        (
            full_tc
            >=
            0.80
        )
    )

    seq_accept = (
        finite
        &
        (
            seq_tc
            >=
            0.80
        )
    )

    both_accept = (
        full_accept
        &
        seq_accept
    )

    false_positive = (
        seq_accept
        &
        ~full_accept
    )

    false_negative = (
        full_accept
        &
        ~seq_accept
    )

    both_reject = (
        finite
        &
        ~full_accept
        &
        ~seq_accept
    )

    tp = int(
        np.count_nonzero(
            both_accept
        )
    )

    fp = int(
        np.count_nonzero(
            false_positive
        )
    )

    fn = int(
        np.count_nonzero(
            false_negative
        )
    )

    tn = int(
        np.count_nonzero(
            both_reject
        )
    )

    precision = (
        tp
        /
        (tp + fp)
        if (tp + fp)
        else np.nan
    )

    recall = (
        tp
        /
        (tp + fn)
        if (tp + fn)
        else np.nan
    )

    jaccard = (
        tp
        /
        (tp + fp + fn)
        if (tp + fp + fn)
        else np.nan
    )

    agreement = (
        (tp + tn)
        /
        (tp + fp + fn + tn)
    )

    print(
        "=" * 108
    )

    print(
        "U3.5a M19 DOLPHIN-STYLE "
        "SEQUENTIAL TEMPORAL COHERENCE"
    )

    print(
        "=" * 108
    )

    print(
        "centers                :",
        f"{rr.size:,}",
    )

    print(
        "finite stage TC        :",
        f"{np.count_nonzero(finite_stage):,}",
        f"({100*np.mean(finite_stage):.6f}%)",
    )

    print()

    print(
        "stage0 TC              :",
        quantiles(
            tc0[
                finite
            ]
        ),
    )

    print(
        "stage1 TC              :",
        quantiles(
            tc1[
                finite
            ]
        ),
    )

    print(
        "sequential avg TC      :",
        quantiles(
            seq_tc[
                finite
            ]
        ),
    )

    print(
        "full38 TC              :",
        quantiles(
            full_tc[
                finite
            ]
        ),
    )

    diff = (
        seq_tc[
            finite
        ]
        -
        full_tc[
            finite
        ]
    )

    print(
        "seqTC-fullTC           :",
        quantiles(
            diff
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
        "sequential accepted    :",
        f"{np.count_nonzero(seq_accept):,}",
    )

    print()

    print(
        "both accepted / TP     :",
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

    report_phase_quality(
        "SEQUENTIAL PRODUCTION CANDIDATE: avg stage TC >= 0.80",
        seq_accept,
        similarity=similarity,
        median_error=median_error,
        p95_error=p95_error,
        full_tc=full_tc,
        seq_tc=seq_tc,
    )

    report_phase_quality(
        "BOTH ACCEPTED",
        both_accept,
        similarity=similarity,
        median_error=median_error,
        p95_error=p95_error,
        full_tc=full_tc,
        seq_tc=seq_tc,
    )

    report_phase_quality(
        "SEQUENTIAL-ONLY: avg stage TC >=0.80 but full38 TC <0.80",
        false_positive,
        similarity=similarity,
        median_error=median_error,
        p95_error=p95_error,
        full_tc=full_tc,
        seq_tc=seq_tc,
    )

    report_phase_quality(
        "FULL38-ONLY: full38 TC >=0.80 but avg stage TC <0.80",
        false_negative,
        similarity=similarity,
        median_error=median_error,
        p95_error=p95_error,
        full_tc=full_tc,
        seq_tc=seq_tc,
    )

    np.savez_compressed(
        SEQDIR
        / "u35a_m19_dolphin_tc_metrics.npz",
        rows=rr,
        cols=cc,
        stage0_temporal_coherence=tc0,
        stage1_temporal_coherence=tc1,
        sequential_average_temporal_coherence=seq_tc,
        full_temporal_coherence=full_tc,
        full_accept=full_accept,
        sequential_accept=seq_accept,
        both_accept=both_accept,
        false_positive=false_positive,
        false_negative=false_negative,
    )

    print()
    print(
        "metrics                :",
        SEQDIR
        / "u35a_m19_dolphin_tc_metrics.npz",
    )

    print()
    print(
        "U3.5a COMPUTATIONAL INTEGRITY: PASS"
    )

    print(
        "U3.5a SCIENTIFIC DECISION: "
        "PENDING OBSERVED METRICS"
    )


if __name__ == "__main__":
    main()
