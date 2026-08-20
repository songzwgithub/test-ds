#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config

from pypsds.phase_linking.emi import (
    image_pairs,
)

from pypsds.phase_linking.coherence import (
    compressed_coherence,
)

from pypsds.phase_linking.shp_coherence_bitset import (
    glrt_support_bitset,
    compressed_coherence_bitset,
)

from pypsds.phase_linking.shp_exact_packed import (
    glrt_support_exact_packed,
    unpack_support_bitset,
)

from pypsds.selection.shp import (
    glrt_statistic,
    glrt_threshold,
)


def legacy_support(
    scale2,
    valid,
    ps,
    rows,
    cols,
    *,
    half_row,
    half_col,
    alpha,
    ndate,
):
    """
    Exact frozen make_support_batch definition.
    """

    B = rows.size

    wh = (
        2 * half_row
        +
        1
    )

    ww = (
        2 * half_col
        +
        1
    )

    out = np.zeros(
        (
            B,
            wh,
            ww,
        ),
        dtype=np.bool_,
    )

    center_scale = (
        scale2[
            rows,
            cols,
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    thr = glrt_threshold(
        alpha
    )

    H, W = valid.shape

    for ky, dy in enumerate(
        range(
            -half_row,
            half_row + 1,
        )
    ):

        for kx, dx in enumerate(
            range(
                -half_col,
                half_col + 1,
            )
        ):

            if (
                dy == 0
                and
                dx == 0
            ):
                continue

            rr = rows + dy
            cc = cols + dx

            inside = (
                (rr >= 0)
                &
                (rr < H)
                &
                (cc >= 0)
                &
                (cc < W)
            )

            if not np.any(
                inside
            ):
                continue

            ids = np.flatnonzero(
                inside
            )

            r2 = rr[
                ids
            ]

            c2 = cc[
                ids
            ]

            ngood = (
                valid[
                    r2,
                    c2,
                ]
                &
                ~ps[
                    r2,
                    c2,
                ]
            )

            if not np.any(
                ngood
            ):
                continue

            ids2 = ids[
                ngood
            ]

            r3 = rr[
                ids2
            ]

            c3 = cc[
                ids2
            ]

            stat = glrt_statistic(
                center_scale[
                    ids2
                ],
                scale2[
                    r3,
                    c3,
                ],
                nslc=ndate,
            )

            out[
                ids2,
                ky,
                kx,
            ] = (
                np.isfinite(
                    stat
                )
                &
                (
                    stat
                    <
                    thr
                )
            )

    return out


def mismatch_summary(
    reference,
    candidate,
):
    diff = (
        reference
        !=
        candidate
    )

    centers = int(
        np.count_nonzero(
            np.any(
                diff,
                axis=(1, 2),
            )
        )
    )

    bits = int(
        np.count_nonzero(
            diff
        )
    )

    ref_K = np.sum(
        reference,
        axis=(1, 2),
    )

    cand_K = np.sum(
        candidate,
        axis=(1, 2),
    )

    K_bad = int(
        np.count_nonzero(
            ref_K
            !=
            cand_K
        )
    )

    return {
        "support_mismatch_centers":
            centers,

        "support_mismatch_bits":
            bits,

        "shp_count_bad":
            K_bad,
    }


def max_complex_diff(
    a,
    b,
):
    finite = (
        np.isfinite(
            a.real
        )
        &
        np.isfinite(
            a.imag
        )
        &
        np.isfinite(
            b.real
        )
        &
        np.isfinite(
            b.imag
        )
    )

    if not np.any(
        finite
    ):
        return 0.0

    return float(
        np.max(
            np.abs(
                a[
                    finite
                ]
                -
                b[
                    finite
                ]
            )
        )
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--sample",
        type=int,
        default=16000,
    )

    ap.add_argument(
        "--coherence-sample",
        type=int,
        default=8000,
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

    ndate = len(
        stack.dates
    )

    processing = (
        Path(
            paths.output_dir
        )
        /
        "processing"
    )

    stats = (
        processing
        /
        "ds_statistics"
    )

    scale2 = np.load(
        stats
        /
        "rayleigh_scale2.npy",
        mmap_mode="r",
    )

    raw_valid = np.load(
        stats
        /
        "raw_valid.npy",
        mmap_mode="r",
    )

    ps_raw = np.load(
        stats
        /
        "ps_mask.npy",
        mmap_mode="r",
    )

    geom = np.load(
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    yxt = np.load(
        processing
        /
        "cache"
        /
        "phase_corrected_yxt.npy",
        mmap_mode="r",
    )

    with np.load(
        processing
        /
        "moraine_center_prior.npz"
    ) as z:

        prior = z[
            "candidate_mask"
        ].astype(
            bool
        )

    valid = (
        np.asarray(
            raw_valid
        )
        &
        np.asarray(
            geom
        )
    )

    ps = (
        np.asarray(
            ps_raw
        )
        &
        valid
    )

    center = (
        prior
        &
        valid
        &
        ~ps
    )

    rr, cc = np.where(
        center
    )

    rr = rr[
        :args.sample
    ].astype(
        np.int32,
        copy=False,
    )

    cc = cc[
        :args.sample
    ].astype(
        np.int32,
        copy=False,
    )

    half_row = 5
    half_col = 11
    wh = 11
    ww = 23
    alpha = 0.005
    threshold = glrt_threshold(
        alpha
    )

    print(
        "=" * 92
    )

    print(
        "pyPSDS-GAMMA U2.3a SHP backend benchmark"
    )

    print(
        "=" * 92
    )

    print(
        "dates             :",
        ndate,
    )

    print(
        "centers           :",
        rr.size,
    )

    print(
        "window            :",
        f"{wh} x {ww}",
    )

    print()

    # --------------------------------------------------------
    # Frozen exact reference.
    # --------------------------------------------------------

    ts = time.perf_counter()

    ref = legacy_support(
        scale2,
        valid,
        ps,
        rr,
        cc,
        half_row=half_row,
        half_col=half_col,
        alpha=alpha,
        ndate=ndate,
    )

    legacy_s = (
        time.perf_counter()
        -
        ts
    )

    ref_K = np.sum(
        ref,
        axis=(1, 2),
    ).astype(
        np.int16
    )

    print(
        "legacy exact      :",
        f"{legacy_s:.3f} s",
    )

    # --------------------------------------------------------
    # Existing Numba bitset: diagnose failure.
    # --------------------------------------------------------

    # warm
    _ = glrt_support_bitset(
        scale2,
        valid,
        ps,
        rr[:1],
        cc[:1],
        half_row=half_row,
        half_col=half_col,
        threshold=threshold,
        nslc=ndate,
    )

    ts = time.perf_counter()

    (
        numba_bits,
        numba_K,
    ) = glrt_support_bitset(
        scale2,
        valid,
        ps,
        rr,
        cc,
        half_row=half_row,
        half_col=half_col,
        threshold=threshold,
        nslc=ndate,
    )

    numba_s = (
        time.perf_counter()
        -
        ts
    )

    numba_bool = (
        unpack_support_bitset(
            numba_bits,
            wh,
            ww,
        )
    )

    numba_diff = mismatch_summary(
        ref,
        numba_bool,
    )

    print()
    print(
        "Numba scalar GLRT"
    )

    print(
        "  seconds                   :",
        f"{numba_s:.3f}",
    )

    for k, v in numba_diff.items():
        print(
            f"  {k:28s}:",
            v,
        )

    # --------------------------------------------------------
    # Exact packed worker benchmark.
    # --------------------------------------------------------

    results = []

    print()
    print(
        f"{'workers':>8s}"
        f"{'chunk':>8s}"
        f"{'seconds':>12s}"
        f"{'speedup':>12s}"
        f"{'bits_bad':>12s}"
        f"{'K_bad':>10s}"
    )

    print(
        "-" * 68
    )

    layouts = [
        (1, 16000),
        (2, 4096),
        (4, 4096),
        (8, 2048),
        (16, 1024),
        (32, 512),
    ]

    best_bits = None

    for workers, chunk in layouts:

        ts = time.perf_counter()

        (
            bits,
            K,
        ) = glrt_support_exact_packed(
            scale2,
            valid,
            ps,
            rr,
            cc,
            half_row=half_row,
            half_col=half_col,
            alpha=alpha,
            nslc=ndate,
            workers=workers,
            chunk_size=chunk,
        )

        seconds = (
            time.perf_counter()
            -
            ts
        )

        b = unpack_support_bitset(
            bits,
            wh,
            ww,
        )

        q = mismatch_summary(
            ref,
            b,
        )

        parity = (
            q[
                "support_mismatch_bits"
            ]
            ==
            0
            and
            q[
                "shp_count_bad"
            ]
            ==
            0
        )

        result = {
            "workers":
                workers,

            "chunk":
                chunk,

            "seconds":
                seconds,

            "speedup":
                legacy_s
                /
                seconds,

            **q,

            "parity":
                parity,
        }

        results.append(
            result
        )

        print(
            f"{workers:8d}"
            f"{chunk:8d}"
            f"{seconds:12.3f}"
            f"{legacy_s/seconds:12.2f}"
            f"{q['support_mismatch_bits']:12d}"
            f"{q['shp_count_bad']:10d}"
        )

    valid_results = [
        x
        for x in results
        if x[
            "parity"
        ]
    ]

    if not valid_results:
        raise RuntimeError(
            "No exact packed SHP layout passed parity."
        )

    winner = min(
        valid_results,
        key=lambda x:
            x[
                "seconds"
            ],
    )

    # Rebuild winner once for coherence test.
    (
        exact_bits,
        exact_K,
    ) = glrt_support_exact_packed(
        scale2,
        valid,
        ps,
        rr,
        cc,
        half_row=half_row,
        half_col=half_col,
        alpha=alpha,
        nslc=ndate,
        workers=winner[
            "workers"
        ],
        chunk_size=winner[
            "chunk"
        ],
    )

    # --------------------------------------------------------
    # Isolate the coherence backend independently of GLRT.
    # --------------------------------------------------------

    good = (
        ref_K
        >=
        48
    )

    ids = np.flatnonzero(
        good
    )[
        :args.coherence_sample
    ]

    gr = rr[
        ids
    ]

    gc = cc[
        ids
    ]

    gs_bool = ref[
        ids
    ]

    gs_bits = exact_bits[
        ids
    ]

    pairs = image_pairs(
        ndate
    )

    pi = pairs[
        :,
        0,
    ]

    pj = pairs[
        :,
        1,
    ]

    print()
    print(
        "coherence sample :",
        gr.size,
    )

    # warm legacy
    _ = compressed_coherence(
        yxt,
        gr[:1],
        gc[:1],
        gs_bool[:1],
        pi,
        pj,
    )

    ts = time.perf_counter()

    coh_ref = compressed_coherence(
        yxt,
        gr,
        gc,
        gs_bool,
        pi,
        pj,
    )

    legacy_coh_s = (
        time.perf_counter()
        -
        ts
    )

    # warm bitset
    _ = compressed_coherence_bitset(
        yxt,
        gr[:1],
        gc[:1],
        gs_bits[:1],
        pi,
        pj,
        half_row=half_row,
        half_col=half_col,
    )

    ts = time.perf_counter()

    coh_bit = compressed_coherence_bitset(
        yxt,
        gr,
        gc,
        gs_bits,
        pi,
        pj,
        half_row=half_row,
        half_col=half_col,
    )

    bitset_coh_s = (
        time.perf_counter()
        -
        ts
    )

    coh_diff = max_complex_diff(
        coh_ref,
        coh_bit,
    )

    # Exact-packed -> bool -> existing coherence.
    ts = time.perf_counter()

    exact_bool = unpack_support_bitset(
        gs_bits,
        wh,
        ww,
    )

    unpack_s = (
        time.perf_counter()
        -
        ts
    )

    ts = time.perf_counter()

    coh_unpack = compressed_coherence(
        yxt,
        gr,
        gc,
        exact_bool,
        pi,
        pj,
    )

    unpack_coh_s = (
        time.perf_counter()
        -
        ts
    )

    unpack_diff = max_complex_diff(
        coh_ref,
        coh_unpack,
    )

    print()
    print(
        "COHERENCE BACKEND ISOLATION"
    )

    print(
        "  legacy bool coherence      :",
        f"{legacy_coh_s:.3f} s",
    )

    print(
        "  bitset coherence           :",
        f"{bitset_coh_s:.3f} s",
    )

    print(
        "  bitset max abs difference  :",
        f"{coh_diff:.9e}",
    )

    print(
        "  unpack packed support      :",
        f"{unpack_s:.3f} s",
    )

    print(
        "  unpack+legacy coherence    :",
        f"{unpack_coh_s:.3f} s",
    )

    print(
        "  unpack max abs difference  :",
        f"{unpack_diff:.9e}",
    )

    out = (
        processing
        /
        "ds_tiled"
        /
        "shp_backend_benchmark.json"
    )

    report = {
        "format":
            "pyPSDS-GAMMA-shp-backend-benchmark-v1",

        "sample":
            int(
                rr.size
            ),

        "legacy_seconds":
            legacy_s,

        "numba_scalar":
            {
                "seconds":
                    numba_s,

                **numba_diff,
            },

        "exact_packed_results":
            results,

        "winner":
            winner,

        "coherence":
            {
                "sample":
                    int(
                        gr.size
                    ),

                "legacy_seconds":
                    legacy_coh_s,

                "bitset_seconds":
                    bitset_coh_s,

                "bitset_max_abs_difference":
                    coh_diff,

                "unpack_seconds":
                    unpack_s,

                "unpack_legacy_seconds":
                    unpack_coh_s,

                "unpack_max_abs_difference":
                    unpack_diff,
            },
    }

    out.write_text(
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
    print(
        "=" * 92
    )

    print(
        "EXACT PACKED WINNER"
    )

    print(
        "=" * 92
    )

    for k, v in winner.items():
        print(
            f"{k:30s}:",
            v,
        )

    print(
        "saved                         :",
        out,
    )

    print()
    print(
        "U2.3a SHP BACKEND BENCHMARK: PASS"
    )


if __name__ == "__main__":
    main()
