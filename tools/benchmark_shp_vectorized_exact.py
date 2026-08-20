#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config
from pypsds.selection.shp import (
    glrt_statistic,
    glrt_threshold,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    prepare_glrt_window_context,
    glrt_support_vectorized_exact,
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
    B = rows.size
    wh = 2 * half_row + 1
    ww = 2 * half_col + 1

    out = np.zeros(
        (B, wh, ww),
        dtype=np.bool_,
    )

    center_scale = (
        scale2[
            rows,
            cols,
        ].astype(
            np.float64,
            copy=False,
        )
    )

    threshold = glrt_threshold(
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
            if dy == 0 and dx == 0:
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

            if not np.any(inside):
                continue

            ids = np.flatnonzero(
                inside
            )

            r2 = rr[ids]
            c2 = cc[ids]

            ngood = (
                valid[r2, c2]
                &
                ~ps[r2, c2]
            )

            if not np.any(ngood):
                continue

            ids2 = ids[ngood]

            r3 = rr[ids2]
            c3 = cc[ids2]

            stat = glrt_statistic(
                center_scale[ids2],
                scale2[r3, c3],
                nslc=ndate,
            )

            out[
                ids2,
                ky,
                kx,
            ] = (
                np.isfinite(stat)
                &
                (stat < threshold)
            )

    return out


def mismatch(
    reference,
    candidate,
):
    d = (
        reference
        !=
        candidate
    )

    return {
        "support_mismatch_bits":
            int(
                np.count_nonzero(
                    d
                )
            ),

        "support_mismatch_centers":
            int(
                np.count_nonzero(
                    np.any(
                        d,
                        axis=(1, 2),
                    )
                )
            ),

        "shp_count_bad":
            int(
                np.count_nonzero(
                    np.sum(
                        reference,
                        axis=(1, 2),
                    )
                    !=
                    np.sum(
                        candidate,
                        axis=(1, 2),
                    )
                )
            ),
    }


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
        Path(paths.output_dir)
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

    centers = (
        prior
        &
        valid
        &
        ~ps
    )

    rr, cc = np.where(
        centers
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

    hr = 5
    hc = 11

    print(
        "=" * 88
    )

    print(
        "pyPSDS-GAMMA U2.3b exact-vectorized GLRT"
    )

    print(
        "=" * 88
    )

    print(
        "dates       :",
        ndate,
    )

    print(
        "centers     :",
        rr.size,
    )

    print(
        "window      :",
        "11 x 23",
    )

    # --------------------------------------------------------
    # Frozen reference
    # --------------------------------------------------------

    ts = time.perf_counter()

    ref = legacy_support(
        scale2,
        valid,
        ps,
        rr,
        cc,
        half_row=hr,
        half_col=hc,
        alpha=0.005,
        ndate=ndate,
    )

    legacy_seconds = (
        time.perf_counter()
        -
        ts
    )

    print(
        "legacy exact:",
        f"{legacy_seconds:.4f} s",
    )

    # Prepare window views outside timed batch operation.
    # In production these are prepared once per tile.
    ts = time.perf_counter()

    ctx = prepare_glrt_window_context(
        scale2,
        valid,
        ps,
        half_row=hr,
        half_col=hc,
    )

    prepare_seconds = (
        time.perf_counter()
        -
        ts
    )

    print(
        "context prep:",
        f"{prepare_seconds:.4f} s",
    )

    layouts = [
        1024,
        2048,
        4096,
        8192,
        16000,
    ]

    results = []

    print()
    print(
        f"{'block':>10s}"
        f"{'seconds':>12s}"
        f"{'speedup':>12s}"
        f"{'bits_bad':>12s}"
        f"{'K_bad':>10s}"
    )

    print(
        "-" * 58
    )

    for block in layouts:

        # warm one small block before measurement
        _ = glrt_support_vectorized_exact(
            ctx,
            rr[:min(256, rr.size)],
            cc[:min(256, cc.size)],
            alpha=0.005,
            nslc=ndate,
            block_size=min(
                block,
                256,
            ),
        )

        ts = time.perf_counter()

        support, K = (
            glrt_support_vectorized_exact(
                ctx,
                rr,
                cc,
                alpha=0.005,
                nslc=ndate,
                block_size=block,
            )
        )

        seconds = (
            time.perf_counter()
            -
            ts
        )

        q = mismatch(
            ref,
            support,
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
            "block_size":
                block,

            "seconds":
                seconds,

            "speedup":
                legacy_seconds
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
            f"{block:10d}"
            f"{seconds:12.4f}"
            f"{legacy_seconds/seconds:12.2f}"
            f"{q['support_mismatch_bits']:12d}"
            f"{q['shp_count_bad']:10d}"
        )

    valid_results = [
        x
        for x in results
        if x["parity"]
    ]

    if not valid_results:
        raise RuntimeError(
            "No vectorized exact layout passed parity."
        )

    winner = min(
        valid_results,
        key=lambda x:
            x["seconds"],
    )

    out = (
        processing
        /
        "ds_tiled"
        /
        "shp_vectorized_exact_benchmark.json"
    )

    out.write_text(
        json.dumps(
            {
                "format":
                    "pyPSDS-GAMMA-shp-vectorized-exact-v1",

                "sample":
                    int(
                        rr.size
                    ),

                "legacy_seconds":
                    legacy_seconds,

                "context_prepare_seconds":
                    prepare_seconds,

                "results":
                    results,

                "winner":
                    winner,
            },
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 88
    )

    print(
        "U2.3b WINNER"
    )

    print(
        "=" * 88
    )

    for k, v in winner.items():
        print(
            f"{k:28s}:",
            v,
        )

    print(
        "saved                       :",
        out,
    )

    print()

    print(
        "U2.3b EXACT VECTORIZED GLRT: PASS"
    )


if __name__ == "__main__":
    main()
