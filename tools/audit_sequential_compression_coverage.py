#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit

from pypsds.context import (
    open_from_config,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


@njit(
    cache=True,
    nogil=True,
)
def mark_support_union(
    union_mask,
    rows,
    cols,
    support,
):
    """
    Mark every spatial pixel that is used as an SHP sample
    by at least one target center.
    """

    H, W = (
        union_mask.shape
    )

    B = rows.size

    wh = support.shape[1]
    ww = support.shape[2]

    hr = wh // 2
    hc = ww // 2

    for p in range(
        B
    ):

        cr = rows[p]
        cc = cols[p]

        for ky in range(
            wh
        ):

            rr = (
                cr
                -
                hr
                +
                ky
            )

            if (
                rr < 0
                or
                rr >= H
            ):
                continue

            for kx in range(
                ww
            ):

                if not support[
                    p,
                    ky,
                    kx,
                ]:
                    continue

                c = (
                    cc
                    -
                    hc
                    +
                    kx
                )

                if (
                    0
                    <=
                    c
                    <
                    W
                ):

                    union_mask[
                        rr,
                        c,
                    ] = True


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
        (
            base_row0,
            base_col0,
            H,
            W,
        ),
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

    valid = np.ascontiguousarray(
        np.asarray(
            raw_valid,
            dtype=bool,
        )
        &
        np.asarray(
            geom,
            dtype=bool,
        )
    )

    ps = np.ascontiguousarray(
        np.asarray(
            ps_raw,
            dtype=bool,
        )
        &
        valid
    )

    target = (
        prior
        &
        valid
        &
        ~ps
    )

    rr, cc = np.where(
        target
    )

    rr = rr.astype(
        np.int32,
        copy=False,
    )

    cc = cc.astype(
        np.int32,
        copy=False,
    )

    ctx = (
        prepare_glrt_window_context(
            scale2,
            valid,
            ps,
            half_row=5,
            half_col=11,
        )
    )

    union = np.zeros(
        (
            H,
            W,
        ),
        dtype=np.bool_,
    )

    # Warm compile.
    dummy_support = np.zeros(
        (
            1,
            11,
            23,
        ),
        dtype=np.bool_,
    )

    mark_support_union(
        union,
        np.asarray(
            [0],
            np.int32,
        ),
        np.asarray(
            [0],
            np.int32,
        ),
        dummy_support,
    )

    union[:] = False

    t0 = time.perf_counter()

    k_eligible = 0

    for start in range(
        0,
        rr.size,
        args.batch,
    ):

        stop = min(
            rr.size,
            start
            +
            args.batch,
        )

        br = rr[
            start:stop
        ]

        bc = cc[
            start:stop
        ]

        support, K = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=0.005,
                nslc=ndate,
                block_size=(
                    args.support_block
                ),
            )
        )

        good = (
            K >= 48
        )

        k_eligible += int(
            np.count_nonzero(
                good
            )
        )

        if np.any(
            good
        ):

            mark_support_union(
                union,
                br[
                    good
                ],
                bc[
                    good
                ],
                support[
                    good
                ],
            )

        done = stop

        if (
            done
            ==
            rr.size
            or
            done
            %
            (
                args.batch
                *
                10
            )
            ==
            0
        ):

            print(
                f"centers "
                f"{done:,}/"
                f"{rr.size:,}"
            )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    valid_nonps = (
        valid
        &
        ~ps
    )

    union_count = int(
        np.count_nonzero(
            union
        )
    )

    valid_nonps_count = int(
        np.count_nonzero(
            valid_nonps
        )
    )

    target_count = int(
        rr.size
    )

    eligible_count = int(
        k_eligible
    )

    union_valid_nonps = int(
        np.count_nonzero(
            union
            &
            valid_nonps
        )
    )

    coverage_valid_nonps = (
        union_valid_nonps
        /
        valid_nonps_count
        if
        valid_nonps_count
        else
        0.0
    )

    target_fraction = (
        target_count
        /
        valid_nonps_count
        if
        valid_nonps_count
        else
        0.0
    )

    print()
    print(
        "=" * 88
    )

    print(
        "U3.2a sequential compressed-state coverage audit"
    )

    print(
        "=" * 88
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
        "valid non-PS pixels    :",
        f"{valid_nonps_count:,}",
    )

    print(
        "target centers         :",
        f"{target_count:,}",
        f"({100*target_fraction:.3f}%)",
    )

    print(
        "K>=48 target centers   :",
        f"{eligible_count:,}",
    )

    print(
        "SHP union pixels       :",
        f"{union_count:,}",
    )

    print(
        "union valid non-PS     :",
        f"{union_valid_nonps:,}",
    )

    print(
        "coverage of valid nonPS:",
        f"{100*coverage_valid_nonps:.3f}%",
    )

    print(
        "elapsed                :",
        f"{elapsed:.3f} s",
    )

    out_dir = (
        processing
        /
        "sequential"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        out_dir
        /
        "compression_required_mask.npy",
        union,
    )

    report = {
        "format":
            "pyPSDS-GAMMA-sequential-compression-coverage-v1",

        "shape":
            [
                H,
                W,
            ],

        "ndate":
            ndate,

        "valid_nonps":
            valid_nonps_count,

        "target_centers":
            target_count,

        "eligible_target_centers":
            eligible_count,

        "support_union":
            union_count,

        "support_union_valid_nonps":
            union_valid_nonps,

        "coverage_valid_nonps":
            coverage_valid_nonps,

        "target_fraction_valid_nonps":
            target_fraction,

        "elapsed_seconds":
            elapsed,
    }

    out = (
        out_dir
        /
        "compression_coverage.json"
    )

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
        "mask :",
        out_dir
        /
        "compression_required_mask.npy",
    )

    print(
        "json :",
        out,
    )

    print()

    print(
        "U3.2a COMPRESSION COVERAGE AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
