#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def wrap(x):
    return np.arctan2(
        np.sin(x),
        np.cos(x),
    ).astype(
        np.float32,
        copy=False,
    )


def load_itab(path, ndate):
    edges = []

    for raw in path.read_text().splitlines():

        f = raw.split()

        if len(f) < 2:
            continue

        i = int(f[0]) - 1
        j = int(f[1]) - 1

        if not (
            0 <= i < ndate
            and
            0 <= j < ndate
        ):
            raise RuntimeError(
                f"Invalid ITAB edge: {raw}"
            )

        edges.append(
            (i, j)
        )

    return edges


def summarize_abs(x):
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    a = np.abs(x)

    return {
        "count": int(a.size),

        "mean_abs":
            float(
                np.mean(
                    a,
                    dtype=np.float64,
                )
            ),

        "median_abs":
            float(
                np.median(a)
            ),

        "p75_abs":
            float(
                np.quantile(
                    a,
                    0.75,
                )
            ),

        "p90_abs":
            float(
                np.quantile(
                    a,
                    0.90,
                )
            ),

        "p95_abs":
            float(
                np.quantile(
                    a,
                    0.95,
                )
            ),

        "p99_abs":
            float(
                np.quantile(
                    a,
                    0.99,
                )
            ),

        "max_abs":
            float(
                np.max(a)
            ),

        "frac_gt_pi_2":
            float(
                np.mean(
                    a > np.pi / 2
                )
            ),

        "frac_gt_2pi_3":
            float(
                np.mean(
                    a > 2*np.pi / 3
                )
            ),

        "frac_gt_0p9pi":
            float(
                np.mean(
                    a > 0.9*np.pi
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
        "--edge-batch",
        type=int,
        default=200000,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        _,
    ) = open_from_config(
        args.config
    )

    outroot = (
        Path(paths.output_dir)
        / "processing"
    )

    pps_dir = (
        outroot
        / "point_phase_stack"
    )

    netdir = (
        outroot
        / "network"
    )

    graphdir = (
        outroot
        / "spatial_graph"
    )

    outdir = (
        outroot
        / "spatial_phase_gradient_quality"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase = np.load(
        pps_dir / "phase_rad.npy",
        mmap_mode="r",
    )

    npoint, ndate = phase.shape

    temporal_edges = load_itab(
        netdir / "network.itab",
        ndate,
    )

    npair = len(
        temporal_edges
    )

    local_u = np.load(
        graphdir / "local_u.npy",
        mmap_mode="r",
    )

    local_v = np.load(
        graphdir / "local_v.npy",
        mmap_mode="r",
    )

    local_dist = np.load(
        graphdir / "local_distance_m.npy",
        mmap_mode="r",
    )

    anchor_u = np.load(
        graphdir / "anchor_u.npy",
    )

    anchor_v = np.load(
        graphdir / "anchor_v.npy",
    )

    anchor_class = np.load(
        graphdir / "anchor_class.npy",
    )

    anchor_radius = np.load(
        graphdir / "anchor_radius.npy",
    )

    anchor_dist = np.load(
        graphdir / "anchor_distance_m.npy",
    )

    if local_u.size != local_v.size:
        raise RuntimeError(
            "local edge length mismatch"
        )

    print("=" * 88)
    print(
        "Spatial wrapped-phase gradient quality"
    )
    print("=" * 88)

    print(
        f"config                     : {config_path}"
    )

    print(
        f"points                     : {npoint:,}"
    )

    print(
        f"acquisitions               : {ndate}"
    )

    print(
        f"IFG pairs                  : {npair}"
    )

    print(
        f"local edges                : {local_u.size:,}"
    )

    print(
        f"anchor edges               : {anchor_u.size:,}"
    )

    print(
        f"edge batch                 : {args.edge_batch:,}"
    )

    print()

    # ========================================================
    # Per-IFG statistics
    # ========================================================

    pair_rows = []

    # Global histogram over |gradient|.
    nbins = 180

    hist_edges = np.linspace(
        0.0,
        np.pi,
        nbins + 1,
        dtype=np.float64,
    )

    hist_local = np.zeros(
        nbins,
        dtype=np.int64,
    )

    # Anchors are tiny, keep all observations.
    anchor_all = {
        1: [],
        2: [],
        3: [],
    }

    for pair_id, (
        i,
        j,
    ) in enumerate(
        temporal_edges,
        start=1,
    ):

        # acquisition -> virtual IFG, point-wise
        ifg = wrap(
            np.asarray(
                phase[:, j],
                dtype=np.float32,
            )
            -
            np.asarray(
                phase[:, i],
                dtype=np.float32,
            )
        )

        # ----------------------------------------------------
        # Local edges: stream in batches
        # ----------------------------------------------------

        local_count = 0
        local_sum_abs = 0.0

        gt_pi2 = 0
        gt_2pi3 = 0
        gt_09pi = 0

        # Quantile sample.
        # Deterministic stride sample to avoid storing
        # 2.3 million values for every IFG.
        sample_parts = []

        for e0 in range(
            0,
            local_u.size,
            args.edge_batch,
        ):

            e1 = min(
                e0 + args.edge_batch,
                local_u.size,
            )

            u = np.asarray(
                local_u[e0:e1],
                dtype=np.int32,
            )

            v = np.asarray(
                local_v[e0:e1],
                dtype=np.int32,
            )

            g = wrap(
                ifg[v]
                -
                ifg[u]
            )

            a = np.abs(g)

            local_count += int(
                a.size
            )

            local_sum_abs += float(
                np.sum(
                    a,
                    dtype=np.float64,
                )
            )

            gt_pi2 += int(
                np.count_nonzero(
                    a > np.pi/2
                )
            )

            gt_2pi3 += int(
                np.count_nonzero(
                    a > 2*np.pi/3
                )
            )

            gt_09pi += int(
                np.count_nonzero(
                    a > 0.9*np.pi
                )
            )

            h, _ = np.histogram(
                a,
                bins=hist_edges,
            )

            hist_local += h

            # ~1/64 deterministic sample.
            sample_parts.append(
                a[::64].copy()
            )

        sample = np.concatenate(
            sample_parts
        )

        # ----------------------------------------------------
        # Anchors
        # ----------------------------------------------------

        anchor_grad = wrap(
            ifg[anchor_v]
            -
            ifg[anchor_u]
        )

        anchor_stats = {}

        for cls in (1, 2, 3):

            m = (
                anchor_class == cls
            )

            x = anchor_grad[m]

            anchor_stats[
                cls
            ] = summarize_abs(
                x
            )

            anchor_all[
                cls
            ].append(
                x.copy()
            )

        # ----------------------------------------------------
        # Per-pair summary
        # ----------------------------------------------------

        row = {
            "pair_id": pair_id,

            "i1": i + 1,
            "j1": j + 1,

            "date1":
                stack.dates[i],

            "date2":
                stack.dates[j],

            "local_mean_abs_rad":
                local_sum_abs
                /
                local_count,

            "local_median_abs_rad":
                float(
                    np.median(
                        sample
                    )
                ),

            "local_p90_abs_rad":
                float(
                    np.quantile(
                        sample,
                        0.90,
                    )
                ),

            "local_p95_abs_rad":
                float(
                    np.quantile(
                        sample,
                        0.95,
                    )
                ),

            "local_p99_abs_rad":
                float(
                    np.quantile(
                        sample,
                        0.99,
                    )
                ),

            "local_frac_gt_pi_2":
                gt_pi2
                /
                local_count,

            "local_frac_gt_2pi_3":
                gt_2pi3
                /
                local_count,

            "local_frac_gt_0p9pi":
                gt_09pi
                /
                local_count,
        }

        for cls, prefix in (
            (1, "anchor_normal"),
            (2, "anchor_extended"),
            (3, "anchor_long"),
        ):

            s = anchor_stats[
                cls
            ]

            row[
                prefix
                + "_median_abs_rad"
            ] = s[
                "median_abs"
            ]

            row[
                prefix
                + "_p90_abs_rad"
            ] = s[
                "p90_abs"
            ]

            row[
                prefix
                + "_frac_gt_pi_2"
            ] = s[
                "frac_gt_pi_2"
            ]

        pair_rows.append(
            row
        )

        if (
            pair_id == 1
            or
            pair_id % 10 == 0
            or
            pair_id == npair
        ):

            print(
                f"  IFG "
                f"{pair_id:3d}/{npair:3d} "
                f"{stack.dates[i]}-"
                f"{stack.dates[j]}: "
                f"median|g|="
                f"{row['local_median_abs_rad']:.3f}, "
                f"p95="
                f"{row['local_p95_abs_rad']:.3f}, "
                f">pi/2="
                f"{100*row['local_frac_gt_pi_2']:.2f}%"
            )

    # ========================================================
    # Global summaries
    # ========================================================

    # Histogram-based local approximate quantiles.
    total_hist = hist_local.sum()

    cdf = np.cumsum(
        hist_local
    ) / total_hist

    def hist_quantile(q):

        idx = int(
            np.searchsorted(
                cdf,
                q,
            )
        )

        idx = min(
            idx,
            nbins - 1,
        )

        return float(
            0.5
            * (
                hist_edges[idx]
                +
                hist_edges[idx+1]
            )
        )

    local_global = {
        "observations":
            int(total_hist),

        "median_abs_rad":
            hist_quantile(0.50),

        "p75_abs_rad":
            hist_quantile(0.75),

        "p90_abs_rad":
            hist_quantile(0.90),

        "p95_abs_rad":
            hist_quantile(0.95),

        "p99_abs_rad":
            hist_quantile(0.99),

        "frac_gt_pi_2":
            float(
                hist_local[
                    hist_edges[:-1]
                    >= np.pi/2
                ].sum()
                /
                total_hist
            ),

        "frac_gt_2pi_3":
            float(
                hist_local[
                    hist_edges[:-1]
                    >= 2*np.pi/3
                ].sum()
                /
                total_hist
            ),

        "frac_gt_0p9pi":
            float(
                hist_local[
                    hist_edges[:-1]
                    >= 0.9*np.pi
                ].sum()
                /
                total_hist
            ),
    }

    anchor_global = {}

    for cls, name in (
        (1, "normal"),
        (2, "extended"),
        (3, "long"),
    ):

        x = np.concatenate(
            anchor_all[
                cls
            ]
        )

        anchor_global[
            name
        ] = summarize_abs(
            x
        )

    # ========================================================
    # Spatial geometry summary
    # ========================================================

    local_dist_q = np.quantile(
        np.asarray(
            local_dist,
            dtype=np.float32,
        ),
        [
            0,
            0.5,
            0.9,
            0.95,
            0.99,
            1.0,
        ],
    )

    # ========================================================
    # Output
    # ========================================================

    csv_path = (
        outdir
        / "per_ifg_spatial_gradient_qa.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                pair_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            pair_rows
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-spatial-phase-gradient-quality-v1.0",

        "points":
            int(npoint),

        "ifg_pairs":
            int(npair),

        "spatial_graph": {
            "local_edges":
                int(local_u.size),

            "anchor_edges":
                int(anchor_u.size),

            "local_distance_m":
                {
                    "min":
                        float(
                            local_dist_q[0]
                        ),

                    "median":
                        float(
                            local_dist_q[1]
                        ),

                    "p90":
                        float(
                            local_dist_q[2]
                        ),

                    "p95":
                        float(
                            local_dist_q[3]
                        ),

                    "p99":
                        float(
                            local_dist_q[4]
                        ),

                    "max":
                        float(
                            local_dist_q[5]
                        ),
                },
        },

        "local_gradient":
            local_global,

        "anchors": {
            "normal":
                anchor_global[
                    "normal"
                ],

            "extended":
                anchor_global[
                    "extended"
                ],

            "long":
                anchor_global[
                    "long"
                ],
        },
    }

    json_path = (
        outdir
        / "spatial_gradient_quality.json"
    )

    json_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 88)
    print(
        "Global local-edge phase-gradient summary"
    )
    print("=" * 88)

    print(
        f"observations             : "
        f"{local_global['observations']:,}"
    )

    print(
        f"|g| median/p90/p95/p99  : "
        f"{local_global['median_abs_rad']:.3f} / "
        f"{local_global['p90_abs_rad']:.3f} / "
        f"{local_global['p95_abs_rad']:.3f} / "
        f"{local_global['p99_abs_rad']:.3f} rad"
    )

    print(
        f"|g| > pi/2             : "
        f"{100*local_global['frac_gt_pi_2']:.3f}%"
    )

    print(
        f"|g| > 2pi/3            : "
        f"{100*local_global['frac_gt_2pi_3']:.3f}%"
    )

    print(
        f"|g| > 0.9pi            : "
        f"{100*local_global['frac_gt_0p9pi']:.3f}%"
    )

    print()
    print("=" * 88)
    print(
        "Residual-anchor phase-gradient summary"
    )
    print("=" * 88)

    for name in (
        "normal",
        "extended",
        "long",
    ):

        s = anchor_global[
            name
        ]

        print(
            f"{name:8s}: "
            f"N={s['count']:,}, "
            f"median={s['median_abs']:.3f}, "
            f"p90={s['p90_abs']:.3f}, "
            f"p95={s['p95_abs']:.3f}, "
            f">pi/2="
            f"{100*s['frac_gt_pi_2']:.2f}%"
        )

    print()
    print(
        f"per-IFG table           : "
        f"{csv_path}"
    )

    print(
        f"manifest                : "
        f"{json_path}"
    )

    print()
    print(
        "STEP 08i STATUS: PASS"
    )


if __name__ == "__main__":
    main()
