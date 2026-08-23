from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.context import open_from_config
from pypsds.filtering.goldstein import (
    goldstein_filter,
)


def wrap(
    x,
):
    return np.arctan2(
        np.sin(
            x
        ),
        np.cos(
            x
        ),
    ).astype(
        np.float32,
        copy=False,
    )


def load_itab(
    path: Path,
    ndate: int,
):
    edges = []

    for raw in path.read_text().splitlines():
        f = raw.split()

        if len(f) < 2:
            continue

        i = int(
            f[0]
        ) - 1

        j = int(
            f[1]
        ) - 1

        if not (
            0
            <=
            i
            <
            ndate
            and
            0
            <=
            j
            <
            ndate
            and
            i
            <
            j
        ):
            raise RuntimeError(
                f"invalid network edge: {raw}"
            )

        edges.append(
            (
                i,
                j,
            )
        )

    return edges


def quantiles(
    x,
):
    v = np.asarray(
        x,
        dtype=np.float64,
    )

    v = v[
        np.isfinite(
            v
        )
    ]

    if not v.size:
        return {
            "count": 0,
        }

    q = np.quantile(
        v,
        [
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ],
    )

    return {
        "count":
            int(
                v.size
            ),

        "q05":
            float(
                q[0]
            ),

        "q25":
            float(
                q[1]
            ),

        "median":
            float(
                q[2]
            ),

        "q75":
            float(
                q[3]
            ),

        "q95":
            float(
                q[4]
            ),

        "q99":
            float(
                q[5]
            ),

        "max":
            float(
                np.max(
                    v
                )
            ),
    }


def edge_metrics(
    phase,
    local_u,
    local_v,
    *,
    edge_batch,
):
    count = 0
    sum_abs = 0.0
    gt_pi2 = 0
    gt_2pi3 = 0
    gt_09pi = 0

    samples = []

    nedge = int(
        local_u.size
    )

    for e0 in range(
        0,
        nedge,
        int(
            edge_batch
        ),
    ):
        e1 = min(
            nedge,
            e0
            +
            int(
                edge_batch
            ),
        )

        u = np.asarray(
            local_u[
                e0:e1
            ],
            dtype=np.int32,
        )

        v = np.asarray(
            local_v[
                e0:e1
            ],
            dtype=np.int32,
        )

        g = wrap(
            phase[
                v
            ]
            -
            phase[
                u
            ]
        )

        a = np.abs(
            g
        )

        count += int(
            a.size
        )

        sum_abs += float(
            np.sum(
                a,
                dtype=np.float64,
            )
        )

        gt_pi2 += int(
            np.count_nonzero(
                a
                >
                np.pi
                /
                2
            )
        )

        gt_2pi3 += int(
            np.count_nonzero(
                a
                >
                2
                *
                np.pi
                /
                3
            )
        )

        gt_09pi += int(
            np.count_nonzero(
                a
                >
                0.9
                *
                np.pi
            )
        )

        # Deterministic quantile sample, same philosophy as production QA.
        samples.append(
            a[
                ::64
            ].copy()
        )

    sample = np.concatenate(
        samples
    )

    return {
        "count":
            count,

        "mean_abs_rad":
            float(
                sum_abs
                /
                count
            ),

        "median_abs_rad":
            float(
                np.median(
                    sample
                )
            ),

        "p90_abs_rad":
            float(
                np.quantile(
                    sample,
                    0.90,
                )
            ),

        "p95_abs_rad":
            float(
                np.quantile(
                    sample,
                    0.95,
                )
            ),

        "p99_abs_rad":
            float(
                np.quantile(
                    sample,
                    0.99,
                )
            ),

        "frac_gt_pi_2":
            float(
                gt_pi2
                /
                count
            ),

        "frac_gt_2pi_3":
            float(
                gt_2pi3
                /
                count
            ),

        "frac_gt_0p9pi":
            float(
                gt_09pi
                /
                count
            ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--alphas",
        default="0.3,0.5,0.7",
    )

    ap.add_argument(
        "--patch-size",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--representative-pairs",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--edge-batch",
        type=int,
        default=200000,
    )

    args = ap.parse_args()

    alphas = [
        float(
            x
        )
        for x in args.alphas.split(",")
        if x.strip()
    ]

    for a in alphas:
        if not (
            0.0
            <
            a
            <=
            1.0
        ):
            raise ValueError(
                "benchmark alpha must be in (0,1]"
            )

    (
        cfg,
        config_path,
        paths,
        stack,
        (
            _,
            _,
            H,
            W,
        ),
    ) = open_from_config(
        args.config
    )

    proc = (
        Path(
            paths.output_dir
        )
        /
        "processing"
    )

    outdir = (
        proc
        /
        "adaptive_filter_benchmark"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    linked = np.load(
        proc
        /
        "linked_phase.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    pps = (
        proc
        /
        "point_phase_stack"
    )

    phase_point = np.load(
        pps
        /
        "phase_rad.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    rows = np.asarray(
        np.load(
            pps
            /
            "rows.npy",
            allow_pickle=False,
        ),
        dtype=np.int32,
    )

    cols = np.asarray(
        np.load(
            pps
            /
            "cols.npy",
            allow_pickle=False,
        ),
        dtype=np.int32,
    )

    point_type = np.asarray(
        np.load(
            pps
            /
            "point_type.npy",
            allow_pickle=False,
        ),
        dtype=np.uint8,
    )

    point_tc = np.asarray(
        np.load(
            pps
            /
            "temporal_coherence.npy",
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    local_u = np.load(
        proc
        /
        "spatial_graph"
        /
        "local_u.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    local_v = np.load(
        proc
        /
        "spatial_graph"
        /
        "local_v.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    edges = load_itab(
        proc
        /
        "network"
        /
        "network.itab",
        len(
            stack.dates
        ),
    )

    if linked.shape != (
        len(
            stack.dates
        ),
        H,
        W,
    ):
        raise RuntimeError(
            "linked_phase shape mismatch"
        )

    # ----------------------------------------------------------
    # Representative temporal-network pairs:
    # evenly span the existing raw spatial-gradient difficulty ranking.
    # ----------------------------------------------------------
    qa_csv = (
        proc
        /
        "spatial_phase_gradient_quality"
        /
        "per_ifg_spatial_gradient_qa.csv"
    )

    rows_qa = []

    with qa_csv.open(
        encoding="utf-8"
    ) as f:
        for row in csv.DictReader(
            f
        ):
            rows_qa.append(
                {
                    "pair_id":
                        int(
                            row[
                                "pair_id"
                            ]
                        ),

                    "frac_gt_pi2":
                        float(
                            row[
                                "local_frac_gt_pi_2"
                            ]
                        ),

                    "p95":
                        float(
                            row[
                                "local_p95_abs_rad"
                            ]
                        ),
                }
            )

    rows_qa.sort(
        key=lambda r:
            (
                r[
                    "frac_gt_pi2"
                ],
                r[
                    "p95"
                ],
                r[
                    "pair_id"
                ],
            )
    )

    m = min(
        int(
            args.representative_pairs
        ),
        len(
            rows_qa
        ),
    )

    pick_pos = np.unique(
        np.linspace(
            0,
            len(
                rows_qa
            )
            -
            1,
            num=m,
            dtype=np.int32,
        )
    )

    selected = [
        rows_qa[
            int(
                q
            )
        ]
        for q in pick_pos
    ]

    print("=" * 92)
    print("P10A adaptive wrapped-IFG filter benchmark")
    print("=" * 92)
    print("config              :", config_path)
    print("scene               :", f"{H} x {W}")
    print("points              :", f"{rows.size:,}")
    print("network pairs       :", len(edges))
    print("representative pairs:", len(selected))
    print("alphas              :", alphas)
    print("patch size          :", args.patch_size)
    print()

    records = []

    # High-quality DS phase is the key preservation diagnostic.
    high_tc_ds = (
        (
            point_type
            ==
            2
        )
        &
        np.isfinite(
            point_tc
        )
        &
        (
            point_tc
            >=
            0.90
        )
    )

    for pair_no, pair_info in enumerate(
        selected,
        start=1,
    ):
        pair_id = int(
            pair_info[
                "pair_id"
            ]
        )

        i, j = edges[
            pair_id
            -
            1
        ]

        print(
            "-" * 92
        )
        print(
            f"[{pair_no:02d}/{len(selected):02d}] "
            f"pair {pair_id:03d}: "
            f"{stack.dates[i]} -> {stack.dates[j]}"
        )

        zi = np.asarray(
            linked[
                i
            ],
            dtype=np.complex64,
        )

        zj = np.asarray(
            linked[
                j
            ],
            dtype=np.complex64,
        )

        valid = (
            np.isfinite(
                zi.real
            )
            &
            np.isfinite(
                zi.imag
            )
            &
            np.isfinite(
                zj.real
            )
            &
            np.isfinite(
                zj.imag
            )
        )

        raw_raster = np.zeros(
            (
                H,
                W,
            ),
            dtype=np.complex64,
        )

        raw_raster[
            valid
        ] = (
            zj[
                valid
            ]
            *
            np.conj(
                zi[
                    valid
                ]
            )
        )

        # Normalize non-zero support to unit wrapped phase.
        mag = np.abs(
            raw_raster
        )

        nz = (
            mag
            >
            0
        )

        raw_raster[
            nz
        ] /= mag[
            nz
        ]

        raw_point = wrap(
            np.asarray(
                phase_point[
                    :,
                    j
                ],
                dtype=np.float32,
            )
            -
            np.asarray(
                phase_point[
                    :,
                    i
                ],
                dtype=np.float32,
            )
        )

        raster_raw_point = np.angle(
            raw_raster[
                rows,
                cols,
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        consistency = np.abs(
            wrap(
                raster_raw_point
                -
                raw_point
            )
        )

        if float(
            np.nanmax(
                consistency
            )
        ) > 5.0e-5:
            raise RuntimeError(
                "linked_phase / PointPhaseStack virtual-IFG "
                f"sign/reference mismatch for pair {pair_id}: "
                f"max={float(np.nanmax(consistency))}"
            )

        raw_metrics = edge_metrics(
            raw_point,
            local_u,
            local_v,
            edge_batch=(
                args.edge_batch
            ),
        )

        records.append(
            {
                "pair_id":
                    pair_id,

                "date1":
                    str(
                        stack.dates[
                            i
                        ]
                    ),

                "date2":
                    str(
                        stack.dates[
                            j
                        ]
                    ),

                "alpha":
                    0.0,

                "patch_size":
                    int(
                        args.patch_size
                    ),

                "valid_raster_fraction":
                    float(
                        np.mean(
                            valid
                        )
                    ),

                "filtered_point_valid_fraction":
                    1.0,

                "gradient":
                    raw_metrics,

                "phase_change_all":
                    {
                        "count":
                            int(
                                raw_point.size
                            ),
                        "median":
                            0.0,
                        "q95":
                            0.0,
                        "q99":
                            0.0,
                        "max":
                            0.0,
                    },

                "phase_change_high_tc_ds":
                    {
                        "count":
                            int(
                                np.count_nonzero(
                                    high_tc_ds
                                )
                            ),
                        "median":
                            0.0,
                        "q95":
                            0.0,
                        "q99":
                            0.0,
                        "max":
                            0.0,
                    },

                "elapsed_seconds":
                    0.0,
            }
        )

        for alpha in alphas:
            t0 = perf_counter()

            filtered = goldstein_filter(
                raw_raster,
                alpha=alpha,
                patch_size=(
                    args.patch_size
                ),
            )

            elapsed = (
                perf_counter()
                -
                t0
            )

            fz = filtered[
                rows,
                cols,
            ]

            fvalid = (
                np.isfinite(
                    fz.real
                )
                &
                np.isfinite(
                    fz.imag
                )
                &
                (
                    np.abs(
                        fz
                    )
                    >
                    0
                )
            )

            filtered_point = np.full(
                rows.size,
                np.nan,
                dtype=np.float32,
            )

            filtered_point[
                fvalid
            ] = np.angle(
                fz[
                    fvalid
                ]
            ).astype(
                np.float32,
                copy=False,
            )

            if not np.all(
                fvalid
            ):
                # Keep benchmark metric domain deterministic. Missing filtered
                # values are not silently substituted for unfiltered values.
                valid_edges = (
                    fvalid[
                        np.asarray(
                            local_u,
                            dtype=np.int32,
                        )
                    ]
                    &
                    fvalid[
                        np.asarray(
                            local_v,
                            dtype=np.int32,
                        )
                    ]
                )

                if float(
                    np.mean(
                        valid_edges
                    )
                ) < 0.999:
                    raise RuntimeError(
                        "Goldstein filtering lost too many PointPhaseStack "
                        f"values for pair {pair_id}, alpha={alpha}"
                    )

                # In the very small number of missing locations, use raw phase
                # ONLY to avoid NaNs in graph metrics and record the validity
                # fraction explicitly. This does not write production products.
                filtered_point[
                    ~fvalid
                ] = raw_point[
                    ~fvalid
                ]

            filt_metrics = edge_metrics(
                filtered_point,
                local_u,
                local_v,
                edge_batch=(
                    args.edge_batch
                ),
            )

            change = np.abs(
                wrap(
                    filtered_point
                    -
                    raw_point
                )
            )

            change_hq = change[
                high_tc_ds
            ]

            row = {
                "pair_id":
                    pair_id,

                "date1":
                    str(
                        stack.dates[
                            i
                        ]
                    ),

                "date2":
                    str(
                        stack.dates[
                            j
                        ]
                    ),

                "alpha":
                    float(
                        alpha
                    ),

                "patch_size":
                    int(
                        args.patch_size
                    ),

                "valid_raster_fraction":
                    float(
                        np.mean(
                            valid
                        )
                    ),

                "filtered_point_valid_fraction":
                    float(
                        np.mean(
                            fvalid
                        )
                    ),

                "gradient":
                    filt_metrics,

                "phase_change_all":
                    quantiles(
                        change
                    ),

                "phase_change_high_tc_ds":
                    quantiles(
                        change_hq
                    ),

                "elapsed_seconds":
                    float(
                        elapsed
                    ),
            }

            records.append(
                row
            )

            frac_reduction = (
                1.0
                -
                (
                    filt_metrics[
                        "frac_gt_pi_2"
                    ]
                    /
                    raw_metrics[
                        "frac_gt_pi_2"
                    ]
                )
                if raw_metrics[
                    "frac_gt_pi_2"
                ]
                >
                0
                else
                0.0
            )

            print(
                f"  alpha={alpha:.2f}: "
                f">pi/2 reduction="
                f"{100*frac_reduction:+7.2f}%, "
                f"p95={filt_metrics['p95_abs_rad']:.3f}, "
                f"HQ phase-change q95="
                f"{row['phase_change_high_tc_ds'].get('q95', float('nan')):.3f} rad, "
                f"{elapsed:.2f}s"
            )

    # ----------------------------------------------------------
    # Aggregate by alpha relative to each pair's raw metrics.
    # ----------------------------------------------------------
    raw_by_pair = {
        int(
            r[
                "pair_id"
            ]
        ):
            r
        for r in records
        if float(
            r[
                "alpha"
            ]
        )
        ==
        0.0
    }

    aggregate = []

    for alpha in alphas:
        rows_a = [
            r
            for r in records
            if abs(
                float(
                    r[
                        "alpha"
                    ]
                )
                -
                alpha
            )
            <
            1.0e-12
        ]

        frac_red = []
        p95_red = []
        hq_q95 = []
        hq_med = []
        valid_frac = []
        elapsed = []

        for r in rows_a:
            raw = raw_by_pair[
                int(
                    r[
                        "pair_id"
                    ]
                )
            ]

            raw_frac = raw[
                "gradient"
            ][
                "frac_gt_pi_2"
            ]

            filt_frac = r[
                "gradient"
            ][
                "frac_gt_pi_2"
            ]

            if raw_frac > 0:
                frac_red.append(
                    1.0
                    -
                    filt_frac
                    /
                    raw_frac
                )

            raw_p95 = raw[
                "gradient"
            ][
                "p95_abs_rad"
            ]

            filt_p95 = r[
                "gradient"
            ][
                "p95_abs_rad"
            ]

            if raw_p95 > 0:
                p95_red.append(
                    1.0
                    -
                    filt_p95
                    /
                    raw_p95
                )

            hq_q95.append(
                r[
                    "phase_change_high_tc_ds"
                ].get(
                    "q95",
                    np.nan,
                )
            )

            hq_med.append(
                r[
                    "phase_change_high_tc_ds"
                ].get(
                    "median",
                    np.nan,
                )
            )

            valid_frac.append(
                r[
                    "filtered_point_valid_fraction"
                ]
            )

            elapsed.append(
                r[
                    "elapsed_seconds"
                ]
            )

        aggregate.append(
            {
                "alpha":
                    float(
                        alpha
                    ),

                "representative_pairs":
                    len(
                        rows_a
                    ),

                "median_frac_gt_pi2_reduction":
                    float(
                        np.nanmedian(
                            frac_red
                        )
                    ),

                "median_p95_gradient_reduction":
                    float(
                        np.nanmedian(
                            p95_red
                        )
                    ),

                "median_high_tc_phase_change_median_rad":
                    float(
                        np.nanmedian(
                            hq_med
                        )
                    ),

                "median_high_tc_phase_change_q95_rad":
                    float(
                        np.nanmedian(
                            hq_q95
                        )
                    ),

                "minimum_filtered_point_valid_fraction":
                    float(
                        np.nanmin(
                            valid_frac
                        )
                    ),

                "total_filter_seconds":
                    float(
                        np.sum(
                            elapsed
                        )
                    ),
            }
        )

    payload = {
        "format":
            "pyPSDS-GAMMA-adaptive-filter-benchmark-v1",

        "production_changed":
            False,

        "filter":
            "Goldstein-Werner adaptive spectral filter",

        "implementation_reference":
            "Dolphin overlapping FFT patch convention",

        "intended_usage":
            (
                "unwrap integer estimation only; transfer integer cycles "
                "back to original unfiltered virtual IFG"
            ),

        "planned_production_position":
            "after_unwrap_policy_before_unwrap",

        "scene":
            [
                int(
                    H
                ),
                int(
                    W
                ),
            ],

        "network_pairs":
            len(
                edges
            ),

        "representative_pair_ids":
            [
                int(
                    x[
                        "pair_id"
                    ]
                )
                for x in selected
            ],

        "alphas":
            alphas,

        "patch_size":
            int(
                args.patch_size
            ),

        "aggregate":
            aggregate,

        "records":
            records,

        "decision_rule":
            (
                "Do not choose alpha from smoothing alone. Prefer only if "
                "high-gradient reduction is material while phase change on "
                "high-TC DS remains small. Full unwrap validation is required "
                "before production integration."
            ),
    }

    json_path = (
        outdir
        /
        "adaptive_filter_benchmark.json"
    )

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    csv_path = (
        outdir
        /
        "adaptive_filter_benchmark.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(
            f
        )

        w.writerow(
            (
                "alpha",
                "representative_pairs",
                "median_frac_gt_pi2_reduction",
                "median_p95_gradient_reduction",
                "median_high_tc_phase_change_median_rad",
                "median_high_tc_phase_change_q95_rad",
                "minimum_filtered_point_valid_fraction",
                "total_filter_seconds",
            )
        )

        for r in aggregate:
            w.writerow(
                (
                    r[
                        "alpha"
                    ],
                    r[
                        "representative_pairs"
                    ],
                    r[
                        "median_frac_gt_pi2_reduction"
                    ],
                    r[
                        "median_p95_gradient_reduction"
                    ],
                    r[
                        "median_high_tc_phase_change_median_rad"
                    ],
                    r[
                        "median_high_tc_phase_change_q95_rad"
                    ],
                    r[
                        "minimum_filtered_point_valid_fraction"
                    ],
                    r[
                        "total_filter_seconds"
                    ],
                )
            )

    print()
    print("=" * 118)
    print("ADAPTIVE-FILTER COMPACT TABLE")
    print("=" * 118)
    print(
        f"{'alpha':>6s} "
        f"{'>pi/2 red.':>14s} "
        f"{'p95 grad red.':>15s} "
        f"{'HQ dphi med':>13s} "
        f"{'HQ dphi q95':>13s} "
        f"{'min valid':>11s} "
        f"{'filter s':>10s}"
    )

    for r in aggregate:
        print(
            f"{r['alpha']:6.2f} "
            f"{100*r['median_frac_gt_pi2_reduction']:13.2f}% "
            f"{100*r['median_p95_gradient_reduction']:14.2f}% "
            f"{r['median_high_tc_phase_change_median_rad']:13.4f} "
            f"{r['median_high_tc_phase_change_q95_rad']:13.4f} "
            f"{100*r['minimum_filtered_point_valid_fraction']:10.3f}% "
            f"{r['total_filter_seconds']:10.1f}"
        )

    print()
    print("JSON:", json_path)
    print("CSV :", csv_path)


if __name__ == "__main__":
    main()
