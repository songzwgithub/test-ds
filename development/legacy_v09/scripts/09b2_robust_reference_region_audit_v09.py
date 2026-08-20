#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter

from pypsds.prototype import open_from_config


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--half-row",
        type=int,
        default=10,
        help="Reference box half-size in azimuth pixels.",
    )

    ap.add_argument(
        "--half-col",
        type=int,
        default=15,
        help="Reference box half-size in range pixels.",
    )

    ap.add_argument(
        "--min-points",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--margin-row",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--margin-col",
        type=int,
        default=75,
    )

    ap.add_argument(
        "--separation-scale",
        type=float,
        default=2.0,
    )

    ap.add_argument(
        "--top",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=20000,
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

    root = (
        Path(paths.output_dir)
        / "v09"
    )

    pps_dir = (
        root
        / "point_phase_stack"
    )

    inversion_dir = (
        root
        / "network_inversion_v09"
    )

    outdir = (
        root
        / "reference_candidate_v09_robust"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Inputs
    # ========================================================

    strict_ids = np.load(
        inversion_dir
        / "strict_point_ids.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    phase = np.load(
        inversion_dir
        / "acquisition_phase_l2_candidate_rad.npy",
        mmap_mode="r",
    )

    network_rms = np.asarray(
        np.load(
            inversion_dir
            / "l2_network_residual_rms_rad.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    all_rows = np.asarray(
        np.load(
            pps_dir
            / "rows.npy",
            mmap_mode="r",
        ),
        dtype=np.int32,
    )

    all_cols = np.asarray(
        np.load(
            pps_dir
            / "cols.npy",
            mmap_mode="r",
        ),
        dtype=np.int32,
    )

    rows = all_rows[
        strict_ids
    ]

    cols = all_cols[
        strict_ids
    ]

    nstrict, ndate = phase.shape

    if nstrict != strict_ids.size:
        raise RuntimeError(
            "strict phase size mismatch"
        )

    H = int(
        all_rows.max()
    ) + 1

    W = int(
        all_cols.max()
    ) + 1

    win_rows = (
        2 * args.half_row + 1
    )

    win_cols = (
        2 * args.half_col + 1
    )

    area = float(
        win_rows
        *
        win_cols
    )

    # ========================================================
    # Time design: remove intercept + linear trend PER POINT.
    # ========================================================

    # IMPORTANT:
    # stack.dates are compact YYYYMMDD strings.
    # np.datetime64("20141006") is interpreted as YEAR 20141006,
    # not 2014-10-06. Convert explicitly to ISO YYYY-MM-DD.
    def parse_yyyymmdd(x):
        s = str(x)
        if len(s) != 8 or not s.isdigit():
            raise ValueError(
                f"Expected YYYYMMDD date, got: {s!r}"
            )
        return np.datetime64(
            f"{s[0:4]}-{s[4:6]}-{s[6:8]}",
            "D",
        )

    dates64 = np.asarray(
        [
            parse_yyyymmdd(x)
            for x in stack.dates
        ],
        dtype="datetime64[D]",
    )

    t_days = (
        dates64
        -
        dates64[0]
    ).astype(
        "timedelta64[D]"
    ).astype(
        np.float64
    )

    t = (
        t_days
        -
        np.mean(
            t_days
        )
    )

    t /= max(
        1.0,
        np.std(t),
    )

    G = np.column_stack(
        [
            np.ones(
                ndate,
                dtype=np.float64,
            ),
            t,
        ]
    )

    P = np.linalg.pinv(
        G
    )

    # beta[:,0] intercept
    # beta[:,1] normalized linear trend
    beta = np.empty(
        (
            nstrict,
            2,
        ),
        dtype=np.float64,
    )

    print("=" * 110)
    print(
        "Step 09b2 - Robust computational "
        "reference-region audit"
    )
    print("=" * 110)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"strict points              : "
        f"{nstrict:,}"
    )

    print(
        f"acquisitions               : "
        f"{ndate}"
    )

    print(
        f"reference box              : "
        f"{win_rows} x {win_cols}"
    )

    print(
        f"minimum points             : "
        f"{args.min_points}"
    )

    print(
        f"scene margin row/col       : "
        f"{args.margin_row} / "
        f"{args.margin_col}"
    )

    print()
    print(
        "Fitting per-point linear trends ..."
    )

    for b0 in range(
        0,
        nstrict,
        args.batch_size,
    ):

        b1 = min(
            b0 + args.batch_size,
            nstrict,
        )

        Y = np.asarray(
            phase[
                b0:b1,
                :
            ],
            dtype=np.float64,
        )

        beta[
            b0:b1
        ] = (
            Y
            @
            P.T
        )

    # ========================================================
    # Static local point count
    # ========================================================

    count_grid = np.zeros(
        (H, W),
        dtype=np.float64,
    )

    count_grid[
        rows,
        cols
    ] = 1.0

    local_count_grid = (
        uniform_filter(
            count_grid,
            size=(
                win_rows,
                win_cols,
            ),
            mode="constant",
            cval=0.0,
        )
        *
        area
    )

    # ========================================================
    # Region-internal temporal consistency.
    #
    # For each acquisition:
    #
    # residual_p(t) =
    # phase_p(t) - fitted linear trend_p(t)
    #
    # Within every candidate window:
    #
    # var_t = E(r^2) - E(r)^2
    #
    # final relative RMS =
    # sqrt(mean_t(var_t))
    #
    # This removes:
    #   - arbitrary point intercept
    #   - individual linear trend
    #   - local common residual phase at each epoch
    #
    # and evaluates only INTERNAL region consistency.
    # ========================================================

    var_accum = np.zeros(
        (H, W),
        dtype=np.float64,
    )

    common_sq_accum = np.zeros(
        (H, W),
        dtype=np.float64,
    )

    point_resid_sq = np.zeros(
        nstrict,
        dtype=np.float64,
    )

    value_grid = np.zeros(
        (H, W),
        dtype=np.float64,
    )

    square_grid = np.zeros(
        (H, W),
        dtype=np.float64,
    )

    print()
    print(
        "Computing local spatiotemporal consistency ..."
    )

    for it in range(
        ndate
    ):

        y = np.asarray(
            phase[
                :,
                it
            ],
            dtype=np.float64,
        )

        fitted = (
            beta[:, 0]
            +
            beta[:, 1]
            *
            t[it]
        )

        residual = (
            y
            -
            fitted
        )

        point_resid_sq += (
            residual
            *
            residual
        )

        value_grid.fill(
            0.0
        )

        square_grid.fill(
            0.0
        )

        value_grid[
            rows,
            cols
        ] = residual

        square_grid[
            rows,
            cols
        ] = (
            residual
            *
            residual
        )

        local_sum = (
            uniform_filter(
                value_grid,
                size=(
                    win_rows,
                    win_cols,
                ),
                mode="constant",
                cval=0.0,
            )
            *
            area
        )

        local_sumsq = (
            uniform_filter(
                square_grid,
                size=(
                    win_rows,
                    win_cols,
                ),
                mode="constant",
                cval=0.0,
            )
            *
            area
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):

            local_mean = (
                local_sum
                /
                local_count_grid
            )

            local_second = (
                local_sumsq
                /
                local_count_grid
            )

            local_var = (
                local_second
                -
                local_mean
                *
                local_mean
            )

        local_var[
            ~np.isfinite(
                local_var
            )
        ] = 0.0

        # tiny negative roundoff -> zero
        np.maximum(
            local_var,
            0.0,
            out=local_var,
        )

        local_mean[
            ~np.isfinite(
                local_mean
            )
        ] = 0.0

        var_accum += (
            local_var
        )

        common_sq_accum += (
            local_mean
            *
            local_mean
        )

        print(
            f"  acquisition "
            f"{it+1:2d}/{ndate}: "
            f"{stack.dates[it]}"
        )

    local_relative_rms_grid = np.sqrt(
        var_accum
        /
        ndate
    )

    local_common_rms_grid = np.sqrt(
        common_sq_accum
        /
        ndate
    )

    point_detrended_rms = np.sqrt(
        point_resid_sq
        /
        ndate
    )

    # ========================================================
    # Metrics at strict-point candidate centers
    # ========================================================

    region_count = (
        local_count_grid[
            rows,
            cols
        ]
    )

    region_relative_rms = (
        local_relative_rms_grid[
            rows,
            cols
        ]
    )

    region_common_rms = (
        local_common_rms_grid[
            rows,
            cols
        ]
    )

    # 09a residual is now only a QA gate.
    # It is NOT part of the score.
    network_limit = 1.0e-5

    interior = (
        (rows >= args.margin_row)
        &
        (
            rows
            <
            H - args.margin_row
        )
        &
        (cols >= args.margin_col)
        &
        (
            cols
            <
            W - args.margin_col
        )
    )

    eligible = (
        interior
        &
        (
            region_count
            >=
            args.min_points
        )
        &
        np.isfinite(
            region_relative_rms
        )
        &
        np.isfinite(
            region_common_rms
        )
        &
        (
            network_rms
            <
            network_limit
        )
    )

    eligible_idx = np.where(
        eligible
    )[0]

    if eligible_idx.size == 0:
        raise RuntimeError(
            "No eligible reference candidates."
        )

    # ========================================================
    # Ranking:
    #
    # Primary criterion ONLY:
    #     minimum local relative RMS.
    #
    # Secondary:
    #     lower common residual RMS.
    #
    # Tertiary:
    #     more points.
    #
    # No arbitrary weighted score.
    # ========================================================

    order_local = np.lexsort(
        (
            -region_count[
                eligible_idx
            ],
            region_common_rms[
                eligible_idx
            ],
            region_relative_rms[
                eligible_idx
            ],
        )
    )

    order = eligible_idx[
        order_local
    ]

    # ========================================================
    # Keep spatially distinct regions
    # ========================================================

    sep_r = max(
        1,
        int(
            args.separation_scale
            *
            win_rows
        ),
    )

    sep_c = max(
        1,
        int(
            args.separation_scale
            *
            win_cols
        ),
    )

    selected = []

    for idx in order.tolist():

        r = int(
            rows[idx]
        )

        c = int(
            cols[idx]
        )

        # Reject if candidate centers are too close in BOTH
        # radar directions.
        too_close = False

        for old_idx, rr, cc in selected:

            if (
                abs(
                    r - rr
                )
                <
                sep_r
                and
                abs(
                    c - cc
                )
                <
                sep_c
            ):

                too_close = True
                break

        if too_close:
            continue

        selected.append(
            (
                idx,
                r,
                c,
            )
        )

        if len(
            selected
        ) >= args.top:
            break

    if not selected:
        raise RuntimeError(
            "No spatially separated candidates."
        )

    # ========================================================
    # Candidate details
    # ========================================================

    rows_out = []

    for rank, (
        idx,
        cr,
        cc,
    ) in enumerate(
        selected,
        start=1,
    ):

        region = (
            (
                np.abs(
                    rows - cr
                )
                <=
                args.half_row
            )
            &
            (
                np.abs(
                    cols - cc
                )
                <=
                args.half_col
            )
        )

        ridx = np.where(
            region
        )[0]

        rows_out.append({
            "rank":
                rank,

            "center_strict_index":
                int(idx),

            "center_point_id":
                int(
                    strict_ids[
                        idx
                    ]
                ),

            "row":
                cr,

            "col":
                cc,

            "window_rows":
                win_rows,

            "window_cols":
                win_cols,

            "point_count":
                int(
                    ridx.size
                ),

            "relative_temporal_rms_rad":
                float(
                    region_relative_rms[
                        idx
                    ]
                ),

            "common_mode_rms_rad":
                float(
                    region_common_rms[
                        idx
                    ]
                ),

            "point_detrended_rms_median_rad":
                float(
                    np.median(
                        point_detrended_rms[
                            ridx
                        ]
                    )
                ),

            "point_detrended_rms_p90_rad":
                float(
                    np.percentile(
                        point_detrended_rms[
                            ridx
                        ],
                        90,
                    )
                ),

            "network_rms_median_rad":
                float(
                    np.median(
                        network_rms[
                            ridx
                        ]
                    )
                ),
        })

    # ========================================================
    # Best region
    # ========================================================

    best = rows_out[0]

    best_r = int(
        best[
            "row"
        ]
    )

    best_c = int(
        best[
            "col"
        ]
    )

    best_region_strict = (
        (
            np.abs(
                rows - best_r
            )
            <=
            args.half_row
        )
        &
        (
            np.abs(
                cols - best_c
            )
            <=
            args.half_col
        )
    )

    best_region_indices = np.where(
        best_region_strict
    )[0]

    # Representative point is informational only.
    # Actual reference should use the REGION median.
    rep_idx = best_region_indices[
        np.argmin(
            point_detrended_rms[
                best_region_indices
            ]
        )
    ]

    rep_point_id = int(
        strict_ids[
            rep_idx
        ]
    )

    full_region_mask = np.zeros(
        all_rows.size,
        dtype=bool,
    )

    full_region_mask[
        strict_ids[
            best_region_strict
        ]
    ] = True

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / "robust_reference_candidates.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows_out[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            rows_out
        )

    np.save(
        outdir
        / "auto_reference_region_mask.npy",
        full_region_mask,
    )

    np.save(
        outdir
        / "point_detrended_rms_rad.npy",
        point_detrended_rms.astype(
            np.float32
        ),
    )

    np.save(
        outdir
        / "local_relative_temporal_rms_grid.npy",
        local_relative_rms_grid.astype(
            np.float32
        ),
    )

    np.save(
        outdir
        / "local_common_mode_rms_grid.npy",
        local_common_rms_grid.astype(
            np.float32
        ),
    )

    # ========================================================
    # Visualization
    # ========================================================

    display = (
        local_relative_rms_grid.astype(
            np.float32
        )
    )

    fig, ax = plt.subplots(
        figsize=(15, 5.5),
        constrained_layout=True,
    )

    valid_display = display[
        np.isfinite(
            display
        )
        &
        (
            local_count_grid
            >=
            args.min_points
        )
    ]

    vmax = float(
        np.percentile(
            valid_display,
            95,
        )
    )

    im = ax.imshow(
        display,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        interpolation="none",
        aspect="auto",
    )

    for r in rows_out[:10]:

        rect = plt.Rectangle(
            (
                r["col"]
                -
                args.half_col,
                r["row"]
                -
                args.half_row,
            ),
            win_cols,
            win_rows,
            fill=False,
            linewidth=1.2,
        )

        ax.add_patch(
            rect
        )

        ax.text(
            r["col"],
            r["row"],
            str(
                r["rank"]
            ),
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.scatter(
        [
            all_cols[
                rep_point_id
            ]
        ],
        [
            all_rows[
                rep_point_id
            ]
        ],
        marker="x",
        s=60,
        linewidths=1.5,
        label=(
            "representative point "
            "(region median will be used)"
        ),
    )

    ax.set_title(
        "Robust computational reference-region candidates\n"
        "background: local relative temporal RMS"
    )

    ax.set_xlabel(
        "range pixel / col"
    )

    ax.set_ylabel(
        "azimuth pixel / row"
    )

    ax.set_xlim(
        0,
        W,
    )

    ax.set_ylim(
        H,
        0,
    )

    ax.legend(
        loc="upper right"
    )

    fig.colorbar(
        im,
        ax=ax,
        label=(
            "local relative temporal RMS [rad]"
        ),
    )

    fig_path = (
        outdir
        / "robust_reference_candidate_map.png"
    )

    fig.savefig(
        fig_path,
        dpi=160,
    )

    plt.close(fig)

    # ========================================================
    # Metadata
    # ========================================================

    manifest = {
        "format":
            "pyPSDS-GAMMA-robust-reference-region-v09",

        "status":
            "CANDIDATE_ONLY",

        "role":
            "computational_reference",

        "geodetic_stability_claim":
            False,

        "ranking": {
            "primary":
                "minimum local relative temporal RMS",

            "secondary":
                "minimum local common-mode RMS",

            "tertiary":
                "maximum point count",

            "network_residual":
                (
                    "hard QA gate only; "
                    "not used as ranking weight"
                ),

            "absolute_velocity_used":
                False,
        },

        "window": {
            "rows":
                win_rows,

            "cols":
                win_cols,
        },

        "minimum_points":
            int(
                args.min_points
            ),

        "best_region": {
            "row":
                best_r,

            "col":
                best_c,

            "points":
                int(
                    np.count_nonzero(
                        best_region_strict
                    )
                ),

            "relative_temporal_rms_rad":
                float(
                    best[
                        "relative_temporal_rms_rad"
                    ]
                ),

            "representative_point_id":
                rep_point_id,
        },

        "phase_modified":
            False,
    }

    manifest_path = (
        outdir
        / "robust_reference_candidate_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n"
    )

    print()
    print("=" * 110)
    print(
        "Robust reference-region candidates"
    )
    print("=" * 110)

    print(
        " rank   row   col  points  "
        "relativeRMS  commonRMS  "
        "pointRMSmed"
    )

    for r in rows_out:

        print(
            f" {r['rank']:4d} "
            f"{r['row']:5d} "
            f"{r['col']:5d} "
            f"{r['point_count']:7d} "
            f"{r['relative_temporal_rms_rad']:11.6f} "
            f"{r['common_mode_rms_rad']:10.6f} "
            f"{r['point_detrended_rms_median_rad']:11.6f}"
        )

    print()
    print(
        "Best computational reference region"
    )

    print(
        f"  center row/col           : "
        f"{best_r}, {best_c}"
    )

    print(
        f"  points                   : "
        f"{np.count_nonzero(best_region_strict)}"
    )

    print(
        f"  local relative RMS       : "
        f"{best['relative_temporal_rms_rad']:.6f} rad"
    )

    print(
        f"  common-mode RMS          : "
        f"{best['common_mode_rms_rad']:.6f} rad"
    )

    print(
        f"  representative point ID  : "
        f"{rep_point_id}"
    )

    print(
        f"  representative row/col   : "
        f"{int(all_rows[rep_point_id])}, "
        f"{int(all_cols[rep_point_id])}"
    )

    print()
    print(
        f"candidate table            : "
        f"{csv_path}"
    )

    print(
        f"candidate map              : "
        f"{fig_path}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP 09b2 STATUS: PASS / "
        "ROBUST REFERENCE CANDIDATES ONLY"
    )

    print(
        "No spatial reference has been applied."
    )


if __name__ == "__main__":
    main()
