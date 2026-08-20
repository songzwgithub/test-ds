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


def robust_normalize(x, valid):
    """
    Robust 0..1 normalization using p05/p95.
    Lower original value -> lower normalized value.
    """
    v = x[valid]
    v = v[np.isfinite(v)]

    if v.size == 0:
        raise RuntimeError("No valid values for normalization")

    lo, hi = np.percentile(
        v,
        [5.0, 95.0],
    )

    if hi <= lo:
        hi = lo + 1e-12

    y = (
        x - lo
    ) / (
        hi - lo
    )

    return np.clip(
        y,
        0.0,
        1.0,
    )


def fit_detrended_rms(
    phase,
    t,
    batch_size=20000,
):
    """
    Remove intercept + linear temporal trend independently
    for every point.

    Absolute slope is NOT used for reference selection.
    """

    npoint, ndate = phase.shape

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

    out = np.empty(
        npoint,
        dtype=np.float32,
    )

    for b0 in range(
        0,
        npoint,
        batch_size,
    ):

        b1 = min(
            b0 + batch_size,
            npoint,
        )

        Y = np.asarray(
            phase[
                b0:b1,
                :
            ],
            dtype=np.float64,
        )

        beta = (
            Y @ P.T
        )

        fitted = (
            beta @ G.T
        )

        residual = (
            Y - fitted
        )

        rms = np.sqrt(
            np.mean(
                residual * residual,
                axis=1,
            )
        )

        out[
            b0:b1
        ] = rms.astype(
            np.float32
        )

    return out


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--window",
        type=int,
        default=21,
        help="Odd radar-coordinate box size.",
    )

    ap.add_argument(
        "--min-points",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--top",
        type=int,
        default=20,
    )

    args = ap.parse_args()

    if (
        args.window < 3
        or
        args.window % 2 == 0
    ):
        raise ValueError(
            "--window must be odd and >= 3"
        )

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

    final_dir = (
        root
        / "final_unwrap_v09"
    )

    outdir = (
        root
        / "reference_candidate_v09"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Strict point domain
    # --------------------------------------------------------

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
            "Strict phase/ID size mismatch"
        )

    # --------------------------------------------------------
    # Time axis in days
    # --------------------------------------------------------

    dates64 = np.asarray(
        [
            np.datetime64(
                str(x)
            )
            for x in stack.dates
        ]
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

    # Scale for better numerical conditioning.
    t = (
        t_days
        -
        t_days.mean()
    )

    t /= max(
        1.0,
        np.std(t)
    )

    print("=" * 108)
    print(
        "Step 09b - Automatic computational "
        "reference-region candidate search"
    )
    print("=" * 108)

    print(
        f"config                     : {config_path}"
    )

    print(
        f"strict points              : {nstrict:,}"
    )

    print(
        f"acquisitions               : {ndate}"
    )

    print(
        f"reference window           : "
        f"{args.window} x {args.window}"
    )

    print(
        f"minimum points / region    : "
        f"{args.min_points}"
    )

    print()
    print(
        "Computing per-point detrended temporal RMS ..."
    )

    detrended_rms = fit_detrended_rms(
        phase,
        t,
    )

    # --------------------------------------------------------
    # Full radar display dimensions
    # --------------------------------------------------------

    H = int(
        all_rows.max()
    ) + 1

    W = int(
        all_cols.max()
    ) + 1

    valid_grid = np.zeros(
        (H, W),
        dtype=np.float32,
    )

    valid_grid[
        rows,
        cols
    ] = 1.0

    # --------------------------------------------------------
    # Per-point quality score.
    #
    # network residual is almost zero in current data,
    # but retain it for generic production use.
    # --------------------------------------------------------

    valid = (
        np.isfinite(
            detrended_rms
        )
        &
        np.isfinite(
            network_rms
        )
    )

    time_noise_n = robust_normalize(
        detrended_rms.astype(
            np.float64
        ),
        valid,
    )

    network_n = robust_normalize(
        network_rms,
        valid,
    )

    # Lower is better.
    #
    # Temporal consistency deliberately dominates.
    point_badness = (
        0.85
        *
        time_noise_n
        +
        0.15
        *
        network_n
    )

    score_grid = np.zeros(
        (H, W),
        dtype=np.float32,
    )

    score_grid[
        rows,
        cols
    ] = point_badness.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Box statistics using normalized uniform filter.
    # --------------------------------------------------------

    area = float(
        args.window
        *
        args.window
    )

    local_count = (
        uniform_filter(
            valid_grid,
            size=args.window,
            mode="constant",
            cval=0.0,
        )
        *
        area
    )

    local_score_sum = (
        uniform_filter(
            score_grid,
            size=args.window,
            mode="constant",
            cval=0.0,
        )
        *
        area
    )

    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):

        local_mean_badness = (
            local_score_sum
            /
            local_count
        )

    candidate_count = local_count[
        rows,
        cols
    ]

    candidate_badness = local_mean_badness[
        rows,
        cols
    ]

    # --------------------------------------------------------
    # Penalize scene edges.
    # --------------------------------------------------------

    half = (
        args.window // 2
    )

    interior = (
        (rows >= half)
        &
        (rows < H - half)
        &
        (cols >= half)
        &
        (cols < W - half)
    )

    eligible = (
        valid
        &
        interior
        &
        (
            candidate_count
            >=
            args.min_points
        )
        &
        np.isfinite(
            candidate_badness
        )
    )

    eligible_idx = np.where(
        eligible
    )[0]

    if eligible_idx.size == 0:

        raise RuntimeError(
            "No eligible automatic reference regions."
        )

    order = eligible_idx[
        np.argsort(
            candidate_badness[
                eligible_idx
            ]
        )
    ]

    # --------------------------------------------------------
    # Keep spatially distinct top regions.
    # --------------------------------------------------------

    selected = []

    min_center_sep = (
        args.window
    )

    for idx in order.tolist():

        r = int(
            rows[idx]
        )

        c = int(
            cols[idx]
        )

        if all(
            max(
                abs(
                    r - rr
                ),
                abs(
                    c - cc
                ),
            )
            >=
            min_center_sep
            for _, rr, cc in selected
        ):

            selected.append(
                (
                    idx,
                    r,
                    c,
                )
            )

        if len(selected) >= args.top:
            break

    # --------------------------------------------------------
    # Detailed region statistics.
    # --------------------------------------------------------

    output_rows = []

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
                half
            )
            &
            (
                np.abs(
                    cols - cc
                )
                <=
                half
            )
        )

        ridx = np.where(
            region
        )[0]

        temporal_values = detrended_rms[
            ridx
        ]

        network_values = network_rms[
            ridx
        ]

        output_rows.append({
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

            "window":
                int(
                    args.window
                ),

            "point_count":
                int(
                    ridx.size
                ),

            "region_badness":
                float(
                    candidate_badness[
                        idx
                    ]
                ),

            "temporal_detrended_rms_median_rad":
                float(
                    np.median(
                        temporal_values
                    )
                ),

            "temporal_detrended_rms_p90_rad":
                float(
                    np.percentile(
                        temporal_values,
                        90
                    )
                ),

            "network_rms_median_rad":
                float(
                    np.median(
                        network_values
                    )
                ),

            "network_rms_p90_rad":
                float(
                    np.percentile(
                        network_values,
                        90
                    )
                ),
        })

    # --------------------------------------------------------
    # Best candidate region
    # --------------------------------------------------------

    best = output_rows[0]

    best_r = int(
        best["row"]
    )

    best_c = int(
        best["col"]
    )

    best_region_strict = (
        (
            np.abs(
                rows - best_r
            )
            <=
            half
        )
        &
        (
            np.abs(
                cols - best_c
            )
            <=
            half
        )
    )

    best_region_full = np.zeros(
        all_rows.size,
        dtype=bool,
    )

    best_region_full[
        strict_ids[
            best_region_strict
        ]
    ] = True

    # --------------------------------------------------------
    # Representative point:
    # lowest point badness within best region.
    # --------------------------------------------------------

    best_region_indices = np.where(
        best_region_strict
    )[0]

    rep_local = best_region_indices[
        np.argmin(
            point_badness[
                best_region_indices
            ]
        )
    ]

    rep_point_id = int(
        strict_ids[
            rep_local
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    csv_path = (
        outdir
        / "reference_candidates.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            output_rows
        )

    np.save(
        outdir
        / "auto_reference_region_mask.npy",
        best_region_full,
    )

    np.save(
        outdir
        / "detrended_temporal_rms_rad.npy",
        detrended_rms,
    )

    ref_txt = (
        outdir
        / "auto_reference_point.txt"
    )

    ref_txt.write_text(
        f"point_id {rep_point_id}\n"
        f"row {int(all_rows[rep_point_id])}\n"
        f"col {int(all_cols[rep_point_id])}\n"
        f"region_center_row {best_r}\n"
        f"region_center_col {best_c}\n"
        f"region_window {args.window}\n"
        f"region_point_count "
        f"{int(np.count_nonzero(best_region_strict))}\n"
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    display = np.full(
        (H, W),
        np.nan,
        dtype=np.float32,
    )

    display[
        rows,
        cols
    ] = detrended_rms

    fig, ax = plt.subplots(
        figsize=(15, 5),
        constrained_layout=True,
    )

    vals = detrended_rms[
        np.isfinite(
            detrended_rms
        )
    ]

    vmax = float(
        np.percentile(
            vals,
            95
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

    for r in output_rows[:10]:

        rect = plt.Rectangle(
            (
                r["col"] - half,
                r["row"] - half,
            ),
            args.window,
            args.window,
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
            fontsize=8,
            ha="center",
            va="center",
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
        label="representative point",
    )

    ax.set_title(
        "Automatic computational reference candidates\n"
        "background: detrended temporal RMS"
    )

    ax.set_xlabel(
        "range pixel / col"
    )

    ax.set_ylabel(
        "azimuth pixel / row"
    )

    ax.set_xlim(
        0,
        W
    )

    ax.set_ylim(
        H,
        0
    )

    ax.legend()

    fig.colorbar(
        im,
        ax=ax,
        label="detrended temporal RMS [rad]",
    )

    fig_path = (
        outdir
        / "reference_candidate_map.png"
    )

    fig.savefig(
        fig_path,
        dpi=160,
    )

    plt.close(fig)

    manifest = {
        "format":
            "pyPSDS-GAMMA-reference-candidate-search-v09",

        "status":
            "CANDIDATE_ONLY",

        "reference_role":
            "computational_reference",

        "geodetic_stability_claim":
            False,

        "strict_points":
            int(
                nstrict
            ),

        "window":
            int(
                args.window
            ),

        "minimum_points":
            int(
                args.min_points
            ),

        "candidate_regions":
            int(
                len(
                    output_rows
                )
            ),

        "best_region": {
            "center_row":
                best_r,

            "center_col":
                best_c,

            "point_count":
                int(
                    np.count_nonzero(
                        best_region_strict
                    )
                ),

            "representative_point_id":
                rep_point_id,
        },

        "score": {
            "temporal_detrended_rms_weight":
                0.85,

            "network_residual_weight":
                0.15,

            "absolute_velocity_used":
                False,
        },

        "phase_modified":
            False,

        "note":
            (
                "Automatic reference selection identifies a "
                "low-noise computational reference region. "
                "It does not prove geodetic stability. "
                "Manual stable-region override remains required "
                "when independent geological/GNSS knowledge exists."
            ),
    }

    manifest_path = (
        outdir
        / "reference_candidate_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 108)
    print(
        "Automatic reference candidates"
    )
    print("=" * 108)

    print(
        " rank   row   col  points "
        "regionScore temporalRMS networkRMS"
    )

    for r in output_rows:

        print(
            f" {r['rank']:4d} "
            f"{r['row']:5d} "
            f"{r['col']:5d} "
            f"{r['point_count']:7d} "
            f"{r['region_badness']:.6f} "
            f"{r['temporal_detrended_rms_median_rad']:.6f} "
            f"{r['network_rms_median_rad']:.3e}"
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
        f"candidate table            : {csv_path}"
    )

    print(
        f"candidate map              : {fig_path}"
    )

    print(
        f"manifest                   : {manifest_path}"
    )

    print()
    print(
        "STEP 09b STATUS: PASS / "
        "REFERENCE CANDIDATES ONLY"
    )

    print(
        "No spatial reference has yet been applied."
    )


if __name__ == "__main__":
    main()
