#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


# ============================================================
# Robust statistics
# ============================================================

def mad_sigma(x):
    """
    Robust sigma estimate:
        1.4826 * median(|x - median(x)|)
    """
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return np.nan

    med = np.median(x)

    return float(
        1.4826
        *
        np.median(
            np.abs(
                x - med
            )
        )
    )


def temporal_rms(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                x * x
            )
        )
    )


def get_region(
    rows,
    cols,
    cr,
    cc,
    hr,
    hc,
):
    return (
        (np.abs(rows - cr) <= hr)
        &
        (np.abs(cols - cc) <= hc)
    )


def region_quadrant_consistency(
    residual,
    rows,
    cols,
    cr,
    cc,
    region_mask,
):
    """
    Compare the median residual time series of four
    spatial quadrants against the whole-region median.
    """

    ridx = np.where(
        region_mask
    )[0]

    if ridx.size < 4:
        return np.nan, 0

    r = rows[
        ridx
    ]

    c = cols[
        ridx
    ]

    R = np.asarray(
        residual[
            ridx,
            :
        ],
        dtype=np.float64,
    )

    whole = np.median(
        R,
        axis=0,
    )

    quadrants = [
        (r <= cr) & (c <= cc),
        (r <= cr) & (c >  cc),
        (r >  cr) & (c <= cc),
        (r >  cr) & (c >  cc),
    ]

    qrms = []

    valid_quadrants = 0

    for q in quadrants:

        # Require enough samples so a tiny quadrant
        # cannot dominate the result.
        if np.count_nonzero(q) < 10:
            continue

        qmed = np.median(
            R[
                q,
                :
            ],
            axis=0,
        )

        qrms.append(
            temporal_rms(
                qmed
                -
                whole
            )
        )

        valid_quadrants += 1

    if not qrms:
        return np.nan, 0

    return (
        float(
            max(qrms)
        ),
        valid_quadrants,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--candidate-count",
        type=int,
        default=20,
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

    candidate_dir = (
        root
        / "reference_candidate_v09_robust"
    )

    outdir = (
        root
        / "reference_candidate_v09_robustness"
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

    candidate_csv = (
        candidate_dir
        / "robust_reference_candidates.csv"
    )

    candidates = []

    with candidate_csv.open() as f:

        for r in csv.DictReader(f):

            candidates.append(
                {
                    "original_rank":
                        int(r["rank"]),

                    "row":
                        int(r["row"]),

                    "col":
                        int(r["col"]),
                }
            )

    candidates = candidates[
        :args.candidate_count
    ]

    # ========================================================
    # Physical time axis
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

    # Use years for interpretable phase rate.
    t_year = (
        t_days
        /
        365.25
    )

    tc = (
        t_year
        -
        np.mean(
            t_year
        )
    )

    denom = float(
        np.sum(
            tc * tc
        )
    )

    if denom <= 0:
        raise RuntimeError(
            "Invalid time axis"
        )

    # ========================================================
    # Fit intercept + linear trend for ALL strict points.
    #
    # slope units:
    #     rad / year
    #
    # residual:
    #     acquisition phase minus individual linear model.
    # ========================================================

    slope = np.empty(
        nstrict,
        dtype=np.float64,
    )

    intercept = np.empty(
        nstrict,
        dtype=np.float64,
    )

    residual = np.empty(
        (
            nstrict,
            ndate,
        ),
        dtype=np.float32,
    )

    print("=" * 112)
    print(
        "Step 09b3 - Reference-region robustness audit"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"strict points              : "
        f"{nstrict:,}"
    )

    print(
        f"candidates audited         : "
        f"{len(candidates)}"
    )

    print()
    print(
        "Computing point-wise linear phase trends ..."
    )

    batch = 20000

    for b0 in range(
        0,
        nstrict,
        batch,
    ):

        b1 = min(
            b0 + batch,
            nstrict,
        )

        Y = np.asarray(
            phase[
                b0:b1,
                :
            ],
            dtype=np.float64,
        )

        ymean = np.mean(
            Y,
            axis=1,
        )

        s = (
            (Y - ymean[:, None])
            @
            tc
            /
            denom
        )

        a = (
            ymean
            -
            s
            *
            np.mean(
                t_year
            )
        )

        fit = (
            a[:, None]
            +
            s[:, None]
            *
            t_year[
                None,
                :
            ]
        )

        slope[
            b0:b1
        ] = s

        intercept[
            b0:b1
        ] = a

        residual[
            b0:b1,
            :
        ] = (
            Y - fit
        ).astype(
            np.float32
        )

    # ========================================================
    # Nested window definitions
    # ========================================================

    windows = [
        (7, 11),   # 15 x 23
        (10, 15),  # 21 x 31
        (15, 22),  # 31 x 45
    ]

    rows_out = []

    print()
    print(
        "Auditing candidate regions ..."
    )

    for cand in candidates:

        cr = cand[
            "row"
        ]

        cc = cand[
            "col"
        ]

        metrics = {}

        for hr, hc in windows:

            region = get_region(
                rows,
                cols,
                cr,
                cc,
                hr,
                hc,
            )

            ridx = np.where(
                region
            )[0]

            npix = (
                (2 * hr + 1)
                *
                (2 * hc + 1)
            )

            n = ridx.size

            density = (
                n
                /
                npix
            )

            if n < 20:

                relative_rms = np.nan
                slope_sigma = np.nan
                quadrant_rms = np.nan
                quadrant_valid = 0

            else:

                R = np.asarray(
                    residual[
                        ridx,
                        :
                    ],
                    dtype=np.float64,
                )

                # Whole-region common residual:
                # remove it epoch by epoch.
                region_med = np.median(
                    R,
                    axis=0,
                )

                internal = (
                    R
                    -
                    region_med[
                        None,
                        :
                    ]
                )

                relative_rms = float(
                    np.sqrt(
                        np.mean(
                            internal
                            *
                            internal
                        )
                    )
                )

                slope_sigma = mad_sigma(
                    slope[
                        ridx
                    ]
                )

                (
                    quadrant_rms,
                    quadrant_valid,
                ) = region_quadrant_consistency(
                    residual,
                    rows,
                    cols,
                    cr,
                    cc,
                    region,
                )

            prefix = (
                f"w{2*hr+1}x{2*hc+1}"
            )

            metrics[
                f"{prefix}_points"
            ] = int(n)

            metrics[
                f"{prefix}_density"
            ] = float(
                density
            )

            metrics[
                f"{prefix}_relative_rms_rad"
            ] = float(
                relative_rms
            )

            metrics[
                f"{prefix}_slope_mad_rad_yr"
            ] = float(
                slope_sigma
            )

            metrics[
                f"{prefix}_quadrant_rms_rad"
            ] = float(
                quadrant_rms
            )

            metrics[
                f"{prefix}_valid_quadrants"
            ] = int(
                quadrant_valid
            )

        # ====================================================
        # Cross-scale robustness
        # ====================================================

        rel_values = np.asarray(
            [
                metrics[
                    "w15x23_relative_rms_rad"
                ],
                metrics[
                    "w21x31_relative_rms_rad"
                ],
                metrics[
                    "w31x45_relative_rms_rad"
                ],
            ],
            dtype=np.float64,
        )

        slope_values = np.asarray(
            [
                metrics[
                    "w15x23_slope_mad_rad_yr"
                ],
                metrics[
                    "w21x31_slope_mad_rad_yr"
                ],
                metrics[
                    "w31x45_slope_mad_rad_yr"
                ],
            ],
            dtype=np.float64,
        )

        quad_values = np.asarray(
            [
                metrics[
                    "w15x23_quadrant_rms_rad"
                ],
                metrics[
                    "w21x31_quadrant_rms_rad"
                ],
                metrics[
                    "w31x45_quadrant_rms_rad"
                ],
            ],
            dtype=np.float64,
        )

        density_values = np.asarray(
            [
                metrics[
                    "w15x23_density"
                ],
                metrics[
                    "w21x31_density"
                ],
                metrics[
                    "w31x45_density"
                ],
            ],
            dtype=np.float64,
        )

        # We do NOT combine them into an arbitrary weighted
        # score yet. Keep interpretable Pareto-style metrics.
        result = {
            "original_rank":
                cand[
                    "original_rank"
                ],

            "row":
                cr,

            "col":
                cc,

            "minimum_density":
                float(
                    np.nanmin(
                        density_values
                    )
                ),

            "median_density":
                float(
                    np.nanmedian(
                        density_values
                    )
                ),

            "relative_rms_median_rad":
                float(
                    np.nanmedian(
                        rel_values
                    )
                ),

            "relative_rms_max_rad":
                float(
                    np.nanmax(
                        rel_values
                    )
                ),

            "relative_rms_scale_range_rad":
                float(
                    np.nanmax(
                        rel_values
                    )
                    -
                    np.nanmin(
                        rel_values
                    )
                ),

            "slope_mad_median_rad_yr":
                float(
                    np.nanmedian(
                        slope_values
                    )
                ),

            "slope_mad_max_rad_yr":
                float(
                    np.nanmax(
                        slope_values
                    )
                ),

            "quadrant_rms_median_rad":
                float(
                    np.nanmedian(
                        quad_values
                    )
                ),

            "quadrant_rms_max_rad":
                float(
                    np.nanmax(
                        quad_values
                    )
                ),
        }

        result.update(
            metrics
        )

        rows_out.append(
            result
        )

    # ========================================================
    # Pareto-oriented ranking
    #
    # First reject clearly weak geometry.
    #
    # Then lexicographic ordering:
    # 1. cross-scale internal consistency
    # 2. slope homogeneity
    # 3. quadrant consistency
    # 4. density
    #
    # This is intentionally NOT a weighted score.
    # ========================================================

    for r in rows_out:

        density_ok = (
            r[
                "minimum_density"
            ]
            >=
            0.30
        )

        quadrants_ok = (
            r[
                "w21x31_valid_quadrants"
            ]
            >=
            4
        )

        r[
            "geometry_ok"
        ] = int(
            density_ok
            and
            quadrants_ok
        )

    ranked = sorted(
        rows_out,
        key=lambda r: (
            -r[
                "geometry_ok"
            ],
            r[
                "relative_rms_max_rad"
            ],
            r[
                "slope_mad_max_rad_yr"
            ],
            r[
                "quadrant_rms_max_rad"
            ],
            -r[
                "minimum_density"
            ],
        )
    )

    for rank, r in enumerate(
        ranked,
        start=1,
    ):

        r[
            "robustness_rank"
        ] = rank

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / "reference_region_robustness.csv"
    )

    fields = []

    for r in ranked:
        for k in r.keys():
            if k not in fields:
                fields.append(k)

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()
        w.writerows(
            ranked
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-reference-region-robustness-v09",

        "status":
            "AUDIT_ONLY",

        "candidate_count":
            int(
                len(
                    ranked
                )
            ),

        "windows": [
            "15x23",
            "21x31",
            "31x45",
        ],

        "ranking": [
            "geometry eligibility",
            "maximum cross-scale relative temporal RMS",
            "maximum local slope MAD",
            "maximum quadrant inconsistency",
            "minimum density",
        ],

        "spatial_reference_applied":
            False,

        "absolute_velocity_used":
            False,

        "note":
            (
                "Slope dispersion is reference-invariant; "
                "absolute slope is not used to claim geodetic "
                "stability."
            ),
    }

    manifest_path = (
        outdir
        / "reference_region_robustness_manifest.json"
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

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Reference-region robustness ranking"
    )
    print("=" * 112)

    print(
        " rank old  row   col geom "
        "densMin relRMSmax slopeMADmax "
        "quadRMSmax scaleRange"
    )

    for r in ranked:

        print(
            f" {r['robustness_rank']:4d} "
            f"{r['original_rank']:3d} "
            f"{r['row']:5d} "
            f"{r['col']:5d} "
            f"{r['geometry_ok']:4d} "
            f"{r['minimum_density']:7.3f} "
            f"{r['relative_rms_max_rad']:9.5f} "
            f"{r['slope_mad_max_rad_yr']:11.5f} "
            f"{r['quadrant_rms_max_rad']:10.5f} "
            f"{r['relative_rms_scale_range_rad']:10.5f}"
        )

    print()
    print(
        "Top robust candidates"
    )

    for r in ranked[:10]:

        print(
            f"  new#{r['robustness_rank']:02d} "
            f"(old#{r['original_rank']:02d}) "
            f"row={r['row']}, "
            f"col={r['col']}, "
            f"density_min="
            f"{r['minimum_density']:.3f}, "
            f"RMSmax="
            f"{r['relative_rms_max_rad']:.5f}, "
            f"slopeMADmax="
            f"{r['slope_mad_max_rad_yr']:.5f}, "
            f"quadRMSmax="
            f"{r['quadrant_rms_max_rad']:.5f}"
        )

    print()
    print(
        f"audit table                : "
        f"{csv_path}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP 09b3 STATUS: PASS / "
        "REFERENCE ROBUSTNESS AUDIT ONLY"
    )

    print(
        "No spatial reference has been applied."
    )


if __name__ == "__main__":
    main()
