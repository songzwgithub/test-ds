#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


def parse_date(x):

    s = str(x)

    if len(s) != 8 or not s.isdigit():
        raise ValueError(
            f"Expected YYYYMMDD, got {s!r}"
        )

    return np.datetime64(
        f"{s[:4]}-{s[4:6]}-{s[6:8]}",
        "D",
    )


def print_quantiles(
    title,
    x,
    qs=(1, 5, 50, 95, 99),
    fmt=".6f",
):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    q = np.percentile(
        x,
        qs,
    )

    print(title)

    print(
        "  "
        +
        " / ".join(
            format(v, fmt)
            for v in q
        )
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=65536,
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
        /
        "v09"
    )

    prod_dir = (
        root
        /
        "scla_v09"
        /
        "production_sensitivity"
    )

    outdir = (
        root
        /
        "scla_v09"
        /
        "scla_identifiability"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Require final Step10a adjudication.
    # ========================================================

    batch_manifest_path = (
        prod_dir
        /
        "batch_context_audit"
        /
        "batch_context_repeatability_manifest.json"
    )

    batch_manifest = json.loads(
        batch_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if not str(
        batch_manifest[
            "status"
        ]
    ).startswith(
        "PASS"
    ):
        raise RuntimeError(
            "Step10a5c has not passed"
        )

    # ========================================================
    # Sensitivity
    # ========================================================

    sensitivity_path = (
        prod_dir
        /
        "topographic_phase_sensitivity_rad_per_m.npy"
    )

    S = np.load(
        sensitivity_path,
        mmap_mode="r",
    )

    strict_ids = np.load(
        prod_dir
        /
        "strict_point_ids.npy"
    ).astype(
        np.int32,
        copy=False,
    )

    npoint = int(
        strict_ids.size
    )

    ndate = len(
        stack.dates
    )

    if S.shape != (
        npoint,
        ndate,
    ):
        raise RuntimeError(
            f"Sensitivity shape "
            f"{S.shape} != "
            f"({npoint}, {ndate})"
        )

    # ========================================================
    # Correct time axis.
    # ========================================================

    dates = np.asarray(
        [
            parse_date(x)
            for x in stack.dates
        ],
        dtype="datetime64[D]",
    )

    days = (
        dates
        -
        dates[0]
    ).astype(
        "timedelta64[D]"
    ).astype(
        np.float64
    )

    years = (
        days
        /
        365.2425
    )

    # Center time to remove intercept/time numerical
    # correlation.
    tc = (
        years
        -
        np.mean(
            years
        )
    )

    var_t = float(
        np.sum(
            tc
            *
            tc
        )
    )

    if var_t <= 0:
        raise RuntimeError(
            "Invalid time axis"
        )

    # ========================================================
    # Output metrics
    # ========================================================

    corr_path = (
        outdir
        /
        "sensitivity_time_correlation.npy"
    )

    rms_center_path = (
        outdir
        /
        "sensitivity_centered_rms_rad_per_m.npy"
    )

    rms_perp_path = (
        outdir
        /
        "sensitivity_time_orthogonal_rms_rad_per_m.npy"
    )

    retained_path = (
        outdir
        /
        "sensitivity_identifiable_fraction.npy"
    )

    vif_path = (
        outdir
        /
        "sensitivity_time_vif.npy"
    )

    corr_out = np.lib.format.open_memmap(
        corr_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint,),
    )

    rms_center_out = np.lib.format.open_memmap(
        rms_center_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint,),
    )

    rms_perp_out = np.lib.format.open_memmap(
        rms_perp_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint,),
    )

    retained_out = np.lib.format.open_memmap(
        retained_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint,),
    )

    vif_out = np.lib.format.open_memmap(
        vif_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint,),
    )

    temporal_ref_max = 0.0
    nonfinite_points = 0
    degenerate_points = 0
    identity_max_error = 0.0

    print("=" * 112)
    print(
        "Step 10b1 - SCLA geometry/time "
        "identifiability audit"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"strict points              : "
        f"{npoint:,}"
    )

    print(
        f"acquisitions               : "
        f"{ndate}"
    )

    print(
        f"time span                  : "
        f"{years[-1]:.4f} yr"
    )

    print(
        f"temporal reference         : "
        f"{stack.dates[0]}"
    )

    print(
        f"batch size                 : "
        f"{args.batch_size:,}"
    )

    # ========================================================
    # Full-scene streaming calculation
    # ========================================================

    for b0 in range(
        0,
        npoint,
        args.batch_size,
    ):

        b1 = min(
            b0
            +
            args.batch_size,
            npoint,
        )

        Sb = np.asarray(
            S[
                b0:b1
            ],
            dtype=np.float64,
        )

        finite_point = np.all(
            np.isfinite(
                Sb
            ),
            axis=1,
        )

        nonfinite_points += int(
            np.count_nonzero(
                ~finite_point
            )
        )

        if np.any(
            finite_point
        ):

            temporal_ref_max = max(
                temporal_ref_max,
                float(
                    np.max(
                        np.abs(
                            Sb[
                                finite_point,
                                0
                            ]
                        )
                    )
                ),
            )

        # -----------------------------------------------
        # Remove constant term.
        # -----------------------------------------------

        mean_s = np.mean(
            Sb,
            axis=1,
            keepdims=True,
        )

        Sc = (
            Sb
            -
            mean_s
        )

        ss_center = np.sum(
            Sc
            *
            Sc,
            axis=1,
        )

        rms_center = np.sqrt(
            ss_center
            /
            ndate
        )

        # -----------------------------------------------
        # Correlation between sensitivity and time.
        # -----------------------------------------------

        cov_st = np.sum(
            Sc
            *
            tc[
                None,
                :
            ],
            axis=1,
        )

        denom_corr = np.sqrt(
            ss_center
            *
            var_t
        )

        corr = np.full(
            b1 - b0,
            np.nan,
            dtype=np.float64,
        )

        good = (
            finite_point
            &
            np.isfinite(
                denom_corr
            )
            &
            (
                denom_corr
                >
                1.0e-15
            )
        )

        corr[
            good
        ] = (
            cov_st[
                good
            ]
            /
            denom_corr[
                good
            ]
        )

        # Numerical clipping only.
        corr[
            good
        ] = np.clip(
            corr[
                good
            ],
            -1.0,
            1.0,
        )

        # -----------------------------------------------
        # Remove best linear time trend from sensitivity.
        #
        # Since Sc already has zero mean and tc has zero
        # mean, this is projection onto time alone.
        # -----------------------------------------------

        slope_st = (
            cov_st
            /
            var_t
        )

        Sperp = (
            Sc
            -
            slope_st[
                :,
                None
            ]
            *
            tc[
                None,
                :
            ]
        )

        ss_perp = np.sum(
            Sperp
            *
            Sperp,
            axis=1,
        )

        rms_perp = np.sqrt(
            ss_perp
            /
            ndate
        )

        retained = np.full(
            b1 - b0,
            np.nan,
            dtype=np.float64,
        )

        retained[
            good
        ] = (
            rms_perp[
                good
            ]
            /
            rms_center[
                good
            ]
        )

        # -----------------------------------------------
        # VIF for the two predictors:
        #
        # time and S_h.
        #
        # VIF = 1 / (1-rho^2)
        # -----------------------------------------------

        vif = np.full(
            b1 - b0,
            np.nan,
            dtype=np.float64,
        )

        one_minus_r2 = (
            1.0
            -
            corr[
                good
            ]
            *
            corr[
                good
            ]
        )

        vif[
            good
        ] = (
            1.0
            /
            np.maximum(
                one_minus_r2,
                1.0e-12,
            )
        )

        degenerate_points += int(
            np.count_nonzero(
                finite_point
                &
                ~good
            )
        )

        # -----------------------------------------------
        # Algebraic QA:
        #
        # retained^2 == 1-corr^2
        # -----------------------------------------------

        if np.any(
            good
        ):

            identity_error = np.abs(
                retained[
                    good
                ]
                *
                retained[
                    good
                ]
                -
                (
                    1.0
                    -
                    corr[
                        good
                    ]
                    *
                    corr[
                        good
                    ]
                )
            )

            identity_max_error = max(
                identity_max_error,
                float(
                    np.max(
                        identity_error
                    )
                ),
            )

        corr_out[
            b0:b1
        ] = corr.astype(
            np.float32
        )

        rms_center_out[
            b0:b1
        ] = rms_center.astype(
            np.float32
        )

        rms_perp_out[
            b0:b1
        ] = rms_perp.astype(
            np.float32
        )

        retained_out[
            b0:b1
        ] = retained.astype(
            np.float32
        )

        vif_out[
            b0:b1
        ] = vif.astype(
            np.float32
        )

        print(
            f"  {b1:,}/"
            f"{npoint:,}"
        )

    for x in (
        corr_out,
        rms_center_out,
        rms_perp_out,
        retained_out,
        vif_out,
    ):
        x.flush()

    # ========================================================
    # Global distributions
    # ========================================================

    corr = np.asarray(
        corr_out,
        dtype=np.float64,
    )

    center_rms = np.asarray(
        rms_center_out,
        dtype=np.float64,
    )

    perp_rms = np.asarray(
        rms_perp_out,
        dtype=np.float64,
    )

    retained = np.asarray(
        retained_out,
        dtype=np.float64,
    )

    vif = np.asarray(
        vif_out,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(corr)
        &
        np.isfinite(center_rms)
        &
        np.isfinite(perp_rms)
        &
        np.isfinite(retained)
        &
        np.isfinite(vif)
    )

    abs_corr = np.abs(
        corr[
            valid
        ]
    )

    print()
    print("=" * 112)
    print(
        "Sensitivity/time correlation"
    )
    print("=" * 112)

    print_quantiles(
        "signed corr p01/p05/p50/p95/p99:",
        corr[
            valid
        ],
    )

    q = np.percentile(
        abs_corr,
        [
            50,
            90,
            95,
            99,
            100,
        ],
    )

    print(
        "|corr| p50/p90/p95/p99/max:"
    )

    print(
        "  "
        +
        " / ".join(
            f"{x:.6f}"
            for x in q
        )
    )

    print()

    for threshold in (
        0.80,
        0.90,
        0.95,
        0.98,
        0.99,
    ):

        n = int(
            np.count_nonzero(
                abs_corr
                >=
                threshold
            )
        )

        print(
            f"|corr| >= {threshold:.2f}      : "
            f"{n:8,d} "
            f"({100.0*n/abs_corr.size:.4f}%)"
        )

    print()
    print("=" * 112)
    print(
        "Topographic-sensitivity observability"
    )
    print("=" * 112)

    print_quantiles(
        "centered S RMS p01/p05/p50/p95/p99:",
        center_rms[
            valid
        ],
        fmt=".6e",
    )

    print_quantiles(
        "time-orthogonal S RMS p01/p05/p50/p95/p99:",
        perp_rms[
            valid
        ],
        fmt=".6e",
    )

    print_quantiles(
        "identifiable fraction p01/p05/p50/p95/p99:",
        retained[
            valid
        ],
    )

    print_quantiles(
        "VIF p50/p90/p95/p99/p99.9:",
        vif[
            valid
        ],
        qs=(
            50,
            90,
            95,
            99,
            99.9,
        ),
    )

    # This is NOT an assumed DEM error.
    # It is only a convenient sensitivity scale:
    # phase RMS produced by 10 m height coefficient after
    # removing the linear-time-degenerate part.
    distinguish_10m = (
        10.0
        *
        perp_rms[
            valid
        ]
    )

    print()
    print(
        "10 m coefficient distinguishable "
        "phase RMS p01/p05/p50/p95/p99:"
    )

    print(
        "  "
        +
        " / ".join(
            f"{x:.6f}"
            for x in np.percentile(
                distinguish_10m,
                [
                    1,
                    5,
                    50,
                    95,
                    99,
                ],
            )
        )
        +
        " rad"
    )

    print()
    print("=" * 112)
    print(
        "Numerical QA"
    )
    print("=" * 112)

    print(
        f"valid points               : "
        f"{np.count_nonzero(valid):,}/"
        f"{npoint:,}"
    )

    print(
        f"nonfinite points           : "
        f"{nonfinite_points:,}"
    )

    print(
        f"degenerate sensitivity pts : "
        f"{degenerate_points:,}"
    )

    print(
        f"temporal-ref max |S|       : "
        f"{temporal_ref_max:.3e} rad/m"
    )

    print(
        "max |retained^2-(1-rho^2)|:"
    )

    print(
        f"  {identity_max_error:.6e}"
    )

    # ========================================================
    # Status
    #
    # This step deliberately does NOT make the final
    # scientific decision about SCLA correction. It only
    # verifies that the design metrics are numerically valid.
    # ========================================================

    if (
        nonfinite_points != 0
        or
        degenerate_points != 0
    ):

        status = (
            "REVIEW_DEGENERATE_DESIGN"
        )

    elif temporal_ref_max > 1.0e-7:

        status = (
            "REVIEW_TEMPORAL_REFERENCE"
        )

    elif identity_max_error > 1.0e-8:

        status = (
            "REVIEW_PROJECTION_IDENTITY"
        )

    else:

        status = (
            "PASS"
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-scla-identifiability-audit-v09",

        "status":
            status,

        "points":
            int(
                npoint
            ),

        "acquisitions":
            int(
                ndate
            ),

        "time_span_years":
            float(
                years[-1]
            ),

        "model_being_tested":
            "phase = intercept + linear_time + height_coefficient * sensitivity",

        "sensitivity":
            str(
                sensitivity_path
            ),

        "metrics": {
            "sensitivity_time_correlation":
                str(
                    corr_path
                ),

            "centered_sensitivity_rms":
                str(
                    rms_center_path
                ),

            "time_orthogonal_sensitivity_rms":
                str(
                    rms_perp_path
                ),

            "identifiable_fraction":
                str(
                    retained_path
                ),

            "time_vif":
                str(
                    vif_path
                ),
        },

        "numerical_qa": {
            "nonfinite_points":
                int(
                    nonfinite_points
                ),

            "degenerate_points":
                int(
                    degenerate_points
                ),

            "temporal_reference_max_abs_rad_per_m":
                float(
                    temporal_ref_max
                ),

            "projection_identity_max_error":
                float(
                    identity_max_error
                ),
        },

        "phase_read":
            False,

        "height_coefficient_estimated":
            False,

        "phase_modified":
            False,

        "residual_dem_correction_applied":
            False,

        "scientific_acceptance_decision":
            False,
    }

    manifest_path = (
        outdir
        /
        "scla_identifiability_manifest.json"
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
    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10b1 STATUS: "
        f"{status} / "
        "SCLA IDENTIFIABILITY AUDIT ONLY"
    )

    print(
        "No phase was read, fitted, or corrected."
    )


if __name__ == "__main__":
    main()
