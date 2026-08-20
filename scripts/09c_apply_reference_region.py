#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def robust_mad_sigma(x, axis=0):
    med = np.median(
        x,
        axis=axis,
    )

    expanded = np.expand_dims(
        med,
        axis=axis,
    )

    mad = np.median(
        np.abs(
            x - expanded
        ),
        axis=axis,
    )

    return (
        1.4826 * mad
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--center-row",
        type=int,
        default=538,
    )

    ap.add_argument(
        "--center-col",
        type=int,
        default=337,
    )

    ap.add_argument(
        "--half-row",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--half-col",
        type=int,
        default=15,
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
        / "processing"
    )

    pps_dir = (
        root
        / "point_phase_stack"
    )

    inversion_dir = (
        root
        / "network_inversion"
    )

    outdir = (
        root
        / "referenced_timeseries"
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
        ),
        dtype=np.int32,
    )

    all_cols = np.asarray(
        np.load(
            pps_dir
            / "cols.npy",
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

    # ========================================================
    # Reference-region membership
    # ========================================================

    region = (
        (
            np.abs(
                rows
                -
                args.center_row
            )
            <=
            args.half_row
        )
        &
        (
            np.abs(
                cols
                -
                args.center_col
            )
            <=
            args.half_col
        )
    )

    region_idx = np.where(
        region
    )[0]

    nref = region_idx.size

    if nref < 100:
        raise RuntimeError(
            f"Reference region has only "
            f"{nref} strict points."
        )

    # ========================================================
    # Epoch-wise robust reference
    # ========================================================

    ref_phase_stack = np.asarray(
        phase[
            region_idx,
            :
        ],
        dtype=np.float64,
    )

    reference_median = np.median(
        ref_phase_stack,
        axis=0,
    )

    reference_sigma = robust_mad_sigma(
        ref_phase_stack,
        axis=0,
    )

    # ========================================================
    # Apply to all strict points
    # ========================================================

    output_path = (
        outdir
        / "acquisition_phase_referenced_rad.npy"
    )

    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            nstrict,
            ndate,
        ),
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

        output[
            b0:b1,
            :
        ] = (
            Y
            -
            reference_median[
                None,
                :
            ]
        ).astype(
            np.float32
        )

    output.flush()

    # ========================================================
    # Reference verification
    # ========================================================

    referenced = np.asarray(
        output[
            region_idx,
            :
        ],
        dtype=np.float64,
    )

    post_ref_median = np.median(
        referenced,
        axis=0,
    )

    post_ref_sigma = robust_mad_sigma(
        referenced,
        axis=0,
    )

    # First acquisition should remain approximately zero.
    first_epoch_max = float(
        np.max(
            np.abs(
                np.asarray(
                    output[:, 0],
                    dtype=np.float64,
                )
            )
        )
    )

    # ========================================================
    # Preliminary relative linear phase rate
    #
    # This is NOT yet final deformation velocity.
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

    t_year = (
        t_days / 365.25
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

    rate_path = (
        outdir
        / "preliminary_phase_rate_rad_per_year.npy"
    )

    rate = np.lib.format.open_memmap(
        rate_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            nstrict,
        ),
    )

    rate_residual_rms_path = (
        outdir
        / "preliminary_linear_residual_rms_rad.npy"
    )

    rate_residual_rms = np.lib.format.open_memmap(
        rate_residual_rms_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            nstrict,
        ),
    )

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
            output[
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
            (
                Y
                -
                ymean[
                    :,
                    None
                ]
            )
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
            a[
                :,
                None
            ]
            +
            s[
                :,
                None
            ]
            *
            t_year[
                None,
                :
            ]
        )

        residual = (
            Y - fit
        )

        rms = np.sqrt(
            np.mean(
                residual
                *
                residual,
                axis=1,
            )
        )

        rate[
            b0:b1
        ] = s.astype(
            np.float32
        )

        rate_residual_rms[
            b0:b1
        ] = rms.astype(
            np.float32
        )

    rate.flush()
    rate_residual_rms.flush()

    # ========================================================
    # Reference-region phase rate QA
    # ========================================================

    ref_rate = np.asarray(
        rate[
            region_idx
        ],
        dtype=np.float64,
    )

    ref_rate_median = float(
        np.median(
            ref_rate
        )
    )

    ref_rate_sigma = float(
        1.4826
        *
        np.median(
            np.abs(
                ref_rate
                -
                ref_rate_median
            )
        )
    )

    # ========================================================
    # Save region masks
    # ========================================================

    full_mask = np.zeros(
        all_rows.size,
        dtype=bool,
    )

    full_mask[
        strict_ids[
            region_idx
        ]
    ] = True

    np.save(
        outdir
        / "reference_region_mask.npy",
        full_mask,
    )

    np.save(
        outdir
        / "reference_strict_indices.npy",
        region_idx.astype(
            np.int32
        ),
    )

    np.save(
        outdir
        / "reference_point_ids.npy",
        strict_ids[
            region_idx
        ].astype(
            np.int32
        ),
    )

    np.save(
        outdir
        / "reference_phase_median_rad.npy",
        reference_median.astype(
            np.float32
        ),
    )

    np.save(
        outdir
        / "reference_phase_mad_sigma_rad.npy",
        reference_sigma.astype(
            np.float32
        ),
    )

    # ========================================================
    # Epoch QA table
    # ========================================================

    qa_rows = []

    for t in range(
        ndate
    ):

        qa_rows.append({
            "index":
                t,

            "date":
                str(
                    stack.dates[
                        t
                    ]
                ),

            "reference_phase_median_before_rad":
                float(
                    reference_median[
                        t
                    ]
                ),

            "reference_phase_mad_sigma_before_rad":
                float(
                    reference_sigma[
                        t
                    ]
                ),

            "reference_phase_median_after_rad":
                float(
                    post_ref_median[
                        t
                    ]
                ),

            "reference_phase_mad_sigma_after_rad":
                float(
                    post_ref_sigma[
                        t
                    ]
                ),
        })

    qa_csv = (
        outdir
        / "reference_epoch_qa.csv"
    )

    with qa_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                qa_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            qa_rows
        )

    # ========================================================
    # Console QA
    # ========================================================

    phase_rate_arr = np.asarray(
        rate,
        dtype=np.float64,
    )

    residual_arr = np.asarray(
        rate_residual_rms,
        dtype=np.float64,
    )

    print("=" * 110)
    print(
        "Step 09c - Apply computational reference region"
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
        f"reference center row/col   : "
        f"{args.center_row}, "
        f"{args.center_col}"
    )

    print(
        f"reference window           : "
        f"{2*args.half_row+1} x "
        f"{2*args.half_col+1}"
    )

    print(
        f"reference points           : "
        f"{nref}"
    )

    print()
    print("=" * 110)
    print(
        "Reference application QA"
    )
    print("=" * 110)

    print(
        f"max |epoch median after|   : "
        f"{np.max(np.abs(post_ref_median)):.3e} rad"
    )

    print(
        f"first-acquisition max |phase|:"
    )

    print(
        f"  {first_epoch_max:.3e} rad"
    )

    print(
        f"reference phase-rate median:"
    )

    print(
        f"  {ref_rate_median:.6e} rad/yr"
    )

    print(
        f"reference phase-rate MAD sigma:"
    )

    print(
        f"  {ref_rate_sigma:.6e} rad/yr"
    )

    print()
    print("=" * 110)
    print(
        "Preliminary relative phase-rate distribution"
    )
    print("=" * 110)

    q_rate = np.percentile(
        phase_rate_arr,
        [
            1,
            5,
            50,
            95,
            99,
        ],
    )

    print(
        "phase rate p01/p05/p50/p95/p99:"
    )

    print(
        "  "
        +
        " / ".join(
            f"{x:.6f}"
            for x in q_rate
        )
        +
        " rad/yr"
    )

    q_res = np.percentile(
        residual_arr,
        [
            50,
            90,
            95,
            99,
        ],
    )

    print(
        "linear residual RMS "
        "p50/p90/p95/p99:"
    )

    print(
        "  "
        +
        " / ".join(
            f"{x:.6f}"
            for x in q_res
        )
        +
        " rad"
    )

    # ========================================================
    # Manifest
    # ========================================================

    manifest = {
        "format":
            "pyPSDS-GAMMA-referenced-timeseries-processing",

        "status":
            "PRELIMINARY_REFERENCED_PHASE",

        "reference_type":
            "computational_region_median",

        "geodetic_stability_claim":
            False,

        "reference_region": {
            "center_row":
                int(
                    args.center_row
                ),

            "center_col":
                int(
                    args.center_col
                ),

            "half_row":
                int(
                    args.half_row
                ),

            "half_col":
                int(
                    args.half_col
                ),

            "points":
                int(
                    nref
                ),
        },

        "products": {
            "referenced_acquisition_phase":
                str(
                    output_path
                ),

            "preliminary_phase_rate":
                str(
                    rate_path
                ),

            "linear_residual_rms":
                str(
                    rate_residual_rms_path
                ),
        },

        "corrections_applied": {
            "spatial_reference":
                True,

            "SCLA":
                False,

            "residual_DEM":
                False,

            "APS":
                False,

            "GACOS":
                False,

            "ERA5":
                False,

            "ramp":
                False,

            "SCN":
                False,
        },

        "LOS_displacement_created":
            False,

        "note":
            (
                "Reference is a computational region median. "
                "No geodetic zero-velocity assumption is made. "
                "LOS displacement sign conversion is intentionally "
                "deferred to a dedicated sign-convention quality."
            ),
    }

    manifest_path = (
        outdir
        / "referenced_timeseries_manifest.json"
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
        f"referenced phase           : "
        f"{output_path}"
    )

    print(
        f"preliminary phase rate     : "
        f"{rate_path}"
    )

    print(
        f"reference epoch QA         : "
        f"{qa_csv}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP 09c STATUS: PASS / "
        "PRELIMINARY REFERENCED PHASE"
    )

    print(
        "No SCLA, APS, atmospheric or "
        "LOS sign conversion has been applied."
    )


if __name__ == "__main__":
    main()
