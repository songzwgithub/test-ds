#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
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

    root = Path(paths.output_dir) / "v09"

    invdir = (
        root
        / "network_inversion_v09"
    )

    step10a4 = (
        root
        / "scla_v09"
        / "fd_stepsize_audit"
    )

    step10a5 = (
        root
        / "scla_v09"
        / "production_sensitivity"
    )

    outdir = (
        root
        / "scla_v09"
        / "production_sensitivity"
        / "reproducibility_audit"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    strict_ids = np.load(
        invdir
        / "strict_point_ids.npy"
    ).astype(
        np.int64,
        copy=False,
    )

    sample_ids = np.load(
        step10a4
        / "sample_strict_point_ids.npy"
    ).astype(
        np.int64,
        copy=False,
    )

    sample20 = np.asarray(
        np.load(
            step10a4
            / "sample_sensitivity_dh20m_rad_per_m.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    production = np.load(
        step10a5
        / "topographic_phase_sensitivity_rad_per_m.npy",
        mmap_mode="r",
    )

    # --------------------------------------------------------
    # Map exact PointPhaseStack IDs back into strict domain.
    # --------------------------------------------------------

    pos = np.searchsorted(
        strict_ids,
        sample_ids,
    )

    if (
        np.any(pos >= strict_ids.size)
        or
        not np.array_equal(
            strict_ids[pos],
            sample_ids,
        )
    ):
        raise RuntimeError(
            "Sample point ID mapping failed"
        )

    prod_sample = np.asarray(
        production[pos],
        dtype=np.float64,
    )

    if prod_sample.shape != sample20.shape:
        raise RuntimeError(
            f"shape mismatch: "
            f"{prod_sample.shape} vs "
            f"{sample20.shape}"
        )

    d = (
        prod_sample
        -
        sample20
    )

    ad = np.abs(d)

    n_total = int(
        d.size
    )

    rms = float(
        np.sqrt(
            np.mean(
                d * d
            )
        )
    )

    mx = float(
        np.max(ad)
    )

    print("=" * 112)
    print(
        "Step 10a5b - Production/sample "
        "reproducibility mismatch audit"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"sample points              : "
        f"{sample_ids.size:,}"
    )

    print(
        f"acquisitions               : "
        f"{len(stack.dates)}"
    )

    print(
        f"total compared values      : "
        f"{n_total:,}"
    )

    print(
        f"RMS difference             : "
        f"{rms:.9e} rad/m"
    )

    print(
        f"maximum difference         : "
        f"{mx:.9e} rad/m"
    )

    # --------------------------------------------------------
    # Threshold counts
    # --------------------------------------------------------

    thresholds = [
        0.0,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        5e-5,
        9e-5,
    ]

    print()
    print("=" * 112)
    print(
        "Mismatch counts"
    )
    print("=" * 112)

    counts = {}

    for t in thresholds:

        n = int(
            np.count_nonzero(
                ad > t
            )
        )

        counts[str(t)] = n

        print(
            f"|difference| > {t:.1e} : "
            f"{n:8,d} "
            f"({100.0*n/n_total:.8f}%)"
        )

    # --------------------------------------------------------
    # Affected points/acquisitions
    # --------------------------------------------------------

    bad = (
        ad > 1e-5
    )

    affected_point_local = np.flatnonzero(
        np.any(
            bad,
            axis=1,
        )
    )

    affected_acq = np.flatnonzero(
        np.any(
            bad,
            axis=0,
        )
    )

    print()
    print("=" * 112)
    print(
        "Affected domain for |difference| > 1e-5"
    )
    print("=" * 112)

    print(
        f"affected sample points     : "
        f"{affected_point_local.size:,}"
    )

    print(
        f"affected acquisitions      : "
        f"{affected_acq.size:,}"
    )

    if affected_acq.size:

        print(
            "acquisitions:"
        )

        for j in affected_acq:

            n = int(
                np.count_nonzero(
                    bad[:, j]
                )
            )

            print(
                f"  {j:2d} "
                f"{stack.dates[j]} "
                f"count={n}"
            )

    # --------------------------------------------------------
    # Difference-value histogram.
    #
    # Round only for display so we can see discrete quantized
    # levels rather than thousands of FLOAT textual variants.
    # --------------------------------------------------------

    nz = d[
        ad > 1e-8
    ]

    rounded = np.round(
        nz,
        decimals=10,
    )

    counter = Counter(
        rounded.tolist()
    )

    print()
    print("=" * 112)
    print(
        "Most common nonzero difference levels"
    )
    print("=" * 112)

    for value, count in counter.most_common(
        30
    ):

        print(
            f"{value:+.10e} "
            f"count={count}"
        )

    # --------------------------------------------------------
    # Largest individual mismatches.
    # --------------------------------------------------------

    flat_order = np.argsort(
        ad.ravel()
    )[::-1]

    print()
    print("=" * 112)
    print(
        "Largest individual mismatches"
    )
    print("=" * 112)

    print(
        " rank sample_idx point_id acq date "
        "sample20 production difference"
    )

    rows_out = []

    nshow = min(
        50,
        flat_order.size,
    )

    for rank, flat in enumerate(
        flat_order[:nshow],
        start=1,
    ):

        ip, jt = np.unravel_index(
            flat,
            d.shape,
        )

        if ad[
            ip,
            jt
        ] == 0:
            break

        row = {
            "rank":
                rank,

            "sample_index":
                int(ip),

            "strict_position":
                int(pos[ip]),

            "point_id":
                int(sample_ids[ip]),

            "acquisition_index":
                int(jt),

            "date":
                str(stack.dates[jt]),

            "sample20_rad_per_m":
                float(sample20[ip, jt]),

            "production20_rad_per_m":
                float(prod_sample[ip, jt]),

            "difference_rad_per_m":
                float(d[ip, jt]),
        }

        rows_out.append(
            row
        )

        print(
            f"{rank:4d} "
            f"{ip:8d} "
            f"{sample_ids[ip]:8d} "
            f"{jt:3d} "
            f"{stack.dates[jt]} "
            f"{sample20[ip,jt]:+.9e} "
            f"{prod_sample[ip,jt]:+.9e} "
            f"{d[ip,jt]:+.9e}"
        )

    # --------------------------------------------------------
    # Batch-boundary check.
    #
    # Production 10a5 used batch_size=65536.
    # Check whether affected strict positions cluster around
    # those boundaries.
    # --------------------------------------------------------

    batch_size = 65536

    affected_strict_pos = pos[
        affected_point_local
    ]

    dist_to_boundary = []

    for p in affected_strict_pos:

        rem = int(
            p % batch_size
        )

        dist = min(
            rem,
            batch_size - rem,
        )

        dist_to_boundary.append(
            dist
        )

    dist_to_boundary = np.asarray(
        dist_to_boundary,
        dtype=np.int64,
    )

    print()
    print("=" * 112)
    print(
        "Production batch-boundary check"
    )
    print("=" * 112)

    if dist_to_boundary.size:

        print(
            f"min distance to 65536 boundary : "
            f"{dist_to_boundary.min()}"
        )

        print(
            f"median distance                : "
            f"{np.median(dist_to_boundary):.1f}"
        )

        print(
            f"points within 1                : "
            f"{np.count_nonzero(dist_to_boundary <= 1)}"
        )

        print(
            f"points within 10               : "
            f"{np.count_nonzero(dist_to_boundary <= 10)}"
        )

        print(
            f"points within 100              : "
            f"{np.count_nonzero(dist_to_boundary <= 100)}"
        )

    else:

        print(
            "No >1e-5 mismatches."
        )

    # --------------------------------------------------------
    # Decision.
    #
    # For reproducibility of FLOAT GAMMA geometry:
    # - global RMS must remain negligible;
    # - mismatch must be sparse;
    # - no systematic acquisition/batch structure;
    # - maximum must remain within the observed single-
    #   quantization-level regime.
    # --------------------------------------------------------

    n_gt_1e5 = int(
        np.count_nonzero(
            ad > 1e-5
        )
    )

    frac_gt_1e5 = (
        n_gt_1e5
        /
        n_total
    )

    if (
        rms <= 1e-6
        and
        frac_gt_1e5 <= 1e-4
        and
        mx <= 1.1e-4
    ):

        status = (
            "PASS_FLOAT_QUANTIZATION_LIMITED"
        )

    else:

        status = (
            "REVIEW_REPRODUCIBILITY"
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-production-sensitivity-reproducibility-audit-v09",

        "status":
            status,

        "sample_points":
            int(sample_ids.size),

        "acquisitions":
            int(len(stack.dates)),

        "total_values":
            n_total,

        "rms_difference_rad_per_m":
            rms,

        "max_difference_rad_per_m":
            mx,

        "counts_above_threshold":
            counts,

        "affected_points_gt_1e5":
            int(
                affected_point_local.size
            ),

        "affected_acquisitions_gt_1e5":
            int(
                affected_acq.size
            ),

        "fraction_values_gt_1e5":
            float(
                frac_gt_1e5
            ),

        "interpretation":
            (
                "Audit distinguishes sparse FLOAT-level "
                "reproducibility differences from systematic "
                "geometry or indexing errors."
            ),
    }

    manifest_path = (
        outdir
        /
        "reproducibility_audit_manifest.json"
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
        f"STEP 10a5b STATUS: {status}"
    )


if __name__ == "__main__":
    main()
