#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def find_geometry_mask(processing_dir: Path):
    candidates = [
        processing_dir / "cache" / "phase_geometry_valid.npy",
        processing_dir / "phase_geometry_valid.npy",
        processing_dir / "cache" / "geometry_valid.npy",
        processing_dir / "geometry_valid.npy",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Classify incomplete PS phase histories and "
            "build the production geometry-valid PS mask."
        )
    )

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
        (_, _, H, W),
    ) = open_from_config(
        args.config
    )

    processing = (
        Path(paths.output_dir)
        / "processing"
    )

    ps_path = (
        processing
        / "ps_mask.npy"
    )

    phase_path = (
        processing
        / "linked_phase.npy"
    )

    print("=" * 80)
    print(
        "Step 06b - Final PS geometry/phase quality"
    )
    print("=" * 80)

    print(
        f"config          : {config_path}"
    )

    print(
        f"scene           : {H} x {W}"
    )

    print(
        f"dates           : {len(stack.dates)}"
    )

    print()
    print(
        "[06b:1/4] Loading PS and linked phase..."
    )

    ps = np.load(
        ps_path,
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    phase = np.load(
        phase_path,
        mmap_mode="r",
    )

    if ps.shape != (H, W):
        raise RuntimeError(
            f"PS shape {ps.shape}, "
            f"expected {(H,W)}"
        )

    if phase.shape != (
        len(stack.dates),
        H,
        W,
    ):
        raise RuntimeError(
            f"phase shape {phase.shape}, "
            f"expected "
            f"{(len(stack.dates),H,W)}"
        )

    phase_finite = np.all(
        np.isfinite(phase.real)
        & np.isfinite(phase.imag),
        axis=0,
    )

    ps_complete = (
        ps
        & phase_finite
    )

    ps_incomplete = (
        ps
        & ~phase_finite
    )

    print(
        f"raw PS          : "
        f"{ps.sum()}"
    )

    print(
        f"complete PS     : "
        f"{ps_complete.sum()}"
    )

    print(
        f"incomplete PS   : "
        f"{ps_incomplete.sum()}"
    )

    # ==========================================================
    # Geometry-valid classification
    # ==========================================================

    print()
    print(
        "[06b:2/4] Classifying against geometry-valid mask..."
    )

    geom_path = find_geometry_mask(
        processing
    )

    if geom_path is None:
        raise RuntimeError(
            "Could not locate phase_geometry_valid.npy "
            "under processing or processing/cache."
        )

    print(
        f"geometry mask   : {geom_path}"
    )

    geom = np.load(
        geom_path,
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    if geom.shape != (
        H,
        W,
    ):
        raise RuntimeError(
            f"geometry mask shape "
            f"{geom.shape}, expected {(H,W)}"
        )

    ps_geom_invalid = (
        ps
        & ~geom
    )

    ps_geom_valid_but_incomplete = (
        ps
        & geom
        & ~phase_finite
    )

    ps_geom_invalid_but_complete = (
        ps
        & ~geom
        & phase_finite
    )

    print(
        f"geometry valid  : "
        f"{geom.sum()}/{H*W} "
        f"({100*geom.mean():.3f}%)"
    )

    print()
    print(
        "PS failure classification:"
    )

    print(
        f"  PS geometry-invalid          : "
        f"{ps_geom_invalid.sum()}"
    )

    print(
        f"  geom-valid but phase-missing : "
        f"{ps_geom_valid_but_incomplete.sum()}"
    )

    print(
        f"  geom-invalid but phase-valid : "
        f"{ps_geom_invalid_but_complete.sum()}"
    )

    # ==========================================================
    # Detailed rejected PS report
    # ==========================================================

    print()
    print(
        "[06b:3/4] Writing rejected-PS diagnostics..."
    )

    outdir = (
        processing
        / "step06_ps_finalize"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rr, cc = np.where(
        ps_incomplete
    )

    rejected_csv = (
        outdir
        / "rejected_ps.csv"
    )

    with open(
        rejected_csv,
        "w",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "row",
            "col",
            "geometry_valid",
            "phase_complete",
            "first_bad_date_index",
            "first_bad_date",
        ])

        for r, c in zip(
            rr,
            cc,
        ):

            good_t = (
                np.isfinite(
                    phase[:, r, c].real
                )
                & np.isfinite(
                    phase[:, r, c].imag
                )
            )

            bad_ids = np.where(
                ~good_t
            )[0]

            if len(bad_ids):
                first_bad = int(
                    bad_ids[0]
                )

                first_bad_date = (
                    stack.dates[
                        first_bad
                    ]
                )
            else:
                first_bad = -1
                first_bad_date = ""

            w.writerow([
                int(r),
                int(c),
                int(geom[r, c]),
                int(phase_finite[r, c]),
                first_bad,
                first_bad_date,
            ])

    # ==========================================================
    # Finalize
    # ==========================================================

    print()
    print(
        "[06b:4/4] Finalizing production PS mask..."
    )

    # Production contract:
    # raw ADI PS
    # AND geometry-valid
    # AND complete phase history
    final_ps = (
        ps
        & geom
        & phase_finite
    )

    rejected = (
        ps
        & ~final_ps
    )

    np.save(
        processing
        / "final_ps_mask.npy",
        final_ps,
    )

    np.save(
        outdir
        / "rejected_ps_mask.npy",
        rejected,
    )

    print()
    print("=" * 80)
    print(
        "Final PS result"
    )
    print("=" * 80)

    print(
        f"raw ADI PS       : "
        f"{ps.sum()}"
    )

    print(
        f"final usable PS  : "
        f"{final_ps.sum()}"
    )

    print(
        f"rejected PS      : "
        f"{rejected.sum()}"
    )

    print(
        f"retained         : "
        f"{100*final_ps.sum()/ps.sum():.6f}%"
    )

    print(
        f"final mask       : "
        f"{processing/'final_ps_mask.npy'}"
    )

    print(
        f"rejected CSV     : "
        f"{rejected_csv}"
    )

    # ----------------------------------------------------------
    # Decision
    # ----------------------------------------------------------

    unexplained = int(
        ps_geom_valid_but_incomplete.sum()
    )

    if unexplained != 0:

        print()
        print(
            "QUALITY STATUS: FAIL"
        )

        print(
            f"{unexplained} PS pixels are geometry-valid "
            "but still have incomplete phase histories."
        )

        print(
            "These require phase-cache / PS-fill debugging."
        )

        raise SystemExit(2)

    print()
    print(
        "QUALITY STATUS: PASS"
    )

    print(
        "All incomplete PS are explained by "
        "the geometry-valid mask."
    )

    print(
        "Use final_ps_mask.npy for PointPhaseStack."
    )


if __name__ == "__main__":
    main()
