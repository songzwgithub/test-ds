#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from pypsds.prototype import open_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ds-mask", required=True)
    ap.add_argument("--ref-tol", type=float, default=1e-5)
    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (_, _, H, W),
    ) = open_from_config(args.config)

    d = Path(paths.output_dir) / "v09"

    ps = np.load(d / "ps_mask.npy").astype(bool)
    ds = np.load(args.ds_mask).astype(bool)
    pl = np.load(d / "pl_valid.npy", mmap_mode="r")
    tc = np.load(d / "temporal_coherence.npy", mmap_mode="r")
    pair = np.load(d / "median_pair_coherence.npy", mmap_mode="r")
    phase = np.load(d / "linked_phase.npy", mmap_mode="r")

    if phase.shape != (len(stack.dates), H, W):
        raise RuntimeError(
            f"linked_phase shape={phase.shape}, "
            f"expected={(len(stack.dates),H,W)}"
        )

    print("=" * 80)
    print("Step 06a - PointPhaseStack input integrity audit")
    print("=" * 80)

    print(f"dates                  : {len(stack.dates)}")
    print(f"reference acquisition  : {stack.dates[0]}")
    print(f"PS                     : {ps.sum()}")
    print(f"final DS               : {ds.sum()}")

    overlap = ps & ds

    print(f"PS/DS overlap          : {overlap.sum()}")

    # ---------------------------------------------------------
    # DS contract
    # ---------------------------------------------------------
    ds_not_pl = ds & ~np.asarray(pl)
    ds_bad_tc = ds & ~np.isfinite(np.asarray(tc))
    ds_bad_pair = ds & ~np.isfinite(np.asarray(pair))

    print()
    print("DS integrity:")
    print(f"  selected but !PL     : {ds_not_pl.sum()}")
    print(f"  non-finite TC        : {ds_bad_tc.sum()}")
    print(f"  non-finite pair coh  : {ds_bad_pair.sum()}")

    # ---------------------------------------------------------
    # Complete complex phase history
    # ---------------------------------------------------------
    finite_phase = np.all(
        np.isfinite(phase.real)
        & np.isfinite(phase.imag),
        axis=0,
    )

    ps_phase_ok = ps & finite_phase
    ds_phase_ok = ds & finite_phase

    ps_phase_bad = ps & ~finite_phase
    ds_phase_bad = ds & ~finite_phase

    print()
    print("Phase-history completeness:")
    print(
        f"  PS complete          : "
        f"{ps_phase_ok.sum()}/{ps.sum()}"
    )
    print(
        f"  PS incomplete        : "
        f"{ps_phase_bad.sum()}"
    )
    print(
        f"  DS complete          : "
        f"{ds_phase_ok.sum()}/{ds.sum()}"
    )
    print(
        f"  DS incomplete        : "
        f"{ds_phase_bad.sum()}"
    )

    # ---------------------------------------------------------
    # Common reference audit
    # ---------------------------------------------------------
    point_valid = (
        (ps | ds)
        & finite_phase
    )

    r, c = np.where(point_valid)

    ref_phase = np.angle(
        phase[0, r, c]
    )

    max_ref = (
        float(np.max(np.abs(ref_phase)))
        if len(ref_phase)
        else np.nan
    )

    p99_ref = (
        float(np.percentile(np.abs(ref_phase), 99))
        if len(ref_phase)
        else np.nan
    )

    print()
    print("Common phase reference:")
    print(
        f"  |phase(t0)| p99      : "
        f"{p99_ref:.9e} rad"
    )
    print(
        f"  |phase(t0)| max      : "
        f"{max_ref:.9e} rad"
    )

    # Unit phasor magnitude is not mathematically required for
    # PointPhaseStack after angle(), but it is a useful integrity check.
    mag = np.abs(
        phase[:, r, c]
    )

    max_mag_err = (
        float(
            np.max(
                np.abs(mag - 1.0)
            )
        )
        if mag.size
        else np.nan
    )

    print(
        f"  max ||z|-1|         : "
        f"{max_mag_err:.9e}"
    )

    # ---------------------------------------------------------
    # Fusion contract
    # PS priority on overlap
    # ---------------------------------------------------------
    usable_ps = ps & finite_phase
    usable_ds = ds & finite_phase & ~usable_ps

    fused = usable_ps | usable_ds

    print()
    print("Fusion preview:")
    print(
        f"  usable PS            : "
        f"{usable_ps.sum()}"
    )
    print(
        f"  usable DS            : "
        f"{usable_ds.sum()}"
    )
    print(
        f"  PointPhaseStack N    : "
        f"{fused.sum()}"
    )

    fail = False

    if overlap.sum() != 0:
        print(
            "WARNING: PS/DS overlap exists; "
            "Step06 PS-priority rule will remove DS duplicates."
        )

    if ds_not_pl.sum() != 0:
        print("FAIL: final DS contains non-phase-linked pixels.")
        fail = True

    if ds_bad_tc.sum() != 0 or ds_bad_pair.sum() != 0:
        print("FAIL: final DS has invalid quality values.")
        fail = True

    if ds_phase_bad.sum() != 0:
        print("FAIL: final DS has incomplete phase histories.")
        fail = True

    if ps_phase_bad.sum() != 0:
        print()
        print(
            "ATTENTION: some PS have no complete geometry-corrected "
            "phase history."
        )
        print(
            "Do NOT build PointPhaseStack with the current Step06 "
            "until these PS are filtered."
        )
        fail = True

    if np.isfinite(max_ref) and max_ref > args.ref_tol:
        print(
            f"FAIL: acquisition-0 phase is not zero "
            f"within tolerance {args.ref_tol} rad."
        )
        fail = True

    print()
    if fail:
        print("AUDIT STATUS: FAIL")
        raise SystemExit(2)

    print("AUDIT STATUS: PASS")
    print(
        "PS and final DS can be fused into one PointPhaseStack."
    )


if __name__ == "__main__":
    main()
