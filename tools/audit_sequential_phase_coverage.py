#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--processing-dir",
        required=True,
        help="pyPSDS processing directory",
    )

    args = ap.parse_args()

    processing = Path(args.processing_dir)
    seqdir = processing / "sequential"

    required_path = (
        seqdir
        / "compression_required_mask.npy"
    )

    pl_path = (
        processing
        / "pl_valid.npy"
    )

    phase_path = (
        processing
        / "linked_phase.npy"
    )

    ps_path = (
        processing
        / "ps_mask.npy"
    )

    # --------------------------------------------------------
    # Required inputs
    # --------------------------------------------------------

    for p in (
        required_path,
        pl_path,
        phase_path,
        ps_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing required file: {p}"
            )

    required = np.load(
        required_path,
        mmap_mode="r",
    )

    pl_valid = np.load(
        pl_path,
        mmap_mode="r",
    )

    ps = np.load(
        ps_path,
        mmap_mode="r",
    )

    linked_phase = np.load(
        phase_path,
        mmap_mode="r",
    )

    if required.ndim != 2:
        raise RuntimeError(
            f"compression_required_mask shape="
            f"{required.shape}, expected 2-D"
        )

    H, W = required.shape

    if pl_valid.shape != (H, W):
        raise RuntimeError(
            f"pl_valid shape={pl_valid.shape}, "
            f"expected {(H, W)}"
        )

    if ps.shape != (H, W):
        raise RuntimeError(
            f"ps_mask shape={ps.shape}, "
            f"expected {(H, W)}"
        )

    # --------------------------------------------------------
    # Detect linked-phase layout.
    #
    # Current production layout:
    #     (T, H, W)
    #
    # HWT is retained defensively for future refactors.
    # --------------------------------------------------------

    if (
        linked_phase.ndim == 3
        and
        linked_phase.shape[1:] == (H, W)
    ):

        layout = "THW"
        ndate = linked_phase.shape[0]

        def phase_slice(t):
            return linked_phase[t, :, :]

    elif (
        linked_phase.ndim == 3
        and
        linked_phase.shape[:2] == (H, W)
    ):

        layout = "HWT"
        ndate = linked_phase.shape[2]

        def phase_slice(t):
            return linked_phase[:, :, t]

    else:

        raise RuntimeError(
            "Cannot identify linked_phase layout: "
            f"{linked_phase.shape}; "
            f"scene={(H, W)}"
        )

    # --------------------------------------------------------
    # Stream through temporal dimension.
    #
    # DO NOT create:
    #
    #     isfinite(linked_phase)
    #
    # for the whole T x H x W cube.
    #
    # Only one H x W temporary boolean slice exists at once.
    # --------------------------------------------------------

    phase_all_finite = np.ones(
        (H, W),
        dtype=np.bool_,
    )

    phase_first_finite = None
    phase_last_finite = None

    for t in range(ndate):

        ph = phase_slice(t)

        finite = (
            np.isfinite(
                ph
            )
            &
            (
                ph
                !=
                np.complex64(0.0)
            )
        )

        if t == 0:
            phase_first_finite = (
                finite.copy()
            )

        if t == ndate - 1:
            phase_last_finite = (
                finite.copy()
            )

        phase_all_finite &= finite

        if (
            t == ndate - 1
            or
            (t + 1) % 10 == 0
        ):
            print(
                "phase integrity "
                f"{t + 1:3d}/{ndate:3d}"
            )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    req = np.asarray(
        required,
        dtype=bool,
    )

    pl = np.asarray(
        pl_valid,
        dtype=bool,
    )

    ps_bool = np.asarray(
        ps,
        dtype=bool,
    )

    # A phase source currently exists if:
    #
    #   - DS phase linking succeeded, or
    #   - the pixel is PS and raw referenced phase was filled.
    #
    declared_phase_source = (
        pl
        |
        ps_bool
    )

    required_pl = (
        req
        &
        pl
    )

    required_ps = (
        req
        &
        ps_bool
    )

    required_source = (
        req
        &
        declared_phase_source
    )

    # Declared PL/PS location but at least one epoch is invalid.
    integrity_failure = (
        req
        &
        declared_phase_source
        &
        ~phase_all_finite
    )

    # Fully usable with current production linked-phase cube.
    available = (
        req
        &
        declared_phase_source
        &
        phase_all_finite
    )

    missing = (
        req
        &
        ~available
    )

    # Important diagnostic:
    # finite phase exists despite neither PL nor PS declaring
    # the pixel as a valid phase source.
    undeclared_but_finite = (
        req
        &
        ~declared_phase_source
        &
        phase_all_finite
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    n_required = int(
        np.count_nonzero(req)
    )

    n_pl = int(
        np.count_nonzero(required_pl)
    )

    n_ps = int(
        np.count_nonzero(required_ps)
    )

    n_source = int(
        np.count_nonzero(required_source)
    )

    n_available = int(
        np.count_nonzero(available)
    )

    n_missing = int(
        np.count_nonzero(missing)
    )

    n_integrity = int(
        np.count_nonzero(
            integrity_failure
        )
    )

    n_undeclared_finite = int(
        np.count_nonzero(
            undeclared_but_finite
        )
    )

    n_first = int(
        np.count_nonzero(
            req
            &
            phase_first_finite
        )
    )

    n_last = int(
        np.count_nonzero(
            req
            &
            phase_last_finite
        )
    )

    # --------------------------------------------------------
    # Save audit masks
    # --------------------------------------------------------

    seqdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    missing_path = (
        seqdir
        /
        "compression_phase_missing_mask.npy"
    )

    integrity_path = (
        seqdir
        /
        "compression_phase_integrity_failure_mask.npy"
    )

    np.save(
        missing_path,
        missing,
    )

    np.save(
        integrity_path,
        integrity_failure,
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if (
        n_missing == 0
        and
        n_integrity == 0
    ):

        decision = (
            "current_phase_output_sufficient"
        )

        recommendation = (
            "Current linked-phase output already covers "
            "all compression-required pixels."
        )

    else:

        decision = (
            "dense_compression_phase_path_required"
        )

        recommendation = (
            "Current sparse center-only PL output is not "
            "sufficient for the sequential compressed-SLC "
            "state. Implement tile-local dense compression "
            "phase estimation and project the compressed SLC "
            "immediately while phase is resident in RAM. "
            "Do not materialize another dense HxWxT "
            "linked-phase cube."
        )

    report = {
        "format":
            "pyPSDS-GAMMA-sequential-phase-coverage-v1",

        "shape":
            [H, W],

        "ndate":
            int(ndate),

        "linked_phase_layout":
            layout,

        "compression_required":
            n_required,

        "required_with_pl_valid":
            n_pl,

        "required_with_ps_phase":
            n_ps,

        "required_with_declared_phase_source":
            n_source,

        "required_first_epoch_finite":
            n_first,

        "required_last_epoch_finite":
            n_last,

        "required_all_epochs_available":
            n_available,

        "required_missing":
            n_missing,

        "required_phase_integrity_failures":
            n_integrity,

        "required_undeclared_but_all_finite":
            n_undeclared_finite,

        "pl_coverage_fraction":
            (
                n_pl / n_required
                if n_required
                else 0.0
            ),

        "usable_coverage_fraction":
            (
                n_available / n_required
                if n_required
                else 0.0
            ),

        "decision":
            decision,

        "recommendation":
            recommendation,
    }

    json_path = (
        seqdir
        /
        "compression_phase_coverage.json"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print(
        "U3.2b sequential compression phase availability audit"
    )
    print("=" * 88)

    print(
        "scene                       :",
        f"{H} x {W}",
    )

    print(
        "dates                       :",
        ndate,
    )

    print(
        "linked phase layout         :",
        layout,
    )

    print(
        "compression required        :",
        f"{n_required:,}",
    )

    print(
        "required + PL valid         :",
        f"{n_pl:,}",
        f"({pct(n_pl, n_required):.3f}%)",
    )

    print(
        "required + PS               :",
        f"{n_ps:,}",
        f"({pct(n_ps, n_required):.3f}%)",
    )

    print(
        "declared phase source       :",
        f"{n_source:,}",
        f"({pct(n_source, n_required):.3f}%)",
    )

    print(
        "first epoch finite          :",
        f"{n_first:,}",
        f"({pct(n_first, n_required):.3f}%)",
    )

    print(
        "last epoch finite           :",
        f"{n_last:,}",
        f"({pct(n_last, n_required):.3f}%)",
    )

    print(
        "all epochs usable           :",
        f"{n_available:,}",
        f"({pct(n_available, n_required):.3f}%)",
    )

    print(
        "missing phase state         :",
        f"{n_missing:,}",
        f"({pct(n_missing, n_required):.3f}%)",
    )

    print(
        "phase integrity failures    :",
        f"{n_integrity:,}",
    )

    print(
        "undeclared but finite       :",
        f"{n_undeclared_finite:,}",
    )

    print()
    print(
        "missing mask :",
        missing_path,
    )

    print(
        "integrity mask:",
        integrity_path,
    )

    print(
        "json         :",
        json_path,
    )

    print()
    print(
        "decision     :",
        decision,
    )

    print()
    print(
        recommendation
    )

    print()
    print(
        "U3.2b AUDIT EXECUTION: PASS"
    )


if __name__ == "__main__":
    main()
