#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


def run_command(cmd, log_file: Path, label: str):

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_file.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write("\n" + "=" * 100 + "\n")
        f.write(label + "\n")
        f.write(" ".join(str(x) for x in cmd) + "\n")
        f.write("=" * 100 + "\n")
        f.flush()

        p = subprocess.run(
            [str(x) for x in cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if p.returncode != 0:
        raise RuntimeError(
            f"{label} failed; see {log_file}"
        )


def write_reference_itab(
    path: Path,
    ref_idx: int,
    secondary_indices,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for rec, sec_idx in enumerate(
            secondary_indices,
            start=1,
        ):

            f.write(
                f"{ref_idx+1} "
                f"{sec_idx+1} "
                f"{rec} 1\n"
            )


def make_plist(
    path: Path,
    rows,
    cols,
):

    # GAMMA IPTA order:
    # range, azimuth = col, row
    arr = np.column_stack(
        (
            cols,
            rows,
        )
    ).astype(
        ">i4",
        copy=False,
    )

    arr.tofile(path)


def run_data2pt(
    *,
    cmd,
    height_path,
    height_par,
    plist,
    reference_par,
    output,
    npoint,
    log_file,
    label,
):

    if output.exists():
        output.unlink()

    run_command(
        [
            cmd,
            str(height_path),
            str(height_par),
            str(plist),
            str(reference_par),
            str(output),
            "1",
            "2",
        ],
        log_file,
        label,
    )

    h = np.fromfile(
        output,
        dtype=">f4",
    )

    if h.size != npoint:
        raise RuntimeError(
            f"{label}: height count "
            f"{h.size} != {npoint}"
        )

    return h.astype(
        np.float32
    )


def simulate_fd(
    *,
    dh,
    h_native,
    phase_sim,
    plist,
    pslc_par,
    itab,
    reference_par,
    nrec,
    workdir,
    log_file,
    label,
):

    npoint = h_native.size

    valid = np.isfinite(
        h_native
    )

    if not np.all(valid):
        raise RuntimeError(
            f"{label}: invalid DEM height"
        )

    h = h_native.copy()

    # Keep same zero-height policy as 10a3-10a5.
    zero = (
        valid
        &
        (
            h == 0.0
        )
    )

    h[
        zero
    ] = np.float32(
        1.0e-3
    )

    hp = h.copy()
    hm = h.copy()

    hp += np.float32(dh)
    hm -= np.float32(dh)

    hp_path = (
        workdir
        /
        f"{label}_hp"
    )

    hm_path = (
        workdir
        /
        f"{label}_hm"
    )

    pmask = (
        workdir
        /
        f"{label}_pmask"
    )

    out_p = (
        workdir
        /
        f"{label}_plus"
    )

    out_m = (
        workdir
        /
        f"{label}_minus"
    )

    hp.astype(
        ">f4"
    ).tofile(
        hp_path
    )

    hm.astype(
        ">f4"
    ).tofile(
        hm_path
    )

    valid.astype(
        np.uint8
    ).tofile(
        pmask
    )

    def one(phgt, output, sign_label):

        if output.exists():
            output.unlink()

        run_command(
            [
                phase_sim,
                str(plist),
                str(pmask),
                str(pslc_par),
                "-",
                str(itab),
                "-",
                str(phgt),
                str(output),
                str(reference_par),
                "-",
                "0",
            ],
            log_file,
            (
                f"{label}:"
                f"{sign_label}"
            ),
        )

        raw = np.fromfile(
            output,
            dtype=">f4",
        )

        expected = (
            nrec
            *
            npoint
        )

        if raw.size != expected:
            raise RuntimeError(
                f"{label}: "
                f"phase output {raw.size} "
                f"!= {expected}"
            )

        return (
            raw.astype(
                np.float64
            )
            .reshape(
                nrec,
                npoint,
            )
        )

    plus = one(
        hp_path,
        out_p,
        f"+{dh:g}m",
    )

    minus = one(
        hm_path,
        out_m,
        f"-{dh:g}m",
    )

    return (
        plus
        -
        minus
    ) / (
        2.0 * dh
    )


def pair_to_acquisition(
    pair_derivative,
    *,
    ndate,
    secondary_indices,
    temporal_ref_idx,
):

    npoint = (
        pair_derivative.shape[1]
    )

    S = np.zeros(
        (
            ndate,
            npoint,
        ),
        dtype=np.float64,
    )

    for rec, sec_idx in enumerate(
        secondary_indices
    ):

        S[
            sec_idx
        ] = pair_derivative[
            rec
        ]

    # Re-reference to Step09 temporal reference.
    S -= (
        S[
            temporal_ref_idx
        ][
            None,
            :
        ]
    )

    return S.T


def compare(a, b):

    d = (
        np.asarray(
            a,
            dtype=np.float64,
        )
        -
        np.asarray(
            b,
            dtype=np.float64,
        )
    )

    ad = np.abs(d)

    n = int(
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
        np.max(
            ad
        )
    )

    counts = {}

    for t in (
        0.0,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        5e-5,
        9e-5,
    ):

        counts[
            str(t)
        ] = int(
            np.count_nonzero(
                ad > t
            )
        )

    return {
        "difference":
            d,

        "rms":
            rms,

        "max":
            mx,

        "count_gt_1e5":
            counts[
                str(1e-5)
            ],

        "fraction_gt_1e5":
            float(
                counts[
                    str(1e-5)
                ]
                /
                n
            ),

        "counts":
            counts,
    }


def print_compare(
    name,
    result,
):

    print(
        f"{name:34s}: "
        f"RMS={result['rms']:.9e}, "
        f"max={result['max']:.9e}, "
        f">1e-5={result['count_gt_1e5']:,}"
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--test-points",
        type=int,
        default=4000,
    )

    ap.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
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

    invdir = (
        root
        /
        "network_inversion_v09"
    )

    fd_dir = (
        root
        /
        "scla_v09"
        /
        "fd_stepsize_audit"
    )

    prod_dir = (
        root
        /
        "scla_v09"
        /
        "production_sensitivity"
    )

    phase_corr_dir = (
        Path(paths.output_dir)
        /
        "phase_correction_v09"
    )

    outdir = (
        prod_dir
        /
        "batch_context_audit"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gamma_log = (
        outdir
        /
        "gamma.log"
    )

    # ========================================================
    # Source products
    # ========================================================

    strict_ids = np.load(
        invdir
        /
        "strict_point_ids.npy"
    ).astype(
        np.int64,
        copy=False,
    )

    sample_ids = np.load(
        fd_dir
        /
        "sample_strict_point_ids.npy"
    ).astype(
        np.int64,
        copy=False,
    )

    sample_rows = np.load(
        fd_dir
        /
        "sample_rows.npy"
    ).astype(
        np.int32,
        copy=False,
    )

    sample_cols = np.load(
        fd_dir
        /
        "sample_cols.npy"
    ).astype(
        np.int32,
        copy=False,
    )

    stored_10a4 = np.asarray(
        np.load(
            fd_dir
            /
            "sample_sensitivity_dh20m_rad_per_m.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    production = np.load(
        prod_dir
        /
        "topographic_phase_sensitivity_rad_per_m.npy",
        mmap_mode="r",
    )

    pos = np.searchsorted(
        strict_ids,
        sample_ids,
    )

    if (
        np.any(
            pos >= strict_ids.size
        )
        or
        not np.array_equal(
            strict_ids[pos],
            sample_ids,
        )
    ):
        raise RuntimeError(
            "Sample IDs cannot be mapped "
            "to strict point domain"
        )

    stored_10a5 = np.asarray(
        production[
            pos
        ],
        dtype=np.float64,
    )

    original_difference = (
        stored_10a5
        -
        stored_10a4
    )

    original_bad_point = np.any(
        np.abs(
            original_difference
        )
        >
        1.0e-5,
        axis=1,
    )

    affected = np.flatnonzero(
        original_bad_point
    )

    # ========================================================
    # Targeted sample:
    # all previously affected points + scene-wide fill.
    # ========================================================

    n_sample_available = (
        sample_ids.size
    )

    ntest = min(
        args.test_points,
        n_sample_available,
    )

    if affected.size > ntest:
        raise RuntimeError(
            "test-points smaller than number "
            "of known affected points"
        )

    remaining = np.flatnonzero(
        ~original_bad_point
    )

    need = (
        ntest
        -
        affected.size
    )

    if need > 0:

        pick = np.linspace(
            0,
            remaining.size - 1,
            need,
            dtype=np.int64,
        )

        fill = remaining[
            pick
        ]

        selected = np.concatenate(
            (
                affected,
                fill,
            )
        )

    else:

        selected = affected.copy()

    selected = np.unique(
        selected
    )

    # linspace integer mapping should already be unique,
    # but guarantee exact requested count.
    if selected.size < ntest:

        present = np.zeros(
            n_sample_available,
            dtype=bool,
        )

        present[
            selected
        ] = True

        extras = np.flatnonzero(
            ~present
        )[
            :(
                ntest
                -
                selected.size
            )
        ]

        selected = np.concatenate(
            (
                selected,
                extras,
            )
        )

    selected = np.sort(
        selected
    )

    if selected.size != ntest:
        raise RuntimeError(
            f"selected={selected.size}, "
            f"expected={ntest}"
        )

    test_ids = sample_ids[
        selected
    ]

    test_rows = sample_rows[
        selected
    ]

    test_cols = sample_cols[
        selected
    ]

    old_a = stored_10a4[
        selected
    ]

    old_b = stored_10a5[
        selected
    ]

    # ========================================================
    # Existing GAMMA geometry preparation
    # ========================================================

    prep = json.loads(
        (
            phase_corr_dir
            /
            "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    pslc_par = (
        phase_corr_dir
        /
        "pSLC_par"
    )

    geom_ref_idx = int(
        prep[
            "geometric_reference_index"
        ]
    )

    geom_ref_date = str(
        prep[
            "geometric_reference_date"
        ]
    )

    height_path = Path(
        prep[
            "height_path"
        ]
    )

    height_par = Path(
        prep[
            "height_geometry_par"
        ]
    )

    reference_par = Path(
        stack.records[
            geom_ref_idx
        ].par
    ).resolve()

    # ========================================================
    # Step09 reference
    # ========================================================

    inv_manifest = json.loads(
        (
            invdir
            /
            "network_inversion_parity_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    temporal_ref_idx = int(
        inv_manifest[
            "reference_acquisition_index"
        ]
    )

    temporal_ref_date = str(
        inv_manifest[
            "reference_date"
        ]
    )

    ndate = len(
        stack.dates
    )

    secondary_indices = [
        i
        for i in range(
            ndate
        )
        if i != geom_ref_idx
    ]

    nrec = len(
        secondary_indices
    )

    itab = (
        outdir
        /
        "reference_to_all.itab"
    )

    write_reference_itab(
        itab,
        geom_ref_idx,
        secondary_indices,
    )

    # ========================================================
    # Commands
    # ========================================================

    data2pt = shutil.which(
        "data2pt"
    )

    phase_sim = shutil.which(
        "phase_sim_orb_pt"
    )

    if not data2pt:
        raise RuntimeError(
            "data2pt not found"
        )

    if not phase_sim:
        raise RuntimeError(
            "phase_sim_orb_pt not found"
        )

    dh = 20.0

    print("=" * 112)
    print(
        "Step 10a5c - GAMMA batch-context "
        "repeatability audit"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"available 10a4 sample      : "
        f"{sample_ids.size:,}"
    )

    print(
        f"known affected sample pts  : "
        f"{affected.size:,}"
    )

    print(
        f"test points                : "
        f"{ntest:,}"
    )

    print(
        f"chunk size                 : "
        f"{args.chunk_size:,}"
    )

    print(
        f"geometric reference        : "
        f"{geom_ref_date}"
    )

    print(
        f"temporal reference         : "
        f"{temporal_ref_date}"
    )

    print(
        f"finite-difference step     : "
        f"±{dh:g} m"
    )

    # ========================================================
    # Full-call calculation
    # ========================================================

    with tempfile.TemporaryDirectory(
        prefix="full_",
        dir=outdir,
    ) as td:

        wd = Path(td)

        plist = (
            wd
            /
            "plist"
        )

        make_plist(
            plist,
            test_rows,
            test_cols,
        )

        h_full = run_data2pt(
            cmd=data2pt,
            height_path=height_path,
            height_par=height_par,
            plist=plist,
            reference_par=reference_par,
            output=(
                wd
                /
                "phgt"
            ),
            npoint=ntest,
            log_file=gamma_log,
            label="data2pt:full",
        )

        pair_full = simulate_fd(
            dh=dh,
            h_native=h_full,
            phase_sim=phase_sim,
            plist=plist,
            pslc_par=pslc_par,
            itab=itab,
            reference_par=reference_par,
            nrec=nrec,
            workdir=wd,
            log_file=gamma_log,
            label="full",
        )

        fresh_full = pair_to_acquisition(
            pair_full,
            ndate=ndate,
            secondary_indices=
                secondary_indices,
            temporal_ref_idx=
                temporal_ref_idx,
        )

    # ========================================================
    # Chunked calculation
    # ========================================================

    fresh_chunk = np.empty(
        (
            ntest,
            ndate,
        ),
        dtype=np.float64,
    )

    h_chunk_joined = np.empty(
        ntest,
        dtype=np.float32,
    )

    for b0 in range(
        0,
        ntest,
        args.chunk_size,
    ):

        b1 = min(
            b0
            +
            args.chunk_size,
            ntest,
        )

        B = b1 - b0

        with tempfile.TemporaryDirectory(
            prefix=f"chunk_{b0}_",
            dir=outdir,
        ) as td:

            wd = Path(td)

            plist = (
                wd
                /
                "plist"
            )

            make_plist(
                plist,
                test_rows[
                    b0:b1
                ],
                test_cols[
                    b0:b1
                ],
            )

            h_chunk = run_data2pt(
                cmd=data2pt,
                height_path=height_path,
                height_par=height_par,
                plist=plist,
                reference_par=reference_par,
                output=(
                    wd
                    /
                    "phgt"
                ),
                npoint=B,
                log_file=gamma_log,
                label=(
                    f"data2pt:chunk:"
                    f"{b0}-{b1}"
                ),
            )

            h_chunk_joined[
                b0:b1
            ] = h_chunk

            pair = simulate_fd(
                dh=dh,
                h_native=h_chunk,
                phase_sim=phase_sim,
                plist=plist,
                pslc_par=pslc_par,
                itab=itab,
                reference_par=reference_par,
                nrec=nrec,
                workdir=wd,
                log_file=gamma_log,
                label=(
                    f"chunk_{b0}_{b1}"
                ),
            )

            fresh_chunk[
                b0:b1
            ] = pair_to_acquisition(
                pair,
                ndate=ndate,
                secondary_indices=
                    secondary_indices,
                temporal_ref_idx=
                    temporal_ref_idx,
            )

        print(
            f"  chunk {b0:,}:{b1:,}"
        )

    # ========================================================
    # data2pt context test
    # ========================================================

    hdiff = (
        h_chunk_joined.astype(
            np.float64
        )
        -
        h_full.astype(
            np.float64
        )
    )

    h_abs = np.abs(
        hdiff
    )

    h_rms = float(
        np.sqrt(
            np.mean(
                hdiff * hdiff
            )
        )
    )

    h_max = float(
        np.max(
            h_abs
        )
    )

    h_nonzero = int(
        np.count_nonzero(
            hdiff != 0
        )
    )

    print()
    print("=" * 112)
    print(
        "data2pt batch-context reproducibility"
    )
    print("=" * 112)

    print(
        f"non-identical heights      : "
        f"{h_nonzero:,}/"
        f"{ntest:,}"
    )

    print(
        f"height RMS difference      : "
        f"{h_rms:.9e} m"
    )

    print(
        f"height max difference      : "
        f"{h_max:.9e} m"
    )

    # ========================================================
    # Main comparisons
    # ========================================================

    r_full_chunk = compare(
        fresh_full,
        fresh_chunk,
    )

    r_full_old_a = compare(
        fresh_full,
        old_a,
    )

    r_full_old_b = compare(
        fresh_full,
        old_b,
    )

    r_chunk_old_a = compare(
        fresh_chunk,
        old_a,
    )

    r_chunk_old_b = compare(
        fresh_chunk,
        old_b,
    )

    r_old_a_b = compare(
        old_a,
        old_b,
    )

    print()
    print("=" * 112)
    print(
        "Sensitivity repeatability"
    )
    print("=" * 112)

    print_compare(
        "fresh full vs fresh chunks",
        r_full_chunk,
    )

    print_compare(
        "fresh full vs stored 10a4",
        r_full_old_a,
    )

    print_compare(
        "fresh full vs production 10a5",
        r_full_old_b,
    )

    print_compare(
        "fresh chunks vs stored 10a4",
        r_chunk_old_a,
    )

    print_compare(
        "fresh chunks vs production 10a5",
        r_chunk_old_b,
    )

    print_compare(
        "stored 10a4 vs production",
        r_old_a_b,
    )

    # ========================================================
    # Original mismatch vs fresh mismatch overlap
    # ========================================================

    old_bad = (
        np.abs(
            old_b
            -
            old_a
        )
        >
        1.0e-5
    )

    fresh_bad = (
        np.abs(
            fresh_chunk
            -
            fresh_full
        )
        >
        1.0e-5
    )

    old_bad_n = int(
        np.count_nonzero(
            old_bad
        )
    )

    fresh_bad_n = int(
        np.count_nonzero(
            fresh_bad
        )
    )

    overlap_n = int(
        np.count_nonzero(
            old_bad
            &
            fresh_bad
        )
    )

    union_n = int(
        np.count_nonzero(
            old_bad
            |
            fresh_bad
        )
    )

    jaccard = (
        float(
            overlap_n
            /
            union_n
        )
        if union_n
        else 1.0
    )

    print()
    print("=" * 112)
    print(
        "Mismatch-location overlap"
    )
    print("=" * 112)

    print(
        f"stored mismatch entries    : "
        f"{old_bad_n:,}"
    )

    print(
        f"fresh batch mismatch       : "
        f"{fresh_bad_n:,}"
    )

    print(
        f"overlap entries            : "
        f"{overlap_n:,}"
    )

    print(
        f"union entries              : "
        f"{union_n:,}"
    )

    print(
        f"Jaccard overlap            : "
        f"{jaccard:.6f}"
    )

    # ========================================================
    # Fresh difference levels
    # ========================================================

    d = r_full_chunk[
        "difference"
    ]

    nz = d[
        np.abs(
            d
        )
        >
        1.0e-8
    ]

    levels = Counter(
        np.round(
            nz,
            decimals=10,
        ).tolist()
    )

    print()
    print("=" * 112)
    print(
        "Fresh full-vs-chunk nonzero difference levels"
    )
    print("=" * 112)

    if not levels:

        print(
            "No differences > 1e-8."
        )

    else:

        for value, count in levels.most_common(
            30
        ):

            print(
                f"{value:+.10e} "
                f"count={count}"
            )

    # ========================================================
    # Known affected point subset
    # ========================================================

    affected_selected = np.isin(
        selected,
        affected,
    )

    if np.any(
        affected_selected
    ):

        affected_full = fresh_full[
            affected_selected
        ]

        affected_chunk = fresh_chunk[
            affected_selected
        ]

        affected_old_a = old_a[
            affected_selected
        ]

        affected_old_b = old_b[
            affected_selected
        ]

        rr = compare(
            affected_full,
            affected_chunk,
        )

        rr_a = compare(
            affected_full,
            affected_old_a,
        )

        rr_b = compare(
            affected_full,
            affected_old_b,
        )

        print()
        print("=" * 112)
        print(
            "Previously affected point subset"
        )
        print("=" * 112)

        print(
            f"points                     : "
            f"{np.count_nonzero(affected_selected):,}"
        )

        print_compare(
            "fresh full vs fresh chunks",
            rr,
        )

        print_compare(
            "fresh full vs stored 10a4",
            rr_a,
        )

        print_compare(
            "fresh full vs production 10a5",
            rr_b,
        )

    # ========================================================
    # Save small QA arrays
    # ========================================================

    np.save(
        outdir
        /
        "test_sample_indices.npy",
        selected.astype(
            np.int32
        ),
    )

    np.save(
        outdir
        /
        "test_point_ids.npy",
        test_ids.astype(
            np.int32
        ),
    )

    np.save(
        outdir
        /
        "fresh_full_sensitivity.npy",
        fresh_full.astype(
            np.float32
        ),
    )

    np.save(
        outdir
        /
        "fresh_chunk_sensitivity.npy",
        fresh_chunk.astype(
            np.float32
        ),
    )

    # ========================================================
    # Decision
    # ========================================================

    numerical_small = (
        r_full_chunk[
            "rms"
        ]
        <=
        2.0e-6
        and
        r_full_chunk[
            "max"
        ]
        <=
        1.1e-4
    )

    if (
        h_max == 0.0
        and
        r_full_chunk[
            "count_gt_1e5"
        ]
        >
        0
        and
        numerical_small
    ):

        status = (
            "PASS_PHASE_SIM_BATCH_CONTEXT_CONFIRMED"
        )

    elif (
        h_max > 0.0
        and
        h_max <= 1.0e-3
        and
        numerical_small
    ):

        status = (
            "PASS_NUMERICAL_BATCH_CONTEXT_CONFIRMED"
        )

    elif (
        h_max == 0.0
        and
        r_full_chunk[
            "max"
        ]
        <=
        1.0e-8
    ):

        status = (
            "PASS_EXACT_REPEAT_FOR_TESTED_CONTEXT"
        )

    else:

        status = (
            "REVIEW_BATCH_CONTEXT"
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-batch-context-repeatability-v09",

        "status":
            status,

        "test_points":
            int(
                ntest
            ),

        "known_affected_points_included":
            int(
                np.count_nonzero(
                    affected_selected
                )
            ),

        "chunk_size":
            int(
                args.chunk_size
            ),

        "delta_height_m":
            20.0,

        "data2pt": {
            "non_identical_heights":
                h_nonzero,

            "rms_difference_m":
                h_rms,

            "max_difference_m":
                h_max,
        },

        "fresh_full_vs_chunks": {
            "rms_rad_per_m":
                r_full_chunk[
                    "rms"
                ],

            "max_rad_per_m":
                r_full_chunk[
                    "max"
                ],

            "count_gt_1e5":
                r_full_chunk[
                    "count_gt_1e5"
                ],

            "fraction_gt_1e5":
                r_full_chunk[
                    "fraction_gt_1e5"
                ],
        },

        "old_vs_fresh_overlap": {
            "stored_bad_entries":
                old_bad_n,

            "fresh_bad_entries":
                fresh_bad_n,

            "overlap_entries":
                overlap_n,

            "union_entries":
                union_n,

            "jaccard":
                jaccard,
        },

        "phase_modified":
            False,

        "production_sensitivity_modified":
            False,
    }

    manifest_path = (
        outdir
        /
        "batch_context_repeatability_manifest.json"
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
        f"STEP 10a5c STATUS: "
        f"{status}"
    )

    print(
        "No production sensitivity, phase, "
        "or residual DEM correction was modified."
    )


if __name__ == "__main__":
    main()
