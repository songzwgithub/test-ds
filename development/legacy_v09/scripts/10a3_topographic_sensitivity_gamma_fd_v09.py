#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


# ============================================================
# Utilities
# ============================================================

def run_command(cmd, *, log_file: Path, label: str):

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_file.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write(f"{label}\n")
        f.write(" ".join(map(str, cmd)) + "\n")
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
            f"{label} failed with "
            f"return code {p.returncode}. "
            f"See {log_file}"
        )


def load_network_itab(
    path: Path,
    ndate: int,
):

    edges = []

    with path.open() as f:

        for raw in f:

            x = raw.split()

            if len(x) < 2:
                continue

            i = int(x[0]) - 1
            j = int(x[1]) - 1

            if not (
                0 <= i < ndate
                and
                0 <= j < ndate
            ):
                raise RuntimeError(
                    f"Invalid network line: {raw}"
                )

            edges.append(
                (i, j)
            )

    return edges


def write_gamma_itab(
    path: Path,
    edges,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for rec, (
            i,
            j,
        ) in enumerate(
            edges,
            start=1,
        ):

            f.write(
                f"{i+1} "
                f"{j+1} "
                f"{rec} "
                f"1\n"
            )


def simulate_phase(
    *,
    phase_sim_cmd,
    plist: Path,
    pmask: Path,
    pslc_par: Path,
    itab: Path,
    phgt: Path,
    output: Path,
    reference_par: Path,
    nrecords: int,
    npoints: int,
    gamma_log: Path,
    label: str,
):

    if output.exists():
        output.unlink()

    run_command(
        [
            phase_sim_cmd,
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
        log_file=gamma_log,
        label=label,
    )

    raw = np.fromfile(
        output,
        dtype=">f4",
    )

    expected = (
        nrecords
        *
        npoints
    )

    if raw.size != expected:

        raise RuntimeError(
            f"{label}: "
            f"output size={raw.size}, "
            f"expected={expected}"
        )

    return (
        raw.astype(
            np.float64
        )
        .reshape(
            nrecords,
            npoints,
        )
    )


def simulate_fd(
    *,
    dh: float,
    native_height,
    valid,
    workdir: Path,
    phase_sim_cmd,
    plist: Path,
    pmask: Path,
    pslc_par: Path,
    itab: Path,
    reference_par: Path,
    nrecords: int,
    npoints: int,
    gamma_log: Path,
    label_prefix: str,
):

    hp = np.asarray(
        native_height,
        dtype=np.float32,
    ).copy()

    hm = hp.copy()

    hp[
        valid
    ] += np.float32(
        dh
    )

    hm[
        valid
    ] -= np.float32(
        dh
    )

    phgt_plus = (
        workdir
        /
        f"phgt_plus_{dh:g}"
    )

    phgt_minus = (
        workdir
        /
        f"phgt_minus_{dh:g}"
    )

    hp.astype(
        ">f4"
    ).tofile(
        phgt_plus
    )

    hm.astype(
        ">f4"
    ).tofile(
        phgt_minus
    )

    out_plus = (
        workdir
        /
        f"sim_plus_{label_prefix}_{dh:g}"
    )

    out_minus = (
        workdir
        /
        f"sim_minus_{label_prefix}_{dh:g}"
    )

    plus = simulate_phase(
        phase_sim_cmd=phase_sim_cmd,
        plist=plist,
        pmask=pmask,
        pslc_par=pslc_par,
        itab=itab,
        phgt=phgt_plus,
        output=out_plus,
        reference_par=reference_par,
        nrecords=nrecords,
        npoints=npoints,
        gamma_log=gamma_log,
        label=(
            f"{label_prefix}:"
            f"+{dh:g}m"
        ),
    )

    minus = simulate_phase(
        phase_sim_cmd=phase_sim_cmd,
        plist=plist,
        pmask=pmask,
        pslc_par=pslc_par,
        itab=itab,
        phgt=phgt_minus,
        output=out_minus,
        reference_par=reference_par,
        nrecords=nrecords,
        npoints=npoints,
        gamma_log=gamma_log,
        label=(
            f"{label_prefix}:"
            f"-{dh:g}m"
        ),
    )

    derivative = (
        plus
        -
        minus
    ) / (
        2.0
        *
        dh
    )

    return derivative


# ============================================================
# Main
# ============================================================

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

    output_base = Path(
        paths.output_dir
    )

    root = (
        output_base
        /
        "v09"
    )

    pps_dir = (
        root
        /
        "point_phase_stack"
    )

    inversion_dir = (
        root
        /
        "network_inversion_v09"
    )

    network_dir = (
        root
        /
        "network"
    )

    phase_corr_dir = (
        output_base
        /
        "phase_correction_v09"
    )

    outdir = (
        root
        /
        "scla_v09"
        /
        "topographic_sensitivity"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scratch_root = (
        outdir
        /
        "scratch"
    )

    scratch_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    gamma_log = (
        outdir
        /
        "gamma.log"
    )

    # ========================================================
    # Existing GAMMA phase-correction preparation
    # ========================================================

    prep_manifest_path = (
        phase_corr_dir
        /
        "manifest.json"
    )

    pslc_par = (
        phase_corr_dir
        /
        "pSLC_par"
    )

    if not prep_manifest_path.is_file():

        raise FileNotFoundError(
            prep_manifest_path
        )

    if not pslc_par.is_file():

        raise FileNotFoundError(
            pslc_par
        )

    prep = json.loads(
        prep_manifest_path.read_text(
            encoding="utf-8",
        )
    )

    expected_dates = [
        str(x)
        for x in stack.dates
    ]

    if list(
        prep.get(
            "dates",
            []
        )
    ) != expected_dates:

        raise RuntimeError(
            "phase_correction_v09 manifest "
            "date list does not match current stack"
        )

    geom_ref_date = str(
        prep[
            "geometric_reference_date"
        ]
    )

    geom_ref_idx = int(
        prep[
            "geometric_reference_index"
        ]
    )

    if (
        expected_dates[
            geom_ref_idx
        ]
        !=
        geom_ref_date
    ):

        raise RuntimeError(
            "Geometric reference index/date mismatch"
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

    for p in (
        height_path,
        height_par,
        reference_par,
    ):

        if not p.is_file():
            raise FileNotFoundError(
                p
            )

    # ========================================================
    # GAMMA commands
    # ========================================================

    data2pt_cmd = shutil.which(
        "data2pt"
    )

    phase_sim_cmd = shutil.which(
        "phase_sim_orb_pt"
    )

    if not data2pt_cmd:
        raise RuntimeError(
            "data2pt not found"
        )

    if not phase_sim_cmd:
        raise RuntimeError(
            "phase_sim_orb_pt not found"
        )

    # ========================================================
    # Strict point coordinates
    # ========================================================

    strict_ids = np.load(
        inversion_dir
        /
        "strict_point_ids.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    all_rows = np.asarray(
        np.load(
            pps_dir
            /
            "rows.npy",
            mmap_mode="r",
        ),
        dtype=np.int32,
    )

    all_cols = np.asarray(
        np.load(
            pps_dir
            /
            "cols.npy",
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

    npoint = int(
        strict_ids.size
    )

    ndate = len(
        stack.dates
    )

    # ========================================================
    # Temporal reference from Step09a
    # ========================================================

    inv_manifest_path = (
        inversion_dir
        /
        "network_inversion_parity_manifest.json"
    )

    inv_manifest = json.loads(
        inv_manifest_path.read_text(
            encoding="utf-8",
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

    if (
        expected_dates[
            temporal_ref_idx
        ]
        !=
        temporal_ref_date
    ):

        raise RuntimeError(
            "Temporal reference date/index mismatch"
        )

    # ========================================================
    # Reference-to-all GAMMA itab
    #
    # Exactly reproduce the existing phase_correction.py
    # ordering:
    # all dates except geometric reference, in stack order.
    # ========================================================

    secondary_indices = [
        i
        for i in range(
            ndate
        )
        if i != geom_ref_idx
    ]

    reference_edges = [
        (
            geom_ref_idx,
            i,
        )
        for i in secondary_indices
    ]

    reference_itab = (
        outdir
        /
        "reference_to_all_gamma.itab"
    )

    write_gamma_itab(
        reference_itab,
        reference_edges,
    )

    nref_records = len(
        reference_edges
    )

    if nref_records != ndate - 1:

        raise RuntimeError(
            "reference-to-all record count mismatch"
        )

    # ========================================================
    # Production network GAMMA itab
    # ========================================================

    production_edges = load_network_itab(
        network_dir
        /
        "network.itab",
        ndate,
    )

    production_itab = (
        outdir
        /
        "production_network_gamma.itab"
    )

    write_gamma_itab(
        production_itab,
        production_edges,
    )

    nifg = len(
        production_edges
    )

    # ========================================================
    # Output main acquisition sensitivity
    #
    # Definition:
    #
    #   d_phase_sim_dh
    #
    # from phase_sim_orb_pt itself.
    #
    # This is deliberately NOT yet assigned the physical sign
    # of residual DEM error in the corrected phase.
    # ========================================================

    sensitivity_path = (
        outdir
        /
        "d_phase_sim_dh_acquisition_rad_per_m.npy"
    )

    sensitivity = np.lib.format.open_memmap(
        sensitivity_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            npoint,
            ndate,
        ),
    )

    network_rms_path = (
        outdir
        /
        "network_parity_rms_by_point_rad_per_m.npy"
    )

    network_rms_by_point = np.lib.format.open_memmap(
        network_rms_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            npoint,
        ),
    )

    linearity_path = (
        outdir
        /
        "linearity_max_abs_difference_by_point_rad_per_m.npy"
    )

    linearity_by_point = np.lib.format.open_memmap(
        linearity_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            npoint,
        ),
    )

    # ========================================================
    # Finite-difference setup
    # ========================================================

    main_dh = 10.0

    test_dh = [
        1.0,
        5.0,
        10.0,
        20.0,
    ]

    # Put main first so it is always immediately available.
    run_dh = [
        10.0,
        1.0,
        5.0,
        20.0,
    ]

    # Linearity accumulators.
    lin_ss = {
        d: 0.0
        for d in test_dh
        if d != main_dh
    }

    lin_n = {
        d: 0
        for d in test_dh
        if d != main_dh
    }

    lin_max = {
        d: 0.0
        for d in test_dh
        if d != main_dh
    }

    # Main sensitivity scale.
    sensitivity_ss = 0.0
    sensitivity_n = 0

    sensitivity_min = np.inf
    sensitivity_max = -np.inf

    # Network parity accumulators.
    network_ss = 0.0
    network_n = 0
    network_max = 0.0

    # Height QA.
    total_valid_height = 0
    total_invalid_height = 0

    temporal_ref_max = 0.0
    geom_ref_raw_max = 0.0

    # ========================================================
    # Header
    # ========================================================

    print("=" * 112)
    print(
        "Step 10a3 - GAMMA point-wise topographic "
        "phase-sensitivity finite-difference audit"
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
        f"production IFGs            : "
        f"{nifg}"
    )

    print(
        f"geometric reference        : "
        f"{geom_ref_date} "
        f"(index {geom_ref_idx})"
    )

    print(
        f"temporal reference         : "
        f"{temporal_ref_date} "
        f"(index {temporal_ref_idx})"
    )

    print(
        f"finite-difference heights  : "
        f"{test_dh} m"
    )

    print(
        f"main sensitivity delta-h   : "
        f"{main_dh:g} m"
    )

    print(
        f"batch size                 : "
        f"{args.batch_size:,}"
    )

    print(
        f"pSLC_par                   : "
        f"{pslc_par}"
    )

    print(
        f"height raster              : "
        f"{height_path}"
    )

    print()

    # ========================================================
    # Stream batches
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

        B = (
            b1 - b0
        )

        batch_rows = rows[
            b0:b1
        ]

        batch_cols = cols[
            b0:b1
        ]

        with tempfile.TemporaryDirectory(
            prefix=(
                f"b{b0:07d}_"
            ),
            dir=scratch_root,
        ) as td:

            workdir = Path(td)

            # ------------------------------------------------
            # IPTA plist:
            #
            # range first, azimuth second.
            # GAMMA big-endian int32.
            # ------------------------------------------------

            plist_arr = np.column_stack(
                (
                    batch_cols,
                    batch_rows,
                )
            ).astype(
                ">i4",
                copy=False,
            )

            plist = (
                workdir
                /
                "plist"
            )

            plist_arr.tofile(
                plist
            )

            # ------------------------------------------------
            # Native DEM height
            # ------------------------------------------------

            phgt_native = (
                workdir
                /
                "phgt_native"
            )

            run_command(
                [
                    data2pt_cmd,
                    str(
                        height_path
                    ),
                    str(
                        height_par
                    ),
                    str(
                        plist
                    ),
                    str(
                        reference_par
                    ),
                    str(
                        phgt_native
                    ),
                    "1",
                    "2",
                ],
                log_file=gamma_log,
                label=(
                    f"data2pt:"
                    f"{b0}-{b1}"
                ),
            )

            h = np.fromfile(
                phgt_native,
                dtype=">f4",
            )

            if h.size != B:

                raise RuntimeError(
                    f"data2pt batch "
                    f"{b0}:{b1}: "
                    f"{h.size} heights, "
                    f"expected {B}"
                )

            h_native = h.astype(
                np.float32
            )

            valid = np.isfinite(
                h_native
            )

            # Strict points came through the original geometry
            # pipeline already. Zero height can be physically
            # valid; avoid GAMMA zero-as-no-data ambiguity with
            # a tiny numerical epsilon.
            zero_valid = (
                valid
                &
                (
                    h_native
                    ==
                    0.0
                )
            )

            h_native[
                zero_valid
            ] = np.float32(
                1.0e-3
            )

            pmask = (
                workdir
                /
                "pmask"
            )

            valid.astype(
                np.uint8
            ).tofile(
                pmask
            )

            nv = int(
                np.count_nonzero(
                    valid
                )
            )

            total_valid_height += nv
            total_invalid_height += (
                B - nv
            )

            # ------------------------------------------------
            # Reference-to-all finite differences.
            # ------------------------------------------------

            derivative_by_dh = {}

            for dh in run_dh:

                pair_derivative = simulate_fd(
                    dh=dh,
                    native_height=h_native,
                    valid=valid,
                    workdir=workdir,
                    phase_sim_cmd=phase_sim_cmd,
                    plist=plist,
                    pmask=pmask,
                    pslc_par=pslc_par,
                    itab=reference_itab,
                    reference_par=reference_par,
                    nrecords=nref_records,
                    npoints=B,
                    gamma_log=gamma_log,
                    label_prefix=(
                        f"refall_b"
                        f"{b0}_{b1}"
                    ),
                )

                # --------------------------------------------
                # Convert the 37 geometric-reference IFGs into
                # a 38-acquisition sensitivity array.
                #
                # Geometric reference initially = exactly zero.
                # --------------------------------------------

                S = np.zeros(
                    (
                        ndate,
                        B,
                    ),
                    dtype=np.float64,
                )

                for rec, sec_idx in enumerate(
                    secondary_indices
                ):

                    S[
                        sec_idx,
                        :
                    ] = (
                        pair_derivative[
                            rec,
                            :
                        ]
                    )

                if np.any(
                    valid
                ):

                    geom_ref_raw_max = max(
                        geom_ref_raw_max,
                        float(
                            np.max(
                                np.abs(
                                    S[
                                        geom_ref_idx,
                                        valid
                                    ]
                                )
                            )
                        ),
                    )

                # --------------------------------------------
                # Re-reference geometry sensitivity from the
                # GAMMA geometric reference acquisition to the
                # same temporal reference used by Step09:
                #
                #     20141006 = 0
                #
                # Pair differences are unaffected.
                # --------------------------------------------

                S -= (
                    S[
                        temporal_ref_idx,
                        :
                    ][
                        None,
                        :
                    ]
                )

                if np.any(
                    valid
                ):

                    temporal_ref_max = max(
                        temporal_ref_max,
                        float(
                            np.max(
                                np.abs(
                                    S[
                                        temporal_ref_idx,
                                        valid
                                    ]
                                )
                            )
                        ),
                    )

                # Mask invalid height points.
                S[
                    :,
                    ~valid
                ] = np.nan

                derivative_by_dh[
                    dh
                ] = S

            # ------------------------------------------------
            # Main Δh = 10 m product
            # ------------------------------------------------

            S_main = derivative_by_dh[
                main_dh
            ]

            sensitivity[
                b0:b1,
                :
            ] = (
                S_main.T
            ).astype(
                np.float32
            )

            finite_main = np.isfinite(
                S_main
            )

            if np.any(
                finite_main
            ):

                vals = S_main[
                    finite_main
                ]

                sensitivity_ss += float(
                    np.sum(
                        vals
                        *
                        vals
                    )
                )

                sensitivity_n += int(
                    vals.size
                )

                sensitivity_min = min(
                    sensitivity_min,
                    float(
                        np.min(
                            vals
                        )
                    ),
                )

                sensitivity_max = max(
                    sensitivity_max,
                    float(
                        np.max(
                            vals
                        )
                    ),
                )

            # ------------------------------------------------
            # Cross-delta linearity
            # ------------------------------------------------

            point_lin_max = np.zeros(
                B,
                dtype=np.float64,
            )

            point_lin_max[
                ~valid
            ] = np.nan

            for dh in (
                1.0,
                5.0,
                20.0,
            ):

                diff = (
                    derivative_by_dh[
                        dh
                    ]
                    -
                    S_main
                )

                finite = np.isfinite(
                    diff
                )

                if np.any(
                    finite
                ):

                    dvals = diff[
                        finite
                    ]

                    lin_ss[
                        dh
                    ] += float(
                        np.sum(
                            dvals
                            *
                            dvals
                        )
                    )

                    lin_n[
                        dh
                    ] += int(
                        dvals.size
                    )

                    lin_max[
                        dh
                    ] = max(
                        lin_max[
                            dh
                        ],
                        float(
                            np.max(
                                np.abs(
                                    dvals
                                )
                            )
                        ),
                    )

                if np.any(
                    valid
                ):

                    pm = np.nanmax(
                        np.abs(
                            diff
                        ),
                        axis=0,
                    )

                    point_lin_max[
                        valid
                    ] = np.maximum(
                        point_lin_max[
                            valid
                        ],
                        pm[
                            valid
                        ],
                    )

            linearity_by_point[
                b0:b1
            ] = (
                point_lin_max
            ).astype(
                np.float32
            )

            # ------------------------------------------------
            # Independent network parity:
            #
            # Directly simulate the 108 production IFGs using
            # ±10 m.
            #
            # Compare:
            #
            # S_direct(i,j)
            #
            # against
            #
            # S_acq(j) - S_acq(i)
            #
            # This is independent of the reference-to-all itab.
            # ------------------------------------------------

            pair_direct = simulate_fd(
                dh=main_dh,
                native_height=h_native,
                valid=valid,
                workdir=workdir,
                phase_sim_cmd=phase_sim_cmd,
                plist=plist,
                pmask=pmask,
                pslc_par=pslc_par,
                itab=production_itab,
                reference_par=reference_par,
                nrecords=nifg,
                npoints=B,
                gamma_log=gamma_log,
                label_prefix=(
                    f"prod108_b"
                    f"{b0}_{b1}"
                ),
            )

            pair_pred = np.empty(
                (
                    nifg,
                    B,
                ),
                dtype=np.float64,
            )

            for e, (
                i,
                j,
            ) in enumerate(
                production_edges
            ):

                pair_pred[
                    e,
                    :
                ] = (
                    S_main[
                        j,
                        :
                    ]
                    -
                    S_main[
                        i,
                        :
                    ]
                )

            net_res = (
                pair_direct
                -
                pair_pred
            )

            net_res[
                :,
                ~valid
            ] = np.nan

            finite_net = np.isfinite(
                net_res
            )

            if np.any(
                finite_net
            ):

                rvals = net_res[
                    finite_net
                ]

                network_ss += float(
                    np.sum(
                        rvals
                        *
                        rvals
                    )
                )

                network_n += int(
                    rvals.size
                )

                network_max = max(
                    network_max,
                    float(
                        np.max(
                            np.abs(
                                rvals
                            )
                        )
                    ),
                )

            point_net_rms = np.full(
                B,
                np.nan,
                dtype=np.float64,
            )

            if np.any(
                valid
            ):

                point_net_rms[
                    valid
                ] = np.sqrt(
                    np.mean(
                        net_res[
                            :,
                            valid
                        ]
                        *
                        net_res[
                            :,
                            valid
                        ],
                        axis=0,
                    )
                )

            network_rms_by_point[
                b0:b1
            ] = (
                point_net_rms
            ).astype(
                np.float32
            )

        # End temp directory.

        print(
            f"  {b1:,}/"
            f"{npoint:,} "
            f"height_valid="
            f"{total_valid_height:,}"
        )

    # ========================================================
    # Flush
    # ========================================================

    sensitivity.flush()
    network_rms_by_point.flush()
    linearity_by_point.flush()

    # ========================================================
    # Summary statistics
    # ========================================================

    if sensitivity_n == 0:
        raise RuntimeError(
            "No valid sensitivity values"
        )

    sensitivity_rms = float(
        np.sqrt(
            sensitivity_ss
            /
            sensitivity_n
        )
    )

    network_rms = (
        float(
            np.sqrt(
                network_ss
                /
                network_n
            )
        )
        if network_n
        else np.nan
    )

    network_relative_rms = (
        network_rms
        /
        sensitivity_rms
        if sensitivity_rms > 0
        else np.nan
    )

    linearity = {}

    for dh in (
        1.0,
        5.0,
        20.0,
    ):

        rms = float(
            np.sqrt(
                lin_ss[
                    dh
                ]
                /
                lin_n[
                    dh
                ]
            )
        )

        rel = (
            rms
            /
            sensitivity_rms
            if sensitivity_rms > 0
            else np.nan
        )

        linearity[
            dh
        ] = {
            "rms":
                rms,

            "max":
                float(
                    lin_max[
                        dh
                    ]
                ),

            "relative_rms":
                float(
                    rel
                ),
        }

    # Exact distributions from point-level QA arrays.
    lin_point = np.asarray(
        linearity_by_point,
        dtype=np.float64,
    )

    net_point = np.asarray(
        network_rms_by_point,
        dtype=np.float64,
    )

    lin_valid = lin_point[
        np.isfinite(
            lin_point
        )
    ]

    net_valid = net_point[
        np.isfinite(
            net_point
        )
    ]

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Topographic sensitivity"
    )
    print("=" * 112)

    print(
        f"height valid points        : "
        f"{total_valid_height:,}/"
        f"{npoint:,}"
    )

    print(
        f"height invalid points      : "
        f"{total_invalid_height:,}"
    )

    print(
        f"sensitivity min/max        : "
        f"{sensitivity_min:.6e} / "
        f"{sensitivity_max:.6e} rad/m"
    )

    print(
        f"sensitivity RMS            : "
        f"{sensitivity_rms:.6e} rad/m"
    )

    print(
        f"geometric-ref raw max      : "
        f"{geom_ref_raw_max:.3e} rad/m"
    )

    print(
        f"temporal-ref max after ref : "
        f"{temporal_ref_max:.3e} rad/m"
    )

    print()
    print("=" * 112)
    print(
        "Finite-difference linearity"
    )
    print("=" * 112)

    print(
        " comparison       RMS diff       "
        "max diff       RMS/main"
    )

    for dh in (
        1.0,
        5.0,
        20.0,
    ):

        q = linearity[
            dh
        ]

        print(
            f" {dh:4.0f}m vs 10m   "
            f"{q['rms']:.6e}   "
            f"{q['max']:.6e}   "
            f"{q['relative_rms']:.6e}"
        )

    if lin_valid.size:

        q = np.percentile(
            lin_valid,
            [
                50,
                90,
                95,
                99,
                100,
            ],
        )

        print()
        print(
            "point max |cross-delta difference| "
            "p50/p90/p95/p99/max:"
        )

        print(
            "  "
            +
            " / ".join(
                f"{x:.6e}"
                for x in q
            )
            +
            " rad/m"
        )

    print()
    print("=" * 112)
    print(
        "Production-network sensitivity parity"
    )
    print("=" * 112)

    print(
        f"direct-vs-acquisition RMS  : "
        f"{network_rms:.6e} rad/m"
    )

    print(
        f"direct-vs-acquisition max  : "
        f"{network_max:.6e} rad/m"
    )

    print(
        f"relative RMS               : "
        f"{network_relative_rms:.6e}"
    )

    if net_valid.size:

        q = np.percentile(
            net_valid,
            [
                50,
                90,
                95,
                99,
                100,
            ],
        )

        print(
            "point network RMS "
            "p50/p90/p95/p99/max:"
        )

        print(
            "  "
            +
            " / ".join(
                f"{x:.6e}"
                for x in q
            )
            +
            " rad/m"
        )

    # ========================================================
    # Conservative status
    # ========================================================

    if total_invalid_height > 0:

        status = (
            "REVIEW_INVALID_HEIGHT"
        )

    elif (
        not np.isfinite(
            network_relative_rms
        )
        or
        network_relative_rms
        >
        1.0e-3
        or
        network_max
        >
        1.0e-3
    ):

        status = (
            "REVIEW_NETWORK_PARITY"
        )

    elif (
        linearity[
            5.0
        ][
            "relative_rms"
        ]
        >
        1.0e-3
        or
        linearity[
            20.0
        ][
            "relative_rms"
        ]
        >
        1.0e-3
    ):

        status = (
            "REVIEW_NONLINEAR_GEOMETRY"
        )

    else:

        status = (
            "PASS"
        )

    # ========================================================
    # Save metadata
    # ========================================================

    np.save(
        outdir
        /
        "strict_point_ids.npy",
        strict_ids,
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-topographic-sensitivity-fd-v09",

        "status":
            status,

        "method":
            "GAMMA_phase_sim_orb_pt_central_finite_difference",

        "main_delta_height_m":
            main_dh,

        "tested_delta_heights_m":
            test_dh,

        "points":
            int(
                npoint
            ),

        "acquisitions":
            int(
                ndate
            ),

        "production_ifgs":
            int(
                nifg
            ),

        "geometric_reference": {
            "index":
                int(
                    geom_ref_idx
                ),

            "date":
                geom_ref_date,
        },

        "temporal_reference": {
            "index":
                int(
                    temporal_ref_idx
                ),

            "date":
                temporal_ref_date,
        },

        "height": {
            "source":
                str(
                    height_path
                ),

            "geometry_par":
                str(
                    height_par
                ),

            "valid_points":
                int(
                    total_valid_height
                ),

            "invalid_points":
                int(
                    total_invalid_height
                ),
        },

        "sensitivity": {
            "definition":
                "d(phase_sim_orb_pt)/d(height)",

            "units":
                "rad_per_m",

            "reference":
                (
                    "re-referenced to temporal "
                    "reference acquisition"
                ),

            "rms_rad_per_m":
                sensitivity_rms,

            "min_rad_per_m":
                float(
                    sensitivity_min
                ),

            "max_rad_per_m":
                float(
                    sensitivity_max
                ),

            "physical_residual_dem_sign_assigned":
                False,
        },

        "linearity": {
            str(
                dh
            ):
                linearity[
                    dh
                ]
            for dh in (
                1.0,
                5.0,
                20.0,
            )
        },

        "production_network_parity": {
            "rms_rad_per_m":
                network_rms,

            "max_rad_per_m":
                float(
                    network_max
                ),

            "relative_rms":
                float(
                    network_relative_rms
                ),
        },

        "outputs": {
            "acquisition_sensitivity":
                str(
                    sensitivity_path
                ),

            "network_rms_by_point":
                str(
                    network_rms_path
                ),

            "linearity_max_abs_by_point":
                str(
                    linearity_path
                ),
        },

        "persistent_ifg_sensitivity_cube":
            False,

        "phase_modified":
            False,

        "residual_dem_correction_applied":
            False,

        "note":
            (
                "Sensitivity is the raw derivative of GAMMA "
                "phase_sim_orb_pt with respect to DEM height. "
                "The physical residual-DEM sign in the corrected "
                "phase is intentionally not assigned at this stage."
            ),
    }

    manifest_path = (
        outdir
        /
        "topographic_sensitivity_manifest.json"
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
        f"acquisition sensitivity    : "
        f"{sensitivity_path}"
    )

    print(
        f"network point QA           : "
        f"{network_rms_path}"
    )

    print(
        f"linearity point QA         : "
        f"{linearity_path}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10a3 STATUS: "
        f"{status} / "
        "TOPOGRAPHIC SENSITIVITY AUDIT"
    )

    print(
        "No phase correction or residual "
        "DEM correction has been applied."
    )


if __name__ == "__main__":
    main()
