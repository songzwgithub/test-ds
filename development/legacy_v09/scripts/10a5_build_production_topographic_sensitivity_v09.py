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


def simulate(
    *,
    phase_sim,
    plist,
    pmask,
    pslc_par,
    itab,
    phgt,
    output,
    reference_par,
    nrec,
    npoint,
    gamma_log,
    label,
):

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
        gamma_log,
        label,
    )

    raw = np.fromfile(
        output,
        dtype=">f4",
    )

    expected = nrec * npoint

    if raw.size != expected:
        raise RuntimeError(
            f"{label}: output size={raw.size}, "
            f"expected={expected}"
        )

    return (
        raw.astype(np.float64)
        .reshape(
            nrec,
            npoint,
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

    phase_corr_dir = (
        output_base
        /
        "phase_correction_v09"
    )

    sensitivity_dir = (
        root
        /
        "scla_v09"
        /
        "topographic_sensitivity"
    )

    stepsize_dir = (
        root
        /
        "scla_v09"
        /
        "fd_stepsize_audit"
    )

    outdir = (
        root
        /
        "scla_v09"
        /
        "production_sensitivity"
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
    # Phase-correction geometry preparation
    # ========================================================

    prep_manifest = json.loads(
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

    if not pslc_par.is_file():
        raise FileNotFoundError(
            pslc_par
        )

    geom_ref_idx = int(
        prep_manifest[
            "geometric_reference_index"
        ]
    )

    geom_ref_date = str(
        prep_manifest[
            "geometric_reference_date"
        ]
    )

    height_path = Path(
        prep_manifest[
            "height_path"
        ]
    )

    height_par = Path(
        prep_manifest[
            "height_geometry_par"
        ]
    )

    reference_par = Path(
        stack.records[
            geom_ref_idx
        ].par
    ).resolve()

    # ========================================================
    # GAMMA executables
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

    # ========================================================
    # Strict points
    # ========================================================

    strict_ids = np.load(
        inversion_dir
        /
        "strict_point_ids.npy"
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

    # IDs must be strictly increasing for sample parity lookup.
    strict_sorted = bool(
        np.all(
            strict_ids[1:]
            >
            strict_ids[:-1]
        )
    )

    if not strict_sorted:
        raise RuntimeError(
            "strict_point_ids are not strictly increasing"
        )

    # ========================================================
    # Step09 temporal reference
    # ========================================================

    inv_manifest = json.loads(
        (
            inversion_dir
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

    if (
        str(
            stack.dates[
                temporal_ref_idx
            ]
        )
        !=
        temporal_ref_date
    ):
        raise RuntimeError(
            "Temporal reference mismatch"
        )

    # ========================================================
    # Confirm Step10a4 decision
    # ========================================================

    fd_manifest_path = (
        stepsize_dir
        /
        "fd_stepsize_audit_manifest.json"
    )

    fd_manifest = json.loads(
        fd_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    recommended_dh = float(
        fd_manifest[
            "recommended_production_delta_height_m"
        ]
    )

    if recommended_dh != 20.0:
        raise RuntimeError(
            "Step10a4 did not recommend 20 m"
        )

    if not str(
        fd_manifest[
            "status"
        ]
    ).startswith(
        "PASS"
    ):
        raise RuntimeError(
            "Step10a4 status is not PASS"
        )

    dh = 20.0

    # ========================================================
    # Existing full-scene 10 m candidate for independent
    # production comparison.
    # ========================================================

    candidate10_path = (
        sensitivity_dir
        /
        "d_phase_sim_dh_acquisition_rad_per_m.npy"
    )

    candidate10 = np.load(
        candidate10_path,
        mmap_mode="r",
    )

    if candidate10.shape != (
        npoint,
        ndate,
    ):
        raise RuntimeError(
            "10 m sensitivity shape mismatch"
        )

    # ========================================================
    # Reference-to-all GAMMA itab
    # ========================================================

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

    if nrec != ndate - 1:
        raise RuntimeError(
            "reference-to-all record count mismatch"
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
    # Production output
    # ========================================================

    production_path = (
        outdir
        /
        "topographic_phase_sensitivity_rad_per_m.npy"
    )

    production = np.lib.format.open_memmap(
        production_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            npoint,
            ndate,
        ),
    )

    diff10_path = (
        outdir
        /
        "fd20_vs_fd10_max_abs_difference_by_point_rad_per_m.npy"
    )

    diff10_point = np.lib.format.open_memmap(
        diff10_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            npoint,
        ),
    )

    # ========================================================
    # Statistics
    # ========================================================

    valid_height_total = 0
    invalid_height_total = 0

    production_ss = 0.0
    production_n = 0

    production_min = np.inf
    production_max = -np.inf

    diff10_ss = 0.0
    diff10_n = 0
    diff10_max = 0.0

    temporal_ref_max = 0.0
    geom_ref_before_max = 0.0

    print("=" * 112)
    print(
        "Step 10a5 - Build production GAMMA "
        "topographic sensitivity"
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
        f"geometric reference        : "
        f"{geom_ref_date}"
    )

    print(
        f"temporal reference         : "
        f"{temporal_ref_date}"
    )

    print(
        f"production delta-h         : "
        f"±{dh:g} m"
    )

    print(
        f"batch size                 : "
        f"{args.batch_size:,}"
    )

    print()

    # ========================================================
    # Full-scene processing
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

        B = b1 - b0

        br = rows[
            b0:b1
        ]

        bc = cols[
            b0:b1
        ]

        with tempfile.TemporaryDirectory(
            prefix=f"b{b0:07d}_",
            dir=scratch_root,
        ) as td:

            workdir = Path(td)

            # IPTA plist = range, azimuth = col, row.
            plist_arr = np.column_stack(
                (
                    bc,
                    br,
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

            # -----------------------------------------------
            # Native DEM height
            # -----------------------------------------------

            phgt_native = (
                workdir
                /
                "phgt_native"
            )

            run_command(
                [
                    data2pt,
                    str(height_path),
                    str(height_par),
                    str(plist),
                    str(reference_par),
                    str(phgt_native),
                    "1",
                    "2",
                ],
                gamma_log,
                f"data2pt:{b0}-{b1}",
            )

            h = np.fromfile(
                phgt_native,
                dtype=">f4",
            )

            if h.size != B:
                raise RuntimeError(
                    f"Height count {h.size} "
                    f"!= {B}"
                )

            h_native = h.astype(
                np.float32
            )

            valid = np.isfinite(
                h_native
            )

            # Same zero-height policy as previous frozen QA.
            z = (
                valid
                &
                (
                    h_native == 0.0
                )
            )

            h_native[
                z
            ] = np.float32(
                1.0e-3
            )

            nv = int(
                np.count_nonzero(
                    valid
                )
            )

            valid_height_total += nv
            invalid_height_total += (
                B - nv
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

            # -----------------------------------------------
            # h + 20 m / h - 20 m
            # -----------------------------------------------

            hp = h_native.copy()
            hm = h_native.copy()

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

            hp_path = (
                workdir
                /
                "phgt_plus20"
            )

            hm_path = (
                workdir
                /
                "phgt_minus20"
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

            plus = simulate(
                phase_sim=phase_sim,
                plist=plist,
                pmask=pmask,
                pslc_par=pslc_par,
                itab=itab,
                phgt=hp_path,
                output=(
                    workdir
                    /
                    "sim_plus20"
                ),
                reference_par=reference_par,
                nrec=nrec,
                npoint=B,
                gamma_log=gamma_log,
                label=(
                    f"phase_sim:+20m:"
                    f"{b0}-{b1}"
                ),
            )

            minus = simulate(
                phase_sim=phase_sim,
                plist=plist,
                pmask=pmask,
                pslc_par=pslc_par,
                itab=itab,
                phgt=hm_path,
                output=(
                    workdir
                    /
                    "sim_minus20"
                ),
                reference_par=reference_par,
                nrec=nrec,
                npoint=B,
                gamma_log=gamma_log,
                label=(
                    f"phase_sim:-20m:"
                    f"{b0}-{b1}"
                ),
            )

            pair_derivative = (
                plus
                -
                minus
            ) / (
                2.0 * dh
            )

            # -----------------------------------------------
            # 37 reference-to-secondary records -> 38
            # acquisitions.
            # -----------------------------------------------

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
                    sec_idx
                ] = pair_derivative[
                    rec
                ]

            if np.any(
                valid
            ):

                geom_ref_before_max = max(
                    geom_ref_before_max,
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

            # -----------------------------------------------
            # Re-reference geometry to Step09 temporal ref.
            # -----------------------------------------------

            S -= (
                S[
                    temporal_ref_idx
                ][
                    None,
                    :
                ]
            )

            S[
                :,
                ~valid
            ] = np.nan

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

            # Point-major output.
            Spt = S.T

            production[
                b0:b1
            ] = Spt.astype(
                np.float32
            )

            # -----------------------------------------------
            # Full-scene 20 m vs previous 10 m candidate.
            # -----------------------------------------------

            S10 = np.asarray(
                candidate10[
                    b0:b1
                ],
                dtype=np.float64,
            )

            d = (
                Spt
                -
                S10
            )

            finite = np.isfinite(
                d
            )

            if np.any(
                finite
            ):

                dv = d[
                    finite
                ]

                diff10_ss += float(
                    np.sum(
                        dv * dv
                    )
                )

                diff10_n += int(
                    dv.size
                )

                diff10_max = max(
                    diff10_max,
                    float(
                        np.max(
                            np.abs(
                                dv
                            )
                        )
                    ),
                )

            pmax = np.nanmax(
                np.abs(
                    d
                ),
                axis=1,
            )

            diff10_point[
                b0:b1
            ] = pmax.astype(
                np.float32
            )

            finite_s = np.isfinite(
                Spt
            )

            if np.any(
                finite_s
            ):

                sv = Spt[
                    finite_s
                ]

                production_ss += float(
                    np.sum(
                        sv * sv
                    )
                )

                production_n += int(
                    sv.size
                )

                production_min = min(
                    production_min,
                    float(
                        np.min(
                            sv
                        )
                    ),
                )

                production_max = max(
                    production_max,
                    float(
                        np.max(
                            sv
                        )
                    ),
                )

        print(
            f"  {b1:,}/"
            f"{npoint:,} "
            f"height_valid="
            f"{valid_height_total:,}"
        )

    production.flush()
    diff10_point.flush()

    # ========================================================
    # Global full-scene statistics
    # ========================================================

    production_rms = float(
        np.sqrt(
            production_ss
            /
            production_n
        )
    )

    diff10_rms = float(
        np.sqrt(
            diff10_ss
            /
            diff10_n
        )
    )

    diff10_relative = (
        diff10_rms
        /
        production_rms
    )

    diff10_arr = np.asarray(
        diff10_point,
        dtype=np.float64,
    )

    # ========================================================
    # Exact Step10a4 sample reproducibility
    # ========================================================

    sample_ids = np.load(
        stepsize_dir
        /
        "sample_strict_point_ids.npy"
    ).astype(
        np.int32,
        copy=False,
    )

    sample20 = np.asarray(
        np.load(
            stepsize_dir
            /
            "sample_sensitivity_dh20m_rad_per_m.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    sample_pos = np.searchsorted(
        strict_ids,
        sample_ids,
    )

    if (
        np.any(
            sample_pos
            >=
            npoint
        )
        or
        not np.array_equal(
            strict_ids[
                sample_pos
            ],
            sample_ids,
        )
    ):
        raise RuntimeError(
            "Could not map Step10a4 sample IDs "
            "into strict point domain"
        )

    production_sample = np.asarray(
        production[
            sample_pos
        ],
        dtype=np.float64,
    )

    sample_diff = (
        production_sample
        -
        sample20
    )

    sample_parity_rms = float(
        np.sqrt(
            np.mean(
                sample_diff
                *
                sample_diff
            )
        )
    )

    sample_parity_max = float(
        np.max(
            np.abs(
                sample_diff
            )
        )
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Production sensitivity summary"
    )
    print("=" * 112)

    print(
        f"height valid points        : "
        f"{valid_height_total:,}/"
        f"{npoint:,}"
    )

    print(
        f"height invalid points      : "
        f"{invalid_height_total:,}"
    )

    print(
        f"sensitivity min/max        : "
        f"{production_min:.6e} / "
        f"{production_max:.6e} rad/m"
    )

    print(
        f"sensitivity RMS            : "
        f"{production_rms:.6e} rad/m"
    )

    print(
        f"geometric-ref raw max      : "
        f"{geom_ref_before_max:.3e} rad/m"
    )

    print(
        f"temporal-ref max           : "
        f"{temporal_ref_max:.3e} rad/m"
    )

    print()
    print("=" * 112)
    print(
        "Full-scene 20m vs previous 10m sensitivity"
    )
    print("=" * 112)

    print(
        f"RMS difference             : "
        f"{diff10_rms:.6e} rad/m"
    )

    print(
        f"maximum difference         : "
        f"{diff10_max:.6e} rad/m"
    )

    print(
        f"relative RMS               : "
        f"{diff10_relative:.6e}"
    )

    q = np.percentile(
        diff10_arr[
            np.isfinite(
                diff10_arr
            )
        ],
        [
            50,
            90,
            95,
            99,
            100,
        ],
    )

    print(
        "point max |20m-10m| "
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
        "Step10a4 subset reproducibility"
    )
    print("=" * 112)

    print(
        f"sample points              : "
        f"{sample_ids.size:,}"
    )

    print(
        f"RMS difference             : "
        f"{sample_parity_rms:.6e} rad/m"
    )

    print(
        f"maximum difference         : "
        f"{sample_parity_max:.6e} rad/m"
    )

    # ========================================================
    # Status
    # ========================================================

    if invalid_height_total != 0:

        status = (
            "REVIEW_INVALID_HEIGHT"
        )

    elif temporal_ref_max > 1.0e-7:

        status = (
            "REVIEW_TEMPORAL_REFERENCE"
        )

    elif sample_parity_max > 1.0e-5:

        status = (
            "REVIEW_SAMPLE_REPRODUCIBILITY"
        )

    elif (
        diff10_relative
        >
        2.0e-3
    ):

        status = (
            "REVIEW_FD20_VS_FD10"
        )

    else:

        status = (
            "PASS"
        )

    # ========================================================
    # Save
    # ========================================================

    np.save(
        outdir
        /
        "strict_point_ids.npy",
        strict_ids,
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-production-topographic-sensitivity-v09",

        "status":
            status,

        "method":
            "GAMMA_phase_sim_orb_pt_central_finite_difference",

        "delta_height_m":
            20.0,

        "delta_height_source":
            str(
                fd_manifest_path
            ),

        "points":
            int(
                npoint
            ),

        "acquisitions":
            int(
                ndate
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

        "sensitivity": {
            "units":
                "rad_per_m",

            "definition":
                "d(phase_sim_orb_pt)/d(height)",

            "rms_rad_per_m":
                production_rms,

            "min_rad_per_m":
                float(
                    production_min
                ),

            "max_rad_per_m":
                float(
                    production_max
                ),
        },

        "fd20_vs_fd10": {
            "rms_rad_per_m":
                diff10_rms,

            "max_rad_per_m":
                float(
                    diff10_max
                ),

            "relative_rms":
                float(
                    diff10_relative
                ),
        },

        "step10a4_reproducibility": {
            "sample_points":
                int(
                    sample_ids.size
                ),

            "rms_rad_per_m":
                sample_parity_rms,

            "max_rad_per_m":
                sample_parity_max,
        },

        "production_output":
            str(
                production_path
            ),

        "persistent_ifg_cube":
            False,

        "physical_residual_dem_sign_assigned":
            False,

        "phase_modified":
            False,

        "residual_dem_correction_applied":
            False,
    }

    manifest_path = (
        outdir
        /
        "production_sensitivity_manifest.json"
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
        f"production sensitivity     : "
        f"{production_path}"
    )

    print(
        f"20m-vs-10m point QA        : "
        f"{diff10_path}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10a5 STATUS: "
        f"{status} / "
        "PRODUCTION TOPOGRAPHIC SENSITIVITY"
    )

    print(
        "No residual DEM or phase "
        "correction has been applied."
    )


if __name__ == "__main__":
    main()
