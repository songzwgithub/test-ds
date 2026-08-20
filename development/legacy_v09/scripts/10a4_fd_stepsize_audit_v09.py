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


def write_itab(path, ref_idx, secondary_indices):

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
    log_file,
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
        log_file,
        label,
    )

    raw = np.fromfile(
        output,
        dtype=">f4",
    )

    expected = nrec * npoint

    if raw.size != expected:

        raise RuntimeError(
            f"{label}: output size "
            f"{raw.size} != {expected}"
        )

    return (
        raw.astype(np.float64)
        .reshape(
            nrec,
            npoint,
        )
    )


def central_difference(
    dh,
    h_native,
    valid,
    workdir,
    *,
    phase_sim,
    plist,
    pmask,
    pslc_par,
    itab,
    reference_par,
    nrec,
    npoint,
    log_file,
):

    hp = h_native.copy()
    hm = h_native.copy()

    hp[valid] += np.float32(dh)
    hm[valid] -= np.float32(dh)

    hp_path = workdir / f"h_plus_{dh:g}"
    hm_path = workdir / f"h_minus_{dh:g}"

    hp.astype(">f4").tofile(hp_path)
    hm.astype(">f4").tofile(hm_path)

    plus = simulate(
        phase_sim,
        plist,
        pmask,
        pslc_par,
        itab,
        hp_path,
        workdir / f"sim_plus_{dh:g}",
        reference_par,
        nrec,
        npoint,
        log_file,
        f"phase_sim:+{dh:g}m",
    )

    minus = simulate(
        phase_sim,
        plist,
        pmask,
        pslc_par,
        itab,
        hm_path,
        workdir / f"sim_minus_{dh:g}",
        reference_par,
        nrec,
        npoint,
        log_file,
        f"phase_sim:-{dh:g}m",
    )

    return (
        plus - minus
    ) / (
        2.0 * dh
    )


def rms(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if not x.size:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                x * x
            )
        )
    )


def compare(a, b):

    d = a - b

    finite = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    dv = d[finite]

    av = a[finite]
    bv = b[finite]

    if not dv.size:
        return {
            "rms": np.nan,
            "max": np.nan,
            "relative_rms": np.nan,
        }

    diff_rms = rms(dv)

    # Symmetric scale: avoid privileging either step.
    scale = 0.5 * (
        rms(av)
        +
        rms(bv)
    )

    return {
        "rms":
            float(diff_rms),

        "max":
            float(
                np.max(
                    np.abs(dv)
                )
            ),

        "relative_rms":
            float(
                diff_rms / scale
                if scale > 0
                else np.nan
            ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--sample-points",
        type=int,
        default=20000,
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

    root = output_base / "v09"

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

    outdir = (
        root
        /
        "scla_v09"
        /
        "fd_stepsize_audit"
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
    # Existing phase-correction assets
    # ========================================================

    manifest_path = (
        phase_corr_dir
        /
        "manifest.json"
    )

    prep = json.loads(
        manifest_path.read_text(
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

    if not pslc_par.is_file():
        raise FileNotFoundError(
            pslc_par
        )

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

    nstrict = strict_ids.size

    nsample = min(
        args.sample_points,
        nstrict,
    )

    # PointPhaseStack point IDs are deterministic row-major.
    # Uniform index spacing therefore provides a deterministic
    # scene-wide numerical sample.
    sample_pos = np.linspace(
        0,
        nstrict - 1,
        nsample,
        dtype=np.int64,
    )

    sample_strict_ids = strict_ids[
        sample_pos
    ]

    rows = all_rows[
        sample_strict_ids
    ]

    cols = all_cols[
        sample_strict_ids
    ]

    ndate = len(
        stack.dates
    )

    # ========================================================
    # Temporal reference
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

    # ========================================================
    # Geometric-reference-to-all itab
    # ========================================================

    secondary_indices = [
        i
        for i in range(ndate)
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

    write_itab(
        itab,
        geom_ref_idx,
        secondary_indices,
    )

    print("=" * 112)
    print(
        "Step 10a4 - GAMMA finite-difference "
        "step-size stability audit"
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
        f"sample points              : "
        f"{nsample:,}"
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
        "tested steps               : "
        "10 / 20 / 40 / 80 m"
    )

    # ========================================================
    # One temporary GAMMA workspace
    # ========================================================

    with tempfile.TemporaryDirectory(
        prefix="fd_stepsize_",
        dir=outdir,
    ) as td:

        workdir = Path(td)

        # range / azimuth = col / row
        plist_arr = np.column_stack(
            (
                cols,
                rows,
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
            "data2pt:sample",
        )

        h = np.fromfile(
            phgt_native,
            dtype=">f4",
        )

        if h.size != nsample:

            raise RuntimeError(
                f"height count {h.size} "
                f"!= sample count {nsample}"
            )

        h_native = h.astype(
            np.float32
        )

        valid = np.isfinite(
            h_native
        )

        # Same zero-height handling as 10a3.
        z = (
            valid
            &
            (
                h_native == 0.0
            )
        )

        h_native[z] = np.float32(
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

        print(
            f"valid sampled heights      : "
            f"{np.count_nonzero(valid):,}/"
            f"{nsample:,}"
        )

        if not np.all(valid):
            raise RuntimeError(
                "Unexpected invalid sampled height"
            )

        # ====================================================
        # Run 10 / 20 / 40 / 80 m
        # ====================================================

        steps = [
            10.0,
            20.0,
            40.0,
            80.0,
        ]

        sensitivity = {}

        for dh in steps:

            print(
                f"  simulating ±{dh:g} m ..."
            )

            pair_derivative = central_difference(
                dh,
                h_native,
                valid,
                workdir,
                phase_sim=phase_sim,
                plist=plist,
                pmask=pmask,
                pslc_par=pslc_par,
                itab=itab,
                reference_par=reference_par,
                nrec=nrec,
                npoint=nsample,
                log_file=gamma_log,
            )

            # Geometric-ref acquisition stack.
            S = np.zeros(
                (
                    ndate,
                    nsample,
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

            # Same Step09 temporal reference.
            S -= (
                S[
                    temporal_ref_idx
                ][
                    None,
                    :
                ]
            )

            sensitivity[
                dh
            ] = S.T.copy()

    # ========================================================
    # Save sample products
    # ========================================================

    np.save(
        outdir
        /
        "sample_strict_point_ids.npy",
        sample_strict_ids,
    )

    np.save(
        outdir
        /
        "sample_rows.npy",
        rows,
    )

    np.save(
        outdir
        /
        "sample_cols.npy",
        cols,
    )

    for dh in (
        10.0,
        20.0,
        40.0,
        80.0,
    ):

        np.save(
            outdir
            /
            f"sample_sensitivity_dh{int(dh)}m_rad_per_m.npy",
            sensitivity[
                dh
            ].astype(
                np.float32
            ),
        )

    # ========================================================
    # Step comparisons
    # ========================================================

    comparisons = {
        "10_vs_20":
            compare(
                sensitivity[10.0],
                sensitivity[20.0],
            ),

        "20_vs_40":
            compare(
                sensitivity[20.0],
                sensitivity[40.0],
            ),

        "20_vs_80":
            compare(
                sensitivity[20.0],
                sensitivity[80.0],
            ),

        "40_vs_80":
            compare(
                sensitivity[40.0],
                sensitivity[80.0],
            ),
    }

    rms_by_step = {
        int(dh):
            rms(
                sensitivity[
                    dh
                ]
            )
        for dh in (
            10.0,
            20.0,
            40.0,
            80.0,
        )
    }

    # ========================================================
    # Point-wise 20-vs-larger stability
    # ========================================================

    d20_40 = np.max(
        np.abs(
            sensitivity[20.0]
            -
            sensitivity[40.0]
        ),
        axis=1,
    )

    d20_80 = np.max(
        np.abs(
            sensitivity[20.0]
            -
            sensitivity[80.0]
        ),
        axis=1,
    )

    # ========================================================
    # Decision
    #
    # Prefer the smallest step already on the stable plateau.
    # ========================================================

    r10_20 = comparisons[
        "10_vs_20"
    ][
        "relative_rms"
    ]

    r20_40 = comparisons[
        "20_vs_40"
    ][
        "relative_rms"
    ]

    r20_80 = comparisons[
        "20_vs_80"
    ][
        "relative_rms"
    ]

    r40_80 = comparisons[
        "40_vs_80"
    ][
        "relative_rms"
    ]

    if (
        r20_40 <= 1.0e-3
        and
        r20_80 <= 1.5e-3
        and
        r40_80 <= 1.0e-3
    ):

        status = (
            "PASS_RECOMMEND_20M"
        )

        recommended_dh = 20.0

    elif (
        r40_80 <= 1.0e-3
    ):

        status = (
            "PASS_RECOMMEND_40M"
        )

        recommended_dh = 40.0

    else:

        status = (
            "REVIEW_STEP_SIZE_NONLINEARITY"
        )

        recommended_dh = None

    # ========================================================
    # Console output
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Sensitivity scale"
    )
    print("=" * 112)

    for dh in (
        10,
        20,
        40,
        80,
    ):

        print(
            f"RMS S({dh:2d}m)             : "
            f"{rms_by_step[dh]:.6e} rad/m"
        )

    print()
    print("=" * 112)
    print(
        "Finite-difference step-size comparison"
    )
    print("=" * 112)

    print(
        " comparison       RMS diff       "
        "max diff       relative RMS"
    )

    for name in (
        "10_vs_20",
        "20_vs_40",
        "20_vs_80",
        "40_vs_80",
    ):

        q = comparisons[
            name
        ]

        print(
            f" {name:10s} "
            f"{q['rms']:.6e}   "
            f"{q['max']:.6e}   "
            f"{q['relative_rms']:.6e}"
        )

    print()
    print(
        "Successive relative differences:"
    )

    print(
        f"  10 -> 20 m              : "
        f"{r10_20:.6e}"
    )

    print(
        f"  20 -> 40 m              : "
        f"{r20_40:.6e}"
    )

    print(
        f"  40 -> 80 m              : "
        f"{r40_80:.6e}"
    )

    print()
    print(
        "20m point max |20-40| "
        "p50/p90/p95/p99/max:"
    )

    q = np.percentile(
        d20_40,
        [
            50,
            90,
            95,
            99,
            100,
        ],
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

    print(
        "20m point max |20-80| "
        "p50/p90/p95/p99/max:"
    )

    q = np.percentile(
        d20_80,
        [
            50,
            90,
            95,
            99,
            100,
        ],
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
    print(
        f"recommended production dh  : "
        f"{recommended_dh if recommended_dh is not None else 'NONE'}"
    )

    # ========================================================
    # Manifest
    # ========================================================

    manifest = {
        "format":
            "pyPSDS-GAMMA-fd-stepsize-audit-v09",

        "status":
            status,

        "strict_points":
            int(
                nstrict
            ),

        "sample_points":
            int(
                nsample
            ),

        "sampling":
            (
                "deterministic uniform sampling "
                "over strict PointPhaseStack order"
            ),

        "tested_delta_height_m": [
            10,
            20,
            40,
            80,
        ],

        "rms_sensitivity_rad_per_m":
            rms_by_step,

        "comparisons":
            comparisons,

        "recommended_production_delta_height_m":
            recommended_dh,

        "interpretation":
            (
                "The smallest finite-difference height on the "
                "stable 20-40-80 m plateau is preferred. "
                "Small-step differences may be dominated by "
                "FLOAT32 phase_sim_orb_pt output quantization."
            ),

        "phase_modified":
            False,

        "residual_dem_correction_applied":
            False,
    }

    manifest_out = (
        outdir
        /
        "fd_stepsize_audit_manifest.json"
    )

    manifest_out.write_text(
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
        f"{manifest_out}"
    )

    print()
    print(
        f"STEP 10a4 STATUS: "
        f"{status}"
    )

    print(
        "No phase or residual DEM "
        "correction has been applied."
    )


if __name__ == "__main__":
    main()
