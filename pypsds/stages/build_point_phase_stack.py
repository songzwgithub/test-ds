#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


TYPE_PS = np.uint8(1)
TYPE_DS = np.uint8(2)

EST_NONE = np.int16(-1)


def optional_load(path: Path, mmap_mode="r"):
    if path.exists():
        return np.load(
            path,
            mmap_mode=mmap_mode,
        )
    return None


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Build the final PS/DS PointPhaseStack "
            "from geometry-valid PS and final DS."
        )
    )

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--ds-mask",
        required=True,
    )

    ap.add_argument(
        "--ps-mask",
        default=None,
        help=(
            "Final usable PS mask. "
            "Default: <processing>/final_ps_mask.npy"
        ),
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=100000,
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

    dates = list(
        stack.dates
    )

    T = len(dates)

    processing = (
        Path(paths.output_dir)
        / "processing"
    )

    ps_path = (
        Path(args.ps_mask)
        if args.ps_mask
        else (
            processing
            / "final_ps_mask.npy"
        )
    )

    ds_path = Path(
        args.ds_mask
    )

    phase_path = (
        processing
        / "linked_phase.npy"
    )

    outdir = (
        processing
        / "point_phase_stack"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "Build final PS/DS PointPhaseStack"
    )
    print("=" * 80)

    print(
        f"config             : {config_path}"
    )

    print(
        f"scene              : {H} x {W}"
    )

    print(
        f"dates              : {T}"
    )

    print(
        f"reference date     : {dates[0]}"
    )

    print(
        f"PS mask            : {ps_path}"
    )

    print(
        f"DS mask            : {ds_path}"
    )

    # =========================================================
    # 1. Inputs
    # =========================================================

    print()
    print(
        "[06:1/7] Loading final PS/DS masks..."
    )

    if not ps_path.exists():
        raise FileNotFoundError(
            f"Missing final PS mask: {ps_path}"
        )

    if not ds_path.exists():
        raise FileNotFoundError(
            f"Missing final DS mask: {ds_path}"
        )

    if not phase_path.exists():
        raise FileNotFoundError(
            f"Missing linked phase: {phase_path}"
        )

    ps = np.load(
        ps_path,
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    ds = np.load(
        ds_path,
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    if ps.shape != (H, W):
        raise RuntimeError(
            f"PS shape={ps.shape}, "
            f"expected={(H,W)}"
        )

    if ds.shape != (H, W):
        raise RuntimeError(
            f"DS shape={ds.shape}, "
            f"expected={(H,W)}"
        )

    phase = np.load(
        phase_path,
        mmap_mode="r",
    )

    if phase.shape != (
        T,
        H,
        W,
    ):
        raise RuntimeError(
            f"linked_phase shape={phase.shape}, "
            f"expected={(T,H,W)}"
        )

    overlap = (
        ps & ds
    )

    print(
        f"final PS           : "
        f"{ps.sum()}"
    )

    print(
        f"final DS           : "
        f"{ds.sum()}"
    )

    print(
        f"PS/DS overlap      : "
        f"{overlap.sum()}"
    )

    # Production contract:
    # PS always has priority over DS.
    ds_unique = (
        ds & ~ps
    )

    fused = (
        ps | ds_unique
    )

    Nps = int(
        ps.sum()
    )

    Nds = int(
        ds_unique.sum()
    )

    N = int(
        fused.sum()
    )

    print(
        f"usable DS          : "
        f"{Nds}"
    )

    print(
        f"PointPhaseStack N  : "
        f"{N}"
    )

    # =========================================================
    # 2. Deterministic point ordering
    # =========================================================

    print()
    print(
        "[06:2/7] Building deterministic point index..."
    )

    # Row-major spatial order.
    rows, cols = np.where(
        fused
    )

    rows = rows.astype(
        np.int32
    )

    cols = cols.astype(
        np.int32
    )

    if len(rows) != N:
        raise RuntimeError(
            "Point count mismatch."
        )

    point_type = np.empty(
        N,
        dtype=np.uint8,
    )

    is_ps_point = ps[
        rows,
        cols,
    ]

    point_type[:] = TYPE_DS

    point_type[
        is_ps_point
    ] = TYPE_PS

    print(
        f"ordered PS         : "
        f"{np.sum(point_type==TYPE_PS)}"
    )

    print(
        f"ordered DS         : "
        f"{np.sum(point_type==TYPE_DS)}"
    )

    # =========================================================
    # 3. Extract phase_rad
    # =========================================================

    print()
    print(
        "[06:3/7] Writing phase_rad.npy..."
    )

    phase_out_path = (
        outdir
        / "phase_rad.npy"
    )

    phase_rad = np.lib.format.open_memmap(
        phase_out_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            N,
            T,
        ),
    )

    for p0 in range(
        0,
        N,
        args.batch_size,
    ):

        p1 = min(
            N,
            p0 + args.batch_size,
        )

        rr = rows[
            p0:p1
        ]

        cc = cols[
            p0:p1
        ]

        # linked_phase:
        # [date,row,col]
        #
        # sampled:
        # [date,point]
        z = phase[
            :,
            rr,
            cc,
        ]

        phase_valid = (
            np.isfinite(
                z.real
            )
            &
            np.isfinite(
                z.imag
            )
            &
            (
                z
                !=
                np.complex64(0.0)
            )
        )

        if not np.all(
            phase_valid
        ):
            bad = np.where(
                ~np.all(
                    phase_valid,
                    axis=0,
                )
            )[0]

            raise RuntimeError(
                f"Incomplete phase history "
                f"inside PointPhaseStack batch "
                f"{p0}:{p1}; "
                f"bad points={len(bad)}"
            )

        ph = np.angle(
            z
        ).T.astype(
            np.float32,
            copy=False,
        )

        # Force exact common reference.
        #
        # Normally this correction is only ~1e-7 rad,
        # but explicitly enforcing it makes the stack
        # contract deterministic.
        ph -= ph[
            :,
            0:1
        ]

        # Wrap back to [-pi, pi).
        ph[:] = (
            (
                ph
                + np.pi
            )
            % (
                2.0 * np.pi
            )
            - np.pi
        )

        # Exact zero at reference epoch.
        ph[
            :,
            0
        ] = 0.0

        phase_rad[
            p0:p1,
            :
        ] = ph

        if (
            p1 == N
            or p1 % 100000 == 0
        ):
            print(
                f"  points "
                f"{p1:7d}/{N:7d} "
                f"({100*p1/N:6.2f}%)"
            )

    phase_rad.flush()

    # =========================================================
    # 4. Save core point arrays
    # =========================================================

    print()
    print(
        "[06:4/7] Saving point metadata arrays..."
    )

    np.save(
        outdir / "rows.npy",
        rows,
    )

    np.save(
        outdir / "cols.npy",
        cols,
    )

    np.save(
        outdir / "point_type.npy",
        point_type,
    )

    point_id = np.arange(
        N,
        dtype=np.int64,
    )

    np.save(
        outdir / "point_id.npy",
        point_id,
    )

    # =========================================================
    # 5. Quality metadata
    # =========================================================

    print()
    print(
        "[06:5/7] Building quality metadata..."
    )

    tc = optional_load(
        processing
        / "temporal_coherence.npy"
    )

    pair = optional_load(
        processing
        / "median_pair_coherence.npy"
    )

    shp = optional_load(
        processing
        / "shp_count.npy"
    )

    est = optional_load(
        processing
        / "estimator_code.npy"
    )

    emi_eig = optional_load(
        processing
        / "emi_eigenvalue.npy"
    )

    evd_eig = optional_load(
        processing
        / "evd_eigenvalue.npy"
    )

    gamma_min = optional_load(
        processing
        / "gamma_min_eigenvalue.npy"
    )

    # PS does not use DS quality metrics.
    point_tc = np.full(
        N,
        np.nan,
        dtype=np.float32,
    )

    point_pair = np.full(
        N,
        np.nan,
        dtype=np.float32,
    )

    point_K = np.full(
        N,
        -1,
        dtype=np.int16,
    )

    point_est = np.full(
        N,
        EST_NONE,
        dtype=np.int16,
    )

    point_emi_eig = np.full(
        N,
        np.nan,
        dtype=np.float32,
    )

    point_evd_eig = np.full(
        N,
        np.nan,
        dtype=np.float32,
    )

    point_gamma_min = np.full(
        N,
        np.nan,
        dtype=np.float32,
    )

    ds_point = (
        point_type
        == TYPE_DS
    )

    rds = rows[
        ds_point
    ]

    cds = cols[
        ds_point
    ]

    if tc is not None:
        point_tc[
            ds_point
        ] = tc[
            rds,
            cds
        ]

    if pair is not None:
        point_pair[
            ds_point
        ] = pair[
            rds,
            cds
        ]

    if shp is not None:
        point_K[
            ds_point
        ] = shp[
            rds,
            cds
        ].astype(
            np.int16
        )

    if est is not None:
        point_est[
            ds_point
        ] = est[
            rds,
            cds
        ].astype(
            np.int16
        )

    if emi_eig is not None:
        point_emi_eig[
            ds_point
        ] = emi_eig[
            rds,
            cds
        ]

    if evd_eig is not None:
        point_evd_eig[
            ds_point
        ] = evd_eig[
            rds,
            cds
        ]

    if gamma_min is not None:
        point_gamma_min[
            ds_point
        ] = gamma_min[
            rds,
            cds
        ]

    np.save(
        outdir
        / "temporal_coherence.npy",
        point_tc,
    )

    np.save(
        outdir
        / "median_pair_coherence.npy",
        point_pair,
    )

    np.save(
        outdir
        / "glrt_support_K.npy",
        point_K,
    )

    np.save(
        outdir
        / "estimator_code.npy",
        point_est,
    )

    np.save(
        outdir
        / "emi_eigenvalue.npy",
        point_emi_eig,
    )

    np.save(
        outdir
        / "evd_eigenvalue.npy",
        point_evd_eig,
    )

    np.save(
        outdir
        / "gamma_min_eigenvalue.npy",
        point_gamma_min,
    )

    # =========================================================
    # 6. Point table / dates / manifest
    # =========================================================

    print()
    print(
        "[06:6/7] Writing point table and manifest..."
    )

    dates_path = (
        outdir
        / "dates.txt"
    )

    with open(
        dates_path,
        "w"
    ) as f:
        for dt in dates:
            f.write(
                f"{dt}\n"
            )

    csv_path = (
        outdir
        / "points.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "point_id",
            "row",
            "col",
            "point_type",
            "point_type_name",
            "temporal_coherence",
            "median_pair_coherence",
            "glrt_support_K",
            "estimator_code",
            "emi_eigenvalue",
            "evd_eigenvalue",
            "gamma_min_eigenvalue",
        ])

        for i in range(N):

            typ = int(
                point_type[i]
            )

            name = (
                "PS"
                if typ == int(TYPE_PS)
                else "DS"
            )

            w.writerow([
                i,
                int(rows[i]),
                int(cols[i]),
                typ,
                name,
                (
                    f"{point_tc[i]:.8f}"
                    if np.isfinite(
                        point_tc[i]
                    )
                    else ""
                ),
                (
                    f"{point_pair[i]:.8f}"
                    if np.isfinite(
                        point_pair[i]
                    )
                    else ""
                ),
                int(
                    point_K[i]
                ),
                int(
                    point_est[i]
                ),
                (
                    f"{point_emi_eig[i]:.8f}"
                    if np.isfinite(
                        point_emi_eig[i]
                    )
                    else ""
                ),
                (
                    f"{point_evd_eig[i]:.8f}"
                    if np.isfinite(
                        point_evd_eig[i]
                    )
                    else ""
                ),
                (
                    f"{point_gamma_min[i]:.8f}"
                    if np.isfinite(
                        point_gamma_min[i]
                    )
                    else ""
                ),
            ])

    manifest = {
        "format": "pyPSDS PointPhaseStack",
        "version": "0.9",
        "scene_shape": [
            int(H),
            int(W),
        ],
        "n_dates": int(T),
        "n_points": int(N),
        "n_ps": int(Nps),
        "n_ds": int(Nds),
        "reference_date": str(
            dates[0]
        ),
        "phase_unit": "radian",
        "phase_range": "[-pi, pi)",
        "phase_reference": (
            "all point phases explicitly referenced "
            "to acquisition index 0"
        ),
        "phase_array": {
            "path": "phase_rad.npy",
            "dtype": "float32",
            "shape": [
                int(N),
                int(T),
            ],
            "axis_0": "point",
            "axis_1": "acquisition",
        },
        "point_type": {
            "1": "PS",
            "2": "DS",
        },
        "estimator_code": {
            "-1": "not_applicable_PS",
            "0": "EVD_fallback",
            "1": "EMI",
        },
        "fusion_rule": (
            "PS priority on overlap"
        ),
        "ps_source": str(
            ps_path
        ),
        "ds_source": str(
            ds_path
        ),
        "source_phase": str(
            phase_path
        ),
    }

    with open(
        outdir
        / "manifest.json",
        "w",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # =========================================================
    # 7. Final integrity quality
    # =========================================================

    print()
    print(
        "[06:7/7] Final PointPhaseStack quality..."
    )

    phase_check = np.load(
        phase_out_path,
        mmap_mode="r",
    )

    if phase_check.shape != (
        N,
        T,
    ):
        raise RuntimeError(
            "phase_rad shape mismatch."
        )

    if not np.all(
        np.isfinite(
            phase_check
        )
    ):
        raise RuntimeError(
            "phase_rad contains NaN/Inf."
        )

    max_ref = float(
        np.max(
            np.abs(
                phase_check[:, 0]
            )
        )
    )

    max_phase = float(
        np.max(
            np.abs(
                phase_check
            )
        )
    )

    # Coordinate uniqueness
    linear = (
        rows.astype(
            np.int64
        )
        * int(W)
        + cols.astype(
            np.int64
        )
    )

    unique_count = int(
        np.unique(
            linear
        ).size
    )

    if unique_count != N:
        raise RuntimeError(
            f"Duplicate point coordinates: "
            f"{N-unique_count}"
        )

    # DS selection contract
    ds_tc_min = float(
        np.nanmin(
            point_tc[
                ds_point
            ]
        )
    )

    ds_pair_nan = int(
        np.sum(
            ~np.isfinite(
                point_pair[
                    ds_point
                ]
            )
        )
    )

    print()
    print("=" * 80)
    print(
        "PointPhaseStack complete"
    )
    print("=" * 80)

    print(
        f"points             : {N}"
    )

    print(
        f"PS                 : {Nps}"
    )

    print(
        f"DS                 : {Nds}"
    )

    print(
        f"dates              : {T}"
    )

    print(
        f"phase shape        : "
        f"{phase_check.shape}"
    )

    print(
        f"phase dtype        : "
        f"{phase_check.dtype}"
    )

    print(
        f"reference max      : "
        f"{max_ref:.9e} rad"
    )

    print(
        f"max |phase|        : "
        f"{max_phase:.9f} rad"
    )

    print(
        f"unique coordinates : "
        f"{unique_count}/{N}"
    )

    print(
        f"DS minimum TC      : "
        f"{ds_tc_min:.8f}"
    )

    print(
        f"DS bad pair values : "
        f"{ds_pair_nan}"
    )

    print()
    print(
        f"output directory   : "
        f"{outdir}"
    )

    print()
    print(
        "STEP 06 STATUS: PASS"
    )


if __name__ == "__main__":
    main()
