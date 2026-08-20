#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


TWOPI = 2.0 * np.pi


def load_itab(path: Path, ndate: int):
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
                    f"Invalid ITAB line: {raw}"
                )

            edges.append((i, j))

    return edges


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=12000,
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

    graph_dir = (
        root
        / "spatial_graph"
    )

    network_dir = (
        root
        / "network"
    )

    unwrap_dir = (
        root
        / "single_ifg_robust_solution"
    )

    policy_dir = (
        root
        / "unwrap_component_policy"
    )

    temporal_dir = (
        root
        / "temporal_integer_closure_quality"
    )

    quality08r_dir = (
        root
        / "safe_conflict_acquisition_quality"
    )

    quality08w_dir = (
        root
        / "fragment_signature_feasibility"
    )

    outdir = (
        root
        / "final_unwrap"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Static data
    # ========================================================

    rows_pt = np.load(
        pps_dir / "rows.npy",
        mmap_mode="r",
    )

    cols_pt = np.load(
        pps_dir / "cols.npy",
        mmap_mode="r",
    )

    npoint = rows_pt.size
    ndate = len(stack.dates)

    temporal_edges = load_itab(
        network_dir / "network.itab",
        ndate,
    )

    nedge = len(temporal_edges)

    if nedge != 108:
        print(
            f"NOTE: production IFG count = {nedge}"
        )

    C = np.load(
        temporal_dir / "cycle_matrix.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    cycle_mode = np.load(
        temporal_dir / "cycle_modal_integer.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    main_mask = np.load(
        policy_dir / "main_component_mask.npy",
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    temporal_bad_count = np.load(
        temporal_dir
        / "point_temporal_bad_cycle_count.npy",
        mmap_mode="r",
    )

    main_temporal_bad = (
        np.asarray(
            temporal_bad_count
        )
        > 0
    )

    # ========================================================
    # Open production unwrapped IFGs and registration masks
    # ========================================================

    phase_maps = []
    registration_maps = []

    all_registered = np.ones(
        npoint,
        dtype=bool,
    )

    for pair_id, (ti, tj) in enumerate(
        temporal_edges,
        start=1,
    ):

        tag = (
            f"pair{pair_id:03d}_"
            f"{stack.dates[ti]}_"
            f"{stack.dates[tj]}"
        )

        phase_path = (
            unwrap_dir
            / f"{tag}_unwrapped_phase_rad.npy"
        )

        reg_path = (
            unwrap_dir
            / f"{tag}_registered_mask.npy"
        )

        if not phase_path.exists():
            raise FileNotFoundError(
                phase_path
            )

        if not reg_path.exists():
            raise FileNotFoundError(
                reg_path
            )

        U = np.load(
            phase_path,
            mmap_mode="r",
        )

        R = np.load(
            reg_path,
            mmap_mode="r",
        )

        if U.size != npoint:
            raise RuntimeError(
                f"{phase_path.name}: "
                "point count mismatch"
            )

        if R.size != npoint:
            raise RuntimeError(
                f"{reg_path.name}: "
                "point count mismatch"
            )

        phase_maps.append(U)
        registration_maps.append(R)

        all_registered &= np.asarray(
            R,
            dtype=bool,
        )

    # ========================================================
    # Derive global 2pi IFG gauge from fundamental cycles
    #
    # Step08t fundamental cycle construction has one unique
    # non-tree edge per cycle:
    #
    # C[:, non_tree_edges] = I
    #
    # Therefore simply setting
    #
    #     g_non_tree = -cycle_mode
    #
    # gives:
    #
    #     C g = -cycle_mode
    #
    # ========================================================

    cycle_csv = (
        temporal_dir
        / "temporal_cycle_qa.csv"
    )

    global_delta = np.zeros(
        nedge,
        dtype=np.int32,
    )

    seen_non_tree = set()

    with cycle_csv.open() as f:

        for r in csv.DictReader(f):

            cid = int(
                r["cycle_id"]
            ) - 1

            eid = int(
                r["non_tree_edge_id"]
            ) - 1

            mode = int(
                r["modal_integer"]
            )

            if eid in seen_non_tree:
                raise RuntimeError(
                    f"duplicate non-tree edge {eid+1}"
                )

            seen_non_tree.add(eid)

            global_delta[eid] = -mode

            if (
                mode
                !=
                cycle_mode[cid]
            ):
                raise RuntimeError(
                    "cycle mode CSV/NPY mismatch"
                )

    if len(seen_non_tree) != C.shape[0]:
        raise RuntimeError(
            "non-tree edge count mismatch"
        )

    gauge_check = (
        C
        @
        global_delta
    )

    if not np.array_equal(
        gauge_check,
        -cycle_mode,
    ):
        raise RuntimeError(
            "Global gauge verification failed: "
            "C @ global_delta != -cycle_mode"
        )

    # ========================================================
    # Residual points:
    # Step08t only qualityed main component.
    #
    # Quality residual points that are registered in ALL IFGs.
    # ========================================================

    residual_all_registered = (
        (~main_mask)
        &
        all_registered
    )

    residual_ids = np.where(
        residual_all_registered
    )[0].astype(
        np.int32
    )

    residual_temporal_bad = np.zeros(
        npoint,
        dtype=bool,
    )

    residual_bad_cycle_count = np.zeros(
        residual_ids.size,
        dtype=np.uint16,
    )

    residual_float_residual_max = 0.0

    if residual_ids.size:

        Xr = np.empty(
            (
                residual_ids.size,
                nedge,
            ),
            dtype=np.float64,
        )

        for e in range(nedge):

            Xr[:, e] = (
                np.asarray(
                    phase_maps[e][
                        residual_ids
                    ],
                    dtype=np.float64,
                )
                +
                TWOPI
                *
                global_delta[e]
            )

        closure = (
            Xr
            @
            C.T.astype(
                np.float64,
                copy=False,
            )
        )

        k = np.rint(
            closure / TWOPI
        ).astype(
            np.int32
        )

        residual_float = (
            closure
            -
            TWOPI * k
        )

        residual_float_residual_max = float(
            np.max(
                np.abs(
                    residual_float
                )
            )
        )

        bad = (
            k != 0
        )

        residual_bad_cycle_count = np.sum(
            bad,
            axis=1,
            dtype=np.uint16,
        )

        bad_ids = residual_ids[
            residual_bad_cycle_count > 0
        ]

        residual_temporal_bad[
            bad_ids
        ] = True

    # ========================================================
    # Complete temporal bad mask
    # ========================================================

    temporal_bad = (
        main_temporal_bad
        |
        residual_temporal_bad
    )

    temporal_valid = (
        all_registered
        &
        (~temporal_bad)
    )

    # ========================================================
    # SAFE-conflict point flag from Step08r
    #
    # Conservative strict-quality layer.
    # Do NOT delete these points.
    # ========================================================

    spatial_safe_conflict = np.zeros(
        npoint,
        dtype=bool,
    )

    suspicious_edge_path = (
        quality08r_dir
        / "suspicious_edge_ids.npy"
    )

    if suspicious_edge_path.exists():

        edge_ids = np.load(
            suspicious_edge_path
        ).astype(
            np.int64,
            copy=False,
        )

        local_u = np.load(
            graph_dir / "local_u.npy",
            mmap_mode="r",
        )

        local_v = np.load(
            graph_dir / "local_v.npy",
            mmap_mode="r",
        )

        pids = np.unique(
            np.concatenate(
                [
                    np.asarray(
                        local_u[edge_ids],
                        dtype=np.int32,
                    ),
                    np.asarray(
                        local_v[edge_ids],
                        dtype=np.int32,
                    ),
                ]
            )
        )

        spatial_safe_conflict[
            pids
        ] = True

    strict_valid = (
        temporal_valid
        &
        (~spatial_safe_conflict)
    )

    # ========================================================
    # Structural-incompatibility flag from Step08w.
    #
    # This should be a subset of temporal_bad.
    # ========================================================

    structural_incompatible = np.zeros(
        npoint,
        dtype=bool,
    )

    point_status_csv = (
        quality08w_dir
        / "temporal_bad_point_signature_status.csv"
    )

    if point_status_csv.exists():

        with point_status_csv.open() as f:

            for r in csv.DictReader(f):

                status = r[
                    "status"
                ]

                if status.startswith(
                    "INCOMPATIBLE"
                ):

                    pid = int(
                        r["point_id"]
                    )

                    structural_incompatible[
                        pid
                    ] = True

    if np.any(
        structural_incompatible
        &
        (~temporal_bad)
    ):
        raise RuntimeError(
            "Structural-incompatible point is "
            "not temporal-bad."
        )

    # ========================================================
    # Final full-scene temporal closure verification
    # on STRICT valid points after global gauge.
    # ========================================================

    strict_ids = np.where(
        strict_valid
    )[0].astype(
        np.int32
    )

    final_bad_occurrences = 0
    final_bad_points = 0
    final_float_residual_max = 0.0

    print("=" * 108)
    print(
        "Final unwrap quality mask "
        "and global temporal integer gauge"
    )
    print("=" * 108)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"points                     : "
        f"{npoint:,}"
    )

    print(
        f"IFGs                       : "
        f"{nedge}"
    )

    print(
        f"temporal cycles            : "
        f"{C.shape[0]}"
    )

    print()
    print(
        "Global IFG integer gauge:"
    )

    print(
        f"  nonzero IFG deltas       : "
        f"{np.count_nonzero(global_delta)}"
    )

    print(
        f"  max |integer delta|      : "
        f"{np.max(np.abs(global_delta))}"
    )

    print(
        f"  C@g + cycle_mode == 0    : "
        f"{np.array_equal(gauge_check + cycle_mode, np.zeros_like(cycle_mode))}"
    )

    print()
    print(
        "Quality-mask construction:"
    )

    print(
        f"  registered in all IFGs   : "
        f"{np.count_nonzero(all_registered):,}"
    )

    print(
        f"  not registered in all    : "
        f"{np.count_nonzero(~all_registered):,}"
    )

    print(
        f"  main temporal bad        : "
        f"{np.count_nonzero(main_temporal_bad):,}"
    )

    print(
        f"  residual all-registered  : "
        f"{residual_ids.size:,}"
    )

    print(
        f"  residual temporal bad    : "
        f"{np.count_nonzero(residual_temporal_bad):,}"
    )

    print(
        f"  total temporal bad       : "
        f"{np.count_nonzero(temporal_bad):,}"
    )

    print(
        f"  SAFE-conflict points     : "
        f"{np.count_nonzero(spatial_safe_conflict):,}"
    )

    print(
        f"  structural incompatible : "
        f"{np.count_nonzero(structural_incompatible):,}"
    )

    print(
        f"  temporal-valid points    : "
        f"{np.count_nonzero(temporal_valid):,}"
    )

    print(
        f"  STRICT-valid points      : "
        f"{strict_ids.size:,}"
    )

    print()
    print(
        "Final strict-mask temporal closure quality ..."
    )

    CT = C.T.astype(
        np.float64,
        copy=False,
    )

    for b0 in range(
        0,
        strict_ids.size,
        args.batch_size,
    ):

        b1 = min(
            b0 + args.batch_size,
            strict_ids.size,
        )

        ids = strict_ids[
            b0:b1
        ]

        B = ids.size

        X = np.empty(
            (
                B,
                nedge,
            ),
            dtype=np.float64,
        )

        for e in range(nedge):

            X[:, e] = (
                np.asarray(
                    phase_maps[e][ids],
                    dtype=np.float64,
                )
                +
                TWOPI
                *
                global_delta[e]
            )

        closure = X @ CT

        k = np.rint(
            closure / TWOPI
        ).astype(
            np.int32
        )

        float_residual = (
            closure
            -
            TWOPI * k
        )

        final_float_residual_max = max(
            final_float_residual_max,
            float(
                np.max(
                    np.abs(
                        float_residual
                    )
                )
            ),
        )

        bad = (
            k != 0
        )

        final_bad_occurrences += int(
            np.count_nonzero(
                bad
            )
        )

        final_bad_points += int(
            np.count_nonzero(
                np.any(
                    bad,
                    axis=1,
                )
            )
        )

        if (
            b0 == 0
            or
            b1 == strict_ids.size
            or
            (
                b1
                //
                args.batch_size
            )
            % 10
            == 0
        ):
            print(
                f"  {b1:,}/"
                f"{strict_ids.size:,}"
            )

    print()
    print("=" * 108)
    print(
        "FINAL STEP08 CANDIDATE QA"
    )
    print("=" * 108)

    print(
        f"strict valid points        : "
        f"{strict_ids.size:,}/"
        f"{npoint:,} "
        f"({100*strict_ids.size/npoint:.6f}%)"
    )

    print(
        f"temporal closure bad points: "
        f"{final_bad_points}"
    )

    print(
        f"closure bad occurrences    : "
        f"{final_bad_occurrences}"
    )

    print(
        f"float closure residual max : "
        f"{final_float_residual_max:.3e} rad"
    )

    print(
        f"residual closure float max : "
        f"{residual_float_residual_max:.3e} rad"
    )

    # ========================================================
    # Save masks and gauge
    # ========================================================

    np.save(
        outdir
        / "global_ifg_integer_delta.npy",
        global_delta.astype(
            np.int16
        ),
    )

    np.save(
        outdir
        / "all_ifg_registered_mask.npy",
        all_registered,
    )

    np.save(
        outdir
        / "temporal_integer_bad_mask.npy",
        temporal_bad,
    )

    np.save(
        outdir
        / "temporal_valid_mask.npy",
        temporal_valid,
    )

    np.save(
        outdir
        / "spatial_safe_conflict_point_mask.npy",
        spatial_safe_conflict,
    )

    np.save(
        outdir
        / "structural_incompatible_mask.npy",
        structural_incompatible,
    )

    np.save(
        outdir
        / "strict_unwrap_valid_mask.npy",
        strict_valid,
    )

    gauge_csv = (
        outdir
        / "global_ifg_integer_gauge.csv"
    )

    gauge_rows = []

    for e0, (ti, tj) in enumerate(
        temporal_edges
    ):

        gauge_rows.append({
            "pair_id":
                e0 + 1,

            "date1":
                str(
                    stack.dates[ti]
                ),

            "date2":
                str(
                    stack.dates[tj]
                ),

            "integer_delta":
                int(
                    global_delta[e0]
                ),

            "phase_delta_rad":
                float(
                    TWOPI
                    *
                    global_delta[e0]
                ),
        })

    with gauge_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                gauge_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            gauge_rows
        )

    flag_csv = (
        outdir
        / "excluded_or_flagged_points.csv"
    )

    flagged = np.where(
        (~strict_valid)
        |
        structural_incompatible
        |
        spatial_safe_conflict
    )[0]

    flag_rows = []

    for pid in flagged.tolist():

        reasons = []

        if not all_registered[pid]:
            reasons.append(
                "not_registered_all_ifgs"
            )

        if temporal_bad[pid]:
            reasons.append(
                "temporal_integer_inconsistent"
            )

        if spatial_safe_conflict[pid]:
            reasons.append(
                "safe_spatial_conflict_endpoint"
            )

        if structural_incompatible[pid]:
            reasons.append(
                "fragment_signature_incompatible"
            )

        flag_rows.append({
            "point_id":
                int(pid),

            "row":
                int(
                    rows_pt[pid]
                ),

            "col":
                int(
                    cols_pt[pid]
                ),

            "all_ifg_registered":
                int(
                    all_registered[pid]
                ),

            "temporal_integer_bad":
                int(
                    temporal_bad[pid]
                ),

            "safe_spatial_conflict":
                int(
                    spatial_safe_conflict[pid]
                ),

            "structural_incompatible":
                int(
                    structural_incompatible[pid]
                ),

            "temporal_valid":
                int(
                    temporal_valid[pid]
                ),

            "strict_valid":
                int(
                    strict_valid[pid]
                ),

            "reasons":
                ";".join(
                    reasons
                ),
        })

    with flag_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                flag_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            flag_rows
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-final-unwrap-candidate-v1.0",

        "status":
            (
                "PASS_CANDIDATE"
                if (
                    final_bad_points == 0
                    and
                    final_bad_occurrences == 0
                )
                else
                "REVIEW_REQUIRED"
            ),

        "points":
            int(npoint),

        "ifgs":
            int(nedge),

        "cycles":
            int(C.shape[0]),

        "global_integer_gauge": {
            "nonzero_ifgs":
                int(
                    np.count_nonzero(
                        global_delta
                    )
                ),

            "max_abs_integer_delta":
                int(
                    np.max(
                        np.abs(
                            global_delta
                        )
                    )
                ),

            "verified":
                bool(
                    np.array_equal(
                        gauge_check,
                        -cycle_mode,
                    )
                ),
        },

        "quality": {
            "all_ifg_registered":
                int(
                    np.count_nonzero(
                        all_registered
                    )
                ),

            "main_temporal_bad":
                int(
                    np.count_nonzero(
                        main_temporal_bad
                    )
                ),

            "residual_temporal_bad":
                int(
                    np.count_nonzero(
                        residual_temporal_bad
                    )
                ),

            "total_temporal_bad":
                int(
                    np.count_nonzero(
                        temporal_bad
                    )
                ),

            "safe_conflict_points":
                int(
                    np.count_nonzero(
                        spatial_safe_conflict
                    )
                ),

            "structural_incompatible":
                int(
                    np.count_nonzero(
                        structural_incompatible
                    )
                ),

            "temporal_valid":
                int(
                    np.count_nonzero(
                        temporal_valid
                    )
                ),

            "strict_valid":
                int(
                    np.count_nonzero(
                        strict_valid
                    )
                ),
        },

        "final_temporal_closure": {
            "bad_points":
                int(
                    final_bad_points
                ),

            "bad_occurrences":
                int(
                    final_bad_occurrences
                ),

            "float_residual_max_rad":
                float(
                    final_float_residual_max
                ),
        },

        "policy": {
            "integer_repair":
                "none",

            "temporal_inconsistent_points":
                "retained_in_point_stack_but_masked_from_timeseries",

            "safe_conflict_points":
                "retained_and_excluded_only_from_strict_quality_mask",

            "global_ifg_integer_gauge":
                (
                    "lazy_per_ifg_2pi_offset; "
                    "original IFG files unchanged"
                ),
        },
    }

    manifest_path = (
        outdir
        / "final_unwrap_candidate_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print(
        f"global gauge               : "
        f"{gauge_csv}"
    )

    print(
        f"flagged point table        : "
        f"{flag_csv}"
    )

    print(
        f"output directory           : "
        f"{outdir}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()

    if (
        final_bad_points == 0
        and
        final_bad_occurrences == 0
    ):

        print(
            "STEP 08x STATUS: PASS / "
            "FINAL UNWRAP CANDIDATE"
        )

        print(
            "No production IFG phase file "
            "has been rewritten."
        )

    else:

        print(
            "STEP 08x STATUS: REVIEW REQUIRED"
        )


if __name__ == "__main__":
    main()
