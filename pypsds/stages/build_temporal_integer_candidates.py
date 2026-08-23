#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import (
    milp,
    LinearConstraint,
    Bounds,
)

from pypsds.context import open_from_config


TWOPI = 2.0 * np.pi


def wrap(x):
    return np.arctan2(
        np.sin(x),
        np.cos(x),
    )


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


def read_batch_qa(path: Path):

    if not path.exists():
        return {}

    out = {}

    with path.open() as f:

        for r in csv.DictReader(f):

            pid = int(r["pair_id"])

            out[pid] = {
                "safe_internal_bad":
                    int(r["safe_internal_bad"]),

                "final_safe_bad":
                    int(r["final_safe_bad"]),

                "selected_non_exact":
                    int(r["selected_non_exact"]),

                "selected_ratio_lt_0p75":
                    int(
                        r[
                            "selected_ratio_lt_0p75"
                        ]
                    ),

                "rejected_cycle_outliers":
                    int(
                        r[
                            "rejected_cycle_outliers"
                        ]
                    ),

                "unsafe_within_bad":
                    int(r["unsafe_within_bad"]),

                "unsafe_cross_bad":
                    int(r["unsafe_cross_bad"]),
            }

    return out


def solve_sparse_integer(
    C,
    d,
):
    """
    min sum(z_e)

    subject to

        C x = d

        -z <= x <= z

        x integer
        z continuous >= 0
    """

    ncycle, nedge = C.shape

    max_d = int(
        np.max(
            np.abs(d)
        )
    )

    # Deliberately generous relative to observed
    # cycle discrepancy. This is NOT the assumed
    # physical correction magnitude.
    M = max(
        4,
        2 * max_d + 2,
    )

    # Variables:
    # [ x_0 ... x_E-1,
    #   z_0 ... z_E-1 ]
    nvar = 2 * nedge

    objective = np.zeros(
        nvar,
        dtype=np.float64,
    )

    objective[
        nedge:
    ] = 1.0

    integrality = np.zeros(
        nvar,
        dtype=np.int32,
    )

    integrality[
        :nedge
    ] = 1

    lb = np.empty(
        nvar,
        dtype=np.float64,
    )

    ub = np.empty(
        nvar,
        dtype=np.float64,
    )

    lb[
        :nedge
    ] = -M

    ub[
        :nedge
    ] = M

    lb[
        nedge:
    ] = 0.0

    ub[
        nedge:
    ] = float(M)

    bounds = Bounds(
        lb,
        ub,
    )

    # Equality:
    #
    # C x = d
    Aeq = np.zeros(
        (
            ncycle,
            nvar,
        ),
        dtype=np.float64,
    )

    Aeq[
        :,
        :nedge
    ] = C

    eq = LinearConstraint(
        Aeq,
        d.astype(
            np.float64
        ),
        d.astype(
            np.float64
        ),
    )

    # Absolute-value constraints:
    #
    #  x - z <= 0
    # -x - z <= 0

    I = np.eye(
        nedge,
        dtype=np.float64,
    )

    Aabs = np.zeros(
        (
            2 * nedge,
            nvar,
        ),
        dtype=np.float64,
    )

    Aabs[
        :nedge,
        :nedge
    ] = I

    Aabs[
        :nedge,
        nedge:
    ] = -I

    Aabs[
        nedge:,
        :nedge
    ] = -I

    Aabs[
        nedge:,
        nedge:
    ] = -I

    abs_constraint = (
        LinearConstraint(
            Aabs,
            -np.inf,
            np.zeros(
                2 * nedge,
                dtype=np.float64,
            ),
        )
    )

    res = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=[
            eq,
            abs_constraint,
        ],
        options={
            "presolve": True,
            "time_limit": 60.0,
        },
    )

    if not res.success:

        raise RuntimeError(
            "MILP failed: "
            f"{res.message}"
        )

    x = np.rint(
        res.x[
            :nedge
        ]
    ).astype(
        np.int32
    )

    lhs = (
        C.astype(
            np.int32
        )
        @
        x
    )

    if not np.array_equal(
        lhs,
        d,
    ):

        raise RuntimeError(
            "MILP integer equality "
            "verification failed."
        )

    return {
        "x":
            x,

        "objective":
            float(
                np.sum(
                    np.abs(x)
                )
            ),

        "nonzero":
            int(
                np.count_nonzero(
                    x
                )
            ),

        "max_abs":
            int(
                np.max(
                    np.abs(x)
                )
            ),

        "bound_M":
            int(M),

        "message":
            str(
                res.message
            ),
    }


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

    root = (
        Path(paths.output_dir)
        / "processing"
    )

    pps_dir = (
        root
        / "point_phase_stack"
    )

    network_dir = (
        root
        / "network"
    )

    unwrap_dir = (
        root
        / "single_ifg_robust_solution"
    )

    temporal_dir = (
        root
        / "temporal_integer_closure_quality"
    )

    batch_qa_path = (
        root
        / "batch_unwrap_validation"
        / "all_ifg_unwrap_qa.csv"
    )

    outdir = (
        root
        / "temporal_sparse_integer_candidate"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_pt = np.load(
        pps_dir
        / "rows.npy",
        mmap_mode="r",
    )

    cols_pt = np.load(
        pps_dir
        / "cols.npy",
        mmap_mode="r",
    )

    npoint = rows_pt.size

    ndate = len(
        stack.dates
    )

    temporal_edges = load_itab(
        network_dir
        / "network.itab",
        ndate,
    )

    nedge = len(
        temporal_edges
    )

    C = np.load(
        temporal_dir
        / "cycle_matrix.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    cycle_mode = np.load(
        temporal_dir
        / "cycle_modal_integer.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    point_bad_count = np.load(
        temporal_dir
        / "point_temporal_bad_cycle_count.npy",
        mmap_mode="r",
    )

    bad_point_ids = np.where(
        np.asarray(
            point_bad_count
        )
        >
        0
    )[0].astype(
        np.int32
    )

    nbad = bad_point_ids.size

    if nbad == 0:

        print(
            "No temporal bad points. "
            "Nothing to solve."
        )

        return

    if C.shape != (
        nedge - ndate + 1,
        nedge,
    ):

        raise RuntimeError(
            f"Unexpected cycle matrix "
            f"shape: {C.shape}"
        )

    if cycle_mode.size != C.shape[0]:

        raise RuntimeError(
            "cycle mode size mismatch"
        )

    # ========================================================
    # Load only 159 x 108 values.
    # No large IFG cube.
    # ========================================================

    X = np.empty(
        (
            nbad,
            nedge,
        ),
        dtype=np.float64,
    )

    phase_maps = []

    for pair_id, (i, j) in enumerate(
        temporal_edges,
        start=1,
    ):

        tag = (
            f"pair{pair_id:03d}_"
            f"{stack.dates[i]}_"
            f"{stack.dates[j]}"
        )

        path = (
            unwrap_dir
            / (
                f"{tag}_"
                "unwrapped_phase_rad.npy"
            )
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        arr = np.load(
            path,
            mmap_mode="r",
        )

        phase_maps.append(
            arr
        )

        X[
            :,
            pair_id - 1
        ] = np.asarray(
            arr[
                bad_point_ids
            ],
            dtype=np.float64,
        )

    # Current temporal closure integer.
    closure = (
        X
        @
        C.T.astype(
            np.float64,
            copy=False,
        )
    )

    current_k = np.rint(
        closure
        /
        TWOPI
    ).astype(
        np.int32
    )

    current_residual = (
        closure
        -
        TWOPI
        *
        current_k
    )

    # Desired:
    #
    # C x = cycle_mode - current_k
    D = (
        cycle_mode[
            None,
            :
        ]
        -
        current_k
    ).astype(
        np.int32
    )

    initial_bad_cycles = np.count_nonzero(
        D,
        axis=1,
    )

    # ========================================================
    # Group identical 71-D error patterns.
    # ========================================================

    (
        unique_D,
        inverse,
        pattern_counts,
    ) = np.unique(
        D,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    npattern = unique_D.shape[0]

    print("=" * 96)
    print(
        "Sparse temporal integer-correction "
        "candidate quality"
    )
    print("=" * 96)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"temporal acquisitions      : "
        f"{ndate}"
    )

    print(
        f"IFGs                       : "
        f"{nedge}"
    )

    print(
        f"cycle constraints          : "
        f"{C.shape[0]}"
    )

    print(
        f"temporal bad points        : "
        f"{nbad}"
    )

    print(
        f"unique closure-error "
        f"patterns : "
        f"{npattern}"
    )

    print(
        f"initial bad cycles "
        f"min/med/max:"
    )

    print(
        f"  {initial_bad_cycles.min()} / "
        f"{np.median(initial_bad_cycles):.1f} / "
        f"{initial_bad_cycles.max()}"
    )

    print(
        f"current closure float "
        f"residual max:"
    )

    print(
        f"  "
        f"{np.max(np.abs(current_residual)):.3e} rad"
    )

    # ========================================================
    # Solve each unique pattern once.
    # ========================================================

    pattern_solutions = []
    solution_matrix = np.zeros(
        (
            npattern,
            nedge,
        ),
        dtype=np.int32,
    )

    print()
    print(
        "Solving unique temporal integer patterns ..."
    )

    for pattern_id in range(
        npattern
    ):

        d = unique_D[
            pattern_id
        ]

        sol = solve_sparse_integer(
            C,
            d,
        )

        solution_matrix[
            pattern_id
        ] = sol["x"]

        points_here = int(
            pattern_counts[
                pattern_id
            ]
        )

        pattern_solutions.append({
            "pattern_id":
                pattern_id,

            "point_count":
                points_here,

            "bad_cycles":
                int(
                    np.count_nonzero(
                        d
                    )
                ),

            "max_abs_cycle_target":
                int(
                    np.max(
                        np.abs(d)
                    )
                ),

            "minimum_L1":
                sol["objective"],

            "corrected_ifgs":
                sol["nonzero"],

            "max_abs_ifg_integer_delta":
                sol["max_abs"],

            "integer_bound":
                sol["bound_M"],
        })

        print(
            f"  pattern "
            f"{pattern_id+1:3d}/"
            f"{npattern:3d}: "
            f"points={points_here:3d}, "
            f"badcycles="
            f"{np.count_nonzero(d):2d}, "
            f"correctIFGs="
            f"{sol['nonzero']:2d}, "
            f"L1="
            f"{sol['objective']:.0f}, "
            f"max|dx|="
            f"{sol['max_abs']}"
        )

    # ========================================================
    # Expand candidate corrections to the 159 points.
    # ========================================================

    Xcorr_integer = (
        solution_matrix[
            inverse
        ]
    )

    # Exact cycle verification.
    D_check = (
        Xcorr_integer
        @
        C.T
    )

    if not np.array_equal(
        D_check,
        D,
    ):

        raise RuntimeError(
            "Expanded correction matrix "
            "does not satisfy cycle equations."
        )

    Xcorr = (
        X
        +
        TWOPI
        *
        Xcorr_integer.astype(
            np.float64
        )
    )

    corrected_closure = (
        Xcorr
        @
        C.T.astype(
            np.float64,
            copy=False,
        )
    )

    corrected_k = np.rint(
        corrected_closure
        /
        TWOPI
    ).astype(
        np.int32
    )

    corrected_residual = (
        corrected_closure
        -
        TWOPI
        *
        corrected_k
    )

    corrected_bad = (
        corrected_k
        !=
        cycle_mode[
            None,
            :
        ]
    )

    corrected_bad_occ = int(
        np.count_nonzero(
            corrected_bad
        )
    )

    corrected_bad_points = int(
        np.count_nonzero(
            np.any(
                corrected_bad,
                axis=1,
            )
        )
    )

    # Wrapped parity.
    parity = np.abs(
        wrap(
            Xcorr
            -
            X
        )
    )

    parity_max = float(
        parity.max()
    )

    # ========================================================
    # Sparse correction table
    # ========================================================

    batch_qa = read_batch_qa(
        batch_qa_path
    )

    sparse_rows = []

    per_ifg_count = np.zeros(
        nedge,
        dtype=np.int32,
    )

    per_ifg_abs_sum = np.zeros(
        nedge,
        dtype=np.int32,
    )

    per_ifg_max_abs = np.zeros(
        nedge,
        dtype=np.int32,
    )

    per_point_nonzero = np.count_nonzero(
        Xcorr_integer,
        axis=1,
    )

    per_point_l1 = np.sum(
        np.abs(
            Xcorr_integer
        ),
        axis=1,
    )

    for q in range(
        nbad
    ):

        pid = int(
            bad_point_ids[q]
        )

        nz = np.where(
            Xcorr_integer[
                q
            ]
            !=
            0
        )[0]

        for eid0 in nz.tolist():

            delta = int(
                Xcorr_integer[
                    q,
                    eid0
                ]
            )

            pair_id = eid0 + 1

            i, j = temporal_edges[
                eid0
            ]

            per_ifg_count[
                eid0
            ] += 1

            per_ifg_abs_sum[
                eid0
            ] += abs(
                delta
            )

            per_ifg_max_abs[
                eid0
            ] = max(
                per_ifg_max_abs[
                    eid0
                ],
                abs(delta),
            )

            sparse_rows.append({
                "point_id":
                    pid,

                "row":
                    int(
                        rows_pt[
                            pid
                        ]
                    ),

                "col":
                    int(
                        cols_pt[
                            pid
                        ]
                    ),

                "pattern_id":
                    int(
                        inverse[q]
                    ),

                "initial_bad_cycle_count":
                    int(
                        initial_bad_cycles[
                            q
                        ]
                    ),

                "candidate_corrected_ifgs_for_point":
                    int(
                        per_point_nonzero[
                            q
                        ]
                    ),

                "candidate_L1_for_point":
                    int(
                        per_point_l1[
                            q
                        ]
                    ),

                "pair_id":
                    pair_id,

                "date1":
                    str(
                        stack.dates[
                            i
                        ]
                    ),

                "date2":
                    str(
                        stack.dates[
                            j
                        ]
                    ),

                "integer_delta":
                    delta,

                "phase_delta_rad":
                    float(
                        TWOPI
                        *
                        delta
                    ),
            })

    # ========================================================
    # Per-IFG candidate correction localization
    # ========================================================

    ifg_rows = []

    for eid0 in range(
        nedge
    ):

        if (
            per_ifg_count[
                eid0
            ]
            ==
            0
        ):
            continue

        pair_id = (
            eid0 + 1
        )

        i, j = temporal_edges[
            eid0
        ]

        qa = batch_qa.get(
            pair_id,
            {}
        )

        ifg_rows.append({
            "pair_id":
                pair_id,

            "date1":
                str(
                    stack.dates[i]
                ),

            "date2":
                str(
                    stack.dates[j]
                ),

            "candidate_point_corrections":
                int(
                    per_ifg_count[
                        eid0
                    ]
                ),

            "candidate_sum_abs_integer_delta":
                int(
                    per_ifg_abs_sum[
                        eid0
                    ]
                ),

            "candidate_max_abs_integer_delta":
                int(
                    per_ifg_max_abs[
                        eid0
                    ]
                ),

            "qa_safe_internal_bad":
                qa.get(
                    "safe_internal_bad",
                    -1,
                ),

            "qa_selected_non_exact":
                qa.get(
                    "selected_non_exact",
                    -1,
                ),

            "qa_selected_ratio_lt_0p75":
                qa.get(
                    "selected_ratio_lt_0p75",
                    -1,
                ),

            "qa_rejected_cycle_outliers":
                qa.get(
                    "rejected_cycle_outliers",
                    -1,
                ),

            "qa_unsafe_within_bad":
                qa.get(
                    "unsafe_within_bad",
                    -1,
                ),

            "qa_unsafe_cross_bad":
                qa.get(
                    "unsafe_cross_bad",
                    -1,
                ),
        })

    ifg_rows.sort(
        key=lambda r: (
            r[
                "candidate_point_corrections"
            ],
            r[
                "candidate_sum_abs_integer_delta"
            ],
        ),
        reverse=True,
    )

    # ========================================================
    # Summary
    # ========================================================

    total_entries = int(
        np.count_nonzero(
            Xcorr_integer
        )
    )

    total_l1 = int(
        np.sum(
            np.abs(
                Xcorr_integer
            )
        )
    )

    max_abs_delta = int(
        np.max(
            np.abs(
                Xcorr_integer
            )
        )
    )

    print()
    print("=" * 96)
    print(
        "Sparse temporal candidate summary"
    )
    print("=" * 96)

    print(
        f"bad points                 : "
        f"{nbad}"
    )

    print(
        f"unique error patterns      : "
        f"{npattern}"
    )

    print(
        f"candidate correction "
        f"entries  : "
        f"{total_entries}"
    )

    print(
        f"corrected IFGs per point "
        f"min/med/max:"
    )

    print(
        f"  "
        f"{per_point_nonzero.min()} / "
        f"{np.median(per_point_nonzero):.1f} / "
        f"{per_point_nonzero.max()}"
    )

    print(
        f"L1 integer correction "
        f"per point min/med/max:"
    )

    print(
        f"  "
        f"{per_point_l1.min()} / "
        f"{np.median(per_point_l1):.1f} / "
        f"{per_point_l1.max()}"
    )

    print(
        f"total |integer delta|      : "
        f"{total_l1}"
    )

    print(
        f"maximum |integer delta|    : "
        f"{max_abs_delta}"
    )

    print(
        f"post-correction bad "
        f"cycle occurrences:"
    )

    print(
        f"  {corrected_bad_occ}"
    )

    print(
        f"post-correction bad points : "
        f"{corrected_bad_points}"
    )

    print(
        f"post-correction float "
        f"residual max:"
    )

    print(
        f"  "
        f"{np.max(np.abs(corrected_residual)):.3e} rad"
    )

    print(
        f"wrapped parity max error   : "
        f"{parity_max:.3e} rad"
    )

    print()
    print(
        "Top IFGs selected by minimum-L1 candidate:"
    )

    print(
        " pair  dates                  "
        "points  sum|dx| max|dx| "
        "safeBad nonExact <0.75"
    )

    for r in ifg_rows[:20]:

        print(
            f" {r['pair_id']:4d}  "
            f"{r['date1']}->{r['date2']} "
            f"{r['candidate_point_corrections']:6d} "
            f"{r['candidate_sum_abs_integer_delta']:8d} "
            f"{r['candidate_max_abs_integer_delta']:7d} "
            f"{r['qa_safe_internal_bad']:7d} "
            f"{r['qa_selected_non_exact']:8d} "
            f"{r['qa_selected_ratio_lt_0p75']:6d}"
        )

    # ========================================================
    # Save
    # ========================================================

    pattern_csv = (
        outdir
        / "temporal_error_patterns.csv"
    )

    with pattern_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                pattern_solutions[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            pattern_solutions
        )

    sparse_csv = (
        outdir
        / "candidate_sparse_integer_corrections.csv"
    )

    with sparse_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                sparse_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            sparse_rows
        )

    ifg_csv = (
        outdir
        / "candidate_corrections_by_ifg.csv"
    )

    with ifg_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                ifg_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            ifg_rows
        )

    np.save(
        outdir
        / "bad_point_ids.npy",
        bad_point_ids,
    )

    np.save(
        outdir
        / "point_integer_correction_matrix.npy",
        Xcorr_integer.astype(
            np.int16
        ),
    )

    np.save(
        outdir
        / "unique_error_patterns.npy",
        unique_D.astype(
            np.int16
        ),
    )

    np.save(
        outdir
        / "pattern_integer_solutions.npy",
        solution_matrix.astype(
            np.int16
        ),
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-temporal-sparse-integer-candidate-v1.0",

        "status":
            "CANDIDATE_ONLY_NOT_APPLIED",

        "bad_points":
            int(
                nbad
            ),

        "unique_error_patterns":
            int(
                npattern
            ),

        "candidate_correction_entries":
            int(
                total_entries
            ),

        "total_abs_integer_delta":
            int(
                total_l1
            ),

        "max_abs_integer_delta":
            int(
                max_abs_delta
            ),

        "corrected_ifgs_per_point": {
            "min":
                int(
                    per_point_nonzero.min()
                ),

            "median":
                float(
                    np.median(
                        per_point_nonzero
                    )
                ),

            "max":
                int(
                    per_point_nonzero.max()
                ),
        },

        "post_candidate_temporal_qa": {
            "bad_cycle_occurrences":
                int(
                    corrected_bad_occ
                ),

            "bad_points":
                int(
                    corrected_bad_points
                ),

            "float_residual_max_rad":
                float(
                    np.max(
                        np.abs(
                            corrected_residual
                        )
                    )
                ),

            "wrapped_parity_max_error_rad":
                float(
                    parity_max
                ),
        },

        "note":
            (
                "Minimum-L1 integer corrections "
                "satisfying temporal cycle modes. "
                "Candidates have NOT been applied "
                "to production IFG phases. "
                "Spatial validation is required."
            ),
    }

    manifest_path = (
        outdir
        / "temporal_sparse_integer_candidate.json"
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
        f"pattern table              : "
        f"{pattern_csv}"
    )

    print(
        f"sparse candidate table     : "
        f"{sparse_csv}"
    )

    print(
        f"IFG candidate summary      : "
        f"{ifg_csv}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP temporal_integer_candidate STATUS: PASS / "
        "SPARSE INTEGER CANDIDATE ONLY"
    )

    print(
        "No unwrapped IFG file has been modified."
    )


if __name__ == "__main__":
    main()
