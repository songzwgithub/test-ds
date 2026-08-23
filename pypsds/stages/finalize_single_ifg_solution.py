#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


TWOPI = 2.0 * np.pi


def wrap(x):
    return np.arctan2(
        np.sin(x),
        np.cos(x),
    )


def load_itab(path: Path, ndate: int):
    out = []

    for raw in path.read_text().splitlines():

        f = raw.split()

        if len(f) < 2:
            continue

        i = int(f[0]) - 1
        j = int(f[1]) - 1

        if not (
            0 <= i < ndate
            and
            0 <= j < ndate
        ):
            raise RuntimeError(
                f"Invalid ITAB line: {raw}"
            )

        out.append((i, j))

    return out


def read_groups(path: Path):

    rows = []

    with path.open() as f:

        for gid, r in enumerate(
            csv.DictReader(f)
        ):

            rows.append({
                "group_id":
                    gid,

                "fragment_a":
                    int(
                        r["fragment_a"]
                    ),

                "fragment_b":
                    int(
                        r["fragment_b"]
                    ),

                "mode_shift":
                    int(
                        r[
                            "mode_shift_b_minus_a"
                        ]
                    ),

                "edge_count":
                    int(
                        r["edge_count"]
                    ),

                "mode_count":
                    int(
                        r["mode_count"]
                    ),

                "consensus_ratio":
                    float(
                        r["consensus_ratio"]
                    ),

                "exact_consensus":
                    int(
                        r["exact_consensus"]
                    ),

                "median_abs_gradient_rad":
                    float(
                        r[
                            "median_abs_gradient_rad"
                        ]
                    ),

                "median_distance_m":
                    float(
                        r[
                            "median_distance_m"
                        ]
                    ),
            })

    return rows


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--pair-id",
        type=int,
        default=19,
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

    pps = (
        root
        / "point_phase_stack"
    )

    network = (
        root
        / "network"
    )

    graph = (
        root
        / "spatial_graph"
    )

    policy = (
        root
        / "unwrap_component_policy"
    )

    qualitysafe_fragment_quality = (
        root
        / "safe_fragment_integer_quality"
    )

    outdir = (
        root
        / "single_ifg_robust_solution"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase = np.load(
        pps
        / "phase_rad.npy",
        mmap_mode="r",
    )

    npoint, ndate = phase.shape

    temporal_edges = load_itab(
        network
        / "network.itab",
        ndate,
    )

    pair_id = args.pair_id

    if not (
        1
        <= pair_id
        <= len(temporal_edges)
    ):
        raise RuntimeError(
            "Invalid pair ID."
        )

    ti, tj = temporal_edges[
        pair_id - 1
    ]

    tag = (
        f"pair{pair_id:03d}_"
        f"{stack.dates[ti]}_"
        f"{stack.dates[tj]}"
    )

    # --------------------------------------------------------
    # safe_fragment_quality robust local-component solution
    # --------------------------------------------------------

    U = np.load(
        qualitysafe_fragment_quality
        / f"{tag}_consensus_unwrapped.npy",
    ).astype(
        np.float64,
        copy=True,
    )

    safe_fragment = np.load(
        qualitysafe_fragment_quality
        / f"{tag}_safe_fragment.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    fragment_shift = np.load(
        qualitysafe_fragment_quality
        / f"{tag}_fragment_shift.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    groups = read_groups(
        qualitysafe_fragment_quality
        / f"{tag}_fragment_pair_consensus.csv"
    )

    # --------------------------------------------------------
    # Local component / anchor policy
    # --------------------------------------------------------

    local_component = np.load(
        policy
        / "local_component.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    point_tier = np.load(
        policy
        / "point_unwrap_tier.npy",
    ).astype(
        np.uint8,
        copy=False,
    )

    anchor_component = np.load(
        policy
        / "anchor_component.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    anchor_u = np.load(
        graph
        / "anchor_u.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    anchor_v = np.load(
        graph
        / "anchor_v.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    anchor_class = np.load(
        graph
        / "anchor_class.npy",
    ).astype(
        np.uint8,
        copy=False,
    )

    local_u = np.load(
        graph
        / "local_u.npy",
        mmap_mode="r",
    )

    local_v = np.load(
        graph
        / "local_v.npy",
        mmap_mode="r",
    )

    # --------------------------------------------------------
    # Wrapped IFG
    # --------------------------------------------------------

    ifg = wrap(
        np.asarray(
            phase[:, tj],
            dtype=np.float64,
        )
        -
        np.asarray(
            phase[:, ti],
            dtype=np.float64,
        )
    )

    # ========================================================
    # 1. Fragment-pair global consistency
    # ========================================================

    accepted_groups = []
    rejected_groups = []

    for r in groups:

        a = r[
            "fragment_a"
        ]

        b = r[
            "fragment_b"
        ]

        predicted = int(
            fragment_shift[b]
            -
            fragment_shift[a]
        )

        observed = r[
            "mode_shift"
        ]

        residual = (
            observed
            -
            predicted
        )

        rr = dict(r)

        rr[
            "predicted_shift"
        ] = predicted

        rr[
            "integer_residual"
        ] = residual

        rr[
            "status"
        ] = (
            "accepted_consistent"
            if residual == 0
            else
            "rejected_cycle_outlier"
        )

        if residual == 0:
            accepted_groups.append(
                rr
            )
        else:
            rejected_groups.append(
                rr
            )

    # ========================================================
    # 2. Verify local spatial-edge consistency
    # ========================================================

    u = np.asarray(
        local_u,
        dtype=np.int32,
    )

    v = np.asarray(
        local_v,
        dtype=np.int32,
    )

    g = wrap(
        ifg[v]
        -
        ifg[u]
    )

    delta = (
        U[v]
        -
        U[u]
        -
        g
    )

    jump = np.rint(
        delta
        /
        TWOPI
    ).astype(
        np.int32
    )

    safe = (
        np.abs(g)
        <=
        np.pi / 2
    )

    same_safe_fragment = (
        safe_fragment[u]
        ==
        safe_fragment[v]
    )

    bad = (
        jump != 0
    )

    safe_bad = int(
        np.count_nonzero(
            safe
            &
            bad
        )
    )

    unsafe_within_bad = int(
        np.count_nonzero(
            (~safe)
            &
            same_safe_fragment
            &
            bad
        )
    )

    unsafe_cross_bad = int(
        np.count_nonzero(
            (~safe)
            &
            (~same_safe_fragment)
            &
            bad
        )
    )

    # ========================================================
    # 3. Residual-component registration to main component
    # ========================================================

    comp_ids, comp_counts = np.unique(
        local_component,
        return_counts=True,
    )

    main_component = int(
        comp_ids[
            np.argmax(
                comp_counts
            )
        ]
    )

    main_mask = (
        local_component
        ==
        main_component
    )

    registered = np.zeros(
        npoint,
        dtype=bool,
    )

    registered[
        main_mask
    ] = True

    residual_components = [
        int(x)
        for x in comp_ids
        if int(x)
        !=
        main_component
    ]

    registration_rows = []

    conflict_components = []

    tier_components = {
        1: 0,
        2: 0,
        3: 0,
    }

    tier_conflicts = {
        1: 0,
        2: 0,
        3: 0,
    }

    tier_registered_points = {
        1: 0,
        2: 0,
        3: 0,
    }

    for comp in residual_components:

        pids = np.where(
            local_component
            ==
            comp
        )[0]

        tier = int(
            point_tier[
                pids[0]
            ]
        )

        tier_components[
            tier
        ] += 1

        aids = np.where(
            anchor_component
            ==
            comp
        )[0]

        if aids.size != 2:

            raise RuntimeError(
                f"Component {comp} "
                f"has {aids.size} anchors."
            )

        shifts = []

        details = []

        for aid in aids:

            a = int(
                anchor_u[
                    aid
                ]
            )

            b = int(
                anchor_v[
                    aid
                ]
            )

            if (
                local_component[a]
                ==
                main_component
            ):

                main_p = a
                residual_p = b

            elif (
                local_component[b]
                ==
                main_component
            ):

                main_p = b
                residual_p = a

            else:

                raise RuntimeError(
                    "Invalid residual anchor."
                )

            g_anchor = float(
                wrap(
                    ifg[
                        main_p
                    ]
                    -
                    ifg[
                        residual_p
                    ]
                )
            )

            raw_shift = (
                U[
                    main_p
                ]
                -
                U[
                    residual_p
                ]
                -
                g_anchor
            ) / TWOPI

            nshift = int(
                np.rint(
                    raw_shift
                )
            )

            residual_rad = (
                U[
                    main_p
                ]
                -
                (
                    U[
                        residual_p
                    ]
                    +
                    TWOPI
                    *
                    nshift
                )
                -
                g_anchor
            )

            shifts.append(
                nshift
            )

            details.append(
                {
                    "anchor_id":
                        int(aid),

                    "class":
                        int(
                            anchor_class[
                                aid
                            ]
                        ),

                    "shift":
                        nshift,

                    "residual_rad":
                        float(
                            residual_rad
                        ),
                }
            )

        agree = (
            shifts[0]
            ==
            shifts[1]
        )

        applied_shift = 0

        if agree:

            applied_shift = (
                shifts[0]
            )

            U[
                pids
            ] += (
                TWOPI
                *
                applied_shift
            )

            registered[
                pids
            ] = True

            tier_registered_points[
                tier
            ] += int(
                pids.size
            )

        else:

            conflict_components.append(
                comp
            )

            tier_conflicts[
                tier
            ] += 1

        registration_rows.append({
            "component_id":
                comp,

            "point_count":
                int(
                    pids.size
                ),

            "tier":
                tier,

            "anchor1_shift":
                details[0][
                    "shift"
                ],

            "anchor1_class":
                details[0][
                    "class"
                ],

            "anchor1_residual_rad":
                details[0][
                    "residual_rad"
                ],

            "anchor2_shift":
                details[1][
                    "shift"
                ],

            "anchor2_class":
                details[1][
                    "class"
                ],

            "anchor2_residual_rad":
                details[1][
                    "residual_rad"
                ],

            "anchors_agree":
                int(
                    agree
                ),

            "applied_shift":
                int(
                    applied_shift
                ),
        })

    registered_points = int(
        registered.sum()
    )

    # ========================================================
    # 4. Final modulo parity
    # ========================================================

    wrap_error = np.abs(
        wrap(
            U
            -
            ifg
        )
    )

    max_wrap_error = float(
        wrap_error.max()
    )

    # ========================================================
    # Print
    # ========================================================

    print("=" * 92)
    print(
        "Final robust single-IFG candidate"
    )
    print("=" * 92)

    print(
        f"config                    : "
        f"{config_path}"
    )

    print(
        f"pair                      : "
        f"{pair_id}/"
        f"{len(temporal_edges)}"
    )

    print(
        f"dates                     : "
        f"{stack.dates[ti]} -> "
        f"{stack.dates[tj]}"
    )

    print()
    print("=" * 92)
    print(
        "Fragment-pair integer constraints"
    )
    print("=" * 92)

    print(
        f"total groups              : "
        f"{len(groups)}"
    )

    print(
        f"accepted consistent       : "
        f"{len(accepted_groups)}"
    )

    print(
        f"rejected cycle outliers   : "
        f"{len(rejected_groups)}"
    )

    for r in rejected_groups:

        print(
            f"  group {r['group_id']:3d}: "
            f"{r['fragment_a']}-"
            f"{r['fragment_b']}, "
            f"observed={r['mode_shift']}, "
            f"predicted={r['predicted_shift']}, "
            f"support="
            f"{r['mode_count']}/"
            f"{r['edge_count']}, "
            f"ratio="
            f"{r['consensus_ratio']:.3f}"
        )

    print()
    print("=" * 92)
    print(
        "Local-edge QA after robust fragment solution"
    )
    print("=" * 92)

    print(
        f"SAFE bad                  : "
        f"{safe_bad}"
    )

    print(
        f"UNSAFE within bad         : "
        f"{unsafe_within_bad}"
    )

    print(
        f"UNSAFE cross bad          : "
        f"{unsafe_cross_bad}"
    )

    print()
    print("=" * 92)
    print(
        "Residual two-anchor registration"
    )
    print("=" * 92)

    print(
        f"residual components       : "
        f"{len(residual_components)}"
    )

    print(
        f"anchor-agree components   : "
        f"{len(residual_components)-len(conflict_components)}"
    )

    print(
        f"anchor-conflict components: "
        f"{len(conflict_components)}"
    )

    print(
        f"registered points         : "
        f"{registered_points:,}/"
        f"{npoint:,} "
        f"({100*registered_points/npoint:.5f}%)"
    )

    for tier, name in (
        (1, "normal"),
        (2, "extended"),
        (3, "long"),
    ):

        print(
            f"{name:8s}: "
            f"components="
            f"{tier_components[tier]:3d}, "
            f"conflicts="
            f"{tier_conflicts[tier]:3d}, "
            f"registered="
            f"{tier_registered_points[tier]:4d}"
        )

    print()

    print(
        f"wrap-back max error       : "
        f"{max_wrap_error:.3e} rad"
    )

    # ========================================================
    # Save
    # ========================================================

    np.save(
        outdir
        / f"{tag}_unwrapped_phase_rad.npy",
        U.astype(
            np.float32
        ),
    )

    np.save(
        outdir
        / f"{tag}_registered_mask.npy",
        registered,
    )

    group_csv = (
        outdir
        / f"{tag}_fragment_constraint_status.csv"
    )

    all_groups = (
        accepted_groups
        +
        rejected_groups
    )

    all_groups.sort(
        key=lambda r:
        r["group_id"]
    )

    with group_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                all_groups[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            all_groups
        )

    reg_csv = (
        outdir
        / f"{tag}_residual_registration.csv"
    )

    with reg_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                registration_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            registration_rows
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-robust-single-ifg-candidate-v1.0",

        "status":
            "CANDIDATE_NOT_BATCH_FROZEN",

        "pair": {
            "pair_id":
                pair_id,

            "date1":
                str(
                    stack.dates[
                        ti
                    ]
                ),

            "date2":
                str(
                    stack.dates[
                        tj
                    ]
                ),
        },

        "fragment_constraints": {
            "total":
                len(groups),

            "accepted":
                len(
                    accepted_groups
                ),

            "rejected_cycle_outliers":
                len(
                    rejected_groups
                ),

            "rejected_group_ids":
                [
                    r[
                        "group_id"
                    ]
                    for r in
                    rejected_groups
                ],
        },

        "local_edge_qa": {
            "safe_bad":
                safe_bad,

            "unsafe_within_bad":
                unsafe_within_bad,

            "unsafe_cross_bad":
                unsafe_cross_bad,
        },

        "residual_registration": {
            "components":
                len(
                    residual_components
                ),

            "agree":
                (
                    len(
                        residual_components
                    )
                    -
                    len(
                        conflict_components
                    )
                ),

            "conflict":
                len(
                    conflict_components
                ),

            "registered_points":
                registered_points,

            "registered_fraction":
                float(
                    registered_points
                    /
                    npoint
                ),
        },

        "wrap_back_max_error_rad":
            max_wrap_error,
    }

    manifest_path = (
        outdir
        / f"{tag}_manifest.json"
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
        f"fragment status          : "
        f"{group_csv}"
    )

    print(
        f"residual registration    : "
        f"{reg_csv}"
    )

    print(
        f"manifest                 : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP single_ifg_solution STATUS: PASS / "
        "SINGLE-IFG ROBUST CANDIDATE"
    )

    print(
        "Do not batch-run temporal-network IFGs yet."
    )


if __name__ == "__main__":
    main()
