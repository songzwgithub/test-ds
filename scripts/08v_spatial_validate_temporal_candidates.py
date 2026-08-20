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

            edges.append(
                (i, j)
            )

    return edges


def read_batch_qa(path: Path):

    out = {}

    if not path.exists():
        return out

    with path.open() as f:

        for r in csv.DictReader(f):

            pid = int(
                r["pair_id"]
            )

            out[pid] = {
                "safe_internal_bad":
                    int(
                        r[
                            "safe_internal_bad"
                        ]
                    ),

                "final_safe_bad":
                    int(
                        r[
                            "final_safe_bad"
                        ]
                    ),

                "selected_non_exact":
                    int(
                        r[
                            "selected_non_exact"
                        ]
                    ),

                "selected_ratio_lt_0p75":
                    int(
                        r[
                            "selected_ratio_lt_0p75"
                        ]
                    ),
            }

    return out


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

    frag_dir = (
        root
        / "safe_fragment_integer_quality"
    )

    candidate_dir = (
        root
        / "temporal_sparse_integer_candidate"
    )

    batch_qa_path = (
        root
        / "batch_unwrap_validation"
        / "all_ifg_unwrap_qa.csv"
    )

    outdir = (
        root
        / "temporal_candidate_spatial_validation"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Inputs
    # ========================================================

    phase = np.load(
        pps_dir
        / "phase_rad.npy",
        mmap_mode="r",
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

    local_u = np.load(
        graph_dir
        / "local_u.npy",
        mmap_mode="r",
    )

    local_v = np.load(
        graph_dir
        / "local_v.npy",
        mmap_mode="r",
    )

    bad_point_ids = np.load(
        candidate_dir
        / "bad_point_ids.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    correction = np.load(
        candidate_dir
        / "point_integer_correction_matrix.npy",
    ).astype(
        np.int16,
        copy=False,
    )

    npoint, ndate = phase.shape

    temporal_edges = load_itab(
        network_dir
        / "network.itab",
        ndate,
    )

    nedge_t = len(
        temporal_edges
    )

    nbad = bad_point_ids.size

    if correction.shape != (
        nbad,
        nedge_t,
    ):
        raise RuntimeError(
            f"Correction matrix mismatch: "
            f"{correction.shape}"
        )

    batch_qa = read_batch_qa(
        batch_qa_path
    )

    # ========================================================
    # Build spatial incident-edge subset once.
    #
    # Only spatial edges touching one of the 159 temporal
    # bad points can possibly change.
    # ========================================================

    u_all = np.asarray(
        local_u,
        dtype=np.int32,
    )

    v_all = np.asarray(
        local_v,
        dtype=np.int32,
    )

    is_temporal_bad_point = np.zeros(
        npoint,
        dtype=bool,
    )

    is_temporal_bad_point[
        bad_point_ids
    ] = True

    incident_mask = (
        is_temporal_bad_point[
            u_all
        ]
        |
        is_temporal_bad_point[
            v_all
        ]
    )

    incident_ids = np.where(
        incident_mask
    )[0].astype(
        np.int64
    )

    iu = u_all[
        incident_ids
    ]

    iv = v_all[
        incident_ids
    ]

    # Map production point ID -> row in the
    # 159 x 108 candidate matrix.
    bad_index = np.full(
        npoint,
        -1,
        dtype=np.int32,
    )

    bad_index[
        bad_point_ids
    ] = np.arange(
        nbad,
        dtype=np.int32,
    )

    iu_idx = bad_index[
        iu
    ]

    iv_idx = bad_index[
        iv
    ]

    iu_is_bad = (
        iu_idx >= 0
    )

    iv_is_bad = (
        iv_idx >= 0
    )

    print("=" * 112)
    print(
        "Step 08v - Spatial validation of sparse "
        "temporal integer candidates"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"points                     : "
        f"{npoint:,}"
    )

    print(
        f"temporal bad points        : "
        f"{nbad}"
    )

    print(
        f"IFGs                       : "
        f"{nedge_t}"
    )

    print(
        f"candidate entries          : "
        f"{np.count_nonzero(correction):,}"
    )

    print(
        f"local spatial edges        : "
        f"{u_all.size:,}"
    )

    print(
        f"edges incident to 159 pts  : "
        f"{incident_ids.size:,}"
    )

    # ========================================================
    # Global counters
    # ========================================================

    pair_rows = []
    fragment_rows = []

    total_candidate_entries = 0

    total_fragment_groups = 0
    total_full_fragment_groups = 0
    total_partial_fragment_groups = 0
    total_mixed_delta_groups = 0

    total_entries_full_fragment = 0
    total_entries_partial_fragment = 0
    total_entries_singleton = 0

    total_boundary_edges = 0

    total_safe_boundary = 0
    total_safe_before_bad = 0
    total_safe_after_bad = 0
    total_safe_resolved = 0
    total_safe_created = 0

    total_unsafe_boundary = 0
    total_unsafe_before_bad = 0
    total_unsafe_after_bad = 0
    total_unsafe_resolved = 0
    total_unsafe_created = 0

    ifgs_touched = 0
    ifgs_safe_created = 0
    ifgs_safe_improved = 0

    # ========================================================
    # Process only IFGs touched by the temporal candidate
    # ========================================================

    for e0 in range(
        nedge_t
    ):

        dx_bad = correction[
            :,
            e0
        ].astype(
            np.int32,
            copy=False,
        )

        nz_bad_idx = np.where(
            dx_bad != 0
        )[0]

        if nz_bad_idx.size == 0:
            continue

        ifgs_touched += 1

        pair_id = e0 + 1

        ti, tj = temporal_edges[
            e0
        ]

        tag = (
            f"pair{pair_id:03d}_"
            f"{stack.dates[ti]}_"
            f"{stack.dates[tj]}"
        )

        safe_fragment_path = (
            frag_dir
            / f"{tag}_safe_fragment.npy"
        )

        unwrap_path = (
            unwrap_dir
            / (
                f"{tag}_"
                "unwrapped_phase_rad.npy"
            )
        )

        if not safe_fragment_path.exists():
            raise FileNotFoundError(
                safe_fragment_path
            )

        if not unwrap_path.exists():
            raise FileNotFoundError(
                unwrap_path
            )

        safe_fragment = np.load(
            safe_fragment_path,
            mmap_mode="r",
        )

        U = np.load(
            unwrap_path,
            mmap_mode="r",
        )

        if safe_fragment.size != npoint:
            raise RuntimeError(
                f"{tag}: fragment size mismatch"
            )

        # ====================================================
        # A. Fragment-coherence quality
        # ====================================================

        corrected_points = bad_point_ids[
            nz_bad_idx
        ]

        corrected_delta = dx_bad[
            nz_bad_idx
        ]

        corrected_frag = np.asarray(
            safe_fragment[
                corrected_points
            ],
            dtype=np.int32,
        )

        frag_sizes = np.bincount(
            np.asarray(
                safe_fragment,
                dtype=np.int32,
            )
        )

        pair_full_fragment_groups = 0
        pair_partial_groups = 0
        pair_mixed_groups = 0

        pair_full_entries = 0
        pair_partial_entries = 0
        pair_singleton_entries = 0

        unique_frags = np.unique(
            corrected_frag
        )

        for frag in unique_frags.tolist():

            m = (
                corrected_frag
                ==
                frag
            )

            pids = corrected_points[
                m
            ]

            deltas = corrected_delta[
                m
            ]

            frag_size = int(
                frag_sizes[
                    frag
                ]
            )

            ncorrected = int(
                pids.size
            )

            unique_delta = np.unique(
                deltas
            )

            same_nonzero_delta = (
                unique_delta.size == 1
            )

            full_fragment = (
                same_nonzero_delta
                and
                ncorrected
                ==
                frag_size
            )

            mixed_delta = (
                unique_delta.size
                >
                1
            )

            if full_fragment:

                pair_full_fragment_groups += 1

                pair_full_entries += (
                    ncorrected
                )

            else:

                pair_partial_groups += 1

                pair_partial_entries += (
                    ncorrected
                )

            if mixed_delta:
                pair_mixed_groups += 1

            if frag_size == 1:

                pair_singleton_entries += (
                    ncorrected
                )

            fragment_rows.append({
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

                "safe_fragment":
                    int(
                        frag
                    ),

                "fragment_size":
                    frag_size,

                "candidate_points":
                    ncorrected,

                "candidate_fraction":
                    float(
                        ncorrected
                        /
                        frag_size
                    ),

                "candidate_delta_values":
                    ",".join(
                        str(
                            int(x)
                        )
                        for x in
                        unique_delta.tolist()
                    ),

                "full_fragment_same_delta":
                    int(
                        full_fragment
                    ),

                "mixed_nonzero_delta":
                    int(
                        mixed_delta
                    ),
            })

        # ====================================================
        # B. Counterfactual spatial-edge quality
        #
        # Construct delta only for incident edges.
        # ====================================================

        dx_u = np.zeros(
            incident_ids.size,
            dtype=np.int32,
        )

        dx_v = np.zeros(
            incident_ids.size,
            dtype=np.int32,
        )

        if np.any(
            iu_is_bad
        ):

            dx_u[
                iu_is_bad
            ] = dx_bad[
                iu_idx[
                    iu_is_bad
                ]
            ]

        if np.any(
            iv_is_bad
        ):

            dx_v[
                iv_is_bad
            ] = dx_bad[
                iv_idx[
                    iv_is_bad
                ]
            ]

        boundary = (
            dx_u
            !=
            dx_v
        )

        nbnd = int(
            np.count_nonzero(
                boundary
            )
        )

        if nbnd:

            bu = iu[
                boundary
            ]

            bv = iv[
                boundary
            ]

            bdx_u = dx_u[
                boundary
            ]

            bdx_v = dx_v[
                boundary
            ]

            # Wrapped IFG only at required endpoints.
            phi_u = wrap(
                np.asarray(
                    phase[
                        bu,
                        tj
                    ],
                    dtype=np.float64,
                )
                -
                np.asarray(
                    phase[
                        bu,
                        ti
                    ],
                    dtype=np.float64,
                )
            )

            phi_v = wrap(
                np.asarray(
                    phase[
                        bv,
                        tj
                    ],
                    dtype=np.float64,
                )
                -
                np.asarray(
                    phase[
                        bv,
                        ti
                    ],
                    dtype=np.float64,
                )
            )

            g = wrap(
                phi_v
                -
                phi_u
            )

            before_delta = (
                np.asarray(
                    U[
                        bv
                    ],
                    dtype=np.float64,
                )
                -
                np.asarray(
                    U[
                        bu
                    ],
                    dtype=np.float64,
                )
                -
                g
            )

            before_jump = np.rint(
                before_delta
                /
                TWOPI
            ).astype(
                np.int32
            )

            # Exact integer update:
            #
            # U'_v-U'_u
            # =
            # U_v-U_u + 2pi(dx_v-dx_u)
            after_jump = (
                before_jump
                +
                bdx_v
                -
                bdx_u
            )

            safe = (
                np.abs(
                    g
                )
                <=
                np.pi / 2
            )

            unsafe = ~safe

            before_bad = (
                before_jump
                !=
                0
            )

            after_bad = (
                after_jump
                !=
                0
            )

            # -----------------------------------------------
            # SAFE
            # -----------------------------------------------

            safe_before_bad = int(
                np.count_nonzero(
                    safe
                    &
                    before_bad
                )
            )

            safe_after_bad = int(
                np.count_nonzero(
                    safe
                    &
                    after_bad
                )
            )

            safe_resolved = int(
                np.count_nonzero(
                    safe
                    &
                    before_bad
                    &
                    (~after_bad)
                )
            )

            safe_created = int(
                np.count_nonzero(
                    safe
                    &
                    (~before_bad)
                    &
                    after_bad
                )
            )

            # -----------------------------------------------
            # UNSAFE
            # -----------------------------------------------

            unsafe_before_bad = int(
                np.count_nonzero(
                    unsafe
                    &
                    before_bad
                )
            )

            unsafe_after_bad = int(
                np.count_nonzero(
                    unsafe
                    &
                    after_bad
                )
            )

            unsafe_resolved = int(
                np.count_nonzero(
                    unsafe
                    &
                    before_bad
                    &
                    (~after_bad)
                )
            )

            unsafe_created = int(
                np.count_nonzero(
                    unsafe
                    &
                    (~before_bad)
                    &
                    after_bad
                )
            )

            safe_boundary = int(
                np.count_nonzero(
                    safe
                )
            )

            unsafe_boundary = int(
                np.count_nonzero(
                    unsafe
                )
            )

            safe_l1_before = int(
                np.sum(
                    np.abs(
                        before_jump[
                            safe
                        ]
                    )
                )
            )

            safe_l1_after = int(
                np.sum(
                    np.abs(
                        after_jump[
                            safe
                        ]
                    )
                )
            )

            unsafe_l1_before = int(
                np.sum(
                    np.abs(
                        before_jump[
                            unsafe
                        ]
                    )
                )
            )

            unsafe_l1_after = int(
                np.sum(
                    np.abs(
                        after_jump[
                            unsafe
                        ]
                    )
                )
            )

        else:

            safe_boundary = 0
            unsafe_boundary = 0

            safe_before_bad = 0
            safe_after_bad = 0
            safe_resolved = 0
            safe_created = 0

            unsafe_before_bad = 0
            unsafe_after_bad = 0
            unsafe_resolved = 0
            unsafe_created = 0

            safe_l1_before = 0
            safe_l1_after = 0

            unsafe_l1_before = 0
            unsafe_l1_after = 0

        qa = batch_qa.get(
            pair_id,
            {}
        )

        if safe_created > 0:
            ifgs_safe_created += 1

        if (
            safe_resolved
            >
            safe_created
        ):
            ifgs_safe_improved += 1

        total_candidate_entries += int(
            nz_bad_idx.size
        )

        total_fragment_groups += int(
            unique_frags.size
        )

        total_full_fragment_groups += (
            pair_full_fragment_groups
        )

        total_partial_fragment_groups += (
            pair_partial_groups
        )

        total_mixed_delta_groups += (
            pair_mixed_groups
        )

        total_entries_full_fragment += (
            pair_full_entries
        )

        total_entries_partial_fragment += (
            pair_partial_entries
        )

        total_entries_singleton += (
            pair_singleton_entries
        )

        total_boundary_edges += (
            nbnd
        )

        total_safe_boundary += (
            safe_boundary
        )

        total_safe_before_bad += (
            safe_before_bad
        )

        total_safe_after_bad += (
            safe_after_bad
        )

        total_safe_resolved += (
            safe_resolved
        )

        total_safe_created += (
            safe_created
        )

        total_unsafe_boundary += (
            unsafe_boundary
        )

        total_unsafe_before_bad += (
            unsafe_before_bad
        )

        total_unsafe_after_bad += (
            unsafe_after_bad
        )

        total_unsafe_resolved += (
            unsafe_resolved
        )

        total_unsafe_created += (
            unsafe_created
        )

        if (
            safe_created == 0
            and
            safe_after_bad
            <=
            safe_before_bad
        ):

            spatial_status = (
                "NO_SAFE_DEGRADATION"
            )

        elif (
            safe_created
            >
            safe_resolved
        ):

            spatial_status = (
                "SAFE_CONTRADICTED"
            )

        else:

            spatial_status = (
                "MIXED"
            )

        pair_rows.append({
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

            "candidate_entries":
                int(
                    nz_bad_idx.size
                ),

            "candidate_fragments":
                int(
                    unique_frags.size
                ),

            "full_fragment_groups":
                pair_full_fragment_groups,

            "partial_fragment_groups":
                pair_partial_groups,

            "mixed_delta_groups":
                pair_mixed_groups,

            "full_fragment_entries":
                pair_full_entries,

            "partial_fragment_entries":
                pair_partial_entries,

            "singleton_entries":
                pair_singleton_entries,

            "changed_boundary_edges":
                nbnd,

            "safe_boundary_edges":
                safe_boundary,

            "safe_before_bad":
                safe_before_bad,

            "safe_after_bad":
                safe_after_bad,

            "safe_resolved":
                safe_resolved,

            "safe_created":
                safe_created,

            "safe_L1_before":
                safe_l1_before,

            "safe_L1_after":
                safe_l1_after,

            "unsafe_boundary_edges":
                unsafe_boundary,

            "unsafe_before_bad":
                unsafe_before_bad,

            "unsafe_after_bad":
                unsafe_after_bad,

            "unsafe_resolved":
                unsafe_resolved,

            "unsafe_created":
                unsafe_created,

            "unsafe_L1_before":
                unsafe_l1_before,

            "unsafe_L1_after":
                unsafe_l1_after,

            "original_IFG_safe_bad":
                qa.get(
                    "safe_internal_bad",
                    -1,
                ),

            "original_selected_non_exact":
                qa.get(
                    "selected_non_exact",
                    -1,
                ),

            "spatial_status":
                spatial_status,
        })

    # ========================================================
    # Summary
    # ========================================================

    pair_rows.sort(
        key=lambda r: (
            r[
                "safe_created"
            ],
            r[
                "partial_fragment_entries"
            ],
            r[
                "candidate_entries"
            ],
        ),
        reverse=True,
    )

    full_entry_fraction = (
        total_entries_full_fragment
        /
        total_candidate_entries
        if total_candidate_entries
        else 0.0
    )

    partial_entry_fraction = (
        total_entries_partial_fragment
        /
        total_candidate_entries
        if total_candidate_entries
        else 0.0
    )

    print()
    print("=" * 112)
    print(
        "A. Candidate / safe-fragment coherence"
    )
    print("=" * 112)

    print(
        f"IFGs touched                : "
        f"{ifgs_touched}"
    )

    print(
        f"candidate entries           : "
        f"{total_candidate_entries}"
    )

    print(
        f"candidate fragment groups   : "
        f"{total_fragment_groups}"
    )

    print(
        f"full-fragment groups        : "
        f"{total_full_fragment_groups}"
    )

    print(
        f"partial-fragment groups     : "
        f"{total_partial_fragment_groups}"
    )

    print(
        f"mixed-delta fragment groups : "
        f"{total_mixed_delta_groups}"
    )

    print(
        f"entries in full fragments   : "
        f"{total_entries_full_fragment} "
        f"({100*full_entry_fraction:.3f}%)"
    )

    print(
        f"entries in partial fragments: "
        f"{total_entries_partial_fragment} "
        f"({100*partial_entry_fraction:.3f}%)"
    )

    print(
        f"entries on singleton "
        f"fragments : "
        f"{total_entries_singleton}"
    )

    print()
    print("=" * 112)
    print(
        "B. Counterfactual spatial-edge consistency"
    )
    print("=" * 112)

    print(
        f"changed spatial boundaries  : "
        f"{total_boundary_edges:,}"
    )

    print()
    print(
        "SAFE edges:"
    )

    print(
        f"  boundary edges            : "
        f"{total_safe_boundary:,}"
    )

    print(
        f"  bad before                : "
        f"{total_safe_before_bad:,}"
    )

    print(
        f"  bad after candidate       : "
        f"{total_safe_after_bad:,}"
    )

    print(
        f"  resolved by candidate     : "
        f"{total_safe_resolved:,}"
    )

    print(
        f"  newly CREATED             : "
        f"{total_safe_created:,}"
    )

    print(
        f"  net bad change            : "
        f"{total_safe_after_bad-total_safe_before_bad:+,}"
    )

    print()
    print(
        "UNSAFE edges:"
    )

    print(
        f"  boundary edges            : "
        f"{total_unsafe_boundary:,}"
    )

    print(
        f"  bad before                : "
        f"{total_unsafe_before_bad:,}"
    )

    print(
        f"  bad after candidate       : "
        f"{total_unsafe_after_bad:,}"
    )

    print(
        f"  resolved by candidate     : "
        f"{total_unsafe_resolved:,}"
    )

    print(
        f"  newly created             : "
        f"{total_unsafe_created:,}"
    )

    print()
    print(
        f"IFGs creating SAFE conflicts: "
        f"{ifgs_safe_created}/{ifgs_touched}"
    )

    print(
        f"IFGs net-improving SAFE     : "
        f"{ifgs_safe_improved}/{ifgs_touched}"
    )

    print()
    print(
        "Top IFGs by newly-created SAFE conflicts:"
    )

    print(
        " pair  dates                  "
        "cand fragFull fragPart "
        "safeBefore safeAfter resolved created"
    )

    for r in pair_rows[:25]:

        print(
            f" {r['pair_id']:4d}  "
            f"{r['date1']}->"
            f"{r['date2']} "
            f"{r['candidate_entries']:4d} "
            f"{r['full_fragment_groups']:8d} "
            f"{r['partial_fragment_groups']:8d} "
            f"{r['safe_before_bad']:10d} "
            f"{r['safe_after_bad']:9d} "
            f"{r['safe_resolved']:8d} "
            f"{r['safe_created']:7d}"
        )

    # ========================================================
    # Save
    # ========================================================

    pair_csv = (
        outdir
        / "candidate_spatial_validation_by_ifg.csv"
    )

    with pair_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                pair_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            pair_rows
        )

    frag_csv = (
        outdir
        / "candidate_safe_fragment_coherence.csv"
    )

    with frag_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                fragment_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            fragment_rows
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-temporal-candidate-spatial-validation-v1.0",

        "status":
            "QUALITY_ONLY_NOT_APPLIED",

        "candidate_entries":
            int(
                total_candidate_entries
            ),

        "ifgs_touched":
            int(
                ifgs_touched
            ),

        "fragment_coherence": {
            "groups":
                int(
                    total_fragment_groups
                ),

            "full_fragment_groups":
                int(
                    total_full_fragment_groups
                ),

            "partial_fragment_groups":
                int(
                    total_partial_fragment_groups
                ),

            "mixed_delta_groups":
                int(
                    total_mixed_delta_groups
                ),

            "entries_full_fragment":
                int(
                    total_entries_full_fragment
                ),

            "entries_partial_fragment":
                int(
                    total_entries_partial_fragment
                ),

            "entries_singleton":
                int(
                    total_entries_singleton
                ),
        },

        "safe_counterfactual": {
            "boundary_edges":
                int(
                    total_safe_boundary
                ),

            "bad_before":
                int(
                    total_safe_before_bad
                ),

            "bad_after":
                int(
                    total_safe_after_bad
                ),

            "resolved":
                int(
                    total_safe_resolved
                ),

            "created":
                int(
                    total_safe_created
                ),

            "net_bad_change":
                int(
                    total_safe_after_bad
                    -
                    total_safe_before_bad
                ),

            "ifgs_creating_safe_conflicts":
                int(
                    ifgs_safe_created
                ),
        },

        "unsafe_counterfactual": {
            "boundary_edges":
                int(
                    total_unsafe_boundary
                ),

            "bad_before":
                int(
                    total_unsafe_before_bad
                ),

            "bad_after":
                int(
                    total_unsafe_after_bad
                ),

            "resolved":
                int(
                    total_unsafe_resolved
                ),

            "created":
                int(
                    total_unsafe_created
                ),
        },

        "note":
            (
                "Temporal minimum-L1 candidates "
                "were evaluated counterfactually. "
                "No production unwrapped phase "
                "file was modified."
            ),
    }

    json_path = (
        outdir
        / "candidate_spatial_validation.json"
    )

    json_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print(
        f"per-IFG table              : "
        f"{pair_csv}"
    )

    print(
        f"fragment coherence table   : "
        f"{frag_csv}"
    )

    print(
        f"manifest                   : "
        f"{json_path}"
    )

    print()
    print(
        "STEP 08v STATUS: PASS / "
        "SPATIAL COUNTERFACTUAL QUALITY ONLY"
    )

    print(
        "No temporal candidate has been applied."
    )


if __name__ == "__main__":
    main()
