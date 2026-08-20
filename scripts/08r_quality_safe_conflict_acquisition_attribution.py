#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def wrap(x):
    return np.arctan2(
        np.sin(x),
        np.cos(x),
    )


def load_network(path: Path):
    out = []

    with path.open() as f:

        for raw in f:

            x = raw.split()

            if len(x) < 2:
                continue

            out.append(
                (
                    int(x[0]) - 1,
                    int(x[1]) - 1,
                )
            )

    return out


def load_qa(path: Path):

    rows = []

    with path.open() as f:

        for r in csv.DictReader(f):

            rows.append({
                "pair_id":
                    int(r["pair_id"]),

                "date1":
                    r["date1"],

                "date2":
                    r["date2"],

                "safe_internal_bad":
                    int(
                        r[
                            "safe_internal_bad"
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

    graph = (
        root
        / "spatial_graph"
    )

    dir08l = (
        root
        / "safe_fragment_integer_quality"
    )

    qa_csv = (
        root
        / "batch_unwrap_validation"
        / "all_ifg_unwrap_qa.csv"
    )

    network_file = (
        root
        / "network"
        / "network.itab"
    )

    outdir = (
        root
        / "safe_conflict_acquisition_quality"
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

    rows_pt = np.load(
        pps
        / "rows.npy",
        mmap_mode="r",
    )

    cols_pt = np.load(
        pps
        / "cols.npy",
        mmap_mode="r",
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

    npoint, ndate = phase.shape

    temporal_edges = load_network(
        network_file
    )

    qa = load_qa(
        qa_csv
    )

    if len(temporal_edges) != len(qa):

        raise RuntimeError(
            "network / QA pair count mismatch"
        )

    conflict_pairs = [
        r
        for r in qa
        if r[
            "safe_internal_bad"
        ] > 0
    ]

    print("=" * 104)
    print(
        "Step 08r - SAFE-conflict acquisition attribution quality"
    )
    print("=" * 104)

    print(
        f"config                     : {config_path}"
    )

    print(
        f"points                     : {npoint:,}"
    )

    print(
        f"acquisitions               : {ndate}"
    )

    print(
        f"SAFE-conflict IFGs         : "
        f"{len(conflict_pairs)}"
    )

    # ========================================================
    # Extract actual bad SAFE edge IDs from each IFG
    # ========================================================

    pair_bad_safe = {}

    all_bad_edge_counter = Counter()

    acquisition_pair_hits = Counter()

    detail_rows = []

    for r in conflict_pairs:

        pair_id = r[
            "pair_id"
        ]

        i, j = temporal_edges[
            pair_id - 1
        ]

        tag = (
            f"pair{pair_id:03d}_"
            f"{stack.dates[i]}_"
            f"{stack.dates[j]}"
        )

        bad_path = (
            dir08l
            / f"{tag}_bad_edge_ids.npy"
        )

        if not bad_path.exists():

            raise FileNotFoundError(
                bad_path
            )

        bad_ids = np.load(
            bad_path
        ).astype(
            np.int64,
            copy=False,
        )

        # Recalculate this IFG's spatial gradient,
        # then retain only bad edges that are SAFE.
        ifg = wrap(
            np.asarray(
                phase[:, j],
                dtype=np.float32,
            )
            -
            np.asarray(
                phase[:, i],
                dtype=np.float32,
            )
        )

        u = np.asarray(
            local_u[
                bad_ids
            ],
            dtype=np.int32,
        )

        v = np.asarray(
            local_v[
                bad_ids
            ],
            dtype=np.int32,
        )

        g = wrap(
            ifg[v]
            -
            ifg[u]
        )

        safe = (
            np.abs(g)
            <=
            np.pi / 2
        )

        ids = bad_ids[
            safe
        ]

        if ids.size != r[
            "safe_internal_bad"
        ]:

            raise RuntimeError(
                f"pair {pair_id}: "
                f"expected "
                f"{r['safe_internal_bad']} "
                f"safe bad edges, "
                f"recovered {ids.size}"
            )

        pair_bad_safe[
            pair_id
        ] = ids

        for eid in ids.tolist():

            all_bad_edge_counter[
                int(eid)
            ] += 1

        acquisition_pair_hits[i] += (
            int(ids.size)
        )

        acquisition_pair_hits[j] += (
            int(ids.size)
        )

        print(
            f"  pair {pair_id:3d} "
            f"{stack.dates[i]}->"
            f"{stack.dates[j]}: "
            f"{ids.size} SAFE bad edges"
        )

    # ========================================================
    # Spatial recurrence
    # ========================================================

    unique_bad_edges = np.array(
        sorted(
            all_bad_edge_counter.keys()
        ),
        dtype=np.int64,
    )

    recurrence = np.array(
        [
            all_bad_edge_counter[
                int(eid)
            ]
            for eid in unique_bad_edges
        ],
        dtype=np.int32,
    )

    bu = np.asarray(
        local_u[
            unique_bad_edges
        ],
        dtype=np.int32,
    )

    bv = np.asarray(
        local_v[
            unique_bad_edges
        ],
        dtype=np.int32,
    )

    unique_points = np.unique(
        np.concatenate(
            [
                bu,
                bv,
            ]
        )
    )

    print()
    print("=" * 104)
    print(
        "Spatial recurrence of SAFE-conflict edges"
    )
    print("=" * 104)

    print(
        f"total SAFE-bad occurrences : "
        f"{recurrence.sum():,}"
    )

    print(
        f"unique spatial edges       : "
        f"{unique_bad_edges.size:,}"
    )

    print(
        f"unique involved points     : "
        f"{unique_points.size:,}"
    )

    print(
        f"edge recurrence min/med/max: "
        f"{recurrence.min()} / "
        f"{np.median(recurrence):.1f} / "
        f"{recurrence.max()}"
    )

    for n in (
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ):

        k = int(
            np.count_nonzero(
                recurrence >= n
            )
        )

        print(
            f"edges seen in >= {n} "
            f"conflict IFGs : {k}"
        )

    # ========================================================
    # Acquisition-domain spatial gradients
    #
    # For every unique suspicious spatial edge e:
    #
    # a_t(e) = wrap(theta_t(v)-theta_t(u))
    # ========================================================

    acquisition_gradient = np.empty(
        (
            unique_bad_edges.size,
            ndate,
        ),
        dtype=np.float32,
    )

    for t in range(ndate):

        acquisition_gradient[
            :,
            t
        ] = wrap(
            np.asarray(
                phase[
                    bv,
                    t
                ],
                dtype=np.float32,
            )
            -
            np.asarray(
                phase[
                    bu,
                    t
                ],
                dtype=np.float32,
            )
        ).astype(
            np.float32
        )

    edge_to_pos = {
        int(eid): k
        for k, eid in enumerate(
            unique_bad_edges.tolist()
        )
    }

    # ========================================================
    # Attribution score:
    #
    # For each actual conflict occurrence,
    # compare the magnitude of the acquisition-domain
    # spatial gradients at the two endpoint dates.
    #
    # This is not yet "proof of fault"; it is an
    # attribution diagnostic.
    # ========================================================

    acquisition_wins = Counter()
    acquisition_ties = Counter()
    acquisition_occurrences = Counter()

    occurrence_rows = []

    for r in conflict_pairs:

        pair_id = r[
            "pair_id"
        ]

        i, j = temporal_edges[
            pair_id - 1
        ]

        ids = pair_bad_safe[
            pair_id
        ]

        for eid in ids.tolist():

            pos = edge_to_pos[
                int(eid)
            ]

            ai = float(
                acquisition_gradient[
                    pos,
                    i
                ]
            )

            aj = float(
                acquisition_gradient[
                    pos,
                    j
                ]
            )

            mai = abs(ai)
            maj = abs(aj)

            acquisition_occurrences[
                i
            ] += 1

            acquisition_occurrences[
                j
            ] += 1

            # Only call a winner if difference is
            # meaningful (>0.15 rad).
            if mai > maj + 0.15:

                winner = i
                acquisition_wins[
                    i
                ] += 1

                attribution = (
                    str(
                        stack.dates[i]
                    )
                )

            elif maj > mai + 0.15:

                winner = j
                acquisition_wins[
                    j
                ] += 1

                attribution = (
                    str(
                        stack.dates[j]
                    )
                )

            else:

                winner = -1

                acquisition_ties[
                    i
                ] += 1

                acquisition_ties[
                    j
                ] += 1

                attribution = (
                    "ambiguous"
                )

            p = int(
                local_u[
                    eid
                ]
            )

            q = int(
                local_v[
                    eid
                ]
            )

            occurrence_rows.append({
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

                "edge_id":
                    int(eid),

                "point_u":
                    p,

                "point_v":
                    q,

                "row_u":
                    int(
                        rows_pt[p]
                    ),

                "col_u":
                    int(
                        cols_pt[p]
                    ),

                "row_v":
                    int(
                        rows_pt[q]
                    ),

                "col_v":
                    int(
                        cols_pt[q]
                    ),

                "acq_gradient_date1_rad":
                    ai,

                "acq_gradient_date2_rad":
                    aj,

                "abs_date1_rad":
                    mai,

                "abs_date2_rad":
                    maj,

                "larger_gradient_date":
                    attribution,
            })

    # ========================================================
    # Per-acquisition summary
    # ========================================================

    acq_rows = []

    for t in range(ndate):

        # Unique suspicious edges appearing in an IFG
        # touching acquisition t.
        touched = set()

        pair_count = 0

        for r in conflict_pairs:

            pair_id = r[
                "pair_id"
            ]

            i, j = temporal_edges[
                pair_id - 1
            ]

            if (
                i != t
                and
                j != t
            ):
                continue

            pair_count += 1

            touched.update(
                int(x)
                for x in
                pair_bad_safe[
                    pair_id
                ].tolist()
            )

        if touched:

            touched_arr = np.array(
                sorted(
                    touched
                ),
                dtype=np.int64,
            )

            positions = np.array(
                [
                    edge_to_pos[
                        int(eid)
                    ]
                    for eid in touched_arr
                ],
                dtype=np.int32,
            )

            vals = np.abs(
                acquisition_gradient[
                    positions,
                    t
                ]
            )

            med = float(
                np.median(vals)
            )

            p90 = float(
                np.quantile(
                    vals,
                    0.90
                )
            )

            vmax = float(
                vals.max()
            )

        else:

            med = 0.0
            p90 = 0.0
            vmax = 0.0

        acq_rows.append({
            "acquisition_index":
                t,

            "date":
                str(
                    stack.dates[t]
                ),

            "conflict_ifgs_touching":
                pair_count,

            "bad_edge_occurrences_touching":
                int(
                    acquisition_occurrences[
                        t
                    ]
                ),

            "unique_bad_edges_touching":
                len(touched),

            "larger_gradient_wins":
                int(
                    acquisition_wins[
                        t
                    ]
                ),

            "ambiguous_ties":
                int(
                    acquisition_ties[
                        t
                    ]
                ),

            "touched_edge_gradient_median_rad":
                med,

            "touched_edge_gradient_p90_rad":
                p90,

            "touched_edge_gradient_max_rad":
                vmax,
        })

    acq_rows_sorted = sorted(
        acq_rows,
        key=lambda r: (
            r[
                "conflict_ifgs_touching"
            ],
            r[
                "unique_bad_edges_touching"
            ],
            r[
                "larger_gradient_wins"
            ],
        ),
        reverse=True,
    )

    print()
    print("=" * 104)
    print(
        "Acquisition attribution ranking"
    )
    print("=" * 104)

    print(
        " date       conflictIFG uniqueEdges "
        "wins  median|a|  p90|a|  max|a|"
    )

    for r in acq_rows_sorted[:12]:

        if (
            r[
                "conflict_ifgs_touching"
            ]
            == 0
        ):
            continue

        print(
            f" {r['date']} "
            f"{r['conflict_ifgs_touching']:11d} "
            f"{r['unique_bad_edges_touching']:11d} "
            f"{r['larger_gradient_wins']:4d} "
            f"{r['touched_edge_gradient_median_rad']:10.3f} "
            f"{r['touched_edge_gradient_p90_rad']:8.3f} "
            f"{r['touched_edge_gradient_max_rad']:8.3f}"
        )

    # Special report for 20150110 if present.
    target = None

    for r in acq_rows:

        if r["date"] == "20150110":
            target = r
            break

    print()
    print("=" * 104)
    print(
        "20150110 targeted quality"
    )
    print("=" * 104)

    if target is None:

        print(
            "20150110 not found in acquisition list."
        )

    else:

        for k, v in target.items():

            print(
                f"{k:34s}: {v}"
            )

    # ========================================================
    # Edge recurrence table
    # ========================================================

    recurrence_rows = []

    for k, eid in enumerate(
        unique_bad_edges.tolist()
    ):

        p = int(
            bu[k]
        )

        q = int(
            bv[k]
        )

        occurrences = []

        for r in conflict_pairs:

            pair_id = r[
                "pair_id"
            ]

            ids = pair_bad_safe[
                pair_id
            ]

            if np.any(
                ids == eid
            ):

                occurrences.append(
                    pair_id
                )

        recurrence_rows.append({
            "edge_id":
                int(eid),

            "point_u":
                p,

            "point_v":
                q,

            "row_u":
                int(
                    rows_pt[p]
                ),

            "col_u":
                int(
                    cols_pt[p]
                ),

            "row_v":
                int(
                    rows_pt[q]
                ),

            "col_v":
                int(
                    cols_pt[q]
                ),

            "conflict_occurrences":
                int(
                    recurrence[k]
                ),

            "pair_ids":
                ",".join(
                    str(x)
                    for x in occurrences
                ),
        })

    # ========================================================
    # Save
    # ========================================================

    occurrence_csv = (
        outdir
        / "safe_conflict_occurrences.csv"
    )

    with occurrence_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                occurrence_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            occurrence_rows
        )

    acq_csv = (
        outdir
        / "acquisition_attribution.csv"
    )

    with acq_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                acq_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            acq_rows
        )

    edge_csv = (
        outdir
        / "recurrent_safe_conflict_edges.csv"
    )

    with edge_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                recurrence_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            recurrence_rows
        )

    np.save(
        outdir
        / "suspicious_edge_ids.npy",
        unique_bad_edges,
    )

    np.save(
        outdir
        / "suspicious_edge_acquisition_gradient.npy",
        acquisition_gradient,
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-safe-conflict-acquisition-attribution-v1.0",

        "status":
            "QUALITY_ONLY",

        "safe_conflict_ifgs":
            len(
                conflict_pairs
            ),

        "total_bad_safe_occurrences":
            int(
                recurrence.sum()
            ),

        "unique_bad_edges":
            int(
                unique_bad_edges.size
            ),

        "unique_bad_points":
            int(
                unique_points.size
            ),

        "max_edge_recurrence":
            int(
                recurrence.max()
            ),

        "top_acquisitions":
            acq_rows_sorted[:10],
    }

    json_path = (
        outdir
        / "acquisition_attribution.json"
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
        f"occurrence table           : "
        f"{occurrence_csv}"
    )

    print(
        f"acquisition table          : "
        f"{acq_csv}"
    )

    print(
        f"recurrent edge table       : "
        f"{edge_csv}"
    )

    print(
        f"manifest                   : "
        f"{json_path}"
    )

    print()
    print(
        "STEP 08r STATUS: PASS / QUALITY ONLY"
    )


if __name__ == "__main__":
    main()
