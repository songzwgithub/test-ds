#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


MASK64 = (1 << 64) - 1


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


def splitmix64(x):
    """
    Vectorized uint64 mixing.
    Overflow is intentional.
    """
    with np.errstate(over="ignore"):
        x = x.astype(
            np.uint64,
            copy=False,
        )

        x = (
            x
            +
            np.uint64(
                0x9E3779B97F4A7C15
            )
        )

        x = (
            (x ^ (x >> np.uint64(30)))
            *
            np.uint64(
                0xBF58476D1CE4E5B9
            )
        )

        x = (
            (x ^ (x >> np.uint64(27)))
            *
            np.uint64(
                0x94D049BB133111EB
            )
        )

        x = (
            x
            ^
            (x >> np.uint64(31))
        )

    return x


def read_bad_point_patterns(
    sparse_csv: Path,
    bad_point_ids,
):
    """
    Steptemporal_integer_candidate sparse correction CSV contains one or more
    entries per bad point and carries pattern_id.

    Recover one pattern ID per bad point.
    """

    pattern_by_point = {}

    with sparse_csv.open() as f:
        for r in csv.DictReader(f):

            pid = int(
                r["point_id"]
            )

            pattern = int(
                r["pattern_id"]
            )

            old = pattern_by_point.get(
                pid
            )

            if (
                old is not None
                and
                old != pattern
            ):
                raise RuntimeError(
                    f"point {pid} has inconsistent "
                    f"pattern IDs: {old}, {pattern}"
                )

            pattern_by_point[
                pid
            ] = pattern

    patterns = np.empty(
        bad_point_ids.size,
        dtype=np.int32,
    )

    for k, pid in enumerate(
        bad_point_ids.tolist()
    ):

        if pid not in pattern_by_point:
            raise RuntimeError(
                f"bad point {pid} not found "
                "in Steptemporal_integer_candidate sparse table"
            )

        patterns[k] = (
            pattern_by_point[
                pid
            ]
        )

    return patterns


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

    network_dir = (
        root
        / "network"
    )

    policy_dir = (
        root
        / "unwrap_component_policy"
    )

    frag_dir = (
        root
        / "safe_fragment_integer_quality"
    )

    temporal_dir = (
        root
        / "temporal_integer_closure_quality"
    )

    candidate_dir = (
        root
        / "temporal_sparse_integer_candidate"
    )

    pps_dir = (
        root
        / "point_phase_stack"
    )

    outdir = (
        root
        / "fragment_signature_feasibility"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Static data
    # ========================================================

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

    main_mask = np.load(
        policy_dir
        / "main_component_mask.npy",
        mmap_mode="r",
    ).astype(
        bool,
        copy=False,
    )

    main_ids = np.where(
        main_mask
    )[0].astype(
        np.int32
    )

    nmain = main_ids.size

    bad_point_ids = np.load(
        candidate_dir
        / "bad_point_ids.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    nbad = bad_point_ids.size

    temporal_bad_count = np.load(
        temporal_dir
        / "point_temporal_bad_cycle_count.npy",
        mmap_mode="r",
    )

    # All Steptemporal_integer_candidate points must lie in the main component.
    bad_pos = np.searchsorted(
        main_ids,
        bad_point_ids,
    )

    if np.any(
        bad_pos >= nmain
    ):
        raise RuntimeError(
            "bad-point position outside main component"
        )

    if not np.array_equal(
        main_ids[
            bad_pos
        ],
        bad_point_ids,
    ):
        raise RuntimeError(
            "Some temporal bad points are not "
            "in main component"
        )

    bad_patterns = read_bad_point_patterns(
        candidate_dir
        / "candidate_sparse_integer_corrections.csv",
        bad_point_ids,
    )

    # ========================================================
    # Compute collision-resistant dual uint64 hash for the
    # complete temporal-network IFG safe-fragment membership signature.
    #
    # No [Npoint,nedge] fragment cube is created.
    # ========================================================

    h1 = np.full(
        nmain,
        np.uint64(
            0x243F6A8885A308D3
        ),
        dtype=np.uint64,
    )

    h2 = np.full(
        nmain,
        np.uint64(
            0x13198A2E03707344
        ),
        dtype=np.uint64,
    )

    print("=" * 112)
    print(
        "Spatiotemporal safe-fragment "
        "signature feasibility quality"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"main spatial points        : "
        f"{nmain:,}"
    )

    print(
        f"temporal bad points        : "
        f"{nbad}"
    )

    print(
        f"IFGs in signature          : "
        f"{nedge}"
    )

    print()
    print(
        "Building dual-64-bit fragment signatures ..."
    )

    for e0, (ti, tj) in enumerate(
        temporal_edges
    ):

        pair_id = e0 + 1

        tag = (
            f"pair{pair_id:03d}_"
            f"{stack.dates[ti]}_"
            f"{stack.dates[tj]}"
        )

        path = (
            frag_dir
            / f"{tag}_safe_fragment.npy"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        frag = np.load(
            path,
            mmap_mode="r",
        )

        if frag.size != npoint:
            raise RuntimeError(
                f"{path.name}: point count mismatch"
            )

        labels = np.asarray(
            frag[
                main_ids
            ],
            dtype=np.uint64,
        )

        salt1 = np.uint64(
            (
                (pair_id * 0x9E3779B97F4A7C15)
                &
                MASK64
            )
        )

        salt2 = np.uint64(
            (
                (pair_id * 0xD1B54A32D192ED03)
                &
                MASK64
            )
        )

        with np.errstate(over="ignore"):

            m1 = splitmix64(
                labels
                ^
                salt1
            )

            m2 = splitmix64(
                labels
                +
                salt2
            )

            h1 = (
                (
                    h1
                    ^
                    m1
                )
                *
                np.uint64(
                    0x9E3779B185EBCA87
                )
            )

            h2 = (
                (
                    h2
                    ^
                    m2
                )
                *
                np.uint64(
                    0xC2B2AE3D27D4EB4F
                )
            )

        if (
            pair_id == 1
            or
            pair_id % 10 == 0
            or
            pair_id == nedge
        ):
            print(
                f"  {pair_id:3d}/"
                f"{nedge:3d}"
            )

    # ========================================================
    # Group main points by full fragment signature.
    # ========================================================

    print()
    print(
        "Grouping identical temporal-network IFG signatures ..."
    )

    order = np.lexsort(
        (
            h2,
            h1,
        )
    )

    hs1 = h1[
        order
    ]

    hs2 = h2[
        order
    ]

    start_flag = np.ones(
        nmain,
        dtype=bool,
    )

    start_flag[
        1:
    ] = (
        (hs1[1:] != hs1[:-1])
        |
        (hs2[1:] != hs2[:-1])
    )

    group_id_sorted = (
        np.cumsum(
            start_flag,
            dtype=np.int64,
        )
        -
        1
    ).astype(
        np.int32
    )

    ngroup = int(
        group_id_sorted[
            -1
        ]
        +
        1
    )

    group_id = np.empty(
        nmain,
        dtype=np.int32,
    )

    group_id[
        order
    ] = group_id_sorted

    group_sizes = np.bincount(
        group_id,
        minlength=ngroup,
    ).astype(
        np.int32
    )

    bad_groups = group_id[
        bad_pos
    ]

    group_bad_counts = np.bincount(
        bad_groups,
        minlength=ngroup,
    ).astype(
        np.int32
    )

    unique_bad_groups = np.unique(
        bad_groups
    )

    # Temporal error patterns present inside each
    # bad-containing signature group.
    pattern_sets = defaultdict(set)

    for gid, pattern in zip(
        bad_groups.tolist(),
        bad_patterns.tolist(),
    ):

        pattern_sets[
            int(gid)
        ].add(
            int(pattern)
        )

    # ========================================================
    # Classify bad-containing signature groups.
    # ========================================================

    group_rows = []

    mixed_good_bad_groups = 0
    mixed_good_bad_bad_points = 0

    pure_bad_single_pattern_groups = 0
    pure_bad_single_pattern_points = 0

    pure_bad_multi_pattern_groups = 0
    pure_bad_multi_pattern_points = 0

    singleton_signature_bad_points = 0

    for gid in unique_bad_groups.tolist():

        size = int(
            group_sizes[
                gid
            ]
        )

        bad_count = int(
            group_bad_counts[
                gid
            ]
        )

        good_count = (
            size
            -
            bad_count
        )

        patterns = sorted(
            pattern_sets[
                int(gid)
            ]
        )

        npattern = len(
            patterns
        )

        if good_count > 0:

            status = (
                "INCOMPATIBLE_GOOD_BAD_SHARED_SIGNATURE"
            )

            mixed_good_bad_groups += 1

            mixed_good_bad_bad_points += (
                bad_count
            )

        elif npattern > 1:

            status = (
                "INCOMPATIBLE_MULTIPLE_BAD_PATTERNS"
            )

            pure_bad_multi_pattern_groups += 1

            pure_bad_multi_pattern_points += (
                bad_count
            )

        else:

            status = (
                "STRUCTURALLY_COMPATIBLE_PURE_BAD_SIGNATURE"
            )

            pure_bad_single_pattern_groups += 1

            pure_bad_single_pattern_points += (
                bad_count
            )

        if (
            size == 1
            and
            bad_count == 1
        ):
            singleton_signature_bad_points += 1

        members_bad_mask = (
            bad_groups
            ==
            gid
        )

        member_bad_ids = (
            bad_point_ids[
                members_bad_mask
            ]
        )

        group_rows.append({
            "signature_group":
                int(gid),

            "group_size":
                size,

            "bad_points":
                bad_count,

            "good_points":
                good_count,

            "distinct_bad_error_patterns":
                npattern,

            "bad_pattern_ids":
                ",".join(
                    str(x)
                    for x in patterns
                ),

            "bad_point_ids":
                ",".join(
                    str(
                        int(x)
                    )
                    for x in
                    member_bad_ids.tolist()
                ),

            "status":
                status,
        })

    # ========================================================
    # Per-bad-point table
    # ========================================================

    point_rows = []

    for q, pid in enumerate(
        bad_point_ids.tolist()
    ):

        gid = int(
            bad_groups[
                q
            ]
        )

        size = int(
            group_sizes[
                gid
            ]
        )

        bad_count = int(
            group_bad_counts[
                gid
            ]
        )

        good_count = (
            size
            -
            bad_count
        )

        patterns = pattern_sets[
            gid
        ]

        if good_count > 0:

            status = (
                "INCOMPATIBLE_GOOD_BAD_SHARED_SIGNATURE"
            )

        elif len(patterns) > 1:

            status = (
                "INCOMPATIBLE_MULTIPLE_BAD_PATTERNS"
            )

        else:

            status = (
                "STRUCTURALLY_COMPATIBLE"
            )

        point_rows.append({
            "point_id":
                int(pid),

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

            "temporal_bad_cycles":
                int(
                    temporal_bad_count[
                        pid
                    ]
                ),

            "error_pattern_id":
                int(
                    bad_patterns[
                        q
                    ]
                ),

            "signature_group":
                gid,

            "signature_group_size":
                size,

            "bad_points_in_signature":
                bad_count,

            "good_points_in_signature":
                good_count,

            "status":
                status,
        })

    # ========================================================
    # Global statistics
    # ========================================================

    bad_signature_sizes = group_sizes[
        bad_groups
    ]

    incompatible_bad_points = (
        mixed_good_bad_bad_points
        +
        pure_bad_multi_pattern_points
    )

    compatible_bad_points = (
        pure_bad_single_pattern_points
    )

    print()
    print("=" * 112)
    print(
        "Fragment-signature structure"
    )
    print("=" * 112)

    print(
        f"main points                : "
        f"{nmain:,}"
    )

    print(
        f"unique temporal-network signatures : "
        f"{ngroup:,}"
    )

    print(
        f"signature groups containing "
        f"temporal-bad points:"
    )

    print(
        f"  {unique_bad_groups.size}"
    )

    print(
        f"bad-point signature size "
        f"min/med/max:"
    )

    print(
        f"  "
        f"{bad_signature_sizes.min()} / "
        f"{np.median(bad_signature_sizes):.1f} / "
        f"{bad_signature_sizes.max()}"
    )

    print(
        f"bad points with singleton "
        f"full signature:"
    )

    print(
        f"  "
        f"{singleton_signature_bad_points}/"
        f"{nbad}"
    )

    print()
    print("=" * 112)
    print(
        "Necessary-condition feasibility"
    )
    print("=" * 112)

    print(
        "A. Pure-bad, single-error-pattern signatures"
    )

    print(
        f"  groups                   : "
        f"{pure_bad_single_pattern_groups}"
    )

    print(
        f"  bad points               : "
        f"{pure_bad_single_pattern_points}"
    )

    print()

    print(
        "B. GOOD + BAD share identical temporal-network IFG signature"
    )

    print(
        f"  incompatible groups      : "
        f"{mixed_good_bad_groups}"
    )

    print(
        f"  affected bad points      : "
        f"{mixed_good_bad_bad_points}"
    )

    print()

    print(
        "C. Pure-bad signature but multiple temporal patterns"
    )

    print(
        f"  incompatible groups      : "
        f"{pure_bad_multi_pattern_groups}"
    )

    print(
        f"  affected bad points      : "
        f"{pure_bad_multi_pattern_points}"
    )

    print()

    print(
        f"structurally compatible "
        f"bad points : "
        f"{compatible_bad_points}/"
        f"{nbad}"
    )

    print(
        f"structurally incompatible "
        f"bad points: "
        f"{incompatible_bad_points}/"
        f"{nbad}"
    )

    if incompatible_bad_points == 0:

        overall = (
            "PASS_NECESSARY_CONDITION"
        )

        print()
        print(
            "RESULT: all temporal-bad points pass "
            "the necessary fragment-signature condition."
        )

        print(
            "A fragment-constant joint integer solution "
            "remains structurally possible."
        )

    else:

        overall = (
            "FAIL_NECESSARY_CONDITION"
        )

        print()
        print(
            "RESULT: the current safe-fragment partition "
            "cannot exactly correct all temporal-bad points "
            "using fragment-constant shifts alone."
        )

        print(
            "The incompatible signature groups must be "
            "locally split/masked before any joint "
            "fragment-level integer correction."
        )

    # Show largest problematic groups.
    problematic = [
        r
        for r in group_rows
        if r[
            "status"
        ].startswith(
            "INCOMPATIBLE"
        )
    ]

    problematic.sort(
        key=lambda r: (
            r[
                "bad_points"
            ],
            r[
                "group_size"
            ],
        ),
        reverse=True,
    )

    if problematic:

        print()
        print(
            "Largest incompatible signature groups:"
        )

        print(
            " group     size   bad   good "
            "patterns  status"
        )

        for r in problematic[:20]:

            print(
                f" {r['signature_group']:6d} "
                f"{r['group_size']:7d} "
                f"{r['bad_points']:5d} "
                f"{r['good_points']:6d} "
                f"{r['distinct_bad_error_patterns']:8d} "
                f"{r['status']}"
            )

    # ========================================================
    # Save
    # ========================================================

    group_rows.sort(
        key=lambda r:
            r[
                "signature_group"
            ]
    )

    group_csv = (
        outdir
        / "bad_signature_groups.csv"
    )

    with group_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                group_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            group_rows
        )

    point_csv = (
        outdir
        / "temporal_bad_point_signature_status.csv"
    )

    with point_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                point_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            point_rows
        )

    np.save(
        outdir
        / "main_signature_hash1.npy",
        h1,
    )

    np.save(
        outdir
        / "main_signature_hash2.npy",
        h2,
    )

    np.save(
        outdir
        / "main_signature_group.npy",
        group_id,
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-fragment-signature-feasibility-v1.0",

        "status":
            "QUALITY_ONLY",

        "result":
            overall,

        "main_points":
            int(
                nmain
            ),

        "ifgs":
            int(
                nedge
            ),

        "unique_fragment_signatures":
            int(
                ngroup
            ),

        "temporal_bad_points":
            int(
                nbad
            ),

        "bad_signature_groups":
            int(
                unique_bad_groups.size
            ),

        "compatible": {
            "pure_bad_single_pattern_groups":
                int(
                    pure_bad_single_pattern_groups
                ),

            "bad_points":
                int(
                    pure_bad_single_pattern_points
                ),
        },

        "incompatible_good_bad_shared_signature": {
            "groups":
                int(
                    mixed_good_bad_groups
                ),

            "bad_points":
                int(
                    mixed_good_bad_bad_points
                ),
        },

        "incompatible_multiple_bad_patterns": {
            "groups":
                int(
                    pure_bad_multi_pattern_groups
                ),

            "bad_points":
                int(
                    pure_bad_multi_pattern_points
                ),
        },

        "singleton_bad_signatures":
            int(
                singleton_signature_bad_points
            ),

        "note":
            (
                "Dual uint64 hashes encode the complete "
                "temporal-network IFG safe-fragment membership signature "
                "without persisting an Npoint x Nifg cube. "
                "This is a necessary-condition quality only; "
                "passing does not yet prove that an integer "
                "fragment-level solution exists."
            ),
    }

    json_path = (
        outdir
        / "fragment_signature_feasibility.json"
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
        f"group table                : "
        f"{group_csv}"
    )

    print(
        f"point table                : "
        f"{point_csv}"
    )

    print(
        f"manifest                   : "
        f"{json_path}"
    )

    print()
    print(
        "STEP unwrap_signature_quality STATUS: PASS / "
        "STRUCTURAL FEASIBILITY QUALITY COMPLETE"
    )


if __name__ == "__main__":
    main()
