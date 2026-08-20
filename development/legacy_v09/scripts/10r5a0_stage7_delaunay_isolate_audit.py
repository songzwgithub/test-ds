#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree


BRIDGE = Path(
    "/home/ubuntu/Downloads/psds/"
    "prototype_outputs/v09/"
    "pystamps_bridge_v09"
)

WORK = (
    BRIDGE
    /
    "_stage7_triangle_work"
)

PS2 = BRIDGE / "ps2.mat"

EDGE = WORK / "scla.2.edge"
NODE = WORK / "scla.1.node"
LOG = WORK / "triangle_scla.log"

OUT = (
    BRIDGE
    /
    "stage7_delaunay_isolate_audit_v09.json"
)


def read_triangle_edges(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        header = f.readline().split()

    if not header:
        raise RuntimeError(
            "Empty Triangle edge file"
        )

    nedge = int(
        header[0]
    )

    edges = np.loadtxt(
        path,
        dtype=np.int64,
        skiprows=1,
        max_rows=nedge,
        usecols=(1, 2),
    )

    if edges.ndim == 1:
        edges = edges.reshape(
            1,
            2,
        )

    return edges


def duplicate_groups(
    xy: np.ndarray,
):

    unique_xy, inv, counts = np.unique(
        xy,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    duplicate_ids = np.flatnonzero(
        counts > 1
    )

    groups = []

    for uid in duplicate_ids:

        members = np.flatnonzero(
            inv == uid
        )

        groups.append(
            members
        )

    return groups


def main():

    if not PS2.is_file():
        raise RuntimeError(
            f"Missing {PS2}"
        )

    if not EDGE.is_file():
        raise RuntimeError(
            f"Missing Triangle edge file: {EDGE}"
        )

    ps = loadmat(
        PS2,
        variable_names=[
            "xy",
            "ij",
            "lonlat",
            "n_ps",
        ],
        squeeze_me=False,
    )

    xy_all = np.asarray(
        ps["xy"],
        dtype=np.float64,
    )

    ij = np.asarray(
        ps["ij"],
        dtype=np.float64,
    )

    lonlat = np.asarray(
        ps["lonlat"],
        dtype=np.float64,
    )

    nps = int(
        round(
            float(
                np.asarray(
                    ps["n_ps"]
                ).reshape(-1)[0]
            )
        )
    )

    if xy_all.shape != (
        nps,
        3,
    ):
        raise RuntimeError(
            f"Unexpected xy shape "
            f"{xy_all.shape}"
        )

    xy = xy_all[
        :,
        1:3
    ]

    rows = (
        np.rint(
            ij[:, 1]
        ).astype(
            np.int64
        )
        -
        1
    )

    cols = (
        np.rint(
            ij[:, 2]
        ).astype(
            np.int64
        )
        -
        1
    )

    edges = read_triangle_edges(
        EDGE
    )

    # Triangle indices are 1-based.
    u = edges[:, 0] - 1
    v = edges[:, 1] - 1

    degree = np.zeros(
        nps,
        dtype=np.int64,
    )

    np.add.at(
        degree,
        u,
        1,
    )

    np.add.at(
        degree,
        v,
        1,
    )

    isolated = np.flatnonzero(
        degree == 0
    )

    # --------------------------------------------------------
    # Exact duplicate coordinates as stored in ps2.mat
    # --------------------------------------------------------

    exact_groups = duplicate_groups(
        xy
    )

    exact_member = np.zeros(
        nps,
        dtype=bool,
    )

    for g in exact_groups:
        exact_member[g] = True

    # --------------------------------------------------------
    # Duplicate coordinates after the exact formatting used
    # by stage7_sbas._triangle_edges_external:
    #
    #     f"{xy:.6f}"
    #
    # Numerically equivalent to rounding to 6 decimals for
    # this audit.
    # --------------------------------------------------------

    xy6 = np.round(
        xy,
        6,
    )

    rounded_groups = duplicate_groups(
        xy6
    )

    rounded_member = np.zeros(
        nps,
        dtype=bool,
    )

    for g in rounded_groups:
        rounded_member[g] = True

    isolated_exact_dup = isolated[
        exact_member[
            isolated
        ]
    ]

    isolated_round_dup = isolated[
        rounded_member[
            isolated
        ]
    ]

    # --------------------------------------------------------
    # Nearest-neighbour distances for isolated points
    # --------------------------------------------------------

    tree = cKDTree(
        xy
    )

    if isolated.size:

        dist, nn = tree.query(
            xy[
                isolated
            ],
            k=2,
            workers=-1,
        )

        nearest_distance = (
            dist[:, 1]
        )

        nearest_id = (
            nn[:, 1]
        )

    else:

        nearest_distance = np.empty(
            0,
            dtype=np.float64,
        )

        nearest_id = np.empty(
            0,
            dtype=np.int64,
        )

    # --------------------------------------------------------
    # Coordinate-boundary audit
    # --------------------------------------------------------

    left_edge = np.isin(
        cols,
        [
            0,
            1,
            2,
            3,
        ],
    )

    right_edge = np.isin(
        cols,
        [
            1996,
            1997,
            1998,
            1999,
        ],
    )

    edge_zone = (
        left_edge
        |
        right_edge
    )

    isolated_edge_zone = isolated[
        edge_zone[
            isolated
        ]
    ]

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("=" * 112)
    print(
        "Step 10R5a0 - Stage7 Delaunay "
        "isolated-point audit"
    )
    print("=" * 112)

    print(
        f"PS                         : "
        f"{nps:,}"
    )

    print(
        f"Triangle edges             : "
        f"{edges.shape[0]:,}"
    )

    print(
        f"degree min/median/max      : "
        f"{degree.min()} / "
        f"{np.median(degree):.0f} / "
        f"{degree.max()}"
    )

    print(
        f"isolated PS                : "
        f"{isolated.size:,}"
    )

    print()
    print("=" * 112)
    print(
        "Coordinate duplicates"
    )
    print("=" * 112)

    print(
        f"exact duplicate groups     : "
        f"{len(exact_groups):,}"
    )

    print(
        f"exact duplicate members    : "
        f"{np.count_nonzero(exact_member):,}"
    )

    print(
        f"6-decimal duplicate groups : "
        f"{len(rounded_groups):,}"
    )

    print(
        f"6-decimal duplicate members: "
        f"{np.count_nonzero(rounded_member):,}"
    )

    print(
        f"isolated + exact duplicate : "
        f"{isolated_exact_dup.size:,}/"
        f"{isolated.size:,}"
    )

    print(
        f"isolated + 6dec duplicate  : "
        f"{isolated_round_dup.size:,}/"
        f"{isolated.size:,}"
    )

    print()
    print("=" * 112)
    print(
        "Raw-radar boundary relationship"
    )
    print("=" * 112)

    print(
        f"isolated in edge zone      : "
        f"{isolated_edge_zone.size:,}/"
        f"{isolated.size:,}"
    )

    if isolated.size:

        unique_cols, col_counts = np.unique(
            cols[
                isolated
            ],
            return_counts=True,
        )

        order = np.argsort(
            col_counts
        )[::-1]

        print()
        print(
            "isolated raw columns "
            "(most frequent):"
        )

        for k in order[:20]:

            print(
                f"  col={unique_cols[k]:4d}  "
                f"count={col_counts[k]:5d}"
            )

    print()
    print("=" * 112)
    print(
        "Isolated-point details"
    )
    print("=" * 112)

    if isolated.size == 0:

        print(
            "No isolated points found."
        )

    else:

        show = min(
            isolated.size,
            50,
        )

        for k in range(show):

            p = int(
                isolated[k]
            )

            q = int(
                nearest_id[k]
            )

            print(
                f"id={p:6d}  "
                f"row={rows[p]:3d} "
                f"col={cols[p]:4d}  "
                f"xy=({xy[p,0]:.6f},"
                f"{xy[p,1]:.6f})  "
                f"nearest={q:6d} "
                f"d={nearest_distance[k]:.9f} m  "
                f"dup_exact="
                f"{bool(exact_member[p])}  "
                f"dup_6dec="
                f"{bool(rounded_member[p])}"
            )

    # --------------------------------------------------------
    # Triangle warning lines
    # --------------------------------------------------------

    warning_lines = []

    if LOG.is_file():

        for line in LOG.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():

            low = line.lower()

            if (
                "warning" in low
                or
                "duplicate" in low
                or
                "vertex" in low
            ):

                warning_lines.append(
                    line
                )

    print()
    print("=" * 112)
    print(
        "Triangle warnings"
    )
    print("=" * 112)

    if warning_lines:

        for line in warning_lines[:100]:
            print(line)

    else:

        print(
            "No warning/duplicate lines found."
        )

    # --------------------------------------------------------
    # Diagnose
    # --------------------------------------------------------

    if isolated.size == 0:

        status = (
            "REVIEW_STAGE7_EDGE_AUDIT_INCONSISTENT"
        )

    elif (
        isolated_round_dup.size
        ==
        isolated.size
    ):

        status = (
            "ISOLATED_POINTS_EXPLAINED_BY_DUPLICATE_XY"
        )

    elif (
        isolated_edge_zone.size
        ==
        isolated.size
    ):

        status = (
            "ISOLATED_POINTS_CONFINED_TO_GRID_BOUNDARY"
        )

    else:

        status = (
            "REVIEW_NON_DUPLICATE_DELAUNAY_ISOLATES"
        )

    payload = {
        "format":
            "pyPSDS-GAMMA-stage7-delaunay-isolate-audit-v09",

        "status":
            status,

        "n_ps":
            int(
                nps
            ),

        "triangle_edges":
            int(
                edges.shape[0]
            ),

        "isolated_ps":
            int(
                isolated.size
            ),

        "exact_duplicate_groups":
            int(
                len(
                    exact_groups
                )
            ),

        "exact_duplicate_members":
            int(
                np.count_nonzero(
                    exact_member
                )
            ),

        "rounded6_duplicate_groups":
            int(
                len(
                    rounded_groups
                )
            ),

        "rounded6_duplicate_members":
            int(
                np.count_nonzero(
                    rounded_member
                )
            ),

        "isolated_exact_duplicate":
            int(
                isolated_exact_dup.size
            ),

        "isolated_rounded6_duplicate":
            int(
                isolated_round_dup.size
            ),

        "isolated_edge_zone":
            int(
                isolated_edge_zone.size
            ),

        "isolated_indices":
            isolated.tolist(),

        "isolated_rows":
            rows[
                isolated
            ].tolist(),

        "isolated_cols":
            cols[
                isolated
            ].tolist(),

        "nearest_distance_m":
            nearest_distance.tolist(),

        "triangle_warning_lines":
            warning_lines[:500],
    }

    OUT.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"audit                      : "
        f"{OUT}"
    )

    print()
    print(
        f"STEP 10R5a0 STATUS: "
        f"{status}"
    )


if __name__ == "__main__":
    main()
