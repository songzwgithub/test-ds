#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from pypsds.context import open_from_config


def component_stats(labels, ncomp):
    if ncomp == 0:
        return {
            "components": 0,
            "largest": 0,
            "second_largest": 0,
            "largest_fraction": 0.0,
            "singleton_components": 0,
        }

    counts = np.bincount(
        labels.ravel(),
        minlength=ncomp + 1,
    )[1:]

    order = np.sort(counts)[::-1]

    largest = int(order[0])

    second = (
        int(order[1])
        if len(order) > 1
        else 0
    )

    total = int(
        counts.sum()
    )

    return {
        "components": int(ncomp),
        "largest": largest,
        "second_largest": second,
        "largest_fraction": (
            largest / total
            if total > 0
            else 0.0
        ),
        "singleton_components": int(
            np.sum(counts == 1)
        ),
    }


def compute_degree8(mask):
    """
    Number of occupied 8-neighbors for each
    occupied radar-grid pixel.
    """

    degree = np.zeros(
        mask.shape,
        dtype=np.uint8,
    )

    offsets = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    ]

    H, W = mask.shape

    for dr, dc in offsets:

        r_src0 = max(
            0,
            -dr,
        )

        r_src1 = min(
            H,
            H - dr,
        )

        c_src0 = max(
            0,
            -dc,
        )

        c_src1 = min(
            W,
            W - dc,
        )

        r_dst0 = r_src0 + dr
        r_dst1 = r_src1 + dr

        c_dst0 = c_src0 + dc
        c_dst1 = c_src1 + dc

        degree[
            r_src0:r_src1,
            c_src0:c_src1
        ] += mask[
            r_dst0:r_dst1,
            c_dst0:c_dst1
        ]

    degree[
        ~mask
    ] = 0

    return degree


def compute_degree4(mask):
    degree = np.zeros(
        mask.shape,
        dtype=np.uint8,
    )

    offsets = [
        (-1, 0),
        ( 1, 0),
        ( 0,-1),
        ( 0, 1),
    ]

    H, W = mask.shape

    for dr, dc in offsets:

        r_src0 = max(
            0,
            -dr,
        )

        r_src1 = min(
            H,
            H - dr,
        )

        c_src0 = max(
            0,
            -dc,
        )

        c_src1 = min(
            W,
            W - dc,
        )

        r_dst0 = r_src0 + dr
        r_dst1 = r_src1 + dr

        c_dst0 = c_src0 + dc
        c_dst1 = c_src1 + dc

        degree[
            r_src0:r_src1,
            c_src0:c_src1
        ] += mask[
            r_dst0:r_dst1,
            c_dst0:c_dst1
        ]

    degree[
        ~mask
    ] = 0

    return degree


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
        _roi,
    ) = open_from_config(
        args.config
    )

    outroot = (
        Path(paths.output_dir)
        / "processing"
    )

    pps_dir = (
        outroot
        / "point_phase_stack"
    )

    qa_dir = (
        outroot
        / "spatial_graph_quality"
    )

    qa_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_path = (
        pps_dir
        / "rows.npy"
    )

    cols_path = (
        pps_dir
        / "cols.npy"
    )

    phase_path = (
        pps_dir
        / "phase_rad.npy"
    )

    for p in (
        rows_path,
        cols_path,
        phase_path,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    rows = np.load(
        rows_path,
        mmap_mode="r",
    )

    cols = np.load(
        cols_path,
        mmap_mode="r",
    )

    phase = np.load(
        phase_path,
        mmap_mode="r",
    )

    if rows.ndim != 1:
        raise RuntimeError(
            f"rows.npy must be 1-D: {rows.shape}"
        )

    if cols.ndim != 1:
        raise RuntimeError(
            f"cols.npy must be 1-D: {cols.shape}"
        )

    npoint = len(rows)

    if len(cols) != npoint:
        raise RuntimeError(
            "rows/cols length mismatch."
        )

    if phase.shape[0] != npoint:
        raise RuntimeError(
            "PointPhaseStack point count mismatch."
        )

    r = np.asarray(
        rows,
        dtype=np.int64,
    )

    c = np.asarray(
        cols,
        dtype=np.int64,
    )

    if np.any(r < 0):
        raise RuntimeError(
            "Negative row coordinate."
        )

    if np.any(c < 0):
        raise RuntimeError(
            "Negative column coordinate."
        )

    # --------------------------------------------------------
    # Exact coordinate uniqueness
    # --------------------------------------------------------

    key = (
        (r.astype(np.uint64) << np.uint64(32))
        |
        c.astype(np.uint64)
    )

    unique_count = int(
        np.unique(key).size
    )

    if unique_count != npoint:
        raise RuntimeError(
            f"Duplicate radar coordinates: "
            f"{npoint-unique_count}"
        )

    H = int(
        r.max()
    ) + 1

    W = int(
        c.max()
    ) + 1

    mask = np.zeros(
        (H, W),
        dtype=bool,
    )

    mask[
        r,
        c
    ] = True

    occupancy = (
        npoint
        /
        (H * W)
    )

    print("=" * 80)
    print(
        "Spatial PS/DS point-graph quality"
    )
    print("=" * 80)

    print(
        f"config                     : {config_path}"
    )

    print(
        f"points                     : {npoint:,}"
    )

    print(
        f"radar-grid extent          : "
        f"{H} rows x {W} cols"
    )

    print(
        f"occupied fraction          : "
        f"{100*occupancy:.3f}%"
    )

    print(
        f"unique coordinates         : "
        f"{unique_count:,}/{npoint:,}"
    )

    # ========================================================
    # 4-neighbor topology
    # ========================================================

    structure4 = np.array(
        [
            [0,1,0],
            [1,1,1],
            [0,1,0],
        ],
        dtype=np.uint8,
    )

    labels4, ncomp4 = ndimage.label(
        mask,
        structure=structure4,
    )

    stat4 = component_stats(
        labels4,
        ncomp4,
    )

    deg4_grid = compute_degree4(
        mask
    )

    deg4 = deg4_grid[
        r,
        c
    ]

    # ========================================================
    # 8-neighbor topology
    # ========================================================

    structure8 = np.ones(
        (3,3),
        dtype=np.uint8,
    )

    labels8, ncomproduction = ndimage.label(
        mask,
        structure=structure8,
    )

    stat8 = component_stats(
        labels8,
        ncomproduction,
    )

    deg8_grid = compute_degree8(
        mask
    )

    deg8 = deg8_grid[
        r,
        c
    ]

    point_component8 = labels8[
        r,
        c
    ].astype(
        np.int32
    )

    # ========================================================
    # Degree distributions
    # ========================================================

    print()
    print("=" * 80)
    print(
        "4-neighbor connectivity"
    )
    print("=" * 80)

    print(
        f"components                 : "
        f"{stat4['components']:,}"
    )

    print(
        f"largest component          : "
        f"{stat4['largest']:,} "
        f"({100*stat4['largest_fraction']:.3f}%)"
    )

    print(
        f"second-largest component   : "
        f"{stat4['second_largest']:,}"
    )

    print(
        f"singleton components       : "
        f"{stat4['singleton_components']:,}"
    )

    print(
        f"degree min/median/max      : "
        f"{deg4.min()} / "
        f"{np.median(deg4):.1f} / "
        f"{deg4.max()}"
    )

    print(
        f"degree=0 points            : "
        f"{np.count_nonzero(deg4 == 0):,}"
    )

    print()
    print("=" * 80)
    print(
        "8-neighbor connectivity"
    )
    print("=" * 80)

    print(
        f"components                 : "
        f"{stat8['components']:,}"
    )

    print(
        f"largest component          : "
        f"{stat8['largest']:,} "
        f"({100*stat8['largest_fraction']:.3f}%)"
    )

    print(
        f"second-largest component   : "
        f"{stat8['second_largest']:,}"
    )

    print(
        f"singleton components       : "
        f"{stat8['singleton_components']:,}"
    )

    print(
        f"degree min/median/max      : "
        f"{deg8.min()} / "
        f"{np.median(deg8):.1f} / "
        f"{deg8.max()}"
    )

    print(
        f"degree=0 points            : "
        f"{np.count_nonzero(deg8 == 0):,}"
    )

    print(
        f"degree>=3 points           : "
        f"{np.count_nonzero(deg8 >= 3):,} "
        f"({100*np.mean(deg8 >= 3):.3f}%)"
    )

    print(
        f"degree>=5 points           : "
        f"{np.count_nonzero(deg8 >= 5):,} "
        f"({100*np.mean(deg8 >= 5):.3f}%)"
    )

    # ========================================================
    # Component-size distribution
    # ========================================================

    counts8 = np.bincount(
        labels8.ravel(),
        minlength=ncomproduction + 1,
    )[1:]

    small_10 = int(
        np.sum(
            counts8 <= 10
        )
    )

    small_100 = int(
        np.sum(
            counts8 <= 100
        )
    )

    points_outside_largest = (
        npoint
        -
        stat8["largest"]
    )

    print()
    print(
        f"8-neighbor components <=10 : "
        f"{small_10:,}"
    )

    print(
        f"8-neighbor components <=100: "
        f"{small_100:,}"
    )

    print(
        f"points outside largest comp : "
        f"{points_outside_largest:,} "
        f"({100*points_outside_largest/npoint:.4f}%)"
    )

    # ========================================================
    # Save products
    # ========================================================

    np.save(
        qa_dir
        / "degree4.npy",
        deg4.astype(
            np.uint8
        ),
    )

    np.save(
        qa_dir
        / "degree8.npy",
        deg8.astype(
            np.uint8
        ),
    )

    np.save(
        qa_dir
        / "component8.npy",
        point_component8,
    )

    np.save(
        qa_dir
        / "occupancy_mask.npy",
        mask,
    )

    summary = {
        "format": (
            "pyPSDS-GAMMA-spatial-"
            "graph-quality-v1.0"
        ),

        "points": int(
            npoint
        ),

        "radar_grid_extent": {
            "rows": H,
            "cols": W,
            "occupied_fraction": float(
                occupancy
            ),
        },

        "coordinate_unique": (
            unique_count == npoint
        ),

        "four_neighbor": {
            **stat4,

            "degree_min": int(
                deg4.min()
            ),

            "degree_median": float(
                np.median(deg4)
            ),

            "degree_max": int(
                deg4.max()
            ),

            "degree_zero_points": int(
                np.count_nonzero(
                    deg4 == 0
                )
            ),
        },

        "eight_neighbor": {
            **stat8,

            "degree_min": int(
                deg8.min()
            ),

            "degree_median": float(
                np.median(deg8)
            ),

            "degree_max": int(
                deg8.max()
            ),

            "degree_zero_points": int(
                np.count_nonzero(
                    deg8 == 0
                )
            ),

            "degree_ge_3_points": int(
                np.count_nonzero(
                    deg8 >= 3
                )
            ),

            "degree_ge_5_points": int(
                np.count_nonzero(
                    deg8 >= 5
                )
            ),

            "points_outside_largest": int(
                points_outside_largest
            ),
        },
    }

    manifest = (
        qa_dir
        / "spatial_graph_quality.json"
    )

    manifest.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print(
        f"degree8                   : "
        f"{qa_dir/'degree8.npy'}"
    )

    print(
        f"component8                : "
        f"{qa_dir/'component8.npy'}"
    )

    print(
        f"quality manifest            : "
        f"{manifest}"
    )

    print()
    print(
        "STEP spatial_graph_quality STATUS: PASS"
    )


if __name__ == "__main__":
    main()
