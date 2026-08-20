#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--max-radius",
        type=int,
        default=30,
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

    outroot = (
        Path(paths.output_dir)
        / "processing"
    )

    pps_dir = (
        outroot
        / "point_phase_stack"
    )

    quality_dir = (
        outroot
        / "spatial_graph_residual_quality"
    )

    outdir = (
        outroot
        / "spatial_graph_two_anchor_quality"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = np.load(
        pps_dir / "rows.npy"
    ).astype(np.int32)

    cols = np.load(
        pps_dir / "cols.npy"
    ).astype(np.int32)

    labels = np.load(
        quality_dir
        / "component_label_r4.npy"
    ).astype(np.int32)

    npoint = rows.size

    if (
        cols.size != npoint
        or
        labels.size != npoint
    ):
        raise RuntimeError(
            "rows/cols/component length mismatch."
        )

    H = int(rows.max()) + 1
    W = int(cols.max()) + 1

    # --------------------------------------------------------
    # Determine main R=4 component
    # --------------------------------------------------------

    comp_counts = np.bincount(
        labels
    )

    main_label = int(
        np.argmax(
            comp_counts
        )
    )

    main_mask_point = (
        labels == main_label
    )

    residual_mask_point = (
        ~main_mask_point
    )

    n_main = int(
        main_mask_point.sum()
    )

    n_residual = int(
        residual_mask_point.sum()
    )

    residual_labels = np.unique(
        labels[
            residual_mask_point
        ]
    )

    # --------------------------------------------------------
    # Radar-grid lookup
    # --------------------------------------------------------

    index_grid = np.full(
        (H, W),
        -1,
        dtype=np.int32,
    )

    index_grid[
        rows,
        cols
    ] = np.arange(
        npoint,
        dtype=np.int32,
    )

    main_grid = np.zeros(
        (H, W),
        dtype=bool,
    )

    main_grid[
        rows[main_mask_point],
        cols[main_mask_point]
    ] = True

    print("=" * 92)
    print(
        "Step 08e - Residual component two-anchor quality"
    )
    print("=" * 92)

    print(
        f"config                     : {config_path}"
    )

    print(
        f"points                     : {npoint:,}"
    )

    print(
        f"R=4 main component        : "
        f"{n_main:,} "
        f"({100*n_main/npoint:.4f}%)"
    )

    print(
        f"residual points           : "
        f"{n_residual:,}"
    )

    print(
        f"residual components       : "
        f"{len(residual_labels)}"
    )

    print(
        f"maximum search radius     : "
        f"{args.max_radius}"
    )

    # ========================================================
    # For each residual component, enumerate possible
    # crossings to the main component.
    # ========================================================

    results = []

    for icomp, lab in enumerate(
        residual_labels,
        start=1,
    ):

        ids = np.where(
            labels == lab
        )[0]

        # Dictionary:
        # (residual point, main point) -> distance
        crossing = {}

        for p in ids:

            rp = int(rows[p])
            cp = int(cols[p])

            r0 = max(
                0,
                rp - args.max_radius,
            )

            r1 = min(
                H,
                rp + args.max_radius + 1,
            )

            c0 = max(
                0,
                cp - args.max_radius,
            )

            c1 = min(
                W,
                cp + args.max_radius + 1,
            )

            rr, cc = np.where(
                main_grid[
                    r0:r1,
                    c0:c1
                ]
            )

            if rr.size == 0:
                continue

            rr = (
                rr + r0
            )

            cc = (
                cc + c0
            )

            for rq, cq in zip(
                rr.tolist(),
                cc.tolist(),
            ):

                d = max(
                    abs(rp - rq),
                    abs(cp - cq),
                )

                if d > args.max_radius:
                    continue

                q = int(
                    index_grid[
                        rq,
                        cq
                    ]
                )

                if q < 0:
                    continue

                key = (
                    int(p),
                    q,
                )

                old = crossing.get(
                    key
                )

                if (
                    old is None
                    or
                    d < old
                ):
                    crossing[key] = int(d)

        edges = [
            (
                p,
                q,
                d,
            )
            for (
                p,
                q,
            ), d in crossing.items()
        ]

        edges.sort(
            key=lambda x: (
                x[2],
                x[0],
                x[1],
            )
        )

        # ----------------------------------------------------
        # Find first radius with:
        # >=1 crossing
        # >=2 crossings to >=2 distinct main anchors
        # >=3 crossings to >=3 distinct main anchors
        # ----------------------------------------------------

        min_r1 = None
        min_r2 = None
        min_r3 = None

        best2 = []
        best3 = []

        for radius in range(
            5,
            args.max_radius + 1,
        ):

            ee = [
                e
                for e in edges
                if e[2] <= radius
            ]

            main_anchors = {
                e[1]
                for e in ee
            }

            if (
                min_r1 is None
                and
                len(ee) >= 1
            ):
                min_r1 = radius

            if (
                min_r2 is None
                and
                len(ee) >= 2
                and
                len(main_anchors) >= 2
            ):

                min_r2 = radius

                # Pick two shortest edges using
                # two different main anchors.
                used_main = set()

                for e in ee:

                    if e[1] in used_main:
                        continue

                    best2.append(e)
                    used_main.add(e[1])

                    if len(best2) == 2:
                        break

            if (
                min_r3 is None
                and
                len(ee) >= 3
                and
                len(main_anchors) >= 3
            ):

                min_r3 = radius

                used_main = set()

                for e in ee:

                    if e[1] in used_main:
                        continue

                    best3.append(e)
                    used_main.add(e[1])

                    if len(best3) == 3:
                        break

            if (
                min_r1 is not None
                and
                min_r2 is not None
                and
                min_r3 is not None
            ):
                break

        row = {
            "component_label":
                int(lab),

            "component_size":
                int(ids.size),

            "crossing_edges_Rmax":
                int(len(edges)),

            "distinct_main_anchors_Rmax":
                int(
                    len({
                        e[1]
                        for e in edges
                    })
                ),

            "min_radius_1_anchor":
                (
                    int(min_r1)
                    if min_r1 is not None
                    else -1
                ),

            "min_radius_2_anchor":
                (
                    int(min_r2)
                    if min_r2 is not None
                    else -1
                ),

            "min_radius_3_anchor":
                (
                    int(min_r3)
                    if min_r3 is not None
                    else -1
                ),
        }

        # Save actual recommended first two links.
        for k in range(2):

            if k < len(best2):

                p, q, d = best2[k]

                row[
                    f"anchor{k+1}_radius"
                ] = int(d)

                row[
                    f"anchor{k+1}_residual_point"
                ] = int(p)

                row[
                    f"anchor{k+1}_residual_row"
                ] = int(rows[p])

                row[
                    f"anchor{k+1}_residual_col"
                ] = int(cols[p])

                row[
                    f"anchor{k+1}_main_point"
                ] = int(q)

                row[
                    f"anchor{k+1}_main_row"
                ] = int(rows[q])

                row[
                    f"anchor{k+1}_main_col"
                ] = int(cols[q])

            else:

                row[
                    f"anchor{k+1}_radius"
                ] = -1

                row[
                    f"anchor{k+1}_residual_point"
                ] = -1

                row[
                    f"anchor{k+1}_residual_row"
                ] = -1

                row[
                    f"anchor{k+1}_residual_col"
                ] = -1

                row[
                    f"anchor{k+1}_main_point"
                ] = -1

                row[
                    f"anchor{k+1}_main_row"
                ] = -1

                row[
                    f"anchor{k+1}_main_col"
                ] = -1

        results.append(
            row
        )

        if (
            icomp == 1
            or
            icomp % 20 == 0
            or
            icomp == len(
                residual_labels
            )
        ):

            print(
                f"  component "
                f"{icomp:3d}/"
                f"{len(residual_labels):3d}"
            )

    # ========================================================
    # Summary
    # ========================================================

    r1 = np.array(
        [
            x["min_radius_1_anchor"]
            for x in results
        ],
        dtype=np.int32,
    )

    r2 = np.array(
        [
            x["min_radius_2_anchor"]
            for x in results
        ],
        dtype=np.int32,
    )

    r3 = np.array(
        [
            x["min_radius_3_anchor"]
            for x in results
        ],
        dtype=np.int32,
    )

    print()
    print("=" * 92)
    print(
        "Residual component anchor summary"
    )
    print("=" * 92)

    radii_to_report = [
        5,
        6,
        7,
        8,
        10,
        12,
        15,
        20,
        25,
        30,
    ]

    print()
    print(
        "Components with >=2 independent "
        "main-component anchors:"
    )

    for radius in radii_to_report:

        n = int(
            np.count_nonzero(
                (r2 > 0)
                &
                (r2 <= radius)
            )
        )

        print(
            f"  R <= {radius:2d}: "
            f"{n:3d}/"
            f"{len(results):3d}"
        )

    valid2 = r2[
        r2 > 0
    ]

    print()

    if valid2.size:

        print(
            "2-anchor radius "
            "min/p25/p50/p75/p90/p95/p99/max:"
        )

        q = np.quantile(
            valid2,
            [
                0.00,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.00,
            ]
        )

        print(
            "  "
            + " / ".join(
                f"{x:.1f}"
                for x in q
            )
        )

    unresolved2 = int(
        np.count_nonzero(
            r2 < 0
        )
    )

    unresolved3 = int(
        np.count_nonzero(
            r3 < 0
        )
    )

    print()

    print(
        f"no 2-anchor solution <=R{args.max_radius}: "
        f"{unresolved2}"
    )

    print(
        f"no 3-anchor solution <=R{args.max_radius}: "
        f"{unresolved3}"
    )

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / "residual_two_anchor_quality.csv"
    )

    fieldnames = list(
        results[0].keys()
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        w.writeheader()
        w.writerows(
            results
        )

    summary = {
        "format":
            "pyPSDS-GAMMA-residual-two-anchor-quality-v1.0",

        "core_radius": 4,

        "main_points":
            n_main,

        "residual_points":
            n_residual,

        "residual_components":
            len(results),

        "maximum_search_radius":
            args.max_radius,

        "two_anchor": {
            "resolved_components":
                int(
                    np.count_nonzero(
                        r2 > 0
                    )
                ),

            "unresolved_components":
                unresolved2,

            "minimum_radius":
                (
                    int(valid2.min())
                    if valid2.size
                    else None
                ),

            "median_radius":
                (
                    float(
                        np.median(valid2)
                    )
                    if valid2.size
                    else None
                ),

            "maximum_radius":
                (
                    int(valid2.max())
                    if valid2.size
                    else None
                ),
        },

        "three_anchor": {
            "resolved_components":
                int(
                    np.count_nonzero(
                        r3 > 0
                    )
                ),

            "unresolved_components":
                unresolved3,
        },
    }

    json_path = (
        outdir
        / "residual_two_anchor_quality.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print(
        f"component table           : "
        f"{csv_path}"
    )

    print(
        f"manifest                  : "
        f"{json_path}"
    )

    print()
    print(
        "STEP 08e STATUS: PASS"
    )


if __name__ == "__main__":
    main()
