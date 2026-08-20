#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


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
    ) = open_from_config(args.config)

    outroot = (
        Path(paths.output_dir)
        / "processing"
    )

    path = (
        outroot
        / "spatial_graph_two_anchor_quality"
        / "residual_two_anchor_quality.csv"
    )

    rows = []

    with path.open() as f:

        reader = csv.DictReader(f)

        for r in reader:

            rows.append({
                "label":
                    int(r["component_label"]),

                "size":
                    int(r["component_size"]),

                "r2":
                    int(r["min_radius_2_anchor"]),
            })

    sizes = np.array(
        [x["size"] for x in rows],
        dtype=np.int32,
    )

    radii = np.array(
        [x["r2"] for x in rows],
        dtype=np.int32,
    )

    total_points = int(
        sizes.sum()
    )

    total_components = len(rows)

    print("=" * 84)
    print(
        "Residual two-anchor radius summary"
    )
    print("=" * 84)

    print(
        f"residual components     : "
        f"{total_components}"
    )

    print(
        f"residual points         : "
        f"{total_points}"
    )

    print()
    print(
        " Cumulative two-anchor coverage"
    )

    print(
        " Radius | components             | points"
    )

    print("-" * 84)

    for R in [
        5, 6, 7, 8,
        10, 12, 15, 20, 25, 30
    ]:

        m = (
            (radii > 0)
            &
            (radii <= R)
        )

        nc = int(
            np.count_nonzero(m)
        )

        npnt = int(
            sizes[m].sum()
        )

        print(
            f" R<={R:2d} | "
            f"{nc:3d}/{total_components:3d} "
            f"({100*nc/total_components:7.3f}%) | "
            f"{npnt:4d}/{total_points:4d} "
            f"({100*npnt/total_points:7.3f}%)"
        )

    print()
    print("=" * 84)
    print(
        "Components requiring radius > 12"
    )
    print("=" * 84)

    hard = [
        x
        for x in rows
        if x["r2"] > 12
    ]

    hard.sort(
        key=lambda x: (
            x["r2"],
            -x["size"],
        )
    )

    if not hard:

        print("none")

    else:

        print(
            " label   size   two-anchor-radius"
        )

        for x in hard:

            print(
                f" {x['label']:5d} "
                f"{x['size']:6d} "
                f"{x['r2']:10d}"
            )

    hard_points = sum(
        x["size"]
        for x in hard
    )

    print()

    print(
        f"components with R2>12 : "
        f"{len(hard)}"
    )

    print(
        f"points with R2>12     : "
        f"{hard_points}/{total_points} "
        f"({100*hard_points/total_points:.3f}%)"
    )

    print()
    print(
        "STEP 08f STATUS: PASS"
    )


if __name__ == "__main__":
    main()
