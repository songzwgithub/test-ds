#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


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

    outroot = (
        Path(paths.output_dir)
        / "v09"
    )

    graphdir = (
        outroot
        / "spatial_graph"
    )

    outdir = (
        outroot
        / "unwrap_component_policy"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = np.load(
        outroot
        / "point_phase_stack"
        / "rows.npy",
        mmap_mode="r",
    )

    cols = np.load(
        outroot
        / "point_phase_stack"
        / "cols.npy",
        mmap_mode="r",
    )

    component = np.load(
        graphdir
        / "local_component.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    anchor_u = np.load(
        graphdir
        / "anchor_u.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    anchor_v = np.load(
        graphdir
        / "anchor_v.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    anchor_class = np.load(
        graphdir
        / "anchor_class.npy",
    ).astype(
        np.uint8,
        copy=False,
    )

    anchor_radius = np.load(
        graphdir
        / "anchor_radius.npy",
    ).astype(
        np.uint8,
        copy=False,
    )

    anchor_distance = np.load(
        graphdir
        / "anchor_distance_m.npy",
    ).astype(
        np.float32,
        copy=False,
    )

    npoint = component.size

    if (
        rows.size != npoint
        or cols.size != npoint
    ):
        raise RuntimeError(
            "Point coordinate/component mismatch."
        )

    if not (
        anchor_u.size
        ==
        anchor_v.size
        ==
        anchor_class.size
        ==
        anchor_radius.size
        ==
        anchor_distance.size
    ):
        raise RuntimeError(
            "Anchor-array length mismatch."
        )

    # ========================================================
    # Local component structure
    # ========================================================

    comp_ids, comp_counts = np.unique(
        component,
        return_counts=True,
    )

    ncomp = comp_ids.size

    main_pos = int(
        np.argmax(
            comp_counts
        )
    )

    main_component = int(
        comp_ids[
            main_pos
        ]
    )

    main_count = int(
        comp_counts[
            main_pos
        ]
    )

    main_mask = (
        component
        ==
        main_component
    )

    residual_mask = (
        ~main_mask
    )

    residual_points = int(
        residual_mask.sum()
    )

    residual_components = [
        int(x)
        for x in comp_ids
        if int(x) != main_component
    ]

    print("=" * 88)
    print(
        "Step 08j - Component-wise unwrapping policy"
    )
    print("=" * 88)

    print(
        f"config                    : "
        f"{config_path}"
    )

    print(
        f"points                    : "
        f"{npoint:,}"
    )

    print(
        f"local components          : "
        f"{ncomp}"
    )

    print(
        f"main component            : "
        f"{main_component}"
    )

    print(
        f"main points               : "
        f"{main_count:,} "
        f"({100*main_count/npoint:.4f}%)"
    )

    print(
        f"residual components       : "
        f"{len(residual_components)}"
    )

    print(
        f"residual points           : "
        f"{residual_points:,}"
    )

    # ========================================================
    # Verify every anchor crosses residual <-> main
    # ========================================================

    anchor_component = np.full(
        anchor_u.size,
        -1,
        dtype=np.int32,
    )

    for k in range(
        anchor_u.size
    ):

        u = int(anchor_u[k])
        v = int(anchor_v[k])

        cu = int(component[u])
        cv = int(component[v])

        u_main = (
            cu == main_component
        )

        v_main = (
            cv == main_component
        )

        if u_main == v_main:

            raise RuntimeError(
                "Anchor does not cross exactly one "
                "residual component and the main component: "
                f"edge {k}, components {cu}/{cv}"
            )

        if u_main:
            residual_comp = cv
        else:
            residual_comp = cu

        anchor_component[k] = (
            residual_comp
        )

    # ========================================================
    # Group two anchors by residual component
    # ========================================================

    component_rows = []

    point_tier = np.zeros(
        npoint,
        dtype=np.uint8,
    )

    # 0 = main
    # 1 = normal residual
    # 2 = extended residual
    # 3 = long residual

    n_tier_components = {
        1: 0,
        2: 0,
        3: 0,
    }

    n_tier_points = {
        1: 0,
        2: 0,
        3: 0,
    }

    for comp in residual_components:

        point_ids = np.where(
            component == comp
        )[0]

        aid = np.where(
            anchor_component
            ==
            comp
        )[0]

        if aid.size != 2:

            raise RuntimeError(
                f"Residual component {comp} has "
                f"{aid.size} anchors; expected exactly 2."
            )

        classes = anchor_class[
            aid
        ]

        radii = anchor_radius[
            aid
        ]

        distances = anchor_distance[
            aid
        ]

        # Component policy is determined by its
        # least-local / weakest geometric anchor.
        tier = int(
            classes.max()
        )

        if tier not in (
            1,
            2,
            3,
        ):
            raise RuntimeError(
                f"Unexpected anchor class {tier}"
            )

        point_tier[
            point_ids
        ] = tier

        n_tier_components[
            tier
        ] += 1

        n_tier_points[
            tier
        ] += int(
            point_ids.size
        )

        row = {
            "component_id":
                comp,

            "point_count":
                int(
                    point_ids.size
                ),

            "unwrap_tier":
                tier,

            "unwrap_tier_name":
                {
                    1: "normal",
                    2: "extended",
                    3: "long",
                }[tier],

            "anchor1_id":
                int(
                    aid[0]
                ),

            "anchor1_class":
                int(
                    classes[0]
                ),

            "anchor1_radius":
                int(
                    radii[0]
                ),

            "anchor1_distance_m":
                float(
                    distances[0]
                ),

            "anchor2_id":
                int(
                    aid[1]
                ),

            "anchor2_class":
                int(
                    classes[1]
                ),

            "anchor2_radius":
                int(
                    radii[1]
                ),

            "anchor2_distance_m":
                float(
                    distances[1]
                ),

            "max_anchor_radius":
                int(
                    radii.max()
                ),

            "max_anchor_distance_m":
                float(
                    distances.max()
                ),
        }

        component_rows.append(
            row
        )

    # ========================================================
    # QA
    # ========================================================

    if np.any(
        point_tier[
            main_mask
        ]
        != 0
    ):
        raise RuntimeError(
            "Main component received residual tier."
        )

    if np.any(
        point_tier[
            residual_mask
        ]
        == 0
    ):
        raise RuntimeError(
            "Residual points missing tier."
        )

    if len(component_rows) != 102:

        print(
            "WARNING: residual component count "
            f"is {len(component_rows)}, not historical 102."
        )

    print()
    print("=" * 88)
    print(
        "Residual attachment policy"
    )
    print("=" * 88)

    print(
        f"normal components          : "
        f"{n_tier_components[1]:3d}, "
        f"points={n_tier_points[1]:4d}"
    )

    print(
        f"extended components        : "
        f"{n_tier_components[2]:3d}, "
        f"points={n_tier_points[2]:4d}"
    )

    print(
        f"long components            : "
        f"{n_tier_components[3]:3d}, "
        f"points={n_tier_points[3]:4d}"
    )

    print()
    print(
        "Unwrapping policy:"
    )

    print(
        "  1. Unwrap each R4-K8 local component "
        "using LOCAL EDGES ONLY."
    )

    print(
        "  2. Never use residual anchors for "
        "phase propagation inside the main component."
    )

    print(
        "  3. After component unwrapping, use the "
        "two anchors only to estimate/check the "
        "integer 2pi component offset."
    )

    print(
        "  4. normal/extended/long remain separate "
        "confidence classes."
    )

    # ========================================================
    # Save
    # ========================================================

    np.save(
        outdir
        / "local_component.npy",
        component,
    )

    np.save(
        outdir
        / "point_unwrap_tier.npy",
        point_tier,
    )

    np.save(
        outdir
        / "main_component_mask.npy",
        main_mask,
    )

    np.save(
        outdir
        / "anchor_component.npy",
        anchor_component,
    )

    csv_path = (
        outdir
        / "residual_component_policy.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                component_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            component_rows
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-unwrapping-component-policy-v0.9",

        "status":
            "FROZEN",

        "points":
            int(npoint),

        "local_components":
            int(ncomp),

        "main_component": {
            "id":
                int(main_component),

            "points":
                int(main_count),

            "fraction":
                float(
                    main_count
                    /
                    npoint
                ),
        },

        "residual": {
            "components":
                int(
                    len(
                        residual_components
                    )
                ),

            "points":
                int(
                    residual_points
                ),

            "normal": {
                "components":
                    int(
                        n_tier_components[1]
                    ),

                "points":
                    int(
                        n_tier_points[1]
                    ),
            },

            "extended": {
                "components":
                    int(
                        n_tier_components[2]
                    ),

                "points":
                    int(
                        n_tier_points[2]
                    ),
            },

            "long": {
                "components":
                    int(
                        n_tier_components[3]
                    ),

                "points":
                    int(
                        n_tier_points[3]
                    ),
            },
        },

        "policy": {
            "within_component_edges":
                "R4-K8 local edges only",

            "anchors_used_during_core_unwrap":
                False,

            "anchor_role":
                (
                    "post-unwrapping integer "
                    "2pi component registration"
                ),

            "anchor_consistency":
                "two independent anchors",
        },
    }

    manifest_path = (
        outdir
        / "unwrap_component_policy.json"
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
        f"component policy table    : "
        f"{csv_path}"
    )

    print(
        f"manifest                  : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP 08j STATUS: PASS / "
        "UNWRAPPING COMPONENT POLICY FROZEN"
    )


if __name__ == "__main__":
    main()
