#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
                    f"Invalid ITAB: {raw}"
                )

            edges.append((i, j))

    return edges


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.s = [1] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[
                self.p[x]
            ]
            x = self.p[x]

        return x

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        if self.s[a] < self.s[b]:
            a, b = b, a

        self.p[b] = a
        self.s[a] += self.s[b]

        return True


def build_design_matrix(
    edges,
    ndate,
    reference_idx=0,
):
    """
    y_ij = theta_j - theta_i

    theta(reference_idx) = 0

    Therefore A has Ndate-1 unknowns.
    """

    col = {}

    c = 0

    for t in range(ndate):
        if t == reference_idx:
            continue

        col[t] = c
        c += 1

    A = np.zeros(
        (
            len(edges),
            ndate - 1,
        ),
        dtype=np.float64,
    )

    for e, (i, j) in enumerate(edges):

        if i != reference_idx:
            A[
                e,
                col[i]
            ] -= 1.0

        if j != reference_idx:
            A[
                e,
                col[j]
            ] += 1.0

    return A


def spanning_tree_edge_ids(
    edges,
    ndate,
):
    """
    Deterministic temporal spanning tree:
    production network edge order.
    """

    dsu = DSU(
        ndate
    )

    tree = []

    for eid, (i, j) in enumerate(edges):

        if dsu.union(i, j):
            tree.append(eid)

    if len(tree) != ndate - 1:
        raise RuntimeError(
            f"Temporal network not connected: "
            f"tree={len(tree)}, "
            f"expected={ndate-1}"
        )

    return np.asarray(
        tree,
        dtype=np.int32,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=12000,
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

    final_dir = (
        root
        / "final_unwrap"
    )

    outdir = (
        root
        / "network_inversion"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Inputs
    # ========================================================

    phase_pl = np.load(
        pps_dir
        / "phase_rad.npy",
        mmap_mode="r",
    )

    rows = np.load(
        pps_dir
        / "rows.npy",
        mmap_mode="r",
    )

    cols = np.load(
        pps_dir
        / "cols.npy",
        mmap_mode="r",
    )

    strict = np.asarray(
        np.load(
            final_dir
            / "strict_unwrap_valid_mask.npy",
        ),
        dtype=bool,
    )

    gauge = np.asarray(
        np.load(
            final_dir
            / "global_ifg_integer_delta.npy",
        ),
        dtype=np.int32,
    )

    npoint, ndate = phase_pl.shape

    edges = load_itab(
        network_dir
        / "network.itab",
        ndate,
    )

    nifg = len(edges)

    if gauge.size != nifg:
        raise RuntimeError(
            "global gauge size mismatch"
        )

    strict_ids = np.where(
        strict
    )[0].astype(
        np.int32
    )

    nstrict = strict_ids.size

    # ========================================================
    # Temporal design matrix
    # ========================================================

    reference_idx = 0

    A = build_design_matrix(
        edges,
        ndate,
        reference_idx,
    )

    rank_A = int(
        np.linalg.matrix_rank(
            A
        )
    )

    expected_rank = (
        ndate - 1
    )

    if rank_A != expected_rank:
        raise RuntimeError(
            f"Design matrix rank "
            f"{rank_A} != {expected_rank}"
        )

    tree_ids = spanning_tree_edge_ids(
        edges,
        ndate,
    )

    A_tree = A[
        tree_ids,
        :
    ]

    rank_tree = int(
        np.linalg.matrix_rank(
            A_tree
        )
    )

    if rank_tree != expected_rank:
        raise RuntimeError(
            "Temporal tree matrix "
            "is rank deficient."
        )

    # Tree inverse.
    A_tree_inv = np.linalg.inv(
        A_tree
    )

    # Full-network ordinary L2 operator:
    #
    # theta = pinv(A) y
    P_l2 = np.linalg.pinv(
        A,
        rcond=1e-12,
    )

    cond_A = float(
        np.linalg.cond(
            A
        )
    )

    cond_tree = float(
        np.linalg.cond(
            A_tree
        )
    )

    # ========================================================
    # Open IFG maps
    # ========================================================

    ifg_maps = []

    for pair_id, (i, j) in enumerate(
        edges,
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

        if arr.size != npoint:
            raise RuntimeError(
                f"{path.name}: "
                "point count mismatch"
            )

        ifg_maps.append(
            arr
        )

    # ========================================================
    # Candidate output
    #
    # Rows correspond exactly to strict_point_ids.npy.
    # ========================================================

    phase_out_path = (
        outdir
        / "acquisition_phase_l2_candidate_rad.npy"
    )

    phase_out = np.lib.format.open_memmap(
        phase_out_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            nstrict,
            ndate,
        ),
    )

    max_tree_l2_by_point = np.lib.format.open_memmap(
        outdir
        / "tree_l2_max_abs_diff_rad.npy",
        mode="w+",
        dtype=np.float32,
        shape=(nstrict,),
    )

    l2_residual_rms = np.lib.format.open_memmap(
        outdir
        / "l2_network_residual_rms_rad.npy",
        mode="w+",
        dtype=np.float32,
        shape=(nstrict,),
    )

    l2_residual_max = np.lib.format.open_memmap(
        outdir
        / "l2_network_residual_max_rad.npy",
        mode="w+",
        dtype=np.float32,
        shape=(nstrict,),
    )

    # ========================================================
    # Global QA accumulators
    # ========================================================

    global_tree_l2_max = 0.0
    global_tree_l2_ss = 0.0
    global_tree_l2_n = 0

    global_l2_residual_max = 0.0
    global_l2_residual_ss = 0.0
    global_l2_residual_n = 0

    global_tree_residual_max = 0.0

    global_wrap_parity_max = 0.0
    global_wrap_parity_ss = 0.0
    global_wrap_parity_n = 0

    print("=" * 108)
    print(
        "Step 09a - Temporal network inversion "
        "tree vs full-network L2 parity"
    )
    print("=" * 108)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"points                     : "
        f"{npoint:,}"
    )

    print(
        f"strict points              : "
        f"{nstrict:,}"
    )

    print(
        f"acquisitions               : "
        f"{ndate}"
    )

    print(
        f"IFGs                       : "
        f"{nifg}"
    )

    print(
        f"reference acquisition      : "
        f"{stack.dates[reference_idx]}"
    )

    print(
        f"unknown acquisition phases : "
        f"{ndate-1}"
    )

    print(
        f"rank(A)                    : "
        f"{rank_A}"
    )

    print(
        f"temporal tree edges        : "
        f"{tree_ids.size}"
    )

    print(
        f"condition number A         : "
        f"{cond_A:.6f}"
    )

    print(
        f"condition number A_tree    : "
        f"{cond_tree:.6f}"
    )

    print()
    print(
        "Streaming strict points ..."
    )

    # ========================================================
    # Batch inversion
    # ========================================================

    for b0 in range(
        0,
        nstrict,
        args.batch_size,
    ):

        b1 = min(
            b0
            +
            args.batch_size,
            nstrict,
        )

        ids = strict_ids[
            b0:b1
        ]

        B = ids.size

        Y = np.empty(
            (
                B,
                nifg,
            ),
            dtype=np.float64,
        )

        for e in range(
            nifg
        ):

            Y[:, e] = (
                np.asarray(
                    ifg_maps[e][
                        ids
                    ],
                    dtype=np.float64,
                )
                +
                TWOPI
                *
                gauge[e]
            )

        # ----------------------------------------------------
        # Tree solution
        # ----------------------------------------------------

        Y_tree = Y[
            :,
            tree_ids
        ]

        theta_tree = (
            Y_tree
            @
            A_tree_inv.T
        )

        # ----------------------------------------------------
        # Full-network ordinary L2
        # ----------------------------------------------------

        theta_l2 = (
            Y
            @
            P_l2.T
        )

        # ----------------------------------------------------
        # Tree / L2 parity
        # ----------------------------------------------------

        diff = (
            theta_tree
            -
            theta_l2
        )

        abs_diff = np.abs(
            diff
        )

        point_max_diff = np.max(
            abs_diff,
            axis=1,
        )

        max_tree_l2_by_point[
            b0:b1
        ] = point_max_diff.astype(
            np.float32
        )

        global_tree_l2_max = max(
            global_tree_l2_max,
            float(
                abs_diff.max()
            ),
        )

        global_tree_l2_ss += float(
            np.sum(
                diff * diff
            )
        )

        global_tree_l2_n += int(
            diff.size
        )

        # ----------------------------------------------------
        # Network residual:
        #
        # y - A theta
        # ----------------------------------------------------

        Yhat_l2 = (
            theta_l2
            @
            A.T
        )

        residual_l2 = (
            Y
            -
            Yhat_l2
        )

        point_rms = np.sqrt(
            np.mean(
                residual_l2
                *
                residual_l2,
                axis=1,
            )
        )

        point_rmax = np.max(
            np.abs(
                residual_l2
            ),
            axis=1,
        )

        l2_residual_rms[
            b0:b1
        ] = point_rms.astype(
            np.float32
        )

        l2_residual_max[
            b0:b1
        ] = point_rmax.astype(
            np.float32
        )

        global_l2_residual_max = max(
            global_l2_residual_max,
            float(
                np.max(
                    np.abs(
                        residual_l2
                    )
                )
            ),
        )

        global_l2_residual_ss += float(
            np.sum(
                residual_l2
                *
                residual_l2
            )
        )

        global_l2_residual_n += int(
            residual_l2.size
        )

        # Tree solution residual against ALL 108 IFGs.
        residual_tree = (
            Y
            -
            theta_tree
            @
            A.T
        )

        global_tree_residual_max = max(
            global_tree_residual_max,
            float(
                np.max(
                    np.abs(
                        residual_tree
                    )
                )
            ),
        )

        # ----------------------------------------------------
        # Restore reference acquisition = 0
        # ----------------------------------------------------

        theta_full = np.zeros(
            (
                B,
                ndate,
            ),
            dtype=np.float64,
        )

        theta_full[
            :,
            1:
        ] = theta_l2

        # Save candidate L2 acquisition phase.
        phase_out[
            b0:b1,
            :
        ] = theta_full.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Acquisition wrapped parity against canonical
        # PointPhaseStack.
        #
        # This is independent of any spatial reference point.
        # ----------------------------------------------------

        original_pl = np.asarray(
            phase_pl[
                ids,
                :
            ],
            dtype=np.float64,
        )

        parity = wrap(
            theta_full
            -
            original_pl
        )

        abs_parity = np.abs(
            parity
        )

        global_wrap_parity_max = max(
            global_wrap_parity_max,
            float(
                abs_parity.max()
            ),
        )

        global_wrap_parity_ss += float(
            np.sum(
                parity * parity
            )
        )

        global_wrap_parity_n += int(
            parity.size
        )

        if (
            b0 == 0
            or
            b1 == nstrict
            or
            (
                b1
                //
                args.batch_size
            )
            %
            10
            ==
            0
        ):

            print(
                f"  {b1:,}/"
                f"{nstrict:,}"
            )

    # Flush memmaps.
    phase_out.flush()
    max_tree_l2_by_point.flush()
    l2_residual_rms.flush()
    l2_residual_max.flush()

    # ========================================================
    # Final statistics
    # ========================================================

    tree_l2_rms = np.sqrt(
        global_tree_l2_ss
        /
        global_tree_l2_n
    )

    l2_residual_global_rms = np.sqrt(
        global_l2_residual_ss
        /
        global_l2_residual_n
    )

    wrap_parity_rms = np.sqrt(
        global_wrap_parity_ss
        /
        global_wrap_parity_n
    )

    # Distribution of point-level QA.
    point_diff_arr = np.asarray(
        max_tree_l2_by_point
    )

    point_rms_arr = np.asarray(
        l2_residual_rms
    )

    point_rmax_arr = np.asarray(
        l2_residual_max
    )

    print()
    print("=" * 108)
    print(
        "Tree vs full-network L2 parity"
    )
    print("=" * 108)

    print(
        f"RMS phase difference       : "
        f"{tree_l2_rms:.3e} rad"
    )

    print(
        f"maximum phase difference   : "
        f"{global_tree_l2_max:.3e} rad"
    )

    print(
        "point max-difference p50/p90/p95/p99/max:"
    )

    q = np.percentile(
        point_diff_arr,
        [
            50,
            90,
            95,
            99,
            100,
        ],
    )

    print(
        "  "
        +
        " / ".join(
            f"{x:.3e}"
            for x in q
        )
        +
        " rad"
    )

    print()
    print("=" * 108)
    print(
        "Full-network L2 residual"
    )
    print("=" * 108)

    print(
        f"global RMS residual        : "
        f"{l2_residual_global_rms:.3e} rad"
    )

    print(
        f"global max residual        : "
        f"{global_l2_residual_max:.3e} rad"
    )

    q = np.percentile(
        point_rms_arr,
        [
            50,
            90,
            95,
            99,
            100,
        ],
    )

    print(
        "point RMS residual p50/p90/p95/p99/max:"
    )

    print(
        "  "
        +
        " / ".join(
            f"{x:.3e}"
            for x in q
        )
        +
        " rad"
    )

    print(
        f"tree all-edge max residual : "
        f"{global_tree_residual_max:.3e} rad"
    )

    print()
    print("=" * 108)
    print(
        "Acquisition wrapped-phase parity"
    )
    print("=" * 108)

    print(
        f"RMS wrap(theta_L2 - PL)    : "
        f"{wrap_parity_rms:.3e} rad"
    )

    print(
        f"max wrap(theta_L2 - PL)    : "
        f"{global_wrap_parity_max:.3e} rad"
    )

    # ========================================================
    # Conservative numerical status
    # ========================================================

    tolerance = 1.0e-4

    passed = (
        global_tree_l2_max
        <
        tolerance
        and
        global_l2_residual_max
        <
        tolerance
        and
        global_wrap_parity_max
        <
        tolerance
    )

    status = (
        "PASS"
        if passed
        else
        "REVIEW"
    )

    # ========================================================
    # Save metadata
    # ========================================================

    np.save(
        outdir
        / "strict_point_ids.npy",
        strict_ids,
    )

    np.save(
        outdir
        / "temporal_tree_edge_ids.npy",
        tree_ids,
    )

    np.save(
        outdir
        / "design_matrix.npy",
        A.astype(
            np.float32
        ),
    )

    dates_path = (
        outdir
        / "dates.txt"
    )

    dates_path.write_text(
        "\n".join(
            str(x)
            for x in stack.dates
        )
        +
        "\n"
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-network-inversion-parity-processing",

        "status":
            status,

        "reference_acquisition_index":
            int(
                reference_idx
            ),

        "reference_date":
            str(
                stack.dates[
                    reference_idx
                ]
            ),

        "points":
            int(
                npoint
            ),

        "strict_points":
            int(
                nstrict
            ),

        "acquisitions":
            int(
                ndate
            ),

        "ifgs":
            int(
                nifg
            ),

        "rank_A":
            int(
                rank_A
            ),

        "condition_number_A":
            float(
                cond_A
            ),

        "condition_number_tree":
            float(
                cond_tree
            ),

        "tree_l2_parity": {
            "rms_rad":
                float(
                    tree_l2_rms
                ),

            "max_rad":
                float(
                    global_tree_l2_max
                ),
        },

        "l2_network_residual": {
            "rms_rad":
                float(
                    l2_residual_global_rms
                ),

            "max_rad":
                float(
                    global_l2_residual_max
                ),
        },

        "acquisition_wrap_parity": {
            "rms_rad":
                float(
                    wrap_parity_rms
                ),

            "max_rad":
                float(
                    global_wrap_parity_max
                ),
        },

        "numerical_tolerance_rad":
            tolerance,

        "spatial_reference_applied":
            False,

        "note":
            (
                "Temporal reference acquisition only. "
                "No deformation spatial reference point "
                "or reference region has been applied."
            ),
    }

    manifest_path = (
        outdir
        / "network_inversion_parity_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n"
    )

    print()
    print(
        f"L2 phase candidate         : "
        f"{phase_out_path}"
    )

    print(
        f"strict point IDs           : "
        f"{outdir / 'strict_point_ids.npy'}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 09a STATUS: {status} / "
        "TREE-vs-L2 NETWORK INVERSION QUALITY"
    )

    print(
        "No spatial deformation reference "
        "has been applied."
    )


if __name__ == "__main__":
    main()
