#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


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


def read_qa(path: Path):
    out = {}

    if not path.exists():
        return out

    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                pid = int(r["pair_id"])
            except Exception:
                continue

            out[pid] = r

    return out


def as_int(row, key, default=-1):
    if row is None:
        return default

    try:
        return int(row[key])
    except Exception:
        return default


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
        / "v09"
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

    qa_path = (
        root
        / "batch_unwrap_validation"
        / "all_ifg_unwrap_qa.csv"
    )

    outdir = (
        root
        / "ifg_visual_qa_v2"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase = np.load(
        pps_dir
        / "phase_rad.npy",
        mmap_mode="r",
    )

    u = np.asarray(
        np.load(
            graph_dir
            / "local_u.npy",
            mmap_mode="r",
        ),
        dtype=np.int32,
    )

    v = np.asarray(
        np.load(
            graph_dir
            / "local_v.npy",
            mmap_mode="r",
        ),
        dtype=np.int32,
    )

    npoint, ndate = phase.shape

    temporal_edges = load_itab(
        network_dir
        / "network.itab",
        ndate,
    )

    nifg = len(
        temporal_edges
    )

    qa = read_qa(
        qa_path
    )

    print("=" * 112)
    print(
        "Step 08y4 - Exact SAFE / UNSAFE spatial integer-edge audit"
    )
    print("=" * 112)

    print(
        f"config                     : {config_path}"
    )

    print(
        f"points                     : {npoint:,}"
    )

    print(
        f"local graph edges          : {u.size:,}"
    )

    print(
        f"IFGs                       : {nifg}"
    )

    rows_out = []

    # Global totals.
    total_safe = 0
    total_unsafe = 0

    total_safe_bad = 0
    total_unsafe_k_nonzero = 0

    total_old_branch = 0
    total_old_mismatch = 0
    total_old_mismatch_safe = 0
    total_old_mismatch_unsafe = 0

    total_identity_bad = 0

    total_safe_relation_bad = 0

    max_integer_identity_residual = 0
    max_float_edge_residual = 0.0

    all_abs_dN_ge2 = 0

    # --------------------------------------------------------
    # Full 108 IFG audit
    # --------------------------------------------------------

    for pair_id, (ti, tj) in enumerate(
        temporal_edges,
        start=1,
    ):

        d1 = str(stack.dates[ti])
        d2 = str(stack.dates[tj])

        tag = (
            f"pair{pair_id:03d}_"
            f"{d1}_{d2}"
        )

        path = (
            unwrap_dir
            / f"{tag}_unwrapped_phase_rad.npy"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        U = np.asarray(
            np.load(
                path,
                mmap_mode="r",
            ),
            dtype=np.float64,
        )

        wrapped = wrap(
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

        # ====================================================
        # Endpoint wrapped difference
        # ====================================================

        raw_diff = (
            wrapped[v]
            -
            wrapped[u]
        )

        # The actual graph observation.
        g = wrap(
            raw_diff
        )

        # Integer removed by spatial wrapping:
        #
        # raw_diff = g + 2*pi*w
        w = np.rint(
            (
                raw_diff
                -
                g
            )
            /
            TWOPI
        ).astype(
            np.int16
        )

        raw_float_residual = (
            raw_diff
            -
            g
            -
            TWOPI
            *
            w.astype(
                np.float64
            )
        )

        # ====================================================
        # Spatial ambiguity at nodes
        # ====================================================

        N = np.rint(
            (
                U
                -
                wrapped
            )
            /
            TWOPI
        ).astype(
            np.int16
        )

        dN = (
            N[v]
            -
            N[u]
        ).astype(
            np.int16
        )

        # ====================================================
        # Actual integer carried by each graph edge:
        #
        # Uv - Uu = g + 2*pi*k
        # ====================================================

        edge_delta = (
            U[v]
            -
            U[u]
            -
            g
        )

        k = np.rint(
            edge_delta
            /
            TWOPI
        ).astype(
            np.int16
        )

        edge_float_residual = (
            edge_delta
            -
            TWOPI
            *
            k.astype(
                np.float64
            )
        )

        # ====================================================
        # Exact algebra:
        #
        # dN = k - w
        #
        # therefore:
        #
        # dN + w - k == 0
        # ====================================================

        identity = (
            dN
            +
            w
            -
            k
        ).astype(
            np.int16
        )

        identity_bad = (
            identity
            !=
            0
        )

        n_identity_bad = int(
            np.count_nonzero(
                identity_bad
            )
        )

        max_identity = int(
            np.max(
                np.abs(
                    identity
                )
            )
        )

        # ====================================================
        # SAFE / UNSAFE exactly as Step08
        # ====================================================

        safe = (
            np.abs(g)
            <=
            np.pi / 2
        )

        unsafe = ~safe

        nsafe = int(
            np.count_nonzero(
                safe
            )
        )

        nunsafe = int(
            np.count_nonzero(
                unsafe
            )
        )

        # The true SAFE integer violation.
        safe_bad = (
            safe
            &
            (k != 0)
        )

        nsafe_bad = int(
            np.count_nonzero(
                safe_bad
            )
        )

        # Equivalent SAFE relation:
        #
        # if k=0 -> dN = -w
        safe_relation_bad = (
            safe
            &
            (
                dN
                !=
                -w
            )
        )

        nsafe_relation_bad = int(
            np.count_nonzero(
                safe_relation_bad
            )
        )

        # UNSAFE k != 0 is diagnostic,
        # NOT automatically an error.
        unsafe_knz = (
            unsafe
            &
            (k != 0)
        )

        nunsafe_knz = int(
            np.count_nonzero(
                unsafe_knz
            )
        )

        # ====================================================
        # Reproduce old 08y3 criterion
        # ====================================================

        old_branch = (
            dN != 0
        )

        raw_wrap_jump = (
            w != 0
        )

        old_match = (
            old_branch
            &
            raw_wrap_jump
        )

        old_mismatch = (
            old_branch
            &
            (~raw_wrap_jump)
        )

        old_mismatch_safe = (
            old_mismatch
            &
            safe
        )

        old_mismatch_unsafe = (
            old_mismatch
            &
            unsafe
        )

        n_old_branch = int(
            np.count_nonzero(
                old_branch
            )
        )

        n_old_mismatch = int(
            np.count_nonzero(
                old_mismatch
            )
        )

        n_old_mismatch_safe = int(
            np.count_nonzero(
                old_mismatch_safe
            )
        )

        n_old_mismatch_unsafe = int(
            np.count_nonzero(
                old_mismatch_unsafe
            )
        )

        # ====================================================
        # Integer distributions
        # ====================================================

        abs_dN_ge2 = int(
            np.count_nonzero(
                np.abs(dN)
                >=
                2
            )
        )

        safe_k_abs2 = int(
            np.count_nonzero(
                safe
                &
                (
                    np.abs(k)
                    >=
                    2
                )
            )
        )

        unsafe_k_m2 = int(
            np.count_nonzero(
                unsafe
                &
                (k <= -2)
            )
        )

        unsafe_k_m1 = int(
            np.count_nonzero(
                unsafe
                &
                (k == -1)
            )
        )

        unsafe_k_0 = int(
            np.count_nonzero(
                unsafe
                &
                (k == 0)
            )
        )

        unsafe_k_p1 = int(
            np.count_nonzero(
                unsafe
                &
                (k == 1)
            )
        )

        unsafe_k_p2 = int(
            np.count_nonzero(
                unsafe
                &
                (k >= 2)
            )
        )

        q = qa.get(
            pair_id
        )

        qa_safe_bad = as_int(
            q,
            "safe_internal_bad",
            -1,
        )

        qa_final_safe_bad = as_int(
            q,
            "final_safe_bad",
            -1,
        )

        safe_bad_matches_qa = (
            qa_safe_bad < 0
            or
            nsafe_bad
            ==
            qa_safe_bad
        )

        row = {
            "pair_id":
                pair_id,

            "date1":
                d1,

            "date2":
                d2,

            "safe_edges":
                nsafe,

            "unsafe_edges":
                nunsafe,

            "safe_k_nonzero":
                nsafe_bad,

            "qa_safe_internal_bad":
                qa_safe_bad,

            "qa_final_safe_bad":
                qa_final_safe_bad,

            "safe_bad_matches_qa":
                int(
                    safe_bad_matches_qa
                ),

            "safe_relation_bad":
                nsafe_relation_bad,

            "unsafe_k_nonzero":
                nunsafe_knz,

            "old_branch_edges":
                n_old_branch,

            "old_branch_mismatch":
                n_old_mismatch,

            "old_mismatch_SAFE":
                n_old_mismatch_safe,

            "old_mismatch_UNSAFE":
                n_old_mismatch_unsafe,

            "old_mismatch_unsafe_fraction":
                (
                    float(
                        n_old_mismatch_unsafe
                        /
                        n_old_mismatch
                    )
                    if n_old_mismatch
                    else 0.0
                ),

            "abs_dN_ge2":
                abs_dN_ge2,

            "safe_k_abs2plus":
                safe_k_abs2,

            "unsafe_k_le_m2":
                unsafe_k_m2,

            "unsafe_k_m1":
                unsafe_k_m1,

            "unsafe_k_0":
                unsafe_k_0,

            "unsafe_k_p1":
                unsafe_k_p1,

            "unsafe_k_ge_p2":
                unsafe_k_p2,

            "integer_identity_bad":
                n_identity_bad,

            "integer_identity_max_abs":
                max_identity,

            "raw_wrap_float_residual_max":
                float(
                    np.max(
                        np.abs(
                            raw_float_residual
                        )
                    )
                ),

            "edge_integer_float_residual_max":
                float(
                    np.max(
                        np.abs(
                            edge_float_residual
                        )
                    )
                ),
        }

        rows_out.append(row)

        total_safe += nsafe
        total_unsafe += nunsafe

        total_safe_bad += nsafe_bad
        total_unsafe_k_nonzero += nunsafe_knz

        total_old_branch += n_old_branch
        total_old_mismatch += n_old_mismatch

        total_old_mismatch_safe += (
            n_old_mismatch_safe
        )

        total_old_mismatch_unsafe += (
            n_old_mismatch_unsafe
        )

        total_identity_bad += (
            n_identity_bad
        )

        total_safe_relation_bad += (
            nsafe_relation_bad
        )

        all_abs_dN_ge2 += (
            abs_dN_ge2
        )

        max_integer_identity_residual = max(
            max_integer_identity_residual,
            max_identity,
        )

        max_float_edge_residual = max(
            max_float_edge_residual,
            float(
                np.max(
                    np.abs(
                        edge_float_residual
                    )
                )
            ),
        )

        print(
            f"{pair_id:3d}: "
            f"{d1}->{d2} "
            f"SAFEbad={nsafe_bad:3d} "
            f"oldMismatch="
            f"{n_old_mismatch:4d} "
            f"[S={n_old_mismatch_safe:3d}, "
            f"U={n_old_mismatch_unsafe:4d}] "
            f"identityBad="
            f"{n_identity_bad}"
        )

    # ========================================================
    # Global summary
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Exact integer-edge consistency"
    )
    print("=" * 112)

    print(
        f"integer identity bad       : "
        f"{total_identity_bad:,}"
    )

    print(
        f"max |dN + w - k|          : "
        f"{max_integer_identity_residual}"
    )

    print(
        f"edge float residual max    : "
        f"{max_float_edge_residual:.3e} rad"
    )

    print()
    print("=" * 112)
    print(
        "SAFE / UNSAFE decomposition"
    )
    print("=" * 112)

    print(
        f"SAFE edge observations     : "
        f"{total_safe:,}"
    )

    print(
        f"UNSAFE edge observations   : "
        f"{total_unsafe:,}"
    )

    print(
        f"SAFE k != 0                : "
        f"{total_safe_bad:,}"
    )

    print(
        f"SAFE relation violations   : "
        f"{total_safe_relation_bad:,}"
    )

    print(
        f"UNSAFE k != 0              : "
        f"{total_unsafe_k_nonzero:,}"
    )

    print()
    print("=" * 112)
    print(
        "Reinterpretation of Step08y3 branch mismatch"
    )
    print("=" * 112)

    print(
        f"old branch edges           : "
        f"{total_old_branch:,}"
    )

    print(
        f"old mismatch edges         : "
        f"{total_old_mismatch:,}"
    )

    print(
        f"  mismatch on SAFE         : "
        f"{total_old_mismatch_safe:,}"
    )

    print(
        f"  mismatch on UNSAFE       : "
        f"{total_old_mismatch_unsafe:,}"
    )

    if total_old_mismatch:

        print(
            f"  UNSAFE share             : "
            f"{100*total_old_mismatch_unsafe/total_old_mismatch:.6f}%"
        )

    print(
        f"|dN| >= 2 edges           : "
        f"{all_abs_dN_ge2:,}"
    )

    # ========================================================
    # Worst old mismatch IFGs, with SAFE / UNSAFE split
    # ========================================================

    weakest = sorted(
        rows_out,
        key=lambda r: (
            r[
                "old_mismatch_SAFE"
            ],
            r[
                "old_branch_mismatch"
            ],
        ),
        reverse=True,
    )

    print()
    print("=" * 112)
    print(
        "Largest old branch mismatches after SAFE/UNSAFE split"
    )
    print("=" * 112)

    print(
        " pair  dates                  "
        "oldMis SAFEmis UNSAFEmis "
        "SAFEk!=0 qaSAFE"
    )

    for r in weakest[:20]:

        print(
            f" {r['pair_id']:4d}  "
            f"{r['date1']}->"
            f"{r['date2']} "
            f"{r['old_branch_mismatch']:6d} "
            f"{r['old_mismatch_SAFE']:7d} "
            f"{r['old_mismatch_UNSAFE']:9d} "
            f"{r['safe_k_nonzero']:8d} "
            f"{r['qa_safe_internal_bad']:6d}"
        )

    # ========================================================
    # Status
    # ========================================================

    qa_disagree = [
        r
        for r in rows_out
        if (
            r[
                "qa_safe_internal_bad"
            ]
            >=
            0
            and
            r[
                "safe_k_nonzero"
            ]
            !=
            r[
                "qa_safe_internal_bad"
            ]
        )
    ]

    if total_identity_bad != 0:

        status = (
            "FAIL_INTEGER_IDENTITY"
        )

    elif qa_disagree:

        status = (
            "REVIEW_SAFE_QA_DISAGREEMENT"
        )

    elif total_safe_relation_bad != total_safe_bad:

        # Mathematically these should represent the same
        # nonzero-k set on SAFE edges.
        status = (
            "FAIL_SAFE_RELATION"
        )

    else:

        status = (
            "PASS"
        )

    # ========================================================
    # Save
    # ========================================================

    csv_path = (
        outdir
        / "visual_qa_v4_safe_unsafe_edge_audit.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        wout = csv.DictWriter(
            f,
            fieldnames=list(
                rows_out[0].keys()
            ),
        )

        wout.writeheader()
        wout.writerows(
            rows_out
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-safe-unsafe-integer-edge-audit-v09",

        "status":
            status,

        "ifgs":
            int(nifg),

        "integer_identity": {
            "bad_edges":
                int(
                    total_identity_bad
                ),

            "max_abs_residual_integer":
                int(
                    max_integer_identity_residual
                ),
        },

        "safe": {
            "observations":
                int(
                    total_safe
                ),

            "k_nonzero":
                int(
                    total_safe_bad
                ),

            "relation_violations":
                int(
                    total_safe_relation_bad
                ),
        },

        "unsafe": {
            "observations":
                int(
                    total_unsafe
                ),

            "k_nonzero":
                int(
                    total_unsafe_k_nonzero
                ),
        },

        "old_y3_mismatch": {
            "total":
                int(
                    total_old_mismatch
                ),

            "safe":
                int(
                    total_old_mismatch_safe
                ),

            "unsafe":
                int(
                    total_old_mismatch_unsafe
                ),

            "unsafe_fraction":
                (
                    float(
                        total_old_mismatch_unsafe
                        /
                        total_old_mismatch
                    )
                    if total_old_mismatch
                    else 0.0
                ),
        },

        "abs_dN_ge2":
            int(
                all_abs_dN_ge2
            ),

        "qa_safe_count_disagreement_ifgs":
            [
                int(
                    r["pair_id"]
                )
                for r in qa_disagree
            ],

        "interpretation":
            (
                "Only SAFE k!=0 is a strict local integer "
                "consistency violation. Nonzero k on UNSAFE "
                "edges is diagnostic and is not automatically "
                "an unwrap failure."
            ),
    }

    manifest_path = (
        outdir
        / "visual_qa_v4_safe_unsafe_edge_audit.json"
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
        f"audit table                : "
        f"{csv_path}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 08y4 STATUS: {status}"
    )


if __name__ == "__main__":
    main()
