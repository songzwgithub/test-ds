#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def wrap_phase(x):
    """
    Production wrapped phase in [-pi, pi].
    Preserve float32 behavior where possible.
    """
    return np.arctan2(
        np.sin(x),
        np.cos(x),
    ).astype(np.float32, copy=False)


def read_dates(path: Path):
    dates = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if len(dates) != len(set(dates)):
        raise RuntimeError(
            "Duplicate acquisition dates in dates.txt"
        )

    return dates


def load_itab(path: Path, ndate: int):
    edges = []

    for line_no, raw in enumerate(
        path.read_text().splitlines(),
        start=1,
    ):
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        f = line.split()

        if len(f) < 2:
            raise RuntimeError(
                f"Invalid ITAB line {line_no}: {raw}"
            )

        i = int(f[0]) - 1
        j = int(f[1]) - 1

        if not (
            0 <= i < ndate
            and
            0 <= j < ndate
        ):
            raise RuntimeError(
                f"ITAB index outside [1,{ndate}] "
                f"at line {line_no}: {raw}"
            )

        if i == j:
            raise RuntimeError(
                f"Self pair at ITAB line {line_no}"
            )

        # Production network is chronological.
        if i > j:
            raise RuntimeError(
                f"Reversed pair at ITAB line {line_no}: "
                f"{i+1} -> {j+1}"
            )

        edges.append((i, j))

    if len(edges) != len(set(edges)):
        raise RuntimeError(
            "Duplicate interferometric pairs in network.itab"
        )

    return edges


def enumerate_triangles(ndate, edges):
    adj = [
        set()
        for _ in range(ndate)
    ]

    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    triangles = []

    for a in range(ndate):
        for b in sorted(adj[a]):

            if b <= a:
                continue

            common = (
                adj[a]
                &
                adj[b]
            )

            for c in sorted(common):

                if c <= b:
                    continue

                triangles.append(
                    (a, b, c)
                )

    return triangles


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Quality pyPSDS virtual interferograms "
            "and wrapped triangle closure identities."
        )
    )

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=16000,
    )

    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0"
        )

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

    netdir = (
        outroot
        / "network"
    )

    qa_dir = (
        outroot
        / "virtual_ifg_quality"
    )

    qa_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase_path = (
        pps_dir
        / "phase_rad.npy"
    )

    dates_path = (
        pps_dir
        / "dates.txt"
    )

    itab_path = (
        netdir
        / "network.itab"
    )

    manifest_path = (
        netdir
        / "network_manifest.json"
    )

    for p in (
        phase_path,
        dates_path,
        itab_path,
        manifest_path,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    # ========================================================
    # Load metadata
    # ========================================================

    dates = read_dates(
        dates_path
    )

    stack_dates = list(
        stack.dates
    )

    if dates != stack_dates:
        raise RuntimeError(
            "PointPhaseStack dates.txt does not "
            "exactly match GAMMA stack dates."
        )

    ndate = len(dates)

    edges = load_itab(
        itab_path,
        ndate,
    )

    npair = len(edges)

    manifest = json.loads(
        manifest_path.read_text()
    )

    manifest_dates = manifest.get(
        "dates",
        []
    )

    if manifest_dates != dates:
        raise RuntimeError(
            "network_manifest dates do not "
            "match PointPhaseStack dates."
        )

    expected_pairs = (
        manifest
        .get("network", {})
        .get("selected_pairs")
    )

    if (
        expected_pairs is not None
        and
        int(expected_pairs) != npair
    ):
        raise RuntimeError(
            f"network.itab has {npair} pairs, "
            f"manifest says {expected_pairs}."
        )

    # ========================================================
    # PointPhaseStack
    # ========================================================

    phase = np.load(
        phase_path,
        mmap_mode="r",
    )

    if phase.ndim != 2:
        raise RuntimeError(
            f"phase_rad must be 2-D, got {phase.shape}"
        )

    npoint, phase_ndate = (
        phase.shape
    )

    if phase_ndate != ndate:
        raise RuntimeError(
            f"phase_rad dates={phase_ndate}, "
            f"dates.txt={ndate}"
        )

    if phase.dtype != np.float32:
        raise RuntimeError(
            f"Expected phase_rad float32, "
            f"got {phase.dtype}"
        )

    # ========================================================
    # Network / triangle structure
    # ========================================================

    triangles = enumerate_triangles(
        ndate,
        edges,
    )

    ntri = len(triangles)

    edge_index = {
        e: k
        for k, e in enumerate(edges)
    }

    tri_edge_idx = np.empty(
        (
            ntri,
            3,
        ),
        dtype=np.int32,
    )

    for q, (
        a,
        b,
        c,
    ) in enumerate(triangles):

        try:
            tri_edge_idx[q, 0] = (
                edge_index[(a, b)]
            )

            tri_edge_idx[q, 1] = (
                edge_index[(b, c)]
            )

            tri_edge_idx[q, 2] = (
                edge_index[(a, c)]
            )

        except KeyError as exc:
            raise RuntimeError(
                "Triangle enumeration produced "
                "an edge absent from network."
            ) from exc

    manifest_triangles = (
        manifest
        .get("cycle_quality", {})
        .get("triangles")
    )

    if (
        manifest_triangles is not None
        and
        int(manifest_triangles) != ntri
    ):
        raise RuntimeError(
            f"Enumerated {ntri} triangles, "
            f"frozen manifest says "
            f"{manifest_triangles}."
        )

    print("=" * 80)
    print(
        "Virtual IFG / closure quality"
    )
    print("=" * 80)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"PointPhaseStack            : "
        f"{phase_path}"
    )

    print(
        f"points                     : "
        f"{npoint:,}"
    )

    print(
        f"acquisitions               : "
        f"{ndate}"
    )

    print(
        f"production IFG pairs       : "
        f"{npair}"
    )

    print(
        f"production triangles       : "
        f"{ntri}"
    )

    print(
        f"phase dtype                : "
        f"{phase.dtype}"
    )

    print(
        f"batch size                 : "
        f"{args.batch_size:,}"
    )

    print()
    print(
        "Virtual IFG convention:"
    )
    print(
        "  phi_ij = wrap(phi_j - phi_i)"
    )
    print(
        "  triangle closure = "
        "wrap(phi_ab + phi_bc - phi_ac)"
    )

    # ========================================================
    # Quality accumulators
    # ========================================================

    input_nonfinite = 0
    ifg_nonfinite = 0

    ifg_abs_max = 0.0

    closure_abs_max = 0.0
    closure_abs_sum = 0.0
    closure_count = 0

    closure_gt_1e7 = 0
    closure_gt_1e6 = 0
    closure_gt_1e5 = 0
    closure_gt_1e4 = 0

    per_triangle_max = np.zeros(
        ntri,
        dtype=np.float64,
    )

    per_triangle_sum = np.zeros(
        ntri,
        dtype=np.float64,
    )

    per_triangle_count = np.zeros(
        ntri,
        dtype=np.int64,
    )

    # ========================================================
    # Full PointPhaseStack quality
    # ========================================================

    print()
    print(
        "Running full-scene virtual IFG quality..."
    )

    total_batches = math.ceil(
        npoint
        /
        args.batch_size
    )

    for batch_no, p0 in enumerate(
        range(
            0,
            npoint,
            args.batch_size,
        ),
        start=1,
    ):

        p1 = min(
            p0 + args.batch_size,
            npoint,
        )

        # Copy current block into RAM.
        ph = np.asarray(
            phase[p0:p1, :],
            dtype=np.float32,
        )

        bad_input = int(
            np.count_nonzero(
                ~np.isfinite(ph)
            )
        )

        input_nonfinite += bad_input

        if bad_input:
            raise RuntimeError(
                f"Non-finite acquisition phase "
                f"in points [{p0}:{p1})."
            )

        B = p1 - p0

        # Temporary only:
        # B x 108, not persisted.
        vifg = np.empty(
            (
                B,
                npair,
            ),
            dtype=np.float32,
        )

        for eidx, (
            i,
            j,
        ) in enumerate(edges):

            vifg[:, eidx] = wrap_phase(
                ph[:, j]
                -
                ph[:, i]
            )

        bad_ifg = int(
            np.count_nonzero(
                ~np.isfinite(vifg)
            )
        )

        ifg_nonfinite += bad_ifg

        if bad_ifg:
            raise RuntimeError(
                f"Non-finite virtual IFG phase "
                f"in points [{p0}:{p1})."
            )

        local_abs_max = float(
            np.max(
                np.abs(vifg)
            )
        )

        ifg_abs_max = max(
            ifg_abs_max,
            local_abs_max,
        )

        if (
            local_abs_max
            >
            np.pi + 2e-6
        ):
            raise RuntimeError(
                "Virtual IFG outside [-pi,pi]: "
                f"max |phase|={local_abs_max}"
            )

        # ----------------------------------------------------
        # All 102 triangle closures
        # ----------------------------------------------------

        for q in range(ntri):

            eab = tri_edge_idx[q, 0]
            ebc = tri_edge_idx[q, 1]
            eac = tri_edge_idx[q, 2]

            residual = wrap_phase(
                vifg[:, eab]
                +
                vifg[:, ebc]
                -
                vifg[:, eac]
            )

            ar = np.abs(
                residual
            )

            qmax = float(
                np.max(ar)
            )

            per_triangle_max[q] = max(
                per_triangle_max[q],
                qmax,
            )

            s = float(
                np.sum(
                    ar,
                    dtype=np.float64,
                )
            )

            per_triangle_sum[q] += s
            per_triangle_count[q] += B

            closure_abs_max = max(
                closure_abs_max,
                qmax,
            )

            closure_abs_sum += s
            closure_count += B

            closure_gt_1e7 += int(
                np.count_nonzero(
                    ar > 1e-7
                )
            )

            closure_gt_1e6 += int(
                np.count_nonzero(
                    ar > 1e-6
                )
            )

            closure_gt_1e5 += int(
                np.count_nonzero(
                    ar > 1e-5
                )
            )

            closure_gt_1e4 += int(
                np.count_nonzero(
                    ar > 1e-4
                )
            )

        if (
            batch_no == 1
            or
            batch_no % 5 == 0
            or
            batch_no == total_batches
        ):

            print(
                f"  batch "
                f"{batch_no:3d}/"
                f"{total_batches:3d}: "
                f"points {p0:,}:{p1:,}, "
                f"IFG max={local_abs_max:.8f}, "
                f"closure max="
                f"{closure_abs_max:.3e}"
            )

    # ========================================================
    # Reversal identity quality
    # ========================================================

    print()
    print(
        "Running deterministic pair-reversal quality..."
    )

    rng = np.random.default_rng(
        20260816
    )

    nsample_point = min(
        10000,
        npoint,
    )

    nsample_pair = min(
        24,
        npair,
    )

    point_ids = np.sort(
        rng.choice(
            npoint,
            size=nsample_point,
            replace=False,
        )
    )

    pair_ids = np.sort(
        rng.choice(
            npair,
            size=nsample_pair,
            replace=False,
        )
    )

    sample_phase = np.asarray(
        phase[point_ids, :],
        dtype=np.float32,
    )

    reversal_abs_max = 0.0

    for eidx in pair_ids:

        i, j = edges[eidx]

        forward = wrap_phase(
            sample_phase[:, j]
            -
            sample_phase[:, i]
        )

        reverse = wrap_phase(
            sample_phase[:, i]
            -
            sample_phase[:, j]
        )

        residual = wrap_phase(
            forward
            +
            reverse
        )

        reversal_abs_max = max(
            reversal_abs_max,
            float(
                np.max(
                    np.abs(residual)
                )
            ),
        )

    # ========================================================
    # Per-triangle output
    # ========================================================

    tri_csv = (
        qa_dir
        / "triangle_closure_qa.csv"
    )

    with tri_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "triangle_id",
            "a1",
            "b1",
            "c1",
            "date_a",
            "date_b",
            "date_c",
            "edge_ab_1",
            "edge_bc_1",
            "edge_ac_1",
            "closure_abs_mean_rad",
            "closure_abs_max_rad",
        ])

        for q, (
            a,
            b,
            c,
        ) in enumerate(
            triangles,
            start=1,
        ):

            idx = q - 1

            mean_abs = (
                per_triangle_sum[idx]
                /
                per_triangle_count[idx]
            )

            w.writerow([
                q,
                a + 1,
                b + 1,
                c + 1,
                dates[a],
                dates[b],
                dates[c],
                int(
                    tri_edge_idx[idx, 0]
                ) + 1,
                int(
                    tri_edge_idx[idx, 1]
                ) + 1,
                int(
                    tri_edge_idx[idx, 2]
                ) + 1,
                f"{mean_abs:.12e}",
                f"{per_triangle_max[idx]:.12e}",
            ])

    # ========================================================
    # Pair mapping
    # ========================================================

    pair_csv = (
        qa_dir
        / "virtual_ifg_pairs.csv"
    )

    with pair_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "pair_id",
            "i0",
            "j0",
            "i1",
            "j1",
            "date1",
            "date2",
        ])

        for q, (
            i,
            j,
        ) in enumerate(
            edges,
            start=1,
        ):

            w.writerow([
                q,
                i,
                j,
                i + 1,
                j + 1,
                dates[i],
                dates[j],
            ])

    closure_abs_mean = (
        closure_abs_sum
        /
        closure_count
    )

    # ========================================================
    # Final QA decision
    # ========================================================

    # Float32 wrapped arithmetic should be very close to zero.
    # 1e-5 rad is already extremely conservative here.
    closure_pass = (
        closure_abs_max
        <= 1e-5
    )

    reversal_pass = (
        reversal_abs_max
        <= 1e-6
    )

    overall_pass = (
        input_nonfinite == 0
        and
        ifg_nonfinite == 0
        and
        ifg_abs_max <= np.pi + 2e-6
        and
        closure_pass
        and
        reversal_pass
    )

    summary = {
        "format": "pyPSDS-GAMMA-virtual-ifg-quality-v1.0",

        "point_phase_stack": {
            "points": int(npoint),
            "acquisitions": int(ndate),
            "dtype": str(phase.dtype),
            "input_nonfinite": int(
                input_nonfinite
            ),
        },

        "network": {
            "pairs": int(npair),
            "triangles": int(ntri),
        },

        "virtual_ifg": {
            "definition": (
                "wrap(phi_j - phi_i)"
            ),
            "persisted_ifg_cube": False,
            "nonfinite": int(
                ifg_nonfinite
            ),
            "max_abs_phase_rad": float(
                ifg_abs_max
            ),
        },

        "triangle_closure": {
            "definition": (
                "wrap(phi_ab + phi_bc - phi_ac)"
            ),
            "evaluations": int(
                closure_count
            ),
            "mean_abs_rad": float(
                closure_abs_mean
            ),
            "max_abs_rad": float(
                closure_abs_max
            ),
            "count_gt_1e-7": int(
                closure_gt_1e7
            ),
            "count_gt_1e-6": int(
                closure_gt_1e6
            ),
            "count_gt_1e-5": int(
                closure_gt_1e5
            ),
            "count_gt_1e-4": int(
                closure_gt_1e4
            ),
        },

        "pair_reversal": {
            "sample_points": int(
                nsample_point
            ),
            "sample_pairs": int(
                nsample_pair
            ),
            "max_abs_residual_rad": float(
                reversal_abs_max
            ),
        },

        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    }

    manifest_out = (
        qa_dir
        / "virtual_ifg_quality_manifest.json"
    )

    manifest_out.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Virtual IFG quality summary"
    )
    print("=" * 80)

    print(
        f"points                     : "
        f"{npoint:,}"
    )

    print(
        f"acquisitions               : "
        f"{ndate}"
    )

    print(
        f"production pairs           : "
        f"{npair}"
    )

    print(
        f"triangles                  : "
        f"{ntri}"
    )

    print(
        f"IFG non-finite             : "
        f"{ifg_nonfinite}"
    )

    print(
        f"IFG max |phase|            : "
        f"{ifg_abs_max:.9f} rad"
    )

    print()
    print(
        f"closure evaluations        : "
        f"{closure_count:,}"
    )

    print(
        f"closure mean |residual|    : "
        f"{closure_abs_mean:.3e} rad"
    )

    print(
        f"closure max |residual|     : "
        f"{closure_abs_max:.3e} rad"
    )

    print(
        f"closure > 1e-7             : "
        f"{closure_gt_1e7:,}"
    )

    print(
        f"closure > 1e-6             : "
        f"{closure_gt_1e6:,}"
    )

    print(
        f"closure > 1e-5             : "
        f"{closure_gt_1e5:,}"
    )

    print(
        f"closure > 1e-4             : "
        f"{closure_gt_1e4:,}"
    )

    print()
    print(
        f"reversal max residual      : "
        f"{reversal_abs_max:.3e} rad"
    )

    print()
    print(
        f"pair mapping               : "
        f"{pair_csv}"
    )

    print(
        f"triangle QA                : "
        f"{tri_csv}"
    )

    print(
        f"quality manifest             : "
        f"{manifest_out}"
    )

    print()

    if overall_pass:

        print(
            "STEP 08a STATUS: PASS"
        )

        print(
            "Virtual IFG convention and "
            "network/date indexing are consistent."
        )

        print(
            "No persistent Npoint x Npair IFG cube "
            "was generated."
        )

    else:

        print(
            "STEP 08a STATUS: FAIL"
        )

        raise RuntimeError(
            "Virtual IFG quality failed. "
            "Do not continue to filtering/unwrapping."
        )


if __name__ == "__main__":
    main()
