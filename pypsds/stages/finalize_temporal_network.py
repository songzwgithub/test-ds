#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pypsds.config import cfg_get
from pypsds.context import open_from_config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pdate(s):
    return datetime.strptime(str(s), "%Y%m%d")


def load_itab(path: Path):
    edges = []

    for raw in path.read_text().splitlines():
        f = raw.split()

        if len(f) < 2:
            continue

        i = int(f[0]) - 1
        j = int(f[1]) - 1

        if i > j:
            i, j = j, i

        edges.append((i, j))

    if len(edges) != len(set(edges)):
        raise RuntimeError(
            f"Duplicate edges in {path}"
        )

    return edges


def components(n, edges):
    adj = [[] for _ in range(n)]

    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    seen = np.zeros(n, dtype=bool)
    out = []

    for s in range(n):

        if seen[s]:
            continue

        seen[s] = True
        todo = [s]
        comp = []

        while todo:
            u = todo.pop()
            comp.append(u)

            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    todo.append(v)

        out.append(sorted(comp))

    return out


def bridge_count(n, edges):
    adj = [[] for _ in range(n)]

    for eid, (i, j) in enumerate(edges):
        adj[i].append((j, eid))
        adj[j].append((i, eid))

    tin = [-1] * n
    low = [-1] * n
    timer = 0
    bridges = []

    def dfs(u, parent_eid):
        nonlocal timer

        tin[u] = timer
        low[u] = timer
        timer += 1

        for v, eid in adj[u]:

            if eid == parent_eid:
                continue

            if tin[v] >= 0:
                low[u] = min(
                    low[u],
                    tin[v],
                )
            else:
                dfs(v, eid)

                low[u] = min(
                    low[u],
                    low[v],
                )

                if low[v] > tin[u]:
                    bridges.append(eid)

    for u in range(n):
        if tin[u] < 0:
            dfs(u, -1)

    return len(bridges)


def read_summary(path: Path):
    out = {}

    if not path.exists():
        return out

    for raw in path.read_text().splitlines():

        if "=" not in raw:
            continue

        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip()

        try:
            if "." in v:
                val = float(v)
            else:
                val = int(v)
        except ValueError:
            val = v

        out[k] = val

    return out


def backup_once(src: Path, dst: Path):

    if (
        src.exists()
        and
        not dst.exists()
    ):
        shutil.copy2(src, dst)


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

    dates = list(stack.dates)
    n = len(dates)

    tmax = float(
        cfg_get(
            cfg,
            "network.max_temporal_baseline_days",
            72,
        )
    )

    bmax = float(
        cfg_get(
            cfg,
            "network.max_perpendicular_baseline_m",
            200,
        )
    )

    target = int(
        cfg_get(
            cfg,
            "network.target_connections_each_side",
            3,
        )
    )

    netdir = (
        Path(paths.output_dir)
        / "processing"
        / "network"
    )

    src_itab = (
        netdir
        / "network_directional.itab"
    )

    src_pairs = (
        netdir
        / "pairs_directional.csv"
    )

    if not src_itab.exists():
        raise FileNotFoundError(src_itab)

    if not src_pairs.exists():
        raise FileNotFoundError(src_pairs)

    bperp_path = (
        netdir
        / "acquisition_bperp_m.npy"
    )

    bperp = np.load(
        bperp_path
    ).astype(np.float64)

    if len(bperp) != n:
        raise RuntimeError(
            "Bperp acquisition count mismatch."
        )

    edges = load_itab(
        src_itab
    )

    E = len(edges)

    # ---------------------------------------------------------
    # Recompute candidate count from frozen thresholds
    # ---------------------------------------------------------

    candidate_count = 0

    # Number of physically admissible candidate edges on each side
    # after applying the temporal/perpendicular-baseline thresholds.
    #
    # The directional target is therefore:
    #
    #   min(target, number_of_available_candidates_on_that_side)
    #
    # rather than min(target, number_of_acquisitions_on_that_side).
    # A side with only one or two admissible candidate edges cannot
    # be required to contain three selected edges.
    candidate_left = np.zeros(
        n,
        dtype=np.int32,
    )

    candidate_right = np.zeros(
        n,
        dtype=np.int32,
    )

    for i in range(n - 1):
        for j in range(i + 1, n):

            dt = abs(
                (
                    pdate(dates[j])
                    - pdate(dates[i])
                ).days
            )

            db = abs(
                float(
                    bperp[j]
                    - bperp[i]
                )
            )

            if (
                dt <= tmax
                and
                db <= bmax
            ):
                candidate_count += 1

                # i < j by construction:
                # edge (i,j) is a right-side candidate for i
                # and a left-side candidate for j.
                candidate_right[i] += 1
                candidate_left[j] += 1

    # ---------------------------------------------------------
    # Degree / directional counts
    # ---------------------------------------------------------

    degree = np.zeros(
        n,
        dtype=np.int32,
    )

    left = np.zeros(
        n,
        dtype=np.int32,
    )

    right = np.zeros(
        n,
        dtype=np.int32,
    )

    for i, j in edges:

        degree[i] += 1
        degree[j] += 1

        right[i] += 1
        left[j] += 1

    # Feasible directional requirement.
    #
    # 07b selects up to `target` best admissible candidates on each
    # side. Therefore the correct production invariant is bounded by
    # the candidate graph itself.
    req_left = np.minimum(
        candidate_left,
        target,
    ).astype(
        np.int32,
        copy=False,
    )

    req_right = np.minimum(
        candidate_right,
        target,
    ).astype(
        np.int32,
        copy=False,
    )

    left_ok = (
        left >= req_left
    )

    right_ok = (
        right >= req_right
    )

    both_ok = (
        left_ok & right_ok
    )

    comps = components(
        n,
        edges,
    )

    bridges = bridge_count(
        n,
        edges,
    )

    cycle_rank = (
        E - n + len(comps)
    )

    # ---------------------------------------------------------
    # Read already completed 07c quality
    # ---------------------------------------------------------

    cycle_dir = (
        netdir
        / "cycle_quality_directional"
    )

    cycle_summary = read_summary(
        cycle_dir / "summary.txt"
    )


    # ---------------------------------------------------------
    # General production-network checks
    # ---------------------------------------------------------

    if len(comps) != 1:
        raise RuntimeError(
            f"Temporal network is disconnected: "
            f"components={len(comps)}"
        )

    if bridges != 0:
        raise RuntimeError(
            f"Temporal network contains "
            f"{bridges} bridge edge(s)."
        )

    if E < n:
        raise RuntimeError(
            "Temporal network has insufficient "
            "redundancy for closure analysis."
        )

    if not np.all(
        both_ok
    ):
        bad = np.where(
            ~both_ok
        )[0]

        raise RuntimeError(
            "Directional connection requirement "
            "is not satisfied for acquisitions: "
            + ", ".join(
                str(
                    dates[int(i)]
                )
                for i in bad
            )
        )

    # Short-cycle quality is required when the
    # corresponding quality stage is present.
    if cycle_summary:

        coverage = int(
            cycle_summary.get(
                "edge_3or4_coverage",
                -1,
            )
        )

        if coverage != E:
            raise RuntimeError(
                "Not every temporal-network edge "
                "is covered by a 3- or 4-cycle."
            )

        shortest_max = int(
            cycle_summary.get(
                "shortest_cycle_max",
                -1,
            )
        )

        if (
            shortest_max < 0
            or
            shortest_max > 4
        ):
            raise RuntimeError(
                "Temporal-network shortest-cycle "
                "maximum exceeds 4."
            )

    # ---------------------------------------------------------
    # Preserve previous production aliases
    # ---------------------------------------------------------

    production_itab = (
        netdir
        / "network.itab"
    )

    production_pairs = (
        netdir
        / "pairs.csv"
    )

    production_degree = (
        netdir
        / "degree.npy"
    )

    backup_once(
        production_itab,
        netdir
        / "network_pre_directional.itab",
    )

    backup_once(
        production_pairs,
        netdir
        / "pairs_pre_directional.csv",
    )

    backup_once(
        production_degree,
        netdir
        / "degree_pre_directional.npy",
    )

    # ---------------------------------------------------------
    # Promote directional products
    # ---------------------------------------------------------

    shutil.copy2(
        src_itab,
        production_itab,
    )

    shutil.copy2(
        src_pairs,
        production_pairs,
    )

    np.save(
        production_degree,
        degree,
    )

    np.save(
        netdir
        / "left_connection_count.npy",
        left,
    )

    np.save(
        netdir
        / "right_connection_count.npy",
        right,
    )

    # ---------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------

    phase_ref = cfg_get(
        cfg,
        "phase_correction.geometric_reference_date",
        None,
    )

    manifest = {
        "format": "pyPSDS-GAMMA-network-v1.0",
        "status": "FROZEN",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "algorithm": (
            "directional_spatiotemporal_target_"
            "with_connectivity_and_bridge_repair"
        ),

        "baseline_backend": {
            "software": "GAMMA",
            "program": "base_calc",
            "orbit_program": "base_orbit",
            "reference_date": phase_ref,
            "all_pair_baseline_file": str(
                netdir
                / "gamma_base_calc"
                / "all_pairs.bperp"
            ),
        },

        "parameters": {
            "max_temporal_baseline_days": tmax,
            "max_perpendicular_baseline_m": bmax,
            "target_connections_each_side": target,
        },

        "network": {
            "acquisitions": n,
            "all_possible_pairs": (
                n * (n - 1) // 2
            ),
            "candidate_pairs": candidate_count,
            "selected_pairs": E,
            "components": len(comps),
            "bridges": bridges,
            "cycle_rank": cycle_rank,

            "degree": {
                "min": int(
                    degree.min()
                ),
                "median": float(
                    np.median(degree)
                ),
                "max": int(
                    degree.max()
                ),
                "mean": float(
                    degree.mean()
                ),
            },

            "directional_target": {
                "left_satisfied": int(
                    left_ok.sum()
                ),
                "right_satisfied": int(
                    right_ok.sum()
                ),
                "both_sides_satisfied": int(
                    both_ok.sum()
                ),
                "total_acquisitions": n,
            },
        },

        "cycle_quality": cycle_summary,

        "products": {
            "production_itab": str(
                production_itab
            ),
            "production_pairs": str(
                production_pairs
            ),
            "degree": str(
                production_degree
            ),
            "left_connection_count": str(
                netdir
                / "left_connection_count.npy"
            ),
            "right_connection_count": str(
                netdir
                / "right_connection_count.npy"
            ),
            "cycle_quality_directory": str(
                cycle_dir
            ),
        },

        "sha256": {
            "network_itab": sha256(
                production_itab
            ),
            "pairs_csv": sha256(
                production_pairs
            ),
            "acquisition_bperp_m": sha256(
                bperp_path
            ),
        },

        "dates": dates,
    }

    manifest_path = (
        netdir
        / "network_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    # ---------------------------------------------------------
    # Human-readable summary
    # ---------------------------------------------------------

    summary_path = (
        netdir
        / "NETWORK_FROZEN.txt"
    )

    summary_path.write_text(
        "\n".join([
            "pyPSDS-GAMMA v1.0 production network",
            "=" * 60,
            "",
            "STATUS: FROZEN",
            "",
            f"Acquisitions             : {n}",
            f"All possible pairs       : {n*(n-1)//2}",
            f"Candidate pairs          : {candidate_count}",
            f"Selected pairs           : {E}",
            "",
            f"Tmax                     : {tmax:g} days",
            f"Bperp max                : {bmax:g} m",
            f"Target connections/side  : {target}",
            "",
            f"Components               : {len(comps)}",
            f"Bridges                  : {bridges}",
            f"Cycle rank               : {cycle_rank}",
            "",
            (
                "Degree min/median/max    : "
                f"{degree.min()} / "
                f"{np.median(degree):.1f} / "
                f"{degree.max()}"
            ),
            "",
            (
                "Left target satisfied    : "
                f"{left_ok.sum()}/{n}"
            ),
            (
                "Right target satisfied   : "
                f"{right_ok.sum()}/{n}"
            ),
            (
                "Both sides satisfied     : "
                f"{both_ok.sum()}/{n}"
            ),
            "",
            (
                "Production ITAB          : "
                f"{production_itab}"
            ),
            (
                "Production pair table    : "
                f"{production_pairs}"
            ),
            (
                "Manifest                 : "
                f"{manifest_path}"
            ),
            "",
        ])
    )

    print("=" * 80)
    print("Finalize production network")
    print("=" * 80)

    print(
        f"parameters                 : "
        f"{tmax:g} d / {bmax:g} m / "
        f"{target} per side"
    )

    print(
        f"all possible pairs         : "
        f"{n*(n-1)//2}"
    )

    print(
        f"candidate pairs            : "
        f"{candidate_count}"
    )

    print(
        f"production pairs           : {E}"
    )

    print(
        f"components / bridges       : "
        f"{len(comps)} / {bridges}"
    )

    print(
        f"cycle rank                 : "
        f"{cycle_rank}"
    )

    print(
        f"degree min/median/max      : "
        f"{degree.min()} / "
        f"{np.median(degree):.1f} / "
        f"{degree.max()}"
    )

    print(
        f"left/right/both target     : "
        f"{left_ok.sum()} / "
        f"{right_ok.sum()} / "
        f"{both_ok.sum()} "
        f"of {n}"
    )

    if cycle_summary:

        print(
            f"triangles                  : "
            f"{cycle_summary.get('triangles')}"
        )

        print(
            f"4-cycles                   : "
            f"{cycle_summary.get('cycles4')}"
        )

        print(
            f"5-cycles                   : "
            f"{cycle_summary.get('cycles5')}"
        )

        print(
            f"3-or-4 edge coverage       : "
            f"{cycle_summary.get('edge_3or4_coverage')}"
            f"/{E}"
        )

        print(
            f"shortest cycle max         : "
            f"{cycle_summary.get('shortest_cycle_max')}"
        )

    print()
    print(
        f"production network.itab    : "
        f"{production_itab}"
    )

    print(
        f"production pairs.csv       : "
        f"{production_pairs}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print(
        f"summary                    : "
        f"{summary_path}"
    )

    print()
    print(
        "STEP 07d STATUS: PASS / NETWORK FROZEN"
    )


if __name__ == "__main__":
    main()
