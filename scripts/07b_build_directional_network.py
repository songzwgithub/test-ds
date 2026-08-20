#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypsds.config import cfg_get
from pypsds.context import open_from_config


def pdate(s):
    return datetime.strptime(str(s), "%Y%m%d")


def components(n, edges):
    adj = [[] for _ in range(n)]

    for e in edges:
        i, j = e[:2]
        adj[i].append(j)
        adj[j].append(i)

    seen = np.zeros(n, dtype=bool)
    out = []

    for s in range(n):
        if seen[s]:
            continue

        todo = [s]
        seen[s] = True
        comp = []

        while todo:
            u = todo.pop()
            comp.append(u)

            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    todo.append(v)

        out.append(comp)

    return out


def find_bridges(n, edges):
    adj = [[] for _ in range(n)]

    for eid, e in enumerate(edges):
        i, j = e[:2]
        adj[i].append((j, eid))
        adj[j].append((i, eid))

    tin = [-1] * n
    low = [-1] * n
    timer = 0
    bridge_ids = []

    def dfs(u, peid):
        nonlocal timer

        tin[u] = timer
        low[u] = timer
        timer += 1

        for v, eid in adj[u]:

            if eid == peid:
                continue

            if tin[v] >= 0:
                low[u] = min(low[u], tin[v])
            else:
                dfs(v, eid)

                low[u] = min(
                    low[u],
                    low[v],
                )

                if low[v] > tin[u]:
                    bridge_ids.append(eid)

    for u in range(n):
        if tin[u] < 0:
            dfs(u, -1)

    return bridge_ids


def component_labels(n, edges):
    comps = components(n, edges)

    lab = np.full(n, -1, dtype=np.int32)

    for k, c in enumerate(comps):
        for u in c:
            lab[u] = k

    return comps, lab


def bridge_cut(n, edges, bridge_id):
    bi, bj = edges[bridge_id][:2]

    adj = [[] for _ in range(n)]

    for eid, e in enumerate(edges):
        if eid == bridge_id:
            continue

        i, j = e[:2]
        adj[i].append(j)
        adj[j].append(i)

    seen = {bi}
    todo = [bi]

    while todo:
        u = todo.pop()

        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                todo.append(v)

    return seen


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

    figdir = netdir / "figures"

    figdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bperp = np.load(
        netdir / "acquisition_bperp_m.npy"
    ).astype(np.float64)

    print("=" * 80)
    print(
        "Step 07b - Directional spatiotemporal network"
    )
    print("=" * 80)

    print(f"nodes                     : {n}")
    print(f"Tmax                      : {tmax:g} d")
    print(f"Bmax                      : {bmax:g} m")
    print(
        f"target connections/side   : {target}"
    )

    # =========================================================
    # Candidate graph
    # =========================================================

    candidates = []

    for i in range(n - 1):
        for j in range(i + 1, n):

            dt = abs(
                (
                    pdate(dates[j])
                    -
                    pdate(dates[i])
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

                score = math.hypot(
                    dt / tmax,
                    db / bmax,
                )

                candidates.append(
                    (
                        i,
                        j,
                        float(dt),
                        db,
                        score,
                    )
                )

    print(
        f"candidate edges            : "
        f"{len(candidates)}"
    )

    # =========================================================
    # Directional target selection
    # =========================================================

    selected_keys = set()

    for i in range(n):

        left = sorted(
            [
                e
                for e in candidates
                if e[1] == i
            ],
            key=lambda e: (
                e[4],
                e[2],
                e[3],
                e[0],
            ),
        )

        right = sorted(
            [
                e
                for e in candidates
                if e[0] == i
            ],
            key=lambda e: (
                e[4],
                e[2],
                e[3],
                e[1],
            ),
        )

        for e in left[:target]:
            selected_keys.add(
                (e[0], e[1])
            )

        for e in right[:target]:
            selected_keys.add(
                (e[0], e[1])
            )

    candidate_map = {
        (e[0], e[1]): e
        for e in candidates
    }

    selected = [
        candidate_map[k]
        for k in sorted(
            selected_keys
        )
    ]

    print(
        f"directional seed edges      : "
        f"{len(selected)}"
    )

    # =========================================================
    # Repair disconnected components
    # =========================================================

    while True:

        comps, labels = component_labels(
            n,
            selected,
        )

        if len(comps) == 1:
            break

        current = {
            (e[0], e[1])
            for e in selected
        }

        remaining = sorted(
            [
                e
                for e in candidates
                if (
                    e[0],
                    e[1],
                ) not in current
                and
                labels[e[0]]
                != labels[e[1]]
            ],
            key=lambda e: e[4],
        )

        if not remaining:
            raise RuntimeError(
                "Cannot connect directional seed "
                "within current thresholds."
            )

        e = remaining[0]
        selected.append(e)

        print(
            f"  add connectivity edge "
            f"{e[0]+1:02d}-{e[1]+1:02d}"
        )

    # =========================================================
    # Repair bridges
    # =========================================================

    for iteration in range(100):

        bids = find_bridges(
            n,
            selected,
        )

        if not bids:
            break

        current = {
            (e[0], e[1])
            for e in selected
        }

        changed = False

        for bid in bids:

            side = bridge_cut(
                n,
                selected,
                bid,
            )

            remaining = []

            for e in candidates:

                key = (
                    e[0],
                    e[1],
                )

                if key in current:
                    continue

                crosses = (
                    (e[0] in side)
                    !=
                    (e[1] in side)
                )

                if crosses:
                    remaining.append(e)

            remaining.sort(
                key=lambda e: e[4]
            )

            if not remaining:
                raise RuntimeError(
                    "Cannot remove bridge "
                    "within current candidate graph."
                )

            e = remaining[0]

            selected.append(e)
            current.add(
                (
                    e[0],
                    e[1],
                )
            )

            print(
                f"  add bridge-repair edge "
                f"{e[0]+1:02d}-{e[1]+1:02d}"
            )

            changed = True

        if not changed:
            break

    selected = sorted(
        {
            (e[0], e[1]): e
            for e in selected
        }.values(),
        key=lambda e: (
            e[0],
            e[1],
        ),
    )

    # =========================================================
    # Final counts
    # =========================================================

    left = np.zeros(
        n,
        dtype=np.int32,
    )

    right = np.zeros(
        n,
        dtype=np.int32,
    )

    degree = np.zeros(
        n,
        dtype=np.int32,
    )

    for e in selected:

        i, j = e[:2]

        right[i] += 1
        left[j] += 1

        degree[i] += 1
        degree[j] += 1

    comps = components(
        n,
        selected,
    )

    bridges = find_bridges(
        n,
        selected,
    )

    E = len(selected)

    print()
    print("=" * 80)
    print("Final directional network")
    print("=" * 80)

    print(f"edges                     : {E}")
    print(f"components                : {len(comps)}")
    print(f"bridges                   : {len(bridges)}")
    print(
        f"cycle rank                : "
        f"{E-n+1}"
    )

    print(
        f"degree min/median/max     : "
        f"{degree.min()} / "
        f"{np.median(degree):.1f} / "
        f"{degree.max()}"
    )

    print()
    print(
        "Directional connection counts:"
    )

    print(
        " idx date      left right degree"
    )

    for i in range(n):

        print(
            f" {i+1:02d}  "
            f"{dates[i]}  "
            f"{left[i]:4d} "
            f"{right[i]:5d} "
            f"{degree[i]:6d}"
        )

    theoretical_left = np.array(
        [
            min(target, i)
            for i in range(n)
        ]
    )

    theoretical_right = np.array(
        [
            min(
                target,
                n - 1 - i,
            )
            for i in range(n)
        ]
    )

    left_ok = (
        left >= theoretical_left
    )

    right_ok = (
        right >= theoretical_right
    )

    both_ok = (
        left_ok
        & right_ok
    )

    print()
    print(
        f"left target satisfied       : "
        f"{left_ok.sum()}/{n}"
    )

    print(
        f"right target satisfied      : "
        f"{right_ok.sum()}/{n}"
    )

    print(
        f"both sides satisfied        : "
        f"{both_ok.sum()}/{n}"
    )

    # =========================================================
    # Save itab
    # =========================================================

    itab = (
        netdir
        / "network_directional.itab"
    )

    with itab.open(
        "w"
    ) as f:

        for eid, e in enumerate(
            selected,
            start=1,
        ):

            f.write(
                f"{e[0]+1:4d} "
                f"{e[1]+1:4d} "
                f"{eid:4d}  1\n"
            )

    np.save(
        netdir
        / "directional_left_count.npy",
        left,
    )

    np.save(
        netdir
        / "directional_right_count.npy",
        right,
    )

    np.save(
        netdir
        / "directional_degree.npy",
        degree,
    )

    # =========================================================
    # Pair table
    # =========================================================

    with (
        netdir
        / "pairs_directional.csv"
    ).open(
        "w",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "edge_id",
            "i1",
            "j1",
            "date1",
            "date2",
            "delta_t_days",
            "delta_bperp_m",
            "score",
        ])

        for eid, e in enumerate(
            selected,
            start=1,
        ):

            i, j, dt, db, score = e

            w.writerow([
                eid,
                i + 1,
                j + 1,
                dates[i],
                dates[j],
                dt,
                db,
                score,
            ])

    # =========================================================
    # Plot
    # =========================================================

    x = [
        pdate(d)
        for d in dates
    ]

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    for e in candidates:

        i, j = e[:2]

        ax.plot(
            [x[i], x[j]],
            [bperp[i], bperp[j]],
            linewidth=0.35,
            alpha=0.04,
        )

    for e in selected:

        i, j = e[:2]

        ax.plot(
            [x[i], x[j]],
            [bperp[i], bperp[j]],
            linewidth=1.1,
            alpha=0.75,
        )

    ax.scatter(
        x,
        bperp,
        s=28,
        zorder=3,
    )

    for i in range(n):
        ax.annotate(
            str(i + 1),
            (
                x[i],
                bperp[i],
            ),
            xytext=(3,3),
            textcoords="offset points",
            fontsize=7,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Acquisition date"
    )

    ax.set_ylabel(
        "Perpendicular baseline (m)"
    )

    ax.set_title(
        "pyPSDS-GAMMA directional interferogram network\n"
        f"|ΔT|≤{tmax:g} d, "
        f"|ΔB⊥|≤{bmax:g} m, "
        f"target={target}/side; "
        f"E={E}, cycle rank={E-n+1}"
    )

    ax.grid(
        alpha=0.2
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    png = (
        figdir
        / "07_directional_time_bperp_network.png"
    )

    fig.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"network itab              : {itab}")
    print(f"network plot              : {png}")

    if (
        len(comps) != 1
        or
        len(bridges) != 0
    ):
        raise RuntimeError(
            "Final network topology failed."
        )

    print()
    print(
        "STEP 07b STATUS: PASS"
    )


if __name__ == "__main__":
    main()
