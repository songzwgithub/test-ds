#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypsds.prototype import open_from_config


def pdate(s):
    return datetime.strptime(
        str(s),
        "%Y%m%d",
    )


def load_network(path: Path):
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
            "Duplicate edges found in network."
        )

    return sorted(edges)


def build_adj(n, edges):
    adj = [
        set()
        for _ in range(n)
    ]

    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    return adj


def enumerate_cycles_k(
    n,
    adj,
    k,
):
    """
    Enumerate each undirected simple cycle of
    exactly length k once.

    Rules:
      - start is smallest node in cycle
      - path[1] < path[-1] removes reversal
    """

    cycles = []

    for start in range(n):

        path = [start]
        used = {start}

        def dfs(u):

            if len(path) == k:

                if start not in adj[u]:
                    return

                # Remove reverse duplicate.
                if path[1] > path[-1]:
                    return

                cycles.append(
                    tuple(path)
                )

                return

            for v in sorted(adj[u]):

                if v == start:
                    continue

                # Ensures start is smallest node
                # in this cycle.
                if v < start:
                    continue

                if v in used:
                    continue

                used.add(v)
                path.append(v)

                dfs(v)

                path.pop()
                used.remove(v)

        dfs(start)

    return cycles


def cycle_edges(cycle):
    k = len(cycle)

    out = []

    for q in range(k):

        i = cycle[q]
        j = cycle[
            (q + 1) % k
        ]

        if i > j:
            i, j = j, i

        out.append(
            (i, j)
        )

    return out


def shortest_cycle_for_edge(
    n,
    adj,
    edge,
):
    """
    Remove edge (u,v), find shortest remaining
    path u -> v.

    shortest cycle length = path length + 1.
    """

    u, v = edge

    q = deque([
        (u, 0)
    ])

    seen = {u}

    while q:

        x, dist = q.popleft()

        for y in adj[x]:

            # Temporarily remove target edge.
            if (
                (x == u and y == v)
                or
                (x == v and y == u)
            ):
                continue

            if y == v:
                return dist + 2

            if y in seen:
                continue

            seen.add(y)

            q.append(
                (
                    y,
                    dist + 1,
                )
            )

    return None


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

    dates = list(stack.dates)
    n = len(dates)

    netdir = (
        Path(paths.output_dir)
        / "v09"
        / "network"
    )

    network_path = (
        netdir
        / "network_directional.itab"
    )

    if not network_path.exists():
        raise FileNotFoundError(
            network_path
        )

    edges = load_network(
        network_path
    )

    E = len(edges)

    adj = build_adj(
        n,
        edges,
    )

    bperp = np.load(
        netdir
        / "acquisition_bperp_m.npy"
    ).astype(
        np.float64
    )

    outdir = (
        netdir
        / "cycle_audit_directional"
    )

    figdir = (
        outdir
        / "figures"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "Step 07c - Short-cycle coverage audit"
    )
    print("=" * 80)

    print(
        f"config             : {config_path}"
    )

    print(
        f"nodes              : {n}"
    )

    print(
        f"edges              : {E}"
    )

    print(
        f"cycle rank         : {E-n+1}"
    )

    # =========================================================
    # 1. Enumerate 3/4/5 cycles
    # =========================================================

    print()
    print(
        "[07c:1/5] Enumerating 3/4/5-node cycles..."
    )

    cycles = {}

    for k in (
        3,
        4,
        5,
    ):

        cycles[k] = enumerate_cycles_k(
            n,
            adj,
            k,
        )

        print(
            f"  {k}-cycles          : "
            f"{len(cycles[k])}"
        )

    # =========================================================
    # 2. Per-edge short-cycle coverage
    # =========================================================

    print()
    print(
        "[07c:2/5] Computing per-edge cycle coverage..."
    )

    edge_index = {
        e: q
        for q, e in enumerate(edges)
    }

    count3 = np.zeros(
        E,
        dtype=np.int32,
    )

    count4 = np.zeros(
        E,
        dtype=np.int32,
    )

    count5 = np.zeros(
        E,
        dtype=np.int32,
    )

    lookup = {
        3: count3,
        4: count4,
        5: count5,
    }

    for k in (
        3,
        4,
        5,
    ):

        arr = lookup[k]

        for cyc in cycles[k]:

            for e in cycle_edges(cyc):

                arr[
                    edge_index[e]
                ] += 1

    shortest = np.full(
        E,
        -1,
        dtype=np.int16,
    )

    for q, e in enumerate(edges):

        L = shortest_cycle_for_edge(
            n,
            adj,
            e,
        )

        if L is None:
            raise RuntimeError(
                f"Edge {e} belongs to no cycle, "
                "but network was expected bridge-free."
            )

        shortest[q] = L

    # =========================================================
    # 3. Coverage statistics
    # =========================================================

    print()
    print(
        "[07c:3/5] Coverage summary..."
    )

    any3 = count3 > 0
    any4 = count4 > 0
    any5 = count5 > 0

    upto4 = (
        any3 | any4
    )

    upto5 = (
        upto4 | any5
    )

    print()
    print("=" * 80)
    print(
        "Short-loop coverage"
    )
    print("=" * 80)

    print(
        f"edges in >=1 triangle       : "
        f"{any3.sum()}/{E} "
        f"({100*any3.mean():.2f}%)"
    )

    print(
        f"edges in >=1 4-cycle        : "
        f"{any4.sum()}/{E} "
        f"({100*any4.mean():.2f}%)"
    )

    print(
        f"edges in >=1 5-cycle        : "
        f"{any5.sum()}/{E} "
        f"({100*any5.mean():.2f}%)"
    )

    print(
        f"covered by 3 or 4 cycle     : "
        f"{upto4.sum()}/{E} "
        f"({100*upto4.mean():.2f}%)"
    )

    print(
        f"covered by 3/4/5 cycle      : "
        f"{upto5.sum()}/{E} "
        f"({100*upto5.mean():.2f}%)"
    )

    print()
    print(
        f"shortest-cycle min/median/max: "
        f"{shortest.min()} / "
        f"{np.median(shortest):.1f} / "
        f"{shortest.max()}"
    )

    vals, nums = np.unique(
        shortest,
        return_counts=True,
    )

    print()
    print(
        "Shortest cycle length distribution:"
    )

    for v, num in zip(
        vals,
        nums,
    ):

        print(
            f"  L={int(v):2d}: "
            f"{int(num):3d} edges "
            f"({100*num/E:6.2f}%)"
        )

    # =========================================================
    # 4. Weakly covered edges
    # =========================================================

    print()
    print(
        "[07c:4/5] Auditing weakly covered edges..."
    )

    print()
    print(
        "Edges not covered by any 3/4/5-cycle:"
    )

    weak = np.where(
        ~upto5
    )[0]

    if len(weak) == 0:

        print(
            "  none"
        )

    else:

        for q in weak:

            i, j = edges[q]

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
                    -
                    bperp[i]
                )
            )

            print(
                f"  edge {i+1:02d}-{j+1:02d} "
                f"{dates[i]}->{dates[j]} "
                f"dT={dt:3d} d "
                f"dB={db:7.2f} m "
                f"shortest_cycle={shortest[q]}"
            )

    # =========================================================
    # 5. Save
    # =========================================================

    print()
    print(
        "[07c:5/5] Saving cycle products..."
    )

    # ---------------------------------------------------------
    # Edge CSV
    # ---------------------------------------------------------

    edge_csv = (
        outdir
        / "edge_cycle_coverage.csv"
    )

    with edge_csv.open(
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
            "triangle_count",
            "cycle4_count",
            "cycle5_count",
            "shortest_cycle_length",
        ])

        for q, e in enumerate(
            edges,
            start=1,
        ):

            i, j = e

            w.writerow([
                q,
                i + 1,
                j + 1,
                dates[i],
                dates[j],
                int(
                    count3[q-1]
                ),
                int(
                    count4[q-1]
                ),
                int(
                    count5[q-1]
                ),
                int(
                    shortest[q-1]
                ),
            ])

    # ---------------------------------------------------------
    # Explicit cycle lists
    # ---------------------------------------------------------

    for k in (
        3,
        4,
        5,
    ):

        p = (
            outdir
            / f"cycles_{k}.csv"
        )

        with p.open(
            "w",
            newline="",
        ) as f:

            w = csv.writer(f)

            w.writerow(
                [
                    "cycle_id"
                ]
                +
                [
                    f"node{q+1}"
                    for q in range(k)
                ]
                +
                [
                    f"date{q+1}"
                    for q in range(k)
                ]
            )

            for cid, cyc in enumerate(
                cycles[k],
                start=1,
            ):

                w.writerow(
                    [cid]
                    +
                    [
                        x + 1
                        for x in cyc
                    ]
                    +
                    [
                        dates[x]
                        for x in cyc
                    ]
                )

    # ---------------------------------------------------------
    # Shortest cycle histogram
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bins = np.arange(
        shortest.min() - 0.5,
        shortest.max() + 1.5,
        1,
    )

    ax.hist(
        shortest,
        bins=bins,
        rwidth=0.85,
    )

    ax.set_xlabel(
        "Shortest cycle length containing edge"
    )

    ax.set_ylabel(
        "Number of network edges"
    )

    ax.set_title(
        "Step 07c - Shortest closure-loop length"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        figdir
        / "07c_shortest_cycle_histogram.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Time-Bperp graph colored by shortest cycle length
    # ---------------------------------------------------------

    x = [
        pdate(d)
        for d in dates
    ]

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    cmap = plt.get_cmap(
        "viridis"
    )

    vmin = float(
        shortest.min()
    )

    vmax = float(
        shortest.max()
    )

    denom = max(
        vmax - vmin,
        1.0,
    )

    for q, (
        i,
        j,
    ) in enumerate(edges):

        frac = (
            shortest[q]
            - vmin
        ) / denom

        ax.plot(
            [
                x[i],
                x[j],
            ],
            [
                bperp[i],
                bperp[j],
            ],
            linewidth=1.5,
            color=cmap(frac),
            alpha=0.9,
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

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    cb = fig.colorbar(
        sm,
        ax=ax,
    )

    cb.set_label(
        "Shortest cycle length"
    )

    ax.set_xlabel(
        "Acquisition date"
    )

    ax.set_ylabel(
        "Perpendicular baseline (m)"
    )

    ax.set_title(
        "Step 07c - Network edge short-cycle coverage"
    )

    ax.grid(
        alpha=0.2
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        figdir
        / "07c_network_shortest_cycle.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Text summary
    # ---------------------------------------------------------

    with (
        outdir
        / "summary.txt"
    ).open(
        "w"
    ) as f:

        f.write(
            f"nodes={n}\n"
        )

        f.write(
            f"edges={E}\n"
        )

        f.write(
            f"cycle_rank={E-n+1}\n"
        )

        f.write(
            f"triangles={len(cycles[3])}\n"
        )

        f.write(
            f"cycles4={len(cycles[4])}\n"
        )

        f.write(
            f"cycles5={len(cycles[5])}\n"
        )

        f.write(
            f"edge_triangle_coverage={any3.sum()}\n"
        )

        f.write(
            f"edge_3or4_coverage={upto4.sum()}\n"
        )

        f.write(
            f"edge_3or4or5_coverage={upto5.sum()}\n"
        )

        f.write(
            f"shortest_cycle_min={shortest.min()}\n"
        )

        f.write(
            f"shortest_cycle_median={np.median(shortest):.6f}\n"
        )

        f.write(
            f"shortest_cycle_max={shortest.max()}\n"
        )

    print()
    print(
        f"edge table         : {edge_csv}"
    )

    print(
        f"cycle products     : {outdir}"
    )

    print()
    print(
        "STEP 07c STATUS: PASS"
    )


if __name__ == "__main__":
    main()
