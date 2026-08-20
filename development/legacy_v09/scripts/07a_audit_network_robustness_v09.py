#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from pypsds.config import cfg_get
from pypsds.prototype import open_from_config


def parse_date(s):
    return datetime.strptime(str(s), "%Y%m%d")


def build_edges(dates, bperp, tmax, bmax):
    edges = []

    for i in range(len(dates)-1):
        for j in range(i+1, len(dates)):

            dt = abs(
                (
                    parse_date(dates[j])
                    - parse_date(dates[i])
                ).days
            )

            db = abs(
                float(bperp[j] - bperp[i])
            )

            if dt <= tmax and db <= bmax:
                edges.append((i, j, dt, db))

    return edges


def degree(n, edges):
    d = np.zeros(n, dtype=np.int32)

    for e in edges:
        i, j = e[0], e[1]
        d[i] += 1
        d[j] += 1

    return d


def components(n, edges):
    adj = [[] for _ in range(n)]

    for e in edges:
        i, j = e[0], e[1]
        adj[i].append(j)
        adj[j].append(i)

    seen = np.zeros(n, dtype=bool)
    comps = []

    for s in range(n):
        if seen[s]:
            continue

        stack = [s]
        seen[s] = True
        comp = []

        while stack:
            u = stack.pop()
            comp.append(u)

            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

        comps.append(sorted(comp))

    return comps


def bridges_with_cuts(n, edges):
    adj = [[] for _ in range(n)]

    for eid, e in enumerate(edges):
        i, j = e[0], e[1]
        adj[i].append((j, eid))
        adj[j].append((i, eid))

    tin = [-1] * n
    low = [-1] * n
    timer = 0
    bridge_ids = []

    def dfs(u, parent_eid):
        nonlocal timer

        tin[u] = timer
        low[u] = timer
        timer += 1

        for v, eid in adj[u]:

            if eid == parent_eid:
                continue

            if tin[v] >= 0:
                low[u] = min(low[u], tin[v])
            else:
                dfs(v, eid)
                low[u] = min(low[u], low[v])

                if low[v] > tin[u]:
                    bridge_ids.append(eid)

    for u in range(n):
        if tin[u] < 0:
            dfs(u, -1)

    # For each bridge, find the two component sizes after removal.
    out = []

    for eid in bridge_ids:
        kept = [
            e
            for k, e in enumerate(edges)
            if k != eid
        ]

        cs = components(n, kept)

        sizes = sorted(
            [len(c) for c in cs]
        )

        out.append(
            (
                edges[eid],
                sizes,
            )
        )

    return out


def load_selected_itab(path, dates, bperp):
    edges = []

    for raw in path.read_text().splitlines():
        f = raw.split()

        if len(f) < 2:
            continue

        i = int(f[0]) - 1
        j = int(f[1]) - 1

        dt = abs(
            (
                parse_date(dates[j])
                - parse_date(dates[i])
            ).days
        )

        db = abs(
            float(bperp[j] - bperp[i])
        )

        edges.append(
            (i, j, dt, db)
        )

    return edges


def print_bridge_table(name, bridges, dates):
    print()
    print(f"{name} bridges: {len(bridges)}")

    if not bridges:
        return

    print(
        "  edge   date1      date2      "
        "dT[d]   dBperp[m]   cut sizes"
    )

    for edge, sizes in bridges:
        i, j, dt, db = edge

        print(
            f"  {i+1:02d}-{j+1:02d}  "
            f"{dates[i]}  {dates[j]}  "
            f"{dt:5.0f}   {db:9.2f}   "
            f"{sizes}"
        )


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
    ) = open_from_config(args.config)

    dates = list(stack.dates)
    n = len(dates)

    t0 = float(
        cfg_get(
            cfg,
            "network.max_temporal_baseline_days",
            72.0,
        )
    )

    b0 = float(
        cfg_get(
            cfg,
            "network.max_perpendicular_baseline_m",
            150.0,
        )
    )

    kmax = int(
        cfg_get(
            cfg,
            "network.max_connections_per_acquisition",
            4,
        )
    )

    netdir = (
        Path(paths.output_dir)
        / "v09"
        / "network"
    )

    bperp = np.load(
        netdir / "acquisition_bperp_m.npy"
    ).astype(np.float64)

    selected = load_selected_itab(
        netdir / "network.itab",
        dates,
        bperp,
    )

    print("=" * 80)
    print("Step 07a - Network robustness audit")
    print("=" * 80)

    print(f"config            : {config_path}")
    print(f"nodes             : {n}")
    print(f"current Tmax      : {t0:g} d")
    print(f"current Bmax      : {b0:g} m")
    print(f"current degree cap: {kmax}")

    # ========================================================
    # Current selected graph
    # ========================================================

    ds = degree(n, selected)
    bs = bridges_with_cuts(
        n,
        selected,
    )

    print()
    print("=" * 80)
    print("Current selected network")
    print("=" * 80)

    print(f"edges             : {len(selected)}")
    print(f"components        : {len(components(n, selected))}")
    print(
        f"degree min/med/max: "
        f"{ds.min()} / {np.median(ds):.1f} / {ds.max()}"
    )
    print(f"bridges           : {len(bs)}")

    print_bridge_table(
        "Selected-network",
        bs,
        dates,
    )

    print()
    print("Selected node degrees:")

    for i in range(n):
        flag = "  ***" if ds[i] <= 1 else ""

        print(
            f"  {i+1:02d} {dates[i]} "
            f"Bperp={bperp[i]:9.2f} "
            f"degree={ds[i]}{flag}"
        )

    # ========================================================
    # Current candidate graph
    # ========================================================

    candidate = build_edges(
        dates,
        bperp,
        t0,
        b0,
    )

    dc = degree(
        n,
        candidate,
    )

    bc = bridges_with_cuts(
        n,
        candidate,
    )

    print()
    print("=" * 80)
    print("Current threshold-candidate graph")
    print("=" * 80)

    print(f"candidate edges   : {len(candidate)}")
    print(
        f"components        : "
        f"{len(components(n,candidate))}"
    )

    print(
        f"degree min/med/max: "
        f"{dc.min()} / {np.median(dc):.1f} / {dc.max()}"
    )

    print(f"bridges           : {len(bc)}")

    print_bridge_table(
        "Candidate-network",
        bc,
        dates,
    )

    # ========================================================
    # Problem nodes: show nearest excluded alternatives
    # ========================================================

    problem_nodes = np.where(
        dc < 2
    )[0]

    print()
    print("=" * 80)
    print("Candidate nodes with degree < 2")
    print("=" * 80)

    if len(problem_nodes) == 0:
        print("none")
    else:
        for i in problem_nodes:

            print()
            print(
                f"Node {i+1:02d} {dates[i]} "
                f"Bperp={bperp[i]:.3f} m "
                f"candidate degree={dc[i]}"
            )

            alternatives = []

            for j in range(n):

                if j == i:
                    continue

                dt = abs(
                    (
                        parse_date(dates[j])
                        - parse_date(dates[i])
                    ).days
                )

                db = abs(
                    float(
                        bperp[j]
                        - bperp[i]
                    )
                )

                currently_valid = (
                    dt <= t0
                    and
                    db <= b0
                )

                if currently_valid:
                    continue

                # How far outside the rectangular
                # threshold box is this pair?
                violation = max(
                    dt / t0,
                    db / b0,
                )

                alternatives.append(
                    (
                        violation,
                        j,
                        dt,
                        db,
                    )
                )

            alternatives.sort()

            print(
                "  nearest excluded alternatives:"
            )

            for _, j, dt, db in alternatives[:5]:

                need = []

                if dt > t0:
                    need.append(
                        f"Tmax>={dt:.0f}d"
                    )

                if db > b0:
                    need.append(
                        f"Bmax>={db:.1f}m"
                    )

                print(
                    f"    -> {j+1:02d} {dates[j]} "
                    f"dT={dt:5.0f} d "
                    f"dB={db:8.2f} m "
                    f"needs: {', '.join(need)}"
                )

    # ========================================================
    # Minimal local threshold sweep
    # ========================================================

    print()
    print("=" * 80)
    print("Candidate-topology threshold sweep")
    print("=" * 80)

    t_values = sorted(
        set([
            t0,
            t0 + 12,
            t0 + 24,
            t0 + 36,
        ])
    )

    b_values = sorted(
        set([
            b0,
            b0 + 10,
            b0 + 20,
            b0 + 30,
            b0 + 50,
        ])
    )

    print(
        " Tmax  Bmax | edges comps minDeg bridges | robust-candidate"
    )

    robust_options = []

    for tmax in t_values:

        for bmax in b_values:

            ee = build_edges(
                dates,
                bperp,
                tmax,
                bmax,
            )

            dd = degree(
                n,
                ee,
            )

            cc = components(
                n,
                ee,
            )

            bb = bridges_with_cuts(
                n,
                ee,
            )

            robust = (
                len(cc) == 1
                and dd.min() >= 2
                and len(bb) == 0
            )

            print(
                f"{tmax:5.0f} {bmax:5.0f} | "
                f"{len(ee):5d} "
                f"{len(cc):5d} "
                f"{dd.min():6d} "
                f"{len(bb):7d} | "
                f"{'YES' if robust else 'no'}"
            )

            if robust:
                robust_options.append(
                    (
                        tmax,
                        bmax,
                        len(ee),
                    )
                )

    print()
    if robust_options:

        # Prefer smallest normalized relaxation
        robust_options.sort(
            key=lambda x: (
                x[0] / t0
                + x[1] / b0,
                x[0],
                x[1],
            )
        )

        print(
            "Smallest tested robust candidate option:"
        )

        print(
            f"  Tmax = {robust_options[0][0]:g} d"
        )

        print(
            f"  Bmax = {robust_options[0][1]:g} m"
        )

    else:
        print(
            "No tested threshold pair produced "
            "a 2-edge-connected candidate graph."
        )

    print()
    print("STEP 07a STATUS: PASS")


if __name__ == "__main__":
    main()
