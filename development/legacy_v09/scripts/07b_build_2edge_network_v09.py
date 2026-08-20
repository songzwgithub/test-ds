#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import (
    Bounds,
    LinearConstraint,
    milp,
)
from scipy.sparse import lil_matrix

from pypsds.config import cfg_get
from pypsds.prototype import open_from_config


def pdate(s):
    return datetime.strptime(
        str(s),
        "%Y%m%d",
    )


def build_candidates(
    dates,
    bperp,
    tmax,
    bmax,
):
    edges = []

    n = len(dates)

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

                score = math.hypot(
                    dt / tmax,
                    db / bmax,
                )

                edges.append(
                    (
                        i,
                        j,
                        float(dt),
                        db,
                        score,
                    )
                )

    return edges


def degrees(n, edges):
    d = np.zeros(
        n,
        dtype=np.int32,
    )

    for e in edges:
        d[e[0]] += 1
        d[e[1]] += 1

    return d


def components(
    n,
    edges,
):
    adj = [
        []
        for _ in range(n)
    ]

    for e in edges:
        i, j = e[0], e[1]

        adj[i].append(j)
        adj[j].append(i)

    seen = np.zeros(
        n,
        dtype=bool,
    )

    out = []

    for s in range(n):

        if seen[s]:
            continue

        todo = [s]
        seen[s] = True
        c = []

        while todo:
            u = todo.pop()
            c.append(u)

            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    todo.append(v)

        out.append(
            sorted(c)
        )

    return out


def bridge_cuts(
    n,
    edges,
):
    """
    Return every bridge together with one side
    of the cut produced by removing that bridge.
    """

    adj = [
        []
        for _ in range(n)
    ]

    for eid, e in enumerate(edges):
        i, j = e[0], e[1]

        adj[i].append(
            (j, eid)
        )
        adj[j].append(
            (i, eid)
        )

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
                    bridge_ids.append(eid)

    for u in range(n):
        if tin[u] < 0:
            dfs(u, -1)

    result = []

    for eid in bridge_ids:

        bi, bj = (
            edges[eid][0],
            edges[eid][1],
        )

        # BFS after removing bridge.
        seen = {bi}
        todo = [bi]

        while todo:
            u = todo.pop()

            for v, e2 in adj[u]:

                if e2 == eid:
                    continue

                if v not in seen:
                    seen.add(v)
                    todo.append(v)

        result.append(
            (
                edges[eid],
                seen,
            )
        )

    return result


def normalize_cut(
    S,
    n,
):
    S = frozenset(S)

    C = frozenset(
        set(range(n))
        - set(S)
    )

    if len(S) < len(C):
        return S

    if len(C) < len(S):
        return C

    # Equal-size deterministic choice.
    return min(
        S,
        C,
        key=lambda x:
            tuple(sorted(x)),
    )


def solve_with_cuts(
    n,
    candidates,
    degree_max,
):
    """
    Binary MILP.

    Primary objective:
        maximize number of selected edges.

    Secondary:
        prefer shorter normalized
        temporal/perpendicular baseline.

    Constraints:
        2 <= degree(i) <= degree_max

    Connectivity and bridge-free conditions are added
    iteratively as violated cut constraints:

        x(delta(S)) >= 2
    """

    m = len(candidates)

    if m == 0:
        raise RuntimeError(
            "No candidate edges."
        )

    score = np.array(
        [
            e[4]
            for e in candidates
        ],
        dtype=np.float64,
    )

    # One extra selected edge improves objective by ~1.
    # Total score penalty is much smaller than 1, so
    # cardinality is strictly the primary objective.
    c = (
        -np.ones(
            m,
            dtype=np.float64,
        )
        + 1.0e-4 * score
    )

    cut_sets = []
    cut_keys = set()

    for iteration in range(
        1,
        51,
    ):

        nrow = (
            n
            + len(cut_sets)
        )

        A = lil_matrix(
            (
                nrow,
                m,
            ),
            dtype=np.float64,
        )

        lower = np.full(
            nrow,
            -np.inf,
            dtype=np.float64,
        )

        upper = np.full(
            nrow,
            +np.inf,
            dtype=np.float64,
        )

        # ----------------------------------------
        # Degree:
        # 2 <= degree <= configured maximum
        # ----------------------------------------

        for k, e in enumerate(
            candidates
        ):

            i, j = e[0], e[1]

            A[i, k] = 1.0
            A[j, k] = 1.0

        lower[:n] = 2.0
        upper[:n] = float(
            degree_max
        )

        # ----------------------------------------
        # Previously discovered cut constraints
        # ----------------------------------------

        for r0, S in enumerate(
            cut_sets,
            start=n,
        ):

            for k, e in enumerate(
                candidates
            ):

                i, j = e[0], e[1]

                if (
                    (i in S)
                    !=
                    (j in S)
                ):

                    A[
                        r0,
                        k
                    ] = 1.0

            lower[r0] = 2.0

        res = milp(
            c=c,

            integrality=np.ones(
                m,
                dtype=np.int8,
            ),

            bounds=Bounds(
                np.zeros(m),
                np.ones(m),
            ),

            constraints=
                LinearConstraint(
                    A.tocsr(),
                    lower,
                    upper,
                ),

            options={
                "time_limit": 120.0,
                "mip_rel_gap": 0.0,
            },
        )

        if (
            not res.success
            or
            res.x is None
        ):

            raise RuntimeError(
                "No feasible bridge-free network "
                "under current time/baseline/"
                f"degree constraints.\n"
                f"MILP: {res.message}"
            )

        ids = np.where(
            res.x > 0.5
        )[0]

        selected = [
            candidates[int(k)]
            for k in ids
        ]

        comps = components(
            n,
            selected,
        )

        bridges = bridge_cuts(
            n,
            selected,
        )

        print(
            f"  MILP iteration {iteration:02d}: "
            f"E={len(selected)}, "
            f"components={len(comps)}, "
            f"bridges={len(bridges)}, "
            f"cuts={len(cut_sets)}"
        )

        new_cuts = []

        # ----------------------------------------
        # Disconnected component:
        # crossing edge count = 0
        # ----------------------------------------

        if len(comps) > 1:

            for comp in comps:

                S = normalize_cut(
                    comp,
                    n,
                )

                if (
                    0 < len(S) < n
                    and S not in cut_keys
                ):
                    new_cuts.append(S)

        # ----------------------------------------
        # Bridge:
        # crossing edge count = 1
        # ----------------------------------------

        for _, side in bridges:

            S = normalize_cut(
                side,
                n,
            )

            if (
                0 < len(S) < n
                and S not in cut_keys
            ):
                new_cuts.append(S)

        if (
            len(comps) == 1
            and
            len(bridges) == 0
        ):
            return selected

        if not new_cuts:
            raise RuntimeError(
                "Cut-generation stalled."
            )

        for S in new_cuts:
            if S not in cut_keys:
                cut_keys.add(S)
                cut_sets.append(S)

    raise RuntimeError(
        "Maximum cut-generation iterations reached."
    )


def write_itab(
    path,
    edges,
):
    with path.open(
        "w"
    ) as f:

        for eid, e in enumerate(
            sorted(
                edges,
                key=lambda x:
                    (
                        x[0],
                        x[1],
                    ),
            ),
            start=1,
        ):

            f.write(
                f"{e[0]+1:4d} "
                f"{e[1]+1:4d} "
                f"{eid:4d}  1\n"
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
        _,
    ) = open_from_config(
        args.config
    )

    dates = list(
        stack.dates
    )

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
            160,
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

    figdir = (
        netdir
        / "figures"
    )

    figdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bperp = np.load(
        netdir
        / "acquisition_bperp_m.npy"
    ).astype(
        np.float64
    )

    print("=" * 80)
    print(
        "Step 07b - Degree-constrained "
        "2-edge-connected network"
    )
    print("=" * 80)

    print(
        f"nodes              : {n}"
    )

    print(
        f"Tmax               : {tmax:g} d"
    )

    print(
        f"Bmax               : {bmax:g} m"
    )

    print(
        f"degree range       : 2 .. {kmax}"
    )

    candidates = build_candidates(
        dates,
        bperp,
        tmax,
        bmax,
    )

    dc = degrees(
        n,
        candidates,
    )

    bc = bridge_cuts(
        n,
        candidates,
    )

    cc = components(
        n,
        candidates,
    )

    print()
    print(
        f"candidate edges    : {len(candidates)}"
    )

    print(
        f"candidate comps    : {len(cc)}"
    )

    print(
        f"candidate degree   : "
        f"{dc.min()} / "
        f"{np.median(dc):.1f} / "
        f"{dc.max()}"
    )

    print(
        f"candidate bridges  : {len(bc)}"
    )

    if (
        len(cc) != 1
        or
        dc.min() < 2
        or
        len(bc) != 0
    ):

        raise RuntimeError(
            "Candidate graph itself is not "
            "2-edge-connected. Adjust config first."
        )

    print()
    print(
        "Solving final network..."
    )

    selected = solve_with_cuts(
        n,
        candidates,
        kmax,
    )

    selected = sorted(
        selected,
        key=lambda x:
            (
                x[0],
                x[1],
            ),
    )

    d = degrees(
        n,
        selected,
    )

    comps = components(
        n,
        selected,
    )

    bridges = bridge_cuts(
        n,
        selected,
    )

    E = len(
        selected
    )

    cycle_rank = (
        E - n + 1
    )

    dtv = np.array(
        [
            e[2]
            for e in selected
        ]
    )

    dbv = np.array(
        [
            e[3]
            for e in selected
        ]
    )

    print()
    print("=" * 80)
    print(
        "Final 2-edge-connected network"
    )
    print("=" * 80)

    print(
        f"nodes              : {n}"
    )

    print(
        f"edges              : {E}"
    )

    print(
        f"components         : {len(comps)}"
    )

    print(
        f"bridges            : {len(bridges)}"
    )

    print(
        f"cycle rank         : {cycle_rank}"
    )

    print(
        f"degree min/med/max : "
        f"{d.min()} / "
        f"{np.median(d):.1f} / "
        f"{d.max()}"
    )

    print(
        f"|dT| min/med/max   : "
        f"{dtv.min():.1f} / "
        f"{np.median(dtv):.1f} / "
        f"{dtv.max():.1f} d"
    )

    print(
        f"|dB| min/med/max   : "
        f"{dbv.min():.1f} / "
        f"{np.median(dbv):.1f} / "
        f"{dbv.max():.1f} m"
    )

    if (
        len(comps) != 1
        or
        len(bridges) != 0
        or
        d.min() < 2
        or
        d.max() > kmax
    ):

        raise RuntimeError(
            "Final network failed topology audit."
        )

    # ========================================================
    # Save
    # ========================================================

    old_itab = (
        netdir
        / "network.itab"
    )

    backup_itab = (
        netdir
        / "network_pre_2edge.itab"
    )

    if (
        old_itab.exists()
        and
        not backup_itab.exists()
    ):
        shutil.copy2(
            old_itab,
            backup_itab,
        )

    final_itab = (
        netdir
        / "network_2edge.itab"
    )

    write_itab(
        final_itab,
        selected,
    )

    np.save(
        netdir
        / "degree_2edge.npy",
        d,
    )

    csv_path = (
        netdir
        / "pairs_2edge.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "edge_id",
            "i0",
            "j0",
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
                i,
                j,
                i + 1,
                j + 1,
                dates[i],
                dates[j],
                f"{dt:.6f}",
                f"{db:.6f}",
                f"{score:.8f}",
            ])

    # ========================================================
    # Time-Bperp figure
    # ========================================================

    x = [
        pdate(d0)
        for d0 in dates
    ]

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    for e in candidates:

        i, j = e[0], e[1]

        ax.plot(
            [x[i], x[j]],
            [bperp[i], bperp[j]],
            linewidth=0.4,
            alpha=0.06,
        )

    for e in selected:

        i, j = e[0], e[1]

        ax.plot(
            [x[i], x[j]],
            [bperp[i], bperp[j]],
            linewidth=1.2,
            alpha=0.78,
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
            xytext=(3, 3),
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
        "pyPSDS-GAMMA v0.9 final "
        "2-edge-connected network\n"
        f"|ΔT|≤{tmax:g} d, "
        f"|ΔB⊥|≤{bmax:g} m, "
        f"2≤degree≤{kmax}; "
        f"E={E}, cycle rank={cycle_rank}, "
        "bridges=0"
    )

    ax.grid(
        alpha=0.2
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    png = (
        figdir
        / "07_final_2edge_time_bperp_network.png"
    )

    fig.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Degree figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    xx = np.arange(
        1,
        n + 1,
    )

    ax.bar(
        xx,
        d,
    )

    ax.axhline(
        2,
        linestyle="--",
        linewidth=1,
        label="minimum degree=2",
    )

    ax.axhline(
        kmax,
        linestyle="--",
        linewidth=1,
        label=f"maximum degree={kmax}",
    )

    ax.set_xticks(xx)

    ax.set_xlabel(
        "Acquisition index"
    )

    ax.set_ylabel(
        "Selected connections"
    )

    ax.set_title(
        "Final interferogram network degree"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        figdir
        / "07_final_2edge_degree.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        f"final itab         : {final_itab}"
    )

    print(
        f"pair table         : {csv_path}"
    )

    print(
        f"network plot       : {png}"
    )

    print()
    print(
        "STEP 07b STATUS: PASS"
    )


if __name__ == "__main__":
    main()
