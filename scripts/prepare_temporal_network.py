#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
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
from pypsds.context import open_from_config


@dataclass(frozen=True, slots=True)
class Edge:
    i: int
    j: int
    dt_days: float
    dbperp_m: float
    score: float


def parse_date(s):
    return datetime.strptime(
        str(s),
        "%Y%m%d",
    )


def find_gamma_command(name):

    p = shutil.which(name)

    if p:
        return p

    for env_name in (
        "GAMMA_HOME",
        "GAMMA_SOFTWARE",
    ):

        root = os.environ.get(
            env_name
        )

        if not root:
            continue

        root = Path(root)

        candidates = [
            root / "DIFF" / "scripts" / name,
            root / "DIFF" / "bin" / name,
            root / "ISP" / "bin" / name,
        ]

        for p in candidates:

            if p.is_file():
                return str(p)

    raise FileNotFoundError(
        f"GAMMA command not found: {name}"
    )


# ============================================================
# GAMMA base_calc
# ============================================================

def run_base_calc(
    stack,
    paths,
    ref_date,
    outdir,
):

    refs = [
        r
        for r in stack.records
        if r.date == ref_date
    ]

    if len(refs) != 1:

        raise RuntimeError(
            f"Baseline reference "
            f"{ref_date} not uniquely "
            f"present in RSLC_tab."
        )

    ref_par = refs[0].par

    base_calc = find_gamma_command(
        "base_calc"
    )

    base_orbit = find_gamma_command(
        "base_orbit"
    )

    gamma_dir = (
        outdir
        / "gamma_base_calc"
    )

    gamma_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bperp_file = (
        gamma_dir
        / "all_pairs.bperp"
    )

    itab_file = (
        gamma_dir
        / "all_pairs.itab"
    )

    log_file = (
        gamma_dir
        / "base_calc_stdout.log"
    )

    # --------------------------------------------------------
    # Build an absolute-path RSLC_tab for GAMMA.
    #
    # base_calc reads filenames in SLC_tab literally.  It does
    # not resolve relative entries against the directory that
    # contains the original SLC_tab.  Since we deliberately run
    # base_calc inside gamma_dir to keep all temporary/log files
    # local, an original entry such as
    #
    #   RSLC/20141006.rslc RSLC/20141006.rslc.par
    #
    # would otherwise be interpreted relative to gamma_dir.
    # --------------------------------------------------------

    src_tab = Path(
        paths.rslc_tab
    ).expanduser().resolve()

    abs_tab = (
        gamma_dir
        / "RSLC_tab.absolute"
    )

    n_tab = 0

    with src_tab.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as fi, abs_tab.open(
        "w",
        encoding="utf-8",
    ) as fo:

        for raw in fi:

            line = raw.strip()

            if (
                not line
                or line.startswith("#")
            ):
                continue

            fields = line.split()

            if len(fields) < 2:
                raise RuntimeError(
                    f"Invalid RSLC_tab line: {raw.rstrip()}"
                )

            slc = Path(
                fields[0]
            ).expanduser()

            par = Path(
                fields[1]
            ).expanduser()

            if not slc.is_absolute():
                slc = (
                    src_tab.parent
                    / slc
                )

            if not par.is_absolute():
                par = (
                    src_tab.parent
                    / par
                )

            slc = slc.resolve()
            par = par.resolve()

            if not slc.exists():
                raise FileNotFoundError(
                    f"RSLC listed in RSLC_tab does not exist: {slc}"
                )

            if not par.exists():
                raise FileNotFoundError(
                    f"RSLC parameter file listed in RSLC_tab "
                    f"does not exist: {par}"
                )

            fo.write(
                f"{slc} {par}\n"
            )

            n_tab += 1

    if n_tab != len(stack.records):
        raise RuntimeError(
            f"Absolute RSLC_tab contains {n_tab} records, "
            f"but stack has {len(stack.records)} acquisitions."
        )

    print(
        f"  absolute RSLC_tab : {abs_tab}"
    )

    print(
        f"  RSLC_tab records  : {n_tab}"
    )

    # Important:
    #
    # Run base_calc with NO baseline filtering.
    # We need all N(N-1)/2 pair baselines first.
    #
    # pyPSDS applies:
    #   max temporal baseline
    #   max perpendicular baseline
    #   max connections/acquisition
    #
    # afterwards.
    #
    # Only plt_flg=0 is specified explicitly.  Remaining
    # base_calc parameters use their defaults.
    cmd = [
        base_calc,
        str(abs_tab),
        str(
            Path(ref_par).resolve()
        ),
        str(bperp_file),
        str(itab_file),
        "1",    # all unique pairs
        "0",    # no GAMMA/Gnuplot plot
    ]

    env = os.environ.copy()

    gamma_bins = [
        str(Path(base_calc).parent),
        str(Path(base_orbit).parent),
    ]

    env["PATH"] = (
        os.pathsep.join(
            gamma_bins
        )
        + os.pathsep
        + env.get(
            "PATH",
            "",
        )
    )

    print()
    print(
        "Running GAMMA base_calc..."
    )

    print(
        "  "
        + " ".join(cmd)
    )

    with log_file.open(
        "w",
        encoding="utf-8",
    ) as log:

        proc = subprocess.run(
            cmd,
            cwd=gamma_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if proc.returncode != 0:

        try:
            log_lines = log_file.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()

            tail = "\\n".join(
                log_lines[-60:]
            )

        except Exception:
            tail = "<unable to read base_calc log>"

        raise RuntimeError(
            "GAMMA base_calc failed "
            f"with exit status {proc.returncode}.\\n"
            f"Command: {' '.join(cmd)}\\n"
            f"Log: {log_file}\\n"
            "---------------- GAMMA log tail ----------------\\n"
            f"{tail}\\n"
            "------------------------------------------------"
        )

    if not bperp_file.is_file():

        raise RuntimeError(
            "base_calc did not create "
            f"{bperp_file}"
        )

    return bperp_file


# ============================================================
# Parse base_calc output
# ============================================================

def parse_base_calc(
    path,
    dates,
):

    date_to_idx = {
        d: i
        for i, d in enumerate(
            dates
        )
    }

    raw_pairs = []

    baseline_samples = {
        d: []
        for d in dates
    }

    lines = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    for raw in lines:

        f = raw.split()

        if len(f) < 5:
            continue

        try:

            d1 = f[1]
            d2 = f[2]

            db = float(
                f[3]
            )

            dt = float(
                f[4]
            )

        except (
            ValueError,
            IndexError,
        ):
            continue

        if (
            d1 not in date_to_idx
            or
            d2 not in date_to_idx
        ):
            continue

        i = date_to_idx[d1]
        j = date_to_idx[d2]

        raw_pairs.append(
            (
                i,
                j,
                dt,
                db,
            )
        )

        # Current GAMMA base_calc all-pairs
        # output contains:
        #
        # edge
        # date1 date2
        # pair_Bperp pair_deltaT
        # MJD1 MJD2
        # Bperp1 Bperp2
        #
        if len(f) >= 9:

            try:

                baseline_samples[
                    d1
                ].append(
                    float(f[7])
                )

                baseline_samples[
                    d2
                ].append(
                    float(f[8])
                )

            except ValueError:
                pass

    n = len(dates)

    expected = (
        n * (n - 1) // 2
    )

    if len(raw_pairs) != expected:

        raise RuntimeError(
            f"Expected {expected} "
            f"all-pair baselines, "
            f"parsed {len(raw_pairs)}."
        )

    # --------------------------------------------------------
    # Per-acquisition Bperp coordinate
    # --------------------------------------------------------

    bperp = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    for i, d in enumerate(
        dates
    ):

        if baseline_samples[d]:

            bperp[i] = np.median(
                baseline_samples[d]
            )

    # Fallback for old GAMMA versions:
    # reconstruct B_i from pair differences.
    if not np.all(
        np.isfinite(bperp)
    ):

        A = np.zeros(
            (
                len(raw_pairs) + 1,
                n,
            ),
            dtype=np.float64,
        )

        y = np.zeros(
            len(raw_pairs) + 1,
            dtype=np.float64,
        )

        for r, (
            i,
            j,
            _dt,
            db,
        ) in enumerate(
            raw_pairs
        ):

            A[r, i] = -1.0
            A[r, j] = +1.0
            y[r] = db

        # Arbitrary Bperp origin.
        A[-1, 0] = 1.0

        bperp = np.linalg.lstsq(
            A,
            y,
            rcond=None,
        )[0]

    # --------------------------------------------------------
    # Recompute pair metrics consistently
    # --------------------------------------------------------

    pair_map = {}

    for i in range(
        n - 1
    ):

        for j in range(
            i + 1,
            n
        ):

            dt = abs(
                (
                    parse_date(
                        dates[j]
                    )
                    -
                    parse_date(
                        dates[i]
                    )
                ).total_seconds()
                / 86400.0
            )

            db = abs(
                float(
                    bperp[j]
                    - bperp[i]
                )
            )

            pair_map[
                (i, j)
            ] = (
                dt,
                db,
            )

    return (
        bperp,
        pair_map,
    )


# ============================================================
# Basic graph functions
# ============================================================

def components(
    n,
    edges,
):

    adj = [
        []
        for _ in range(n)
    ]

    for e in edges:

        adj[e.i].append(
            e.j
        )

        adj[e.j].append(
            e.i
        )

    seen = [
        False
    ] * n

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

        out.append(
            sorted(comp)
        )

    return out


def degree_vector(
    n,
    edges,
):

    d = np.zeros(
        n,
        dtype=np.int32,
    )

    for e in edges:

        d[e.i] += 1
        d[e.j] += 1

    return d


# ============================================================
# Degree-constrained connected backbone
# ============================================================

def solve_spanning_tree(
    n,
    edges,
    degree_cap,
):

    """
    Minimum-spatiotemporal-distance spanning tree
    under a STRICT maximum node degree.

    MILP:
      x_e in {0,1}

    Constraints:
      sum x_e = N-1
      degree_i <= degree_cap
      single-commodity flow guarantees connectivity
    """

    m = len(edges)

    # x_e + forward flow + backward flow
    nvar = 3 * m

    c = np.zeros(
        nvar,
        dtype=np.float64,
    )

    c[:m] = np.array(
        [
            e.score
            for e in edges
        ],
        dtype=np.float64,
    )

    integrality = np.zeros(
        nvar,
        dtype=np.int8,
    )

    integrality[:m] = 1

    lb_var = np.zeros(
        nvar,
        dtype=np.float64,
    )

    ub_var = np.full(
        nvar,
        float(n - 1),
        dtype=np.float64,
    )

    ub_var[:m] = 1.0

    nrow = (
        1
        + n
        + n
        + 2 * m
    )

    A = lil_matrix(
        (
            nrow,
            nvar,
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

    r = 0

    # --------------------------------------------------------
    # Exactly N-1 edges
    # --------------------------------------------------------

    A[
        r,
        :m
    ] = 1.0

    lower[r] = n - 1
    upper[r] = n - 1

    r += 1

    # --------------------------------------------------------
    # Degree cap
    # --------------------------------------------------------

    for u in range(n):

        for k, e in enumerate(
            edges
        ):

            if (
                e.i == u
                or
                e.j == u
            ):

                A[
                    r,
                    k
                ] = 1.0

        lower[r] = 0.0
        upper[r] = float(
            degree_cap
        )

        r += 1

    # --------------------------------------------------------
    # Flow connectivity
    # --------------------------------------------------------

    for u in range(n):

        for k, e in enumerate(
            edges
        ):

            fij = m + k
            fji = (
                2 * m + k
            )

            if e.i == u:

                A[
                    r,
                    fij
                ] += 1.0

                A[
                    r,
                    fji
                ] -= 1.0

            elif e.j == u:

                A[
                    r,
                    fji
                ] += 1.0

                A[
                    r,
                    fij
                ] -= 1.0

        target = (
            float(n - 1)
            if u == 0
            else -1.0
        )

        lower[r] = target
        upper[r] = target

        r += 1

    # --------------------------------------------------------
    # Flow only through selected edges
    # --------------------------------------------------------

    M = float(
        n - 1
    )

    for k in range(m):

        # i -> j
        A[
            r,
            m + k
        ] = 1.0

        A[
            r,
            k
        ] = -M

        upper[r] = 0.0

        r += 1

        # j -> i
        A[
            r,
            2 * m + k
        ] = 1.0

        A[
            r,
            k
        ] = -M

        upper[r] = 0.0

        r += 1

    result = milp(
        c=c,

        integrality=
            integrality,

        bounds=Bounds(
            lb_var,
            ub_var,
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
        not result.success
        or
        result.x is None
    ):

        raise RuntimeError(
            "No connected network "
            "satisfies the current "
            "baseline constraints and "
            f"degree cap={degree_cap}.\n"
            f"MILP: {result.message}"
        )

    ids = np.where(
        result.x[:m] > 0.5
    )[0]

    tree = [
        edges[int(k)]
        for k in ids
    ]

    if len(tree) != n - 1:

        raise RuntimeError(
            "Internal spanning-tree "
            "edge-count error."
        )

    return tree


# ============================================================
# Add redundant network edges
# ============================================================

def add_redundancy(
    n,
    candidates,
    tree,
    degree_cap,
):

    """
    Keep the connected tree fixed.

    Then maximize the number of additional edges
    subject to the same node-degree cap.

    A very small score term prefers shorter
    spatiotemporal edges among equal-cardinality solutions.
    """

    tree_keys = {
        (
            e.i,
            e.j,
        )
        for e in tree
    }

    remaining = [
        e
        for e in candidates
        if (
            e.i,
            e.j,
        ) not in tree_keys
    ]

    if not remaining:

        return list(tree)

    d0 = degree_vector(
        n,
        tree,
    )

    capacity = (
        degree_cap
        - d0
    )

    m = len(
        remaining
    )

    score = np.array(
        [
            e.score
            for e in remaining
        ],
        dtype=np.float64,
    )

    # Primary objective:
    # maximize edge count.
    #
    # Secondary:
    # prefer smaller normalized baseline.
    c = (
        -np.ones(
            m,
            dtype=np.float64,
        )
        + 1.0e-4 * score
    )

    A = lil_matrix(
        (
            n,
            m,
        ),
        dtype=np.float64,
    )

    for k, e in enumerate(
        remaining
    ):

        A[
            e.i,
            k
        ] = 1.0

        A[
            e.j,
            k
        ] = 1.0

    result = milp(
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
                np.zeros(n),
                capacity.astype(
                    float
                ),
            ),

        options={
            "time_limit": 60.0,
            "mip_rel_gap": 0.0,
        },
    )

    if (
        not result.success
        or
        result.x is None
    ):

        raise RuntimeError(
            "Redundancy selection failed: "
            f"{result.message}"
        )

    extra = [
        remaining[int(k)]
        for k in np.where(
            result.x > 0.5
        )[0]
    ]

    return (
        list(tree)
        + extra
    )


# ============================================================
# Bridge quality
# ============================================================

def find_bridges(
    n,
    edges,
):

    adj = [
        []
        for _ in range(n)
    ]

    for eid, e in enumerate(
        edges
    ):

        adj[e.i].append(
            (
                e.j,
                eid,
            )
        )

        adj[e.j].append(
            (
                e.i,
                eid,
            )
        )

    tin = [-1] * n
    low = [-1] * n

    timer = 0

    out = set()

    def dfs(
        u,
        parent_eid,
    ):

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

                dfs(
                    v,
                    eid,
                )

                low[u] = min(
                    low[u],
                    low[v],
                )

                if (
                    low[v]
                    > tin[u]
                ):

                    out.add(
                        tuple(
                            sorted(
                                (
                                    u,
                                    v,
                                )
                            )
                        )
                    )

    for u in range(n):

        if tin[u] < 0:

            dfs(
                u,
                -1,
            )

    return out


def write_itab(
    path,
    edges,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for k, e in enumerate(
            sorted(
                edges,
                key=lambda x:
                    (
                        x.i,
                        x.j,
                    ),
            ),
            start=1,
        ):

            f.write(
                f"{e.i+1:4d} "
                f"{e.j+1:4d} "
                f"{k:4d}  1\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--force-base-calc",
        action="store_true",
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        _roi,
    ) = open_from_config(
        args.config
    )

    dates = list(
        stack.dates
    )

    n = len(dates)

    dt_max = float(
        cfg_get(
            cfg,
            "network.max_temporal_baseline_days",
            72.0,
        )
    )

    bp_max = float(
        cfg_get(
            cfg,
            "network.max_perpendicular_baseline_m",
            150.0,
        )
    )

    degree_cap = int(
        cfg_get(
            cfg,
            "network.max_connections_per_acquisition",
            4,
        )
    )

    if (
        dt_max <= 0
        or
        bp_max <= 0
    ):
        raise ValueError(
            "Baseline thresholds "
            "must be positive."
        )

    if degree_cap < 2:
        raise ValueError(
            "max_connections_per_acquisition "
            "must be >= 2."
        )

    ref_date = str(
        cfg_get(
            cfg,
            "phase_correction.geometric_reference_date",
            dates[0],
        )
    )

    outdir = (
        Path(
            paths.output_dir
        )
        / "processing"
        / "network"
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
        "Spatiotemporal interferogram network"
    )
    print("=" * 80)

    print(
        f"config                       : "
        f"{config_path}"
    )

    print(
        f"acquisitions                 : "
        f"{n}"
    )

    print(
        f"baseline reference           : "
        f"{ref_date}"
    )

    print(
        f"max temporal baseline        : "
        f"{dt_max:.3f} days"
    )

    print(
        f"max perpendicular baseline   : "
        f"{bp_max:.3f} m"
    )

    print(
        f"max connections/acquisition  : "
        f"{degree_cap}"
    )

    # ========================================================
    # 1. GAMMA baselines
    # ========================================================

    bperp_path = (
        outdir
        / "gamma_base_calc"
        / "all_pairs.bperp"
    )

    if (
        args.force_base_calc
        or
        not bperp_path.exists()
    ):

        bperp_path = run_base_calc(
            stack,
            paths,
            ref_date,
            outdir,
        )

    else:

        print(
            f"Reusing base_calc output     : "
            f"{bperp_path}"
        )

    (
        bperp,
        pair_map,
    ) = parse_base_calc(
        bperp_path,
        dates,
    )

    # ========================================================
    # 2. Candidate edges
    # ========================================================

    all_edges = []

    candidates = []

    for i in range(
        n - 1
    ):

        for j in range(
            i + 1,
            n
        ):

            dt, db = pair_map[
                (i, j)
            ]

            score = math.hypot(
                dt / dt_max,
                db / bp_max,
            )

            e = Edge(
                i=i,
                j=j,
                dt_days=dt,
                dbperp_m=db,
                score=score,
            )

            all_edges.append(e)

            if (
                dt
                <= dt_max + 1e-9
                and
                db
                <= bp_max + 1e-9
            ):

                candidates.append(e)

    comps = components(
        n,
        candidates,
    )

    print()
    print(
        f"all possible pairs            : "
        f"{len(all_edges)}"
    )

    print(
        f"threshold-admissible pairs    : "
        f"{len(candidates)}"
    )

    print(
        f"candidate components          : "
        f"{len(comps)}"
    )

    if len(comps) != 1:

        print()
        print(
            "Disconnected components:"
        )

        for c in comps:

            print(
                "  "
                + ", ".join(
                    f"{i+1}:{dates[i]}"
                    for i in c
                )
            )

        raise RuntimeError(
            "Temporal/spatial baseline "
            "thresholds disconnect the graph. "
            "Increase one or both thresholds."
        )

    # ========================================================
    # 3. Connected degree-capped network
    # ========================================================

    print()
    print(
        "Solving degree-capped "
        "connected backbone..."
    )

    tree = solve_spanning_tree(
        n,
        candidates,
        degree_cap,
    )

    print(
        "Adding redundant edges..."
    )

    selected = add_redundancy(
        n,
        candidates,
        tree,
        degree_cap,
    )

    selected = sorted(
        selected,
        key=lambda e:
            (
                e.i,
                e.j,
            ),
    )

    # ========================================================
    # 4. Quality
    # ========================================================

    deg = degree_vector(
        n,
        selected,
    )

    bridges = find_bridges(
        n,
        selected,
    )

    cycle_rank = (
        len(selected)
        - n
        + 1
    )

    print()
    print("=" * 80)
    print(
        "Network summary"
    )
    print("=" * 80)

    print(
        f"nodes                         : "
        f"{n}"
    )

    print(
        f"selected edges                : "
        f"{len(selected)}"
    )

    print(
        f"cycle rank E-V+1              : "
        f"{cycle_rank}"
    )

    print(
        f"bridge edges                  : "
        f"{len(bridges)}"
    )

    print(
        f"degree min/median/max         : "
        f"{deg.min()} / "
        f"{np.median(deg):.1f} / "
        f"{deg.max()}"
    )

    dt_selected = np.array(
        [
            e.dt_days
            for e in selected
        ]
    )

    bp_selected = np.array(
        [
            e.dbperp_m
            for e in selected
        ]
    )

    print(
        f"|dT| min/median/max days      : "
        f"{dt_selected.min():.1f} / "
        f"{np.median(dt_selected):.1f} / "
        f"{dt_selected.max():.1f}"
    )

    print(
        f"|dBperp| min/median/max m     : "
        f"{bp_selected.min():.1f} / "
        f"{np.median(bp_selected):.1f} / "
        f"{bp_selected.max():.1f}"
    )

    # ========================================================
    # 5. Save network
    # ========================================================

    np.save(
        outdir
        / "acquisition_bperp_m.npy",
        bperp.astype(
            np.float32
        ),
    )

    np.save(
        outdir
        / "degree.npy",
        deg,
    )

    write_itab(
        outdir
        / "network.itab",
        selected,
    )

    # --------------------------------------------------------
    # acquisitions.csv
    # --------------------------------------------------------

    with (
        outdir
        / "acquisitions.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "index0",
            "index1",
            "date",
            "bperp_m",
            "degree",
        ])

        for i, d in enumerate(
            dates
        ):

            w.writerow([
                i,
                i + 1,
                d,
                f"{bperp[i]:.6f}",
                int(deg[i]),
            ])

    # --------------------------------------------------------
    # pairs.csv
    # --------------------------------------------------------

    tree_keys = {
        (
            e.i,
            e.j,
        )
        for e in tree
    }

    with (
        outdir
        / "pairs.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
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
            "tree_edge",
            "bridge_edge",
        ])

        for eid, e in enumerate(
            selected,
            start=1,
        ):

            key = (
                e.i,
                e.j,
            )

            w.writerow([
                eid,
                e.i,
                e.j,
                e.i + 1,
                e.j + 1,
                dates[e.i],
                dates[e.j],
                f"{e.dt_days:.6f}",
                f"{e.dbperp_m:.6f}",
                f"{e.score:.8f}",
                int(
                    key
                    in tree_keys
                ),
                int(
                    key
                    in bridges
                ),
            ])

    # ========================================================
    # 6. Classic time-Bperp network figure
    # ========================================================

    xdates = [
        parse_date(d)
        for d in dates
    ]

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    # Threshold-admissible candidate network
    for e in candidates:

        ax.plot(
            [
                xdates[e.i],
                xdates[e.j],
            ],
            [
                bperp[e.i],
                bperp[e.j],
            ],
            linewidth=0.45,
            alpha=0.08,
        )

    # Final selected network
    for e in selected:

        ax.plot(
            [
                xdates[e.i],
                xdates[e.j],
            ],
            [
                bperp[e.i],
                bperp[e.j],
            ],
            linewidth=1.2,
            alpha=0.75,
        )

    ax.scatter(
        xdates,
        bperp,
        s=28,
        zorder=3,
    )

    for i, (
        x,
        y,
    ) in enumerate(
        zip(
            xdates,
            bperp,
        )
    ):

        ax.annotate(
            str(i + 1),
            (
                x,
                y,
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.set_xlabel(
        "Acquisition date"
    )

    ax.set_ylabel(
        "Perpendicular baseline "
        f"relative to {ref_date} (m)"
    )

    ax.set_title(
        "pyPSDS-GAMMA v1.0 "
        "spatiotemporal baseline network\n"
        f"|ΔT| ≤ {dt_max:g} d, "
        f"|ΔB⊥| ≤ {bp_max:g} m, "
        f"degree ≤ {degree_cap}; "
        f"E={len(selected)}, "
        f"cycle rank={cycle_rank}"
    )

    ax.grid(
        alpha=0.2
    )

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        figdir
        / "07_time_perp_baseline_network.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # 7. Pair baseline plane
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.scatter(
        [
            e.dt_days
            for e in all_edges
        ],
        [
            e.dbperp_m
            for e in all_edges
        ],
        s=9,
        alpha=0.12,
        label="All pairs",
    )

    ax.scatter(
        [
            e.dt_days
            for e in candidates
        ],
        [
            e.dbperp_m
            for e in candidates
        ],
        s=13,
        alpha=0.30,
        label="Within hard thresholds",
    )

    ax.scatter(
        [
            e.dt_days
            for e in selected
        ],
        [
            e.dbperp_m
            for e in selected
        ],
        s=28,
        alpha=0.90,
        label="Selected network",
    )

    ax.axvline(
        dt_max,
        linewidth=1.0,
        linestyle="--",
    )

    ax.axhline(
        bp_max,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "|Temporal baseline| (days)"
    )

    ax.set_ylabel(
        "|Perpendicular baseline| (m)"
    )

    ax.set_title(
        "Pairwise temporal-spatial baseline plane"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figdir
        / "07_pair_baseline_plane.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # 8. Connection count
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.bar(
        np.arange(
            1,
            n + 1,
        ),
        deg,
    )

    ax.axhline(
        degree_cap,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Acquisition index"
    )

    ax.set_ylabel(
        "Selected connection count"
    )

    ax.set_title(
        "Per-acquisition network degree"
    )

    ax.set_xticks(
        np.arange(
            1,
            n + 1,
        )
    )

    ax.tick_params(
        axis="x",
        labelsize=7,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        figdir
        / "07_connection_count.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    manifest = {
        "version": "0.9",
        "baseline_backend":
            "GAMMA base_calc/base_orbit",
        "baseline_reference_date":
            ref_date,

        "max_temporal_baseline_days":
            dt_max,

        "max_perpendicular_baseline_m":
            bp_max,

        "max_connections_per_acquisition":
            degree_cap,

        "n_acquisitions":
            n,

        "n_all_pairs":
            len(all_edges),

        "n_threshold_candidates":
            len(candidates),

        "n_selected_edges":
            len(selected),

        "cycle_rank":
            cycle_rank,

        "n_bridges":
            len(bridges),

        "degree_min":
            int(deg.min()),

        "degree_median":
            float(
                np.median(deg)
            ),

        "degree_max":
            int(deg.max()),
    }

    (
        outdir
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"network itab                  : "
        f"{outdir/'network.itab'}"
    )

    print(
        f"pair table                    : "
        f"{outdir/'pairs.csv'}"
    )

    print(
        f"time-Bperp plot               : "
        f"{figdir/'07_time_perp_baseline_network.png'}"
    )

    print(
        f"pair baseline plane           : "
        f"{figdir/'07_pair_baseline_plane.png'}"
    )

    print(
        f"connection-count plot         : "
        f"{figdir/'07_connection_count.png'}"
    )

    print()
    print(
        "STEP 07 STATUS: PASS"
    )


if __name__ == "__main__":
    main()
