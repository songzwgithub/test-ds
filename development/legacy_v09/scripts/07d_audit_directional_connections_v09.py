#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


def pdate(s):
    return datetime.strptime(str(s), "%Y%m%d")


def counts_for_threshold(
    dates,
    bperp,
    tmax,
    bmax,
):
    n = len(dates)

    left = np.zeros(n, dtype=np.int32)
    right = np.zeros(n, dtype=np.int32)

    edges = []

    for i in range(n - 1):
        for j in range(i + 1, n):

            dt = abs(
                (pdate(dates[j]) - pdate(dates[i])).days
            )

            db = abs(
                float(bperp[j] - bperp[i])
            )

            if dt <= tmax and db <= bmax:
                edges.append((i, j, dt, db))
                right[i] += 1
                left[j] += 1

    return left, right, edges


def required_counts(n, k):
    req_left = np.zeros(n, dtype=np.int32)
    req_right = np.zeros(n, dtype=np.int32)

    for i in range(n):
        req_left[i] = min(k, i)
        req_right[i] = min(k, n - 1 - i)

    return req_left, req_right


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--min-each-side",
        type=int,
        default=3,
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

    netdir = (
        Path(paths.output_dir)
        / "v09"
        / "network"
    )

    bperp = np.load(
        netdir / "acquisition_bperp_m.npy"
    ).astype(np.float64)

    k = args.min_each_side

    req_left, req_right = required_counts(
        n,
        k,
    )

    print("=" * 100)
    print(
        "Step 07d - Directional connection feasibility audit"
    )
    print("=" * 100)

    print(f"nodes                 : {n}")
    print(f"minimum each side     : {k}")

    # =========================================================
    # Current parameters
    # =========================================================

    t0 = 72
    b0 = 160

    left, right, edges = counts_for_threshold(
        dates,
        bperp,
        t0,
        b0,
    )

    print()
    print("=" * 100)
    print(
        f"Current candidate graph: Tmax={t0} d, Bmax={b0} m"
    )
    print("=" * 100)

    print(
        " idx date      Bperp      "
        "left(req) right(req) status"
    )

    bad = []

    for i in range(n):

        ok = (
            left[i] >= req_left[i]
            and
            right[i] >= req_right[i]
        )

        if not ok:
            bad.append(i)

        print(
            f"{i+1:3d} "
            f"{dates[i]} "
            f"{bperp[i]:9.2f}   "
            f"{left[i]:2d}({req_left[i]:1d})     "
            f"{right[i]:2d}({req_right[i]:1d})   "
            f"{'OK' if ok else 'FAIL'}"
        )

    print()
    print(
        f"candidate edges       : {len(edges)}"
    )

    print(
        f"failed acquisitions   : {len(bad)}"
    )

    if bad:
        print(
            "failed indices        : "
            + ", ".join(
                f"{i+1}:{dates[i]}"
                for i in bad
            )
        )

    # =========================================================
    # Threshold sweep
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Searching minimum feasible Tmax / Bmax"
    )
    print("=" * 100)

    t_values = list(
        range(
            72,
            181,
            12,
        )
    )

    b_values = list(
        range(
            150,
            401,
            10,
        )
    )

    feasible = []

    print(
        " Tmax Bmax | edges "
        "minL/req minR/req failed"
    )

    for tmax in t_values:

        for bmax in b_values:

            l, r, ee = counts_for_threshold(
                dates,
                bperp,
                tmax,
                bmax,
            )

            ok = (
                (l >= req_left)
                &
                (r >= req_right)
            )

            nf = int(
                np.sum(~ok)
            )

            if nf == 0:

                feasible.append(
                    (
                        tmax,
                        bmax,
                        len(ee),
                    )
                )

                print(
                    f"{tmax:5d} {bmax:4d} | "
                    f"{len(ee):5d} "
                    f"      FEASIBLE"
                )

    print()
    print("=" * 100)
    print("Minimum feasible candidates")
    print("=" * 100)

    if not feasible:

        print(
            "No feasible solution in tested range."
        )

        return

    # Pareto-like ordering:
    # first smallest Tmax,
    # then smallest Bmax.
    feasible.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    for x in feasible[:15]:
        print(
            f"Tmax={x[0]:3d} d  "
            f"Bmax={x[1]:3d} m  "
            f"candidate_edges={x[2]}"
        )

    # Also show minimum normalized relaxation.
    best = min(
        feasible,
        key=lambda x:
            (
                x[0] / 72.0
                +
                x[1] / 160.0
            )
    )

    print()
    print(
        "Smallest normalized relaxation:"
    )

    print(
        f"  Tmax = {best[0]} d"
    )

    print(
        f"  Bmax = {best[1]} m"
    )

    print(
        f"  candidate edges = {best[2]}"
    )

    print()
    print(
        "STEP 07d STATUS: PASS"
    )


if __name__ == "__main__":
    main()
