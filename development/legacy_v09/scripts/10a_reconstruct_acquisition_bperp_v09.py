#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


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
                    f"Invalid ITAB line: {raw}"
                )

            edges.append((i, j))

    return edges


def build_A(edges, ndate, ref_idx=0):

    col = {}
    k = 0

    for t in range(ndate):

        if t == ref_idx:
            continue

        col[t] = k
        k += 1

    A = np.zeros(
        (
            len(edges),
            ndate - 1,
        ),
        dtype=np.float64,
    )

    for e, (i, j) in enumerate(edges):

        if i != ref_idx:
            A[e, col[i]] -= 1.0

        if j != ref_idx:
            A[e, col[j]] += 1.0

    return A


class DSU:

    def __init__(self, n):
        self.p = list(range(n))

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

        self.p[b] = a
        return True


def tree_edges(edges, ndate):

    dsu = DSU(ndate)

    out = []

    for e, (i, j) in enumerate(edges):

        if dsu.union(i, j):
            out.append(e)

    if len(out) != ndate - 1:

        raise RuntimeError(
            "Temporal network is not connected"
        )

    return np.asarray(
        out,
        dtype=np.int32,
    )


def read_pairs_csv(
    path: Path,
    edges,
    dates,
):

    with path.open() as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        if not fieldnames:
            raise RuntimeError(
                "pairs.csv has no header"
            )

        rows = list(reader)

    # --------------------------------------------------------
    # Find likely signed perpendicular-baseline field.
    # Do NOT accept explicitly absolute fields.
    # --------------------------------------------------------

    candidates = []

    for name in fieldnames:

        low = name.lower()

        if (
            "bperp" in low
            or
            "perp" in low
            or
            "baseline" in low
        ):

            if (
                "abs" in low
                or
                "absolute" in low
            ):
                continue

            candidates.append(name)

    print(
        "Candidate baseline fields   : "
        + ", ".join(candidates)
    )

    if not candidates:

        raise RuntimeError(
            "Could not identify signed perpendicular "
            "baseline field in pairs.csv"
        )

    # Prefer explicit pair difference names.
    priority_tokens = [
        "signed",
        "db",
        "delta",
        "bperp",
        "perpendicular",
    ]

    def score(name):

        low = name.lower()

        s = 0

        for k, tok in enumerate(
            priority_tokens
        ):

            if tok in low:
                s += (
                    len(priority_tokens)
                    -
                    k
                )

        return s

    candidates.sort(
        key=score,
        reverse=True,
    )

    baseline_field = candidates[0]

    print(
        f"Selected baseline field    : "
        f"{baseline_field}"
    )

    if len(rows) != len(edges):

        raise RuntimeError(
            f"pairs.csv rows={len(rows)}, "
            f"network edges={len(edges)}"
        )

    db = np.empty(
        len(edges),
        dtype=np.float64,
    )

    for e, r in enumerate(rows):

        try:

            db[e] = float(
                r[
                    baseline_field
                ]
            )

        except Exception as exc:

            raise RuntimeError(
                f"Cannot parse baseline on row {e+1}: "
                f"{r.get(baseline_field)!r}"
            ) from exc

    return (
        db,
        baseline_field,
        rows,
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

    root = (
        Path(paths.output_dir)
        / "v09"
    )

    network_dir = (
        root
        / "network"
    )

    outdir = (
        root
        / "scla_v09"
        / "baseline_audit"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ndate = len(
        stack.dates
    )

    edges = load_itab(
        network_dir
        / "network.itab",
        ndate,
    )

    nifg = len(
        edges
    )

    pairs_csv = (
        network_dir
        / "pairs.csv"
    )

    if not pairs_csv.exists():

        raise FileNotFoundError(
            pairs_csv
        )

    (
        db,
        baseline_field,
        pair_rows,
    ) = read_pairs_csv(
        pairs_csv,
        edges,
        stack.dates,
    )

    # ========================================================
    # Basic signed-baseline sanity
    # ========================================================

    print("=" * 108)
    print(
        "Step 10a - Acquisition perpendicular-baseline reconstruction"
    )
    print("=" * 108)

    print(
        f"config                     : "
        f"{config_path}"
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
        f"pair baseline field        : "
        f"{baseline_field}"
    )

    print(
        f"pair Bperp min/med/max     : "
        f"{db.min():.3f} / "
        f"{np.median(db):.3f} / "
        f"{db.max():.3f} m"
    )

    npos = int(
        np.count_nonzero(
            db > 0
        )
    )

    nneg = int(
        np.count_nonzero(
            db < 0
        )
    )

    nzero = int(
        np.count_nonzero(
            db == 0
        )
    )

    print(
        f"signed counts + / - / 0   : "
        f"{npos} / {nneg} / {nzero}"
    )

    # A baseline list containing only positive values is
    # suspicious for this inversion because it may be |db|.
    signed_suspicious = (
        npos > 0
        and
        nneg == 0
    )

    # ========================================================
    # Network inversion
    # ========================================================

    ref_idx = 0

    A = build_A(
        edges,
        ndate,
        ref_idx,
    )

    rank_A = int(
        np.linalg.matrix_rank(
            A
        )
    )

    if rank_A != ndate - 1:

        raise RuntimeError(
            f"rank(A)={rank_A}, expected={ndate-1}"
        )

    tree_ids = tree_edges(
        edges,
        ndate,
    )

    At = A[
        tree_ids,
        :
    ]

    bt = db[
        tree_ids
    ]

    # Exact tree reconstruction.
    x_tree = np.linalg.solve(
        At,
        bt,
    )

    # Full network L2.
    x_l2 = np.linalg.lstsq(
        A,
        db,
        rcond=None,
    )[0]

    b_tree = np.zeros(
        ndate,
        dtype=np.float64,
    )

    b_l2 = np.zeros(
        ndate,
        dtype=np.float64,
    )

    b_tree[
        1:
    ] = x_tree

    b_l2[
        1:
    ] = x_l2

    pred_tree = (
        A
        @
        x_tree
    )

    pred_l2 = (
        A
        @
        x_l2
    )

    res_tree = (
        db
        -
        pred_tree
    )

    res_l2 = (
        db
        -
        pred_l2
    )

    tree_l2_diff = (
        b_tree
        -
        b_l2
    )

    # ========================================================
    # QA
    # ========================================================

    print()
    print("=" * 108)
    print(
        "Acquisition Bperp reconstruction"
    )
    print("=" * 108)

    print(
        f"rank(A)                    : "
        f"{rank_A}"
    )

    print(
        f"reference acquisition      : "
        f"{stack.dates[ref_idx]}"
    )

    print(
        f"acquisition Bperp min/max  : "
        f"{b_l2.min():.3f} / "
        f"{b_l2.max():.3f} m"
    )

    print(
        f"tree-vs-L2 RMS             : "
        f"{np.sqrt(np.mean(tree_l2_diff**2)):.6e} m"
    )

    print(
        f"tree-vs-L2 max             : "
        f"{np.max(np.abs(tree_l2_diff)):.6e} m"
    )

    print()
    print("=" * 108)
    print(
        "Pair-baseline closure residual"
    )
    print("=" * 108)

    print(
        f"L2 residual RMS            : "
        f"{np.sqrt(np.mean(res_l2**2)):.6e} m"
    )

    print(
        f"L2 residual max            : "
        f"{np.max(np.abs(res_l2)):.6e} m"
    )

    print(
        f"tree all-edge residual max : "
        f"{np.max(np.abs(res_tree)):.6e} m"
    )

    worst = np.argsort(
        np.abs(
            res_l2
        )
    )[::-1]

    print()
    print(
        "Largest pair-baseline residuals:"
    )

    print(
        " pair   dates                  "
        "observed[m] predicted[m] residual[m]"
    )

    for e in worst[:15]:

        i, j = edges[e]

        print(
            f" {e+1:4d}   "
            f"{stack.dates[i]}->"
            f"{stack.dates[j]} "
            f"{db[e]:11.5f} "
            f"{pred_l2[e]:12.5f} "
            f"{res_l2[e]:11.6f}"
        )

    # ========================================================
    # Save acquisition baseline
    # ========================================================

    np.save(
        outdir
        / "acquisition_bperp_m.npy",
        b_l2.astype(
            np.float64
        ),
    )

    np.save(
        outdir
        / "acquisition_bperp_tree_m.npy",
        b_tree.astype(
            np.float64
        ),
    )

    np.save(
        outdir
        / "pair_bperp_observed_m.npy",
        db.astype(
            np.float64
        ),
    )

    np.save(
        outdir
        / "pair_bperp_residual_m.npy",
        res_l2.astype(
            np.float64
        ),
    )

    # Acquisition CSV
    acquisition_csv = (
        outdir
        / "acquisition_bperp.csv"
    )

    acquisition_rows = []

    for t in range(
        ndate
    ):

        acquisition_rows.append({
            "acquisition_index":
                t,

            "date":
                str(
                    stack.dates[t]
                ),

            "bperp_l2_m":
                float(
                    b_l2[t]
                ),

            "bperp_tree_m":
                float(
                    b_tree[t]
                ),

            "tree_l2_difference_m":
                float(
                    tree_l2_diff[t]
                ),
        })

    with acquisition_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                acquisition_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            acquisition_rows
        )

    # Pair QA CSV
    pair_csv = (
        outdir
        / "pair_bperp_closure_qa.csv"
    )

    pair_out = []

    for e, (i, j) in enumerate(
        edges
    ):

        pair_out.append({
            "pair_id":
                e + 1,

            "date1":
                str(
                    stack.dates[i]
                ),

            "date2":
                str(
                    stack.dates[j]
                ),

            "observed_bperp_m":
                float(
                    db[e]
                ),

            "predicted_bperp_m":
                float(
                    pred_l2[e]
                ),

            "residual_m":
                float(
                    res_l2[e]
                ),
        })

    with pair_csv.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                pair_out[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            pair_out
        )

    # ========================================================
    # Status
    #
    # Do not impose an unrealistically tiny threshold yet,
    # because GAMMA pair baseline values may be rounded.
    # ========================================================

    rms_res = float(
        np.sqrt(
            np.mean(
                res_l2
                *
                res_l2
            )
        )
    )

    max_res = float(
        np.max(
            np.abs(
                res_l2
            )
        )
    )

    if signed_suspicious:

        status = (
            "REVIEW_UNSIGNED_BASELINE"
        )

    elif max_res > 1.0:

        status = (
            "REVIEW_BASELINE_CLOSURE"
        )

    else:

        status = (
            "PASS"
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-acquisition-bperp-audit-v09",

        "status":
            status,

        "source":
            str(
                pairs_csv
            ),

        "source_field":
            baseline_field,

        "acquisitions":
            int(
                ndate
            ),

        "ifgs":
            int(
                nifg
            ),

        "reference_date":
            str(
                stack.dates[
                    ref_idx
                ]
            ),

        "signed_counts": {
            "positive":
                npos,

            "negative":
                nneg,

            "zero":
                nzero,
        },

        "tree_l2": {
            "rms_m":
                float(
                    np.sqrt(
                        np.mean(
                            tree_l2_diff
                            *
                            tree_l2_diff
                        )
                    )
                ),

            "max_m":
                float(
                    np.max(
                        np.abs(
                            tree_l2_diff
                        )
                    )
                ),
        },

        "pair_closure": {
            "rms_m":
                rms_res,

            "max_m":
                max_res,
        },

        "phase_modified":
            False,

        "note":
            (
                "Acquisition-domain perpendicular baselines "
                "are reconstructed from signed pair baseline "
                "differences. No SCLA correction is applied."
            ),
    }

    manifest_path = (
        outdir
        / "acquisition_bperp_manifest.json"
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
        f"acquisition baseline       : "
        f"{acquisition_csv}"
    )

    print(
        f"pair closure QA            : "
        f"{pair_csv}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10a STATUS: {status} / "
        "BPERP RECONSTRUCTION AUDIT ONLY"
    )

    print(
        "No SCLA or phase correction "
        "has been applied."
    )


if __name__ == "__main__":
    main()
