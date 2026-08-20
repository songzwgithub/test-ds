#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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
                    f"Invalid ITAB line: {raw}"
                )

            edges.append(
                (i, j)
            )

    return edges


def robust_limits(x, mask=None):

    if mask is not None:
        v = x[
            mask
        ]
    else:
        v = x.ravel()

    v = v[
        np.isfinite(v)
    ]

    if v.size == 0:
        return -1.0, 1.0

    lo, hi = np.percentile(
        v,
        [1.0, 99.0],
    )

    if not np.isfinite(lo):
        lo = float(
            np.nanmin(v)
        )

    if not np.isfinite(hi):
        hi = float(
            np.nanmax(v)
        )

    if hi <= lo:

        c = float(
            np.nanmedian(v)
        )

        lo = c - 1.0
        hi = c + 1.0

    return (
        float(lo),
        float(hi),
    )


def point_to_grid(
    values,
    rows,
    cols,
    H,
    W,
    fill=np.nan,
    dtype=np.float32,
):

    out = np.full(
        (H, W),
        fill,
        dtype=dtype,
    )

    out[
        rows,
        cols
    ] = values

    return out


def read_batch_qa(path: Path):

    if not path.exists():
        return {}

    out = {}

    with path.open() as f:

        for r in csv.DictReader(f):

            try:
                pid = int(
                    r["pair_id"]
                )
            except Exception:
                continue

            out[
                pid
            ] = r

    return out


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--height",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--width",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--dpi",
        type=int,
        default=150,
    )

    ap.add_argument(
        "--only",
        type=int,
        nargs="*",
        default=None,
        help="Optional pair IDs, e.g. --only 19 24 32 108",
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

    network_dir = (
        root
        / "network"
    )

    unwrap_dir = (
        root
        / "single_ifg_robust_solution"
    )

    final_dir = (
        root
        / "final_unwrap_v09"
    )

    batch_qa_path = (
        root
        / "batch_unwrap_validation"
        / "all_ifg_unwrap_qa.csv"
    )

    outdir = (
        root
        / "ifg_visual_qa"
    )

    panel_dir = (
        outdir
        / "panels"
    )

    wrapped_dir = (
        outdir
        / "wrapped"
    )

    unwrapped_dir = (
        outdir
        / "unwrapped"
    )

    ambiguity_dir = (
        outdir
        / "ambiguity"
    )

    for d in (
        outdir,
        panel_dir,
        wrapped_dir,
        unwrapped_dir,
        ambiguity_dir,
    ):
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Point stack
    # ========================================================

    phase = np.load(
        pps_dir
        / "phase_rad.npy",
        mmap_mode="r",
    )

    rows = np.asarray(
        np.load(
            pps_dir
            / "rows.npy",
            mmap_mode="r",
        ),
        dtype=np.int32,
    )

    cols = np.asarray(
        np.load(
            pps_dir
            / "cols.npy",
            mmap_mode="r",
        ),
        dtype=np.int32,
    )

    npoint, ndate = phase.shape

    H = (
        args.height
        if args.height is not None
        else int(
            rows.max()
        ) + 1
    )

    W = (
        args.width
        if args.width is not None
        else int(
            cols.max()
        ) + 1
    )

    if (
        rows.max() >= H
        or
        cols.max() >= W
    ):

        raise RuntimeError(
            f"Grid too small: "
            f"H={H}, W={W}, "
            f"max row={rows.max()}, "
            f"max col={cols.max()}"
        )

    temporal_edges = load_itab(
        network_dir
        / "network.itab",
        ndate,
    )

    nedge = len(
        temporal_edges
    )

    global_delta = np.load(
        final_dir
        / "global_ifg_integer_delta.npy",
    ).astype(
        np.int32,
        copy=False,
    )

    strict_valid = np.load(
        final_dir
        / "strict_unwrap_valid_mask.npy",
    ).astype(
        bool,
        copy=False,
    )

    temporal_bad = np.load(
        final_dir
        / "temporal_integer_bad_mask.npy",
    ).astype(
        bool,
        copy=False,
    )

    safe_conflict = np.load(
        final_dir
        / "spatial_safe_conflict_point_mask.npy",
    ).astype(
        bool,
        copy=False,
    )

    all_registered = np.load(
        final_dir
        / "all_ifg_registered_mask.npy",
    ).astype(
        bool,
        copy=False,
    )

    batch_qa = read_batch_qa(
        batch_qa_path
    )

    # ========================================================
    # Fixed QA grid
    #
    # 0 = strict valid
    # 1 = temporal bad
    # 2 = SAFE conflict
    # 3 = not registered in all IFGs
    #
    # precedence:
    # not registered > temporal bad > safe conflict > strict
    # ========================================================

    qa_code = np.zeros(
        npoint,
        dtype=np.uint8,
    )

    qa_code[
        safe_conflict
    ] = 2

    qa_code[
        temporal_bad
    ] = 1

    qa_code[
        ~all_registered
    ] = 3

    qa_grid = point_to_grid(
        qa_code,
        rows,
        cols,
        H,
        W,
        fill=255,
        dtype=np.uint8,
    )

    # transparent / valid / temporal / safe / unregistered
    qa_cmap = ListedColormap(
        [
            "#d9d9d9",
            "#d73027",
            "#fdae61",
            "#542788",
        ]
    )

    qa_norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5],
        qa_cmap.N,
    )

    # ========================================================
    # Pair selection
    # ========================================================

    if args.only:

        selected = {
            int(x)
            for x in args.only
        }

    else:

        selected = set(
            range(
                1,
                nedge + 1,
            )
        )

    summary = []

    print("=" * 104)
    print(
        "Step 08y - Wrapped / unwrapped IFG visual QA"
    )
    print("=" * 104)

    print(
        f"config                 : {config_path}"
    )

    print(
        f"points                 : {npoint:,}"
    )

    print(
        f"dates                  : {ndate}"
    )

    print(
        f"IFGs                   : {nedge}"
    )

    print(
        f"display grid           : {H} x {W}"
    )

    print(
        f"strict valid           : "
        f"{np.count_nonzero(strict_valid):,}"
    )

    print()

    # ========================================================
    # Render
    # ========================================================

    for pair_id, (i, j) in enumerate(
        temporal_edges,
        start=1,
    ):

        if pair_id not in selected:
            continue

        d1 = str(
            stack.dates[i]
        )

        d2 = str(
            stack.dates[j]
        )

        tag = (
            f"pair{pair_id:03d}_"
            f"{d1}_{d2}"
        )

        unw_path = (
            unwrap_dir
            / f"{tag}_unwrapped_phase_rad.npy"
        )

        if not unw_path.exists():

            raise FileNotFoundError(
                unw_path
            )

        U_raw = np.asarray(
            np.load(
                unw_path,
                mmap_mode="r",
            ),
            dtype=np.float64,
        )

        # -----------------------------------------------
        # Wrapped virtual IFG from canonical
        # acquisition phase stack.
        # -----------------------------------------------

        wrapped = wrap(
            np.asarray(
                phase[:, j],
                dtype=np.float64,
            )
            -
            np.asarray(
                phase[:, i],
                dtype=np.float64,
            )
        )

        # -----------------------------------------------
        # Lazy global integer gauge from Step08x.
        # -----------------------------------------------

        gauge = int(
            global_delta[
                pair_id - 1
            ]
        )

        U = (
            U_raw
            +
            TWOPI * gauge
        )

        # -----------------------------------------------
        # Integer ambiguity map.
        #
        # IMPORTANT:
        # compute BEFORE any optional display-only
        # continuous spatial re-referencing.
        # -----------------------------------------------

        integer_n = np.rint(
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

        # -----------------------------------------------
        # Point -> radar grid.
        # No interpolation.
        # -----------------------------------------------

        wrapped_grid = point_to_grid(
            wrapped.astype(
                np.float32
            ),
            rows,
            cols,
            H,
            W,
        )

        unw_grid = point_to_grid(
            U.astype(
                np.float32
            ),
            rows,
            cols,
            H,
            W,
        )

        n_grid = point_to_grid(
            integer_n.astype(
                np.float32
            ),
            rows,
            cols,
            H,
            W,
        )

        # Robust display limits only.
        uvmin, uvmax = robust_limits(
            U,
            strict_valid,
        )

        n_valid = integer_n[
            strict_valid
        ]

        if n_valid.size:

            nmax = int(
                np.max(
                    np.abs(
                        n_valid
                    )
                )
            )

        else:

            nmax = 1

        nmax = max(
            1,
            nmax,
        )

        # ====================================================
        # 4-panel QA figure
        # ====================================================

        fig, ax = plt.subplots(
            2,
            2,
            figsize=(15, 7.8),
            constrained_layout=True,
        )

        im0 = ax[0, 0].imshow(
            wrapped_grid,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="none",
            aspect="auto",
        )

        ax[0, 0].set_title(
            "Wrapped virtual IFG [rad]"
        )

        fig.colorbar(
            im0,
            ax=ax[0, 0],
            fraction=0.025,
            pad=0.02,
        )

        im1 = ax[0, 1].imshow(
            unw_grid,
            cmap="turbo",
            vmin=uvmin,
            vmax=uvmax,
            interpolation="none",
            aspect="auto",
        )

        ax[0, 1].set_title(
            "Unwrapped IFG + global integer gauge [rad]"
        )

        fig.colorbar(
            im1,
            ax=ax[0, 1],
            fraction=0.025,
            pad=0.02,
        )

        im2 = ax[1, 0].imshow(
            n_grid,
            cmap="RdBu_r",
            vmin=-nmax,
            vmax=nmax,
            interpolation="none",
            aspect="auto",
        )

        ax[1, 0].set_title(
            "Integer ambiguity N = round((U - wrapped)/2π)"
        )

        cbar2 = fig.colorbar(
            im2,
            ax=ax[1, 0],
            fraction=0.025,
            pad=0.02,
        )

        cbar2.set_label(
            "integer cycles"
        )

        qshow = np.ma.masked_where(
            qa_grid == 255,
            qa_grid,
        )

        im3 = ax[1, 1].imshow(
            qshow,
            cmap=qa_cmap,
            norm=qa_norm,
            interpolation="none",
            aspect="auto",
        )

        ax[1, 1].set_title(
            "Final QA classes"
        )

        cbar3 = fig.colorbar(
            im3,
            ax=ax[1, 1],
            fraction=0.025,
            pad=0.02,
            ticks=[0, 1, 2, 3],
        )

        cbar3.ax.set_yticklabels(
            [
                "strict valid",
                "temporal bad",
                "SAFE conflict",
                "not all-registered",
            ]
        )

        for a in ax.ravel():

            a.set_xlabel(
                "range pixel / col"
            )

            a.set_ylabel(
                "azimuth pixel / row"
            )

        qa = batch_qa.get(
            pair_id,
            {}
        )

        extra = []

        for key in (
            "safe_internal_bad",
            "selected_non_exact",
            "rejected_cycle_outliers",
        ):

            if key in qa:

                extra.append(
                    f"{key}={qa[key]}"
                )

        qa_text = (
            " | ".join(extra)
            if extra
            else "QA table unavailable"
        )

        fig.suptitle(
            f"{pair_id:03d}: {d1} → {d2}"
            f" | global gauge={gauge:+d} × 2π"
            f"\n{qa_text}",
            fontsize=12,
        )

        panel_path = (
            panel_dir
            / f"{tag}_QA.png"
        )

        fig.savefig(
            panel_path,
            dpi=args.dpi,
        )

        plt.close(fig)

        # ====================================================
        # Separate products for quick browsing
        # ====================================================

        for name, grid, cmap, vmin, vmax, title, target in (
            (
                "wrapped",
                wrapped_grid,
                "twilight",
                -np.pi,
                np.pi,
                f"{tag} wrapped phase [rad]",
                wrapped_dir
                / f"{tag}_wrapped.png",
            ),
            (
                "unwrapped",
                unw_grid,
                "turbo",
                uvmin,
                uvmax,
                f"{tag} unwrapped phase [rad]",
                unwrapped_dir
                / f"{tag}_unwrapped.png",
            ),
            (
                "ambiguity",
                n_grid,
                "RdBu_r",
                -nmax,
                nmax,
                f"{tag} integer ambiguity [cycles]",
                ambiguity_dir
                / f"{tag}_ambiguity.png",
            ),
        ):

            fig2, a2 = plt.subplots(
                figsize=(13, 4.5),
                constrained_layout=True,
            )

            im = a2.imshow(
                grid,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="none",
                aspect="auto",
            )

            a2.set_title(
                title
            )

            a2.set_xlabel(
                "range pixel / col"
            )

            a2.set_ylabel(
                "azimuth pixel / row"
            )

            fig2.colorbar(
                im,
                ax=a2,
                fraction=0.025,
                pad=0.02,
            )

            fig2.savefig(
                target,
                dpi=args.dpi,
            )

            plt.close(fig2)

        # ====================================================
        # Statistics
        # ====================================================

        strict_n = integer_n[
            strict_valid
        ]

        row = {
            "pair_id":
                pair_id,

            "date1":
                d1,

            "date2":
                d2,

            "global_integer_gauge":
                gauge,

            "unwrapped_p01_rad":
                float(
                    uvmin
                ),

            "unwrapped_p99_rad":
                float(
                    uvmax
                ),

            "integer_min":
                int(
                    strict_n.min()
                )
                if strict_n.size
                else 0,

            "integer_median":
                float(
                    np.median(
                        strict_n
                    )
                )
                if strict_n.size
                else 0.0,

            "integer_max":
                int(
                    strict_n.max()
                )
                if strict_n.size
                else 0,

            "integer_abs_max":
                int(
                    np.max(
                        np.abs(
                            strict_n
                        )
                    )
                )
                if strict_n.size
                else 0,

            "panel":
                str(
                    panel_path
                ),
        }

        if qa:

            for key in (
                "safe_internal_bad",
                "final_safe_bad",
                "selected_non_exact",
                "selected_ratio_lt_0p75",
                "rejected_cycle_outliers",
                "unsafe_within_bad",
                "unsafe_cross_bad",
            ):

                if key in qa:

                    row[
                        key
                    ] = qa[
                        key
                    ]

        summary.append(
            row
        )

        print(
            f"{pair_id:3d}/{nedge}: "
            f"{d1}->{d2} "
            f"gauge={gauge:+d}, "
            f"N=["
            f"{row['integer_min']},"
            f"{row['integer_max']}"
            f"]"
        )

    # ========================================================
    # Save visual index
    # ========================================================

    if summary:

        # Put known numerical-risk metrics first if present.
        def risk(r):

            x = 0.0

            for key, weight in (
                (
                    "safe_internal_bad",
                    1000.0
                ),
                (
                    "final_safe_bad",
                    1000.0
                ),
                (
                    "rejected_cycle_outliers",
                    100.0
                ),
                (
                    "selected_ratio_lt_0p75",
                    10.0
                ),
                (
                    "selected_non_exact",
                    1.0
                ),
                (
                    "unsafe_cross_bad",
                    0.1
                ),
                (
                    "unsafe_within_bad",
                    0.05
                ),
            ):

                if key in r:

                    try:
                        x += (
                            weight
                            *
                            float(
                                r[key]
                            )
                        )
                    except Exception:
                        pass

            return x

        for r in summary:

            r[
                "visual_review_risk_score"
            ] = risk(r)

        ranked = sorted(
            summary,
            key=lambda r:
                r[
                    "visual_review_risk_score"
                ],
            reverse=True,
        )

        csv_path = (
            outdir
            / "visual_qa_index.csv"
        )

        fields = []

        for r in ranked:

            for k in r.keys():

                if k not in fields:
                    fields.append(k)

        with csv_path.open(
            "w",
            newline="",
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            w.writeheader()
            w.writerows(
                ranked
            )

        print()
        print("=" * 104)
        print(
            "Top IFGs recommended for visual inspection"
        )
        print("=" * 104)

        for r in ranked[:20]:

            print(
                f"pair {r['pair_id']:3d} "
                f"{r['date1']}->{r['date2']} "
                f"risk="
                f"{r['visual_review_risk_score']:.1f} "
                f"Nmax="
                f"{r['integer_abs_max']}"
            )

    manifest = {
        "format":
            "pyPSDS-GAMMA-ifg-visual-qa-v0.9",

        "status":
            "VISUAL_QA_ONLY",

        "points":
            int(
                npoint
            ),

        "acquisitions":
            int(
                ndate
            ),

        "ifgs":
            int(
                nedge
            ),

        "display_grid": [
            int(H),
            int(W),
        ],

        "rendered_pairs":
            len(
                summary
            ),

        "phase_source":
            (
                "wrapped = PointPhaseStack acquisition "
                "phase difference"
            ),

        "unwrapped_source":
            (
                "single_ifg_robust_solution + "
                "Step08x global integer gauge"
            ),

        "interpolation":
            "NONE",

        "note":
            (
                "Radar-grid visualization only. "
                "Empty pixels are NaN. "
                "No phase product was modified."
            ),
    }

    manifest_path = (
        outdir
        / "visual_qa_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print(
        f"output directory       : {outdir}"
    )

    print(
        f"visual QA index        : "
        f"{outdir / 'visual_qa_index.csv'}"
    )

    print(
        f"manifest               : {manifest_path}"
    )

    print()
    print(
        "STEP 08y STATUS: PASS / VISUAL QA ONLY"
    )

    print(
        "No phase file has been modified."
    )


if __name__ == "__main__":
    main()
