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
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm

from pypsds.prototype import open_from_config


TWOPI = 2.0 * np.pi


# ============================================================
# Utilities
# ============================================================

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


def point_to_grid(
    values,
    rows,
    cols,
    height,
    width,
    dtype=np.float32,
):

    out = np.full(
        (height, width),
        np.nan,
        dtype=dtype,
    )

    out[
        rows,
        cols
    ] = values

    return out


def robust_limits(
    values,
    mask=None,
    qlo=1.0,
    qhi=99.0,
):

    if mask is None:
        x = np.asarray(
            values
        )
    else:
        x = np.asarray(
            values[
                mask
            ]
        )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return -1.0, 1.0

    lo, hi = np.percentile(
        x,
        [qlo, qhi],
    )

    if hi <= lo:

        c = float(
            np.median(x)
        )

        lo = c - 1.0
        hi = c + 1.0

    return (
        float(lo),
        float(hi),
    )


def read_batch_qa(path: Path):

    out = {}

    if not path.exists():
        return out

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


# ============================================================
# Automatic profile row selection
# ============================================================

def choose_profile_rows(
    rows,
    branch_u,
    branch_v,
    n_profiles=3,
    min_sep=60,
):

    max_row = int(
        rows.max()
    )

    # Prefer rows crossed by integer branch boundaries.
    score = np.zeros(
        max_row + 1,
        dtype=np.int64,
    )

    if branch_u.size:

        bu_rows = rows[
            branch_u
        ]

        bv_rows = rows[
            branch_v
        ]

        np.add.at(
            score,
            bu_rows,
            1,
        )

        np.add.at(
            score,
            bv_rows,
            1,
        )

    # Add weak point-density term so selected rows
    # still have enough samples.
    density = np.bincount(
        rows,
        minlength=max_row + 1,
    )

    combined = (
        score.astype(
            np.float64
        )
        * 1000.0
        +
        density.astype(
            np.float64
        )
    )

    order = np.argsort(
        combined
    )[::-1]

    chosen = []

    for r in order.tolist():

        if density[r] == 0:
            continue

        if all(
            abs(r - old)
            >=
            min_sep
            for old in chosen
        ):

            chosen.append(
                int(r)
            )

        if len(chosen) >= n_profiles:
            break

    # Fallback to scene fractions.
    if len(chosen) < n_profiles:

        fallback = [
            int(
                0.25 * max_row
            ),
            int(
                0.50 * max_row
            ),
            int(
                0.75 * max_row
            ),
        ]

        for r in fallback:

            if all(
                abs(r - old)
                >=
                min_sep // 2
                for old in chosen
            ):

                chosen.append(r)

            if len(chosen) >= n_profiles:
                break

    return sorted(
        chosen[:n_profiles]
    )


def extract_profile(
    values,
    rows,
    cols,
    center_row,
    half_width=2,
):

    mask = (
        np.abs(
            rows
            -
            center_row
        )
        <=
        half_width
    )

    idx = np.where(
        mask
    )[0]

    if idx.size == 0:

        return (
            np.empty(0),
            np.empty(0),
        )

    x = cols[
        idx
    ]

    y = values[
        idx
    ]

    order = np.argsort(
        x
    )

    return (
        x[
            order
        ],
        y[
            order
        ],
    )


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--height",
        type=int,
        default=600,
    )

    ap.add_argument(
        "--width",
        type=int,
        default=2000,
    )

    ap.add_argument(
        "--only",
        type=int,
        nargs="*",
        default=None,
    )

    ap.add_argument(
        "--dpi",
        type=int,
        default=160,
    )

    ap.add_argument(
        "--profile-half-width",
        type=int,
        default=2,
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

    graph_dir = (
        root
        / "spatial_graph"
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
        / "ifg_visual_qa_v2"
    )

    panel_dir = (
        outdir
        / "panels"
    )

    profile_dir = (
        outdir
        / "profiles"
    )

    panel_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Inputs
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

    local_u = np.asarray(
        np.load(
            graph_dir
            / "local_u.npy",
            mmap_mode="r",
        ),
        dtype=np.int32,
    )

    local_v = np.asarray(
        np.load(
            graph_dir
            / "local_v.npy",
            mmap_mode="r",
        ),
        dtype=np.int32,
    )

    global_gauge = np.asarray(
        np.load(
            final_dir
            / "global_ifg_integer_delta.npy",
        ),
        dtype=np.int32,
    )

    strict_valid = np.asarray(
        np.load(
            final_dir
            / "strict_unwrap_valid_mask.npy",
        ),
        dtype=bool,
    )

    temporal_bad = np.asarray(
        np.load(
            final_dir
            / "temporal_integer_bad_mask.npy",
        ),
        dtype=bool,
    )

    safe_conflict = np.asarray(
        np.load(
            final_dir
            / "spatial_safe_conflict_point_mask.npy",
        ),
        dtype=bool,
    )

    all_registered = np.asarray(
        np.load(
            final_dir
            / "all_ifg_registered_mask.npy",
        ),
        dtype=bool,
    )

    npoint, ndate = phase.shape

    temporal_edges = load_itab(
        network_dir
        / "network.itab",
        ndate,
    )

    nedge = len(
        temporal_edges
    )

    qa = read_batch_qa(
        batch_qa_path
    )

    if args.only:

        wanted = set(
            int(x)
            for x in args.only
        )

    else:

        wanted = set(
            range(
                1,
                nedge + 1
            )
        )

    print("=" * 108)
    print(
        "Step 08y2 - Spatial ambiguity / branch-boundary visual QA"
    )
    print("=" * 108)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"points                     : "
        f"{npoint:,}"
    )

    print(
        f"local graph edges          : "
        f"{local_u.size:,}"
    )

    print(
        f"IFGs                       : "
        f"{nedge}"
    )

    print(
        f"rendering                  : "
        f"{len(wanted)} IFGs"
    )

    summary_rows = []

    # ========================================================
    # IFG loop
    # ========================================================

    for pair_id, (i, j) in enumerate(
        temporal_edges,
        start=1,
    ):

        if pair_id not in wanted:
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

        path = (
            unwrap_dir
            / (
                f"{tag}_"
                "unwrapped_phase_rad.npy"
            )
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        U_raw = np.asarray(
            np.load(
                path,
                mmap_mode="r",
            ),
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Canonical wrapped IFG
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DISPLAY unwrapped phase:
        # apply global temporal gauge.
        # ----------------------------------------------------

        gauge = int(
            global_gauge[
                pair_id - 1
            ]
        )

        U_display = (
            U_raw
            +
            TWOPI
            *
            gauge
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # spatial ambiguity EXCLUDES global gauge.
        # ----------------------------------------------------

        N_spatial = np.rint(
            (
                U_raw
                -
                wrapped
            )
            /
            TWOPI
        ).astype(
            np.int16
        )

        # ----------------------------------------------------
        # Wrapped parity QA
        # ----------------------------------------------------

        parity = np.abs(
            wrap(
                U_raw
                -
                wrapped
            )
        )

        parity_max = float(
            parity.max()
        )

        # ====================================================
        # Branch boundaries in LOCAL graph
        #
        # A boundary exists where two linked points carry
        # different spatial integer ambiguity.
        # ====================================================

        Nu = N_spatial[
            local_u
        ]

        Nv = N_spatial[
            local_v
        ]

        dN = (
            Nv
            -
            Nu
        ).astype(
            np.int16
        )

        branch = (
            dN
            !=
            0
        )

        branch_ids = np.where(
            branch
        )[0]

        branch_u = local_u[
            branch_ids
        ]

        branch_v = local_v[
            branch_ids
        ]

        branch_dN = dN[
            branch_ids
        ]

        nbranch = int(
            branch_ids.size
        )

        branch_abs1 = int(
            np.count_nonzero(
                np.abs(
                    branch_dN
                )
                ==
                1
            )
        )

        branch_abs2plus = int(
            np.count_nonzero(
                np.abs(
                    branch_dN
                )
                >=
                2
            )
        )

        # ----------------------------------------------------
        # How many branch boundaries correspond to a raw
        # wrapped +/-pi discontinuity?
        #
        # This is VISUAL QA only, not an algorithm criterion.
        # ----------------------------------------------------

        raw_diff = (
            wrapped[
                local_v
            ]
            -
            wrapped[
                local_u
            ]
        )

        raw_wrap_jump = (
            np.abs(
                raw_diff
            )
            >
            np.pi
        )

        branch_with_raw_wrap_jump = int(
            np.count_nonzero(
                branch
                &
                raw_wrap_jump
            )
        )

        branch_wrap_fraction = (
            branch_with_raw_wrap_jump
            /
            nbranch
            if nbranch
            else 0.0
        )

        # ====================================================
        # Grid images
        # ====================================================

        wrapped_grid = point_to_grid(
            wrapped.astype(
                np.float32
            ),
            rows,
            cols,
            args.height,
            args.width,
        )

        unw_grid = point_to_grid(
            U_display.astype(
                np.float32
            ),
            rows,
            cols,
            args.height,
            args.width,
        )

        N_grid = point_to_grid(
            N_spatial.astype(
                np.float32
            ),
            rows,
            cols,
            args.height,
            args.width,
        )

        uvmin, uvmax = robust_limits(
            U_display,
            strict_valid,
            1,
            99,
        )

        n_strict = N_spatial[
            strict_valid
        ]

        if n_strict.size:

            nmin = int(
                n_strict.min()
            )

            nmax = int(
                n_strict.max()
            )

        else:

            nmin = -1
            nmax = +1

        lo_n = min(
            nmin,
            -1,
        )

        hi_n = max(
            nmax,
            +1,
        )

        bounds = (
            np.arange(
                lo_n - 0.5,
                hi_n + 1.5,
                1.0,
            )
        )

        cmap_N = plt.get_cmap(
            "RdBu_r",
            len(bounds) - 1,
        )

        norm_N = BoundaryNorm(
            bounds,
            cmap_N.N,
        )

        # ====================================================
        # Main 4-panel figure
        # ====================================================

        fig, ax = plt.subplots(
            2,
            2,
            figsize=(17, 9),
            constrained_layout=True,
        )

        # ----------------------------------------------------
        # 1. Wrapped
        # ----------------------------------------------------

        im0 = ax[
            0,
            0
        ].imshow(
            wrapped_grid,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="none",
            aspect="auto",
        )

        ax[
            0,
            0
        ].set_title(
            "Wrapped virtual IFG [rad]"
        )

        fig.colorbar(
            im0,
            ax=ax[
                0,
                0
            ],
            fraction=0.025,
            pad=0.02,
        )

        # ----------------------------------------------------
        # 2. Unwrapped
        # ----------------------------------------------------

        im1 = ax[
            0,
            1
        ].imshow(
            unw_grid,
            cmap="turbo",
            vmin=uvmin,
            vmax=uvmax,
            interpolation="none",
            aspect="auto",
        )

        ax[
            0,
            1
        ].set_title(
            "Unwrapped IFG [rad] "
            f"(display gauge {gauge:+d} × 2π)"
        )

        fig.colorbar(
            im1,
            ax=ax[
                0,
                1
            ],
            fraction=0.025,
            pad=0.02,
        )

        # ----------------------------------------------------
        # 3. TRUE spatial ambiguity
        # ----------------------------------------------------

        im2 = ax[
            1,
            0
        ].imshow(
            N_grid,
            cmap=cmap_N,
            norm=norm_N,
            interpolation="none",
            aspect="auto",
        )

        ax[
            1,
            0
        ].set_title(
            "Spatial ambiguity "
            "Nspatial = round((Uraw - wrapped)/2π)\n"
            "GLOBAL GAUGE EXCLUDED"
        )

        cb2 = fig.colorbar(
            im2,
            ax=ax[
                1,
                0
            ],
            fraction=0.025,
            pad=0.02,
            ticks=np.arange(
                lo_n,
                hi_n + 1,
            ),
        )

        cb2.set_label(
            "integer cycles"
        )

        # ----------------------------------------------------
        # 4. Branch boundary map
        # ----------------------------------------------------

        ax[
            1,
            1
        ].imshow(
            wrapped_grid,
            cmap="gray",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="none",
            aspect="auto",
            alpha=0.45,
        )

        if nbranch:

            segments = np.stack(
                [
                    np.stack(
                        [
                            cols[
                                branch_u
                            ],
                            rows[
                                branch_u
                            ],
                        ],
                        axis=1,
                    ),
                    np.stack(
                        [
                            cols[
                                branch_v
                            ],
                            rows[
                                branch_v
                            ],
                        ],
                        axis=1,
                    ),
                ],
                axis=1,
            ).astype(
                np.float64
            )

            pos = (
                branch_dN
                >
                0
            )

            neg = (
                branch_dN
                <
                0
            )

            if np.any(pos):

                lc = LineCollection(
                    segments[
                        pos
                    ],
                    linewidths=0.55,
                    alpha=0.80,
                    colors="red",
                )

                ax[
                    1,
                    1
                ].add_collection(
                    lc
                )

            if np.any(neg):

                lc = LineCollection(
                    segments[
                        neg
                    ],
                    linewidths=0.55,
                    alpha=0.80,
                    colors="blue",
                )

                ax[
                    1,
                    1
                ].add_collection(
                    lc
                )

        # QA points
        bad_temporal_ids = np.where(
            temporal_bad
        )[0]

        safe_conflict_ids = np.where(
            safe_conflict
        )[0]

        unregister_ids = np.where(
            ~all_registered
        )[0]

        if bad_temporal_ids.size:

            ax[
                1,
                1
            ].scatter(
                cols[
                    bad_temporal_ids
                ],
                rows[
                    bad_temporal_ids
                ],
                s=8,
                marker="x",
                c="black",
                linewidths=0.7,
                label="temporal bad",
            )

        if safe_conflict_ids.size:

            ax[
                1,
                1
            ].scatter(
                cols[
                    safe_conflict_ids
                ],
                rows[
                    safe_conflict_ids
                ],
                s=10,
                marker="o",
                facecolors="none",
                edgecolors="orange",
                linewidths=0.8,
                label="SAFE-conflict endpoint",
            )

        if unregister_ids.size:

            ax[
                1,
                1
            ].scatter(
                cols[
                    unregister_ids
                ],
                rows[
                    unregister_ids
                ],
                s=12,
                marker="s",
                facecolors="none",
                edgecolors="purple",
                linewidths=0.8,
                label="not all-registered",
            )

        ax[
            1,
            1
        ].set_title(
            "Spatial integer branch boundaries\n"
            "red: ΔN>0 | blue: ΔN<0"
        )

        if (
            bad_temporal_ids.size
            or
            safe_conflict_ids.size
            or
            unregister_ids.size
        ):

            ax[
                1,
                1
            ].legend(
                loc="upper right",
                fontsize=7,
            )

        for a in ax.ravel():

            a.set_xlabel(
                "range pixel / col"
            )

            a.set_ylabel(
                "azimuth pixel / row"
            )

            a.set_xlim(
                0,
                args.width
            )

            a.set_ylim(
                args.height,
                0,
            )

        q = qa.get(
            pair_id,
            {}
        )

        safe_internal_bad = q.get(
            "safe_internal_bad",
            "?"
        )

        selected_non_exact = q.get(
            "selected_non_exact",
            "?"
        )

        rejected_cycle_outliers = q.get(
            "rejected_cycle_outliers",
            "?"
        )

        fig.suptitle(
            f"{pair_id:03d}: {d1} → {d2}"
            f" | global gauge={gauge:+d} × 2π\n"
            f"safe_internal_bad={safe_internal_bad}"
            f" | selected_non_exact={selected_non_exact}"
            f" | rejected_cycle_outliers="
            f"{rejected_cycle_outliers}"
            f" | branch_edges={nbranch:,}",
            fontsize=12,
        )

        panel_path = (
            panel_dir
            / f"{tag}_QA_v2.png"
        )

        fig.savefig(
            panel_path,
            dpi=args.dpi,
        )

        plt.close(fig)

        # ====================================================
        # Automatic wrapped/unwrapped profiles
        # ====================================================

        profile_rows = choose_profile_rows(
            rows,
            branch_u,
            branch_v,
            n_profiles=3,
            min_sep=60,
        )

        figp, axes = plt.subplots(
            len(profile_rows),
            1,
            figsize=(
                15,
                3.2
                *
                len(profile_rows)
            ),
            constrained_layout=True,
        )

        if len(profile_rows) == 1:
            axes = [
                axes
            ]

        for a, center_row in zip(
            axes,
            profile_rows,
        ):

            xw, yw = extract_profile(
                wrapped,
                rows,
                cols,
                center_row,
                half_width=args.profile_half_width,
            )

            xu, yu = extract_profile(
                U_display,
                rows,
                cols,
                center_row,
                half_width=args.profile_half_width,
            )

            if xw.size:

                a.scatter(
                    xw,
                    yw,
                    s=3,
                    alpha=0.55,
                    label="wrapped",
                )

            if xu.size:

                a.scatter(
                    xu,
                    yu,
                    s=3,
                    alpha=0.55,
                    label="unwrapped",
                )

            a.axhline(
                np.pi,
                linewidth=0.7,
                linestyle="--",
                alpha=0.5,
            )

            a.axhline(
                -np.pi,
                linewidth=0.7,
                linestyle="--",
                alpha=0.5,
            )

            a.set_xlim(
                0,
                args.width,
            )

            a.set_ylabel(
                "phase [rad]"
            )

            a.set_title(
                f"row ≈ {center_row} "
                f"(±{args.profile_half_width} px)"
            )

            a.grid(
                alpha=0.15,
            )

            a.legend(
                loc="best",
                fontsize=8,
            )

        axes[
            -1
        ].set_xlabel(
            "range pixel / col"
        )

        figp.suptitle(
            f"{pair_id:03d}: {d1} → {d2}\n"
            "Wrapped vs unwrapped spatial profiles",
            fontsize=12,
        )

        profile_path = (
            profile_dir
            / f"{tag}_profiles.png"
        )

        figp.savefig(
            profile_path,
            dpi=args.dpi,
        )

        plt.close(figp)

        # ====================================================
        # Summary row
        # ====================================================

        row_out = {
            "pair_id":
                pair_id,

            "date1":
                d1,

            "date2":
                d2,

            "global_gauge":
                gauge,

            "N_spatial_min":
                int(
                    n_strict.min()
                )
                if n_strict.size
                else 0,

            "N_spatial_median":
                float(
                    np.median(
                        n_strict
                    )
                )
                if n_strict.size
                else 0.0,

            "N_spatial_max":
                int(
                    n_strict.max()
                )
                if n_strict.size
                else 0,

            "branch_edges":
                nbranch,

            "branch_abs1":
                branch_abs1,

            "branch_abs2plus":
                branch_abs2plus,

            "branch_with_raw_wrap_jump":
                branch_with_raw_wrap_jump,

            "branch_raw_wrap_fraction":
                branch_wrap_fraction,

            "wrap_parity_max_rad":
                parity_max,

            "safe_internal_bad":
                int(
                    safe_internal_bad
                )
                if str(
                    safe_internal_bad
                ).lstrip("-").isdigit()
                else -1,

            "selected_non_exact":
                int(
                    selected_non_exact
                )
                if str(
                    selected_non_exact
                ).lstrip("-").isdigit()
                else -1,

            "rejected_cycle_outliers":
                int(
                    rejected_cycle_outliers
                )
                if str(
                    rejected_cycle_outliers
                ).lstrip("-").isdigit()
                else -1,

            "panel":
                str(
                    panel_path
                ),

            "profiles":
                str(
                    profile_path
                ),
        }

        summary_rows.append(
            row_out
        )

        print(
            f"{pair_id:3d}: "
            f"{d1}->{d2} "
            f"gauge={gauge:+d}, "
            f"Nspatial=["
            f"{row_out['N_spatial_min']},"
            f"{row_out['N_spatial_max']}], "
            f"branches={nbranch:,}, "
            f"|dN|>=2={branch_abs2plus:,}, "
            f"rawWrapMatch="
            f"{100*branch_wrap_fraction:.2f}%"
        )

    # ========================================================
    # Risk ranking
    # ========================================================

    for r in summary_rows:

        # QA only.
        # Large |dN| and known integer conflicts dominate.
        r[
            "visual_risk_score"
        ] = (
            10000.0
            *
            r[
                "branch_abs2plus"
            ]
            +
            1000.0
            *
            max(
                0,
                r[
                    "safe_internal_bad"
                ]
            )
            +
            100.0
            *
            max(
                0,
                r[
                    "rejected_cycle_outliers"
                ]
            )
            +
            1.0
            *
            max(
                0,
                r[
                    "selected_non_exact"
                ]
            )
        )

    summary_rows.sort(
        key=lambda r:
            r[
                "visual_risk_score"
            ],
        reverse=True,
    )

    if summary_rows:

        csv_path = (
            outdir
            / "visual_qa_v2_index.csv"
        )

        with csv_path.open(
            "w",
            newline="",
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=list(
                    summary_rows[
                        0
                    ].keys()
                ),
            )

            w.writeheader()

            w.writerows(
                summary_rows
            )

        print()
        print("=" * 108)
        print(
            "Highest visual-QA risk IFGs"
        )
        print("=" * 108)

        for r in summary_rows[
            :min(
                20,
                len(
                    summary_rows
                ),
            )
        ]:

            print(
                f"pair {r['pair_id']:3d} "
                f"{r['date1']}->{r['date2']} "
                f"branches={r['branch_edges']:6d} "
                f"|dN|>=2="
                f"{r['branch_abs2plus']:4d} "
                f"safeBad="
                f"{r['safe_internal_bad']:3d} "
                f"score="
                f"{r['visual_risk_score']:.1f}"
            )

    manifest = {
        "format":
            "pyPSDS-GAMMA-ifg-visual-qa-v2",

        "status":
            "VISUAL_QA_ONLY",

        "points":
            int(
                npoint
            ),

        "ifgs":
            int(
                nedge
            ),

        "rendered":
            int(
                len(
                    summary_rows
                )
            ),

        "spatial_ambiguity_definition":
            (
                "round((U_raw - wrapped)/2pi); "
                "global temporal gauge explicitly excluded"
            ),

        "branch_definition":
            (
                "local R4-K8 edge with "
                "N_spatial[v] != N_spatial[u]"
            ),

        "phase_modified":
            False,
    }

    manifest_path = (
        outdir
        / "visual_qa_v2_manifest.json"
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
        f"output directory           : "
        f"{outdir}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        "STEP 08y2 STATUS: PASS / "
        "VISUAL QA ONLY"
    )

    print(
        "No phase or unwrap product "
        "has been modified."
    )


if __name__ == "__main__":
    main()
