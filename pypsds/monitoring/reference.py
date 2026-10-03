from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pypsds.config import cfg_get
from pypsds.context import open_from_config


def rank01(values):
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    good = np.flatnonzero(np.isfinite(x))
    if good.size == 0:
        return out
    if good.size == 1:
        out[good] = 1.0
        return out
    order = good[np.argsort(x[good], kind="mergesort")]
    out[order] = np.linspace(0.0, 1.0, order.size)
    return out


# PYPSDS_FAST_AUTO_REFERENCE_V1
def choose_reference_region(
    xy_m,
    rate_abs,
    residual,
    *,
    radius_m,
    cell_size_m,
    min_points,
    rate_weight=0.60,
    residual_weight=0.30,
    density_weight=0.10,
):
    """
    Full-scene scalable automatic reference selector.

    Scientific definition is unchanged:
      1. partition valid points into cell_size_m cells;
      2. use median XY of each sufficiently populated cell as candidate center;
      3. select all points within radius_m;
      4. rank by low |rate|, low residual, and high density.

    Performance change:
      - one O(N log N) cell sort instead of O(N * Ncell) repeated scans;
      - one cKDTree for exact circular neighborhoods;
      - batched parallel radius queries;
      - only the winning candidate retains its full point-index vector.
    """
    import hashlib
    import os
    import time

    xy = np.asarray(
        xy_m,
        dtype=np.float64,
    )

    rate = np.asarray(
        rate_abs,
        dtype=np.float64,
    )

    rms = np.asarray(
        residual,
        dtype=np.float64,
    )

    valid = (
        np.all(
            np.isfinite(xy),
            axis=1,
        )
        &
        np.isfinite(rate)
        &
        np.isfinite(rms)
    )

    valid_ids = np.flatnonzero(
        valid
    )

    if valid_ids.size < min_points:
        raise RuntimeError(
            "Too few finite points for automatic reference"
        )

    vxy = np.ascontiguousarray(
        xy[valid_ids],
        dtype=np.float64,
    )

    print(
        "[AUTO REF] valid points:",
        f"{valid_ids.size:,}",
        flush=True,
    )

    # ================================================================
    # Spatial index
    # ================================================================

    t0 = time.perf_counter()

    print(
        "[AUTO REF] building cKDTree ...",
        flush=True,
    )

    tree = cKDTree(
        vxy,
        compact_nodes=True,
        balanced_tree=True,
    )

    print(
        "[AUTO REF] cKDTree ready:",
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    # ================================================================
    # Cell grouping -- sort ONCE
    # ================================================================

    x0 = float(
        np.min(
            vxy[:, 0]
        )
    )

    y0 = float(
        np.min(
            vxy[:, 1]
        )
    )

    cx = np.floor(
        (
            vxy[:, 0]
            -
            x0
        )
        /
        cell_size_m
    ).astype(
        np.int64
    )

    cy = np.floor(
        (
            vxy[:, 1]
            -
            y0
        )
        /
        cell_size_m
    ).astype(
        np.int64
    )

    cx0 = int(
        cx.min()
    )

    cy0 = int(
        cy.min()
    )

    cy_span = (
        int(
            cy.max()
        )
        -
        cy0
        +
        1
    )

    key = (
        (
            cx
            -
            cx0
        )
        *
        cy_span
        +
        (
            cy
            -
            cy0
        )
    )

    print(
        "[AUTO REF] sorting spatial cells ...",
        flush=True,
    )

    order = np.argsort(
        key,
        kind="stable",
    )

    key_sorted = key[
        order
    ]

    (
        unique_keys,
        starts,
        counts,
    ) = np.unique(
        key_sorted,
        return_index=True,
        return_counts=True,
    )

    seed_min = max(
        3,
        min_points // 10,
    )

    eligible = np.flatnonzero(
        counts
        >=
        seed_min
    )

    if eligible.size == 0:
        raise RuntimeError(
            "No sufficiently populated automatic-reference cells"
        )

    print(
        "[AUTO REF] occupied cells:",
        f"{unique_keys.size:,}",
        "| candidate cells:",
        f"{eligible.size:,}",
        flush=True,
    )

    # ================================================================
    # Exact original candidate-center definition:
    # median XY of points inside each candidate cell
    # ================================================================

    centers = np.empty(
        (
            eligible.size,
            2,
        ),
        dtype=np.float64,
    )

    for q, ui in enumerate(
        eligible
    ):
        a = int(
            starts[ui]
        )

        b = (
            a
            +
            int(
                counts[ui]
            )
        )

        local = order[
            a:b
        ]

        centers[
            q,
            :
        ] = np.median(
            vxy[
                local,
                :
            ],
            axis=0,
        )

        if (
            q == 0
            or
            (q + 1) % 2000 == 0
            or
            q + 1 == eligible.size
        ):
            print(
                "[AUTO REF CENTER]",
                f"{q + 1:,}/{eligible.size:,}",
                f"({100.0*(q+1)/eligible.size:.1f}%)",
                flush=True,
            )

    # Large temporary cell arrays are no longer needed.
    del (
        key_sorted,
        key,
        cx,
        cy,
        order,
    )

    # ================================================================
    # Exact circular region evaluation
    # ================================================================

    workers = min(
        16,
        max(
            1,
            os.cpu_count()
            or
            1,
        ),
    )

    query_batch = 128

    candidates = []
    seen = set()

    print(
        "[AUTO REF] evaluating exact 500-m regions",
        f"with {workers} workers ...",
        flush=True,
    )

    t_query = time.perf_counter()

    for q0 in range(
        0,
        centers.shape[0],
        query_batch,
    ):
        q1 = min(
            centers.shape[0],
            q0
            +
            query_batch,
        )

        regions = tree.query_ball_point(
            centers[
                q0:q1
            ],
            r=radius_m,
            workers=workers,
        )

        for kk, region_list in enumerate(
            regions
        ):
            if len(
                region_list
            ) < min_points:
                continue

            region_local = np.asarray(
                region_list,
                dtype=np.int64,
            )

            region = np.sort(
                valid_ids[
                    region_local
                ]
            )

            # Same logical duplicate suppression as the old tuple(region)
            # approach without retaining huge Python tuples.
            digest = hashlib.blake2b(
                memoryview(
                    np.ascontiguousarray(
                        region
                    )
                ),
                digest_size=16,
            ).digest()

            if digest in seen:
                continue

            seen.add(
                digest
            )

            centre = centers[
                q0 + kk
            ]

            candidates.append(
                {
                    "center_x_m":
                        float(
                            centre[0]
                        ),

                    "center_y_m":
                        float(
                            centre[1]
                        ),

                    "n_points":
                        int(
                            region.size
                        ),

                    "median_abs_rate":
                        float(
                            np.median(
                                rate[
                                    region
                                ]
                            )
                        ),

                    "median_residual":
                        float(
                            np.median(
                                rms[
                                    region
                                ]
                            )
                        ),
                }
            )

        print(
            "[AUTO REF REGION]",
            f"{q1:,}/{centers.shape[0]:,}",
            f"({100.0*q1/centers.shape[0]:.1f}%)",
            "accepted=",
            f"{len(candidates):,}",
            flush=True,
        )

    print(
        "[AUTO REF] region evaluation:",
        f"{time.perf_counter() - t_query:.1f}s",
        flush=True,
    )

    if not candidates:
        raise RuntimeError(
            "No automatic reference region satisfies "
            "radius/min_points constraints"
        )

    # ================================================================
    # Original scoring definition
    # ================================================================

    q_rate = rank01(
        -np.asarray(
            [
                x[
                    "median_abs_rate"
                ]
                for x in candidates
            ]
        )
    )

    q_rms = rank01(
        -np.asarray(
            [
                x[
                    "median_residual"
                ]
                for x in candidates
            ]
        )
    )

    q_den = rank01(
        np.log1p(
            np.asarray(
                [
                    x[
                        "n_points"
                    ]
                    for x in candidates
                ],
                dtype=float,
            )
        )
    )

    weights = np.asarray(
        [
            rate_weight,
            residual_weight,
            density_weight,
        ],
        dtype=np.float64,
    )

    if (
        np.any(
            weights
            <
            0
        )
        or
        float(
            weights.sum()
        )
        <=
        0
    ):
        raise ValueError(
            "automatic-reference weights must be non-negative"
        )

    weights /= weights.sum()

    score = (
        weights[0]
        *
        q_rate
        +
        weights[1]
        *
        q_rms
        +
        weights[2]
        *
        q_den
    )

    for i, value in enumerate(
        score
    ):
        candidates[
            i
        ][
            "score"
        ] = float(
            value
        )

    ranked = sorted(
        candidates,
        key=lambda x:
            x[
                "score"
            ],
        reverse=True,
    )

    # Re-query only the winning region to retain its exact full-scene IDs.
    best = dict(
        ranked[0]
    )

    winner_local = np.asarray(
        tree.query_ball_point(
            np.asarray(
                [
                    best[
                        "center_x_m"
                    ],
                    best[
                        "center_y_m"
                    ],
                ],
                dtype=np.float64,
            ),
            r=radius_m,
        ),
        dtype=np.int64,
    )

    best[
        "indices"
    ] = np.sort(
        valid_ids[
            winner_local
        ]
    )

    print(
        "[AUTO REF] winner:",
        "points=",
        f"{best['indices'].size:,}",
        "rate=",
        f"{best['median_abs_rate']:.6e}",
        "rms=",
        f"{best['median_residual']:.6e}",
        "score=",
        f"{best['score']:.6f}",
        flush=True,
    )

    return (
        best,
        ranked,
    )


def _local_xy(lon, lat):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    R = 6371008.8
    x = np.deg2rad(lon - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return np.column_stack((x, y))


def select_auto_reference(config_path):
    """
    Select a high-quality *relative* reference region.

    The acquisition phase is first centered by a robust scene epoch median,
    then regions are ranked by low relative rate, low residual, and density.
    This cannot prove absolute physical stability.
    """
    cfg, _, paths, stack, _ = open_from_config(config_path)
    proc = Path(paths.output_dir) / "processing"
    inv = proc / "network_inversion"
    geom = proc / "point_geometry"
    outdir = proc / "referenced_timeseries"
    outdir.mkdir(parents=True, exist_ok=True)

    strict_ids = np.asarray(
        np.load(inv / "strict_point_ids.npy"),
        dtype=np.int32,
    )
    phase = np.load(
        inv
        / "acquisition_phase_l2_candidate_rad.npy",
        mmap_mode="r",
    )
    lon = np.asarray(
        np.load(geom / "longitude_deg.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    lat = np.asarray(
        np.load(geom / "latitude_deg.npy", mmap_mode="r"),
        dtype=np.float64,
    )

    n, ndate = phase.shape
    if strict_ids.size != n or lon.size != n or lat.size != n:
        raise RuntimeError("automatic-reference geometry/phase contract mismatch")

    dates = [datetime.strptime(str(x), "%Y%m%d") for x in stack.dates]
    years = np.asarray(
        [(x - dates[0]).days / 365.25 for x in dates],
        dtype=np.float64,
    )
    tc = years - years.mean()
    denom = float(tc @ tc)
    if denom <= 0:
        raise RuntimeError("invalid time axis")

    sample_n = min(
        n,
        int(cfg_get(cfg, "reference.auto.scene_median_sample", 100000)),
    )
    sample_idx = np.linspace(0, n - 1, sample_n, dtype=np.int64)
    scene_epoch_median = np.median(
        np.asarray(phase[sample_idx], dtype=np.float64),
        axis=0,
    )

    rate = np.empty(n, dtype=np.float64)
    temporal_rms = np.empty(n, dtype=np.float64)

    auto_batch = int(
        cfg_get(
            cfg,
            "reference.auto.batch_size",
            131072,
        )
    )

    print(
        "[AUTO REF] temporal metrics:",
        f"{n:,} points",
        f"batch={auto_batch:,}",
        flush=True,
    )

    for b0 in range(0, n, auto_batch):
        b1 = min(b0 + auto_batch, n)

        Y = (
            np.asarray(
                phase[b0:b1],
                dtype=np.float64,
            )
            -
            scene_epoch_median[
                None,
                :
            ]
        )

        ym = np.mean(
            Y,
            axis=1,
        )

        slope = (
            (
                Y
                -
                ym[
                    :,
                    None
                ]
            )
            @
            tc
        ) / denom

        intercept = (
            ym
            -
            slope
            *
            float(
                np.mean(
                    years
                )
            )
        )

        fit = (
            intercept[
                :,
                None
            ]
            +
            slope[
                :,
                None
            ]
            *
            years[
                None,
                :
            ]
        )

        rr = (
            Y
            -
            fit
        )

        rate[
            b0:b1
        ] = np.abs(
            slope
        )

        temporal_rms[
            b0:b1
        ] = np.sqrt(
            np.mean(
                rr
                *
                rr,
                axis=1,
            )
        )

        if (
            b1 == n
            or
            b1 // 1000000
            !=
            b0 // 1000000
        ):
            print(
                "[AUTO REF METRIC]",
                f"{b1:,}/{n:,}",
                f"({100.0*b1/n:.2f}%)",
                flush=True,
            )

    network_rms_path = inv / "l2_network_residual_rms_rad.npy"
    if network_rms_path.is_file():
        nr = np.asarray(np.load(network_rms_path, mmap_mode="r"), dtype=np.float64)
        quality_rms = (
            np.hypot(temporal_rms, nr)
            if nr.shape == temporal_rms.shape
            else temporal_rms
        )
    else:
        quality_rms = temporal_rms

    xy = _local_xy(lon, lat)
    radius = float(cfg_get(cfg, "reference.auto.radius_m", 500.0))
    cell = float(cfg_get(cfg, "reference.auto.cell_size_m", 500.0))
    min_points = int(
        cfg_get(
            cfg,
            "reference.auto.min_points",
            cfg_get(cfg, "reference.min_points", 100),
        )
    )

    best, candidates = choose_reference_region(
        xy,
        rate,
        quality_rms,
        radius_m=radius,
        cell_size_m=cell,
        min_points=min_points,
        rate_weight=float(cfg_get(cfg, "reference.auto.rate_weight", 0.60)),
        residual_weight=float(
            cfg_get(cfg, "reference.auto.residual_weight", 0.30)
        ),
        density_weight=float(
            cfg_get(cfg, "reference.auto.density_weight", 0.10)
        ),
    )

    idx = np.asarray(best["indices"], dtype=np.int64)
    selected = outdir / "auto_reference_point_ids.npy"
    np.save(selected, strict_ids[idx].astype(np.int32))

    report = {
        "status": "PASS_AUTO_REFERENCE",
        "version": "1.3.0",
        "method": "auto_stable_relative_region",
        "points": int(idx.size),
        "longitude_median": float(np.median(lon[idx])),
        "latitude_median": float(np.median(lat[idx])),
        "radius_m": radius,
        "cell_size_m": cell,
        "median_abs_relative_phase_rate_rad_per_year": best["median_abs_rate"],
        "median_quality_residual_rad": best["median_residual"],
        "score": best["score"],
        "candidate_count": len(candidates),
        "top_candidates": [
            {k: v for k, v in row.items() if k != "indices"}
            for row in candidates[:10]
        ],
        "note": (
            "Automatic selection establishes a high-quality relative InSAR "
            "datum; it does not prove zero physical deformation. Prefer an "
            "externally validated stable reference when available."
        ),
        "point_ids_file": str(selected),
    }
    (outdir / "reference_selection.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("AUTOMATIC STABLE REFERENCE")
    print("=" * 96)
    print("points       :", idx.size)
    print("median lon   :", np.median(lon[idx]))
    print("median lat   :", np.median(lat[idx]))
    print("relative rate:", best["median_abs_rate"], "rad/yr")
    print("quality RMS  :", best["median_residual"], "rad")
    print("score        :", best["score"])
    print("=" * 96)
    return selected
