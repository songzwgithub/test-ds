#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


def resolve_pystamps_source(explicit=None):
    candidates = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.extend([
        Path("/home/ubuntu/software/pystamps-gamma"),
        Path("/home/ubuntu/software/pystamps-gamma-main"),
        Path.home() / "software" / "pystamps-gamma",
    ])

    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            continue

        if (
            (p / "pystamps").is_dir()
            and
            (p / "pystamps" / "pipeline").is_dir()
        ):
            sys.path.insert(0, str(p))
            return p

    raise RuntimeError(
        "Cannot locate pystamps-gamma; "
        "use --pystamps-source."
    )


def matlab_datenum(text):
    d = datetime.strptime(
        str(text),
        "%Y%m%d",
    )
    return float(
        d.toordinal() + 366
    )


def read_itab(path, ndate):
    pairs = []

    with Path(path).open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:
            s = line.strip()

            if (
                not s
                or s.startswith("#")
            ):
                continue

            vals = []

            for token in s.split():
                try:
                    vals.append(int(token))
                except ValueError:
                    pass

            if len(vals) < 2:
                continue

            i, j = vals[:2]

            if (
                1 <= i <= ndate
                and
                1 <= j <= ndate
                and
                i != j
            ):
                pairs.append((i, j))

    a = np.asarray(
        pairs,
        dtype=np.int64,
    )

    if (
        a.ndim != 2
        or
        a.shape[1] != 2
    ):
        raise RuntimeError(
            f"Invalid network: {path}"
        )

    return a


def network_matrix(pairs, ndate):
    G = np.zeros(
        (pairs.shape[0], ndate),
        dtype=np.float64,
    )

    r = np.arange(
        pairs.shape[0]
    )

    G[
        r,
        pairs[:, 0] - 1
    ] = -1.0

    G[
        r,
        pairs[:, 1] - 1
    ] = +1.0

    return G


def qline(
    title,
    x,
    qs=(1,5,50,95,99),
    fmt=".6e",
):
    a = np.asarray(
        x,
        dtype=np.float64,
    )

    a = a[
        np.isfinite(a)
    ]

    q = np.percentile(
        a,
        qs,
    )

    print(title)
    print(
        "  "
        +
        " / ".join(
            format(v, fmt)
            for v in q
        )
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--pystamps-source",
        default=None,
    )

    # Frozen computational reference region.
    ap.add_argument(
        "--ref-row",
        type=int,
        default=539,
    )

    ap.add_argument(
        "--ref-col",
        type=int,
        default=337,
    )

    ap.add_argument(
        "--ref-half-row",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--ref-half-col",
        type=int,
        default=15,
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
        /
        "v09"
    )

    invdir = (
        root
        /
        "network_inversion_v09"
    )

    ppsdir = (
        root
        /
        "point_phase_stack"
    )

    netdir = (
        root
        /
        "network"
    )

    r3b = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge"
        /
        "r3b_grid_adapter"
    )

    r3c2 = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge"
        /
        "r3c2_pointwise_bperp"
    )

    outdir = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge"
        /
        "r4a_stage7_contract"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Require R3c2 PASS
    # ========================================================

    r3_manifest_path = (
        r3c2
        /
        "pointwise_bperp_manifest.json"
    )

    r3_manifest = json.loads(
        r3_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        r3_manifest.get("status")
        !=
        "PASS_PRODUCTION_POINTWISE_BPERP_READY"
    ):
        raise RuntimeError(
            "Step10R3c2 is not PASS"
        )

    # ========================================================
    # Inputs
    # ========================================================

    strict_ids = np.asarray(
        np.load(
            invdir
            /
            "strict_point_ids.npy",
            mmap_mode="r",
        ),
        dtype=np.int64,
    )

    all_rows = np.load(
        ppsdir
        /
        "rows.npy",
        mmap_mode="r",
    )

    all_cols = np.load(
        ppsdir
        /
        "cols.npy",
        mmap_mode="r",
    )

    rows = np.asarray(
        all_rows[strict_ids],
        dtype=np.int32,
    )

    cols = np.asarray(
        all_cols[strict_ids],
        dtype=np.int32,
    )

    phase = np.load(
        invdir
        /
        "acquisition_phase_l2_candidate_rad.npy",
        mmap_mode="r",
    )

    lon = np.load(
        r3b
        /
        "longitude_deg.npy",
        mmap_mode="r",
    )

    lat = np.load(
        r3b
        /
        "latitude_deg.npy",
        mmap_mode="r",
    )

    local_xy = np.load(
        r3b
        /
        "local_xy_m.npy",
        mmap_mode="r",
    )

    ll0 = np.asarray(
        np.load(
            r3b
            /
            "ll0_lonlat_deg.npy"
        ),
        dtype=np.float64,
    ).reshape(-1)[:2]

    bmean = np.asarray(
        np.load(
            r3c2
            /
            "bperp_mean_by_ifg_m.npy"
        ),
        dtype=np.float64,
    )

    npoint = strict_ids.size

    dates = [
        str(x)
        for x in stack.dates
    ]

    ndate = len(dates)

    pairs = read_itab(
        netdir
        /
        "network.itab",
        ndate,
    )

    nedge = pairs.shape[0]

    if phase.shape != (
        npoint,
        ndate,
    ):
        raise RuntimeError(
            "Phase shape mismatch"
        )

    if bmean.shape != (
        nedge,
    ):
        raise RuntimeError(
            "Bperp mean shape mismatch"
        )

    for name, x in (
        ("lon", lon),
        ("lat", lat),
    ):
        if np.asarray(x).shape != (
            npoint,
        ):
            raise RuntimeError(
                f"{name} shape mismatch"
            )

    if np.asarray(
        local_xy
    ).shape != (
        npoint,
        2,
    ):
        raise RuntimeError(
            "local_xy shape mismatch"
        )

    # ========================================================
    # Import exact mature StaMPS reference selector
    # ========================================================

    pystamps_source = (
        resolve_pystamps_source(
            args.pystamps_source
        )
    )

    from pystamps.pipeline.stage6_sbas import (
        _stage6_reference_indices,
        _llh2local_m,
    )

    # ========================================================
    # Temporal metadata
    # ========================================================

    day = np.asarray(
        [
            matlab_datenum(d)
            for d in dates
        ],
        dtype=np.float64,
    )

    ifgday_ix = (
        pairs.astype(
            np.int32
        )
    )

    master_ix = 1
    master_day = day[0]

    if (
        dates[0]
        != "20141006"
    ):
        raise RuntimeError(
            "Expected frozen temporal reference "
            "20141006 at acquisition 1"
        )

    G = network_matrix(
        pairs,
        ndate,
    )

    rankG = int(
        np.linalg.matrix_rank(
            G
        )
    )

    # ========================================================
    # Stable radar-coordinate reference window
    # ========================================================

    target = (
        (rows >= args.ref_row - args.ref_half_row)
        &
        (rows <= args.ref_row + args.ref_half_row)
        &
        (cols >= args.ref_col - args.ref_half_col)
        &
        (cols <= args.ref_col + args.ref_half_col)
    )

    target_ix = np.flatnonzero(
        target
    )

    if target_ix.size == 0:
        raise RuntimeError(
            "Frozen reference window contains no strict points"
        )

    # Use nearest strict point to frozen radar centre.
    dgrid2 = (
        (
            rows.astype(np.float64)
            -
            args.ref_row
        ) ** 2
        +
        (
            cols.astype(np.float64)
            -
            args.ref_col
        ) ** 2
    )

    centre_ix = int(
        np.argmin(
            dgrid2
        )
    )

    centre_lonlat = np.asarray(
        [
            float(lon[centre_ix]),
            float(lat[centre_ix]),
        ],
        dtype=np.float64,
    )

    lonlat = np.column_stack(
        (
            np.asarray(
                lon,
                dtype=np.float64,
            ),
            np.asarray(
                lat,
                dtype=np.float64,
            ),
        )
    )

    # Use the exact metric conversion used by
    # _stage6_reference_indices.
    all_xy_stage6 = (
        _llh2local_m(
            lonlat,
            ll0,
        )
    )

    centre_xy_stage6 = (
        _llh2local_m(
            centre_lonlat.reshape(
                1,
                2,
            ),
            ll0,
        )[0]
    )

    dist = np.sqrt(
        np.sum(
            (
                all_xy_stage6
                -
                centre_xy_stage6[
                    None,
                    :
                ]
            ) ** 2,
            axis=1,
        )
    )

    outside_dist = dist[
        ~target
    ]

    if outside_dist.size == 0:
        raise RuntimeError(
            "Reference target unexpectedly covers whole scene"
        )

    # Largest circular reference that cannot include
    # any point outside the frozen stable radar window.
    first_outside = float(
        np.min(
            outside_dist
        )
    )

    ref_radius = np.nextafter(
        first_outside,
        0.0,
    )

    ps2_probe = {
        "lonlat":
            lonlat,

        "ll0":
            ll0.reshape(
                1,
                2,
            ),
    }

    parms_probe = {
        "ref_lon":
            np.asarray(
                [
                    float(
                        np.min(lon)
                    )
                    - 1e-6,
                    float(
                        np.max(lon)
                    )
                    + 1e-6,
                ],
                dtype=np.float64,
            ),

        "ref_lat":
            np.asarray(
                [
                    float(
                        np.min(lat)
                    )
                    - 1e-6,
                    float(
                        np.max(lat)
                    )
                    + 1e-6,
                ],
                dtype=np.float64,
            ),

        "ref_centre_lonlat":
            centre_lonlat,

        "ref_radius":
            float(
                ref_radius
            ),

        "ref_radius_m":
            float(
                ref_radius
            ),
    }

    ref_ix = (
        _stage6_reference_indices(
            ps2_probe,
            parms_probe,
            npoint,
        )
    )

    outside_selected = int(
        np.count_nonzero(
            ~target[
                ref_ix
            ]
        )
    )

    target_coverage = (
        ref_ix.size
        /
        target_ix.size
    )

    # ========================================================
    # Compare mature Stage7 mean reference against
    # frozen 09c target-region median gauge.
    # Diagnostic only.
    # ========================================================

    ref_mean = np.mean(
        np.asarray(
            phase[
                ref_ix,
                :
            ],
            dtype=np.float64,
        ),
        axis=0,
    )

    target_median = np.median(
        np.asarray(
            phase[
                target_ix,
                :
            ],
            dtype=np.float64,
        ),
        axis=0,
    )

    ref_gauge_delta = (
        ref_mean
        -
        target_median
    )

    ref_delta_rms = float(
        np.sqrt(
            np.mean(
                ref_gauge_delta
                *
                ref_gauge_delta
            )
        )
    )

    ref_delta_max = float(
        np.max(
            np.abs(
                ref_gauge_delta
            )
        )
    )

    # ========================================================
    # Bridge covariance:
    #
    # Current pyPSDS does not contain StaMPS Stage6
    # rc2/pm2 noise products.
    #
    # Use transparent equal-variance IFG model:
    #
    #     sb_cov = I
    #
    # Then propagate using the exact official Stage6 GLS
    # network covariance relation.
    # ========================================================

    sb_cov = np.eye(
        nedge,
        dtype=np.float64,
    )

    G_stage6 = G.copy()

    # Official Stage6:
    # G(:,master_ix)=0
    G_stage6[
        :,
        master_ix - 1
    ] = 0.0

    nzc = (
        np.sum(
            np.abs(
                G_stage6
            ),
            axis=0,
        )
        != 0
    )

    G2 = G_stage6[
        :,
        nzc,
    ]

    rank_G2 = int(
        np.linalg.matrix_rank(
            G2
        )
    )

    if rank_G2 != G2.shape[1]:
        raise RuntimeError(
            f"Stage6 bridge G2 rank "
            f"{rank_G2}/{G2.shape[1]}"
        )

    # C = I, so:
    #
    # sm_active = inv(G2' G2)
    #
    normal = (
        G2.T
        @ G2
    )

    sm_active = np.linalg.inv(
        normal
    )

    active_cols = np.flatnonzero(
        nzc
    )

    sm_cov = np.zeros(
        (
            ndate,
            ndate,
        ),
        dtype=np.float64,
    )

    sm_cov[
        np.ix_(
            active_cols,
            active_cols,
        )
    ] = sm_active

    eig_sm = np.linalg.eigvalsh(
        sm_active
    )

    cond_sm = float(
        np.linalg.cond(
            sm_active
        )
    )

    # ========================================================
    # Exact Stage7 design preflight
    # ========================================================

    dt_sb = (
        day[
            pairs[:, 1] - 1
        ]
        -
        day[
            pairs[:, 0] - 1
        ]
    )

    A1 = np.column_stack(
        (
            np.ones(
                nedge,
                dtype=np.float64,
            ),
            bmean,
            dt_sb,
        )
    )

    rank_A1 = int(
        np.linalg.matrix_rank(
            A1
        )
    )

    # Stage7 PASS3 uses all non-master images.
    img0 = np.asarray(
        [
            i
            for i in range(ndate)
            if i != master_ix - 1
        ],
        dtype=np.int64,
    )

    Gbase = G[
        :,
        img0,
    ]

    rank_Gbase = int(
        np.linalg.matrix_rank(
            Gbase
        )
    )

    Pbase = np.linalg.pinv(
        Gbase
    )

    # Mean acquisition Bperp is enough to test
    # the shared Stage7 design rank.
    bperp_some_mean = (
        bmean
        @ Pbase.T
    )

    mean_bdiff = np.diff(
        bperp_some_mean
    )

    day_seq = np.diff(
        day[
            img0
        ]
    )

    A2 = np.column_stack(
        (
            np.ones(
                mean_bdiff.size,
                dtype=np.float64,
            ),
            mean_bdiff,
            day_seq,
        )
    )

    rank_A2 = int(
        np.linalg.matrix_rank(
            A2
        )
    )

    Ac = np.column_stack(
        (
            np.ones(
                img0.size,
                dtype=np.float64,
            ),
            day[
                img0
            ]
            -
            master_day,
        )
    )

    rank_Ac = int(
        np.linalg.matrix_rank(
            Ac
        )
    )

    Csm = sm_cov[
        np.ix_(
            img0,
            img0,
        )
    ]

    eig_Csm = np.linalg.eigvalsh(
        Csm
    )

    cond_Csm = float(
        np.linalg.cond(
            Csm
        )
    )

    # ========================================================
    # Report
    # ========================================================

    print("=" * 112)
    print(
        "Step 10R4a - pyPSDS -> mature pySTAMPS "
        "Stage7 bridge contract preflight"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"pystamps source            : "
        f"{pystamps_source}"
    )

    print(
        f"strict points              : "
        f"{npoint:,}"
    )

    print(
        f"acquisitions               : "
        f"{ndate}"
    )

    print(
        f"SB IFGs                    : "
        f"{nedge}"
    )

    print(
        f"network rank               : "
        f"{rankG}/{ndate-1}"
    )

    print(
        f"master / temporal ref      : "
        f"{dates[0]} "
        f"(master_ix={master_ix})"
    )

    print()
    print("=" * 112)
    print(
        "StaMPS spatial reference adapter"
    )
    print("=" * 112)

    print(
        f"frozen radar centre        : "
        f"row={args.ref_row}, "
        f"col={args.ref_col}"
    )

    print(
        f"frozen target window       : "
        f"{2*args.ref_half_row+1} x "
        f"{2*args.ref_half_col+1}"
    )

    print(
        f"strict points in target    : "
        f"{target_ix.size}"
    )

    print(
        f"reference centre point     : "
        f"id={centre_ix}, "
        f"row={rows[centre_ix]}, "
        f"col={cols[centre_ix]}"
    )

    print(
        f"reference centre lon/lat   : "
        f"{centre_lonlat[0]:.9f}, "
        f"{centre_lonlat[1]:.9f}"
    )

    print(
        f"largest safe radius        : "
        f"{ref_radius:.3f} m"
    )

    print(
        f"Stage7 selected ref PS     : "
        f"{ref_ix.size}"
    )

    print(
        f"selected outside target    : "
        f"{outside_selected}"
    )

    print(
        f"target retained fraction   : "
        f"{target_coverage:.4f}"
    )

    print(
        f"mean-vs-target-median RMS  : "
        f"{ref_delta_rms:.6f} rad"
    )

    print(
        f"mean-vs-target-median max  : "
        f"{ref_delta_max:.6f} rad"
    )

    print()
    print("=" * 112)
    print(
        "Bridge covariance"
    )
    print("=" * 112)

    print(
        "sb_cov model               : "
        "identity (equal IFG variance)"
    )

    print(
        "reason                     : "
        "pyPSDS has no StaMPS rc2/pm2 "
        "Stage6 noise products"
    )

    print(
        f"sb_cov shape               : "
        f"{sb_cov.shape}"
    )

    print(
        f"Stage6 G2 rank             : "
        f"{rank_G2}/{G2.shape[1]}"
    )

    print(
        f"sm_cov shape               : "
        f"{sm_cov.shape}"
    )

    print(
        f"sm_active eig min/max      : "
        f"{eig_sm.min():.9e} / "
        f"{eig_sm.max():.9e}"
    )

    print(
        f"sm_active condition        : "
        f"{cond_sm:.6f}"
    )

    print()
    print("=" * 112)
    print(
        "Stage7 design matrices"
    )
    print("=" * 112)

    print(
        f"PASS1 A=[1,Bperp,dt]       : "
        f"shape={A1.shape}, "
        f"rank={rank_A1}/3"
    )

    print(
        f"PASS3 Gbase               : "
        f"shape={Gbase.shape}, "
        f"rank={rank_Gbase}/"
        f"{img0.size}"
    )

    print(
        f"PASS3 A=[1,dBperp,dt]      : "
        f"shape={A2.shape}, "
        f"rank={rank_A2}/3"
    )

    print(
        f"C fit A=[1,dt]             : "
        f"shape={Ac.shape}, "
        f"rank={rank_Ac}/2"
    )

    print(
        f"Csm eig min/max            : "
        f"{eig_Csm.min():.9e} / "
        f"{eig_Csm.max():.9e}"
    )

    print(
        f"Csm condition              : "
        f"{cond_Csm:.6f}"
    )

    qline(
        "mean IFG Bperp "
        "p01/p05/p50/p95/p99 [m]:",
        bmean,
        fmt=".3f",
    )

    qline(
        "sequential mean dBperp "
        "p01/p05/p50/p95/p99 [m]:",
        mean_bdiff,
        fmt=".3f",
    )

    # ========================================================
    # Decision
    # ========================================================

    if outside_selected != 0:

        status = (
            "REVIEW_REFERENCE_LEAKAGE"
        )

    elif ref_ix.size < 50:

        status = (
            "REVIEW_TOO_FEW_REFERENCE_POINTS"
        )

    elif rank_A1 != 3:

        status = (
            "REVIEW_STAGE7_PASS1_DESIGN"
        )

    elif rank_Gbase != img0.size:

        status = (
            "REVIEW_STAGE7_BASELINE_RANK"
        )

    elif rank_A2 != 3:

        status = (
            "REVIEW_STAGE7_PASS3_DESIGN"
        )

    elif rank_Ac != 2:

        status = (
            "REVIEW_STAGE7_C_DESIGN"
        )

    elif (
        eig_Csm.min() <= 0
        or
        not np.isfinite(
            cond_Csm
        )
    ):

        status = (
            "REVIEW_STAGE7_SM_COV"
        )

    else:

        status = (
            "PASS_STAGE7_BRIDGE_CONTRACT_READY"
        )

    np.save(
        outdir
        /
        "stage7_reference_point_indices.npy",
        ref_ix.astype(
            np.int64
        ),
    )

    np.save(
        outdir
        /
        "bridge_sb_cov_identity.npy",
        sb_cov,
    )

    np.save(
        outdir
        /
        "bridge_sm_cov_unit_ifg.npy",
        sm_cov,
    )

    bridge_parameters = {
        "small_baseline_flag":
            "y",

        "scla_method":
            "L2",

        "scla_deramp":
            "n",

        "subtr_tropo":
            "n",

        "drop_ifg_index":
            [],

        "sb_scla_drop_index":
            [],

        "scla_drop_index":
            [],

        "ref_lon":
            parms_probe[
                "ref_lon"
            ].tolist(),

        "ref_lat":
            parms_probe[
                "ref_lat"
            ].tolist(),

        "ref_centre_lonlat":
            centre_lonlat.tolist(),

        "ref_radius_m":
            float(
                ref_radius
            ),
    }

    manifest = {
        "format":
            "pyPSDS-GAMMA-pystamps-stage7-bridge-contract-v09",

        "status":
            status,

        "points":
            int(
                npoint
            ),

        "images":
            int(
                ndate
            ),

        "ifgs":
            int(
                nedge
            ),

        "master": {
            "ix_1based":
                int(
                    master_ix
                ),

            "date":
                dates[0],

            "day":
                float(
                    master_day
                ),
        },

        "reference": {
            "radar_centre":
                [
                    int(
                        args.ref_row
                    ),
                    int(
                        args.ref_col
                    ),
                ],

            "target_points":
                int(
                    target_ix.size
                ),

            "stage7_reference_points":
                int(
                    ref_ix.size
                ),

            "outside_target":
                int(
                    outside_selected
                ),

            "radius_m":
                float(
                    ref_radius
                ),

            "centre_lonlat":
                centre_lonlat.tolist(),

            "mean_vs_target_median_rms_rad":
                ref_delta_rms,

            "mean_vs_target_median_max_rad":
                ref_delta_max,
        },

        "covariance": {
            "sb_model":
                "identity_equal_ifg_variance",

            "reason":
                (
                    "pyPSDS bypasses StaMPS Stage6 "
                    "and therefore has no rc2/pm2 "
                    "noise covariance products"
                ),

            "sm_model":
                (
                    "official Stage6 GLS propagation "
                    "from identity SB covariance"
                ),

            "sm_condition":
                cond_Csm,
        },

        "stage7_design": {
            "pass1_rank":
                int(
                    rank_A1
                ),

            "pass3_network_rank":
                int(
                    rank_Gbase
                ),

            "pass3_rank":
                int(
                    rank_A2
                ),

            "constant_fit_rank":
                int(
                    rank_Ac
                ),
        },

        "bridge_parameters":
            bridge_parameters,

        "mat_files_generated":
            False,

        "stage7_executed":
            False,

        "phase_modified":
            False,
    }

    manifest_path = (
        outdir
        /
        "stage7_bridge_contract_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10R4a STATUS: "
        f"{status}"
    )

    print(
        "No Stage7 MAT bridge dataset "
        "was created."
    )

    print(
        "No SCLA/APS correction was applied."
    )


if __name__ == "__main__":
    main()
