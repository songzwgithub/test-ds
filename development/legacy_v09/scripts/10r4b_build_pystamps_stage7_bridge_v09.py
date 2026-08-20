#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pypsds.prototype import open_from_config


# ============================================================
# Helpers
# ============================================================

def resolve_pystamps_source(explicit=None):

    candidates = []

    if explicit:
        candidates.append(
            Path(explicit).expanduser()
        )

    candidates.extend([
        Path(
            "/home/ubuntu/software/pystamps-gamma"
        ),
        Path(
            "/home/ubuntu/software/pystamps-gamma-main"
        ),
        Path.home()
        /
        "software"
        /
        "pystamps-gamma",
    ])

    seen = set()

    for p in candidates:

        try:
            p = p.resolve()
        except Exception:
            continue

        if p in seen:
            continue

        seen.add(p)

        if (
            (p / "pystamps").is_dir()
            and
            (
                p
                /
                "pystamps"
                /
                "pipeline"
                /
                "stage7_sbas.py"
            ).is_file()
        ):

            sys.path.insert(
                0,
                str(p),
            )

            return p

    raise RuntimeError(
        "Cannot locate pystamps-gamma. "
        "Use --pystamps-source."
    )


def matlab_datenum_to_yyyymmdd(value):

    x = float(value)

    ordinal = (
        int(
            np.floor(x)
        )
        -
        366
    )

    return (
        datetime
        .fromordinal(
            ordinal
        )
        .strftime(
            "%Y%m%d"
        )
    )


def read_itab(
    path,
    ndate,
):

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
                or
                s.startswith("#")
            ):
                continue

            vals = []

            for token in s.split():

                try:
                    vals.append(
                        int(token)
                    )

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

                pairs.append(
                    (i, j)
                )

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
            f"Invalid network.itab: "
            f"{path}"
        )

    return a


def network_matrix(
    pairs,
    ndate,
):

    G = np.zeros(
        (
            pairs.shape[0],
            ndate,
        ),
        dtype=np.float64,
    )

    rr = np.arange(
        pairs.shape[0],
        dtype=np.int64,
    )

    G[
        rr,
        pairs[:, 0] - 1,
    ] = -1.0

    G[
        rr,
        pairs[:, 1] - 1,
    ] = +1.0

    return G


def mat_scalar(
    value,
):

    return np.asarray(
        [[value]],
        dtype=np.float64,
    )


def file_mib(path):

    return (
        Path(path).stat().st_size
        /
        1024.0
        /
        1024.0
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
        "--pystamps-source",
        default=None,
    )

    ap.add_argument(
        "--old-pystamps-root",
        default=(
            "/home/ubuntu/Downloads/pystamps"
        ),
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=32768,
    )

    ap.add_argument(
        "--rebuild",
        action="store_true",
    )

    args = ap.parse_args()

    # ========================================================
    # Resolve pyPSDS project
    # ========================================================

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

    r4a = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge"
        /
        "r4a_stage7_contract"
    )

    bridge_root = (
        root
        /
        "pystamps_bridge_v09"
    )

    # ========================================================
    # Require frozen upstream PASS
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

    r4_manifest_path = (
        r4a
        /
        "stage7_bridge_contract_manifest.json"
    )

    r4_manifest = json.loads(
        r4_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        r4_manifest.get("status")
        !=
        "PASS_STAGE7_BRIDGE_CONTRACT_READY"
    ):

        raise RuntimeError(
            "Step10R4a is not PASS"
        )

    # ========================================================
    # Bridge directory isolation
    # ========================================================

    if bridge_root.exists():

        if not args.rebuild:

            raise RuntimeError(
                f"{bridge_root} already exists. "
                "Use --rebuild to archive it "
                "and create a new bridge."
            )

        stamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        backup = (
            root
            /
            (
                "pystamps_bridge_v09_backup_"
                +
                stamp
            )
        )

        bridge_root.rename(
            backup
        )

        print(
            f"Existing bridge archived   : "
            f"{backup}"
        )

    bridge_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    workdir = (
        bridge_root
        /
        "_raw_work"
    )

    workdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Mature pySTAMPS imports
    # ========================================================

    pystamps_source = (
        resolve_pystamps_source(
            args.pystamps_source
        )
    )

    from pystamps.io.mat import (
        write_mat,
        read_mat,
        read_mat_variables,
    )

    from pystamps.pipeline.stage6_sbas import (
        _stage6_reference_indices,
    )

    from pystamps.pipeline.stage7_sbas import (
        _preflight,
    )

    # ========================================================
    # Current pyPSDS data
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
        all_rows[
            strict_ids
        ],
        dtype=np.int32,
    )

    cols = np.asarray(
        all_cols[
            strict_ids
        ],
        dtype=np.int32,
    )

    phase09 = np.load(
        invdir
        /
        "acquisition_phase_l2_candidate_rad.npy",
        mmap_mode="r",
    )

    lon = np.asarray(
        np.load(
            r3b
            /
            "longitude_deg.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    lat = np.asarray(
        np.load(
            r3b
            /
            "latitude_deg.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    local_xy = np.asarray(
        np.load(
            r3b
            /
            "local_xy_m.npy",
            mmap_mode="r",
        ),
        dtype=np.float32,
    )

    ll0 = np.asarray(
        np.load(
            r3b
            /
            "ll0_lonlat_deg.npy"
        ),
        dtype=np.float64,
    ).reshape(-1)[:2]

    look_angle = np.load(
        r3b
        /
        "look_angle_rad.npy",
        mmap_mode="r",
    )

    slant_range = np.load(
        r3b
        /
        "slant_range_m.npy",
        mmap_mode="r",
    )

    ref_ix_expected = np.asarray(
        np.load(
            r4a
            /
            "stage7_reference_point_indices.npy"
        ),
        dtype=np.int64,
    )

    bmean = np.asarray(
        np.load(
            r3c2
            /
            "bperp_mean_by_ifg_m.npy"
        ),
        dtype=np.float64,
    )

    b_layout = json.loads(
        (
            r3c2
            /
            "bperp_ifg_layout.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    bperp_raw_path = Path(
        b_layout["file"]
    )

    npoint = strict_ids.size

    dates = [
        str(x)
        for x in stack.dates
    ]

    ndate = len(
        dates
    )

    pairs = read_itab(
        netdir
        /
        "network.itab",
        ndate,
    )

    nedge = pairs.shape[0]

    if phase09.shape != (
        npoint,
        ndate,
    ):

        raise RuntimeError(
            "Step09 phase shape mismatch"
        )

    if bmean.shape != (
        nedge,
    ):

        raise RuntimeError(
            "Bperp mean shape mismatch"
        )

    expected_b_shape = (
        npoint,
        nedge,
    )

    if tuple(
        b_layout["shape"]
    ) != expected_b_shape:

        raise RuntimeError(
            "Bperp layout shape mismatch"
        )

    expected_b_bytes = (
        npoint
        *
        nedge
        *
        np.dtype(
            np.float32
        ).itemsize
    )

    if (
        not bperp_raw_path.is_file()
        or
        bperp_raw_path.stat().st_size
        != expected_b_bytes
    ):

        raise RuntimeError(
            "Point-wise Bperp backing file "
            "is missing or has wrong size"
        )

    # ========================================================
    # Recover mature pySTAMPS master semantics
    # ========================================================

    old_ps_path = (
        Path(
            args.old_pystamps_root
        )
        /
        "ps2.mat"
    )

    old = loadmat(
        old_ps_path,
        squeeze_me=False,
        struct_as_record=False,
    )

    old_day = np.asarray(
        old["day"],
        dtype=np.float64,
    ).reshape(-1)

    old_dates = [
        matlab_datenum_to_yyyymmdd(
            d
        )
        for d in old_day
    ]

    if (
        old_dates
        !=
        dates
    ):

        raise RuntimeError(
            "Old mature pySTAMPS dates/order "
            "do not match current stack"
        )

    master_ix = int(
        round(
            float(
                np.asarray(
                    old["master_ix"]
                ).reshape(-1)[0]
            )
        )
    )

    master_day = float(
        np.asarray(
            old["master_day"]
        ).reshape(-1)[0]
    )

    if not (
        1 <= master_ix <= ndate
    ):

        raise RuntimeError(
            "Invalid mature master_ix"
        )

    master_idx0 = (
        master_ix - 1
    )

    master_date = dates[
        master_idx0
    ]

    if (
        matlab_datenum_to_yyyymmdd(
            master_day
        )
        !=
        master_date
    ):

        raise RuntimeError(
            "master_day/master_ix mismatch"
        )

    if master_date != "20150709":

        raise RuntimeError(
            "Expected mature pySTAMPS "
            "master 20150709"
        )

    # ========================================================
    # Exact Stage7 reference parameters from R4a
    # ========================================================

    bridge_params = (
        r4_manifest[
            "bridge_parameters"
        ]
    )

    ref_lon = np.asarray(
        bridge_params[
            "ref_lon"
        ],
        dtype=np.float64,
    )

    ref_lat = np.asarray(
        bridge_params[
            "ref_lat"
        ],
        dtype=np.float64,
    )

    ref_centre_lonlat = np.asarray(
        bridge_params[
            "ref_centre_lonlat"
        ],
        dtype=np.float64,
    )

    ref_radius = float(
        bridge_params[
            "ref_radius_m"
        ]
    )

    lonlat = np.column_stack(
        (
            lon,
            lat,
        )
    )

    # ========================================================
    # Construct ps2 probe and verify reference selection
    # ========================================================

    ps_probe = {
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
            ref_lon,

        "ref_lat":
            ref_lat,

        "ref_centre_lonlat":
            ref_centre_lonlat,

        "ref_radius":
            ref_radius,

        "ref_radius_m":
            ref_radius,
    }

    ref_ix = (
        _stage6_reference_indices(
            ps_probe,
            parms_probe,
            npoint,
        )
    )

    if not np.array_equal(
        ref_ix,
        ref_ix_expected,
    ):

        raise RuntimeError(
            "Reference selection changed "
            "between R4a and R4b"
        )

    # ========================================================
    # Network and bridge covariance
    #
    # IMPORTANT:
    # Recompute for mature master_ix=20.
    # Do NOT reuse R4a master_ix=1 sm_cov.
    # ========================================================

    G = network_matrix(
        pairs,
        ndate,
    )

    rank_G = int(
        np.linalg.matrix_rank(
            G
        )
    )

    if rank_G != ndate - 1:

        raise RuntimeError(
            f"Network rank {rank_G}/"
            f"{ndate-1}"
        )

    sb_cov = np.eye(
        nedge,
        dtype=np.float64,
    )

    G_stage6 = G.copy()

    G_stage6[
        :,
        master_idx0
    ] = 0.0

    nzc = (
        np.sum(
            np.abs(
                G_stage6
            ),
            axis=0,
        )
        !=
        0
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
            f"Stage6 covariance G2 rank "
            f"{rank_G2}/"
            f"{G2.shape[1]}"
        )

    # C_SB = I:
    #
    # sm_active =
    #   inv(G2' C^-1 G2)
    # = inv(G2' G2)
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

    img0 = np.asarray(
        [
            i
            for i in range(
                ndate
            )
            if i != master_idx0
        ],
        dtype=np.int64,
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

    if eig_Csm.min() <= 0:

        raise RuntimeError(
            "Bridge sm_cov is not "
            "positive definite on "
            "non-master acquisitions"
        )

    # ========================================================
    # Build bridge phase
    #
    # Step 1:
    #   mature spatial reference using 238 ref points
    #
    # Step 2:
    #   temporal gauge to mature master 20150709
    #
    # Canonical Step09a is untouched.
    # ========================================================

    print("=" * 112)
    print(
        "Step 10R4b - build isolated mature "
        "pySTAMPS Stage7 bridge"
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
        f"bridge root                : "
        f"{bridge_root}"
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
        f"mature master              : "
        f"{master_date}"
    )

    print(
        f"master_ix                  : "
        f"{master_ix}"
    )

    print(
        f"reference PS               : "
        f"{ref_ix.size}"
    )

    print()
    print("=" * 112)
    print(
        "Bridge phase gauge"
    )
    print("=" * 112)

    ref_mean09 = np.mean(
        np.asarray(
            phase09[
                ref_ix,
                :
            ],
            dtype=np.float64,
        ),
        axis=0,
    )

    phase_raw = (
        workdir
        /
        "phuw2_bridge.float32.dat"
    )

    ph_sm = np.memmap(
        phase_raw,
        mode="w+",
        dtype=np.float32,
        shape=(
            npoint,
            ndate,
        ),
    )

    t0 = time.perf_counter()

    for start in range(
        0,
        npoint,
        args.batch_size,
    ):

        stop = min(
            start + args.batch_size,
            npoint,
        )

        y = np.asarray(
            phase09[
                start:stop,
                :
            ],
            dtype=np.float64,
        )

        # Mature spatial reference.
        y -= (
            ref_mean09[
                None,
                :
            ]
        )

        # Mature temporal master gauge.
        master_col = (
            y[
                :,
                master_idx0
            ].copy()
        )

        y -= (
            master_col[
                :,
                None
            ]
        )

        ph_sm[
            start:stop,
            :
        ] = y.astype(
            np.float32
        )

        print(
            f"[bridge phuw2] "
            f"{stop:,}/{npoint:,} "
            f"({100*stop/npoint:.1f}%)",
            flush=True,
        )

    ph_sm.flush()

    master_max = float(
        np.max(
            np.abs(
                np.asarray(
                    ph_sm[
                        :,
                        master_idx0
                    ],
                    dtype=np.float64,
                )
            )
        )
    )

    ref_mean_bridge = np.mean(
        np.asarray(
            ph_sm[
                ref_ix,
                :
            ],
            dtype=np.float64,
        ),
        axis=0,
    )

    ref_mean_max = float(
        np.max(
            np.abs(
                ref_mean_bridge
            )
        )
    )

    print(
        f"master column max |phase|  : "
        f"{master_max:.9e} rad"
    )

    print(
        f"reference mean max |phase| : "
        f"{ref_mean_max:.9e} rad"
    )

    if master_max > 1e-7:

        raise RuntimeError(
            "Bridge master column is not zero"
        )

    if ref_mean_max > 5e-6:

        raise RuntimeError(
            "Bridge reference mean is not zero"
        )

    # ========================================================
    # Build virtual spatially-referenced SB IFGs
    # ========================================================

    ii = (
        pairs[:, 0]
        -
        1
    )

    jj = (
        pairs[:, 1]
        -
        1
    )

    sb_raw = (
        workdir
        /
        "phuw_sb2_bridge.float32.dat"
    )

    ph_sb = np.memmap(
        sb_raw,
        mode="w+",
        dtype=np.float32,
        shape=(
            npoint,
            nedge,
        ),
    )

    for start in range(
        0,
        npoint,
        args.batch_size,
    ):

        stop = min(
            start + args.batch_size,
            npoint,
        )

        y = np.asarray(
            ph_sm[
                start:stop,
                :
            ],
            dtype=np.float32,
        )

        ph_sb[
            start:stop,
            :
        ] = (
            y[
                :,
                jj
            ]
            -
            y[
                :,
                ii
            ]
        ).astype(
            np.float32
        )

        print(
            f"[bridge phuw_sb2] "
            f"{stop:,}/{npoint:,} "
            f"({100*stop/npoint:.1f}%)",
            flush=True,
        )

    ph_sb.flush()

    ref_mean_sb = np.mean(
        np.asarray(
            ph_sb[
                ref_ix,
                :
            ],
            dtype=np.float64,
        ),
        axis=0,
    )

    ref_sb_max = float(
        np.max(
            np.abs(
                ref_mean_sb
            )
        )
    )

    print(
        f"SB reference mean max      : "
        f"{ref_sb_max:.9e} rad"
    )

    if ref_sb_max > 1e-5:

        raise RuntimeError(
            "Bridge SB reference mean "
            "is not zero"
        )

    # ========================================================
    # Bperp production memmap
    # ========================================================

    bperp = np.memmap(
        bperp_raw_path,
        mode="r",
        dtype=np.float32,
        shape=(
            npoint,
            nedge,
        ),
        order="C",
    )

    # ========================================================
    # ps2.mat
    # ========================================================

    point_id = np.arange(
        1,
        npoint + 1,
        dtype=np.float32,
    )

    xy = np.empty(
        (
            npoint,
            3,
        ),
        dtype=np.float32,
    )

    xy[:, 0] = point_id
    xy[:, 1:3] = local_xy

    ij = np.empty(
        (
            npoint,
            3,
        ),
        dtype=np.float64,
    )

    ij[:, 0] = np.arange(
        1,
        npoint + 1,
        dtype=np.float64,
    )

    # StaMPS pixel indices are conventionally 1-based.
    ij[:, 1] = (
        rows.astype(
            np.float64
        )
        +
        1.0
    )

    ij[:, 2] = (
        cols.astype(
            np.float64
        )
        +
        1.0
    )

    mean_incidence = float(
        np.mean(
            np.asarray(
                look_angle,
                dtype=np.float64,
            )
        )
    )

    mean_range = float(
        np.mean(
            np.asarray(
                slant_range,
                dtype=np.float64,
            )
        )
    )

    ps_payload = {
        "bperp":
            bmean.astype(
                np.float32
            ).reshape(
                -1,
                1,
            ),

        "day":
            old_day.reshape(
                -1,
                1,
            ),

        "ij":
            ij,

        "ll0":
            ll0.reshape(
                1,
                2,
            ),

        "lonlat":
            lonlat,

        "master_day":
            mat_scalar(
                master_day
            ),

        "master_ix":
            mat_scalar(
                master_ix
            ),

        "n_ifg":
            mat_scalar(
                nedge
            ),

        "n_image":
            mat_scalar(
                ndate
            ),

        "n_ps":
            mat_scalar(
                npoint
            ),

        "xy":
            xy,

        "mean_incidence":
            mat_scalar(
                mean_incidence
            ),

        "mean_range":
            mat_scalar(
                mean_range
            ),

        "ifgday_ix":
            pairs.astype(
                np.int32
            ),
    }

    # ========================================================
    # parms.mat
    # ========================================================

    empty_index = np.empty(
        (0, 0),
        dtype=np.float64,
    )

    parms_payload = {
        "small_baseline_flag":
            "y",

        "scla_method":
            "L2",

        "scla_deramp":
            "n",

        "subtr_tropo":
            "n",

        "drop_ifg_index":
            empty_index,

        "sb_scla_drop_index":
            empty_index,

        "scla_drop_index":
            empty_index,

        "ref_lon":
            ref_lon.reshape(
                1,
                2,
            ),

        "ref_lat":
            ref_lat.reshape(
                1,
                2,
            ),

        "ref_centre_lonlat":
            ref_centre_lonlat.reshape(
                1,
                2,
            ),

        "ref_radius":
            mat_scalar(
                ref_radius
            ),

        "ref_radius_m":
            mat_scalar(
                ref_radius
            ),
    }

    # ========================================================
    # Write Stage7 bridge MAT files
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Writing Stage7 compatibility MAT files"
    )
    print("=" * 112)

    def save_mat(
        name,
        payload,
    ):

        path = (
            bridge_root
            /
            name
        )

        started = (
            time.perf_counter()
        )

        write_mat(
            path,
            payload,
        )

        elapsed = (
            time.perf_counter()
            -
            started
        )

        print(
            f"{name:24s} "
            f"{file_mib(path):9.1f} MiB "
            f"{elapsed:8.1f} s"
        )

        return path

    ps_path = save_mat(
        "ps2.mat",
        ps_payload,
    )

    parms_path = save_mat(
        "parms.mat",
        parms_payload,
    )

    bp_path = save_mat(
        "bp2.mat",
        {
            "bperp_mat":
                bperp,
        },
    )

    phuw2_path = save_mat(
        "phuw2.mat",
        {
            "ph_uw":
                ph_sm,

            "unwrap_ifg_index_sm":
                np.arange(
                    1,
                    ndate + 1,
                    dtype=np.float64,
                ).reshape(
                    1,
                    -1,
                ),
        },
    )

    phuw_sb_path = save_mat(
        "phuw_sb2.mat",
        {
            "ph_uw":
                ph_sb,
        },
    )

    cov_path = save_mat(
        "phuw_sb_res2.mat",
        {
            "sb_cov":
                sb_cov,

            "sm_cov":
                sm_cov,
        },
    )

    # ========================================================
    # Read-back contract checks
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Bridge read-back validation"
    )
    print("=" * 112)

    ps_check = read_mat(
        ps_path
    )

    ref_check = (
        _stage6_reference_indices(
            ps_check,
            read_mat(
                parms_path
            ),
            npoint,
        )
    )

    print(
        f"reference PS read-back     : "
        f"{ref_check.size}"
    )

    print(
        f"reference set exact        : "
        f"{np.array_equal(ref_check, ref_ix)}"
    )

    if not np.array_equal(
        ref_check,
        ref_ix,
    ):

        raise RuntimeError(
            "Reference selection changed "
            "after MAT serialization"
        )

    bp_check = read_mat_variables(
        bp_path,
        (
            "bperp_mat",
        ),
    )

    phuw2_check = read_mat_variables(
        phuw2_path,
        (
            "ph_uw",
            "unwrap_ifg_index_sm",
        ),
    )

    sb_check = read_mat_variables(
        phuw_sb_path,
        (
            "ph_uw",
        ),
    )

    cov_check = read_mat_variables(
        cov_path,
        (
            "sb_cov",
            "sm_cov",
        ),
    )

    print(
        f"bp2 shape                  : "
        f"{np.asarray(bp_check['bperp_mat']).shape}"
    )

    print(
        f"phuw2 shape                : "
        f"{np.asarray(phuw2_check['ph_uw']).shape}"
    )

    print(
        f"phuw_sb2 shape             : "
        f"{np.asarray(sb_check['ph_uw']).shape}"
    )

    print(
        f"sb_cov shape               : "
        f"{np.asarray(cov_check['sb_cov']).shape}"
    )

    print(
        f"sm_cov shape               : "
        f"{np.asarray(cov_check['sm_cov']).shape}"
    )

    # Free read-back large arrays before native preflight.
    del bp_check
    del phuw2_check
    del sb_check
    del cov_check
    del ps_check

    gc.collect()

    # ========================================================
    # Native mature Stage7 preflight
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Calling mature pySTAMPS "
        "stage7_sbas._preflight()"
    )
    print("=" * 112)

    _preflight(
        bridge_root,
        phase_file="phuw2.mat",
    )

    # If we get here, mature Stage7 itself accepted
    # the complete bridge dataset.
    status = (
        "PASS_STAGE7_NATIVE_PREFLIGHT"
    )

    # ========================================================
    # Manifest
    # ========================================================

    manifest = {
        "format":
            (
                "pyPSDS-GAMMA-"
                "pystamps-stage7-bridge-v09"
            ),

        "status":
            status,

        "canonical_step09_modified":
            False,

        "bridge_phase": {
            "source":
                str(
                    invdir
                    /
                    (
                        "acquisition_phase_"
                        "l2_candidate_rad.npy"
                    )
                ),

            "spatial_reference":
                (
                    "mean over mature StaMPS "
                    "reference subset"
                ),

            "reference_points":
                int(
                    ref_ix.size
                ),

            "temporal_gauge":
                master_date,

            "master_ix_1based":
                int(
                    master_ix
                ),

            "master_column_max_abs_rad":
                master_max,

            "reference_mean_max_abs_rad":
                ref_mean_max,

            "sb_reference_mean_max_abs_rad":
                ref_sb_max,
        },

        "network": {
            "images":
                int(
                    ndate
                ),

            "ifgs":
                int(
                    nedge
                ),

            "rank":
                int(
                    rank_G
                ),
        },

        "covariance": {
            "sb_model":
                "identity_equal_variance",

            "sm_model":
                (
                    "official Stage6 GLS "
                    "network propagation"
                ),

            "master_ix_1based":
                int(
                    master_ix
                ),

            "nonmaster_eigen_min":
                float(
                    eig_Csm.min()
                ),

            "nonmaster_eigen_max":
                float(
                    eig_Csm.max()
                ),

            "nonmaster_condition":
                float(
                    cond_Csm
                ),
        },

        "files": {
            "ps2.mat":
                str(
                    ps_path
                ),

            "parms.mat":
                str(
                    parms_path
                ),

            "bp2.mat":
                str(
                    bp_path
                ),

            "phuw2.mat":
                str(
                    phuw2_path
                ),

            "phuw_sb2.mat":
                str(
                    phuw_sb_path
                ),

            "phuw_sb_res2.mat":
                str(
                    cov_path
                ),
        },

        "pointwise_bperp_source":
            str(
                bperp_raw_path
            ),

        "stage7_native_preflight":
            "PASS",

        "stage7_executed":
            False,

        "scla_applied":
            False,

        "aps_applied":
            False,
    }

    manifest_path = (
        bridge_root
        /
        "bridge_manifest.json"
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

    # ========================================================
    # Remove temporary raw phase/IFG backing files
    #
    # Compatibility MAT files remain.
    # Canonical R3c2 Bperp backing file remains.
    # ========================================================

    del ph_sm
    del ph_sb
    del bperp

    gc.collect()

    try:
        phase_raw.unlink()
    except FileNotFoundError:
        pass

    try:
        sb_raw.unlink()
    except FileNotFoundError:
        pass

    try:
        workdir.rmdir()
    except OSError:
        pass

    print()
    print("=" * 112)
    print(
        "Final Stage7 bridge"
    )
    print("=" * 112)

    for name in (
        "ps2.mat",
        "parms.mat",
        "bp2.mat",
        "phuw2.mat",
        "phuw_sb2.mat",
        "phuw_sb_res2.mat",
    ):

        p = (
            bridge_root
            /
            name
        )

        print(
            f"{name:24s} "
            f"{file_mib(p):9.1f} MiB"
        )

    print()
    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10R4b STATUS: "
        f"{status}"
    )

    print(
        "Mature pySTAMPS Stage7 "
        "accepted the bridge dataset."
    )

    print(
        "Stage7 SCLA has NOT been executed."
    )

    print(
        "Canonical pyPSDS Step09 products "
        "were NOT modified."
    )


if __name__ == "__main__":
    main()
