#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pypsds.prototype import open_from_config


def matlab_datenum_to_yyyymmdd(value):
    """
    MATLAB datenum:
      Python ordinal = floor(datenum) - 366
    """
    x = float(value)

    ordinal = int(np.floor(x)) - 366

    dt = datetime.fromordinal(
        ordinal
    )

    return dt.strftime(
        "%Y%m%d"
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

    return np.asarray(
        pairs,
        dtype=np.int64,
    )


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


def stage7_pass3_design(
    *,
    master_idx0,
    G,
    day,
    bmean,
):

    ndate = day.size

    img0 = np.asarray(
        [
            i
            for i in range(ndate)
            if i != master_idx0
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

    bsome_mean = (
        bmean
        @ Pbase.T
    )

    db = np.diff(
        bsome_mean
    )

    dt = np.diff(
        day[
            img0
        ]
    )

    A2 = np.column_stack(
        (
            np.ones(
                db.size,
                dtype=np.float64,
            ),
            db,
            dt,
        )
    )

    rank_A2 = int(
        np.linalg.matrix_rank(
            A2
        )
    )

    # Standardised condition number, so Bperp/day units
    # do not dominate comparison artificially.
    X = A2.copy()

    for j in (1, 2):

        sd = np.std(
            X[:, j]
        )

        if sd > 0:

            X[:, j] = (
                X[:, j]
                -
                np.mean(
                    X[:, j]
                )
            ) / sd

    cond_scaled = float(
        np.linalg.cond(
            X
        )
    )

    return {
        "img0":
            img0,

        "rank_Gbase":
            rank_Gbase,

        "rank_A2":
            rank_A2,

        "condition_scaled":
            cond_scaled,

        "db":
            db,

        "dt":
            dt,
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--old-pystamps-root",
        default="/home/ubuntu/Downloads/pystamps",
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

    netdir = (
        root
        /
        "network"
    )

    bdir = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge"
        /
        "r3c2_pointwise_bperp"
    )

    old_ps_path = (
        Path(
            args.old_pystamps_root
        )
        /
        "ps2.mat"
    )

    if not old_ps_path.is_file():

        raise RuntimeError(
            f"Missing old pySTAMPS ps2.mat: "
            f"{old_ps_path}"
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

    old_master_ix = int(
        round(
            float(
                np.asarray(
                    old["master_ix"]
                ).reshape(-1)[0]
            )
        )
    )

    old_master_day = float(
        np.asarray(
            old["master_day"]
        ).reshape(-1)[0]
    )

    old_dates = [
        matlab_datenum_to_yyyymmdd(
            d
        )
        for d in old_day
    ]

    current_dates = [
        str(x)
        for x in stack.dates
    ]

    ndate = len(
        current_dates
    )

    if old_master_ix < 1 or old_master_ix > len(
        old_dates
    ):

        raise RuntimeError(
            f"Invalid old master_ix="
            f"{old_master_ix}"
        )

    old_master_date_by_ix = (
        old_dates[
            old_master_ix - 1
        ]
    )

    old_master_date_by_day = (
        matlab_datenum_to_yyyymmdd(
            old_master_day
        )
    )

    exact_order_match = (
        old_dates
        ==
        current_dates
    )

    same_date_set = (
        set(
            old_dates
        )
        ==
        set(
            current_dates
        )
    )

    old_master_in_current = (
        old_master_date_by_ix
        in current_dates
    )

    if old_master_in_current:

        mapped_master_idx0 = (
            current_dates.index(
                old_master_date_by_ix
            )
        )

    else:

        mapped_master_idx0 = None

    # ========================================================
    # Current network / Bperp mean
    # ========================================================

    pairs = read_itab(
        netdir
        /
        "network.itab",
        ndate,
    )

    G = network_matrix(
        pairs,
        ndate,
    )

    bmean = np.asarray(
        np.load(
            bdir
            /
            "bperp_mean_by_ifg_m.npy"
        ),
        dtype=np.float64,
    )

    if bmean.shape != (
        pairs.shape[0],
    ):

        raise RuntimeError(
            "Bperp mean/network mismatch"
        )

    day = np.asarray(
        [
            matlab_datenum(
                d
            )
            for d in current_dates
        ],
        dtype=np.float64,
    )

    current_gauge_master = 0

    design_gauge = (
        stage7_pass3_design(
            master_idx0=(
                current_gauge_master
            ),
            G=G,
            day=day,
            bmean=bmean,
        )
    )

    design_old = None

    if mapped_master_idx0 is not None:

        design_old = (
            stage7_pass3_design(
                master_idx0=(
                    mapped_master_idx0
                ),
                G=G,
                day=day,
                bmean=bmean,
            )
        )

    print("=" * 112)
    print(
        "Step 10R4a1 - mature pySTAMPS "
        "master semantics audit"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"old ps2.mat                : "
        f"{old_ps_path}"
    )

    print()
    print("=" * 112)
    print(
        "Acquisition-date parity"
    )
    print("=" * 112)

    print(
        f"current acquisitions       : "
        f"{len(current_dates)}"
    )

    print(
        f"old pySTAMPS acquisitions : "
        f"{len(old_dates)}"
    )

    print(
        f"exact date/order match     : "
        f"{exact_order_match}"
    )

    print(
        f"same acquisition set       : "
        f"{same_date_set}"
    )

    if not exact_order_match:

        print()
        print(
            "First date-order differences:"
        )

        shown = 0

        for i in range(
            min(
                len(old_dates),
                len(current_dates),
            )
        ):

            if (
                old_dates[i]
                !=
                current_dates[i]
            ):

                print(
                    f"  {i+1:2d}: "
                    f"current={current_dates[i]}  "
                    f"old={old_dates[i]}"
                )

                shown += 1

                if shown >= 10:
                    break

    print()
    print("=" * 112)
    print(
        "Old mature pySTAMPS master"
    )
    print("=" * 112)

    print(
        f"old master_ix              : "
        f"{old_master_ix}"
    )

    print(
        f"master date from day[]     : "
        f"{old_master_date_by_ix}"
    )

    print(
        f"master date from master_day: "
        f"{old_master_date_by_day}"
    )

    print(
        f"master exists current stack: "
        f"{old_master_in_current}"
    )

    if mapped_master_idx0 is not None:

        print(
            f"mapped current index       : "
            f"{mapped_master_idx0} "
            f"(0-based), "
            f"{mapped_master_idx0+1} "
            f"(1-based)"
        )

    print()
    print("=" * 112)
    print(
        "Stage7 Pass-3 master comparison"
    )
    print("=" * 112)

    print(
        "Current pyPSDS temporal gauge:"
    )

    print(
        f"  master                   : "
        f"{current_dates[0]} "
        f"(1-based 1)"
    )

    print(
        f"  Gbase rank               : "
        f"{design_gauge['rank_Gbase']}/"
        f"{ndate-1}"
    )

    print(
        f"  A2 rank                  : "
        f"{design_gauge['rank_A2']}/3"
    )

    print(
        f"  scaled A2 condition      : "
        f"{design_gauge['condition_scaled']:.6f}"
    )

    if design_old is not None:

        print()

        print(
            "Old mature pySTAMPS master:"
        )

        print(
            f"  master                   : "
            f"{old_master_date_by_ix} "
            f"(1-based "
            f"{mapped_master_idx0+1})"
        )

        print(
            f"  Gbase rank               : "
            f"{design_old['rank_Gbase']}/"
            f"{ndate-1}"
        )

        print(
            f"  A2 rank                  : "
            f"{design_old['rank_A2']}/3"
        )

        print(
            f"  scaled A2 condition      : "
            f"{design_old['condition_scaled']:.6f}"
        )

    # ========================================================
    # Decision
    # ========================================================

    if (
        exact_order_match
        and
        old_master_date_by_ix
        ==
        old_master_date_by_day
        and
        mapped_master_idx0
        is not None
        and
        design_old is not None
        and
        design_old[
            "rank_Gbase"
        ]
        ==
        ndate - 1
        and
        design_old[
            "rank_A2"
        ]
        ==
        3
    ):

        status = (
            "PASS_REUSE_MATURE_PYSTAMPS_MASTER"
        )

    elif (
        same_date_set
        and
        mapped_master_idx0
        is not None
    ):

        status = (
            "REVIEW_DATE_ORDER_BEFORE_MASTER_REUSE"
        )

    else:

        status = (
            "REVIEW_OLD_MASTER_NOT_DIRECTLY_REUSABLE"
        )

    print()
    print(
        f"STEP 10R4a1 STATUS: "
        f"{status}"
    )

    if status == (
        "PASS_REUSE_MATURE_PYSTAMPS_MASTER"
    ):

        print(
            "Bridge recommendation:"
        )

        print(
            "  keep canonical Step09a unchanged;"
        )

        print(
            "  inside pystamps_bridge_v09, "
            "re-gauge acquisition phase to the "
            "old mature pySTAMPS master;"
        )

        print(
            "  virtual SB IFG phase remains "
            "mathematically unchanged."
        )


if __name__ == "__main__":
    main()
