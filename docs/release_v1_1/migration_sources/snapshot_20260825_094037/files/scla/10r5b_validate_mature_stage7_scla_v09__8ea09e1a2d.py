#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree


BRIDGE = Path(
    "/home/ubuntu/Downloads/psds/"
    "prototype_outputs/v09/"
    "pystamps_bridge_v09"
)

R3C2 = Path(
    "/home/ubuntu/Downloads/psds/"
    "prototype_outputs/v09/"
    "scla_v09/pystamps_bridge/"
    "r3c2_pointwise_bperp"
)

R3B = Path(
    "/home/ubuntu/Downloads/psds/"
    "prototype_outputs/v09/"
    "scla_v09/pystamps_bridge/"
    "r3b_grid_adapter"
)

R4A = Path(
    "/home/ubuntu/Downloads/psds/"
    "prototype_outputs/v09/"
    "scla_v09/pystamps_bridge/"
    "r4a_stage7_contract"
)

OUTDIR = (
    BRIDGE
    /
    "r5b_scientific_validation"
)


def qline(
    title,
    x,
    qs=(1, 5, 50, 95, 99),
    fmt=".6e",
):
    a = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

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


def network_matrix(
    ifgday_ix,
    nimage,
):
    G = np.zeros(
        (
            ifgday_ix.shape[0],
            nimage,
        ),
        dtype=np.float64,
    )

    rr = np.arange(
        ifgday_ix.shape[0]
    )

    G[
        rr,
        ifgday_ix[:, 0] - 1,
    ] = -1.0

    G[
        rr,
        ifgday_ix[:, 1] - 1,
    ] = +1.0

    return G


def run_cv(
    *,
    name,
    folds,
    dph,
    db_mean,
    dt,
    k_full,
    batch=65536,
):

    npoint, nobs = dph.shape
    nfold = len(folds)

    sse_full = np.zeros(
        npoint,
        dtype=np.float64,
    )

    sse_reduced = np.zeros(
        npoint,
        dtype=np.float64,
    )

    k_fold = np.empty(
        (
            npoint,
            nfold,
        ),
        dtype=np.float32,
    )

    all_ix = np.arange(
        nobs,
        dtype=np.int64,
    )

    print()
    print("=" * 112)
    print(
        f"{name} held-out validation"
    )
    print("=" * 112)

    for f, test_ix in enumerate(
        folds
    ):

        test_ix = np.asarray(
            test_ix,
            dtype=np.int64,
        )

        train_ix = np.setdiff1d(
            all_ix,
            test_ix,
        )

        A_train = np.column_stack(
            (
                np.ones(
                    train_ix.size
                ),
                db_mean[
                    train_ix
                ],
                dt[
                    train_ix
                ],
            )
        )

        A_test = np.column_stack(
            (
                np.ones(
                    test_ix.size
                ),
                db_mean[
                    test_ix
                ],
                dt[
                    test_ix
                ],
            )
        )

        A0_train = np.column_stack(
            (
                np.ones(
                    train_ix.size
                ),
                dt[
                    train_ix
                ],
            )
        )

        A0_test = np.column_stack(
            (
                np.ones(
                    test_ix.size
                ),
                dt[
                    test_ix
                ],
            )
        )

        if (
            np.linalg.matrix_rank(
                A_train
            )
            != 3
        ):
            raise RuntimeError(
                f"{name} fold {f}: "
                "full design rank deficient"
            )

        if (
            np.linalg.matrix_rank(
                A0_train
            )
            != 2
        ):
            raise RuntimeError(
                f"{name} fold {f}: "
                "reduced design rank deficient"
            )

        P = np.linalg.pinv(
            A_train
        )

        P0 = np.linalg.pinv(
            A0_train
        )

        for start in range(
            0,
            npoint,
            batch,
        ):

            stop = min(
                start + batch,
                npoint,
            )

            Ytr = np.asarray(
                dph[
                    start:stop,
                    :
                ][
                    :,
                    train_ix
                ],
                dtype=np.float64,
            )

            Yte = np.asarray(
                dph[
                    start:stop,
                    :
                ][
                    :,
                    test_ix
                ],
                dtype=np.float64,
            )

            coef = (
                Ytr
                @
                P.T
            )

            coef0 = (
                Ytr
                @
                P0.T
            )

            pred = (
                coef
                @
                A_test.T
            )

            pred0 = (
                coef0
                @
                A0_test.T
            )

            err = (
                Yte
                -
                pred
            )

            err0 = (
                Yte
                -
                pred0
            )

            sse_full[
                start:stop
            ] += np.sum(
                err * err,
                axis=1,
            )

            sse_reduced[
                start:stop
            ] += np.sum(
                err0 * err0,
                axis=1,
            )

            k_fold[
                start:stop,
                f
            ] = coef[
                :,
                1
            ].astype(
                np.float32
            )

        print(
            f"fold {f+1}/{nfold}: "
            f"train={train_ix.size}, "
            f"test={test_ix.size}"
        )

    improvement = np.divide(
        sse_reduced
        -
        sse_full,
        sse_reduced,
        out=np.full(
            npoint,
            np.nan,
            dtype=np.float64,
        ),
        where=(
            sse_reduced > 0
        ),
    )

    global_improvement = (
        1.0
        -
        np.sum(
            sse_full
        )
        /
        np.sum(
            sse_reduced
        )
    )

    fraction_improved = float(
        np.mean(
            sse_full
            <
            sse_reduced
        )
    )

    k_mean = np.mean(
        k_fold.astype(
            np.float64
        ),
        axis=1,
    )

    k_std = np.std(
        k_fold.astype(
            np.float64
        ),
        axis=1,
        ddof=1,
    )

    k_diff = (
        k_mean
        -
        k_full
    )

    k_corr = float(
        np.corrcoef(
            k_mean,
            k_full,
        )[0, 1]
    )

    abs_full = np.abs(
        k_full
    )

    strong = (
        abs_full
        >=
        np.median(
            abs_full
        )
    )

    sign_same = (
        np.sign(
            k_fold[
                strong,
                :
            ]
        )
        ==
        np.sign(
            k_full[
                strong
            ][
                :,
                None
            ]
        )
    )

    sign_agreement = float(
        np.mean(
            sign_same
        )
    )

    print()
    print(
        f"global held-out improvement: "
        f"{global_improvement:.6f}"
    )

    print(
        f"points improved            : "
        f"{100*fraction_improved:.3f}%"
    )

    qline(
        "point held-out improvement "
        "p01/p05/p50/p95/p99:",
        improvement,
    )

    print(
        f"mean-CV K vs full K corr   : "
        f"{k_corr:.6f}"
    )

    qline(
        "CV K std "
        "p01/p05/p50/p95/p99 [rad/m]:",
        k_std,
    )

    qline(
        "|meanCV K-full K| "
        "p01/p05/p50/p95/p99 [rad/m]:",
        np.abs(
            k_diff
        ),
    )

    print(
        "fold sign agreement "
        "(upper half |K|)        : "
        f"{100*sign_agreement:.3f}%"
    )

    return {
        "sse_full":
            sse_full,

        "sse_reduced":
            sse_reduced,

        "improvement":
            improvement,

        "global_improvement":
            float(
                global_improvement
            ),

        "fraction_improved":
            fraction_improved,

        "k_fold":
            k_fold,

        "k_mean":
            k_mean,

        "k_std":
            k_std,

        "k_corr":
            k_corr,

        "sign_agreement":
            sign_agreement,
    }


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug = json.loads(
        (
            BRIDGE
            /
            "stage7_sbas_debug.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        debug.get("status")
        !=
        "completed"
    ):
        raise RuntimeError(
            "Stage7 not completed"
        )

    # ========================================================
    # Inputs
    # ========================================================

    ps = loadmat(
        BRIDGE
        /
        "ps2.mat",
        variable_names=[
            "day",
            "ifgday_ix",
            "master_ix",
        ],
        squeeze_me=False,
    )

    day = np.asarray(
        ps["day"],
        dtype=np.float64,
    ).reshape(-1)

    ifgday_ix = np.asarray(
        ps["ifgday_ix"],
        dtype=np.int64,
    )

    master_ix = int(
        round(
            float(
                np.asarray(
                    ps["master_ix"]
                ).reshape(-1)[0]
            )
        )
    )

    master0 = (
        master_ix - 1
    )

    ph = np.asarray(
        loadmat(
            BRIDGE
            /
            "phuw2.mat",
            variable_names=[
                "ph_uw",
            ],
            squeeze_me=False,
        )[
            "ph_uw"
        ],
        dtype=np.float32,
    )

    final = loadmat(
        BRIDGE
        /
        "scla2.mat",
        variable_names=[
            "K_ps_uw",
            "C_ps_uw",
            "ph_scla",
        ],
        squeeze_me=False,
    )

    K = np.asarray(
        final[
            "K_ps_uw"
        ],
        dtype=np.float64,
    ).reshape(-1)

    C = np.asarray(
        final[
            "C_ps_uw"
        ],
        dtype=np.float64,
    ).reshape(-1)

    ph_scla = np.asarray(
        final[
            "ph_scla"
        ],
        dtype=np.float32,
    )

    npoint, nimage = (
        ph.shape
    )

    if K.size != npoint:
        raise RuntimeError(
            "K size mismatch"
        )

    bmean_ifg = np.asarray(
        np.load(
            R3C2
            /
            "bperp_mean_by_ifg_m.npy"
        ),
        dtype=np.float64,
    )

    local_xy = np.asarray(
        np.load(
            R3B
            /
            "local_xy_m.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    ref_ix = np.asarray(
        np.load(
            R4A
            /
            "stage7_reference_point_indices.npy"
        ),
        dtype=np.int64,
    )

    # ========================================================
    # Reproduce exact Stage7 PASS3 domain
    # ========================================================

    G = network_matrix(
        ifgday_ix,
        nimage,
    )

    img0 = np.asarray(
        [
            i
            for i in range(
                nimage
            )
            if i != master0
        ],
        dtype=np.int64,
    )

    Gbase = G[
        :,
        img0
    ]

    if (
        np.linalg.matrix_rank(
            Gbase
        )
        !=
        img0.size
    ):
        raise RuntimeError(
            "Gbase rank deficient"
        )

    Pbase = np.linalg.pinv(
        Gbase
    )

    bmean_sm = (
        bmean_ifg
        @
        Pbase.T
    )

    db_mean = np.diff(
        bmean_sm
    )

    dt = np.diff(
        day[
            img0
        ]
    )

    ref_mean = np.mean(
        ph[
            ref_ix,
            :
        ].astype(
            np.float64
        ),
        axis=0,
    )

    ph_some = (
        ph[
            :,
            img0
        ].astype(
            np.float64
        )
        -
        ref_mean[
            img0
        ][
            None,
            :
        ]
    )

    dph = np.diff(
        ph_some,
        axis=1,
    ).astype(
        np.float32
    )

    nobs = dph.shape[1]

    if nobs != 36:
        raise RuntimeError(
            f"Expected 36 PASS3 increments, "
            f"got {nobs}"
        )

    print("=" * 112)
    print(
        "Step 10R5b - mature Stage7 "
        "scientific validity audit"
    )
    print("=" * 112)

    print(
        f"points                     : "
        f"{npoint:,}"
    )

    print(
        f"PASS3 increments           : "
        f"{nobs}"
    )

    print(
        f"master_ix                  : "
        f"{master_ix}"
    )

    # ========================================================
    # 1. Interleaved 6-fold CV
    # ========================================================

    interleaved = [
        np.arange(
            f,
            nobs,
            6,
            dtype=np.int64,
        )
        for f in range(6)
    ]

    cv_inter = run_cv(
        name="Interleaved 6-fold",
        folds=interleaved,
        dph=dph,
        db_mean=db_mean,
        dt=dt,
        k_full=K,
    )

    # ========================================================
    # 2. Contiguous blocked 6-fold CV
    # ========================================================

    blocked = [
        x.astype(
            np.int64
        )
        for x in np.array_split(
            np.arange(
                nobs,
                dtype=np.int64,
            ),
            6,
        )
    ]

    cv_block = run_cv(
        name="Blocked 6-fold",
        folds=blocked,
        dph=dph,
        db_mean=db_mean,
        dt=dt,
        k_full=K,
    )

    # ========================================================
    # 3. Spatial consistency of final K
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Final K spatial consistency"
    )
    print("=" * 112)

    tree = cKDTree(
        local_xy
    )

    dist, nbr = tree.query(
        local_xy,
        k=7,
        workers=-1,
    )

    neighbors = (
        nbr[
            :,
            1:
        ]
    )

    neighbor_K = K[
        neighbors
    ]

    local_median = np.median(
        neighbor_K,
        axis=1,
    )

    local_delta = (
        K
        -
        local_median
    )

    spatial_corr = float(
        np.corrcoef(
            K,
            local_median,
        )[0, 1]
    )

    print(
        f"K vs 6-neighbour median corr: "
        f"{spatial_corr:.6f}"
    )

    qline(
        "|K-local median| "
        "p01/p05/p50/p95/p99 [rad/m]:",
        np.abs(
            local_delta
        ),
    )

    qline(
        "6th-neighbour distance "
        "p01/p05/p50/p95/p99 [m]:",
        dist[
            :,
            -1
        ],
        fmt=".3f",
    )

    # ========================================================
    # 4. Counterfactual rate preservation
    #
    # C is constant in time, therefore irrelevant to rate.
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Long-term linear-rate preservation"
    )
    print("=" * 112)

    t_year = (
        day
        -
        np.mean(
            day
        )
    ) / 365.25

    denom = float(
        np.dot(
            t_year,
            t_year,
        )
    )

    rate_before = (
        ph.astype(
            np.float64
        )
        @
        t_year
    ) / denom

    rate_after = (
        (
            ph.astype(
                np.float64
            )
            -
            ph_scla.astype(
                np.float64
            )
        )
        @
        t_year
    ) / denom

    rate_delta = (
        rate_after
        -
        rate_before
    )

    qline(
        "rate BEFORE "
        "p01/p05/p50/p95/p99 [rad/yr]:",
        rate_before,
    )

    qline(
        "rate AFTER "
        "p01/p05/p50/p95/p99 [rad/yr]:",
        rate_after,
    )

    qline(
        "rate CHANGE "
        "p01/p05/p50/p95/p99 [rad/yr]:",
        rate_delta,
    )

    qline(
        "|rate change| "
        "p01/p05/p50/p95/p99 [rad/yr]:",
        np.abs(
            rate_delta
        ),
    )

    rate_corr = float(
        np.corrcoef(
            rate_before,
            rate_after,
        )[0, 1]
    )

    print(
        f"before-vs-after rate corr  : "
        f"{rate_corr:.6f}"
    )

    # ========================================================
    # Save compact diagnostics
    # ========================================================

    np.save(
        OUTDIR
        /
        "interleaved_cv_improvement.npy",
        cv_inter[
            "improvement"
        ].astype(
            np.float32
        ),
    )

    np.save(
        OUTDIR
        /
        "blocked_cv_improvement.npy",
        cv_block[
            "improvement"
        ].astype(
            np.float32
        ),
    )

    np.save(
        OUTDIR
        /
        "interleaved_cv_K_std.npy",
        cv_inter[
            "k_std"
        ].astype(
            np.float32
        ),
    )

    np.save(
        OUTDIR
        /
        "blocked_cv_K_std.npy",
        cv_block[
            "k_std"
        ].astype(
            np.float32
        ),
    )

    np.save(
        OUTDIR
        /
        "final_K_local_median.npy",
        local_median.astype(
            np.float32
        ),
    )

    np.save(
        OUTDIR
        /
        "rate_change_rad_per_year.npy",
        rate_delta.astype(
            np.float32
        ),
    )

    # ========================================================
    # Diagnostic status only.
    #
    # Do NOT automatically accept SCLA here.
    # ========================================================

    status = (
        "PASS_R5B_DIAGNOSTICS_COMPLETE"
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-mature-stage7-scientific-validation-v09",

        "status":
            status,

        "interleaved_cv": {
            "global_improvement":
                cv_inter[
                    "global_improvement"
                ],

            "fraction_points_improved":
                cv_inter[
                    "fraction_improved"
                ],

            "K_mean_vs_full_corr":
                cv_inter[
                    "k_corr"
                ],

            "strong_K_sign_agreement":
                cv_inter[
                    "sign_agreement"
                ],
        },

        "blocked_cv": {
            "global_improvement":
                cv_block[
                    "global_improvement"
                ],

            "fraction_points_improved":
                cv_block[
                    "fraction_improved"
                ],

            "K_mean_vs_full_corr":
                cv_block[
                    "k_corr"
                ],

            "strong_K_sign_agreement":
                cv_block[
                    "sign_agreement"
                ],
        },

        "spatial": {
            "K_neighbor_median_corr":
                spatial_corr,
        },

        "rate": {
            "before_after_corr":
                rate_corr,

            "abs_change_p50_rad_per_year":
                float(
                    np.percentile(
                        np.abs(
                            rate_delta
                        ),
                        50,
                    )
                ),

            "abs_change_p95_rad_per_year":
                float(
                    np.percentile(
                        np.abs(
                            rate_delta
                        ),
                        95,
                    )
                ),
        },

        "scla_applied":
            False,

        "stage8_executed":
            False,
    }

    manifest_path = (
        OUTDIR
        /
        "r5b_scientific_validation_manifest.json"
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
        f"STEP 10R5b STATUS: "
        f"{status}"
    )

    print(
        "This step did NOT accept or apply "
        "the SCLA correction."
    )


if __name__ == "__main__":
    main()
