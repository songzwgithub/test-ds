#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


def parse_date(x):
    s = str(x)
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Expected YYYYMMDD, got {s!r}")
    return np.datetime64(
        f"{s[:4]}-{s[4:6]}-{s[6:8]}",
        "D",
    )


def qprint(title, x, qs, fmt=".6f"):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    q = np.percentile(x, qs)
    print(title)
    print(
        "  "
        + " / ".join(
            format(v, fmt)
            for v in q
        )
    )


def fit_full(y, s, t):
    """
    Exact FWL OLS:
        y = a + v*t + k*s
    """

    tc = t - np.mean(t)
    ss_t = np.sum(tc * tc)

    ym = np.mean(y, axis=1, keepdims=True)
    yc = y - ym

    v0 = (
        np.sum(
            yc * tc[None, :],
            axis=1,
        )
        / ss_t
    )

    yp = (
        yc
        -
        v0[:, None] * tc[None, :]
    )

    sm = np.mean(s, axis=1, keepdims=True)
    sc = s - sm

    st = (
        np.sum(
            sc * tc[None, :],
            axis=1,
        )
        / ss_t
    )

    sp = (
        sc
        -
        st[:, None] * tc[None, :]
    )

    ss_s = np.sum(
        sp * sp,
        axis=1,
    )

    num = np.sum(
        sp * yp,
        axis=1,
    )

    good = (
        np.isfinite(ss_s)
        &
        (ss_s > 1e-12)
    )

    k = np.full(
        y.shape[0],
        np.nan,
        dtype=np.float64,
    )

    k[good] = (
        num[good]
        /
        ss_s[good]
    )

    # training incremental R2
    sse0 = np.sum(
        yp * yp,
        axis=1,
    )

    res1 = (
        yp
        -
        k[:, None] * sp
    )

    sse1 = np.sum(
        res1 * res1,
        axis=1,
    )

    r2 = np.full(
        y.shape[0],
        np.nan,
        dtype=np.float64,
    )

    ok = (
        good
        &
        (sse0 > 1e-15)
    )

    r2[ok] = (
        1.0
        -
        sse1[ok]
        /
        sse0[ok]
    )

    return k, r2


def fit_train_predict(
    y_train,
    s_train,
    t_train,
    s_test,
    t_test,
):

    tm = np.mean(t_train)
    tc = t_train - tm
    ss_t = np.sum(tc * tc)

    # ---------------- reduced model ----------------

    ym = np.mean(
        y_train,
        axis=1,
    )

    yc = (
        y_train
        -
        ym[:, None]
    )

    v0 = (
        np.sum(
            yc * tc[None, :],
            axis=1,
        )
        /
        ss_t
    )

    a0 = (
        ym
        -
        v0 * tm
    )

    pred0 = (
        a0[:, None]
        +
        v0[:, None]
        *
        t_test[None, :]
    )

    # ---------------- FWL sensitivity ----------------

    sm = np.mean(
        s_train,
        axis=1,
    )

    sc = (
        s_train
        -
        sm[:, None]
    )

    st = (
        np.sum(
            sc * tc[None, :],
            axis=1,
        )
        /
        ss_t
    )

    sperp = (
        sc
        -
        st[:, None]
        *
        tc[None, :]
    )

    ss_s = np.sum(
        sperp * sperp,
        axis=1,
    )

    yperp = (
        yc
        -
        v0[:, None]
        *
        tc[None, :]
    )

    num = np.sum(
        sperp * yperp,
        axis=1,
    )

    good = (
        np.isfinite(ss_s)
        &
        (ss_s > 1e-12)
    )

    k = np.full(
        y_train.shape[0],
        np.nan,
        dtype=np.float64,
    )

    k[good] = (
        num[good]
        /
        ss_s[good]
    )

    # ---------------- full model ----------------

    z = (
        y_train
        -
        k[:, None] * s_train
    )

    zm = np.mean(
        z,
        axis=1,
    )

    zc = (
        z
        -
        zm[:, None]
    )

    v1 = (
        np.sum(
            zc * tc[None, :],
            axis=1,
        )
        /
        ss_t
    )

    a1 = (
        zm
        -
        v1 * tm
    )

    pred1 = (
        a1[:, None]
        +
        v1[:, None]
        *
        t_test[None, :]
        +
        k[:, None]
        *
        s_test
    )

    return pred0, pred1, k, good


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=65536,
    )

    ap.add_argument(
        "--folds",
        type=int,
        default=5,
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

    root = Path(paths.output_dir) / "v09"

    invdir = (
        root
        / "network_inversion_v09"
    )

    sdir = (
        root
        / "scla_v09"
        / "production_sensitivity"
    )

    cvdir = (
        root
        / "scla_v09"
        / "scla_temporal_cv"
    )

    outdir = (
        root
        / "scla_v09"
        / "scla_common_mode_audit"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Require 10b3
    # ========================================================

    cv_manifest = json.loads(
        (
            cvdir
            / "scla_temporal_cv_manifest.json"
        ).read_text()
    )

    if cv_manifest["status"] != "PASS":
        raise RuntimeError(
            "Step10b3 did not PASS numerically"
        )

    # ========================================================
    # Inputs
    # ========================================================

    phase_path = (
        invdir
        / "acquisition_phase_l2_candidate_rad.npy"
    )

    sensitivity_path = (
        sdir
        / "topographic_phase_sensitivity_rad_per_m.npy"
    )

    Y = np.load(
        phase_path,
        mmap_mode="r",
    )

    S = np.load(
        sensitivity_path,
        mmap_mode="r",
    )

    if Y.shape != S.shape:
        raise RuntimeError(
            "phase/sensitivity shape mismatch"
        )

    npoint, ndate = Y.shape

    # ========================================================
    # Time axis
    # ========================================================

    dates = np.asarray(
        [
            parse_date(x)
            for x in stack.dates
        ],
        dtype="datetime64[D]",
    )

    years = (
        (
            dates - dates[0]
        )
        .astype("timedelta64[D]")
        .astype(np.float64)
        /
        365.2425
    )

    # ========================================================
    # Spatial common modes.
    #
    # These are gauge/reference quantities only.
    # No corrected phase cube is written.
    # ========================================================

    print("=" * 112)
    print(
        "Step 10b4 - SCLA spatial common-mode "
        "/ reference-consistency audit"
    )
    print("=" * 112)

    print(f"config                     : {config_path}")
    print(f"strict points              : {npoint:,}")
    print(f"acquisitions               : {ndate}")
    print(f"folds                      : {args.folds}")
    print(
        "reference                  : "
        "scene-wide strict-point median per acquisition"
    )

    phase_median = np.empty(
        ndate,
        dtype=np.float64,
    )

    sensitivity_median = np.empty(
        ndate,
        dtype=np.float64,
    )

    for j in range(ndate):

        phase_median[j] = float(
            np.median(
                np.asarray(
                    Y[:, j],
                    dtype=np.float64,
                )
            )
        )

        sensitivity_median[j] = float(
            np.median(
                np.asarray(
                    S[:, j],
                    dtype=np.float64,
                )
            )
        )

    np.save(
        outdir
        / "phase_spatial_median_by_acquisition_rad.npy",
        phase_median,
    )

    np.save(
        outdir
        / "sensitivity_spatial_median_by_acquisition_rad_per_m.npy",
        sensitivity_median,
    )

    print()
    print("=" * 112)
    print("Spatial common modes")
    print("=" * 112)

    qprint(
        "phase median p01/p05/p50/p95/p99 [rad]:",
        phase_median,
        [1,5,50,95,99],
        ".6f",
    )

    qprint(
        "sensitivity median p01/p05/p50/p95/p99 [rad/m]:",
        sensitivity_median,
        [1,5,50,95,99],
        ".6e",
    )

    # ========================================================
    # Output maps
    # ========================================================

    def mmap(name, dtype=np.float32):
        return np.lib.format.open_memmap(
            outdir / name,
            mode="w+",
            dtype=dtype,
            shape=(npoint,),
        )

    kfull_out = mmap(
        "scene_median_ref_full_k_m.npy"
    )

    train_r2_out = mmap(
        "scene_median_ref_training_incremental_r2.npy"
    )

    cv_r2_out = mmap(
        "scene_median_ref_cv_incremental_r2.npy"
    )

    cv_rms0_out = mmap(
        "scene_median_ref_cv_rms_reduced_rad.npy"
    )

    cv_rms1_out = mmap(
        "scene_median_ref_cv_rms_scla_rad.npy"
    )

    kmed_out = mmap(
        "scene_median_ref_cv_k_median_m.npy"
    )

    kmad_out = mmap(
        "scene_median_ref_cv_k_mad_m.npy"
    )

    improve_out = mmap(
        "scene_median_ref_improved_fold_count.npy",
        dtype=np.uint8,
    )

    # ========================================================
    # CV definition
    # ========================================================

    nfold = args.folds

    fold_id = (
        np.arange(ndate)
        %
        nfold
    )

    nonfinite = 0
    degenerate = 0

    # ========================================================
    # Full-scene streaming
    # ========================================================

    for b0 in range(
        0,
        npoint,
        args.batch_size,
    ):

        b1 = min(
            b0 + args.batch_size,
            npoint,
        )

        y = (
            np.asarray(
                Y[b0:b1],
                dtype=np.float64,
            )
            -
            phase_median[None, :]
        )

        s = (
            np.asarray(
                S[b0:b1],
                dtype=np.float64,
            )
            -
            sensitivity_median[None, :]
        )

        B = b1 - b0

        finite = (
            np.all(np.isfinite(y), axis=1)
            &
            np.all(np.isfinite(s), axis=1)
        )

        nonfinite += int(
            np.count_nonzero(~finite)
        )

        # ---------------- full fit ----------------

        kfull, train_r2 = fit_full(
            y,
            s,
            years,
        )

        # ---------------- CV ----------------

        total_sse0 = np.zeros(
            B,
            dtype=np.float64,
        )

        total_sse1 = np.zeros(
            B,
            dtype=np.float64,
        )

        improve = np.zeros(
            B,
            dtype=np.uint8,
        )

        K = np.full(
            (B, nfold),
            np.nan,
            dtype=np.float64,
        )

        for f in range(nfold):

            test_idx = np.flatnonzero(
                fold_id == f
            )

            train_idx = np.flatnonzero(
                fold_id != f
            )

            pred0, pred1, k, good = (
                fit_train_predict(
                    y[:, train_idx],
                    s[:, train_idx],
                    years[train_idx],
                    s[:, test_idx],
                    years[test_idx],
                )
            )

            obs = y[:, test_idx]

            e0 = obs - pred0
            e1 = obs - pred1

            ss0 = np.sum(
                e0 * e0,
                axis=1,
            )

            ss1 = np.sum(
                e1 * e1,
                axis=1,
            )

            good = (
                good
                &
                finite
            )

            degenerate += int(
                np.count_nonzero(
                    finite & ~good
                )
            )

            total_sse0[good] += ss0[good]
            total_sse1[good] += ss1[good]

            improve[good] += (
                ss1[good] < ss0[good]
            ).astype(np.uint8)

            K[:, f] = k

        valid_cv = (
            finite
            &
            np.all(np.isfinite(K), axis=1)
            &
            (total_sse0 > 1e-15)
        )

        cv_r2 = np.full(
            B,
            np.nan,
            dtype=np.float64,
        )

        cv_rms0 = np.full(
            B,
            np.nan,
            dtype=np.float64,
        )

        cv_rms1 = np.full(
            B,
            np.nan,
            dtype=np.float64,
        )

        cv_r2[valid_cv] = (
            1.0
            -
            total_sse1[valid_cv]
            /
            total_sse0[valid_cv]
        )

        cv_rms0[valid_cv] = np.sqrt(
            total_sse0[valid_cv]
            /
            ndate
        )

        cv_rms1[valid_cv] = np.sqrt(
            total_sse1[valid_cv]
            /
            ndate
        )

        kmed = np.nanmedian(
            K,
            axis=1,
        )

        kmad = np.nanmedian(
            np.abs(
                K - kmed[:, None]
            ),
            axis=1,
        )

        # ---------------- save ----------------

        kfull_out[b0:b1] = (
            kfull.astype(np.float32)
        )

        train_r2_out[b0:b1] = (
            train_r2.astype(np.float32)
        )

        cv_r2_out[b0:b1] = (
            cv_r2.astype(np.float32)
        )

        cv_rms0_out[b0:b1] = (
            cv_rms0.astype(np.float32)
        )

        cv_rms1_out[b0:b1] = (
            cv_rms1.astype(np.float32)
        )

        kmed_out[b0:b1] = (
            kmed.astype(np.float32)
        )

        kmad_out[b0:b1] = (
            kmad.astype(np.float32)
        )

        improve_out[b0:b1] = improve

        print(
            f"  {b1:,}/{npoint:,}"
        )

    for x in (
        kfull_out,
        train_r2_out,
        cv_r2_out,
        cv_rms0_out,
        cv_rms1_out,
        kmed_out,
        kmad_out,
        improve_out,
    ):
        x.flush()

    # ========================================================
    # Results
    # ========================================================

    kfull = np.asarray(
        kfull_out,
        dtype=np.float64,
    )

    train_r2 = np.asarray(
        train_r2_out,
        dtype=np.float64,
    )

    cv_r2 = np.asarray(
        cv_r2_out,
        dtype=np.float64,
    )

    rms0 = np.asarray(
        cv_rms0_out,
        dtype=np.float64,
    )

    rms1 = np.asarray(
        cv_rms1_out,
        dtype=np.float64,
    )

    kmed = np.asarray(
        kmed_out,
        dtype=np.float64,
    )

    kmad = np.asarray(
        kmad_out,
        dtype=np.float64,
    )

    improve = np.asarray(
        improve_out,
        dtype=np.int16,
    )

    valid = (
        np.isfinite(kfull)
        &
        np.isfinite(train_r2)
        &
        np.isfinite(cv_r2)
        &
        np.isfinite(rms0)
        &
        np.isfinite(rms1)
        &
        np.isfinite(kmed)
        &
        np.isfinite(kmad)
    )

    nvalid = int(
        np.count_nonzero(valid)
    )

    print()
    print("=" * 112)
    print(
        "Scene-median-referenced coefficient"
    )
    print("=" * 112)

    qprint(
        "full k p01/p05/p50/p95/p99 [m]:",
        kfull[valid],
        [1,5,50,95,99],
        ".3f",
    )

    qprint(
        "CV median k p01/p05/p50/p95/p99 [m]:",
        kmed[valid],
        [1,5,50,95,99],
        ".3f",
    )

    qprint(
        "k MAD p50/p90/p95/p99/max [m]:",
        kmad[valid],
        [50,90,95,99,100],
        ".3f",
    )

    print()
    print("=" * 112)
    print(
        "Scene-median-referenced training fit"
    )
    print("=" * 112)

    qprint(
        "training incremental R2 "
        "p01/p05/p50/p95/p99:",
        train_r2[valid],
        [1,5,50,95,99],
        ".6f",
    )

    print()
    print("=" * 112)
    print(
        "Scene-median-referenced held-out prediction"
    )
    print("=" * 112)

    qprint(
        "CV RMS reduced p50/p90/p95/p99 [rad]:",
        rms0[valid],
        [50,90,95,99],
        ".6f",
    )

    qprint(
        "CV RMS SCLA    p50/p90/p95/p99 [rad]:",
        rms1[valid],
        [50,90,95,99],
        ".6f",
    )

    qprint(
        "CV incremental R2 "
        "p01/p05/p50/p95/p99:",
        cv_r2[valid],
        [1,5,50,95,99],
        ".6f",
    )

    for threshold in (
        0.0,
        0.01,
        0.05,
        0.10,
    ):

        n = int(
            np.count_nonzero(
                cv_r2[valid]
                >
                threshold
            )
        )

        print(
            f"CV R2 > {threshold:.2f}          : "
            f"{n:8,d} "
            f"({100*n/nvalid:.3f}%)"
        )

    print()

    for nf in range(
        nfold + 1
    ):

        n = int(
            np.count_nonzero(
                improve[valid]
                == nf
            )
        )

        print(
            f"held-out improved folds = {nf}: "
            f"{n:8,d} "
            f"({100*n/nvalid:.3f}%)"
        )

    # ========================================================
    # Compare directly with Step10b3 raw CV
    # ========================================================

    raw_cv = np.asarray(
        np.load(
            cvdir
            / "cv_incremental_r2.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    raw_valid = np.isfinite(raw_cv)

    print()
    print("=" * 112)
    print(
        "Before / after common-mode reference"
    )
    print("=" * 112)

    print(
        f"raw median CV R2           : "
        f"{np.median(raw_cv[raw_valid]):+.6f}"
    )

    print(
        f"referenced median CV R2    : "
        f"{np.median(cv_r2[valid]):+.6f}"
    )

    print(
        f"raw CV R2 > 0             : "
        f"{100*np.mean(raw_cv[raw_valid] > 0):.3f}%"
    )

    print(
        f"referenced CV R2 > 0      : "
        f"{100*np.mean(cv_r2[valid] > 0):.3f}%"
    )

    print(
        f"raw median k              : "
        f"-8.544 m"
    )

    print(
        f"referenced median k       : "
        f"{np.median(kfull[valid]):+.3f} m"
    )

    # ========================================================
    # Numerical status only
    # ========================================================

    if (
        nonfinite != 0
        or
        degenerate != 0
        or
        nvalid != npoint
    ):
        status = "REVIEW_INVALID_COMMON_MODE_AUDIT"
    else:
        status = "PASS"

    manifest = {
        "format":
            "pyPSDS-GAMMA-scla-common-mode-audit-v09",

        "status":
            status,

        "reference_definition":
            (
                "per-acquisition spatial median "
                "of all strict points"
            ),

        "phase_source":
            str(phase_path),

        "sensitivity_source":
            str(sensitivity_path),

        "phase_and_sensitivity_referenced_consistently":
            True,

        "phase_modified":
            False,

        "production_phase_written":
            False,

        "residual_dem_correction_applied":
            False,

        "scientific_decision":
            False,
    }

    manifest_path = (
        outdir
        / "scla_common_mode_audit_manifest.json"
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
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10b4 STATUS: {status} / "
        "COMMON-MODE AUDIT ONLY"
    )

    print(
        "No production phase or SCLA correction "
        "has been applied."
    )


if __name__ == "__main__":
    main()
