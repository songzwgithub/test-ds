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


def qprint(title, x, qs=(1,5,50,95,99), fmt=".6f"):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    q = np.percentile(x, qs)
    print(title)
    print("  " + " / ".join(format(v, fmt) for v in q))


def fit_train_predict(y, s, t_train, t_test, s_test):
    """
    Exact OLS:
        reduced: y = a + v*t
        full   : y = a + v*t + k*s

    Returns predictions on held-out dates plus k and
    sensitivity identifiability of the training subset.
    """

    n = y.shape[0]

    tm = float(np.mean(t_train))
    tc = t_train - tm
    ss_t = float(np.sum(tc * tc))

    if ss_t <= 0:
        raise RuntimeError("Degenerate training time axis")

    # --------------------------------------------------------
    # Reduced model
    # --------------------------------------------------------

    ym = np.mean(y, axis=1)
    yc = y - ym[:, None]

    v0 = np.sum(
        yc * tc[None, :],
        axis=1,
    ) / ss_t

    a0 = ym - v0 * tm

    pred0 = (
        a0[:, None]
        +
        v0[:, None] * t_test[None, :]
    )

    # --------------------------------------------------------
    # FWL sensitivity residualization
    # --------------------------------------------------------

    sm = np.mean(s, axis=1)
    sc = s - sm[:, None]

    cov_st = np.sum(
        sc * tc[None, :],
        axis=1,
    )

    slope_st = cov_st / ss_t

    sperp = (
        sc
        -
        slope_st[:, None] * tc[None, :]
    )

    ss_s = np.sum(
        sperp * sperp,
        axis=1,
    )

    ss_sc = np.sum(
        sc * sc,
        axis=1,
    )

    retained = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    good = (
        np.isfinite(ss_s)
        &
        np.isfinite(ss_sc)
        &
        (ss_s > 1e-12)
        &
        (ss_sc > 1e-12)
    )

    retained[good] = np.sqrt(
        ss_s[good] / ss_sc[good]
    )

    # y residual from intercept+time
    yperp = (
        yc
        -
        v0[:, None] * tc[None, :]
    )

    num = np.sum(
        sperp * yperp,
        axis=1,
    )

    k = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    k[good] = (
        num[good] / ss_s[good]
    )

    # --------------------------------------------------------
    # Full model coefficients
    # --------------------------------------------------------

    ym2_data = (
        y
        -
        k[:, None] * s
    )

    ym2 = np.mean(
        ym2_data,
        axis=1,
    )

    yc2 = (
        ym2_data
        -
        ym2[:, None]
    )

    v1 = np.sum(
        yc2 * tc[None, :],
        axis=1,
    ) / ss_t

    a1 = (
        ym2
        -
        v1 * tm
    )

    pred1 = (
        a1[:, None]
        +
        v1[:, None] * t_test[None, :]
        +
        k[:, None] * s_test
    )

    return (
        pred0,
        pred1,
        k,
        retained,
    )


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg, config_path, paths, stack, _ = open_from_config(
        args.config
    )

    root = Path(paths.output_dir) / "v09"

    invdir = root / "network_inversion_v09"

    sdir = (
        root
        / "scla_v09"
        / "production_sensitivity"
    )

    fitdir = (
        root
        / "scla_v09"
        / "scla_candidate_fit"
    )

    outdir = (
        root
        / "scla_v09"
        / "scla_temporal_cv"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Require 10b2
    # ========================================================

    fit_manifest = json.loads(
        (
            fitdir
            / "scla_candidate_fit_manifest.json"
        ).read_text()
    )

    if fit_manifest["status"] != "PASS":
        raise RuntimeError(
            "Step10b2 did not PASS"
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

    full_k_path = (
        fitdir
        / "scla_sensitivity_coefficient_m.npy"
    )

    Y = np.load(
        phase_path,
        mmap_mode="r",
    )

    S = np.load(
        sensitivity_path,
        mmap_mode="r",
    )

    full_k = np.load(
        full_k_path,
        mmap_mode="r",
    )

    npoint, ndate = Y.shape

    if S.shape != Y.shape:
        raise RuntimeError(
            "phase/sensitivity shape mismatch"
        )

    if full_k.shape != (npoint,):
        raise RuntimeError(
            "full k shape mismatch"
        )

    # ========================================================
    # Dates
    # ========================================================

    dates = np.asarray(
        [parse_date(x) for x in stack.dates],
        dtype="datetime64[D]",
    )

    years = (
        (
            dates - dates[0]
        ).astype(
            "timedelta64[D]"
        ).astype(
            np.float64
        )
        / 365.2425
    )

    nfold = args.folds

    if nfold < 3:
        raise RuntimeError(
            "Use at least 3 folds"
        )

    fold_id = (
        np.arange(ndate)
        %
        nfold
    )

    print("=" * 112)
    print(
        "Step 10b3 - SCLA temporal "
        "cross-validation / stability audit"
    )
    print("=" * 112)

    print(f"config                     : {config_path}")
    print(f"strict points              : {npoint:,}")
    print(f"acquisitions               : {ndate}")
    print(f"folds                      : {nfold}")
    print(f"batch size                 : {args.batch_size:,}")
    print()

    for f in range(nfold):

        test = np.flatnonzero(
            fold_id == f
        )

        train = np.flatnonzero(
            fold_id != f
        )

        print(
            f"fold {f}: "
            f"train={train.size}, "
            f"test={test.size}, "
            f"test dates="
            +
            ",".join(
                str(stack.dates[i])
                for i in test
            )
        )

    # ========================================================
    # Outputs
    # ========================================================

    kfold_path = (
        outdir
        / "k_train_by_fold_m.npy"
    )

    kfold = np.lib.format.open_memmap(
        kfold_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint, nfold),
    )

    retained_path = (
        outdir
        / "training_identifiable_fraction_by_fold.npy"
    )

    retained_fold = np.lib.format.open_memmap(
        retained_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint, nfold),
    )

    sse0_path = (
        outdir
        / "heldout_sse_reduced.npy"
    )

    sse1_path = (
        outdir
        / "heldout_sse_scla.npy"
    )

    sse0_out = np.lib.format.open_memmap(
        sse0_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint,),
    )

    sse1_out = np.lib.format.open_memmap(
        sse1_path,
        mode="w+",
        dtype=np.float32,
        shape=(npoint,),
    )

    improve_count_path = (
        outdir
        / "heldout_improved_fold_count.npy"
    )

    improve_count_out = np.lib.format.open_memmap(
        improve_count_path,
        mode="w+",
        dtype=np.uint8,
        shape=(npoint,),
    )

    # ========================================================
    # Batch processing
    # ========================================================

    nonfinite = 0
    degenerate = 0

    for b0 in range(
        0,
        npoint,
        args.batch_size,
    ):

        b1 = min(
            b0 + args.batch_size,
            npoint,
        )

        y = np.asarray(
            Y[b0:b1],
            dtype=np.float64,
        )

        s = np.asarray(
            S[b0:b1],
            dtype=np.float64,
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

        cv0 = np.zeros(
            B,
            dtype=np.float64,
        )

        cv1 = np.zeros(
            B,
            dtype=np.float64,
        )

        improve_count = np.zeros(
            B,
            dtype=np.uint8,
        )

        for f in range(nfold):

            test_idx = np.flatnonzero(
                fold_id == f
            )

            train_idx = np.flatnonzero(
                fold_id != f
            )

            pred0, pred1, k, retained = (
                fit_train_predict(
                    y[:, train_idx],
                    s[:, train_idx],
                    years[train_idx],
                    years[test_idx],
                    s[:, test_idx],
                )
            )

            obs = y[:, test_idx]

            e0 = (
                obs - pred0
            )

            e1 = (
                obs - pred1
            )

            fold_sse0 = np.sum(
                e0 * e0,
                axis=1,
            )

            fold_sse1 = np.sum(
                e1 * e1,
                axis=1,
            )

            good = (
                finite
                &
                np.isfinite(k)
                &
                np.isfinite(retained)
            )

            degenerate += int(
                np.count_nonzero(
                    finite & ~good
                )
            )

            cv0[good] += (
                fold_sse0[good]
            )

            cv1[good] += (
                fold_sse1[good]
            )

            improve_count[
                good
            ] += (
                fold_sse1[good]
                <
                fold_sse0[good]
            ).astype(
                np.uint8
            )

            kfold[
                b0:b1,
                f
            ] = k.astype(
                np.float32
            )

            retained_fold[
                b0:b1,
                f
            ] = retained.astype(
                np.float32
            )

        cv0[~finite] = np.nan
        cv1[~finite] = np.nan

        sse0_out[
            b0:b1
        ] = cv0.astype(
            np.float32
        )

        sse1_out[
            b0:b1
        ] = cv1.astype(
            np.float32
        )

        improve_count_out[
            b0:b1
        ] = improve_count

        print(
            f"  {b1:,}/{npoint:,}"
        )

    for x in (
        kfold,
        retained_fold,
        sse0_out,
        sse1_out,
        improve_count_out,
    ):
        x.flush()

    # ========================================================
    # Global metrics
    # ========================================================

    K = np.asarray(
        kfold,
        dtype=np.float64,
    )

    R = np.asarray(
        retained_fold,
        dtype=np.float64,
    )

    sse0 = np.asarray(
        sse0_out,
        dtype=np.float64,
    )

    sse1 = np.asarray(
        sse1_out,
        dtype=np.float64,
    )

    improve_count = np.asarray(
        improve_count_out,
        dtype=np.int16,
    )

    k_full = np.asarray(
        full_k,
        dtype=np.float64,
    )

    valid = (
        np.all(np.isfinite(K), axis=1)
        &
        np.all(np.isfinite(R), axis=1)
        &
        np.isfinite(sse0)
        &
        np.isfinite(sse1)
        &
        np.isfinite(k_full)
        &
        (sse0 > 1e-15)
    )

    nvalid = int(
        np.count_nonzero(valid)
    )

    # --------------------------------------------------------
    # CV prediction
    # --------------------------------------------------------

    cv_r2 = np.full(
        npoint,
        np.nan,
        dtype=np.float64,
    )

    cv_r2[valid] = (
        1.0
        -
        sse1[valid]
        /
        sse0[valid]
    )

    cv_rms0 = np.full(
        npoint,
        np.nan,
        dtype=np.float64,
    )

    cv_rms1 = np.full(
        npoint,
        np.nan,
        dtype=np.float64,
    )

    cv_rms0[valid] = np.sqrt(
        sse0[valid] / ndate
    )

    cv_rms1[valid] = np.sqrt(
        sse1[valid] / ndate
    )

    # --------------------------------------------------------
    # k stability
    # --------------------------------------------------------

    k_med = np.nanmedian(
        K,
        axis=1,
    )

    k_mad = np.nanmedian(
        np.abs(
            K - k_med[:, None]
        ),
        axis=1,
    )

    k_range = (
        np.nanmax(K, axis=1)
        -
        np.nanmin(K, axis=1)
    )

    full_diff = np.abs(
        k_med
        -
        k_full
    )

    sign_full = np.sign(
        k_full
    )

    same_sign = np.mean(
        np.sign(K)
        ==
        sign_full[:, None],
        axis=1,
    )

    internal_sign = np.abs(
        np.mean(
            np.sign(K),
            axis=1,
        )
    )

    min_retained = np.nanmin(
        R,
        axis=1,
    )

    # ========================================================
    # Save derived metrics
    # ========================================================

    np.save(
        outdir
        / "cv_incremental_r2.npy",
        cv_r2.astype(np.float32),
    )

    np.save(
        outdir
        / "cv_rms_reduced_rad.npy",
        cv_rms0.astype(np.float32),
    )

    np.save(
        outdir
        / "cv_rms_scla_rad.npy",
        cv_rms1.astype(np.float32),
    )

    np.save(
        outdir
        / "k_cv_median_m.npy",
        k_med.astype(np.float32),
    )

    np.save(
        outdir
        / "k_cv_mad_m.npy",
        k_mad.astype(np.float32),
    )

    np.save(
        outdir
        / "k_cv_range_m.npy",
        k_range.astype(np.float32),
    )

    np.save(
        outdir
        / "k_cv_vs_full_absdiff_m.npy",
        full_diff.astype(np.float32),
    )

    np.save(
        outdir
        / "k_same_sign_as_full_fraction.npy",
        same_sign.astype(np.float32),
    )

    np.save(
        outdir
        / "k_internal_sign_consistency.npy",
        internal_sign.astype(np.float32),
    )

    np.save(
        outdir
        / "minimum_training_identifiable_fraction.npy",
        min_retained.astype(np.float32),
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Training-subset identifiability"
    )
    print("=" * 112)

    qprint(
        "minimum retained fraction "
        "p01/p05/p50/p95/p99:",
        min_retained[valid],
    )

    print()
    print("=" * 112)
    print(
        "Coefficient stability"
    )
    print("=" * 112)

    qprint(
        "CV median k p01/p05/p50/p95/p99 [m]:",
        k_med[valid],
        fmt=".3f",
    )

    qprint(
        "k MAD p50/p90/p95/p99/max [m]:",
        k_mad[valid],
        [50,90,95,99,100],
        ".3f",
    )

    qprint(
        "k fold range p50/p90/p95/p99/max [m]:",
        k_range[valid],
        [50,90,95,99,100],
        ".3f",
    )

    qprint(
        "|CV median k - full k| "
        "p50/p90/p95/p99/max [m]:",
        full_diff[valid],
        [50,90,95,99,100],
        ".3f",
    )

    qprint(
        "same-sign-as-full fraction "
        "p01/p05/p50/p95/p99:",
        same_sign[valid],
    )

    for threshold in (
        0.6,
        0.8,
        1.0,
    ):

        n = int(
            np.count_nonzero(
                same_sign[valid]
                >= threshold
            )
        )

        print(
            f"same-sign fraction >= {threshold:.1f}: "
            f"{n:8,d} "
            f"({100*n/nvalid:.3f}%)"
        )

    print()
    print("=" * 112)
    print(
        "Held-out prediction"
    )
    print("=" * 112)

    qprint(
        "CV RMS reduced p50/p90/p95/p99 [rad]:",
        cv_rms0[valid],
        [50,90,95,99],
    )

    qprint(
        "CV RMS SCLA    p50/p90/p95/p99 [rad]:",
        cv_rms1[valid],
        [50,90,95,99],
    )

    qprint(
        "CV incremental R2 "
        "p01/p05/p50/p95/p99:",
        cv_r2[valid],
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
        0,
        nfold + 1,
    ):

        n = int(
            np.count_nonzero(
                improve_count[valid]
                == nf
            )
        )

        print(
            f"held-out improved folds = {nf}: "
            f"{n:8,d} "
            f"({100*n/nvalid:.3f}%)"
        )

    print()
    print("=" * 112)
    print(
        "Numerical QA"
    )
    print("=" * 112)

    print(
        f"valid points               : "
        f"{nvalid:,}/{npoint:,}"
    )

    print(
        f"nonfinite input points     : "
        f"{nonfinite:,}"
    )

    print(
        f"degenerate fold fits       : "
        f"{degenerate:,}"
    )

    # Numerical status only.
    if (
        nonfinite != 0
        or
        degenerate != 0
        or
        nvalid != npoint
    ):
        status = "REVIEW_INVALID_CV"
    else:
        status = "PASS"

    manifest = {
        "format":
            "pyPSDS-GAMMA-scla-temporal-cv-v09",

        "status":
            status,

        "points":
            int(npoint),

        "acquisitions":
            int(ndate),

        "folds":
            int(nfold),

        "fold_scheme":
            "interleaved acquisition-index modulo folds",

        "model_reduced":
            "phase = intercept + linear_time",

        "model_scla":
            (
                "phase = intercept + linear_time "
                "+ k*sensitivity"
            ),

        "phase_source":
            str(phase_path),

        "sensitivity_source":
            str(sensitivity_path),

        "scientific_decision":
            False,

        "phase_modified":
            False,

        "residual_dem_correction_applied":
            False,
    }

    manifest_path = (
        outdir
        / "scla_temporal_cv_manifest.json"
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
        f"STEP 10b3 STATUS: {status} / "
        "TEMPORAL CV AUDIT ONLY"
    )

    print(
        "No SCLA or residual DEM correction "
        "has been applied."
    )


if __name__ == "__main__":
    main()
