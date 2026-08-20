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


def qprint(title, x, qs, fmt=".6e"):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    q = np.percentile(x, qs)
    print(title)
    print("  " + " / ".join(format(v, fmt) for v in q))


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=65536)
    args = ap.parse_args()

    cfg, config_path, paths, stack, _ = open_from_config(
        args.config
    )

    root = Path(paths.output_dir) / "v09"

    invdir = root / "network_inversion_v09"

    sensitivity_dir = (
        root
        / "scla_v09"
        / "production_sensitivity"
    )

    ident_dir = (
        root
        / "scla_v09"
        / "scla_identifiability"
    )

    outdir = (
        root
        / "scla_v09"
        / "scla_candidate_fit"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Require upstream QA
    # ========================================================

    ident_manifest = json.loads(
        (
            ident_dir
            / "scla_identifiability_manifest.json"
        ).read_text()
    )

    if ident_manifest["status"] != "PASS":
        raise RuntimeError(
            "Step10b1 identifiability did not PASS"
        )

    batch_manifest = json.loads(
        (
            sensitivity_dir
            / "batch_context_audit"
            / "batch_context_repeatability_manifest.json"
        ).read_text()
    )

    if not str(batch_manifest["status"]).startswith("PASS"):
        raise RuntimeError(
            "Step10a5c did not PASS"
        )

    # ========================================================
    # Input phase:
    #
    # IMPORTANT:
    # Use Step09a pre-spatial-reference acquisition phase.
    # Do NOT use Step09c referenced phase for SCLA fitting.
    # ========================================================

    phase_path = (
        invdir
        / "acquisition_phase_l2_candidate_rad.npy"
    )

    sensitivity_path = (
        sensitivity_dir
        / "topographic_phase_sensitivity_rad_per_m.npy"
    )

    strict_ids_path = (
        invdir
        / "strict_point_ids.npy"
    )

    Y = np.load(
        phase_path,
        mmap_mode="r",
    )

    S = np.load(
        sensitivity_path,
        mmap_mode="r",
    )

    strict_ids = np.load(
        strict_ids_path
    ).astype(
        np.int32,
        copy=False,
    )

    npoint = strict_ids.size
    ndate = len(stack.dates)

    if Y.shape != (npoint, ndate):
        raise RuntimeError(
            f"Phase shape mismatch: {Y.shape}"
        )

    if S.shape != (npoint, ndate):
        raise RuntimeError(
            f"Sensitivity shape mismatch: {S.shape}"
        )

    # ========================================================
    # Correct time axis
    # ========================================================

    dates = np.asarray(
        [parse_date(x) for x in stack.dates],
        dtype="datetime64[D]",
    )

    days = (
        dates - dates[0]
    ).astype(
        "timedelta64[D]"
    ).astype(
        np.float64
    )

    years = days / 365.2425

    tc = years - np.mean(years)

    ss_t = float(
        np.sum(tc * tc)
    )

    if ss_t <= 0:
        raise RuntimeError("Invalid time axis")

    # ========================================================
    # Outputs
    # ========================================================

    def mmap(name):
        return np.lib.format.open_memmap(
            outdir / name,
            mode="w+",
            dtype=np.float32,
            shape=(npoint,),
        )

    k_out = mmap(
        "scla_sensitivity_coefficient_m.npy"
    )

    k_se_out = mmap(
        "scla_coefficient_formal_se_m.npy"
    )

    t_out = mmap(
        "scla_coefficient_t_stat.npy"
    )

    rms0_out = mmap(
        "linear_model_residual_rms_rad.npy"
    )

    rms1_out = mmap(
        "scla_model_residual_rms_rad.npy"
    )

    r2_out = mmap(
        "scla_incremental_r2.npy"
    )

    v0_out = mmap(
        "linear_phase_rate_rad_per_year.npy"
    )

    v1_out = mmap(
        "scla_model_phase_rate_rad_per_year.npy"
    )

    dv_out = mmap(
        "scla_induced_phase_rate_change_rad_per_year.npy"
    )

    component_out = mmap(
        "scla_time_orthogonal_component_rms_rad.npy"
    )

    # ========================================================
    # QA accumulators
    # ========================================================

    nonfinite_points = 0
    degenerate_points = 0
    sse_increase_points = 0

    first_phase_max = 0.0
    first_sens_max = 0.0

    max_residual_mean = 0.0
    max_residual_time_corr = 0.0
    max_residual_sens_corr = 0.0

    print("=" * 112)
    print(
        "Step 10b2 - Full-scene SCLA candidate fit"
    )
    print("=" * 112)

    print(f"config                     : {config_path}")
    print(f"strict points              : {npoint:,}")
    print(f"acquisitions               : {ndate}")
    print(f"time span                  : {years[-1]:.4f} yr")
    print(f"phase input                : {phase_path}")
    print(f"sensitivity input          : {sensitivity_path}")
    print(f"batch size                 : {args.batch_size:,}")

    print()
    print(
        "Model: phase = intercept + linear_time "
        "+ k * topographic_sensitivity"
    )

    print(
        "k is a sensitivity coefficient in metres; "
        "physical DEM-error sign is NOT assigned yet."
    )

    # ========================================================
    # Streaming fit
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

        nonfinite_points += int(
            np.count_nonzero(~finite)
        )

        if np.any(finite):
            first_phase_max = max(
                first_phase_max,
                float(
                    np.max(
                        np.abs(y[finite, 0])
                    )
                ),
            )

            first_sens_max = max(
                first_sens_max,
                float(
                    np.max(
                        np.abs(s[finite, 0])
                    )
                ),
            )

        # ----------------------------------------------------
        # Reduced model:
        #
        # y = a + v*t
        # ----------------------------------------------------

        ymean = np.mean(
            y,
            axis=1,
            keepdims=True,
        )

        yc = y - ymean

        cov_yt = np.sum(
            yc * tc[None, :],
            axis=1,
        )

        v0 = cov_yt / ss_t

        y_perp = (
            yc
            -
            v0[:, None]
            *
            tc[None, :]
        )

        sse0 = np.sum(
            y_perp * y_perp,
            axis=1,
        )

        # ----------------------------------------------------
        # Residualize S against [1,t]
        # ----------------------------------------------------

        smean = np.mean(
            s,
            axis=1,
            keepdims=True,
        )

        sc = s - smean

        cov_st = np.sum(
            sc * tc[None, :],
            axis=1,
        )

        slope_st = (
            cov_st / ss_t
        )

        s_perp = (
            sc
            -
            slope_st[:, None]
            *
            tc[None, :]
        )

        ss_sperp = np.sum(
            s_perp * s_perp,
            axis=1,
        )

        good = (
            finite
            &
            np.isfinite(ss_sperp)
            &
            (ss_sperp > 1.0e-12)
        )

        degenerate_points += int(
            np.count_nonzero(
                finite & ~good
            )
        )

        k = np.full(
            B,
            np.nan,
            dtype=np.float64,
        )

        # FWL coefficient
        numerator = np.sum(
            s_perp * y_perp,
            axis=1,
        )

        k[good] = (
            numerator[good]
            /
            ss_sperp[good]
        )

        # ----------------------------------------------------
        # Full model after removing k*S
        # ----------------------------------------------------

        y_minus = (
            y
            -
            k[:, None] * s
        )

        mean_minus = np.mean(
            y_minus,
            axis=1,
            keepdims=True,
        )

        yc_minus = (
            y_minus
            -
            mean_minus
        )

        cov_minus_t = np.sum(
            yc_minus
            *
            tc[None, :],
            axis=1,
        )

        v1 = (
            cov_minus_t
            /
            ss_t
        )

        res = (
            yc_minus
            -
            v1[:, None]
            *
            tc[None, :]
        )

        sse1 = np.sum(
            res * res,
            axis=1,
        )

        # Numerical tolerance only.
        increased = (
            good
            &
            (
                sse1
                >
                sse0
                +
                np.maximum(
                    1.0e-10,
                    1.0e-10 * sse0,
                )
            )
        )

        sse_increase_points += int(
            np.count_nonzero(
                increased
            )
        )

        rms0 = np.sqrt(
            sse0 / ndate
        )

        rms1 = np.sqrt(
            sse1 / ndate
        )

        inc_r2 = np.full(
            B,
            np.nan,
            dtype=np.float64,
        )

        positive_sse0 = (
            good
            &
            (
                sse0 > 1.0e-15
            )
        )

        inc_r2[
            positive_sse0
        ] = (
            1.0
            -
            sse1[
                positive_sse0
            ]
            /
            sse0[
                positive_sse0
            ]
        )

        # ----------------------------------------------------
        # Formal OLS uncertainty.
        #
        # Diagnostic only:
        # temporal residuals need not be iid.
        # ----------------------------------------------------

        dof = ndate - 3

        sigma2 = np.full(
            B,
            np.nan,
            dtype=np.float64,
        )

        sigma2[good] = (
            sse1[good]
            /
            dof
        )

        k_se = np.full(
            B,
            np.nan,
            dtype=np.float64,
        )

        k_se[good] = np.sqrt(
            sigma2[good]
            /
            ss_sperp[good]
        )

        tstat = np.full(
            B,
            np.nan,
            dtype=np.float64,
        )

        okse = (
            good
            &
            np.isfinite(k_se)
            &
            (k_se > 0)
        )

        tstat[okse] = (
            k[okse]
            /
            k_se[okse]
        )

        dv = (
            v1 - v0
        )

        # Distinguishable SCLA component after removing the
        # part degenerate with intercept + linear time.
        sperp_rms = np.sqrt(
            ss_sperp / ndate
        )

        component_rms = (
            np.abs(k)
            *
            sperp_rms
        )

        # ----------------------------------------------------
        # OLS normal-equation QA
        # ----------------------------------------------------

        if np.any(good):

            rg = res[good]

            residual_mean = np.abs(
                np.mean(
                    rg,
                    axis=1,
                )
            )

            max_residual_mean = max(
                max_residual_mean,
                float(
                    np.max(
                        residual_mean
                    )
                ),
            )

            sse_g = np.sum(
                rg * rg,
                axis=1,
            )

            denom_t = np.sqrt(
                np.maximum(
                    sse_g * ss_t,
                    1.0e-30,
                )
            )

            time_corr = np.abs(
                np.sum(
                    rg
                    *
                    tc[None, :],
                    axis=1,
                )
                /
                denom_t
            )

            max_residual_time_corr = max(
                max_residual_time_corr,
                float(
                    np.max(
                        time_corr
                    )
                ),
            )

            sg = s_perp[good]

            denom_s = np.sqrt(
                np.maximum(
                    sse_g
                    *
                    ss_sperp[good],
                    1.0e-30,
                )
            )

            sens_corr = np.abs(
                np.sum(
                    rg * sg,
                    axis=1,
                )
                /
                denom_s
            )

            max_residual_sens_corr = max(
                max_residual_sens_corr,
                float(
                    np.max(
                        sens_corr
                    )
                ),
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        k_out[b0:b1] = k.astype(np.float32)
        k_se_out[b0:b1] = k_se.astype(np.float32)
        t_out[b0:b1] = tstat.astype(np.float32)

        rms0_out[b0:b1] = rms0.astype(np.float32)
        rms1_out[b0:b1] = rms1.astype(np.float32)

        r2_out[b0:b1] = inc_r2.astype(np.float32)

        v0_out[b0:b1] = v0.astype(np.float32)
        v1_out[b0:b1] = v1.astype(np.float32)
        dv_out[b0:b1] = dv.astype(np.float32)

        component_out[
            b0:b1
        ] = component_rms.astype(
            np.float32
        )

        print(
            f"  {b1:,}/{npoint:,}"
        )

    outputs = (
        k_out,
        k_se_out,
        t_out,
        rms0_out,
        rms1_out,
        r2_out,
        v0_out,
        v1_out,
        dv_out,
        component_out,
    )

    for x in outputs:
        x.flush()

    # ========================================================
    # Global distributions
    # ========================================================

    k = np.asarray(
        k_out,
        dtype=np.float64,
    )

    k_se = np.asarray(
        k_se_out,
        dtype=np.float64,
    )

    tstat = np.asarray(
        t_out,
        dtype=np.float64,
    )

    rms0 = np.asarray(
        rms0_out,
        dtype=np.float64,
    )

    rms1 = np.asarray(
        rms1_out,
        dtype=np.float64,
    )

    inc_r2 = np.asarray(
        r2_out,
        dtype=np.float64,
    )

    dv = np.asarray(
        dv_out,
        dtype=np.float64,
    )

    component = np.asarray(
        component_out,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(k)
        &
        np.isfinite(k_se)
        &
        np.isfinite(tstat)
        &
        np.isfinite(rms0)
        &
        np.isfinite(rms1)
        &
        np.isfinite(inc_r2)
        &
        np.isfinite(dv)
        &
        np.isfinite(component)
    )

    nvalid = int(
        np.count_nonzero(valid)
    )

    print()
    print("=" * 112)
    print(
        "SCLA sensitivity coefficient"
    )
    print("=" * 112)

    qprint(
        "k p01/p05/p50/p95/p99 [m]:",
        k[valid],
        [1, 5, 50, 95, 99],
        ".3f",
    )

    qprint(
        "|k| p50/p90/p95/p99/max [m]:",
        np.abs(k[valid]),
        [50, 90, 95, 99, 100],
        ".3f",
    )

    qprint(
        "formal SE(k) p50/p90/p95/p99 [m]:",
        k_se[valid],
        [50, 90, 95, 99],
        ".3f",
    )

    for threshold in (
        10,
        20,
        50,
        100,
    ):

        n = int(
            np.count_nonzero(
                np.abs(k[valid])
                >=
                threshold
            )
        )

        print(
            f"|k| >= {threshold:3d} m       : "
            f"{n:8,d} "
            f"({100*n/nvalid:.3f}%)"
        )

    print()
    print("=" * 112)
    print(
        "Fit improvement"
    )
    print("=" * 112)

    qprint(
        "RMS before p50/p90/p95/p99 [rad]:",
        rms0[valid],
        [50, 90, 95, 99],
    )

    qprint(
        "RMS after  p50/p90/p95/p99 [rad]:",
        rms1[valid],
        [50, 90, 95, 99],
    )

    improvement = (
        rms0[valid]
        -
        rms1[valid]
    )

    qprint(
        "absolute RMS reduction p50/p90/p95/p99 [rad]:",
        improvement,
        [50, 90, 95, 99],
    )

    qprint(
        "incremental R2 p01/p05/p50/p95/p99:",
        inc_r2[valid],
        [1, 5, 50, 95, 99],
        ".6f",
    )

    for threshold in (
        0.01,
        0.05,
        0.10,
        0.20,
        0.50,
    ):

        n = int(
            np.count_nonzero(
                inc_r2[valid]
                >=
                threshold
            )
        )

        print(
            f"incremental R2 >= {threshold:.2f}: "
            f"{n:8,d} "
            f"({100*n/nvalid:.3f}%)"
        )

    print()
    print("=" * 112)
    print(
        "Coefficient significance diagnostics"
    )
    print("=" * 112)

    qprint(
        "|t| p50/p90/p95/p99/max:",
        np.abs(tstat[valid]),
        [50, 90, 95, 99, 100],
        ".3f",
    )

    for threshold in (
        2,
        3,
        5,
    ):

        n = int(
            np.count_nonzero(
                np.abs(
                    tstat[valid]
                )
                >= threshold
            )
        )

        print(
            f"|t| >= {threshold}               : "
            f"{n:8,d} "
            f"({100*n/nvalid:.3f}%)"
        )

    print()
    print("=" * 112)
    print(
        "Interaction with linear phase rate"
    )
    print("=" * 112)

    qprint(
        "velocity change p01/p05/p50/p95/p99 [rad/yr]:",
        dv[valid],
        [1, 5, 50, 95, 99],
    )

    qprint(
        "|velocity change| p50/p90/p95/p99/max [rad/yr]:",
        np.abs(dv[valid]),
        [50, 90, 95, 99, 100],
    )

    qprint(
        "time-orthogonal fitted SCLA RMS "
        "p50/p90/p95/p99 [rad]:",
        component[valid],
        [50, 90, 95, 99],
    )

    print()
    print("=" * 112)
    print(
        "Numerical QA"
    )
    print("=" * 112)

    print(
        f"valid fitted points        : "
        f"{nvalid:,}/{npoint:,}"
    )

    print(
        f"nonfinite input points     : "
        f"{nonfinite_points:,}"
    )

    print(
        f"degenerate points          : "
        f"{degenerate_points:,}"
    )

    print(
        f"SSE increase points        : "
        f"{sse_increase_points:,}"
    )

    print(
        f"first acquisition max |phase|:"
        f" {first_phase_max:.3e} rad"
    )

    print(
        f"first acquisition max |S|  : "
        f"{first_sens_max:.3e} rad/m"
    )

    print(
        f"max residual mean          : "
        f"{max_residual_mean:.3e} rad"
    )

    print(
        f"max residual-time corr     : "
        f"{max_residual_time_corr:.3e}"
    )

    print(
        f"max residual-Sperp corr    : "
        f"{max_residual_sens_corr:.3e}"
    )

    # ========================================================
    # Status: numerical validity only.
    #
    # Scientific acceptance of SCLA correction is NOT made
    # here.
    # ========================================================

    if (
        nonfinite_points != 0
        or
        degenerate_points != 0
    ):
        status = "REVIEW_INVALID_FIT"

    elif sse_increase_points != 0:
        status = "REVIEW_OLS_INCONSISTENCY"

    elif (
        first_phase_max > 1.0e-5
        or
        first_sens_max > 1.0e-7
    ):
        status = "REVIEW_TEMPORAL_REFERENCE"

    elif (
        max_residual_time_corr > 1.0e-8
        or
        max_residual_sens_corr > 1.0e-8
    ):
        status = "REVIEW_NORMAL_EQUATIONS"

    else:
        status = "PASS"

    manifest = {
        "format":
            "pyPSDS-GAMMA-scla-candidate-fit-v09",

        "status":
            status,

        "points":
            int(npoint),

        "acquisitions":
            int(ndate),

        "phase_source":
            str(phase_path),

        "phase_source_stage":
            "Step09a_pre_spatial_reference",

        "sensitivity_source":
            str(sensitivity_path),

        "model":
            "phase = intercept + linear_time + k*sensitivity",

        "solver":
            "Frisch-Waugh-Lovell exact OLS",

        "coefficient_units":
            "m",

        "physical_residual_dem_sign_assigned":
            False,

        "formal_uncertainty_note":
            (
                "OLS standard error assumes iid temporal "
                "residuals and is diagnostic only."
            ),

        "outputs": {
            "coefficient_m":
                str(
                    outdir
                    / "scla_sensitivity_coefficient_m.npy"
                ),

            "formal_se_m":
                str(
                    outdir
                    / "scla_coefficient_formal_se_m.npy"
                ),

            "t_stat":
                str(
                    outdir
                    / "scla_coefficient_t_stat.npy"
                ),

            "rms_before":
                str(
                    outdir
                    / "linear_model_residual_rms_rad.npy"
                ),

            "rms_after":
                str(
                    outdir
                    / "scla_model_residual_rms_rad.npy"
                ),

            "incremental_r2":
                str(
                    outdir
                    / "scla_incremental_r2.npy"
                ),

            "velocity_before":
                str(
                    outdir
                    / "linear_phase_rate_rad_per_year.npy"
                ),

            "velocity_after":
                str(
                    outdir
                    / "scla_model_phase_rate_rad_per_year.npy"
                ),
        },

        "phase_modified":
            False,

        "residual_dem_correction_applied":
            False,

        "scientific_acceptance_decision":
            False,
    }

    manifest_path = (
        outdir
        / "scla_candidate_fit_manifest.json"
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
        f"STEP 10b2 STATUS: {status} / "
        "SCLA CANDIDATE FIT ONLY"
    )

    print(
        "No phase correction has been applied."
    )


if __name__ == "__main__":
    main()
