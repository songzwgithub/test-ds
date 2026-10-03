#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

YEAR_DAYS = 365.25


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict) -> None:
    atomic_json(path, state)


def open_resume_npy(path: Path, dtype, shape):
    if path.is_file():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape or arr.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"resume array contract failed: {path} "
                f"{arr.shape}/{arr.dtype} != {shape}/{np.dtype(dtype)}"
            )
        return arr
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=shape
    )


def percentiles(path: Path, q):
    a = np.load(path, mmap_mode="r")
    x = np.asarray(a, dtype=np.float64)
    return np.percentile(x, q)


def project_tp_la(C: np.ndarray, P: np.ndarray, X: np.ndarray, master0: int):
    """
    Remove constant + linear + annual sin/cos model from the STRICT SCN
    correction, while keeping the correction exactly zero at the geometric
    master epoch.

    Model:
        c(t) = a + b*t + s*sin(2*pi*t) + c*cos(2*pi*t) + residual
    Returned correction contains only the residual component.
    """
    beta = C @ P.T
    fit = beta @ X.T
    fit -= fit[:, master0][:, None]
    out = C - fit
    out[:, master0] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Build full-resolution TP-LA SCN and final Ningbo LOS products."
    )
    ap.add_argument(
        "--root",
        default="/mnt/ningbo_process/ds",
        help="pyPSDS project root",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=131072,
        help="point chunk size",
    )
    ap.add_argument(
        "--qa-sample",
        type=int,
        default=200000,
        help="sample points for final scientific QA",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    proc = root / "output" / "processing"
    strict_prod = root / "output" / "products"

    pre_path = proc / "scla" / "acquisition_phase_pre_scn_rad.npy"
    strict_scn_path = proc / "scn" / "ph_scn_slave_rad.npy"
    strict_scn_manifest_path = proc / "scn" / "scn_manifest.json"
    ref_path = (
        proc
        / "referenced_timeseries"
        / "reference_strict_indices.npy"
    )
    strict_final_manifest_path = (
        proc / "final_los" / "final_los_manifest.json"
    )
    time_contract_path = strict_prod / "time_axis_contract.npz"

    for p in (
        pre_path,
        strict_scn_path,
        strict_scn_manifest_path,
        ref_path,
        strict_final_manifest_path,
        time_contract_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    strict_scn_manifest = json.loads(strict_scn_manifest_path.read_text())
    if strict_scn_manifest.get("status") != "PASS_SCN":
        raise RuntimeError("strict SCN manifest is not PASS_SCN")

    fm = json.loads(strict_final_manifest_path.read_text())
    if fm.get("status") != "PASS_FINAL_LOS":
        raise RuntimeError("strict final LOS manifest is not PASS_FINAL_LOS")

    mm_per_rad = float(
        fm["scientific_contract"]["los_factor_mm_per_rad"]
    )
    master0 = int(
        fm["scientific_contract"]["geometric_master_index_0based"]
    )

    tc = np.load(time_contract_path)
    dates = [str(x) for x in tc["acquisition_dates"]]
    years = np.asarray(
        tc["years_since_reference"], dtype=np.float64
    )
    slope_weights = np.asarray(
        tc["slope_weights_per_year"], dtype=np.float64
    )

    pre = np.load(pre_path, mmap_mode="r")
    strict_scn = np.load(strict_scn_path, mmap_mode="r")

    if pre.shape != strict_scn.shape:
        raise RuntimeError(
            f"pre/SCN shape mismatch: {pre.shape} vs {strict_scn.shape}"
        )

    npoint, nepoch = pre.shape
    if nepoch != len(dates) or nepoch != years.size:
        raise RuntimeError("time-axis size mismatch")
    if not (0 <= master0 < nepoch):
        raise RuntimeError(f"invalid master index {master0}")

    ref_idx = np.load(ref_path).astype(np.int64)
    if ref_idx.min() < 0 or ref_idx.max() >= npoint:
        raise RuntimeError("reference indices outside point domain")

    # Scientific model protected from SCN removal.
    w = 2.0 * np.pi
    X = np.column_stack(
        (
            np.ones(nepoch, dtype=np.float64),
            years,
            np.sin(w * years),
            np.cos(w * years),
        )
    )
    rank = int(np.linalg.matrix_rank(X))
    cond = float(np.linalg.cond(X))
    if rank != 4:
        raise RuntimeError(f"TP-LA design rank={rank}, expected 4")
    P = np.linalg.pinv(X)  # 4 x Nepoch
    XtX_inv = np.linalg.inv(X.T @ X)

    tp_dir = proc / "scn_tp_la"
    final_dir = proc / "final_los_tp_la"
    prod_dir = root / "output" / "products_tp_la"

    for d in (tp_dir, final_dir, prod_dir):
        d.mkdir(parents=True, exist_ok=True)

    state_path = tp_dir / "build_progress.json"
    state = load_state(state_path)

    tp_tmp = tp_dir / ".ph_scn_tp_la_rad.tmp.npy"
    tp_out_path = tp_dir / "ph_scn_tp_la_rad.npy"

    print("=" * 100)
    print("NINGBO TP-LA PRODUCTION")
    print("=" * 100)
    print("points / epochs        :", f"{npoint:,} / {nepoch}")
    print("dates                  :", dates[0], "->", dates[-1])
    print("geometric master       :", dates[master0], f"(0b={master0})")
    print("protected model        : constant + linear + annual sin/cos")
    print("design rank / cond     :", rank, f"/ {cond:.6f}")
    print("LOS factor             :", f"{mm_per_rad:.12f} mm/rad")
    print("reference points       :", f"{ref_idx.size:,}")
    print("chunk                   :", f"{args.chunk:,}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Stage 1: strict SCN -> trend + annual preserving SCN correction
    # ------------------------------------------------------------------
    if not tp_out_path.is_file():
        out = open_resume_npy(
            tp_tmp, np.float32, (npoint, nepoch)
        )
        start0 = int(state.get("tp_la_next", 0))
        if start0 < 0 or start0 > npoint:
            start0 = 0

        t0 = time.perf_counter()
        for start in range(start0, npoint, args.chunk):
            stop = min(start + args.chunk, npoint)
            C = np.asarray(
                strict_scn[start:stop, :], dtype=np.float64
            )
            Ctp = project_tp_la(C, P, X, master0)
            out[start:stop, :] = Ctp.astype(np.float32)
            out.flush()

            state["tp_la_next"] = int(stop)
            save_state(state_path, state)

            print(
                f"[TP-LA SCN] {stop:,}/{npoint:,} "
                f"({100.0*stop/npoint:.1f}%)",
                flush=True,
            )

        out.flush()
        del out
        os.replace(tp_tmp, tp_out_path)
        state["tp_la_next"] = int(npoint)
        save_state(state_path, state)

        print(
            "TP-LA correction seconds:",
            f"{time.perf_counter()-t0:.3f}",
        )
    else:
        print("[TP-LA SCN] existing complete output -> skip")

    tp = np.load(tp_out_path, mmap_mode="r")
    if tp.shape != (npoint, nepoch):
        raise RuntimeError("TP-LA output shape failed")

    # QA: protected model coefficients of the remaining correction.
    ns = min(args.qa_sample, npoint)
    qa_idx = np.linspace(0, npoint - 1, ns, dtype=np.int64)
    tp_qa = np.asarray(tp[qa_idx, :], dtype=np.float64)
    beta_tp = tp_qa @ P.T

    slope_abs_q = np.percentile(
        np.abs(beta_tp[:, 1]), [50, 95, 99]
    )
    sin_abs_q = np.percentile(
        np.abs(beta_tp[:, 2]), [50, 95, 99]
    )
    cos_abs_q = np.percentile(
        np.abs(beta_tp[:, 3]), [50, 95, 99]
    )
    master_max = float(
        np.max(np.abs(tp_qa[:, master0]))
    )

    # float32 storage leaves tiny numerical projection residue.
    if (
        slope_abs_q[-1] > 5e-6
        or sin_abs_q[-1] > 5e-6
        or cos_abs_q[-1] > 5e-6
        or master_max != 0.0
    ):
        raise RuntimeError(
            "TP-LA projection QA failed: "
            f"slope={slope_abs_q}, sin={sin_abs_q}, "
            f"cos={cos_abs_q}, master={master_max}"
        )

    tp_manifest = {
        "status": "PASS_TP_LA_SCN",
        "method": (
            "strict StaMPS SCN correction projected onto complement "
            "of [1, t, sin(2*pi*t), cos(2*pi*t)]"
        ),
        "points": int(npoint),
        "acquisitions": int(nepoch),
        "dates": dates,
        "scientific_contract": {
            "protected_terms": [
                "constant",
                "linear",
                "annual_sin",
                "annual_cos",
            ],
            "year_days": YEAR_DAYS,
            "geometric_master_index_0based": master0,
            "geometric_master_date": dates[master0],
            "master_correction_forced_zero": True,
            "strict_scn_input": str(strict_scn_path),
            "pre_scn_phase": str(pre_path),
        },
        "design": {
            "rank": rank,
            "condition_number": cond,
        },
        "hard_qa": {
            "sample_points": int(ns),
            "remaining_slope_abs_p50_p95_p99_rad_per_year": [
                float(x) for x in slope_abs_q
            ],
            "remaining_annual_sin_abs_p50_p95_p99_rad": [
                float(x) for x in sin_abs_q
            ],
            "remaining_annual_cos_abs_p50_p95_p99_rad": [
                float(x) for x in cos_abs_q
            ],
            "sample_master_max_abs_rad": master_max,
        },
        "output": {
            "path": str(tp_out_path),
            "dtype": "float32",
        },
    }
    atomic_json(tp_dir / "tp_la_manifest.json", tp_manifest)

    print()
    print("TP-LA correction projection QA")
    print("  |slope| p50/p95/p99 :", slope_abs_q)
    print("  |sin|   p50/p95/p99 :", sin_abs_q)
    print("  |cos|   p50/p95/p99 :", cos_abs_q)
    print("  master max |rad|     :", master_max)

    # ------------------------------------------------------------------
    # Spatial reference for TP-LA final phase.
    # Use the STORED float32 TP-LA correction for exact lineage.
    # ------------------------------------------------------------------
    ref_pre = np.asarray(pre[ref_idx, :], dtype=np.float64)
    ref_tp = np.asarray(tp[ref_idx, :], dtype=np.float64)

    ref_raw = ref_pre - ref_tp
    ref_time = ref_raw - ref_raw[:, 0][:, None]
    ref_median = np.median(ref_time, axis=0)
    ref_median[0] = 0.0

    np.save(
        final_dir / "spatial_reference_median_rad.npy",
        ref_median.astype(np.float64),
    )

    # ------------------------------------------------------------------
    # Stage 2: TP-LA final referenced phase + LOS
    # ------------------------------------------------------------------
    phase_out_path = final_dir / "acquisition_phase_final_rad.npy"
    los_m_out_path = final_dir / "los_displacement_toward_satellite_m.npy"
    los_mm_out_path = final_dir / "los_displacement_toward_satellite_mm.npy"

    phase_tmp = final_dir / ".acquisition_phase_final_rad.tmp.npy"
    los_m_tmp = final_dir / ".los_m.tmp.npy"
    los_mm_tmp = final_dir / ".los_mm.tmp.npy"

    final_complete = (
        phase_out_path.is_file()
        and los_m_out_path.is_file()
        and los_mm_out_path.is_file()
    )

    if not final_complete:
        phase_out = open_resume_npy(
            phase_tmp, np.float32, (npoint, nepoch)
        )
        los_m_out = open_resume_npy(
            los_m_tmp, np.float32, (npoint, nepoch)
        )
        los_mm_out = open_resume_npy(
            los_mm_tmp, np.float32, (npoint, nepoch)
        )

        start0 = int(state.get("final_next", 0))
        if start0 < 0 or start0 > npoint:
            start0 = 0

        t0 = time.perf_counter()
        for start in range(start0, npoint, args.chunk):
            stop = min(start + args.chunk, npoint)

            Y = np.asarray(
                pre[start:stop, :], dtype=np.float64
            )
            C = np.asarray(
                tp[start:stop, :], dtype=np.float64
            )

            raw = Y - C
            temporal = raw - raw[:, 0][:, None]
            final = temporal - ref_median[None, :]
            final[:, 0] = 0.0

            los_mm = final * mm_per_rad

            phase_out[start:stop, :] = final.astype(np.float32)
            los_mm_out[start:stop, :] = los_mm.astype(np.float32)
            los_m_out[start:stop, :] = (los_mm / 1000.0).astype(
                np.float32
            )

            phase_out.flush()
            los_mm_out.flush()
            los_m_out.flush()

            state["final_next"] = int(stop)
            save_state(state_path, state)

            print(
                f"[TP-LA FINAL LOS] {stop:,}/{npoint:,} "
                f"({100.0*stop/npoint:.1f}%)",
                flush=True,
            )

        phase_out.flush()
        los_m_out.flush()
        los_mm_out.flush()
        del phase_out, los_m_out, los_mm_out

        os.replace(phase_tmp, phase_out_path)
        os.replace(los_m_tmp, los_m_out_path)
        os.replace(los_mm_tmp, los_mm_out_path)

        state["final_next"] = int(npoint)
        save_state(state_path, state)

        print(
            "final LOS seconds:",
            f"{time.perf_counter()-t0:.3f}",
        )
    else:
        print("[TP-LA FINAL LOS] existing complete outputs -> skip")

    phase_final = np.load(phase_out_path, mmap_mode="r")
    los_mm = np.load(los_mm_out_path, mmap_mode="r")

    epoch0_max = float(
        np.max(np.abs(np.asarray(phase_final[:, 0])))
    )
    ref_final = np.asarray(
        phase_final[ref_idx, :], dtype=np.float64
    )
    ref_median_final = np.median(ref_final, axis=0)
    ref_median_max = float(
        np.max(np.abs(ref_median_final))
    )

    if epoch0_max != 0.0:
        raise RuntimeError(f"TP-LA final epoch0 not zero: {epoch0_max}")
    if ref_median_max > 2e-6:
        raise RuntimeError(
            f"TP-LA final reference median failed: {ref_median_max}"
        )

    final_manifest = {
        "status": "PASS_FINAL_LOS_TP_LA",
        "formula": {
            "scn_correction": "phi_raw = phi_preSCN - phi_SCN_TP_LA",
            "temporal_reference": "phi_t = phi_raw - phi_raw[:,0]",
            "spatial_reference": (
                "phi_final = phi_t - median(phi_t[reference_points], epoch)"
            ),
            "los": "d_LOS_toward = +lambda/(4*pi) * phi_final",
        },
        "scientific_contract": {
            "temporal_reference_date": dates[0],
            "temporal_reference_index_0based": 0,
            "geometric_master_date": dates[master0],
            "geometric_master_index_0based": master0,
            "spatial_reference_points": int(ref_idx.size),
            "los_positive_direction": "toward_satellite",
            "los_factor_mm_per_rad": mm_per_rad,
            "scn_mode": "TP-LA signal-preserving",
        },
        "hard_qa": {
            "epoch0_phase_max_abs_rad": epoch0_max,
            "reference_phase_median_max_abs_rad": ref_median_max,
        },
        "inputs": {
            "pre_scn_phase": str(pre_path),
            "tp_la_scn": str(tp_out_path),
            "reference_indices": str(ref_path),
        },
        "outputs": {
            "phase_rad": str(phase_out_path),
            "los_m": str(los_m_out_path),
            "los_mm": str(los_mm_out_path),
            "dtype": "float32",
        },
    }
    atomic_json(
        final_dir / "final_los_tp_la_manifest.json",
        final_manifest,
    )

    # ------------------------------------------------------------------
    # Stage 3: final scientific products.
    # Primary velocity = linear + annual secular slope.
    # ------------------------------------------------------------------
    output_specs = {
        "velocity_la": (
            prod_dir / "los_velocity_linear_annual_mm_per_year.npy",
            np.float32,
        ),
        "velocity_ols": (
            prod_dir / "los_velocity_ols_mm_per_year.npy",
            np.float32,
        ),
        "annual_sin": (
            prod_dir / "annual_sin_coefficient_mm.npy",
            np.float32,
        ),
        "annual_cos": (
            prod_dir / "annual_cos_coefficient_mm.npy",
            np.float32,
        ),
        "annual_amp": (
            prod_dir / "annual_amplitude_mm.npy",
            np.float32,
        ),
        "annual_phase": (
            prod_dir / "annual_phase_rad.npy",
            np.float32,
        ),
        "annual_peak_lag": (
            prod_dir / "annual_peak_lag_days_from_temporal_reference.npy",
            np.float32,
        ),
        "rms_la": (
            prod_dir / "linear_annual_residual_rms_mm.npy",
            np.float32,
        ),
        "velocity_se": (
            prod_dir
            / "velocity_linear_annual_standard_error_mm_per_year.npy",
            np.float32,
        ),
        "cumulative": (
            prod_dir / "los_cumulative_toward_satellite_mm.npy",
            np.float32,
        ),
    }

    products_complete = all(
        path.is_file()
        for path, _dtype in output_specs.values()
    )

    if not products_complete:
        outs = {}
        tmps = {}

        for name, (final_path, dtype) in output_specs.items():
            tmp_path = final_path.with_name(
                "." + final_path.name + ".tmp.npy"
            )
            tmps[name] = tmp_path
            outs[name] = open_resume_npy(
                tmp_path, dtype, (npoint,)
            )

        start0 = int(state.get("products_next", 0))
        if start0 < 0 or start0 > npoint:
            start0 = 0

        t0 = time.perf_counter()
        dof = nepoch - X.shape[1]
        if dof <= 0:
            raise RuntimeError("non-positive LA regression DOF")

        for start in range(start0, npoint, args.chunk):
            stop = min(start + args.chunk, npoint)

            Y = np.asarray(
                los_mm[start:stop, :], dtype=np.float64
            )

            beta = Y @ P.T

            vel_la = beta[:, 1]
            annual_s = beta[:, 2]
            annual_c = beta[:, 3]

            annual_amp = np.sqrt(
                annual_s * annual_s
                + annual_c * annual_c
            )

            # y = A*sin(2*pi*t + phase)
            phase = np.arctan2(
                annual_c, annual_s
            )

            peak_fraction = np.mod(
                (0.5 * np.pi - phase)
                / (2.0 * np.pi),
                1.0,
            )
            peak_lag_days = (
                peak_fraction * YEAR_DAYS
            )

            fit = beta @ X.T
            resid = Y - fit
            sse = np.sum(resid * resid, axis=1)
            rms = np.sqrt(sse / nepoch)

            sigma2 = sse / dof
            vel_se = np.sqrt(
                sigma2 * XtX_inv[1, 1]
            )

            vel_ols = Y @ slope_weights
            cumulative = Y[:, -1] - Y[:, 0]

            arrays = {
                "velocity_la": vel_la,
                "velocity_ols": vel_ols,
                "annual_sin": annual_s,
                "annual_cos": annual_c,
                "annual_amp": annual_amp,
                "annual_phase": phase,
                "annual_peak_lag": peak_lag_days,
                "rms_la": rms,
                "velocity_se": vel_se,
                "cumulative": cumulative,
            }

            for name, x in arrays.items():
                if not np.all(np.isfinite(x)):
                    raise RuntimeError(
                        f"non-finite {name} at {start}:{stop}"
                    )
                outs[name][start:stop] = x.astype(np.float32)

            for a in outs.values():
                a.flush()

            state["products_next"] = int(stop)
            save_state(state_path, state)

            print(
                f"[TP-LA PRODUCTS] {stop:,}/{npoint:,} "
                f"({100.0*stop/npoint:.1f}%)",
                flush=True,
            )

        for a in outs.values():
            a.flush()
        del outs

        for name, (final_path, _dtype) in output_specs.items():
            os.replace(tmps[name], final_path)

        state["products_next"] = int(npoint)
        save_state(state_path, state)

        print(
            "point products seconds:",
            f"{time.perf_counter()-t0:.3f}",
        )
    else:
        print("[TP-LA PRODUCTS] existing complete outputs -> skip")

    # ------------------------------------------------------------------
    # Final science QA: compare TP-LA to no-SCN on same sample/reference.
    # ------------------------------------------------------------------
    vel_path = output_specs["velocity_la"][0]
    amp_path = output_specs["annual_amp"][0]
    rms_path = output_specs["rms_la"][0]
    cum_path = output_specs["cumulative"][0]
    vse_path = output_specs["velocity_se"][0]

    V = np.load(vel_path, mmap_mode="r")
    AMP = np.load(amp_path, mmap_mode="r")
    RMS = np.load(rms_path, mmap_mode="r")
    CUM = np.load(cum_path, mmap_mode="r")
    VSE = np.load(vse_path, mmap_mode="r")

    ref_pre0 = ref_pre - ref_pre[:, 0][:, None]
    ref_no_scn_median = np.median(ref_pre0, axis=0)

    pre_qa = np.asarray(pre[qa_idx, :], dtype=np.float64)
    no_scn_phase = (
        pre_qa
        - pre_qa[:, 0][:, None]
        - ref_no_scn_median[None, :]
    )
    no_scn_phase[:, 0] = 0.0
    no_scn_los = no_scn_phase * mm_per_rad

    beta0 = no_scn_los @ P.T
    v0 = beta0[:, 1]
    amp0 = np.sqrt(beta0[:, 2] ** 2 + beta0[:, 3] ** 2)
    fit0 = beta0 @ X.T
    rms0 = np.sqrt(
        np.mean((no_scn_los - fit0) ** 2, axis=1)
    )

    v1 = np.asarray(V[qa_idx], dtype=np.float64)
    amp1 = np.asarray(AMP[qa_idx], dtype=np.float64)
    rms1 = np.asarray(RMS[qa_idx], dtype=np.float64)

    corr_v = float(np.corrcoef(v0, v1)[0, 1])
    dv_abs_q = np.percentile(
        np.abs(v1 - v0), [50, 95, 99]
    )
    damp_abs_q = np.percentile(
        np.abs(amp1 - amp0), [50, 95, 99]
    )
    rms_ratio = float(np.median(rms1 / rms0))
    rms_improved = float(np.mean(rms1 < rms0))

    if corr_v < 0.999999:
        raise RuntimeError(
            f"TP-LA velocity preservation failed: corr={corr_v}"
        )
    if dv_abs_q[-1] > 0.05:
        raise RuntimeError(
            f"TP-LA velocity delta too large: {dv_abs_q}"
        )
    if damp_abs_q[-1] > 0.10:
        raise RuntimeError(
            f"TP-LA annual amplitude delta too large: {damp_abs_q}"
        )

    statistics = {
        "velocity_linear_annual_p01_p05_p50_p95_p99_mm_per_year": [
            float(x)
            for x in percentiles(vel_path, [1, 5, 50, 95, 99])
        ],
        "annual_amplitude_p50_p90_p95_p99_mm": [
            float(x)
            for x in percentiles(amp_path, [50, 90, 95, 99])
        ],
        "linear_annual_residual_rms_p50_p95_p99_mm": [
            float(x)
            for x in percentiles(rms_path, [50, 95, 99])
        ],
        "velocity_standard_error_p50_p95_p99_mm_per_year": [
            float(x)
            for x in percentiles(vse_path, [50, 95, 99])
        ],
        "cumulative_p01_p05_p50_p95_p99_mm": [
            float(x)
            for x in percentiles(cum_path, [1, 5, 50, 95, 99])
        ],
    }

    product_manifest = {
        "status": "PASS_POINT_PRODUCTS_TP_LA",
        "points": int(npoint),
        "acquisitions": int(nepoch),
        "scientific_contract": {
            "primary_velocity_model": (
                "LOS = intercept + secular_velocity*t "
                "+ annual_sin*sin(2*pi*t) "
                "+ annual_cos*cos(2*pi*t)"
            ),
            "primary_velocity_unit": "mm/year",
            "year_days": YEAR_DAYS,
            "annual_amplitude": "sqrt(sin_coeff^2 + cos_coeff^2)",
            "annual_phase": (
                "phase in y=A*sin(2*pi*t + phase), "
                "t measured from temporal reference"
            ),
            "annual_peak_lag": (
                "days after temporal reference within one 365.25-day cycle"
            ),
            "cumulative": "last acquisition minus first acquisition",
            "los_positive": "toward_satellite",
            "temporal_reference_date": dates[0],
            "SCN": "TP-LA signal-preserving SCN",
        },
        "hard_scientific_qa": {
            "sample_points": int(ns),
            "velocity_noSCN_vs_TPLA_correlation": corr_v,
            "abs_velocity_delta_p50_p95_p99_mm_per_year": [
                float(x) for x in dv_abs_q
            ],
            "abs_annual_amplitude_delta_p50_p95_p99_mm": [
                float(x) for x in damp_abs_q
            ],
            "median_linear_annual_RMS_ratio_TPLA_over_noSCN": rms_ratio,
            "fraction_linear_annual_RMS_improved": rms_improved,
        },
        "statistics": statistics,
        "outputs": {
            name: str(path)
            for name, (path, _dtype) in output_specs.items()
        },
        "geometry": {
            "longitude": str(
                proc / "point_geometry" / "longitude_deg.npy"
            ),
            "latitude": str(
                proc / "point_geometry" / "latitude_deg.npy"
            ),
            "reference_indices": str(ref_path),
        },
    }

    atomic_json(
        prod_dir / "point_products_tp_la_manifest.json",
        product_manifest,
    )

    np.savez(
        prod_dir / "time_axis_and_model_contract.npz",
        acquisition_dates=np.asarray(dates, dtype="U8"),
        years_since_reference=years,
        design_linear_annual=X,
        design_pseudoinverse=P,
        xtx_inverse=XtX_inv,
        temporal_reference_date=np.asarray(dates[0]),
        geometric_master_date=np.asarray(dates[master0]),
        year_days=np.asarray(YEAR_DAYS),
    )

    print()
    print("=" * 100)
    print("FINAL TP-LA SCIENTIFIC PRODUCTS")
    print("=" * 100)
    print("velocity model      : linear + annual")
    print("corr(noSCN, TP-LA)  :", f"{corr_v:.12f}")
    print("|dv| p50/p95/p99    :", dv_abs_q)
    print("|dAnnual| p50/95/99 :", damp_abs_q)
    print("median RMS ratio     :", f"{rms_ratio:.12f}")
    print("fraction RMS improved:", f"{rms_improved:.12f}")
    print()
    for k, v in statistics.items():
        print(k, ":", v)
    print()
    print("TP-LA SCN            :", tp_out_path)
    print("TP-LA LOS [mm]       :", los_mm_out_path)
    print("TP-LA products       :", prod_dir)
    print("manifest             :", prod_dir / "point_products_tp_la_manifest.json")
    print("=" * 100)
    print("FINAL RESULT: PASS_TP_LA_PRODUCTION")
    print("=" * 100)


if __name__ == "__main__":
    main()
