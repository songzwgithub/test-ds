from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypsds.prototype import open_from_config
from pypsds.ds.shp_dolphin import glrt_statistic, glrt_threshold
from pypsds.ds.covariance_pc_v09 import compressed_coherence_pc_v09
from pypsds.ds.phase_link_v09 import (
    ESTIMATOR_EVD,
    ESTIMATOR_EMI,
    image_pairs,
    robust_emi_threaded_v09,
    temporal_coherence_v09,
    median_pair_coherence_v09,
)


def make_support_batch(
    scale2,
    valid,
    ps,
    rows,
    cols,
    *,
    half_row,
    half_col,
    alpha,
    ndate,
):
    B = rows.size
    wh = 2 * half_row + 1
    ww = 2 * half_col + 1
    out = np.zeros((B, wh, ww), dtype=bool)
    center_scale = scale2[rows, cols]
    thr = glrt_threshold(alpha)
    H, W = valid.shape

    for ky, dy in enumerate(range(-half_row, half_row + 1)):
        for kx, dx in enumerate(range(-half_col, half_col + 1)):
            if dy == 0 and dx == 0:
                continue
            rr = rows + dy
            cc = cols + dx
            inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            if not np.any(inside):
                continue
            ids = np.flatnonzero(inside)
            r2 = rr[ids]
            c2 = cc[ids]
            ngood = valid[r2, c2] & ~ps[r2, c2]
            if not np.any(ngood):
                continue
            ids2 = ids[ngood]
            r3 = rr[ids2]
            c3 = cc[ids2]
            stat = glrt_statistic(
                center_scale[ids2],
                scale2[r3, c3],
                nslc=ndate,
            )
            out[ids2, ky, kx] = np.isfinite(stat) & (stat < thr)
    return out


def open_or_create_npy(path: Path, *, shape, dtype, fill, resume: bool):
    if resume and path.is_file():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != tuple(shape) or arr.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"resume cache mismatch: {path}: shape={arr.shape}, dtype={arr.dtype}, "
                f"expected={shape}/{np.dtype(dtype)}"
            )
        return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[...] = fill
    arr.flush()
    return arr


def save_map(arr, path, title, label, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(arr, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--center-mode", choices=["moraine", "all"], default="moraine")
    ap.add_argument("--moraine-npz", default=None)
    ap.add_argument("--yxt-cache", default=None)
    ap.add_argument("--geom-valid", default=None)
    ap.add_argument("--half-row", type=int, default=5)
    ap.add_argument("--half-col", type=int, default=11)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--min-shp", type=int, default=48)
    ap.add_argument("--adi-max", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--gamma-jitter", type=float, default=1e-6)
    ap.add_argument("--emi-mu", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=16000)
    ap.add_argument("--pl-workers", type=int, default=16)
    ap.add_argument("--pl-chunk-size", type=int, default=512)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-centers", type=int, default=0)
    args = ap.parse_args()

    cfg, config_path, paths, stack, (row0, col0, H, W) = open_from_config(args.config)
    ndate = len(stack.dates)
    pairs = image_pairs(ndate)
    pi, pj = pairs[:, 0], pairs[:, 1]

    outdir = Path(paths.output_dir) / "v09"
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("pyPSDS-GAMMA v0.9 - CPU production (full standalone)")
    print("=" * 80)
    print(f"config          : {config_path}")
    print(f"scene           : {H} x {W}")
    print(f"dates           : {ndate}")
    print(f"center mode     : {args.center_mode}")
    print(f"GLRT            : {2*args.half_row+1} x {2*args.half_col+1}, alpha={args.alpha}")
    print(f"Kmin            : {args.min_shp}")
    print(f"batch size      : {args.batch_size}")
    print(f"PL workers      : {args.pl_workers}")
    print(f"PL chunk        : {args.pl_chunk_size}")
    print(f"Numba threads   : {os.environ.get('NUMBA_NUM_THREADS', 'runtime-default')}")
    print(f"resume          : {args.resume}")

    # ---------------------------------------------------------
    # Raw amplitude / validity / PS
    # ---------------------------------------------------------
    t_all = time.time()
    raw = stack.read_window(row0=row0, col0=col0, rows=H, cols=W).astype(np.complex64, copy=False)
    zero = (raw.real == 0) & (raw.imag == 0)
    valid = ~(
        np.any(zero, axis=0)
        | np.any(~np.isfinite(raw.real), axis=0)
        | np.any(~np.isfinite(raw.imag), axis=0)
    )
    amp = np.abs(raw).astype(np.float32)
    mean_amp = np.full((H, W), np.nan, np.float32)
    var_amp = np.full((H, W), np.nan, np.float32)
    std_amp = np.full((H, W), np.nan, np.float32)
    av = amp[:, valid].astype(np.float64)
    mean_amp[valid] = np.mean(av, axis=0)
    var_amp[valid] = np.var(av, axis=0, ddof=0)
    std_amp[valid] = np.std(av, axis=0, ddof=0)
    adi = np.full((H, W), np.nan, np.float32)
    adi[valid] = std_amp[valid] / mean_amp[valid]
    ps = valid & np.isfinite(adi) & (adi <= args.adi_max)
    scale2 = ((var_amp + mean_amp * mean_amp) / 2.0).astype(np.float64)
    del amp, av, raw
    print(f"raw valid       : {valid.sum()} ({100*valid.mean():.3f}%)")
    print(f"PS              : {ps.sum()} ({100*ps.mean():.3f}%)")

    # ---------------------------------------------------------
    # Fast C-contiguous corrected stack
    # ---------------------------------------------------------
    if args.yxt_cache:
        yxt_path = Path(args.yxt_cache)
    else:
        yxt_path = outdir / "cache" / "phase_corrected_yxt.npy"
    if not yxt_path.is_file():
        raise FileNotFoundError(
            f"Missing v0.9 YXT cache: {yxt_path}\n"
            "Run scripts/07d_prepare_yxt_cache_v09.py first."
        )
    rslc_yxt = np.load(yxt_path, mmap_mode="r")
    if rslc_yxt.shape != (H, W, ndate):
        raise RuntimeError(f"YXT cache shape={rslc_yxt.shape}, expected={(H,W,ndate)}")

    if args.geom_valid:
        geom_path = Path(args.geom_valid)
    else:
        geom_path = Path(paths.output_dir) / "v09" / "cache" / "phase_geometry_valid.npy"
    geom_valid = np.load(geom_path, mmap_mode="r").astype(bool, copy=False)
    valid &= geom_valid

    # ---------------------------------------------------------
    # Center prior
    # ---------------------------------------------------------
    if args.center_mode == "moraine":
        prior_path = Path(args.moraine_npz) if args.moraine_npz else Path(paths.output_dir) / "v09" / "moraine_center_prior.npz"
        z = np.load(prior_path)
        center_prior = z["candidate_mask"].astype(bool)
    else:
        center_prior = valid.copy()
    center_prior &= valid
    center_prior &= ~ps
    rr, cc = np.where(center_prior)
    rr = rr.astype(np.int32)
    cc = cc.astype(np.int32)
    if args.max_centers > 0 and len(rr) > args.max_centers:
        ids = np.linspace(0, len(rr) - 1, args.max_centers, dtype=np.int64)
        rr, cc = rr[ids], cc[ids]
    total = len(rr)
    print(f"DS centers      : {total}")

    # Save exact center order for reliable resume.
    rows_path = outdir / "center_rows.npy"
    cols_path = outdir / "center_cols.npy"
    if args.resume and rows_path.is_file() and cols_path.is_file():
        old_r = np.load(rows_path)
        old_c = np.load(cols_path)
        if not (np.array_equal(old_r, rr) and np.array_equal(old_c, cc)):
            raise RuntimeError("Resume center list does not match current configuration")
    else:
        np.save(rows_path, rr)
        np.save(cols_path, cc)

    # ---------------------------------------------------------
    # Output mmap arrays
    # ---------------------------------------------------------
    shp_count = open_or_create_npy(outdir / "shp_count.npy", shape=(H, W), dtype=np.int16, fill=-1, resume=args.resume)
    tc_map = open_or_create_npy(outdir / "temporal_coherence.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    pair_map = open_or_create_npy(outdir / "median_pair_coherence.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    estimator = open_or_create_npy(outdir / "estimator_code.npy", shape=(H, W), dtype=np.uint8, fill=255, resume=args.resume)
    emi_eig_map = open_or_create_npy(outdir / "emi_eigenvalue.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    evd_eig_map = open_or_create_npy(outdir / "evd_eigenvalue.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    gamma_min_map = open_or_create_npy(outdir / "gamma_min_eigenvalue.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    pl_valid = open_or_create_npy(outdir / "pl_valid.npy", shape=(H, W), dtype=np.bool_, fill=False, resume=args.resume)
    processed = open_or_create_npy(outdir / "processed_centers.npy", shape=(total,), dtype=np.bool_, fill=False, resume=args.resume)
    linked_phase = open_or_create_npy(
        outdir / "linked_phase.npy",
        shape=(ndate, H, W),
        dtype=np.complex64,
        fill=np.nan + 1j * np.nan,
        resume=args.resume,
    )

    # Persist static masks for downstream use.
    np.save(outdir / "ps_mask.npy", ps)
    np.save(outdir / "center_prior.npy", center_prior)

    # Warm-up covariance JIT with tiny legal shapes, excluded from timings.
    if total:
        warm_support = make_support_batch(
            scale2, valid, ps, rr[:1], cc[:1],
            half_row=args.half_row, half_col=args.half_col,
            alpha=args.alpha, ndate=ndate,
        )
        _ = compressed_coherence_pc_v09(rslc_yxt, rr[:1], cc[:1], warm_support, pi, pj)

    # ---------------------------------------------------------
    # Production batches
    # ---------------------------------------------------------
    linked_n = int(np.count_nonzero(pl_valid))
    t0 = time.time()
    support_s = covariance_s = pl_s = write_s = 0.0

    for b0 in range(0, total, args.batch_size):
        b1 = min(total, b0 + args.batch_size)
        if np.all(processed[b0:b1]):
            print(f"centers {b1:8d}/{total:8d}: RESUME skip")
            continue

        br = rr[b0:b1]
        bc = cc[b0:b1]

        ts = time.time()
        support = make_support_batch(
            scale2, valid, ps, br, bc,
            half_row=args.half_row,
            half_col=args.half_col,
            alpha=args.alpha,
            ndate=ndate,
        )
        K = np.sum(support, axis=(1, 2)).astype(np.int16)
        shp_count[br, bc] = K
        support_s += time.time() - ts

        good = K >= args.min_shp
        nlinked_batch = 0
        if np.any(good):
            gr = br[good]
            gc = bc[good]
            gs = support[good]

            ts = time.time()
            coh = compressed_coherence_pc_v09(rslc_yxt, gr, gc, gs, pi, pj)
            covariance_s += time.time() - ts

            ts = time.time()
            ph, est, emi_eig, evd_eig, gamma_min = robust_emi_threaded_v09(
                coh,
                n_images=ndate,
                pairs=pairs,
                beta=args.beta,
                gamma_jitter=args.gamma_jitter,
                emi_mu=args.emi_mu,
                reference_idx=0,
                workers=args.pl_workers,
                chunk_size=args.pl_chunk_size,
            )
            tc = temporal_coherence_v09(coh, ph, pairs)
            pc = median_pair_coherence_v09(coh)
            pl_s += time.time() - ts

            ok = (est != 255) & np.isfinite(tc)
            gr2 = gr[ok]
            gc2 = gc[ok]
            if gr2.size:
                ts = time.time()
                tc_map[gr2, gc2] = tc[ok]
                pair_map[gr2, gc2] = pc[ok]
                estimator[gr2, gc2] = est[ok]
                emi_eig_map[gr2, gc2] = emi_eig[ok]
                evd_eig_map[gr2, gc2] = evd_eig[ok]
                gamma_min_map[gr2, gc2] = gamma_min[ok]
                pl_valid[gr2, gc2] = True
                linked_phase[:, gr2, gc2] = ph[ok].T
                write_s += time.time() - ts
                nlinked_batch = int(gr2.size)
                linked_n += nlinked_batch

        processed[b0:b1] = True
        for arr in (
            shp_count, tc_map, pair_map, estimator, emi_eig_map,
            evd_eig_map, gamma_min_map, pl_valid, processed, linked_phase,
        ):
            arr.flush()

        elapsed = time.time() - t0
        done = b1
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"centers {done:8d}/{total:8d} ({100*done/total:6.2f}%) "
            f"linked={linked_n:8d} batch_link={nlinked_batch:6d} "
            f"wall={elapsed/60:6.2f}m ETA={eta/60:6.2f}m | "
            f"support={support_s:6.1f}s cov={covariance_s:6.1f}s "
            f"PL={pl_s:6.1f}s write={write_s:6.1f}s"
        )

    # ---------------------------------------------------------
    # Fill PS phase, but DO NOT set DS TC=1 for PS QA.
    # ---------------------------------------------------------
    pr, pcx = np.where(ps & geom_valid)
    if pr.size:
        pph = rslc_yxt[pr, pcx, :]
        pph = np.exp(1j * np.angle(pph)).astype(np.complex64)
        pph *= np.exp(-1j * np.angle(pph[:, 0]))[:, None]
        linked_phase[:, pr, pcx] = pph.T
        linked_phase.flush()

    # ---------------------------------------------------------
    # QA only on actually calculated DS pixels.
    # ---------------------------------------------------------
    ds_mask = np.asarray(pl_valid) & ~ps
    tc_view = np.where(ds_mask, np.asarray(tc_map), np.nan)
    pc_view = np.where(ds_mask, np.asarray(pair_map), np.nan)
    shp_view = np.where(np.asarray(shp_count) >= 0, np.asarray(shp_count).astype(np.float32), np.nan)
    est_view = np.where(ds_mask, np.asarray(estimator).astype(np.float32), np.nan)

    save_map(shp_view, figdir / "v09_glrt_shp_count.png", "v0.9 - GLRT covariance support count", "SHP count")
    save_map(tc_view, figdir / "v09_temporal_coherence.png", "v0.9 - DS temporal coherence", "TC", 0, 1)
    save_map(pc_view, figdir / "v09_median_pair_coherence.png", "v0.9 - DS median pair coherence", "Median |Cij|", 0, 1)
    save_map(est_view, figdir / "v09_estimator.png", "v0.9 - Phase linking estimator (0=EVD, 1=EMI)", "Estimator", 0, 1)

    ds_tc = np.asarray(tc_map)[ds_mask]
    print("=" * 80)
    print("v0.9 complete")
    print("=" * 80)
    print(f"PS                 : {ps.sum()} ({100*ps.mean():.3f}%)")
    print(f"phase-linked DS    : {ds_mask.sum()} ({100*ds_mask.mean():.3f}%)")
    if ds_tc.size:
        print(f"DS TC median       : {np.nanmedian(ds_tc):.4f}")
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            m = ds_mask & (np.asarray(tc_map) >= t)
            print(f"TC >= {t:.1f}          : {m.sum()} ({100*m.mean():.3f}%)")
    print(f"EMI used           : {np.sum(ds_mask & (np.asarray(estimator)==ESTIMATOR_EMI))}")
    print(f"EVD fallback       : {np.sum(ds_mask & (np.asarray(estimator)==ESTIMATOR_EVD))}")
    print(f"total wall         : {(time.time()-t_all)/60:.2f} min")
    print(f"stage seconds      : support={support_s:.1f}, covariance={covariance_s:.1f}, PL={pl_s:.1f}, write={write_s:.1f}")
    print(f"outputs            : {outdir}")


if __name__ == "__main__":
    main()
