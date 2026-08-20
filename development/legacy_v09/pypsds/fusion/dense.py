from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Any
import math
import numpy as np

from pypsds.ds.covariance import _cov_from_samples, _normalize_covariance
from pypsds.ds.phase_linking import link_one_diagnostic, median_pair_coherence, ESTIMATOR_EVD
from pypsds.ds.shp import select_shp, unpack_pixel_support
from pypsds.progress import ProgressTracker, log
from .point_stack import POINT_DS, POINT_PS


@dataclass(frozen=True, slots=True)
class DenseFusionResult:
    row: np.ndarray
    col: np.ndarray
    point_type: np.ndarray
    phase_rad: np.ndarray
    quality: np.ndarray
    temporal_coherence: np.ndarray
    pair_coherence: np.ndarray
    amplitude_dispersion: np.ndarray
    shp_count: np.ndarray
    eigenvalue: np.ndarray
    estimator_code: np.ndarray
    pl_status_code: np.ndarray
    dates: np.ndarray
    reference_idx: int

    ps_mask: np.ndarray
    shp_count_map: np.ndarray
    ds_candidate_mask: np.ndarray
    ds_evaluated_mask: np.ndarray
    ds_tc_map: np.ndarray
    ds_pair_coherence_map: np.ndarray
    ds_eigenvalue_map: np.ndarray
    ds_estimator_map: np.ndarray
    ds_pl_status_map: np.ndarray
    geometry_valid_mask: np.ndarray
    ds_min_shp: int
    search_support_size: int


def _reference_center_phase(x: np.ndarray, reference_idx: int) -> np.ndarray:
    ref = x[reference_idx]
    ph = np.angle(x * np.conj(ref)).astype(np.float32)
    ph[reference_idx] = 0.0
    return ph


def _tile_bounds(rows: int, cols: int, tile_rows: int, tile_cols: int):
    for r0 in range(0, rows, tile_rows):
        r1 = min(rows, r0 + tile_rows)
        for c0 in range(0, cols, tile_cols):
            c1 = min(cols, c0 + tile_cols)
            yield r0, r1, c0, c1


def _fmt_sec(x: float) -> str:
    return f"{x:.1f}s"


def run_dense_psds(
    stack,
    *,
    roi: tuple[int, int, int, int],
    dates: np.ndarray,
    ps_mask: np.ndarray,
    ps_dispersion: np.ndarray,
    shp_half_window: tuple[int, int],
    shp_offsets: np.ndarray | None = None,
    shp_method: str = "glrt",
    shp_alpha: float = 0.001,
    ds_min_shp_absolute: int = 30,
    ds_min_shp_per_date: float = 1.25,
    ds_min_shp_for_estimation: int | None = None,  # v0.4 compatibility alias
    exclude_ps_from_ds_support: bool = True,
    phase_link_method: str = "emi",
    reference_idx: int = 0,
    beta: float = 0.05,
    weighted_temp_coh: bool = False,
    tile_shape: tuple[int, int] = (128, 128),
    phase_correction_provider: Any | None = None,
    progress: Callable[[str], None] | None = print,
    candidate_progress_updates: int = 4,
) -> DenseFusionResult:
    """v0.5 candidate-first CPU tiled PS/DS processing.

    Order per tile:
      raw RSLC -> amplitude SHP -> DS candidate mask -> optional GAMMA geometry
      correction -> candidate-only covariance -> robust EMI/EVD -> QA.

    Geometry correction is therefore skipped for tiles containing neither a PS
    center nor a DS candidate. Full covariance matrices are never persisted.
    """
    sink = progress or (lambda _: None)
    row0, col0, rows, cols = map(int, roi)
    if ps_mask.shape != (rows, cols) or ps_dispersion.shape != (rows, cols):
        raise ValueError("PS arrays must match ROI shape")
    ndate = len(dates)
    if reference_idx < 0:
        reference_idx += ndate
    if not (0 <= reference_idx < ndate):
        raise ValueError("reference_idx outside date stack")

    hy, hx = map(int, shp_half_window)
    tile_rows, tile_cols = map(int, tile_shape)
    min_shp = (
        int(ds_min_shp_for_estimation)
        if ds_min_shp_for_estimation is not None
        else max(int(ds_min_shp_absolute), int(math.ceil(float(ds_min_shp_per_date) * ndate)))
    )
    offsets = None if shp_offsets is None else np.asarray(shp_offsets, dtype=np.int16).reshape(-1, 2)
    support_size = ((2 * hy + 1) * (2 * hx + 1)) if offsets is None else len(offsets)
    if offsets is not None:
        support_size -= int(np.any(np.all(offsets == 0, axis=1)))
    else:
        support_size -= 1
    if min_shp > support_size:
        raise ValueError(
            f"DS min SHP={min_shp} exceeds maximum non-center search support={support_size}. "
            "Increase shp.search_radius_m or lower ds.min_shp settings."
        )

    shp_map = np.zeros((rows, cols), np.uint16)
    candidate_mask = np.zeros((rows, cols), bool)
    eval_mask = np.zeros((rows, cols), bool)
    tc_map = np.full((rows, cols), np.nan, np.float32)
    pair_map = np.full((rows, cols), np.nan, np.float32)
    eig_map = np.full((rows, cols), np.nan, np.float32)
    est_map = np.full((rows, cols), -1, np.int8)
    status_map = np.full((rows, cols), -1, np.int8)
    geometry_valid = np.ones((rows, cols), bool) if phase_correction_provider is None else np.zeros((rows, cols), bool)

    row_chunks=[]; col_chunks=[]; type_chunks=[]; phase_chunks=[]; quality_chunks=[]
    tc_chunks=[]; pair_chunks=[]; adi_chunks=[]; k_chunks=[]; eig_chunks=[]; est_chunks=[]; status_chunks=[]

    bounds = list(_tile_bounds(rows, cols, tile_rows, tile_cols))
    tracker = ProgressTracker("Step08 v0.5 candidate tiles", len(bounds), sink=sink)
    total_candidates = 0; total_linked = 0; total_ps = 0; total_fallback = 0

    for tile_i, (cr0, cr1, cc0, cc1) in enumerate(bounds, 1):
        tile_t0 = perf_counter()
        global_cr0, global_cr1 = row0 + cr0, row0 + cr1
        global_cc0, global_cc1 = col0 + cc0, col0 + cc1
        in_r0 = max(0, global_cr0 - hy); in_r1 = min(stack.shape[1], global_cr1 + hy)
        in_c0 = max(0, global_cc0 - hx); in_c1 = min(stack.shape[2], global_cc1 + hx)
        label = f"t{tile_i:05d}_r{global_cr0}_c{global_cc0}"

        t0 = perf_counter()
        block = stack.read_window(row0=in_r0, col0=in_c0, rows=in_r1-in_r0, cols=in_c1-in_c0)
        read_sec = perf_counter() - t0
        if block.shape[0] != ndate:
            raise ValueError(f"Expected {ndate} dates, reader returned {block.shape[0]}")

        core_br0 = global_cr0 - in_r0; core_br1 = global_cr1 - in_r0
        core_bc0 = global_cc0 - in_c0; core_bc1 = global_cc1 - in_c0

        # Map the ROI PS mask onto this tile+halo so known bright PS can be
        # excluded from DS covariance support.
        local_ps = np.zeros(block.shape[1:], bool)
        gr0=max(in_r0,row0); gr1=min(in_r1,row0+rows)
        gc0=max(in_c0,col0); gc1=min(in_c1,col0+cols)
        if gr1>gr0 and gc1>gc0:
            local_ps[gr0-in_r0:gr1-in_r0, gc0-in_c0:gc1-in_c0] = ps_mask[
                gr0-row0:gr1-row0, gc0-col0:gc1-col0
            ]

        t0 = perf_counter()
        shp = select_shp(
            block,
            half_window=(hy, hx),
            method=shp_method,
            alpha=shp_alpha,
            offsets=offsets,
            min_samples_absolute=min_shp,
            min_samples_per_date=0.0,
            progress=None,
        )
        shp_sec = perf_counter() - t0
        core_counts = shp.counts[core_br0:core_br1, core_bc0:core_bc1]
        shp_map[cr0:cr1, cc0:cc1] = core_counts

        center_valid_raw = np.all(
            np.isfinite(block[:, core_br0:core_br1, core_bc0:core_bc1].real)
            & np.isfinite(block[:, core_br0:core_br1, core_bc0:core_bc1].imag),
            axis=0,
        )
        cand_core = (core_counts >= min_shp) & (~ps_mask[cr0:cr1, cc0:cc1]) & center_valid_raw
        candidate_mask[cr0:cr1, cc0:cc1] = cand_core
        cand_coords = np.argwhere(cand_core)
        n_cand = len(cand_coords); n_ps_tile = int(ps_mask[cr0:cr1, cc0:cc1].sum())
        total_candidates += n_cand

        # Only now pay the GAMMA geometry-correction cost.
        corr_sec = 0.0; corr_detail = "corr=off"
        valid_height_block = np.ones(block.shape[1:], bool)
        need_corrected_phase = (n_cand > 0 or n_ps_tile > 0)
        if phase_correction_provider is not None and need_corrected_phase:
            t0 = perf_counter()
            block, valid_height_block, cs = phase_correction_provider.correct_block(
                block, global_row0=in_r0, global_col0=in_c0, tile_label=label,
            )
            corr_sec = perf_counter() - t0
            corr_detail = f"corr={corr_sec:.1f}s hgt={cs.n_valid_height}/{cs.n_points} sim=[{cs.phase_min:.1f},{cs.phase_max:.1f}]"
        elif phase_correction_provider is not None:
            corr_detail = "corr=SKIP(no PS/DS candidate)"

        geometry_valid[cr0:cr1, cc0:cc1] = valid_height_block[
            core_br0:core_br1, core_bc0:core_bc1
        ] if need_corrected_phase else True

        # Collect PS centers first; PS always has priority over DS.
        tile_rows_out=[]; tile_cols_out=[]; tile_types=[]; tile_ph=[]; tile_quality=[]
        tile_tc=[]; tile_pair=[]; tile_adi=[]; tile_k=[]; tile_eig=[]; tile_est=[]; tile_status=[]
        for lr, lc in np.argwhere(ps_mask[cr0:cr1, cc0:cc1]):
            rr_roi=cr0+int(lr); cc_roi=cc0+int(lc)
            br=core_br0+int(lr); bc=core_bc0+int(lc)
            if not valid_height_block[br,bc]:
                continue
            x=block[:,br,bc]
            if not np.all(np.isfinite(x.real)&np.isfinite(x.imag)):
                continue
            adi=float(ps_dispersion[rr_roi,cc_roi])
            tile_rows_out.append(rr_roi); tile_cols_out.append(cc_roi); tile_types.append(int(POINT_PS))
            tile_ph.append(_reference_center_phase(x,reference_idx)); tile_quality.append(float(np.clip(1-adi,0,1)))
            tile_tc.append(np.nan); tile_pair.append(np.nan); tile_adi.append(adi); tile_k.append(0)
            tile_eig.append(np.nan); tile_est.append(-1); tile_status.append(-1)
        total_ps += len(tile_rows_out)

        offsets_i = shp.offsets.astype(np.int32)
        pl_t0 = perf_counter(); linked=0; fallback=0
        report_every = max(1, n_cand // max(1, int(candidate_progress_updates))) if n_cand else 1

        for ii, (lr0, lc0) in enumerate(cand_coords.tolist(), 1):
            rr_roi=cr0+int(lr0); cc_roi=cc0+int(lc0)
            br=core_br0+int(lr0); bc=core_bc0+int(lc0)
            if not valid_height_block[br,bc]:
                continue

            support = unpack_pixel_support(shp.packed_bits[br, bc], len(offsets_i))
            selected = offsets_i[support]
            if len(selected) < min_shp:
                continue
            r_idx=br+selected[:,0]; c_idx=bc+selected[:,1]
            inside=(r_idx>=0)&(r_idx<block.shape[1])&(c_idx>=0)&(c_idx<block.shape[2])
            r_idx=r_idx[inside]; c_idx=c_idx[inside]
            if exclude_ps_from_ds_support and len(r_idx):
                keep=~local_ps[r_idx,c_idx]; r_idx=r_idx[keep]; c_idx=c_idx[keep]
            if len(r_idx)<min_shp:
                continue
            samples=block[:,r_idx,c_idx]
            valid_samples=np.all(np.isfinite(samples.real)&np.isfinite(samples.imag),axis=0)
            samples=samples[:,valid_samples]
            if samples.shape[1]<min_shp:
                continue

            try:
                C=_normalize_covariance(_cov_from_samples(samples))
                v,eig,tc,est,status=link_one_diagnostic(
                    C, method=phase_link_method, reference_idx=reference_idx,
                    beta=beta, weighted_temp_coh=weighted_temp_coh,
                )
            except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                continue

            pair=median_pair_coherence(C); k=int(samples.shape[1])
            ph=np.angle(v).astype(np.float32); ph[reference_idx]=0.0
            eval_mask[rr_roi,cc_roi]=True; tc_map[rr_roi,cc_roi]=tc; pair_map[rr_roi,cc_roi]=pair
            eig_map[rr_roi,cc_roi]=eig; est_map[rr_roi,cc_roi]=est; status_map[rr_roi,cc_roi]=status
            tile_rows_out.append(rr_roi); tile_cols_out.append(cc_roi); tile_types.append(int(POINT_DS))
            tile_ph.append(ph); tile_quality.append(float(tc)); tile_tc.append(float(tc)); tile_pair.append(float(pair))
            tile_adi.append(np.nan); tile_k.append(k); tile_eig.append(float(eig)); tile_est.append(int(est)); tile_status.append(int(status))
            linked+=1
            if int(est)==int(ESTIMATOR_EVD) and phase_link_method.lower() not in {"evd"}:
                fallback+=1
            if n_cand and (ii % report_every == 0 or ii == n_cand):
                log(
                    f"Step08 tile {tile_i}/{len(bounds)} candidate PL: {ii}/{n_cand} "
                    f"({100*ii/n_cand:.1f}%) linked={linked} fallback={fallback}", sink=sink
                )

        pl_sec=perf_counter()-pl_t0; total_linked+=linked; total_fallback+=fallback

        if tile_rows_out:
            row_chunks.append(np.asarray(tile_rows_out,np.int32)); col_chunks.append(np.asarray(tile_cols_out,np.int32))
            type_chunks.append(np.asarray(tile_types,np.uint8)); phase_chunks.append(np.asarray(tile_ph,np.float32))
            quality_chunks.append(np.asarray(tile_quality,np.float32)); tc_chunks.append(np.asarray(tile_tc,np.float32))
            pair_chunks.append(np.asarray(tile_pair,np.float32)); adi_chunks.append(np.asarray(tile_adi,np.float32))
            k_chunks.append(np.asarray(tile_k,np.uint16)); eig_chunks.append(np.asarray(tile_eig,np.float32))
            est_chunks.append(np.asarray(tile_est,np.int8)); status_chunks.append(np.asarray(tile_status,np.int8))

        total_sec=perf_counter()-tile_t0
        shp_vals=core_counts[np.isfinite(core_counts)]
        tracker.update(
            tile_i,
            detail=(
                f"core={cr0}:{cr1},{cc0}:{cc1} read={_fmt_sec(read_sec)} shp={_fmt_sec(shp_sec)} "
                f"SHPmed={float(np.median(shp_vals)) if shp_vals.size else 0:.1f} cand={n_cand} PS={n_ps_tile} "
                f"{corr_detail} PL={_fmt_sec(pl_sec)} linked={linked} fb={fallback} total={_fmt_sec(total_sec)}"
            ),
        )

    def cat(chunks,dtype,shape_tail=()):
        if chunks:
            return np.concatenate(chunks,axis=0).astype(dtype,copy=False)
        return np.empty((0,*shape_tail),dtype=dtype)

    phase = cat(phase_chunks,np.float32,(ndate,))
    log(
        f"Step08 v0.5 totals: PS={total_ps} DS-candidates={total_candidates} "
        f"DS-linked={total_linked} EMI->EVD fallback={total_fallback}", sink=sink
    )
    return DenseFusionResult(
        row=cat(row_chunks,np.int32), col=cat(col_chunks,np.int32), point_type=cat(type_chunks,np.uint8),
        phase_rad=phase, quality=cat(quality_chunks,np.float32), temporal_coherence=cat(tc_chunks,np.float32),
        pair_coherence=cat(pair_chunks,np.float32), amplitude_dispersion=cat(adi_chunks,np.float32),
        shp_count=cat(k_chunks,np.uint16), eigenvalue=cat(eig_chunks,np.float32), estimator_code=cat(est_chunks,np.int8),
        pl_status_code=cat(status_chunks,np.int8), dates=np.asarray(dates), reference_idx=int(reference_idx),
        ps_mask=ps_mask.copy(), shp_count_map=shp_map, ds_candidate_mask=candidate_mask,
        ds_evaluated_mask=eval_mask, ds_tc_map=tc_map, ds_pair_coherence_map=pair_map,
        ds_eigenvalue_map=eig_map, ds_estimator_map=est_map, ds_pl_status_map=status_map,
        geometry_valid_mask=geometry_valid, ds_min_shp=min_shp, search_support_size=support_size,
    )
