from __future__ import annotations

from dataclasses import dataclass
import numpy as np


POINT_PS = np.uint8(1)
POINT_DS = np.uint8(2)


@dataclass(frozen=True, slots=True)
class PointPhaseStack:
    row: np.ndarray
    col: np.ndarray
    point_type: np.ndarray       # 1=PS, 2=DS
    phase_rad: np.ndarray        # (npoint, ndate), referenced
    quality: np.ndarray          # PS: 1-ADI normalized proxy; DS: temporal coherence
    temporal_coherence: np.ndarray
    amplitude_dispersion: np.ndarray
    shp_count: np.ndarray
    dates: np.ndarray
    reference_idx: int


def _reference_ps_phase(stack: np.ndarray, rr: np.ndarray, cc: np.ndarray, reference_idx: int) -> np.ndarray:
    x = stack[:, rr, cc].T.astype(np.complex64, copy=False)  # point,date
    ref = x[:, reference_idx:reference_idx+1]
    ph = np.angle(x * np.conj(ref)).astype(np.float32)
    ph[:, reference_idx] = 0.0
    return ph


def fuse_ps_ds(
    stack: np.ndarray,
    dates: np.ndarray,
    ps_mask: np.ndarray,
    ps_dispersion: np.ndarray,
    ds_centers: np.ndarray,
    ds_phase_rad: np.ndarray,
    ds_temporal_coherence: np.ndarray,
    ds_shp_count: np.ndarray,
    *,
    reference_idx: int = 0,
    ds_temp_coh_threshold: float = 0.6,
    ds_min_shp: int = 20,
    ps_priority: bool = True,
) -> PointPhaseStack:
    """Fuse PS and DS into one radar-coordinate PointPhaseStack.

    PS phases are taken directly from the center SLC pixel and referenced to one
    acquisition. DS phases come from phase linking. On overlap, PS has priority
    by default so no pixel is duplicated.
    """
    ps_rc = np.argwhere(ps_mask).astype(np.int32)
    ds_keep = (
        np.isfinite(ds_temporal_coherence)
        & (ds_temporal_coherence >= float(ds_temp_coh_threshold))
        & (ds_shp_count >= int(ds_min_shp))
    )
    ds_rc = np.asarray(ds_centers[ds_keep], dtype=np.int32)
    ds_phase = np.asarray(ds_phase_rad[ds_keep], dtype=np.float32)
    ds_tc = np.asarray(ds_temporal_coherence[ds_keep], dtype=np.float32)
    ds_k = np.asarray(ds_shp_count[ds_keep], dtype=np.uint16)

    if ps_priority and len(ps_rc) and len(ds_rc):
        ps_keys = {(int(r), int(c)) for r, c in ps_rc.tolist()}
        keep = np.asarray([(int(r), int(c)) not in ps_keys for r, c in ds_rc.tolist()], dtype=bool)
        ds_rc, ds_phase, ds_tc, ds_k = ds_rc[keep], ds_phase[keep], ds_tc[keep], ds_k[keep]

    if len(ps_rc):
        ps_phase = _reference_ps_phase(stack, ps_rc[:, 0], ps_rc[:, 1], int(reference_idx))
        ps_adi = ps_dispersion[ps_rc[:, 0], ps_rc[:, 1]].astype(np.float32)
        # Simple bounded candidate-quality proxy; final PS quality belongs in later PS estimation.
        ps_q = np.clip(1.0 - ps_adi, 0.0, 1.0).astype(np.float32)
    else:
        ps_phase = np.empty((0, stack.shape[0]), dtype=np.float32)
        ps_adi = np.empty(0, dtype=np.float32)
        ps_q = np.empty(0, dtype=np.float32)

    row = np.concatenate([ps_rc[:, 0] if len(ps_rc) else np.empty(0, np.int32), ds_rc[:, 0] if len(ds_rc) else np.empty(0, np.int32)]).astype(np.int32)
    col = np.concatenate([ps_rc[:, 1] if len(ps_rc) else np.empty(0, np.int32), ds_rc[:, 1] if len(ds_rc) else np.empty(0, np.int32)]).astype(np.int32)
    point_type = np.concatenate([np.full(len(ps_rc), POINT_PS, np.uint8), np.full(len(ds_rc), POINT_DS, np.uint8)])
    phase = np.concatenate([ps_phase, ds_phase], axis=0)
    tc = np.concatenate([np.full(len(ps_rc), np.nan, np.float32), ds_tc])
    adi = np.concatenate([ps_adi, np.full(len(ds_rc), np.nan, np.float32)])
    shp = np.concatenate([np.zeros(len(ps_rc), np.uint16), ds_k])
    quality = np.concatenate([ps_q, ds_tc])

    return PointPhaseStack(
        row=row, col=col, point_type=point_type, phase_rad=phase,
        quality=quality, temporal_coherence=tc, amplitude_dispersion=adi,
        shp_count=shp, dates=np.asarray(dates), reference_idx=int(reference_idx),
    )
