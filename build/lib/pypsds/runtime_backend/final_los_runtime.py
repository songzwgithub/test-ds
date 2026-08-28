import os as _pypsds_os
from pathlib import Path as _PyPSDSPath
PUBLIC_PROJECT = _PyPSDSPath(_pypsds_os.environ['PYPSDS_PUBLIC_PROJECT'])
PUBLIC_DATA_ROOT = _PyPSDSPath(_pypsds_os.environ['PYPSDS_PUBLIC_DATA_ROOT'])
PUBLIC_PROC = _PyPSDSPath(_pypsds_os.environ['PYPSDS_PUBLIC_PROC'])
PUBLIC_PRODUCTS = _PyPSDSPath(_pypsds_os.environ['PYPSDS_PUBLIC_PRODUCTS'])
PUBLIC_REF_FILE = _PyPSDSPath(_pypsds_os.environ['PYPSDS_PUBLIC_REF_FILE'])
PUBLIC_RSLC_PAR = _PyPSDSPath(_pypsds_os.environ['PYPSDS_PUBLIC_RSLC_PAR'])
PUBLIC_POINT_GEOMETRY = _PyPSDSPath(_pypsds_os.environ['PYPSDS_PUBLIC_POINT_GEOMETRY'])
PUBLIC_GEOM_COMPAT = PUBLIC_PROC / 'atmosphere_correction' / '_geometry_compat'
PUBLIC_ATM = PUBLIC_PROC / 'atmosphere_correction'
PUBLIC_ATM_CACHE = PUBLIC_ATM / 'mapping_cache'
PUBLIC_SCLA = PUBLIC_PROC / 'scla'
PUBLIC_SCN = PUBLIC_PROC / 'scn'
PUBLIC_SCN_SUPPORT = PUBLIC_SCN / 'support'
PUBLIC_NDATE = int(_pypsds_os.environ['PYPSDS_PUBLIC_NDATE'])
PUBLIC_NIFG = int(_pypsds_os.environ['PYPSDS_PUBLIC_NIFG'])
PUBLIC_NSLAVE = int(_pypsds_os.environ['PYPSDS_PUBLIC_NSLAVE'])
PUBLIC_NREF = int(_pypsds_os.environ['PYPSDS_PUBLIC_NREF'])
PUBLIC_REF_DATE = _pypsds_os.environ['PYPSDS_PUBLIC_REF_DATE']
PUBLIC_MASTER_DATE = _pypsds_os.environ['PYPSDS_PUBLIC_MASTER_DATE']
PUBLIC_GACOS = _PyPSDSPath(_pypsds_os.environ.get('PYPSDS_PUBLIC_GACOS', '.'))
PUBLIC_SCN_TIME_WIN = float(_pypsds_os.environ.get('PYPSDS_PUBLIC_SCN_TIME_WIN', '365.0'))
PUBLIC_SCN_WAVELENGTH = float(_pypsds_os.environ.get('PYPSDS_PUBLIC_SCN_WAVELENGTH', '100.0'))
PUBLIC_SCN_RADIUS = float(_pypsds_os.environ.get('PYPSDS_PUBLIC_SCN_RADIUS', '400.0'))
PUBLIC_SCN_CELL_SIZE = float(_pypsds_os.environ.get('PYPSDS_PUBLIC_SCN_CELL_SIZE', '200.0'))
from pathlib import Path
import json
import os
import re
import time
import numpy as np
ROOT = PUBLIC_DATA_ROOT
PSDS = PUBLIC_PROJECT
PROC = PUBLIC_PROC
PRE = PUBLIC_SCLA / 'acquisition_phase_pre_scn_rad.npy'
SCN = PUBLIC_SCN / 'ph_scn_slave_rad.npy'
SCN_MANIFEST = PUBLIC_SCN / 'scn_manifest.json'
GMAN = PUBLIC_ATM / 'atmosphere_correction_manifest.json'
REF_FILE = PUBLIC_REF_FILE
RSLC_PAR = PUBLIC_RSLC_PAR
OUTDIR = PUBLIC_PROC / 'final_los'
OUTDIR.mkdir(parents=True, exist_ok=True)
PHASE_OUT = OUTDIR / 'acquisition_phase_final_rad.npy'
LOS_M_OUT = OUTDIR / 'los_displacement_toward_satellite_m.npy'
LOS_MM_OUT = OUTDIR / 'los_displacement_toward_satellite_mm.npy'
MANIFEST = PUBLIC_PROC / 'final_los' / 'final_los_manifest.json'
PHASE_TMP = OUTDIR / '.acquisition_phase_final_rad.tmp.npy'
LOS_M_TMP = OUTDIR / '.los_displacement_toward_satellite_m.tmp.npy'
LOS_MM_TMP = OUTDIR / '.los_displacement_toward_satellite_mm.tmp.npy'
TEMPORAL_REFERENCE_DATE = PUBLIC_REF_DATE
GEOMETRIC_MASTER_DATE = PUBLIC_MASTER_DATE
REFERENCE_POINTS_EXPECTED = PUBLIC_NREF
C0 = 299792458.0
CHUNK = 131072
REF_MEDIAN_TOL_RAD = 1e-07
EPOCH0_TOL_RAD = 0.0
PHASE_STORAGE_TOL_RAD = 2e-06
LOS_STORAGE_TOL_M = 2e-08
NUM_RE = re.compile('[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[Ee][-+]?\\d+)?')

def read_par(path):
    result = {}
    for line in path.read_text(errors='ignore').splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        result[k.strip().lower()] = v.strip()
    return result

def par_scalar(pars, keys):
    for key in keys:
        x = pars.get(key.lower())
        if x is None:
            continue
        m = NUM_RE.search(x)
        if m:
            return float(m.group(0))
    raise KeyError(' / '.join(keys))
for p in (PRE, SCN, SCN_MANIFEST, GMAN, REF_FILE, RSLC_PAR):
    if not p.is_file():
        raise FileNotFoundError(p)
pre = np.load(PRE, mmap_mode='r')
scn = np.load(SCN, mmap_mode='r')
if pre.shape != scn.shape:
    raise RuntimeError(f'pre-SCN / SCN shape mismatch: {pre.shape} vs {scn.shape}')
npoint, nepoch = pre.shape
if nepoch != PUBLIC_NDATE:
    raise RuntimeError(f'acquisition count mismatch: {nepoch} != {PUBLIC_NDATE}')
gman = json.loads(GMAN.read_text())
dates = list(gman['acquisition_dates'])
if len(dates) != nepoch:
    raise RuntimeError('date count mismatch')
if dates[0] != TEMPORAL_REFERENCE_DATE:
    raise RuntimeError(f'temporal reference mismatch: first date={dates[0]}, expected={TEMPORAL_REFERENCE_DATE}')
tref0 = dates.index(TEMPORAL_REFERENCE_DATE)
master0 = dates.index(GEOMETRIC_MASTER_DATE)
scn_manifest = json.loads(SCN_MANIFEST.read_text())
if scn_manifest.get('status') != 'PASS_SCN':
    raise RuntimeError('SCN stage is not PASS')
if int(scn_manifest['scientific_contract']['geometric_master_index_0based']) != master0:
    raise RuntimeError('SCN geometric-master contract changed')
ref_idx = np.load(REF_FILE).astype(np.int64)
if ref_idx.size != REFERENCE_POINTS_EXPECTED:
    raise RuntimeError(f'reference set changed: {ref_idx.size}')
if ref_idx.min() < 0 or ref_idx.max() >= npoint:
    raise RuntimeError('invalid reference-point index')
pars = read_par(RSLC_PAR)
radar_frequency = par_scalar(pars, ('radar_frequency',))
wavelength = C0 / radar_frequency
los_factor_m_per_rad = wavelength / (4.0 * np.pi)
ref_pre = np.asarray(pre[ref_idx, :], dtype=np.float64)
ref_scn = np.asarray(scn[ref_idx, :], dtype=np.float64)
ref_raw = ref_pre - ref_scn
ref_time = ref_raw - ref_raw[:, tref0][:, None]
spatial_reference_median = np.median(ref_time, axis=0)
if spatial_reference_median[tref0] != 0.0:
    raise RuntimeError(f'reference median at temporal reference is not exactly zero: {spatial_reference_median[tref0]}')
for p in (PHASE_TMP, LOS_M_TMP, LOS_MM_TMP):
    if p.exists():
        p.unlink()
phase_out = np.lib.format.open_memmap(PHASE_TMP, mode='w+', dtype=np.float32, shape=(npoint, nepoch))
los_m_out = np.lib.format.open_memmap(LOS_M_TMP, mode='w+', dtype=np.float32, shape=(npoint, nepoch))
los_mm_out = np.lib.format.open_memmap(LOS_MM_TMP, mode='w+', dtype=np.float32, shape=(npoint, nepoch))
t0 = time.perf_counter()
raw_scn_correction_ss = 0.0
raw_scn_correction_n = 0
raw_scn_correction_max = 0.0
phase_abs_max = 0.0
los_abs_max_m = 0.0
sample_phase_storage_max = 0.0
sample_los_storage_max = 0.0
for start in range(0, npoint, CHUNK):
    stop = min(start + CHUNK, npoint)
    pre64 = np.asarray(pre[start:stop, :], dtype=np.float64)
    scn64 = np.asarray(scn[start:stop, :], dtype=np.float64)
    raw = pre64 - scn64
    raw_scn_correction_ss += float(np.sum(scn64 * scn64))
    raw_scn_correction_n += int(scn64.size)
    raw_scn_correction_max = max(raw_scn_correction_max, float(np.max(np.abs(scn64))))
    temporal = raw - raw[:, tref0][:, None]
    final64 = temporal - spatial_reference_median[None, :]
    final64[:, tref0] = 0.0
    los64 = final64 * los_factor_m_per_rad
    phase32 = final64.astype(np.float32)
    los_m32 = los64.astype(np.float32)
    los_mm32 = (los64 * 1000.0).astype(np.float32)
    phase_out[start:stop, :] = phase32
    los_m_out[start:stop, :] = los_m32
    los_mm_out[start:stop, :] = los_mm32
    pd = float(np.max(np.abs(phase32.astype(np.float64) - final64)))
    ld = float(np.max(np.abs(los_m32.astype(np.float64) - los64)))
    sample_phase_storage_max = max(sample_phase_storage_max, pd)
    sample_los_storage_max = max(sample_los_storage_max, ld)
    phase_abs_max = max(phase_abs_max, float(np.max(np.abs(final64))))
    los_abs_max_m = max(los_abs_max_m, float(np.max(np.abs(los64))))
    print(f'[FINAL LOS] {stop:,}/{npoint:,} ({100 * stop / npoint:.1f}%)', flush=True)
phase_out.flush()
los_m_out.flush()
los_mm_out.flush()
materialization_seconds = time.perf_counter() - t0
if sample_phase_storage_max > PHASE_STORAGE_TOL_RAD:
    raise RuntimeError(f'float32 phase storage precision failed: {sample_phase_storage_max}')
if sample_los_storage_max > LOS_STORAGE_TOL_M:
    raise RuntimeError(f'float32 LOS storage precision failed: {sample_los_storage_max}')
del phase_out
del los_m_out
del los_mm_out
os.replace(PHASE_TMP, PHASE_OUT)
os.replace(LOS_M_TMP, LOS_M_OUT)
os.replace(LOS_MM_TMP, LOS_MM_OUT)
phase_final = np.load(PHASE_OUT, mmap_mode='r')
los_m = np.load(LOS_M_OUT, mmap_mode='r')
los_mm = np.load(LOS_MM_OUT, mmap_mode='r')
if phase_final.shape != (npoint, nepoch):
    raise RuntimeError('final phase shape failed')
finite_phase = float(np.mean(np.isfinite(phase_final)))
finite_los = float(np.mean(np.isfinite(los_m)))
if finite_phase != 1.0 or finite_los != 1.0:
    raise RuntimeError('final output contains non-finite values')
epoch0_max = float(np.max(np.abs(phase_final[:, tref0])))
los_epoch0_max = float(np.max(np.abs(los_m[:, tref0])))
if epoch0_max != EPOCH0_TOL_RAD:
    raise RuntimeError(f'final temporal reference failed: {epoch0_max}')
if los_epoch0_max != 0.0:
    raise RuntimeError(f'LOS temporal reference failed: {los_epoch0_max}')
ref_phase_final = np.asarray(phase_final[ref_idx, :], dtype=np.float64)
ref_median_final = np.median(ref_phase_final, axis=0)
ref_median_max = float(np.max(np.abs(ref_median_final)))
if ref_median_max > REF_MEDIAN_TOL_RAD:
    raise RuntimeError(f'final reference median failed: {ref_median_max}')
ref_los_median = np.median(np.asarray(los_mm[ref_idx, :], dtype=np.float64), axis=0)
ref_los_median_max_mm = float(np.max(np.abs(ref_los_median)))
rng = np.random.default_rng(0)
nsample = min(100000, npoint)
sample_idx = rng.choice(npoint, size=nsample, replace=False)
p_sample = np.asarray(phase_final[sample_idx, :], dtype=np.float64)
l_sample = np.asarray(los_m[sample_idx, :], dtype=np.float64)
los_factor_parity = float(np.max(np.abs(l_sample - p_sample * los_factor_m_per_rad)))
los_sample_mm = l_sample * 1000.0
los_abs_q_mm = np.percentile(np.abs(los_sample_mm), [50, 95, 99, 99.9])
phase_abs_q = np.percentile(np.abs(p_sample), [50, 95, 99, 99.9])
scn_rms = float(np.sqrt(raw_scn_correction_ss / raw_scn_correction_n))
manifest = {'status': 'PASS_FINAL_LOS', 'formula': {'scn_correction': 'phi_raw = phi_preSCN - ph_scn_slave', 'temporal_reference': 'phi_t = phi_raw - phi_raw[:, temporal_reference_index]', 'spatial_reference': 'phi_final = phi_t - median(phi_t[reference points], epoch)', 'los': 'd_LOS_toward = +lambda/(4*pi) * phi_final'}, 'scientific_contract': {'temporal_reference_date': TEMPORAL_REFERENCE_DATE, 'temporal_reference_index_0based': tref0, 'spatial_reference_method': 'median', 'spatial_reference_points': int(ref_idx.size), 'geometric_master_date': GEOMETRIC_MASTER_DATE, 'geometric_master_index_0based': master0, 'los_positive_direction': 'toward_satellite', 'radar_frequency_hz': radar_frequency, 'wavelength_m': wavelength, 'los_factor_m_per_rad': los_factor_m_per_rad, 'los_factor_mm_per_rad': los_factor_m_per_rad * 1000.0}, 'hard_qa': {'finite_phase_fraction': finite_phase, 'finite_los_fraction': finite_los, 'epoch0_phase_max_abs_rad': epoch0_max, 'epoch0_los_max_abs_m': los_epoch0_max, 'reference_phase_median_max_abs_rad': ref_median_max, 'reference_los_median_max_abs_mm': ref_los_median_max_mm, 'los_factor_sample_max_abs_error_m': los_factor_parity, 'float32_phase_storage_max_error_rad': sample_phase_storage_max, 'float32_los_storage_max_error_m': sample_los_storage_max}, 'correction_statistics': {'scn_rms_rad': scn_rms, 'scn_max_abs_rad': raw_scn_correction_max, 'final_phase_max_abs_rad': phase_abs_max, 'final_los_max_abs_m': los_abs_max_m}, 'sample_statistics': {'points': nsample, 'abs_phase_p50_p95_p99_p999_rad': [float(x) for x in phase_abs_q], 'abs_los_p50_p95_p99_p999_mm': [float(x) for x in los_abs_q_mm]}, 'performance': {'materialization_seconds': materialization_seconds, 'point_epochs_per_second': npoint * nepoch / materialization_seconds}, 'inputs': {'pre_scn_phase': str(PRE), 'ph_scn_slave': str(SCN), 'reference_indices': str(REF_FILE)}, 'outputs': {'final_phase_rad': str(PHASE_OUT), 'los_toward_m': str(LOS_M_OUT), 'los_toward_mm': str(LOS_MM_OUT), 'dtype': 'float32'}, 'upstream_modified': False, 'next': 'point_products'}
MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
print('=' * 96)
print('FINAL REFERENCED LOS TIMESERIES')
print('=' * 96)
print('points / acquisitions           :', f'{npoint:,} / {nepoch}')
print('temporal reference              :', f'{TEMPORAL_REFERENCE_DATE} (0b={tref0})')
print('spatial reference               :', f'{ref_idx.size} points / median')
print('geometric master                :', f'{GEOMETRIC_MASTER_DATE} (0b={master0})')
print()
print('radar frequency                 :', f'{radar_frequency:.6f} Hz')
print('wavelength                      :', f'{wavelength:.15f} m')
print('LOS factor                      :', f'+{los_factor_m_per_rad:.15e} m/rad')
print('LOS positive                    :', 'toward satellite')
print()
print('epoch0 phase max |rad|          :', f'{epoch0_max:.12e}')
print('epoch0 LOS max |m|              :', f'{los_epoch0_max:.12e}')
print('reference median max |rad|      :', f'{ref_median_max:.12e}')
print('reference LOS median max |mm|   :', f'{ref_los_median_max_mm:.12e}')
print('LOS-factor parity max |m|       :', f'{los_factor_parity:.12e}')
print()
print('phase float32 storage max err   :', f'{sample_phase_storage_max:.12e} rad')
print('LOS float32 storage max err     :', f'{sample_los_storage_max:.12e} m')
print()
print('SCN RMS                         :', f'{scn_rms:.6f} rad')
print('SCN max |rad|                   :', f'{raw_scn_correction_max:.6f}')
print('|final phase| p50/95/99/999    :', phase_abs_q)
print('|final LOS| p50/95/99/999 mm   :', los_abs_q_mm)
print()
print('materialization seconds         :', f'{materialization_seconds:.6f}')
print('throughput                      :', f'{npoint * nepoch / materialization_seconds:,.0f} point-epochs/s')
print()
print('final phase                     :', PHASE_OUT)
print('LOS toward satellite [m]        :', LOS_M_OUT)
print('LOS toward satellite [mm]       :', LOS_MM_OUT)
print('upstream modified               :', False)
print('manifest                        :', MANIFEST)
print('=' * 96)
print('FINAL RESULT: PASS_FINAL_LOS')
print('=' * 96)
