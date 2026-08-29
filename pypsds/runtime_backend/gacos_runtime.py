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
import csv
import json
import math
import os
import re
import time
import numpy as np
from numba import njit, prange
PSDS = PUBLIC_PROJECT
PROC = PUBLIC_PROC
REFDIR = PROC / 'referenced_timeseries'
SRC_PHASE = REFDIR / 'acquisition_phase_referenced_rad.npy'
GACOS = PUBLIC_GACOS
GEOM = PUBLIC_GEOM_COMPAT
SMOKE = PUBLIC_ATM
CACHE = PUBLIC_ATM_CACHE
OUTDIR = PUBLIC_ATM
OUTDIR.mkdir(parents=True, exist_ok=True)
FINAL = PUBLIC_ATM / 'acquisition_phase_corrected_rad.npy'
TMP = PUBLIC_ATM / '.acquisition_phase_corrected_rad.tmp.npy'
STATS = PUBLIC_ATM / 'atmosphere_correction_epoch_stats.csv'
MANIFEST = PUBLIC_ATM / 'atmosphere_correction_manifest.json'
REF_DATE = PUBLIC_REF_DATE
C0 = 299792458.0
CHUNK = int(os.environ.get('P15_CHUNK_POINTS', '262144'))
FORCE = (os.environ.get('PYPSDS_FORCE', '0') == '1' or os.environ.get('P15_FORCE', '0') == '1')
DATE_RE = re.compile('^\\d{8}$')

def par_scalar(path, key):
    rx = re.compile('[-+]?\\d+(?:\\.\\d*)?(?:[Ee][-+]?\\d+)?')
    for line in path.read_text(errors='ignore').splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        if k.strip().lower() == key.lower():
            m = rx.search(v)
            if m:
                return float(m.group(0))
    raise KeyError(key)

def read_rsc(path):
    d = {}
    for line in path.read_text(errors='ignore').splitlines():
        p = line.strip().split()
        if len(p) >= 2 and (not line.lstrip().startswith('#')):
            d[p[0].upper()] = p[1]
    return d

def normalize_dates(x):
    a = np.asarray(x).reshape(-1)
    out = []
    for v in a:
        if isinstance(v, (bytes, np.bytes_)):
            s = bytes(v).decode(errors='ignore')
        else:
            s = str(v)
        s = re.sub('[^0-9]', '', s)
        if len(s) >= 8:
            s = s[:8]
        if not DATE_RE.match(s):
            return None
        out.append(s)
    return out

def discover_phase_dates(expected_dates):
    expected_set = set(expected_dates)
    hits = []
    candidates = sorted(PROC.rglob('*date*.npy')) + sorted(PROC.rglob('*dates*.npy'))
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        try:
            vals = normalize_dates(np.load(p, allow_pickle=False))
        except Exception:
            continue
        if vals and len(vals) == len(expected_dates) and (set(vals) == expected_set):
            hits.append((str(p), vals))

    def walk(obj, path=''):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                kp = f'{path}.{k}' if path else str(k)
                if isinstance(v, list):
                    vals = normalize_dates(v)
                    if vals and len(vals) == len(expected_dates) and (set(vals) == expected_set):
                        found.append((kp, vals))
                found.extend(walk(v, kp))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                found.extend(walk(v, f'{path}[{i}]'))
        return found
    for p in sorted(PROC.rglob('*.json')):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        for keypath, vals in walk(obj):
            hits.append((f'{p}:{keypath}', vals))
    if not hits:
        raise RuntimeError('Cannot prove acquisition-date order. No date sequence matching the GACOS set was found.')
    orders = {}
    for src, vals in hits:
        orders.setdefault(tuple(vals), []).append(src)
    if len(orders) != 1:
        raise RuntimeError('Conflicting acquisition-date orders found.')
    order = list(next(iter(orders.keys())))
    sources = next(iter(orders.values()))
    return (order, sources)
if not SRC_PHASE.is_file():
    raise FileNotFoundError(SRC_PHASE)
for p in (CACHE / 'base.npy', CACHE / 'fx.npy', CACHE / 'fy.npy', CACHE / 'sec_inc.npy', CACHE / 'ref_idx.npy'):
    if not p.is_file():
        raise FileNotFoundError(p)
if FINAL.exists() and (not FORCE):
    raise RuntimeError(f'\nRefusing to overwrite existing product:\n{FINAL}\n\nUse `pypsds run --force` for an intentional rebuild (legacy P15_FORCE=1 is also accepted).')
if TMP.exists():
    TMP.unlink()
src = np.load(SRC_PHASE, mmap_mode='r')
if src.ndim != 2:
    raise RuntimeError(f'source phase must be 2-D: {src.shape}')
if src.dtype not in (np.float32, np.float64):
    raise RuntimeError(f'unexpected source phase dtype: {src.dtype}')
ztd_paths = sorted(GACOS.glob('*.ztd'))
gacos_dates = [p.stem for p in ztd_paths]
if len(gacos_dates) != PUBLIC_NDATE or len(set(gacos_dates)) != PUBLIC_NDATE:
    raise RuntimeError(f'expected 38 unique GACOS dates; got {len(gacos_dates)}')
if REF_DATE not in gacos_dates:
    raise RuntimeError(f'reference date missing: {REF_DATE}')
phase_dates, date_sources = discover_phase_dates(gacos_dates)
if phase_dates[0] != REF_DATE:
    raise RuntimeError(f'phase epoch 0 is not {REF_DATE}: {phase_dates[0]}')
path_by_date = {p.stem: p for p in ztd_paths}
ztd_paths = [path_by_date[d] for d in phase_dates]
r0 = read_rsc(Path(str(ztd_paths[0]) + '.rsc'))
width = int(r0['WIDTH'])
length = int(r0['FILE_LENGTH'])
x_first = float(r0['X_FIRST'])
y_first = float(r0['Y_FIRST'])
x_step = float(r0['X_STEP'])
y_step = float(r0['Y_STEP'])
grid_sig = (width, length, x_first, y_first, x_step, y_step)
grid_n = width * length
for p in ztd_paths:
    r = read_rsc(Path(str(p) + '.rsc'))
    sig = (int(r['WIDTH']), int(r['FILE_LENGTH']), float(r['X_FIRST']), float(r['Y_FIRST']), float(r['X_STEP']), float(r['Y_STEP']))
    if sig != grid_sig:
        raise RuntimeError(f'GACOS grid mismatch: {p}')
    if p.stat().st_size != grid_n * 4:
        raise RuntimeError(f'GACOS byte-size mismatch: {p}')
ndate = len(phase_dates)
if src.shape[1] == ndate:
    npoint = src.shape[0]
    point_major = True
elif src.shape[0] == ndate:
    npoint = src.shape[1]
    point_major = False
else:
    raise RuntimeError(f'phase shape {src.shape} incompatible with {ndate} dates')
base = np.load(CACHE / 'base.npy', mmap_mode='r')
fx = np.load(CACHE / 'fx.npy', mmap_mode='r')
fy = np.load(CACHE / 'fy.npy', mmap_mode='r')
sec_inc = np.load(CACHE / 'sec_inc.npy', mmap_mode='r')
ref_idx = np.load(CACHE / 'ref_idx.npy')
if not base.size == fx.size == fy.size == sec_inc.size == npoint:
    raise RuntimeError('GACOS interpolation-cache size mismatch')
if ref_idx.ndim != 1 or ref_idx.size != PUBLIC_NREF:
    raise RuntimeError(f'reference-index contract failed: {ref_idx.shape}')
ztd_px_date = np.empty((grid_n, ndate), dtype=np.float32)
ztd_median = np.empty(ndate, dtype=np.float64)
t_read0 = time.perf_counter()
for e, p in enumerate(ztd_paths):
    z = np.fromfile(p, dtype='<f4')
    if z.size != grid_n or not np.all(np.isfinite(z)):
        raise RuntimeError(f'invalid GACOS ZTD: {p}')
    ztd_px_date[:, e] = z
    ztd_median[e] = np.median(z)
gacos_read_seconds = time.perf_counter() - t_read0

@njit(parallel=True, fastmath=False, cache=True)
def sample_dlos_block(ztd, base, fx, fy, sec_inc, width, ref_epoch, out):
    n = base.size
    ne = ztd.shape[1]
    for k in prange(n):
        b = base[k]
        x = fx[k]
        y = fy[k]
        w00 = (1.0 - x) * (1.0 - y)
        w01 = x * (1.0 - y)
        w10 = (1.0 - x) * y
        w11 = x * y
        s = sec_inc[k]
        z0 = (w00 * ztd[b, ref_epoch] + w01 * ztd[b + 1, ref_epoch] + w10 * ztd[b + width, ref_epoch] + w11 * ztd[b + width + 1, ref_epoch]) * s
        for e in range(ne):
            ze = (w00 * ztd[b, e] + w01 * ztd[b + 1, e] + w10 * ztd[b + width, e] + w11 * ztd[b + width + 1, e]) * s
            out[k, e] = ze - z0

@njit(parallel=True, fastmath=False, cache=True)
def apply_phase_block(src_block, dlos, dlos_ref_med, final_ref_phase_offset, phase_factor, out):
    n, ne = src_block.shape
    for k in prange(n):
        for e in range(ne):
            corr = phase_factor * (dlos[k, e] - dlos_ref_med[e])
            out[k, e] = src_block[k, e] + corr - final_ref_phase_offset[e]
warm_n = min(1024, npoint)
warm_d = np.empty((warm_n, ndate), dtype=np.float32)
sample_dlos_block(ztd_px_date, np.asarray(base[:warm_n]), np.asarray(fx[:warm_n]), np.asarray(fy[:warm_n]), np.asarray(sec_inc[:warm_n]), width, 0, warm_d)
warm_s = np.zeros_like(warm_d)
warm_o = np.empty_like(warm_d)
apply_phase_block(warm_s, warm_d, np.zeros(ndate, np.float32), np.zeros(ndate, np.float32), 1.0, warm_o)
ref_dlos = np.empty((ref_idx.size, ndate), dtype=np.float32)
sample_dlos_block(ztd_px_date, np.asarray(base[ref_idx]), np.asarray(fx[ref_idx]), np.asarray(fy[ref_idx]), np.asarray(sec_inc[ref_idx]), width, 0, ref_dlos)
dlos_ref_med = np.median(ref_dlos.astype(np.float64), axis=0).astype(np.float32)
if point_major:
    src_ref = np.asarray(src[ref_idx, :], dtype=np.float32)
    src_first = np.asarray(src[:, 0], dtype=np.float32)
else:
    src_ref = np.asarray(src[:, ref_idx].T, dtype=np.float32)
    src_first = np.asarray(src[0, :], dtype=np.float32)
src_ref_med = np.median(src_ref.astype(np.float64), axis=0)
src_ref_maxabs = float(np.max(np.abs(src_ref_med)))
src_first_maxabs = float(np.max(np.abs(src_first)))
if src_ref_maxabs > 1e-05:
    raise RuntimeError(f'source reference median not zero enough: {src_ref_maxabs}')
if src_first_maxabs > 1e-05:
    raise RuntimeError(f'source first epoch not zero enough: {src_first_maxabs}')
radar_frequency = par_scalar(PUBLIC_RSLC_PAR, 'radar_frequency')
wavelength = C0 / radar_frequency
phase_factor = 4.0 * math.pi / wavelength
ref_phase_pre = src_ref.astype(np.float64) + phase_factor * (ref_dlos.astype(np.float64) - dlos_ref_med.astype(np.float64)[None, :])
final_ref_phase_offset = np.median(ref_phase_pre, axis=0).astype(np.float32)
ref_phase_post = ref_phase_pre - final_ref_phase_offset[None, :]
ref_post_med = np.median(ref_phase_post, axis=0)
if np.max(np.abs(ref_post_med)) > 1e-06:
    raise RuntimeError('corrected reference median invariant failed')
if abs(float(dlos_ref_med[0])) > 1e-08:
    raise RuntimeError('reference-date atmospheric delay is not zero')
if abs(float(final_ref_phase_offset[0])) > 1e-06:
    raise RuntimeError('reference-date final phase offset is not zero')
dst = np.lib.format.open_memmap(TMP, mode='w+', dtype=np.float32, shape=(npoint, ndate), fortran_order=False)
t_prod0 = time.perf_counter()
for s in range(0, npoint, CHUNK):
    e = min(npoint, s + CHUNK)
    m = e - s
    dlos = np.empty((m, ndate), dtype=np.float32)
    sample_dlos_block(ztd_px_date, np.asarray(base[s:e]), np.asarray(fx[s:e]), np.asarray(fy[s:e]), np.asarray(sec_inc[s:e]), width, 0, dlos)
    if point_major:
        src_block = np.asarray(src[s:e, :], dtype=np.float32)
    else:
        src_block = np.asarray(src[:, s:e].T, dtype=np.float32)
    out_block = np.empty_like(src_block)
    apply_phase_block(src_block, dlos, dlos_ref_med, final_ref_phase_offset, phase_factor, out_block)
    dst[s:e, :] = out_block
dst.flush()
production_seconds = time.perf_counter() - t_prod0
del dst
os.replace(TMP, FINAL)
corr = np.load(FINAL, mmap_mode='r')
if corr.shape != (npoint, ndate) or corr.dtype != np.float32:
    raise RuntimeError(f'output contract failed: {corr.shape} {corr.dtype}')
corr_ref = np.asarray(corr[ref_idx, :], dtype=np.float64)
corr_ref_med = np.median(corr_ref, axis=0)
corr_ref_maxabs = float(np.max(np.abs(corr_ref_med)))
corr_first_maxabs = float(np.max(np.abs(np.asarray(corr[:, 0], dtype=np.float64))))
if corr_ref_maxabs > 1e-06:
    raise RuntimeError(f'corrected reference median failed: {corr_ref_maxabs}')
if corr_first_maxabs > 1e-06:
    raise RuntimeError(f'corrected first epoch failed: {corr_first_maxabs}')
diag_count = min(npoint, 200000)
diag_idx = np.linspace(0, npoint - 1, diag_count, dtype=np.int64)
if point_major:
    src_diag = np.asarray(src[diag_idx, :], dtype=np.float64)
else:
    src_diag = np.asarray(src[:, diag_idx].T, dtype=np.float64)
corr_diag = np.asarray(corr[diag_idx, :], dtype=np.float64)
delta_diag = corr_diag - src_diag
rows = []
for e, d in enumerate(phase_dates):
    q = np.percentile(delta_diag[:, e], [1, 5, 50, 95, 99])
    rows.append({'date': d, 'ztd_median_m': float(ztd_median[e]), 'dlos_ref_median_m': float(dlos_ref_med[e]), 'final_phase_reref_offset_rad': float(final_ref_phase_offset[e]), 'phase_add_p01_rad': float(q[0]), 'phase_add_p05_rad': float(q[1]), 'phase_add_p50_rad': float(q[2]), 'phase_add_p95_rad': float(q[3]), 'phase_add_p99_rad': float(q[4]), 'corrected_ref_median_rad': float(corr_ref_med[e])})
with STATS.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
manifest = {'status': 'PASS_GACOS_CORRECTED_REFERENCED_PHASE', 'product_role': 'atmosphere-corrected referenced phase; NOT final geodetic LOS', 'source_phase': str(SRC_PHASE), 'source_phase_immutable': True, 'output_phase': str(FINAL), 'output_dtype': 'float32', 'output_shape': [int(npoint), int(ndate)], 'acquisition_dates': phase_dates, 'date_order_sources': date_sources, 'reference_date': REF_DATE, 'reference_points': int(ref_idx.size), 'gacos_grid': {'width': width, 'length': length, 'x_first': x_first, 'y_first': y_first, 'x_step': x_step, 'y_step': y_step}, 'incidence': {'source': str(GEOM / 'incidence_gamma_compatible_fast_rad.npy'), 'mapping': 'ZTD / cos(theta)'}, 'phase_convention': {'correction_sign': 'PLUS', 'formula': 'phi_corr = phi_obs + (4*pi/lambda)*dL_ref, followed by exact same-region phase re-reference', 'wavelength_m': wavelength, 'phase_factor_rad_per_m': phase_factor}, 'performance': {'chunk_points': CHUNK, 'gacos_read_seconds': gacos_read_seconds, 'production_seconds': production_seconds, 'point_epochs_per_second': npoint * ndate / production_seconds}, 'qa': {'source_ref_median_max_abs_rad': src_ref_maxabs, 'source_first_epoch_max_abs_rad': src_first_maxabs, 'corrected_ref_median_max_abs_rad': corr_ref_maxabs, 'corrected_first_epoch_max_abs_rad': corr_first_maxabs, 'reference_dlos_epoch0_m': float(dlos_ref_med[0]), 'final_reref_offset_epoch0_rad': float(final_ref_phase_offset[0])}, 'intermediate_atmospheric_matrix_persisted': False, 'next': '5B_RESIDUAL_DEM_SCLA_ESTIMATION'}
MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
print('=' * 88)
print('5A FAST PRODUCTION GACOS CORRECTION')
print('=' * 88)
print('points                    :', f'{npoint:,}')
print('epochs                    :', ndate)
print('source shape              :', tuple(src.shape))
print('source dtype              :', src.dtype)
print('source point-major        :', point_major)
print('output shape              :', tuple(corr.shape))
print('chunk points              :', f'{CHUNK:,}')
print('GACOS read seconds        :', f'{gacos_read_seconds:.6f}')
print('production seconds        :', f'{production_seconds:.6f}')
print('throughput                :', f'{npoint * ndate / production_seconds:,.0f} point-epochs/s')
print('source ref max |median|   :', f'{src_ref_maxabs:.3e} rad')
print('corrected ref max |median|:', f'{corr_ref_maxabs:.3e} rad')
print('corrected epoch0 max |phi|:', f'{corr_first_maxabs:.3e} rad')
print('phase factor              :', f'+{phase_factor:.12f} rad/m')
print('output                    :', FINAL)
print('stats                     :', STATS)
print('manifest                  :', MANIFEST)
print('original phase modified   :', False)
print('=' * 88)
print('5A FINAL RESULT: PASS_GACOS_CORRECTED_REFERENCED_PHASE')
print('=' * 88)
