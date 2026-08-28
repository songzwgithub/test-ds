from pathlib import Path
from datetime import datetime
import json
import math
import re
import time
import numpy as np
import os as _pypsds_os
from pathlib import Path as _PypsdsPath
PUBLIC_PROJECT = _PypsdsPath(_pypsds_os.environ['PYPSDS_SCLA_PROJECT'])
PUBLIC_DATA_ROOT = _PypsdsPath(_pypsds_os.environ['PYPSDS_SCLA_DATA_ROOT'])
PUBLIC_PROC = _PypsdsPath(_pypsds_os.environ['PYPSDS_SCLA_PROC'])
PUBLIC_NETWORK_LOG_DIR = _PypsdsPath(_pypsds_os.environ['PYPSDS_SCLA_NETWORK_LOG_DIR'])
PUBLIC_BASELINE_CONTRACT = _PypsdsPath(_pypsds_os.environ['PYPSDS_SCLA_BASELINE_CONTRACT'])
PUBLIC_NIMAGE = int(_pypsds_os.environ['PYPSDS_SCLA_NIMAGE'])
PUBLIC_NIFG = int(_pypsds_os.environ['PYPSDS_SCLA_NIFG'])
PUBLIC_NSLAVE = int(_pypsds_os.environ['PYPSDS_SCLA_NSLAVE'])
PUBLIC_REFERENCE_COUNT = int(_pypsds_os.environ['PYPSDS_SCLA_REFERENCE_COUNT'])
PUBLIC_MISSING_COUNT = int(_pypsds_os.environ['PYPSDS_SCLA_MISSING_COUNT'])
PUBLIC_ORIGINAL_COUNT = int(_pypsds_os.environ['PYPSDS_SCLA_ORIGINAL_COUNT'])
PUBLIC_GEOMETRIC_MASTER = _pypsds_os.environ['PYPSDS_SCLA_GEOMETRIC_MASTER']
PUBLIC_RSLC_PAR = _PypsdsPath(_pypsds_os.environ['PYPSDS_SCLA_RSLC_PAR'])
PUBLIC_RANGE_LOOKS = int(_pypsds_os.environ['PYPSDS_SCLA_RANGE_LOOKS'])
PUBLIC_AZIMUTH_LOOKS = int(_pypsds_os.environ['PYPSDS_SCLA_AZIMUTH_LOOKS'])
PUBLIC_PLIST = PUBLIC_PROC / 'point_geometry' / 'strict_points.plist'
PUBLIC_REF_FILE = PUBLIC_PROC / 'referenced_timeseries' / 'reference_strict_indices.npy'
PUBLIC_REFERENCE_COVARIANCE = PUBLIC_PROC / '.optional_scla_reference_covariance'
PUBLIC_REGENERATED_BASES = PUBLIC_PROC / 'scla' / 'generated_missing'
ROOT = PUBLIC_DATA_ROOT
PSDS = PUBLIC_PROJECT
PROC = PUBLIC_PROC
PHASE = PROC / 'atmosphere_correction' / 'acquisition_phase_corrected_rad.npy'
GACOS_MANIFEST = PROC / 'atmosphere_correction' / 'atmosphere_correction_manifest.json'
CONTRACT = PUBLIC_BASELINE_CONTRACT
BASE_DIR = PROC / 'scla'
GENERATED = BASE_DIR / 'generated_missing'
CATALOG = BASE_DIR / 'baseline_source_catalog.json'
PLIST = PUBLIC_PLIST
REF_IDX_FILE = PUBLIC_REF_FILE
RSLC_PAR = PUBLIC_RSLC_PAR
NETWORK_LOG_DIR = PUBLIC_NETWORK_LOG_DIR
DIAG_BETA = PROC / 'scla_residual_dem_estimation' / 'scla_beta_rad_per_m_bperp.npy'
GLOBAL_BPERP = PROC / 'network' / 'acquisition_bperp_m.npy'
OUT = PROC / 'scla'
OUT.mkdir(parents=True, exist_ok=True)
K_OUT = OUT / 'K_ps_uw_rad_per_m_bperp.npy'
COEFF_OUT = OUT / 'baseline_sm_transform_coefficients.npz'
MANIFEST = OUT / 'scla_k_manifest.json'
GEOMETRIC_MASTER = PUBLIC_GEOMETRIC_MASTER
CHUNK = 262144
SAMPLE_N = 4096
BPERP_PARITY_TOL_M = 1e-08
K_PARITY_TOL = 1e-10
NUM_RE = re.compile('[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[Ee][-+]?\\d+)?')
PAIR_RE = re.compile('(20\\d{6})[_-](20\\d{6})')

def read_par(path):
    d = {}
    for line in path.read_text(errors='ignore').splitlines():
        if ':' not in line:
            continue
        key, rhs = line.split(':', 1)
        d[key.strip().lower()] = rhs.strip()
    return d

def scalar_from_par(d, keys):
    for key in keys:
        rhs = d.get(key.lower())
        if rhs is None:
            continue
        m = NUM_RE.search(rhs)
        if m:
            return float(m.group(0))
    raise KeyError(' / '.join(keys))

def parse_vector(text, labels):
    for line in text.splitlines():
        if not any((label in line for label in labels)):
            continue
        rhs = line.split(':', 1)[1] if ':' in line else line
        values = NUM_RE.findall(rhs)
        if len(values) >= 3:
            return np.asarray([float(x) for x in values[:3]], dtype=np.float64)
    return None

def parse_base(path):
    text = path.read_text(errors='ignore')
    B = parse_vector(text, ('initial_baseline(TCN)', 'initial_baseline'))
    Br = parse_vector(text, ('initial_baseline_rate', 'baseline_rate(TCN)'))
    if B is None or Br is None or (not np.all(np.isfinite(B))) or (not np.all(np.isfinite(Br))):
        raise RuntimeError(f'invalid GAMMA baseline model: {path}')
    return (B, Br)

def path_from_source_entry(entry):
    if isinstance(entry, str):
        return Path(entry)
    if isinstance(entry, dict):
        for key in ('path', 'base_file', 'file'):
            value = entry.get(key)
            if value:
                return Path(value)
    return None

def pair_orientation(path, date_i, date_j):
    m = PAIR_RE.search(path.name)
    if m is None:
        m = PAIR_RE.search(str(path))
    if m is None:
        raise RuntimeError(f'cannot determine .base orientation: {path}')
    a = m.group(1)
    b = m.group(2)
    if a == date_i and b == date_j:
        return 1
    if a == date_j and b == date_i:
        return -1
    raise RuntimeError(f'.base date mismatch: {path}')
log_re = re.compile('pair(\\d+)_(20\\d{6})_(20\\d{6})_single_ifg\\.log$')
network = []
for p in NETWORK_LOG_DIR.glob('pair*_single_ifg.log'):
    m = log_re.match(p.name)
    if not m:
        continue
    network.append({'edge': int(m.group(1)), 'date_i': m.group(2), 'date_j': m.group(3)})
network.sort(key=lambda x: x['edge'])
if len(network) != PUBLIC_NIFG:
    raise RuntimeError(f'IFG count mismatch, got {len(network)}')
if [x['edge'] for x in network] != list(range(1, PUBLIC_NIFG + 1)):
    raise RuntimeError('production IFG ordering failed')

def ensure_catalog():
    if CATALOG.is_file():
        obj = json.loads(CATALOG.read_text())
        rows = obj.get('pairs', [])
        if len(rows) == PUBLIC_NIFG:
            good = True
            for i, row in enumerate(rows, start=1):
                if int(row.get('edge', -1)) != i:
                    good = False
                    break
                p = Path(row.get('base_file', ''))
                if not p.is_file():
                    good = False
                    break
                parse_base(p)
            if good:
                return (rows, 'existing_complete_catalog')
    if not CONTRACT.is_file():
        raise FileNotFoundError(CONTRACT)
    contract = json.loads(CONTRACT.read_text())
    missing = set(contract.get('missing', []))
    sources = contract.get('sources', {})
    if len(missing) != PUBLIC_MISSING_COUNT:
        raise RuntimeError(f'missing baseline count mismatch, got {len(missing)}')
    rows = []
    n_original = 0
    n_generated = 0
    for item in network:
        edge = item['edge']
        d1 = item['date_i']
        d2 = item['date_j']
        pair = f'{d1}_{d2}'
        if pair in missing:
            p = GENERATED / f'{pair}.base'
            if not p.is_file():
                raise FileNotFoundError(p)
            source_type = 'regenerated_gamma_base_orbit'
            n_generated += 1
        else:
            entries = sources.get(pair)
            if entries is None:
                entries = sources.get(f'{d2}_{d1}')
            if entries is None:
                raise RuntimeError(f'no source mapping for {pair}')
            if not isinstance(entries, list):
                entries = [entries]
            paths = []
            for entry in entries:
                q = path_from_source_entry(entry)
                if q is not None and q.is_file():
                    paths.append(q.resolve())
            if not paths:
                raise RuntimeError(f'no readable original .base for {pair}')
            paths = sorted(set(paths), key=str)
            p = paths[0]
            B0, Br0 = parse_base(p)
            for q in paths[1:]:
                Bq, Brq = parse_base(q)
                if not (np.allclose(Bq, B0, rtol=0.0, atol=1e-06) and np.allclose(Brq, Br0, rtol=0.0, atol=1e-08)):
                    raise RuntimeError(f'conflicting original .base sources for {pair}')
            source_type = 'original_gamma_base'
            n_original += 1
        orientation = pair_orientation(p, d1, d2)
        B, Br = parse_base(p)
        rows.append({'edge': edge, 'date_i': d1, 'date_j': d2, 'orientation': orientation, 'source_type': source_type, 'base_file': str(p), 'baseline_tcn': B.tolist(), 'baseline_rate_tcn': Br.tolist()})
    if len(rows) != PUBLIC_NIFG or n_original != PUBLIC_ORIGINAL_COUNT or n_generated != PUBLIC_MISSING_COUNT:
        raise RuntimeError(f'catalog composition failed: {len(rows)} / {n_original} / {n_generated}')
    payload = {'status': 'PASS_COMPLETE_108_GAMMA_BASE_MODELS', 'original_base_pairs': n_original, 'regenerated_base_pairs': n_generated, 'pairs': rows}
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(json.dumps(payload, indent=2) + '\n')
    return (rows, 'rebuilt_from_current_contract')
catalog_rows, catalog_source = ensure_catalog()
for p in (PHASE, GACOS_MANIFEST, PLIST, REF_IDX_FILE, RSLC_PAR):
    if not p.is_file():
        raise FileNotFoundError(p)
phase = np.load(PHASE, mmap_mode='r')
gman = json.loads(GACOS_MANIFEST.read_text())
dates = list(gman['acquisition_dates'])
if len(dates) != PUBLIC_NIMAGE or phase.shape != (phase.shape[0], PUBLIC_NIMAGE):
    raise RuntimeError(f'phase/date contract failed: {phase.shape}, {len(dates)}')
npoint, nimage = phase.shape
if GEOMETRIC_MASTER not in dates:
    raise RuntimeError('geometric master missing')
master0 = dates.index(GEOMETRIC_MASTER)
master1 = master0 + 1
img0 = np.asarray([i for i in range(nimage) if i != master0], dtype=np.int64)
if img0.size != PUBLIC_NSLAVE:
    raise RuntimeError('non-master acquisition count mismatch')
plist = np.fromfile(PLIST, dtype='>i4').reshape(-1, 2)
if plist.shape[0] != npoint:
    raise RuntimeError('strict point count mismatch')
col = plist[:, 0].astype(np.float64)
row = plist[:, 1].astype(np.float64)
ref_idx = np.load(REF_IDX_FILE).astype(np.int64)
if ref_idx.size != PUBLIC_REFERENCE_COUNT:
    raise RuntimeError(f'reference point count = {ref_idx.size}')
date_to_ix = {d: i for i, d in enumerate(dates)}
G = np.zeros((PUBLIC_NIFG, PUBLIC_NIMAGE), dtype=np.float64)
for e, item in enumerate(network):
    d1 = item['date_i']
    d2 = item['date_j']
    if d1 not in date_to_ix or d2 not in date_to_ix:
        raise RuntimeError(f'network date not in acquisition list: {d1}_{d2}')
    G[e, date_to_ix[d1]] = -1.0
    G[e, date_to_ix[d2]] = +1.0
Gbase = G[:, img0]
rank_Gbase = int(np.linalg.matrix_rank(Gbase))
if rank_Gbase != PUBLIC_NSLAVE:
    raise RuntimeError(f'StaMPS baseline reconstruction rank deficient: {rank_Gbase}/37')
Pbase = np.linalg.pinv(Gbase)
B_T = np.empty(PUBLIC_NIFG, dtype=np.float64)
B_C = np.empty(PUBLIC_NIFG, dtype=np.float64)
B_N = np.empty(PUBLIC_NIFG, dtype=np.float64)
R_T = np.empty(PUBLIC_NIFG, dtype=np.float64)
R_C = np.empty(PUBLIC_NIFG, dtype=np.float64)
R_N = np.empty(PUBLIC_NIFG, dtype=np.float64)
for e, rowcat in enumerate(catalog_rows):
    net = network[e]
    if int(rowcat['edge']) != e + 1 or rowcat['date_i'] != net['date_i'] or rowcat['date_j'] != net['date_j']:
        raise RuntimeError(f'catalog/network mismatch at edge {e + 1}')
    path = Path(rowcat['base_file'])
    B, Br = parse_base(path)
    orientation = int(rowcat.get('orientation', 1))
    if orientation not in (-1, 1):
        raise RuntimeError('invalid baseline orientation')
    B = B * orientation
    Br = Br * orientation
    B_T[e] = B[0]
    B_C[e] = B[1]
    B_N[e] = B[2]
    R_T[e] = Br[0]
    R_C[e] = Br[1]
    R_N[e] = Br[2]
C_SM = Pbase @ B_C
N_SM = Pbase @ B_N
CR_SM = Pbase @ R_C
NR_SM = Pbase @ R_N
rpar = read_par(RSLC_PAR)
rslc_length = int(round(scalar_from_par(rpar, ('azimuth_lines', 'nlines'))))
range_spacing = scalar_from_par(rpar, ('range_pixel_spacing',))
near_range = scalar_from_par(rpar, ('near_range_slc', 'near_range'))
sar_to_earth = scalar_from_par(rpar, ('sar_to_earth_center',))
earth_radius = scalar_from_par(rpar, ('earth_radius_below_sensor',))
prf = scalar_from_par(rpar, ('prf',))
range_looks = PUBLIC_RANGE_LOOKS
azimuth_looks = PUBLIC_AZIMUTH_LOOKS
looks_source = 'derived_from_current_geometry'
mean_azimuth = rslc_length / 2.0 - 0.5

def geometry_factors(row_chunk, col_chunk):
    range_original = col_chunk * range_looks + (range_looks - 1) / 2.0
    azimuth_original = row_chunk * azimuth_looks + (azimuth_looks - 1) / 2.0
    slant_range = near_range + range_original * range_spacing
    look_arg = (sar_to_earth ** 2 + slant_range ** 2 - earth_radius ** 2) / (2.0 * sar_to_earth * slant_range)
    look_arg = np.clip(look_arg, -1.0, 1.0)
    look = np.arccos(look_arg)
    c = np.cos(look)
    s = np.sin(look)
    dt = (azimuth_original - mean_azimuth) / prf
    return (c, s, dt)
sum_c = 0.0
sum_s = 0.0
sum_dtc = 0.0
sum_dts = 0.0
t_geom = time.perf_counter()
for start in range(0, npoint, CHUNK):
    stop = min(start + CHUNK, npoint)
    c, s, dt = geometry_factors(row[start:stop], col[start:stop])
    sum_c += np.sum(c, dtype=np.float64)
    sum_s += np.sum(s, dtype=np.float64)
    sum_dtc += np.sum(dt * c, dtype=np.float64)
    sum_dts += np.sum(dt * s, dtype=np.float64)
geometry_seconds = time.perf_counter() - t_geom
mean_c = sum_c / npoint
mean_s = sum_s / npoint
mean_dtc = sum_dtc / npoint
mean_dts = sum_dts / npoint
mean_bperp_nonmaster = mean_c * C_SM - mean_s * N_SM + mean_dtc * CR_SM - mean_dts * NR_SM
mean_dbperp = np.diff(mean_bperp_nonmaster)
mean_bperp_full = np.zeros(nimage, dtype=np.float64)
mean_bperp_full[img0] = mean_bperp_nonmaster
sample_n = min(SAMPLE_N, npoint)
sample_idx = np.linspace(0, npoint - 1, sample_n, dtype=np.int64)
cs, ss, dts = geometry_factors(row[sample_idx], col[sample_idx])
b_ifg_explicit = cs[:, None] * B_C[None, :] - ss[:, None] * B_N[None, :] + (dts * cs)[:, None] * R_C[None, :] - (dts * ss)[:, None] * R_N[None, :]
b_sm_explicit = b_ifg_explicit @ Pbase.T
b_sm_fast = cs[:, None] * C_SM[None, :] - ss[:, None] * N_SM[None, :] + (dts * cs)[:, None] * CR_SM[None, :] - (dts * ss)[:, None] * NR_SM[None, :]
bperp_parity_max = float(np.max(np.abs(b_sm_fast - b_sm_explicit)))
bperp_parity_rms = float(np.sqrt(np.mean((b_sm_fast - b_sm_explicit) ** 2)))
if bperp_parity_max > BPERP_PARITY_TOL_M:
    raise RuntimeError(f'accelerated Bperp algebra does not match explicit 108->37 reconstruction: {bperp_parity_max}')
date_obj = [datetime.strptime(d, '%Y%m%d') for d in dates]
day = np.asarray([(x - date_obj[0]).days for x in date_obj], dtype=np.float64)
day_seq = np.diff(day[img0])
if mean_dbperp.size != 36 or day_seq.size != 36:
    raise RuntimeError('final-pass sequential design size mismatch')
A2 = np.column_stack((np.ones(36, dtype=np.float64), mean_dbperp, day_seq))
rank_A2 = int(np.linalg.matrix_rank(A2))
if rank_A2 != 3:
    raise RuntimeError(f'StaMPS final-pass design rank deficient: {rank_A2}/3')
norms = np.linalg.norm(A2, axis=0)
cond_A2 = float(np.linalg.cond(A2 / norms[None, :]))
P2 = np.linalg.pinv(A2)
k_diff_weights = P2[1, :]
phase_weights = np.empty(PUBLIC_NSLAVE, dtype=np.float64)
phase_weights[0] = -k_diff_weights[0]
phase_weights[1:-1] = k_diff_weights[:-1] - k_diff_weights[1:]
phase_weights[-1] = k_diff_weights[-1]
phase_weight_sum = float(np.sum(phase_weights))
if abs(phase_weight_sum) > 1e-12:
    raise RuntimeError(f'fused phase-weight constant annihilation failed: {phase_weight_sum}')
ref_phase = np.asarray(phase[ref_idx, :], dtype=np.float64)
ref_mean_sm = np.nanmean(ref_phase, axis=0)
ref_projection = float(ref_mean_sm[img0] @ phase_weights)
sample_phase = np.asarray(phase[sample_idx, :], dtype=np.float64)
sample_centered = sample_phase[:, img0] - ref_mean_sm[img0][None, :]
sample_dph = np.diff(sample_centered, axis=1)
sample_coeff = sample_dph @ P2.T
k_explicit = sample_coeff[:, 1]
k_fast = sample_phase[:, img0] @ phase_weights - ref_projection
k_parity_max = float(np.max(np.abs(k_fast - k_explicit)))
k_parity_rms = float(np.sqrt(np.mean((k_fast - k_explicit) ** 2)))
if k_parity_max > K_PARITY_TOL:
    raise RuntimeError(f'fused K solve does not match explicit StaMPS diff+pinv solve: {k_parity_max}')
K = np.lib.format.open_memmap(K_OUT, mode='w+', dtype=np.float32, shape=(npoint,))
t_k = time.perf_counter()
for start in range(0, npoint, CHUNK):
    stop = min(start + CHUNK, npoint)
    ph = np.asarray(phase[start:stop, :], dtype=np.float32)
    k = ph[:, img0].astype(np.float64) @ phase_weights - ref_projection
    if not np.all(np.isfinite(k)):
        raise RuntimeError(f'non-finite K found in chunk {start}:{stop}')
    K[start:stop] = k.astype(np.float32)
K.flush()
k_seconds = time.perf_counter() - t_k
K_read = np.load(K_OUT, mmap_mode='r')
finite_fraction = float(np.mean(np.isfinite(K_read)))
if finite_fraction != 1.0:
    raise RuntimeError('K finite fraction < 1')
K64 = np.asarray(K_read, dtype=np.float64)
k_q = np.percentile(K64, [1, 5, 50, 95, 99])
abs_k_q = np.percentile(np.abs(K64), [50, 95, 99])
ref_k_mean = float(np.mean(K64[ref_idx]))
ref_k_median = float(np.median(K64[ref_idx]))
H = int(np.max(row)) + 1
W = int(np.max(col)) + 1
grid = np.full((H, W), np.nan, dtype=np.float32)
grid[row.astype(np.int64), col.astype(np.int64)] = K_read
dh = np.abs(grid[:, 1:] - grid[:, :-1])
dv = np.abs(grid[1:, :] - grid[:-1, :])
adj = np.concatenate((dh[np.isfinite(dh)], dv[np.isfinite(dv)]))
adj_med = float(np.median(adj))
rng = np.random.default_rng(0)
nr = min(500000, npoint)
ia = rng.integers(0, npoint, size=nr)
ib = rng.integers(0, npoint, size=nr)
rand_med = float(np.median(np.abs(K64[ia] - K64[ib])))
spatial_ratio = adj_med / rand_med if rand_med > 0.0 else np.nan
diag_corr = None
diag_diff_q = None
if DIAG_BETA.is_file():
    beta_diag = np.load(DIAG_BETA, mmap_mode='r')
    if beta_diag.shape == (npoint,):
        bd = np.asarray(beta_diag, dtype=np.float64)
        diag_corr = float(np.corrcoef(bd, K64)[0, 1])
        diag_diff_q = [float(x) for x in np.percentile(K64 - bd, [1, 50, 99])]
global_baseline_maxdiff = None
if GLOBAL_BPERP.is_file():
    bg = np.load(GLOBAL_BPERP).astype(np.float64).reshape(-1)
    if bg.size == PUBLIC_NIMAGE:
        global_baseline_maxdiff = float(np.max(np.abs(mean_bperp_full - bg)))
np.savez(COEFF_OUT, dates=np.asarray(dates), master0=np.asarray(master0, dtype=np.int64), img0=img0, Pbase=Pbase, C_SM=C_SM, N_SM=N_SM, CR_SM=CR_SM, NR_SM=NR_SM, mean_bperp_full_m=mean_bperp_full, mean_dbperp_m=mean_dbperp, day=day, day_seq=day_seq, A2=A2, P2=P2, phase_weights=phase_weights)
manifest = {'status': 'PASS_STAMPS_FINAL_PASS_K', 'implementation': 'StaMPS / pySTAMPS-GAMMA ps_calc_scla(0,1) final-pass algebraically fused', 'production_phase_modified': False, 'phase_input': str(PHASE), 'points': int(npoint), 'images': int(nimage), 'ifgs': PUBLIC_NIFG, 'geometric_master': {'date': GEOMETRIC_MASTER, 'index_0based': master0, 'index_1based': master1}, 'reference': {'points': int(ref_idx.size), 'internal_method': 'StaMPS arithmetic mean', 'K_reference_mean': ref_k_mean, 'K_reference_median': ref_k_median}, 'baseline_source': {'catalog': str(CATALOG), 'catalog_source': catalog_source, 'pointwise_IFG_matrix_persisted': False, 'pointwise_SM_matrix_persisted': False, 'algebra': 'GAMMA TCN/rate coefficients are transformed by pinv(Gbase) before point evaluation', 'Gbase_shape': list(Gbase.shape), 'Gbase_rank': rank_Gbase}, 'geometry': {'rslc_par': str(RSLC_PAR), 'range_looks': range_looks, 'azimuth_looks': azimuth_looks, 'looks_source': looks_source, 'rslc_length': rslc_length, 'range_pixel_spacing_m': range_spacing, 'near_range_m': near_range, 'prf_hz': prf}, 'design': {'A2_shape': list(A2.shape), 'rank': rank_A2, 'normalized_condition_number': cond_A2, 'phase_weight_sum': phase_weight_sum}, 'hard_parity': {'sample_points': sample_n, 'Bperp_explicit_vs_fast_max_abs_m': bperp_parity_max, 'Bperp_explicit_vs_fast_rms_m': bperp_parity_rms, 'Bperp_tolerance_m': BPERP_PARITY_TOL_M, 'K_explicit_vs_fast_max_abs': k_parity_max, 'K_explicit_vs_fast_rms': k_parity_rms, 'K_tolerance': K_PARITY_TOL}, 'K_statistics': {'p01_p05_p50_p95_p99': [float(x) for x in k_q], 'abs_p50_p95_p99': [float(x) for x in abs_k_q], 'finite_fraction': finite_fraction, 'adjacent_abs_dK_median': adj_med, 'random_abs_dK_median': rand_med, 'adjacent_to_random_ratio': spatial_ratio}, 'diagnostic_only_comparison': {'P15_5B_beta_correlation': diag_corr, 'K_minus_P15_5B_beta_p01_p50_p99': diag_diff_q, 'mean_pointwise_vs_existing_global_Bperp_max_abs_m': global_baseline_maxdiff}, 'performance': {'geometry_mean_seconds': geometry_seconds, 'K_estimation_seconds': k_seconds, 'K_points_per_second': npoint / k_seconds}, 'outputs': {'K_ps_uw': str(K_OUT), 'baseline_transform_coefficients': str(COEFF_OUT)}, 'next': '5B5 pointwise ph_scla + sm_cov/C_ps_uw parity'}
MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
print('=' * 92)
print('5B4 STAMPS FINAL PASS K_ps_uw')
print('=' * 92)
print('points                         :', f'{npoint:,}')
print('images / IFGs                  :', f'{nimage} / 108')
print('geometric master               :', f'{GEOMETRIC_MASTER} (0b={master0}, 1b={master1})')
print('non-master images              :', img0.size)
print('Gbase rank                     :', f'{rank_Gbase} / {img0.size}')
print('baseline catalog               :', catalog_source)
print()
print('A2 shape                       :', A2.shape)
print('A2 rank                        :', rank_A2)
print('A2 normalized condition        :', f'{cond_A2:.6f}')
print()
print('Bperp fast parity max |diff| m :', f'{bperp_parity_max:.12e}')
print('Bperp fast parity RMS m        :', f'{bperp_parity_rms:.12e}')
print('K fast parity max |diff|       :', f'{k_parity_max:.12e}')
print('K fast parity RMS              :', f'{k_parity_rms:.12e}')
print()
print('geometry mean seconds          :', f'{geometry_seconds:.6f}')
print('K estimation seconds           :', f'{k_seconds:.6f}')
print('K throughput                   :', f'{npoint / k_seconds:,.0f} points/s')
print()
print('K p01/p05/p50/p95/p99         :', k_q)
print('|K| p50/p95/p99               :', abs_k_q)
print('reference K mean               :', f'{ref_k_mean:.12e}')
print('reference K median             :', f'{ref_k_median:.12e}')
print('adjacent/random K ratio        :', f'{spatial_ratio:.6f}')
if diag_corr is not None:
    print('corr(K, 5B diagnostic)    :', f'{diag_corr:.6f}')
if global_baseline_maxdiff is not None:
    print('mean-point/global Bperp max d :', f'{global_baseline_maxdiff:.6f} m')
print()
print('108-column Bperp persisted     :', False)
print('37-column Bperp persisted      :', False)
print('production phase modified      :', False)
print('K output                       :', K_OUT)
print('coefficients                   :', COEFF_OUT)
print('manifest                       :', MANIFEST)
print('=' * 92)
print('5B4 FINAL RESULT: PASS_STAMPS_FINAL_PASS_K')
print('=' * 92)
