from pathlib import Path
from datetime import datetime
import json
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
GMAN = PROC / 'atmosphere_correction' / 'atmosphere_correction_manifest.json'
SCLA = PROC / 'scla'
K_FILE = SCLA / 'K_ps_uw_rad_per_m_bperp.npy'
COEFF_FILE = SCLA / 'baseline_sm_transform_coefficients.npz'
PLIST = PUBLIC_PLIST
REF_FILE = PUBLIC_REF_FILE
RSLC_PAR = PUBLIC_RSLC_PAR
NETWORK_LOG_DIR = PUBLIC_NETWORK_LOG_DIR
OLD_SM_COV = PUBLIC_REFERENCE_COVARIANCE
OUT = PROC / 'scla'
C_OUT = OUT / 'C_ps_uw_rad.npy'
SMCOV_OUT = OUT / 'sm_cov_unit_ifg_geom_master.npy'
GLS_OUT = OUT / 'C_ps_uw_gls_contract.npz'
MANIFEST = OUT / 'scla_c_manifest.json'
GEOMETRIC_MASTER = PUBLIC_GEOMETRIC_MASTER
CHUNK = 262144
SAMPLE_N = 4096
SMCOV_PARITY_TOL = 1e-12
C_PARITY_TOL = 1e-10
NUM_RE = re.compile('[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[Ee][-+]?\\d+)?')

def read_par(path):
    out = {}
    for line in path.read_text(errors='ignore').splitlines():
        if ':' not in line:
            continue
        key, rhs = line.split(':', 1)
        out[key.strip().lower()] = rhs.strip()
    return out

def par_scalar(pars, keys):
    for key in keys:
        rhs = pars.get(key.lower())
        if rhs is None:
            continue
        m = NUM_RE.search(rhs)
        if m:
            return float(m.group(0))
    raise KeyError(' / '.join(keys))

def gls_projector(A, covariance):
    """
    pySTAMPS-GAMMA / MATLAB lscov equivalent:

        P = (A' C^-1 A)^-1 A' C^-1

        coefficient_row = y @ P.T
    """
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(covariance, dtype=np.float64)
    CiA = np.linalg.solve(C, A)
    normal = A.T @ CiA
    if np.linalg.matrix_rank(normal) != normal.shape[0]:
        raise RuntimeError('GLS normal matrix rank deficient')
    return np.linalg.solve(normal, CiA.T)
for p in (PHASE, GMAN, K_FILE, COEFF_FILE, PLIST, REF_FILE, RSLC_PAR):
    if not p.is_file():
        raise FileNotFoundError(p)
phase = np.load(PHASE, mmap_mode='r')
K = np.load(K_FILE, mmap_mode='r')
coef = np.load(COEFF_FILE, allow_pickle=False)
gman = json.loads(GMAN.read_text())
dates = list(gman['acquisition_dates'])
npoint, nimage = phase.shape
if nimage != PUBLIC_NIMAGE or len(dates) != PUBLIC_NIMAGE or K.shape != (npoint,):
    raise RuntimeError(f'input contract failed: phase={phase.shape}, K={K.shape}, dates={len(dates)}')
master0 = int(np.asarray(coef['master0']).reshape(-1)[0])
img0 = np.asarray(coef['img0'], dtype=np.int64).reshape(-1)
if dates[master0] != GEOMETRIC_MASTER:
    raise RuntimeError(f'geometric-master contract failed: {dates[master0]}')
if img0.size != PUBLIC_NSLAVE:
    raise RuntimeError(f'non-master acquisition count mismatch, got {img0.size}')
Pbase = np.asarray(coef['Pbase'], dtype=np.float64)
C_SM = np.asarray(coef['C_SM'], dtype=np.float64).reshape(-1)
N_SM = np.asarray(coef['N_SM'], dtype=np.float64).reshape(-1)
CR_SM = np.asarray(coef['CR_SM'], dtype=np.float64).reshape(-1)
NR_SM = np.asarray(coef['NR_SM'], dtype=np.float64).reshape(-1)
if Pbase.shape != (PUBLIC_NSLAVE, PUBLIC_NIFG):
    raise RuntimeError(f'Pbase shape={Pbase.shape}')
for name, x in (('C_SM', C_SM), ('N_SM', N_SM), ('CR_SM', CR_SM), ('NR_SM', NR_SM)):
    if x.shape != (PUBLIC_NSLAVE,):
        raise RuntimeError(f'{name} shape={x.shape}')
log_re = re.compile('pair(\\d+)_(20\\d{6})_(20\\d{6})_single_ifg\\.log$')
network = []
for p in NETWORK_LOG_DIR.glob('pair*_single_ifg.log'):
    m = log_re.match(p.name)
    if not m:
        continue
    network.append((int(m.group(1)), m.group(2), m.group(3)))
network.sort()
if len(network) != PUBLIC_NIFG or [x[0] for x in network] != list(range(1, PUBLIC_NIFG + 1)):
    raise RuntimeError('network contract failed')
date_to_ix = {d: i for i, d in enumerate(dates)}
G = np.zeros((PUBLIC_NIFG, PUBLIC_NIMAGE), dtype=np.float64)
for e, (_, d1, d2) in enumerate(network):
    G[e, date_to_ix[d1]] = -1.0
    G[e, date_to_ix[d2]] = +1.0
Gbase = G[:, img0]
rank_g = int(np.linalg.matrix_rank(Gbase))
if rank_g != PUBLIC_NSLAVE:
    raise RuntimeError(f'Gbase rank={rank_g}/37')
sm_cov_direct = np.linalg.inv(Gbase.T @ Gbase)
sm_cov_from_pinv = Pbase @ Pbase.T
smcov_internal_max = float(np.max(np.abs(sm_cov_direct - sm_cov_from_pinv)))
if smcov_internal_max > SMCOV_PARITY_TOL:
    raise RuntimeError(f'sm_cov derivation parity failed: {smcov_internal_max}')
sm_cov_full = np.zeros((PUBLIC_NIMAGE, PUBLIC_NIMAGE), dtype=np.float64)
sm_cov_full[np.ix_(img0, img0)] = sm_cov_direct
np.save(SMCOV_OUT, sm_cov_full)
old_cov_max = None
old_cov_rms = None
if OLD_SM_COV.is_file():
    old = np.load(OLD_SM_COV).astype(np.float64)
    if old.shape == (PUBLIC_NIMAGE, PUBLIC_NIMAGE):
        diff = sm_cov_full - old
        old_cov_max = float(np.max(np.abs(diff)))
        old_cov_rms = float(np.sqrt(np.mean(diff * diff)))
date_obj = [datetime.strptime(d, '%Y%m%d') for d in dates]
day = np.asarray([(d - date_obj[0]).days for d in date_obj], dtype=np.float64)
Ac = np.column_stack((np.ones(PUBLIC_NSLAVE, dtype=np.float64), day[img0] - day[master0]))
if np.linalg.matrix_rank(Ac) != 2:
    raise RuntimeError('C design rank deficient')
Pc = gls_projector(Ac, sm_cov_direct)
if Pc.shape != (2, PUBLIC_NSLAVE):
    raise RuntimeError(f'Pc shape={Pc.shape}')
c_weights = Pc[0, :]
velocity_weights = Pc[1, :]
constant_gain = float(np.sum(c_weights))
time_leak = float(c_weights @ (day[img0] - day[master0]))
if abs(constant_gain - 1.0) > 1e-12:
    raise RuntimeError(f'C intercept constant gain failed: {constant_gain}')
if abs(time_leak) > 1e-08:
    raise RuntimeError(f'C intercept time leakage failed: {time_leak}')
ref_idx = np.load(REF_FILE).astype(np.int64)
if ref_idx.size != PUBLIC_REFERENCE_COUNT:
    raise RuntimeError(f'reference count={ref_idx.size}')
ref_mean_sm = np.nanmean(np.asarray(phase[ref_idx, :], dtype=np.float64), axis=0)
phase_reference_projection = float(ref_mean_sm[img0] @ c_weights)
plist = np.fromfile(PLIST, dtype='>i4').reshape(-1, 2)
if plist.shape[0] != npoint:
    raise RuntimeError('strict point count mismatch')
col = plist[:, 0].astype(np.float64)
row = plist[:, 1].astype(np.float64)
pars = read_par(RSLC_PAR)
rslc_length = int(round(par_scalar(pars, ('azimuth_lines', 'nlines'))))
range_spacing = par_scalar(pars, ('range_pixel_spacing',))
near_range = par_scalar(pars, ('near_range_slc', 'near_range'))
sar_to_earth = par_scalar(pars, ('sar_to_earth_center',))
earth_radius = par_scalar(pars, ('earth_radius_below_sensor',))
prf = par_scalar(pars, ('prf',))
range_looks = PUBLIC_RANGE_LOOKS
azimuth_looks = PUBLIC_AZIMUTH_LOOKS
mean_azimuth = rslc_length / 2.0 - 0.5

def geometry_factors(rr, cc):
    range_original = cc * range_looks + (range_looks - 1) / 2.0
    azimuth_original = rr * azimuth_looks + (azimuth_looks - 1) / 2.0
    slant_range = near_range + range_original * range_spacing
    look_arg = (sar_to_earth ** 2 + slant_range ** 2 - earth_radius ** 2) / (2.0 * sar_to_earth * slant_range)
    look = np.arccos(np.clip(look_arg, -1.0, 1.0))
    cs = np.cos(look)
    ss = np.sin(look)
    dt = (azimuth_original - mean_azimuth) / prf
    return (cs, ss, dt)
CW_C = float(C_SM @ c_weights)
CW_N = float(N_SM @ c_weights)
CW_CR = float(CR_SM @ c_weights)
CW_NR = float(NR_SM @ c_weights)

def baseline_c_projection(rr, cc):
    cs, ss, dt = geometry_factors(rr, cc)
    return cs * CW_C - ss * CW_N + dt * cs * CW_CR - dt * ss * CW_NR
sample_n = min(SAMPLE_N, npoint)
sample_idx = np.linspace(0, npoint - 1, sample_n, dtype=np.int64)
cs, ss, dt = geometry_factors(row[sample_idx], col[sample_idx])
bsm = cs[:, None] * C_SM[None, :] - ss[:, None] * N_SM[None, :] + (dt * cs)[:, None] * CR_SM[None, :] - (dt * ss)[:, None] * NR_SM[None, :]
sample_phase = np.asarray(phase[sample_idx, :], dtype=np.float64)
sample_K = np.asarray(K[sample_idx], dtype=np.float64)
sample_y = sample_phase[:, img0] - ref_mean_sm[img0][None, :] - sample_K[:, None] * bsm
sample_coeff = sample_y @ Pc.T
C_explicit = sample_coeff[:, 0]
bproj_fast = baseline_c_projection(row[sample_idx], col[sample_idx])
C_fast = sample_phase[:, img0] @ c_weights - phase_reference_projection - sample_K * bproj_fast
C_parity_diff = C_fast - C_explicit
C_parity_max = float(np.max(np.abs(C_parity_diff)))
C_parity_rms = float(np.sqrt(np.mean(C_parity_diff * C_parity_diff)))
if C_parity_max > C_PARITY_TOL:
    raise RuntimeError(f'fused C solve parity failed: {C_parity_max}')
Cout = np.lib.format.open_memmap(C_OUT, mode='w+', dtype=np.float32, shape=(npoint,))
t0 = time.perf_counter()
for start in range(0, npoint, CHUNK):
    stop = min(start + CHUNK, npoint)
    ph = np.asarray(phase[start:stop, :], dtype=np.float32)
    kk = np.asarray(K[start:stop], dtype=np.float64)
    bproj = baseline_c_projection(row[start:stop], col[start:stop])
    cc = ph[:, img0].astype(np.float64) @ c_weights - phase_reference_projection - kk * bproj
    if not np.all(np.isfinite(cc)):
        raise RuntimeError(f'non-finite C in {start}:{stop}')
    Cout[start:stop] = cc.astype(np.float32)
Cout.flush()
production_seconds = time.perf_counter() - t0
Cread = np.load(C_OUT, mmap_mode='r')
C64 = np.asarray(Cread, dtype=np.float64)
finite_fraction = float(np.mean(np.isfinite(C64)))
if finite_fraction != 1.0:
    raise RuntimeError('C finite fraction < 1')
c_q = np.percentile(C64, [1, 5, 50, 95, 99])
abs_c_q = np.percentile(np.abs(C64), [50, 95, 99])
ref_c_mean = float(np.mean(C64[ref_idx]))
ref_c_median = float(np.median(C64[ref_idx]))
H = int(row.max()) + 1
W = int(col.max()) + 1
grid = np.full((H, W), np.nan, dtype=np.float32)
grid[row.astype(np.int64), col.astype(np.int64)] = Cread
dh = np.abs(grid[:, 1:] - grid[:, :-1])
dv = np.abs(grid[1:, :] - grid[:-1, :])
adj = np.concatenate((dh[np.isfinite(dh)], dv[np.isfinite(dv)]))
adj_med = float(np.median(adj))
rng = np.random.default_rng(0)
nr = min(500000, npoint)
ia = rng.integers(0, npoint, size=nr)
ib = rng.integers(0, npoint, size=nr)
rand_med = float(np.median(np.abs(C64[ia] - C64[ib])))
spatial_ratio = adj_med / rand_med if rand_med > 0 else np.nan
np.savez(GLS_OUT, sm_cov_full=sm_cov_full, sm_cov_nonmaster=sm_cov_direct, Ac=Ac, Pc=Pc, c_weights=c_weights, velocity_weights=velocity_weights, phase_reference_mean=ref_mean_sm, phase_reference_projection=np.asarray(phase_reference_projection, dtype=np.float64), baseline_C_projection_coefficients=np.asarray([CW_C, CW_N, CW_CR, CW_NR], dtype=np.float64))
manifest = {'status': 'PASS_STAMPS_FINAL_PASS_C', 'implementation': 'StaMPS / pySTAMPS-GAMMA ps_calc_scla(0,1) C_ps_uw GLS', 'phase_modified': False, 'points': int(npoint), 'ifgs': PUBLIC_NIFG, 'images': PUBLIC_NIMAGE, 'geometric_master': {'date': GEOMETRIC_MASTER, 'index_0based': master0}, 'covariance_contract': {'current_network_inversion': 'ordinary_unweighted_L2', 'sb_cov': 'identity_up_to_global_scalar', 'sm_cov_formula': 'inv(Gbase.T @ Gbase)', 'Pbase_cov_formula': 'Pbase @ Pbase.T', 'internal_max_abs_diff': smcov_internal_max, 'old_prototype_max_abs_diff': old_cov_max, 'old_prototype_rms_diff': old_cov_rms, 'global_variance_scale_relevant': False}, 'C_design': {'shape': list(Ac.shape), 'rank': int(np.linalg.matrix_rank(Ac)), 'constant_gain': constant_gain, 'time_leak': time_leak}, 'hard_parity': {'sample_points': sample_n, 'C_explicit_vs_fused_max_abs': C_parity_max, 'C_explicit_vs_fused_rms': C_parity_rms, 'tolerance': C_PARITY_TOL}, 'C_statistics': {'p01_p05_p50_p95_p99': [float(x) for x in c_q], 'abs_p50_p95_p99': [float(x) for x in abs_c_q], 'finite_fraction': finite_fraction, 'reference_mean': ref_c_mean, 'reference_median': ref_c_median, 'adjacent_abs_dC_median': adj_med, 'random_abs_dC_median': rand_med, 'adjacent_to_random_ratio': spatial_ratio}, 'performance': {'production_seconds': production_seconds, 'points_per_second': npoint / production_seconds}, 'outputs': {'C_ps_uw': str(C_OUT), 'sm_cov': str(SMCOV_OUT), 'gls_contract': str(GLS_OUT)}, 'next': '5B6 materialize StaMPS SCLA-corrected phase and prepare Stage-8 SCN'}
MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
print('=' * 92)
print('5B5 STAMPS FINAL PASS C_ps_uw')
print('=' * 92)
print('points                         :', f'{npoint:,}')
print('Gbase rank                     :', f'{rank_g}/37')
print('network covariance model       :', 'ordinary L2 -> identity IFG covariance')
print()
print('sm_cov direct/pinv max diff    :', f'{smcov_internal_max:.12e}')
if old_cov_max is not None:
    print('sm_cov vs prototype max diff  :', f'{old_cov_max:.12e}')
    print('sm_cov vs prototype RMS       :', f'{old_cov_rms:.12e}')
print()
print('C design shape / rank          :', f'{Ac.shape} / {np.linalg.matrix_rank(Ac)}')
print('C constant gain                :', f'{constant_gain:.12e}')
print('C time leakage                 :', f'{time_leak:.12e}')
print()
print('C fast parity max |diff|       :', f'{C_parity_max:.12e}')
print('C fast parity RMS              :', f'{C_parity_rms:.12e}')
print()
print('production seconds             :', f'{production_seconds:.6f}')
print('C throughput                   :', f'{npoint / production_seconds:,.0f} points/s')
print()
print('C p01/p05/p50/p95/p99         :', c_q)
print('|C| p50/p95/p99               :', abs_c_q)
print('reference C mean               :', f'{ref_c_mean:.12e}')
print('reference C median             :', f'{ref_c_median:.12e}')
print('adjacent/random C ratio        :', f'{spatial_ratio:.6f}')
print()
print('37-column Bperp persisted      :', False)
print('ph_scla matrix persisted       :', False)
print('production phase modified      :', False)
print('C output                       :', C_OUT)
print('sm_cov                         :', SMCOV_OUT)
print('GLS contract                   :', GLS_OUT)
print('manifest                       :', MANIFEST)
print('=' * 92)
print('5B5 FINAL RESULT: PASS_STAMPS_FINAL_PASS_C')
print('=' * 92)
