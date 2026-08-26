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
from datetime import datetime
import json
import math
import os
import time
import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange, get_num_threads, set_num_threads
ROOT = PUBLIC_DATA_ROOT
PSDS = PUBLIC_PROJECT
PROC = PUBLIC_PROC
PRE = PUBLIC_SCLA / 'acquisition_phase_pre_scn_rad.npy'
GMAN = PUBLIC_ATM / 'atmosphere_correction_manifest.json'
XY = PUBLIC_SCN_SUPPORT / 'stamps_xy_exact_float32_m.npy'
SORT_IX = PUBLIC_SCN_SUPPORT / 'stamps_sort_index.npy'
COUNTS = PUBLIC_SCN_SUPPORT / 'neighbor_count_r400m.npy'
OUTDIR = PUBLIC_SCN
HPT_FINAL = OUTDIR / 'ph_hpt_rad.npy'
SCN_FINAL = OUTDIR / 'ph_scn_slave_rad.npy'
HPT_TMP = OUTDIR / '.ph_hpt_rad.tmp.npy'
SCN_TMP = OUTDIR / '.ph_scn_slave_rad.tmp.npy'
MANIFEST = PUBLIC_SCN / 'scn_manifest.json'
MASTER_DATE = PUBLIC_MASTER_DATE
TIME_WIN_DAYS = PUBLIC_SCN_TIME_WIN
SCN_WAVELENGTH_M = PUBLIC_SCN_WAVELENGTH
RADIUS_M = PUBLIC_SCN_RADIUS
CELL_SIZE_M = PUBLIC_SCN_CELL_SIZE
TEMP_CHUNK = 131072
SPATIAL_CHUNK = 32768
ORACLE_N = 512
EDGE_CHECK_N = 4096
TEMP_REF_TOL_RAD = 1e-12
EDGE_EQUIV_TOL_RAD = 1e-05
SPATIAL_ORACLE_TOL_RAD = 1e-09

def temporal_weight_matrix(day, master0, time_win):
    dt = day[:, None] - day[None, :]
    W = np.exp(-(dt * dt) / (2.0 * time_win * time_win))
    W[:, master0] = 0.0
    den = np.sum(W, axis=1)
    if np.any(den <= 0.0):
        raise RuntimeError('temporal Gaussian denominator <= 0')
    W /= den[:, None]
    return W

def build_cell_index(coords, cell_size):
    xmin = float(coords[:, 0].min())
    ymin = float(coords[:, 1].min())
    cx = np.floor((coords[:, 0] - xmin) / cell_size).astype(np.int32)
    cy = np.floor((coords[:, 1] - ymin) / cell_size).astype(np.int32)
    nx = int(cx.max()) + 1
    ny = int(cy.max()) + 1
    cell_id = cy.astype(np.int64) * nx + cx
    order = np.argsort(cell_id, kind='stable').astype(np.int32)
    ncell = nx * ny
    occupancy = np.bincount(cell_id, minlength=ncell).astype(np.int64)
    starts = np.empty(ncell + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(occupancy, out=starts[1:])
    return (cx, cy, nx, ny, order, starts)

def build_safe_offsets(cell_size, radius):
    kmax = int(math.ceil(radius / cell_size)) + 1
    radius_sq = radius * radius
    offsets = []
    for dy in range(-kmax, kmax + 1):
        min_y = max(abs(dy) - 1, 0) * cell_size
        for dx in range(-kmax, kmax + 1):
            min_x = max(abs(dx) - 1, 0) * cell_size
            if min_x * min_x + min_y * min_y < radius_sq:
                offsets.append((dx, dy))
    return np.asarray(offsets, dtype=np.int32)

@njit(parallel=True, fastmath=False, cache=False)
def spatial_gaussian_exact(coords, values, targets, cx, cy, nx, ny, order, starts, offsets, radius_sq, sigma2x2):
    ntarget = targets.size
    nepoch = values.shape[1]
    output = np.empty((ntarget, nepoch), dtype=np.float64)
    true_count = np.zeros(ntarget, dtype=np.int64)
    candidate_count = np.zeros(ntarget, dtype=np.int64)
    for ii in prange(ntarget):
        p = targets[ii]
        x0 = coords[p, 0]
        y0 = coords[p, 1]
        pcx = cx[p]
        pcy = cy[p]
        den = 0.0
        acc = np.zeros(nepoch, dtype=np.float64)
        ntrue = 0
        ncandidate = 0
        for kk in range(offsets.shape[0]):
            qx = pcx + offsets[kk, 0]
            qy = pcy + offsets[kk, 1]
            if qx < 0 or qx >= nx or qy < 0 or (qy >= ny):
                continue
            cell = qy * nx + qx
            q0 = starts[cell]
            q1 = starts[cell + 1]
            ncandidate += q1 - q0
            for jj in range(q0, q1):
                q = order[jj]
                dx = coords[q, 0] - x0
                dy = coords[q, 1] - y0
                dist_sq = dx * dx + dy * dy
                if dist_sq < radius_sq:
                    w = math.exp(-dist_sq / sigma2x2)
                    den += w
                    ntrue += 1
                    for e in range(nepoch):
                        acc[e] += w * values[q, e]
        if den <= 0.0:
            for e in range(nepoch):
                output[ii, e] = np.nan
        else:
            inv_den = 1.0 / den
            for e in range(nepoch):
                output[ii, e] = acc[e] * inv_den
        true_count[ii] = ntrue
        candidate_count[ii] = ncandidate
    return (output, true_count, candidate_count)
for p in (PRE, GMAN, XY, SORT_IX, COUNTS):
    if not p.is_file():
        raise FileNotFoundError(p)
pre = np.load(PRE, mmap_mode='r')
coords = np.load(XY, mmap_mode='r').astype(np.float64)
sort_ix = np.load(SORT_IX).astype(np.int64)
census_count = np.load(COUNTS, mmap_mode='r')
gman = json.loads(GMAN.read_text())
dates = list(gman['acquisition_dates'])
npoint, nepoch = pre.shape
if nepoch != PUBLIC_NDATE or coords.shape != (npoint, 2) or sort_ix.size != npoint or (len(dates) != PUBLIC_NDATE):
    raise RuntimeError('Stage-8 input contract failed')
master0 = dates.index(MASTER_DATE)
official_first = int(sort_ix[0])
available_threads = int(get_num_threads())
requested_threads = available_threads
threads = min(requested_threads, available_threads)
set_num_threads(threads)
date_objects = [datetime.strptime(d, '%Y%m%d') for d in dates]
day = np.asarray([(d - date_objects[0]).days for d in date_objects], dtype=np.float64)
temporal_W = temporal_weight_matrix(day, master0, TIME_WIN_DAYS)
for p in (HPT_TMP, SCN_TMP):
    if p.exists():
        p.unlink()
first_phase = np.asarray(pre[official_first, :], dtype=np.float64)
h0 = first_phase - first_phase @ temporal_W.T
ph_hpt_out = np.lib.format.open_memmap(HPT_TMP, mode='w+', dtype=np.float32, shape=(npoint, nepoch))
t_temporal = time.perf_counter()
for start in range(0, npoint, TEMP_CHUNK):
    stop = min(start + TEMP_CHUNK, npoint)
    y = np.asarray(pre[start:stop, :], dtype=np.float64)
    h = y - y @ temporal_W.T - h0[None, :]
    ph_hpt_out[start:stop, :] = h.astype(np.float32)
    print(f'[TEMPORAL] {stop:,}/{npoint:,} ({100 * stop / npoint:.1f}%)', flush=True)
ph_hpt_out.flush()
temporal_seconds = time.perf_counter() - t_temporal
del ph_hpt_out
ph_hpt = np.load(HPT_TMP, mmap_mode='r')
first_hpt_max = float(np.max(np.abs(np.asarray(ph_hpt[official_first, :], dtype=np.float64))))
if first_hpt_max > TEMP_REF_TOL_RAD:
    raise RuntimeError(f'temporal first-PS reference failed: {first_hpt_max}')
tree = cKDTree(coords, compact_nodes=True, balanced_tree=True)
rng = np.random.default_rng(20260824)
edge_u = []
edge_v = []
seed_points = rng.choice(npoint, size=min(512, npoint), replace=False)
for p in seed_points:
    qlist = tree.query_ball_point(coords[p], r=RADIUS_M)
    for q in qlist:
        if q == p:
            continue
        edge_u.append(int(p))
        edge_v.append(int(q))
        if len(edge_u) >= EDGE_CHECK_N:
            break
    if len(edge_u) >= EDGE_CHECK_N:
        break
if len(edge_u) < EDGE_CHECK_N:
    raise RuntimeError('insufficient temporal edge QA pairs')
u = np.asarray(edge_u[:EDGE_CHECK_N], dtype=np.int64)
v = np.asarray(edge_v[:EDGE_CHECK_N], dtype=np.int64)
dph = np.asarray(pre[v, :], dtype=np.float64) - np.asarray(pre[u, :], dtype=np.float64)
dph_hpt = dph - dph @ temporal_W.T
node_edge = np.asarray(ph_hpt[v, :], dtype=np.float64) - np.asarray(ph_hpt[u, :], dtype=np.float64)
edge_equiv_max = float(np.max(np.abs(dph_hpt - node_edge)))
if edge_equiv_max > EDGE_EQUIV_TOL_RAD:
    raise RuntimeError(f'temporal edge-domain equivalence failed: {edge_equiv_max}')
cx, cy, nx, ny, order, starts = build_cell_index(coords, CELL_SIZE_M)
offsets = build_safe_offsets(CELL_SIZE_M, RADIUS_M)
if offsets.shape[0] != 25:
    raise RuntimeError(f'200m cell offset contract changed: {offsets.shape[0]}')
compile_targets = np.arange(0, min(4, npoint), dtype=np.int64)
_ = spatial_gaussian_exact(coords, ph_hpt, compile_targets, cx, cy, nx, ny, order, starts, offsets, RADIUS_M ** 2, 2.0 * SCN_WAVELENGTH_M ** 2)
oracle_targets = rng.choice(npoint, size=ORACLE_N, replace=False).astype(np.int64)
oracle_fast, oracle_fast_count, oracle_candidate_count = spatial_gaussian_exact(coords, ph_hpt, oracle_targets, cx, cy, nx, ny, order, starts, offsets, RADIUS_M ** 2, 2.0 * SCN_WAVELENGTH_M ** 2)
neighbor_lists = tree.query_ball_point(coords[oracle_targets], r=RADIUS_M, workers=-1)
oracle_kdtree = np.empty((ORACLE_N, nepoch), dtype=np.float64)
oracle_kdtree_count = np.empty(ORACLE_N, dtype=np.int64)
for ii in range(ORACLE_N):
    p = oracle_targets[ii]
    q = np.asarray(neighbor_lists[ii], dtype=np.int64)
    dxy = coords[q, :] - coords[p, :]
    d2 = np.sum(dxy * dxy, axis=1)
    keep = d2 < RADIUS_M ** 2
    q = q[keep]
    d2 = d2[keep]
    w = np.exp(-d2 / (2.0 * SCN_WAVELENGTH_M ** 2))
    oracle_kdtree[ii, :] = w @ np.asarray(ph_hpt[q, :], dtype=np.float64) / np.sum(w, dtype=np.float64)
    oracle_kdtree_count[ii] = q.size
oracle_count_diff = int(np.max(np.abs(oracle_fast_count - oracle_kdtree_count)))
if oracle_count_diff != 0:
    raise RuntimeError(f'pre-production neighbour parity failed: {oracle_count_diff}')
oracle_diff = oracle_fast - oracle_kdtree
oracle_max = float(np.max(np.abs(oracle_diff)))
oracle_rms = float(np.sqrt(np.mean(oracle_diff * oracle_diff)))
if oracle_max > SPATIAL_ORACLE_TOL_RAD:
    raise RuntimeError(f'pre-production spatial oracle failed: {oracle_max}')
del neighbor_lists
ph_scn_out = np.lib.format.open_memmap(SCN_TMP, mode='w+', dtype=np.float64, shape=(npoint, nepoch))
strict_interactions = 0
candidate_interactions = 0
census_count_diff_max = 0
t_spatial = time.perf_counter()
for start in range(0, npoint, SPATIAL_CHUNK):
    stop = min(start + SPATIAL_CHUNK, npoint)
    targets = np.arange(start, stop, dtype=np.int64)
    smooth, true_count, candidate_count = spatial_gaussian_exact(coords, ph_hpt, targets, cx, cy, nx, ny, order, starts, offsets, RADIUS_M ** 2, 2.0 * SCN_WAVELENGTH_M ** 2)
    if not np.all(np.isfinite(smooth)):
        raise RuntimeError(f'non-finite SCN at {start}:{stop}')
    ph_scn_out[start:stop, :] = smooth
    strict_interactions += int(np.sum(true_count, dtype=np.int64))
    candidate_interactions += int(np.sum(candidate_count, dtype=np.int64))
    census_diff = np.asarray(census_count[start:stop], dtype=np.int64) - true_count
    census_count_diff_max = max(census_count_diff_max, int(np.max(np.abs(census_diff))))
    print(f'[SPATIAL] {stop:,}/{npoint:,} ({100 * stop / npoint:.1f}%) true={np.sum(true_count):,} candidate={np.sum(candidate_count):,}', flush=True)
ph_scn_out.flush()
spatial_seconds = time.perf_counter() - t_spatial
ref_scn = np.asarray(ph_scn_out[official_first, :], dtype=np.float64).copy()
t_reference = time.perf_counter()
for start in range(0, npoint, TEMP_CHUNK):
    stop = min(start + TEMP_CHUNK, npoint)
    ph_scn_out[start:stop, :] -= ref_scn[None, :]
ph_scn_out[:, master0] = 0.0
ph_scn_out.flush()
reference_seconds = time.perf_counter() - t_reference
first_scn_max = float(np.max(np.abs(ph_scn_out[official_first, :])))
master_scn_max = float(np.max(np.abs(ph_scn_out[:, master0])))
if first_scn_max != 0.0:
    raise RuntimeError(f'final first-PS SCN is not exact zero: {first_scn_max}')
if master_scn_max != 0.0:
    raise RuntimeError(f'master SCN not zero: {master_scn_max}')
sample_n = min(100000, npoint)
qa_idx = rng.choice(npoint, size=sample_n, replace=False)
qa_scn = np.asarray(ph_scn_out[qa_idx, :], dtype=np.float64)
abs_scn = np.abs(qa_scn)
scn_abs_q = np.percentile(abs_scn, [50, 95, 99, 99.9])
scn_rms_sample = float(np.sqrt(np.mean(qa_scn * qa_scn)))
finite_fraction_sample = float(np.mean(np.isfinite(qa_scn)))
if finite_fraction_sample != 1.0:
    raise RuntimeError('SCN sample contains non-finite values')
del ph_scn_out
del ph_hpt
os.replace(HPT_TMP, HPT_FINAL)
os.replace(SCN_TMP, SCN_FINAL)
candidate_true_ratio = float(candidate_interactions / strict_interactions)
manifest = {'status': 'PASS_STAMPS_STAGE8_SCN', 'implementation': 'StaMPS ps_scn_filt parity; exact 200m Numba cell-list spatial engine', 'scientific_contract': {'scn_time_win_days': TIME_WIN_DAYS, 'scn_wavelength_m': SCN_WAVELENGTH_M, 'spatial_radius_m': RADIUS_M, 'spatial_radius_test': 'distance_squared < radius_squared', 'spatial_weight': 'exp(-distance_squared/(2*wavelength^2))', 'all_neighbors': True, 'coordinate_source': str(XY), 'official_first_ps_current_index': official_first, 'geometric_master_date': MASTER_DATE, 'geometric_master_index_0based': master0}, 'engine': {'cell_size_m': CELL_SIZE_M, 'offset_cells': int(offsets.shape[0]), 'numba_threads': threads, 'available_numba_threads': available_threads, 'spatial_chunk_points': SPATIAL_CHUNK, 'strict_interactions': strict_interactions, 'candidate_interactions': candidate_interactions, 'candidate_to_true_ratio': candidate_true_ratio, 'census_vs_strict_max_count_diff': census_count_diff_max}, 'hard_parity': {'temporal_first_ps_max_abs_rad': first_hpt_max, 'temporal_reference_tolerance_rad': TEMP_REF_TOL_RAD, 'temporal_edge_pairs': EDGE_CHECK_N, 'temporal_edge_equivalence_max_rad': edge_equiv_max, 'temporal_edge_tolerance_rad': EDGE_EQUIV_TOL_RAD, 'spatial_oracle_points': ORACLE_N, 'spatial_neighbor_count_max_diff': oracle_count_diff, 'spatial_cell_vs_kdtree_max_abs_rad': oracle_max, 'spatial_cell_vs_kdtree_rms_rad': oracle_rms, 'spatial_oracle_tolerance_rad': SPATIAL_ORACLE_TOL_RAD, 'final_first_ps_scn_max_abs_rad': first_scn_max, 'final_master_scn_max_abs_rad': master_scn_max}, 'performance': {'temporal_seconds': temporal_seconds, 'spatial_seconds': spatial_seconds, 'reference_seconds': reference_seconds, 'total_seconds': temporal_seconds + spatial_seconds + reference_seconds, 'true_interactions_per_second': strict_interactions / spatial_seconds, 'candidate_tests_per_second': candidate_interactions / spatial_seconds, 'point_epoch_neighbor_accumulations_per_second': strict_interactions * nepoch / spatial_seconds}, 'qa_sample': {'points': sample_n, 'finite_fraction': finite_fraction_sample, 'scn_rms_rad': scn_rms_sample, 'abs_scn_p50_p95_p99_p999_rad': [float(x) for x in scn_abs_q]}, 'outputs': {'ph_hpt': str(HPT_FINAL), 'ph_scn_slave': str(SCN_FINAL), 'ph_hpt_dtype': 'float32', 'ph_scn_slave_dtype': 'float64'}, 'input_phase_modified': False, 'next': 'P15-6C apply SCN correction and restore frozen temporal/spatial datum'}
MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
print()
print('=' * 96)
print('P15-6B2 FULL STAMPS STAGE-8 SCN')
print('=' * 96)
print('points / epochs                 :', f'{npoint:,} / {nepoch}')
print('official first PS current idx   :', official_first)
print('geometric master                :', f'{MASTER_DATE} (0b={master0})')
print()
print('temporal first PS max |H|       :', f'{first_hpt_max:.12e}')
print('edge-domain equivalence max     :', f'{edge_equiv_max:.12e} rad')
print()
print('oracle neighbour count diff     :', oracle_count_diff)
print('cell vs KDTree max diff         :', f'{oracle_max:.12e} rad')
print('cell vs KDTree RMS              :', f'{oracle_rms:.12e} rad')
print()
print('cell size                       :', f'{CELL_SIZE_M:.1f} m')
print('cell offsets                    :', offsets.shape[0])
print('strict interactions             :', f'{strict_interactions:,}')
print('candidate interactions          :', f'{candidate_interactions:,}')
print('candidate / true                :', f'{candidate_true_ratio:.6f}')
print('census / strict max count diff  :', census_count_diff_max)
print()
print('temporal seconds                :', f'{temporal_seconds:.6f}')
print('spatial seconds                 :', f'{spatial_seconds:.6f}')
print('reference seconds               :', f'{reference_seconds:.6f}')
print('total core seconds              :', f'{temporal_seconds + spatial_seconds + reference_seconds:.6f}')
print('true interactions/s             :', f'{strict_interactions / spatial_seconds:,.0f}')
print()
print('final first PS SCN max          :', f'{first_scn_max:.12e}')
print('final master SCN max            :', f'{master_scn_max:.12e}')
print('sample SCN RMS                  :', f'{scn_rms_sample:.6f} rad')
print('|SCN| p50/p95/p99/p999         :', scn_abs_q)
print()
print('ph_hpt                          :', HPT_FINAL)
print('ph_scn_slave                    :', SCN_FINAL)
print('original pre-SCN modified       :', False)
print('manifest                        :', MANIFEST)
print('=' * 96)
print('P15-6B2 FINAL RESULT: PASS_STAMPS_STAGE8_SCN')
print('=' * 96)
