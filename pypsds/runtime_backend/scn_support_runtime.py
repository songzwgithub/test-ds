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
import re
import time
import numpy as np
from scipy.spatial import cKDTree
ROOT = PUBLIC_DATA_ROOT
PSDS = PUBLIC_PROJECT
PROC = PUBLIC_PROC
LON = PROC / 'point_geometry' / 'longitude_deg.npy'
LAT = PROC / 'point_geometry' / 'latitude_deg.npy'
PLIST = PROC / 'point_geometry' / 'strict_points.plist'
RSLC_PAR = PUBLIC_RSLC_PAR
OUTDIR = PUBLIC_SCN_SUPPORT
OUTDIR.mkdir(parents=True, exist_ok=True)
XY_OUT = OUTDIR / 'stamps_xy_exact_float32_m.npy'
SORT_OUT = OUTDIR / 'stamps_sort_index.npy'
COUNT_OUT = OUTDIR / 'neighbor_count.npy'
MANIFEST = OUTDIR / 'neighbor_census_manifest.json'
RADIUS = PUBLIC_SCN_RADIUS
SIGMA = PUBLIC_SCN_WAVELENGTH
QUERY_CHUNK = 65536
A = 6378137.0
E = 0.0820944379497
NUM_RE = re.compile('[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[Ee][-+]?\\d+)?')

def read_par(path):
    out = {}
    for line in path.read_text(errors='ignore').splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        out[k.strip().lower()] = v.strip()
    return out

def par_scalar(pars, names):
    for name in names:
        x = pars.get(name.lower())
        if x is None:
            continue
        m = NUM_RE.search(x)
        if m:
            return float(m.group(0))
    raise KeyError(' / '.join(names))

def meridian_arc(lat):
    e2 = E * E
    e4 = e2 * e2
    e6 = e4 * e2
    return A * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * lat) + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * lat) - 35 * e6 / 3072 * np.sin(6 * lat))

def llh2local_exact_m(lon_deg, lat_deg, lon0_deg, lat0_deg):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon0 = np.deg2rad(float(lon0_deg))
    lat0 = np.deg2rad(float(lat0_deg))
    M = meridian_arc(lat)
    M0 = float(meridian_arc(np.asarray(lat0, dtype=np.float64)))
    N = A / np.sqrt(1.0 - E * E * np.sin(lat) ** 2)
    dlambda = lon - lon0
    xy = np.empty((lon.size, 2), dtype=np.float64)
    nz = lat != 0.0
    Ee = dlambda[nz] * np.sin(lat[nz])
    xy[nz, 0] = N[nz] / np.tan(lat[nz]) * np.sin(Ee)
    xy[nz, 1] = M[nz] - M0 + N[nz] / np.tan(lat[nz]) * (1.0 - np.cos(Ee))
    if np.any(~nz):
        xy[~nz, 0] = A * dlambda[~nz]
        xy[~nz, 1] = -M0
    return xy
for p in (LON, LAT, PLIST, RSLC_PAR):
    if not p.is_file():
        raise FileNotFoundError(p)
lon = np.load(LON, mmap_mode='r').astype(np.float64)
lat = np.load(LAT, mmap_mode='r').astype(np.float64)
plist = np.fromfile(PLIST, dtype='>i4').reshape(-1, 2)
n = lon.size
if lat.shape != lon.shape or plist.shape != (n, 2):
    raise RuntimeError('point geometry contract failed')
pars = read_par(RSLC_PAR)
heading = par_scalar(pars, ('heading',))
lon0 = float((lon.max() + lon.min()) / 2.0)
lat0 = float((lat.max() + lat.min()) / 2.0)
t_xy = time.perf_counter()
xy_raw = llh2local_exact_m(lon, lat, lon0, lat0)
theta = np.deg2rad(180.0 - heading)
if theta > np.pi:
    theta -= 2.0 * np.pi
rotm = np.asarray([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], dtype=np.float64)
xynew = (rotm @ xy_raw.T).T
raw_span = np.ptp(xy_raw, axis=0)
rot_span = np.ptp(xynew, axis=0)
rotation_accepted = bool(rot_span[0] < raw_span[0] and rot_span[1] < raw_span[1])
xy_selected = xynew if rotation_accepted else xy_raw
xy32 = xy_selected.astype(np.float32)
xy32 = (np.round(xy32 * np.float32(1000.0)) / np.float32(1000.0)).astype(np.float32)
xy_seconds = time.perf_counter() - t_xy
if not np.all(np.isfinite(xy32)):
    raise RuntimeError('non-finite StaMPS XY')
np.save(XY_OUT, xy32)
sort_ix = np.lexsort((xy32[:, 0], xy32[:, 1])).astype(np.int32)
np.save(SORT_OUT, sort_ix)
first_ps_current_index = int(sort_ix[0])
coords = xy32.astype(np.float64)
t_tree = time.perf_counter()
tree = cKDTree(coords, compact_nodes=True, balanced_tree=True)
tree_seconds = time.perf_counter() - t_tree
counts = np.lib.format.open_memmap(COUNT_OUT, mode='w+', dtype=np.int32, shape=(n,))
t_query = time.perf_counter()
for start in range(0, n, QUERY_CHUNK):
    stop = min(start + QUERY_CHUNK, n)
    c = tree.query_ball_point(coords[start:stop], r=RADIUS, workers=-1, return_length=True)
    c = np.asarray(c, dtype=np.int64)
    if c.size != stop - start:
        raise RuntimeError('KDTree count size mismatch')
    if np.any(c <= 0):
        raise RuntimeError('point without self neighbour')
    counts[start:stop] = c.astype(np.int32)
    print(f'[NEIGHBOUR COUNT] {stop:,}/{n:,} ({100 * stop / n:.1f}%)', flush=True)
counts.flush()
query_seconds = time.perf_counter() - t_query
count64 = np.asarray(counts, dtype=np.int64)
count_q = np.percentile(count64, [1, 5, 50, 90, 95, 99, 99.9])
mean_count = float(np.mean(count64))
max_count = int(count64.max())
directed_interactions = int(np.sum(count64, dtype=np.int64))
undirected_nonself = (directed_interactions - n) // 2
undirected_with_self = undirected_nonself + n
xmin = float(coords[:, 0].min())
ymin = float(coords[:, 1].min())
bx = np.floor((coords[:, 0] - xmin) / RADIUS).astype(np.int32)
by = np.floor((coords[:, 1] - ymin) / RADIUS).astype(np.int32)
nx = int(bx.max()) + 1
ny = int(by.max()) + 1
occupancy = np.zeros((ny, nx), dtype=np.int64)
np.add.at(occupancy, (by, bx), 1)
pad = np.pad(occupancy, 1, mode='constant')
neighbour_cell_population = np.zeros_like(occupancy, dtype=np.int64)
for dy in (0, 1, 2):
    for dx in (0, 1, 2):
        neighbour_cell_population += pad[dy:dy + ny, dx:dx + nx]
cell_candidate_directed = int(np.sum(occupancy * neighbour_cell_population, dtype=np.int64))
candidate_to_true_ratio = float(cell_candidate_directed / directed_interactions)
occ_nonzero = occupancy[occupancy > 0]
occ_q = np.percentile(occ_nonzero, [50, 90, 95, 99])
csr_float32_gib = float(directed_interactions * 8 / 1024 ** 3)
csr_float64_gib = float(directed_interactions * 12 / 1024 ** 3)
chunk_4096_mean_nnz = float(mean_count * 4096.0)
chunk_8192_mean_nnz = float(mean_count * 8192.0)
naive_point_epoch_interactions = directed_interactions * PUBLIC_NDATE
if directed_interactions <= 500000000:
    recommendation = 'CKDTREE_STREAMING_EXACT'
elif candidate_to_true_ratio <= 1.8:
    recommendation = 'NUMBA_CELL_LIST_EXACT_FIRST'
else:
    recommendation = 'CELL_LIST_VS_CKDTREE_EXACT'
manifest = {'status': 'PASS_EXACT_STAMPS_XY_NEIGHBOUR_CENSUS', 'points': int(n), 'coordinate_contract': {'origin_lonlat_deg': [lon0, lat0], 'heading_deg': heading, 'theta_deg': float(np.rad2deg(theta)), 'rotation_accepted': rotation_accepted, 'raw_span_xy_m': [float(x) for x in raw_span], 'rotated_span_xy_m': [float(x) for x in rot_span], 'dtype': 'float32', 'rounding': '1 mm', 'official_first_ps_current_index': first_ps_current_index, 'xy_file': str(XY_OUT), 'sort_index_file': str(SORT_OUT)}, 'exact_neighbour_census': {'radius_m': RADIUS, 'sigma_m': SIGMA, 'mean': mean_count, 'max': max_count, 'p01_p05_p50_p90_p95_p99_p999': [float(x) for x in count_q], 'directed_interactions': directed_interactions, 'undirected_nonself_pairs': int(undirected_nonself), 'undirected_pairs_with_self': int(undirected_with_self), 'count_file': str(COUNT_OUT)}, 'cell_list_upper_bound': {'cell_size_m': RADIUS, 'grid_nx': nx, 'grid_ny': ny, 'occupied_cells': int(occ_nonzero.size), 'occupancy_p50_p90_p95_p99': [float(x) for x in occ_q], 'candidate_directed_interactions': cell_candidate_directed, 'candidate_to_true_ratio': candidate_to_true_ratio}, 'memory_estimate': {'global_CSR_float32_weight_GiB': csr_float32_gib, 'global_CSR_float64_weight_GiB': csr_float64_gib, 'mean_nnz_per_4096_point_chunk': chunk_4096_mean_nnz, 'mean_nnz_per_8192_point_chunk': chunk_8192_mean_nnz}, 'work_estimate': {'naive_38_epoch_weighted_interactions': int(naive_point_epoch_interactions)}, 'timing': {'xy_seconds': xy_seconds, 'tree_build_seconds': tree_seconds, 'count_query_seconds': query_seconds}, 'next_engine_recommendation': recommendation, 'phase_modified': False}
MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
print('=' * 96)
print('EXACT STAMPS XY + NEIGHBOUR CENSUS')
print('=' * 96)
print('points                         :', f'{n:,}')
print('heading                        :', f'{heading:.6f} deg')
print('StaMPS rotation theta          :', f'{np.rad2deg(theta):.6f} deg')
print('rotation accepted              :', rotation_accepted)
print('raw xy span                    :', raw_span)
print('rotated xy span                :', rot_span)
print('official first PS current idx  :', first_ps_current_index)
print()
print('KDTree build seconds           :', f'{tree_seconds:.6f}')
print('neighbour census seconds       :', f'{query_seconds:.6f}')
print()
print('neighbours mean                :', f'{mean_count:,.2f}')
print('neighbours max                 :', f'{max_count:,}')
print('neigh p01/05/50/90/95/99/999 :', count_q)
print()
print('directed interactions          :', f'{directed_interactions:,}')
print('undirected non-self pairs      :', f'{undirected_nonself:,}')
print('naive interactions ×N epochs :', f'{naive_point_epoch_interactions:,}')
print()
print('spatial cell grid                 :', f'{ny} x {nx}')
print('occupied cells                 :', f'{occ_nonzero.size:,}')
print('cell occupancy p50/90/95/99    :', occ_q)
print('cell candidate interactions    :', f'{cell_candidate_directed:,}')
print('candidate / true ratio         :', f'{candidate_to_true_ratio:.4f}')
print()
print('global CSR f32 estimate        :', f'{csr_float32_gib:.2f} GiB')
print('global CSR f64 estimate        :', f'{csr_float64_gib:.2f} GiB')
print('mean nnz / 4096-point chunk    :', f'{chunk_4096_mean_nnz:,.0f}')
print()
print('recommended next benchmark     :', recommendation)
print('exact XY                       :', XY_OUT)
print('neighbor count                 :', COUNT_OUT)
print('manifest                       :', MANIFEST)
print('=' * 96)
print('FINAL RESULT: PASS_EXACT_STAMPS_XY_NEIGHBOUR_CENSUS')
print('=' * 96)
print('NO PHASE MODIFIED')
