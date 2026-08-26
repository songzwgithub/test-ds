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
import re
import time
import numpy as np
from numba import njit, prange
PSDS = PUBLIC_PROJECT
ROOT = PUBLIC_GEOM_COMPAT
OUT = PUBLIC_ATM
GACOS = PUBLIC_GACOS
RSLC_PAR = PUBLIC_RSLC_PAR
OUT.mkdir(parents=True, exist_ok=True)
C0 = 299792458.0

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
        s = line.strip()
        if not s:
            continue
        if s.startswith('#'):
            continue
        p = s.split()
        if len(p) >= 2:
            d[p[0].upper()] = p[1]
    return d

def req(d, key, cast=float):
    if key not in d:
        raise KeyError(f'missing {key}')
    return cast(d[key])
ztd_files = sorted(GACOS.glob('*.ztd'))
dates = [p.stem for p in ztd_files]
if len(ztd_files) != PUBLIC_NDATE or len(set(dates)) != PUBLIC_NDATE:
    raise RuntimeError(f'GACOS date contract failed: expected={PUBLIC_NDATE}, actual={len(dates)}, unique={len(set(dates))}')
r0 = read_rsc(Path(str(ztd_files[0]) + '.rsc'))
width = req(r0, 'WIDTH', int)
length = req(r0, 'FILE_LENGTH', int)
x_first = req(r0, 'X_FIRST')
y_first = req(r0, 'Y_FIRST')
x_step = req(r0, 'X_STEP')
y_step = req(r0, 'Y_STEP')
geom0 = (width, length, x_first, y_first, x_step, y_step)
expected_bytes = width * length * 4
for ztd in ztd_files:
    rsc = Path(str(ztd) + '.rsc')
    if not rsc.is_file():
        raise RuntimeError(f'missing RSC: {rsc}')
    rr = read_rsc(rsc)
    geom = (req(rr, 'WIDTH', int), req(rr, 'FILE_LENGTH', int), req(rr, 'X_FIRST'), req(rr, 'Y_FIRST'), req(rr, 'X_STEP'), req(rr, 'Y_STEP'))
    if geom != geom0:
        raise RuntimeError(f'RSC geometry mismatch: {ztd.name}')
    if ztd.stat().st_size != expected_bytes:
        raise RuntimeError(f'ZTD byte-size mismatch: {ztd.name}')
lon = np.load(ROOT / 'longitude_deg.npy', mmap_mode='r')
lat = np.load(ROOT / 'latitude_deg.npy', mmap_mode='r')
inc = np.load(ROOT / 'incidence_gamma_compatible_fast_rad.npy', mmap_mode='r')
n = lon.size
if not lat.size == inc.size == n:
    raise RuntimeError('point-array size mismatch')
ref_idx = np.load(PUBLIC_REF_FILE, allow_pickle=False)
if ref_idx.ndim != 1:
    raise RuntimeError('reference_strict_indices.npy must be 1-D')
ref_idx = np.asarray(ref_idx, dtype=np.int32)
if ref_idx.size != PUBLIC_NREF:
    raise RuntimeError(f'reference point count mismatch: {ref_idx.size} != {PUBLIC_NREF}')
if np.any(ref_idx < 0) or np.any(ref_idx >= n):
    raise RuntimeError('reference indices outside point domain')
valid_inc = np.isfinite(inc) & (inc > 0) & (inc < np.pi / 2)
if not np.all(valid_inc):
    raise RuntimeError('invalid incidence')
sec_inc = (1.0 / np.cos(np.asarray(inc, dtype=np.float64))).astype(np.float32)

@njit(parallel=True, fastmath=False, cache=True)
def build_interp(lon, lat, x0, y0, dx, dy, width, length):
    n = lon.size
    base = np.empty(n, np.int32)
    fx = np.empty(n, np.float32)
    fy = np.empty(n, np.float32)
    bad = np.zeros(n, np.uint8)
    for k in prange(n):
        u = (lon[k] - x0) / dx
        v = (lat[k] - y0) / dy
        if not np.isfinite(u) or not np.isfinite(v) or u < -1e-09 or (v < -1e-09) or (u > width - 1 + 1e-09) or (v > length - 1 + 1e-09):
            bad[k] = 1
        u = min(max(u, 0.0), width - 1.0)
        v = min(max(v, 0.0), length - 1.0)
        j = int(math.floor(u))
        i = int(math.floor(v))
        if j >= width - 1:
            j = width - 2
        if i >= length - 1:
            i = length - 2
        base[k] = i * width + j
        fx[k] = u - j
        fy[k] = v - i
    return (base, fx, fy, bad)

@njit(parallel=True, fastmath=False, cache=True)
def sample_los(z, base, fx, fy, sec_inc, width, out):
    for k in prange(base.size):
        b = base[k]
        x = fx[k]
        y = fy[k]
        z00 = z[b]
        z01 = z[b + 1]
        z10 = z[b + width]
        z11 = z[b + width + 1]
        a = z00 + x * (z01 - z00)
        c = z10 + x * (z11 - z10)
        ztd = a + y * (c - a)
        out[k] = ztd * sec_inc[k]
_ = build_interp(np.asarray(lon[:1024], np.float64), np.asarray(lat[:1024], np.float64), x_first, y_first, x_step, y_step, width, length)
dummy = np.zeros(width * length, np.float32)
tmp = np.empty(1024, np.float32)
sample_los(dummy, np.zeros(1024, np.int32), np.zeros(1024, np.float32), np.zeros(1024, np.float32), np.ones(1024, np.float32), width, tmp)
tg = time.perf_counter()
base, fx, fy, bad = build_interp(lon, lat, x_first, y_first, x_step, y_step, width, length)
geometry_seconds = time.perf_counter() - tg
if np.any(bad):
    raise RuntimeError(f'GACOS coverage failed: bad={int(bad.sum())}')
cache_dir = OUT / 'project_local_interp'
cache_dir.mkdir(exist_ok=True)
np.save(cache_dir / 'base.npy', base)
np.save(cache_dir / 'fx.npy', fx)
np.save(cache_dir / 'fy.npy', fy)
np.save(cache_dir / 'sec_inc.npy', sec_inc)
np.save(cache_dir / 'ref_idx.npy', ref_idx)
print('=' * 88)
print('PORTABLE GACOS MAPPING: PASS')
print('points :', n)
print('dates  :', PUBLIC_NDATE)
print('refs   :', ref_idx.size)
print('cache  :', cache_dir)
print('=' * 88)
