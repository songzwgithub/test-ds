from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import subprocess
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
PUBLIC_PLIST = PUBLIC_PROC / 'point_geometry' / 'strict_points.plist'
PUBLIC_REF_FILE = PUBLIC_PROC / 'referenced_timeseries' / 'reference_strict_indices.npy'
PUBLIC_DISABLED_ORACLE = PUBLIC_PROC / '.disabled_scla_prototype_oracle'
PUBLIC_REGENERATED_BASES = PUBLIC_PROC / 'scla' / 'generated_missing'
ROOT = PUBLIC_DATA_ROOT
PSDS = PUBLIC_PROJECT
PROC = PUBLIC_PROC
CONTRACT = PUBLIC_BASELINE_CONTRACT
RSLC = ROOT / 'RSLC'
OLD_GENERATED = PUBLIC_REGENERATED_BASES
OUT = PROC / 'scla'
GEN = OUT / 'generated_missing'
LOG = OUT / 'base_orbit_logs'
CATALOG = OUT / 'baseline_source_catalog.json'
REPORT = OUT / 'p15_5b3b_regeneration_report.json'
BASE_ORBIT_CANDIDATES = []
q = shutil.which('base_orbit')
if q:
    BASE_ORBIT_CANDIDATES.append(Path(q))
BASE_ORBIT = next((p for p in BASE_ORBIT_CANDIDATES if p.is_file()), None)
if BASE_ORBIT is None:
    raise FileNotFoundError('GAMMA base_orbit not found')
OUT.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
NUM_RE = re.compile('[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[Ee][-+]?\\d+)?')
PAIR_RE = re.compile('^(20\\d{6})_(20\\d{6})$')

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def parse_vector(text, labels):
    for line in text.splitlines():
        if not any((label in line for label in labels)):
            continue
        rhs = line.split(':', 1)[1] if ':' in line else line
        x = NUM_RE.findall(rhs)
        if len(x) >= 3:
            return np.asarray([float(v) for v in x[:3]], dtype=np.float64)
    return None

def parse_base(path):
    text = path.read_text(errors='ignore')
    B = parse_vector(text, ('initial_baseline(TCN)', 'initial_baseline'))
    Br = parse_vector(text, ('initial_baseline_rate', 'baseline_rate(TCN)'))
    if B is None or Br is None or (not np.all(np.isfinite(B))) or (not np.all(np.isfinite(Br))):
        raise RuntimeError(f'invalid baseline model: {path}')
    return (B, Br)

def source_path_from_entry(entry):
    if isinstance(entry, str):
        return Path(entry)
    if isinstance(entry, dict):
        for key in ('path', 'base_file', 'file'):
            value = entry.get(key)
            if value:
                return Path(value)
    return None
if not CONTRACT.is_file():
    raise FileNotFoundError(CONTRACT)
contract = json.loads(CONTRACT.read_text())
missing = list(contract.get('missing', []))
original_sources = contract.get('sources', {})
if len(missing) != PUBLIC_MISSING_COUNT:
    raise RuntimeError(f'expected current 16 missing pairs; got {len(missing)}')
LOGDIR = PUBLIC_NETWORK_LOG_DIR
lp = re.compile('pair(\\d+)_(20\\d{6})_(20\\d{6})_single_ifg\\.log$')
network = []
for p in LOGDIR.glob('pair*_single_ifg.log'):
    m = lp.match(p.name)
    if not m:
        continue
    network.append({'edge': int(m.group(1)), 'date_i': m.group(2), 'date_j': m.group(3)})
network.sort(key=lambda x: x['edge'])
if len(network) != PUBLIC_NIFG or [x['edge'] for x in network] != list(range(1, PUBLIC_NIFG + 1)):
    raise RuntimeError('production 108-IFG order contract failed')
missing_set = set(missing)
network_missing = {f"{x['date_i']}_{x['date_j']}" for x in network if f"{x['date_i']}_{x['date_j']}" in missing_set}
if network_missing != missing_set:
    raise RuntimeError('missing-pair/network mismatch')
needed_dates = sorted({date for pair in missing for date in pair.split('_')})
for date in needed_dates:
    p = RSLC / f'{date}.rslc.par'
    if not p.is_file():
        raise FileNotFoundError(p)
print('=' * 92)
print('P15-5B3B GAMMA BASE_ORBIT REGENERATION')
print('=' * 92)
print('base_orbit              :', BASE_ORBIT)
print('missing pairs           :', len(missing))
print('unique acquisitions     :', len(needed_dates))
print('workers                 :', min(8, len(missing)))
print()

def generate_one(pair):
    m = PAIR_RE.match(pair)
    if not m:
        raise RuntimeError(f'bad pair: {pair}')
    d1 = m.group(1)
    d2 = m.group(2)
    par1 = RSLC / f'{d1}.rslc.par'
    par2 = RSLC / f'{d2}.rslc.par'
    final = GEN / f'{pair}.base'
    tmp = GEN / f'.{pair}.base.tmp'
    log = LOG / f'{pair}.log'
    if tmp.exists():
        tmp.unlink()
    t0 = time.perf_counter()
    cp = subprocess.run([str(BASE_ORBIT), str(par1), str(par2), str(tmp)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.perf_counter() - t0
    log.write_text(cp.stdout, encoding='utf-8')
    if cp.returncode != 0:
        raise RuntimeError(f'base_orbit failed {pair}: rc={cp.returncode}\n{cp.stdout}')
    if not tmp.is_file():
        raise RuntimeError(f'base_orbit returned success but output missing: {pair}')
    B, Br = parse_base(tmp)
    os.replace(tmp, final)
    return {'pair': pair, 'date_i': d1, 'date_j': d2, 'par_i': str(par1), 'par_j': str(par2), 'base_file': str(final), 'log': str(log), 'elapsed_s': elapsed, 'baseline_tcn': B.tolist(), 'baseline_rate_tcn': Br.tolist(), 'sha256': sha256(final)}
t_all = time.perf_counter()
generated = {}
workers = min(8, len(missing))
with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='base-orbit') as executor:
    jobs = {executor.submit(generate_one, pair): pair for pair in missing}
    for fut in as_completed(jobs):
        pair = jobs[fut]
        result = fut.result()
        generated[pair] = result
        print(f"[PASS] {pair}  {result['elapsed_s']:.4f} s")
generation_seconds = time.perf_counter() - t_all
if set(generated) != missing_set:
    raise RuntimeError('not all missing baselines regenerated')
OLD_B_TOL_M = 0.001
OLD_RATE_TOL_M_S = 1e-05
old_parity = {}
max_dB = 0.0
max_dBr = 0.0
for pair in missing:
    new_path = Path(generated[pair]['base_file'])
    old_path = OLD_GENERATED / f'{pair}.base'
    if not old_path.is_file():
        raise FileNotFoundError(old_path)
    Bn, Brn = parse_base(new_path)
    Bo, Bro = parse_base(old_path)
    dB = float(np.max(np.abs(Bn - Bo)))
    dBr = float(np.max(np.abs(Brn - Bro)))
    max_dB = max(max_dB, dB)
    max_dBr = max(max_dBr, dBr)
    old_parity[pair] = {'old_file': str(old_path), 'new_file': str(new_path), 'max_abs_baseline_tcn_diff_m': dB, 'max_abs_baseline_rate_diff_m_s': dBr, 'old_sha256': sha256(old_path), 'new_sha256': sha256(new_path)}
print()
print('-' * 92)
print('OLD-PROTOTYPE NUMERICAL PARITY')
print('-' * 92)
print('max |delta B(TCN)| m     :', f'{max_dB:.12e}')
print('max |delta rate| m/s     :', f'{max_dBr:.12e}')
if max_dB > OLD_B_TOL_M:
    raise RuntimeError(f'new base_orbit baseline does not reproduce old prototype: {max_dB} m')
if max_dBr > OLD_RATE_TOL_M_S:
    raise RuntimeError(f'new base_orbit rate does not reproduce old prototype: {max_dBr} m/s')
catalog_rows = []
n_original = 0
n_generated = 0
for item in network:
    edge = item['edge']
    d1 = item['date_i']
    d2 = item['date_j']
    pair = f'{d1}_{d2}'
    if pair in missing_set:
        p = Path(generated[pair]['base_file'])
        source_type = 'regenerated_gamma_base_orbit'
        n_generated += 1
    else:
        entries = original_sources.get(pair)
        if not entries:
            rev = f'{d2}_{d1}'
            entries = original_sources.get(rev)
        if not entries:
            raise RuntimeError(f'missing original source for supposedly valid pair {pair}')
        if not isinstance(entries, list):
            entries = [entries]
        paths = []
        for x in entries:
            pp = source_path_from_entry(x)
            if pp is not None and pp.is_file():
                paths.append(pp.resolve())
        if not paths:
            raise RuntimeError(f'no readable source for {pair}')
        paths = sorted(set(paths), key=str)
        p = paths[0]
        B0, Br0 = parse_base(p)
        for q in paths[1:]:
            Bq, Brq = parse_base(q)
            if not (np.allclose(Bq, B0, rtol=0.0, atol=1e-06) and np.allclose(Brq, Br0, rtol=0.0, atol=1e-08)):
                raise RuntimeError(f'conflicting original .base sources for {pair}')
        source_type = 'original_gamma_base'
        n_original += 1
    B, Br = parse_base(p)
    catalog_rows.append({'edge': edge, 'date_i': d1, 'date_j': d2, 'orientation': 1, 'source_type': source_type, 'base_file': str(p), 'baseline_tcn': B.tolist(), 'baseline_rate_tcn': Br.tolist(), 'sha256': sha256(p)})
if len(catalog_rows) != PUBLIC_NIFG or n_original != PUBLIC_ORIGINAL_COUNT or n_generated != PUBLIC_MISSING_COUNT:
    raise RuntimeError(f'complete catalog contract failed: rows={len(catalog_rows)}, original={n_original}, generated={n_generated}')
for expected_edge, item in enumerate(catalog_rows, start=1):
    if item['edge'] != expected_edge:
        raise RuntimeError('catalog edge ordering failed')
catalog_payload = {'status': 'PASS_COMPLETE_108_GAMMA_BASE_MODELS', 'method': '92 original GAMMA .base + 16 regenerated with GAMMA base_orbit from current RSLC state vectors', 'base_orbit': str(BASE_ORBIT), 'production_ifgs': PUBLIC_NIFG, 'original_base_pairs': n_original, 'regenerated_base_pairs': n_generated, 'old_prototype_role': 'numerical_parity_oracle_only', 'old_parity': {'max_abs_baseline_tcn_diff_m': max_dB, 'max_abs_baseline_rate_diff_m_s': max_dBr, 'hard_tolerance_baseline_m': OLD_B_TOL_M, 'hard_tolerance_rate_m_s': OLD_RATE_TOL_M_S}, 'generation_wall_seconds': generation_seconds, 'pairs': catalog_rows, 'production_policy': {'prototype_base_files_reused': False, 'DIFF_directory_modified': False, 'phase_modified': False, 'next': 'P15-5B4 FAST pointwise SB Bperp -> SM Bperp StaMPS final-pass parity'}}
CATALOG.write_text(json.dumps(catalog_payload, indent=2) + '\n', encoding='utf-8')
REPORT.write_text(json.dumps({'status': 'PASS_P15_5B3B', 'generation_seconds': generation_seconds, 'generated': generated, 'old_parity': old_parity, 'catalog': str(CATALOG)}, indent=2) + '\n', encoding='utf-8')
print()
print('=' * 92)
print('P15-5B3B FINAL')
print('=' * 92)
print('original GAMMA .base     :', n_original)
print('regenerated base_orbit   :', n_generated)
print('total baseline models    :', len(catalog_rows))
print('generation wall seconds  :', f'{generation_seconds:.6f}')
print('old parity max dB m      :', f'{max_dB:.12e}')
print('old parity max dRate m/s :', f'{max_dBr:.12e}')
print('DIFF modified            :', False)
print('phase modified           :', False)
print('catalog                  :', CATALOG)
print('report                   :', REPORT)
print('=' * 92)
print('P15-5B3B FINAL RESULT: PASS_COMPLETE_108_GAMMA_BASE_MODELS')
print('=' * 92)
