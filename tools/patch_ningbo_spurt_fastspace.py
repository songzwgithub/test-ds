#!/usr/bin/env python3
"""Safe in-place speed patch for NEW Ningbo EMCF island processes only.

1. s_worker_count uses configurable default 3 rather than 1.
2. Linux-fork Spurt parallel spatial MCF shares read-only graph, cost and
   gradient cube per island, instead of serializing graph/cost on EACH of 219 IFGs.

No optimizer, graph, cost, phase, or integer-flow objective changes.
Works only with a recognized Spurt 0.1.x _solver.py source layout. Refuses
unrecognized layouts without editing ANY file. Existing processes unchanged.

Usage:
  python tools/patch_ningbo_spurt_fastspace.py --spatial-workers 3 --dry-run
  python tools/patch_ningbo_spurt_fastspace.py --spatial-workers 3 --apply
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import sys

BASE = Path('/mnt/ningbo_process/ds/test-ds-main/pypsds/unwrap/native_emcf_backend.py')
BACKEND_ENV = re.compile(r"s_worker_count\s*=\s*int\(__import__\('os'\)\.environ\.get\('NINGBO_EMCF_SPATIAL_WORKERS',\s*'\d+'\)\)")
BACKEND_ONE = re.compile(r'(?<!\w)s_worker_count\s*=\s*1\b')
PATCH_MARKER = '_SPURT_SPATIAL_SHARED_NINGBO_V1'

OLD_BLOCK = '''        mp_context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=nworkers, mp_context=mp_context
        ) as executor:
            futures = {
                executor.submit(
                    _unwrap_ifg_in_space,
                    grad_space[ii, :],
                    self._solver_space,
                    cost,
                    ii,
                ): ii
                for ii in range(self.nifgs)
            }
            for fut in as_completed(futures):
                ii, data = fut.result()
                futures.pop(fut)
                uw_data[ii, :] = data
        return uw_data'''
NEW_BLOCK = '''        # Linux fork COW: pass ONLY a small IFG index to workers.  The graph,
        # costs, and temporal gradient matrix are inherited once, read-only.
        global _SPURT_SPATIAL_SHARED_NINGBO_V1
        _SPURT_SPATIAL_SHARED_NINGBO_V1 = (grad_space, self._solver_space, cost)
        try:
            mp_context = mp.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=nworkers, mp_context=mp_context
            ) as executor:
                futures = {
                    executor.submit(_ningbo_fastspace_run_ifg_v1, ii): ii
                    for ii in range(self.nifgs)
                }
                for fut in as_completed(futures):
                    ii, data = fut.result()
                    futures.pop(fut)
                    uw_data[ii, :] = data
        finally:
            _SPURT_SPATIAL_SHARED_NINGBO_V1 = None
        return uw_data'''
HELPER = '''
# Ningbo native EMCF speed patch: private read-only state inherited by fork.
_SPURT_SPATIAL_SHARED_NINGBO_V1 = None


def _ningbo_fastspace_run_ifg_v1(ii):
    state = _SPURT_SPATIAL_SHARED_NINGBO_V1
    if state is None:
        raise RuntimeError("Ningbo fastspace worker was not fork-initialized")
    gradients, solver_space, costs = state
    return _unwrap_ifg_in_space(gradients[ii, :], solver_space, costs, ii)

'''


def patch_backend(src: str, nworkers: int):
    expr = "s_worker_count=int(__import__('os').environ.get('NINGBO_EMCF_SPATIAL_WORKERS', '%d'))" % nworkers
    e = list(BACKEND_ENV.finditer(src))
    o = list(BACKEND_ONE.finditer(src))
    if len(e) == 1 and not o:
        return src[:e[0].start()] + expr + src[e[0].end():], 'backend: adjustable spatial workers default %d' % nworkers
    if len(o) == 1 and not e:
        return src[:o[0].start()] + expr + src[o[0].end():], 'backend: spatial 1 -> %d' % nworkers
    raise RuntimeError('Backend s_worker_count pattern not recognized; STOP without changes')


def patch_spurt(src: str):
    if PATCH_MARKER in src:
        if src.count(PATCH_MARKER) < 3:
            raise RuntimeError('Spurt partly patched; STOP and inspect')
        return src, 'spurt: existing Ningbo shared-fork patch detected'
    if src.count(OLD_BLOCK) != 1:
        raise RuntimeError('Spurt source differs from tested 0.1.x layout; STOP without changes. ' +
                           'Do not blindly patch a different version.')
    target = '\nclass EMCFSolver:'
    if src.count(target) != 1 or src.count('def _unwrap_ifg_in_space(') != 1:
        raise RuntimeError('Unexpected Spurt class/function layout')
    return src.replace(target, HELPER+target, 1).replace(OLD_BLOCK, NEW_BLOCK, 1), 'spurt: shared-fork spatial graph/cost/gradients'


def digest(src): return hashlib.sha256(src.encode('utf-8')).hexdigest()[:12]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', type=Path, default=BASE)
    p.add_argument('--spurt-solver', type=Path, help='Path to installed spurt/workflows/emcf/_solver.py')
    p.add_argument('--spatial-workers', type=int, default=3)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--apply', action='store_true')
    a = p.parse_args()
    if not 1 <= a.spatial_workers <= 8: p.error('spatial-workers must be 1..8')
    backend = a.backend.resolve()
    if a.spurt_solver:
        spurt = a.spurt_solver.resolve()
    else:
        spec = importlib.util.find_spec('spurt')
        if spec is None or spec.origin is None:
            p.error('spurt not installed in active Python environment')
        spurt = Path(spec.origin).parent/'workflows'/'emcf'/'_solver.py'
    if not backend.is_file() or not spurt.is_file():
        p.error(f'Missing backend={backend} or spurt={spurt}')
    originals = {backend: backend.read_text(encoding='utf-8'),
                 spurt: spurt.read_text(encoding='utf-8')}
    # PREPARE BOTH before modifying EITHER.
    patched_backend, msg_b = patch_backend(originals[backend], a.spatial_workers)
    patched_spurt, msg_s = patch_spurt(originals[spurt])
    altered = {backend: patched_backend, spurt: patched_spurt}
    for path, data in altered.items():
        ast.parse(data, filename=str(path))
    print(msg_b)
    print(msg_s)
    for path in altered:
        print('PATH', path)
        print('SHA256 before->after', digest(originals[path]), digest(altered[path]))
    if a.dry_run:
        print('DRY_RUN_OK_NO_CHANGES')
        return 0
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    changed = [path for path in altered if originals[path] != altered[path]]
    backups = {}
    done = []
    try:
        for path in changed:
            backup = path.with_name(path.name+f'.before_ningbo_fastspace_{stamp}')
            shutil.copy2(path, backup)
            backups[path] = backup
        for path in changed:
            temp = path.with_name(path.name + '.ningbo_fastspace_tmp')
            temp.write_text(altered[path], encoding='utf-8')
            temp.chmod(path.stat().st_mode)
            os.replace(temp, path)
            done.append(path)
    except BaseException:
        for path in reversed(done):
            shutil.copy2(backups[path], path)
        raise
    print('PATCH_APPLIED', len(changed), 'files')
    for path, backup in backups.items(): print('BACKUP', path, '=>', backup)
    print('CURRENT WORKERS UNCHANGED. Only newly launched island subprocesses import this code.')
    print('CAUTION: keep jobs-spatial=8; max 8 x %d spatial MCF workers. Watch memory peak.' % a.spatial_workers)
    return 0

if __name__ == '__main__':
    try: sys.exit(main())
    except (RuntimeError, OSError, SyntaxError) as exc:
        print('STOP: ', str(exc), file=sys.stderr)
        sys.exit(2)
