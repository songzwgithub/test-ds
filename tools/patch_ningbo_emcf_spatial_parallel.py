#!/usr/bin/env python3
"""Enable Spurt IFG-parallel spatial MCF in Ningbo's existing backend.

No changes to phase data, graph topology, integer estimation, or QC.
Backs up original source and refuses unexpected source shape.
By default each NEW process uses 4 spatial workers. Override with
NINGBO_EMCF_SPATIAL_WORKERS=<positive integer> in the process environment.
Running Python processes retain their already-imported code.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys

DEFAULT = Path('/mnt/ningbo_process/ds/test-ds-main/pypsds/unwrap/native_emcf_backend.py')
SEARCH = re.compile(r'(?<![A-Za-z0-9_])s_worker_count\s*=\s*1\b')
ENV_EXPR = "s_worker_count=int(__import__('os').environ.get('NINGBO_EMCF_SPATIAL_WORKERS', '4'))"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',type=Path,default=DEFAULT)
    p.add_argument('--dry-run',action='store_true')
    a = p.parse_args(argv)
    path=a.backend.resolve()
    if not path.is_file():p.error(f'Backend source file does not exist: {path}')
    src=path.read_text(encoding='utf-8')
    already=src.count(ENV_EXPR)
    if already==1:
        print('ALREADY_PATCHED',path)
        return 0
    occurrences=list(SEARCH.finditer(src))
    if len(occurrences)!=1:
        print(f'STOP: expected exactly one literal s_worker_count=1, found {len(occurrences)}.',file=sys.stderr)
        print('Do not patch blindly. Examine local native_emcf_backend.py for actual SolverSettings call.',file=sys.stderr)
        return 2
    modified=src[:occurrences[0].start()]+ENV_EXPR+src[occurrences[0].end():]
    ast.parse(modified,filename=str(path))
    print('SOURCE',path)
    print('OLD   ',occurrences[0].group(0))
    print('NEW   ',ENV_EXPR)
    if a.dry_run:
        print('DRY_RUN_NO_CHANGES')
        return 0
    backup=path.with_name(f'{path.name}.before_s_worker_patch_{datetime.now():%Y%m%d_%H%M%S}')
    shutil.copy2(path,backup)
    temp=path.with_name(path.name+'.spatial_parallel_tmp')
    try:
        temp.write_text(modified,encoding='utf-8')
        temp.chmod(path.stat().st_mode)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    print('PATCHED_AND_SYNTAX_OK')
    print('BACKUP',backup)
    print('NOTE: already running island computations are not affected; new worker subprocesses use 4 spatial MCF processes unless overridden.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
