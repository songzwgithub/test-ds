"""Read-only, same-GAMMA-IFG Whirlwind/SNAPHU benchmark for pyPSDS.

The experiment never changes PointPhaseStack or production unwrapping. Its
coherence input must be the genuine interferometric sample coherence for the
same raster and its --nlooks must be justified from the coherence estimator.
Do not use phase-linking temporal coherence as a substitute.

Both solvers see the SAME complex IFG, coherence and byte mask. This is a
2-D solver benchmark, not a validation of a spatial-temporal PS/DS product.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

TAU = 2.0 * np.pi


def _verify_file_size(path: Path, expected: int):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path}: bytes={actual:,}; expected={expected:,}. "
                         "Check dimensions, byte order and coherence layout.")


def _dtype(endian: str, kind: str):
    if endian not in ('big', 'little'):
        raise ValueError('endian must be big or little')
    if kind not in ('c8', 'f4'):
        raise ValueError('kind must be c8 or f4')
    return np.dtype(('>' if endian == 'big' else '<') + kind)


def load_gamma_crop(*, ifg_path: Path, corr_path: Path, full_height: int,
                    full_width: int, row0: int, col0: int,
                    height: int, width: int, endian: str,
                    corr_layout: str, mask_path: Path | None = None):
    """Read a spatially identical crop of GAMMA complex IFG and coherence."""
    if min(full_height, full_width, height, width) <= 0:
        raise ValueError('Raster dimensions must be positive')
    if row0 < 0 or col0 < 0 or row0 + height > full_height or col0 + width > full_width:
        raise ValueError('Crop is outside full GAMMA raster')
    total = full_height * full_width
    _verify_file_size(ifg_path, total * 8)
    if corr_layout not in ('float32', 'alt-line'):
        raise ValueError('corr_layout must be float32 or alt-line')
    _verify_file_size(corr_path, total * 4 * (2 if corr_layout == 'alt-line' else 1))
    ifg_m = np.memmap(ifg_path, dtype=_dtype(endian, 'c8'), mode='r',
                      shape=(full_height, full_width))
    if corr_layout == 'float32':
        corr_m = np.memmap(corr_path, dtype=_dtype(endian, 'f4'), mode='r',
                           shape=(full_height, full_width))
        corr_view = corr_m[row0:row0+height, col0:col0+width]
    else:
        corr_m = np.memmap(corr_path, dtype=_dtype(endian, 'f4'), mode='r',
                           shape=(full_height, 2, full_width))
        corr_view = corr_m[row0:row0+height, 1, col0:col0+width]
    igram = np.array(ifg_m[row0:row0+height,
                           col0:col0+width], dtype=np.complex64, order='C', copy=True)
    corr = np.array(corr_view, dtype=np.float32, order='C', copy=True)
    valid = (np.isfinite(igram.real) & np.isfinite(igram.imag) &
             (np.abs(igram) > 0) & np.isfinite(corr) &
             (corr > 0) & (corr <= 1))
    # Values > 1 are invalid measurements, not a reason to silently clip.
    if mask_path is not None:
        if Path(mask_path).suffix == '.npy':
            m = np.load(mask_path, allow_pickle=False)
            if m.shape == (full_height, full_width):
                m = m[row0:row0+height, col0:col0+width]
            elif m.shape != (height, width):
                raise ValueError('NPY mask does not match full or cropped shape')
        else:
            _verify_file_size(mask_path, total)
            full_mask = np.memmap(mask_path, dtype=np.uint8, mode='r',
                                  shape=(full_height, full_width))
            m = full_mask[row0:row0+height, col0:col0+width]
        valid &= np.asarray(m, dtype=bool)
    if np.count_nonzero(valid) < 16:
        raise ValueError('Insufficient valid interferogram/coherence pixels')
    igram[~valid] = 0
    corr[~valid] = 0
    return igram, corr, np.ascontiguousarray(valid)


def _wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def _parity_max(unwrapped, igram, mask):
    if not np.all(np.isfinite(np.asarray(unwrapped)[mask])):
        raise ValueError('Nonfinite unwrapped phase in valid pixels')
    return float(np.max(np.abs(_wrap(np.asarray(unwrapped)[mask] - np.angle(igram[mask])))))


def unwrap_whirlwind(igram, coherence, mask, nlooks, *, unwrap_func=None):
    if unwrap_func is None:
        try:
            import whirlwind as ww
        except ImportError as exc:
            raise RuntimeError('Install the official whirlwind-insar package first') from exc
        unwrap_func = ww.unwrap
    # Explicitly suppress all synthetic bridges across invalid regions.
    t0 = time.perf_counter()
    uw, conn = unwrap_func(igram, coherence, nlooks=float(nlooks), mask=mask,
                           bridge=False, connect_gaps=False, interpolate=False)
    seconds = time.perf_counter() - t0
    uw, conn = np.asarray(uw), np.asarray(conn)
    if uw.shape != igram.shape or conn.shape != igram.shape:
        raise RuntimeError('Whirlwind outputs do not match input dimensions')
    if not np.issubdtype(conn.dtype, np.integer):
        raise RuntimeError('Whirlwind component labels are not integral')
    err = _parity_max(uw, igram, mask)
    if err > 1e-3:
        raise RuntimeError(f'Whirlwind wrapped parity mismatch: {err:g} rad')
    return np.asarray(uw, np.float32), conn, {'seconds':seconds, 'wrap_parity_max_rad':err}


def snaphu_config(nlooks: float, wavelength_m: float):
    if not (np.isfinite(nlooks) and nlooks >= 1):
        raise ValueError('effective coherence looks must be >= 1')
    if not (np.isfinite(wavelength_m) and wavelength_m > 0):
        raise ValueError('wavelength_m must be positive')
    return '\n'.join([
        'INFILE input.cpx', 'OUTFILE unwrapped.f32', 'CORRFILE coherence.f32',
        'BYTEMASKFILE valid.u8',
        'INFILEFORMAT COMPLEX_DATA', 'OUTFILEFORMAT FLOAT_DATA',
        'CORRFILEFORMAT FLOAT_DATA', 'STATCOSTMODE DEFO',
        'INITMETHOD MCF',
        f'NCORRLOOKS {nlooks:.10g}',
        f'LAMBDA {wavelength_m:.10g}', '',
    ])


def unwrap_snaphu(igram, coherence, mask, nlooks, wavelength_m,
                   *, scratch: Path, executable: str = 'snaphu'):
    exe = shutil.which(executable)
    if not exe:
        raise RuntimeError(f'SNAPHU executable not found: {executable}')
    scratch.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(igram, np.complex64).tofile(scratch/'input.cpx')
    np.ascontiguousarray(coherence, np.float32).tofile(scratch/'coherence.f32')
    np.ascontiguousarray(mask, np.uint8).tofile(scratch/'valid.u8')
    (scratch/'snaphu.conf').write_text(snaphu_config(nlooks,wavelength_m), encoding='utf-8')
    t0 = time.perf_counter()
    with (scratch/'snaphu.log').open('w',encoding='utf-8') as log:
        p = subprocess.run([exe, '-d', '-f', 'snaphu.conf', str(igram.shape[1]),
                            '-g', 'conncomp.u8'], cwd=scratch, stdout=log,
                           stderr=subprocess.STDOUT, check=False)
    seconds = time.perf_counter()-t0
    if p.returncode != 0:
        raise RuntimeError(f'SNAPHU failed (exit {p.returncode}); see {scratch/"snaphu.log"}')
    _verify_file_size(scratch/'unwrapped.f32', igram.size*4)
    _verify_file_size(scratch/'conncomp.u8', igram.size)
    uw = np.fromfile(scratch/'unwrapped.f32',np.float32).reshape(igram.shape)
    conn = np.fromfile(scratch/'conncomp.u8',np.uint8).reshape(igram.shape)
    err=_parity_max(uw,igram,mask)
    if err > 1e-3:
        raise RuntimeError(f'SNAPHU wrapped parity mismatch: {err:g} rad')
    return uw,conn,{'seconds':seconds,'wrap_parity_max_rad':err}


def compare_same_input(a, b, comp_a, comp_b, mask):
    """Compare cycles only within intersections of solver-defined components."""
    if not (a.shape == b.shape == comp_a.shape == comp_b.shape == mask.shape):
        raise ValueError('Comparison raster dimensions differ')
    valid = mask & (comp_a != 0) & (comp_b != 0)
    nvalid = int(np.count_nonzero(valid))
    if not nvalid:
        return {'comparable_pixels':0,'fraction_cycle_mismatch':None,
                'joint_components':0}
    cycles = np.rint((np.asarray(a[valid],np.float64) -
                      np.asarray(b[valid],np.float64))/TAU).astype(np.int32)
    comp = np.column_stack((np.asarray(comp_a[valid],np.int64),
                            np.asarray(comp_b[valid],np.int64)))
    _, inv = np.unique(comp,axis=0,return_inverse=True)
    # The median integer is a gauge for diagnostics, not a physical level.
    order = np.argsort(inv, kind='stable')
    ordered_group = inv[order]
    ordered_cycles = cycles[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(ordered_group)) + 1, len(order)]
    count = 0
    for start, stop in zip(cuts[:-1], cuts[1:]):
        z = ordered_cycles[start:stop]
        offset = int(np.rint(np.median(z)))
        count += int(np.count_nonzero(z != offset))
    return {'comparable_pixels':nvalid,
            'fraction_cycle_mismatch':float(count/nvalid),
            'joint_components':int(inv.max())+1,
            'scientific_note':'Integer differences gauge-aligned separately for each component intersection; neither solution is ground truth'}


def audit_inputs(*, ifg, corr, mask, nlooks, wavelength_m, args):
    return {
      'status':'INPUT_AUDIT' if args.audit_only else 'EXPERIMENT_ONLY_NOT_PRODUCTION',
      'ifg_path':str(Path(args.ifg).resolve()),
      'coherence_path':str(Path(args.corr).resolve()),
      'raster_shape_full':[args.full_height,args.full_width],
      'crop':[args.row0,args.col0,args.height,args.width],
      'file_endian':args.endian,
      'coherence_layout':args.corr_layout,
      'nlooks_effective_coherence':None if nlooks is None else float(nlooks),
      'wavelength_m':None if wavelength_m is None else float(wavelength_m),
      'valid_pixels':int(mask.sum()),
      'fraction_valid':float(mask.mean()),
      'coherence_p05_p50_p95':[float(x) for x in np.percentile(corr[mask],[5,50,95])],
      'input_phase_finite_fraction':float(np.mean(np.isfinite(np.angle(ifg[mask])))),
      'warning':'Requires genuine interferometric sample coherence and defensible effective looks; differs from PointPhaseStack phase-linking time coherence.'
    }


def main(argv=None, *, whirlwind_func=None):
    p=argparse.ArgumentParser(description='Same-input GAMMA IFG: Whirlwind vs SNAPHU A/B; never writes production')
    p.add_argument('--ifg',type=Path,required=True)
    p.add_argument('--corr',type=Path,required=True)
    p.add_argument('--full-height',type=int,required=True)
    p.add_argument('--full-width',type=int,required=True)
    p.add_argument('--row0',type=int,default=0)
    p.add_argument('--col0',type=int,default=0)
    p.add_argument('--height',type=int,required=True)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--endian',choices=('big','little'),default='big')
    p.add_argument('--corr-layout',choices=('float32','alt-line'),default='float32')
    p.add_argument('--mask',type=Path)
    p.add_argument('--nlooks',type=float,default=None,
                   help='Effective independent looks of the actual coherence estimator, not nominal number of multilooks')
    p.add_argument('--wavelength-m',type=float,default=None)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--snaphu-exe',default='snaphu')
    args=p.parse_args(argv)
    if args.nlooks is not None and not (math.isfinite(args.nlooks) and args.nlooks >= 1):
        p.error('--nlooks must be a finite value >=1')
    if args.wavelength_m is not None and not (math.isfinite(args.wavelength_m) and args.wavelength_m > 0):
        p.error('--wavelength-m must be a positive finite value')
    if not args.audit_only and (args.nlooks is None or args.wavelength_m is None):
        p.error('Real A/B requires --nlooks and --wavelength-m from the actual data processing metadata')
    igram,corr,mask=load_gamma_crop(ifg_path=args.ifg,corr_path=args.corr,
        full_height=args.full_height,full_width=args.full_width,
        row0=args.row0,col0=args.col0,height=args.height,width=args.width,
        endian=args.endian,corr_layout=args.corr_layout,mask_path=args.mask)
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    summary=audit_inputs(ifg=igram,corr=corr,mask=mask,
                         nlooks=args.nlooks,wavelength_m=args.wavelength_m,args=args)
    (out/'input_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if args.audit_only:
        print(json.dumps(summary,indent=2));return 0
    import importlib.metadata
    try:
        package_version=importlib.metadata.version('whirlwind-insar')
    except importlib.metadata.PackageNotFoundError:
        package_version='unavailable'
    # Preserve the original IFG/correlation in memory; neither solver mutates them.
    wuw,wcc,wt=unwrap_whirlwind(igram,corr,mask,args.nlooks,unwrap_func=whirlwind_func)
    suw,scc,st=unwrap_snaphu(igram,corr,mask,args.nlooks,args.wavelength_m,
                            scratch=out/'snaphu_scratch',executable=args.snaphu_exe)
    np.save(out/'whirlwind_unwrapped_rad.npy',wuw)
    np.save(out/'whirlwind_conncomp.npy',wcc)
    np.save(out/'snaphu_unwrapped_rad.npy',suw)
    np.save(out/'snaphu_conncomp.npy',scc)
    summary.update({
        'status':'EXPERIMENT_ONLY_NOT_PRODUCTION',
        'whirlwind_version':package_version,
        'whirlwind':wt, 'snaphu':st,
        'whirlwind_labeled_pixels':int(np.count_nonzero(mask&(wcc!=0))),
        'snaphu_labeled_pixels':int(np.count_nonzero(mask&(scc!=0))),
        'component_gauge_invariant_comparison':compare_same_input(wuw,suw,wcc,scc,mask),
        'production_phase_modified':False,
        'scientific_note':'Same input IFG, coherence and mask: tests 2D solver only, not PS/DS time-series accuracy.'
    })
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));return 0


if __name__=='__main__':
    raise SystemExit(main())
