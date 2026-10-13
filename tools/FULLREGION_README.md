# Ningbo original-resolution Native EMCF engineering run

**Scope:** Directly use existing `point_phase_stack` (75 acquisitions; 219 IFG edges). No 200 m aggregation, nearest-neighbor point fill, or production overwrite. The output is **relative phase within linked spatial components**, NOT independently validated physical deformation.

## Install (four Python files)

Place these alongside the existing scripts in:

`/mnt/ningbo_process/ds/test-ds-main/tools/`

- `native_emcf_fullregion_run.py` — tile scanner, island queue, checkpoints
- `native_emcf_cached_worker.py` — exact original ROI phase cache, invoking the *local patched backend*
- `native_emcf_timeinv_general.py` — chunked 75-date relative inversion and per-point QA
- `native_emcf_fullregion_stitch.py` — overlap agreement, component gauge graph, optional mosaic

Nothing in `pypsds/unwrap/native_emcf_backend.py` is edited by these scripts. The local backend **must** offer `--enable-hole-faces`; the runner verifies this and stops if absent.

## One-time preflight

```bash
conda activate pypsds-emcf
cd /mnt/ningbo_process/ds/test-ds-main
python -m pypsds.unwrap.native_emcf_backend --help | grep enable-hole-faces
free -h
df -h /mnt/ningbo_process
python - <<'PY'
import numpy as np
p='/mnt/ningbo_process/ds/output/processing/point_phase_stack'
for k in ('phase_rad','rows','cols','point_type'):
 a=np.load(f'{p}/{k}.npy',mmap_mode='r')
 print(k,a.shape,a.dtype)
PY
```

Verify RSLC radar geometry: `12225` rows × `44842` columns. Change the runner's `--rows`, `--cols` if actual geometry differs. `min-points=32` intentionally omits small islands from this *engineering* run; the omitted counts are reported, **not** silently treated as solved. To request every triangulatable component, set `--min-points 3` (potentially far more runtime).

## Start with one full-sized tile (fast operational smoke test)

```bash
ROOT=/mnt/ningbo_process/ds
cd "$ROOT/test-ds-main"
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python tools/native_emcf_fullregion_run.py \
  --root "$ROOT" \
  --only-tile r07168_c14336 \
  --max-islands 1 --jobs 1 --min-points 32 \
  | tee /tmp/ningbo_fullres_smoke.log
```

This tile overlaps the formerly verified ROI and uses the new global grid. It is **not** the previously tested exact window, so do not infer success in advance. Confirm `LIMITED_PILOT`, and inspect one tile's `timeinv/qa_summary.json`, `emcf.log` and `timeinv.log`. If a substantial island exceeds 60,000 points, this pilot explicitly reports `island_too_large_split_tile` rather than exhausting RAM.

## Continue from checkpoints through the ENTIRE region

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python tools/native_emcf_fullregion_run.py \
  --root "$ROOT" --jobs 2 --workers 4 --blas-threads 2 \
  --min-points 32 \
  | tee "$ROOT/output/diagnostics/native_emcf_fullregion_engineering_global.log"
```

On a 64 GiB machine, start with `--jobs 1` if memory is near capacity, switch to `--jobs 2` only if peak usage is safe. This is a potentially long batch, not a seconds/minutes operation. **Resume by running the SAME command again**: completed islands with valid raw and inversion QA are skipped. Partial/failed folders are preserved (not overwritten); inspect `tiles/*/run_status.json` and corresponding logs. An incomplete run is NOT a successful release.

Outputs:

`$ROOT/output/diagnostics/native_emcf_fullregion_engineering/tiles/r*_c*/island_*/raw/`

`$ROOT/output/diagnostics/native_emcf_fullregion_engineering/tiles/r*_c*/island_*/timeinv/`

`$ROOT/output/diagnostics/native_emcf_fullregion_engineering/last_run_summary.json`

The tile ROI phase caches are exact original observations, not interpolation. They are retained for resuming. Disk usage may be substantial; monitor free space.

## Stitch only after every tile has reported DONE

```bash
python tools/native_emcf_fullregion_stitch.py \
  --run "$ROOT/output/diagnostics/native_emcf_fullregion_engineering" \
  --assemble \
  | tee /tmp/ningbo_fullregion_stitch.log
```

The overlap stitcher matches **identical original point IDs** in neighboring overlapping tiles. It rejects inconsistent overlaps and conflicting gauge cycles; disconnected components retain distinct `alignment_component_id`. It refuses ambiguous duplicates assigned to different unaligned components. The out-of-core `.npy` mosaic uses `point × 75 date` layout. The output is **not** a single physical reference frame if there are multiple disconnected alignment components; it never inserts fake points or offsets.

Important limitations:

1. Passing IFG reconstruction/parity and time cycles is a **numerical consistency check**, not proof against a physically wrong coherent integer alias.
2. The 40 m threshold is nominal radar geometry, not guaranteed strain continuity; false edges may propagate correlated unwrapping errors.
3. Tiny islands, holes with unsupported topology, >60k-point components, low-overlap bridges, or unsupported points remain explicitly **unresolved**, requiring separate scientific judgment or a revised tiling plan.
4. The script is tested for Python syntax and general inversion + stitching against synthetic data. It has **not** been run on your remote patched backend or complete 26.4-million-point data. The one-tile operational smoke test is mandatory.
5. No SCLA, SCN, GACOS, LOS unit conversion, reference-network validation, or final GIS export is performed by these scripts. Integrate only after engineering QA.
