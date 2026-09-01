# pyPSDS-GAMMA

pyPSDS-GAMMA is a production-oriented PS/DS-InSAR processing pipeline for
GAMMA-coregistered RSLC stacks. It provides statistically homogeneous pixel
(SHP) selection, PS/DS processing, robust Phase Linking, network quality
control, phase unwrapping, time-series inversion, optional atmospheric and
post-inversion corrections, and final LOS deformation products.

This README is the primary installation and user manual.

---

## 1. Main workflow

The public workflow is organized into nine logical modules:

```text
data_ps
  ↓
shp
  ↓
phase_linking
  ↓
ps_ds
  ↓
network_qc
  ↓
unwrap
  ↓
timeseries
  ↓
corrections
  ↓
products
```

The module interface is recommended for normal processing. Internal stages are
retained for checkpoint recovery, debugging, and advanced control.

List the public modules:

```bash
pypsds modules
```

List the internal stages:

```bash
pypsds run \
  --config pypsds.yaml \
  --list-stages
```

---

## 2. Scientific conventions

The main conventions used by the production pipeline are:

- Final LOS displacement is positive toward the satellite.
- LOS conversion is

  ```text
  d_LOS = +lambda / (4*pi) * phase
  ```

- Radar wavelength is read from the project GAMMA parameter files.
- `reference_date` is the GAMMA geometric/coregistration reference acquisition.
- `phase_linking.temporal_reference_index` is the mathematical temporal phase
  gauge and is independent of `reference_date`.
- The final spatial deformation reference is configured under `reference:`.
- Automatic reference selection establishes a relative InSAR datum; it does
  not prove that the selected area is physically motionless.
- LOS-to-vertical conversion requires the explicit approximation of negligible
  horizontal motion.
- Runtime scheduling parameters control performance only. They must not change
  the scientific PS/DS/SHP/Phase-Linking definitions.

---

## 3. Requirements

Recommended environment:

- Linux
- Ubuntu 22.04 or 24.04
- Python 3.11 or newer
- Python 3.12 recommended
- licensed GAMMA installation
- sufficient local RAM and storage for large full-scene arrays

Required GAMMA commands include:

```text
SLC2pt
data2pt
phase_sim_orb_pt
base_calc
base_orbit
```

They must be executable from the shell used to run pyPSDS-GAMMA.

Check:

```bash
which SLC2pt
which data2pt
which phase_sim_orb_pt
which base_calc
which base_orbit
```

If GAMMA uses a setup script, load it before activating/running pyPSDS-GAMMA.

---

## 4. Installation

### 4.1 Recommended Conda installation

```bash
conda create -n pypsds python=3.12 pip -y
conda activate pypsds

git clone https://github.com/songzwgithub/test-ds.git
cd test-ds
```

Core installation:

```bash
python -m pip install .
```

Complete product/export installation:

```bash
python -m pip install ".[products]"
```

Development installation:

```bash
python -m pip install -e ".[dev,products]"
```

The `products` extra installs the optional monitoring/export stack including
pandas, pyarrow, geopandas, rasterio, pyproj and pyshp.

### 4.2 Verify installation

```bash
python -m pip check
```

```bash
python - <<'PY'
import importlib.metadata as md
import pypsds

print("runtime version :", pypsds.__version__)
print("metadata version:", md.version("pypsds-gamma"))
print("source          :", pypsds.__file__)
PY
```

For an editable development installation, `source` should point to the cloned
repository.

### 4.3 Run tests

```bash
pytest -q
```

---

## 5. GAMMA environment

A typical shell session is:

```bash
conda activate pypsds

# Load GAMMA environment if it is not already in PATH.
# Example only; use the actual GAMMA installation on your server.
export GAMMA_HOME=/path/to/GAMMA_SOFTWARE
export PATH="$GAMMA_HOME/IPTA/bin:$GAMMA_HOME/DIFF/bin:$GAMMA_HOME/ISP/bin:$PATH"

which SLC2pt
which data2pt
which phase_sim_orb_pt
```

`pypsds doctor` will also check required GAMMA commands.

---

## 6. Create a project

Create a new project:

```bash
pypsds init /path/to/project
```

Example:

```bash
mkdir -p ~/insar/ningbo
pypsds init ~/insar/ningbo
cd ~/insar/ningbo
```

This creates:

```text
/path/to/project/pypsds.yaml
```

`pypsds init` always reads the currently installed packaged
`pypsds/resources/default_config.yaml`, so a newly generated project receives
the current configuration template.

To overwrite an existing configuration with the current template:

```bash
pypsds init /path/to/project --force
```

**Warning:** `--force` replaces the complete existing `pypsds.yaml`. Back up
project-specific paths, reference dates, ROI, correction settings and spatial
reference definitions first.

---

## 7. Recommended project layout

Conventional auto-discovered layout:

```text
project/
├── pypsds.yaml
├── RSLC/
│   ├── YYYYMMDD.rslc
│   ├── YYYYMMDD.rslc.par
│   ├── YYYYMMDD.rslc
│   ├── YYYYMMDD.rslc.par
│   └── ...
├── RSLC_tab
├── DEM_prep/
├── GACOS/                  # optional
└── output/
```

The project can use different directories by setting explicit paths in
`pypsds.yaml`.

---

## 8. Input contract

### 8.1 RSLC stack

The input must already be a GAMMA-coregistered RSLC stack.

Sensor-specific raw/SLC preprocessing and coregistration are outside the
pyPSDS-GAMMA production pipeline.

`RSLC_tab` defines the acquisition stack.

Each acquisition must have:

```text
YYYYMMDD.rslc
YYYYMMDD.rslc.par
```

The stack dates, dimensions, wavelength and other project properties are read
from the current project rather than hard-coded.

### 8.2 Geometry and DEM

The processing project requires the GAMMA geometry needed by point-based
topographic/orbital phase simulation.

The radar-coordinate height raster must correspond to the processing geometry.

The project-level:

```yaml
reference_date: YYYYMMDD
```

must be the actual GAMMA geometric/coregistration reference acquisition and
must exist in `RSLC_tab`.

### 8.3 Optional GACOS

GACOS is optional.

When enabled, the correction products are matched by acquisition date. With
strict date checking enabled, missing dates stop processing rather than being
silently ignored.

---

## 9. Essential configuration

Start from the generated `pypsds.yaml`.

The most important project-specific settings are:

```yaml
reference_date: YYYYMMDD

paths:
  work_dir: .
  data_dir: .
  rslc_dir: null
  rslc_tab: null
  output_dir: output
  dem_dir: null
  gacos_dir: null

processing:
  roi:
    row0: 0
    col0: 0
    rows: null
    cols: null
```

`null` path values use project auto-discovery.

---

## 10. Current performance-oriented runtime defaults

The performance build uses hardware-aware scheduling:

```yaml
runtime:
  cpu: auto
  memory_fraction: 0.85
  io_workers: auto

  autotune:
    enabled: true
    sample_points: 16384
    repeats: 2

  phase_link_prefetch_tiles: 1
```

Phase-correction runtime defaults:

```yaml
phase_correction:
  enabled: true
  backend: gamma_phase_sim_orb_pt
  command_timeout_seconds: 300.0

  command_retries: 2
  retry_backoff_seconds: 1.0

  parallel:
    spatial_workers: auto
    pair_workers: auto
```

### 10.1 Canonical Phase-Linking geometry

The canonical GAMMA phase cell remains:

```text
128 x 256 pixels
```

This is part of the validated numerical grouping and should not be enlarged
simply for performance.

Only execution scheduling is parallelized.

On a validated 32-logical-CPU host, the performance fallback is:

```text
6 spatial workers x 3 pair workers
= maximum 18 concurrent GAMMA phase_sim_orb_pt processes
```

If a parity-validated:

```text
output/processing/canonical_phase_parallel_autotune.json
```

exists, its validated winner overrides the hardware fallback.

### 10.2 Phase-Linking prefetch

With:

```yaml
runtime:
  phase_link_prefetch_tiles: 1
```

the next real-acquisition tile can be streamed/corrected while the current tile
performs SHP/coherence/EMI/compression work.

This improves CPU overlap on large scenes.

The asynchronous prefetch path includes a fail-fast watchdog. To use the
conservative synchronous path:

```yaml
runtime:
  phase_link_prefetch_tiles: 0
```

### 10.3 GAMMA retry

External GAMMA failures are retried in place:

```yaml
phase_correction:
  command_retries: 2
  retry_backoff_seconds: 1.0
```

This means:

```text
initial attempt
  ↓ failure
retry after 1 s
  ↓ failure
retry after 2 s
```

A transient `phase_sim_orb_pt` failure such as Linux return code `-11`
(SIGSEGV) therefore does not immediately terminate the complete
Phase-Linking stage.

If all retries fail, the final failed temporary tile bundle is preserved under:

```text
output/phase_correction/failed_tiles/
```

for reproduction and diagnosis.

---

## 11. PS / SHP / DS defaults

Typical production settings:

```yaml
selection:
  ps:
    amplitude_dispersion_max: 0.25

  shp:
    method: rayleigh_glrt
    alpha: 0.005
    half_row: 5
    half_col: 11
    min_count: 48
    exclude_ps: true
    policy: solver_aware
    rank_guard: true

    adaptive_window:
      enabled: true

  ds:
    center_mode: all
    temporal_coherence_min: 0.8
    pair_coherence_min: 0.0
    accept_evd: true
```

The exact SHP support cache is reused by sequential Phase Linking.

---

## 12. Phase Linking

Typical sequential configuration:

```yaml
phase_linking:
  method: robust_emi
  beta: 0.0
  gamma_jitter: 1.0e-06
  target_eigenvalue: 0.99
  evd_fallback: true
  temporal_reference_index: 0

  temporal:
    strategy: sequential
    ministack_size: 19
    max_num_compressed: 5
    state_min_shp: 24
    full_scm_fallback: true
    fullspan_batch_size: 16384
    emi_backend: threshold_cholesky
    use_exact_support_cache: true
```

The sequential solver is intended for large stacks and large scenes where a
full-scene full-SCM representation would be too expensive.

---

## 13. Reference, corrections and products

### 13.1 Spatial reference

Example automatic relative reference:

```yaml
reference:
  method: auto
  min_points: 100
  statistic: median
```

An externally validated stable reference should be used when available.

### 13.2 Atmospheric correction

Disabled:

```yaml
corrections:
  atmosphere:
    mode: disabled
```

GACOS:

```yaml
corrections:
  atmosphere:
    mode: gacos
    backend: gacos
    strict_dates: true
```

### 13.3 SCLA and SCN

Example:

```yaml
corrections:
  scla:
    mode: stamps
    backend: stamps

  scn:
    mode: stamps
    backend: stamps
    temporal_window_days: 365.0
    wavelength_m: 100.0
    radius_factor: 4.0
    cell_size_m: 200.0
```

### 13.4 Final products

The products module creates final LOS and point products. Optional product
extras enable formats such as GeoTIFF, GeoPackage, Shapefile and Parquet.

---

## 14. Preflight before a production run

For every new project:

```bash
pypsds config-check \
  --config pypsds.yaml
```

```bash
pypsds doctor \
  --config pypsds.yaml
```

```bash
pypsds plan \
  --config pypsds.yaml
```

Dry run:

```bash
pypsds run \
  --config pypsds.yaml \
  --dry-run
```

The doctor output should confirm:

- stack dimensions;
- acquisition count;
- reference date;
- GAMMA geometry;
- available CPU/RAM;
- Phase-Linking runtime schedule;
- required GAMMA executables.

---

## 15. Runtime autotuning

The package can benchmark the numerical Phase-Linking runtime schedule on a
bounded real-data sample:

```bash
pypsds tune \
  --config pypsds.yaml
```

Force recalibration:

```bash
pypsds tune \
  --config pypsds.yaml \
  --force
```

The tuned runtime profile is reused when its runtime signature remains valid.

Autotuning is intended to change scheduling only. Numerical parity checks are
part of the tuning contract.

---

## 16. Run the pipeline

### 16.1 Complete processing

```bash
pypsds run \
  --config pypsds.yaml
```

### 16.2 Run one public module

```bash
pypsds run \
  --config pypsds.yaml \
  --module phase_linking
```

### 16.3 Run a module interval

```bash
pypsds run \
  --config pypsds.yaml \
  --from-module shp \
  --to-module products
```

### 16.4 Start from an internal stage

Advanced recovery/debugging:

```bash
pypsds run \
  --config pypsds.yaml \
  --from-stage phase_linking
```

or:

```bash
pypsds run \
  --config pypsds.yaml \
  --from-stage exact_support_cache
```

Do not mix module selectors and internal-stage selectors in one command.

---

## 17. Public modules

| Module | Main purpose |
|---|---|
| `data_ps` | RSLC/phase-source preparation, statistics and PS candidates |
| `shp` | exact statistically homogeneous pixel support |
| `phase_linking` | covariance/coherence processing and sequential robust Phase Linking |
| `ps_ds` | DS selection, PS finalization and PS/DS merge |
| `network_qc` | temporal/spatial network and quality-control processing |
| `unwrap` | phase unwrapping and unwrap validation |
| `timeseries` | geometry, residual-ramp processing and time-series inversion |
| `corrections` | spatial reference, atmosphere, SCLA and SCN |
| `products` | final LOS and point/product export |

Use:

```bash
pypsds modules
```

for the authoritative module list in the installed version.

---

## 18. Resume and checkpoint behavior

pyPSDS-GAMMA uses persistent stage/tile checkpoints for large production runs.

Normal restart:

```bash
pypsds run \
  --config pypsds.yaml \
  --from-stage phase_linking
```

Do **not** use `--force` for ordinary recovery.

Do **not** set:

```bash
PYPSDS_FORCE_FRESH_TILES=1
```

unless a deliberate fresh sequential-tile recomputation is required.

### 18.1 Sequential checkpoint safety

Sequential tile markers are published only after the corresponding persistent
arrays have been flushed.

Empty state tiles are also checkpointed so that the durable checkpoint prefix
remains contiguous.

### 18.2 Repair legacy empty-tile checkpoint gaps

Older runs may contain empty-tile marker gaps.

Audit:

```bash
python tools/repair_empty_sequential_checkpoints.py \
  --processing-dir /path/to/output/processing \
  --stage 0
```

Apply only after the audit passes:

```bash
python tools/repair_empty_sequential_checkpoints.py \
  --processing-dir /path/to/output/processing \
  --stage 0 \
  --apply
```

The repair tool refuses to manufacture markers for missing non-empty tiles.

---

## 19. Important recovery rules

When resuming an expensive production run:

1. keep the same project configuration;
2. keep the same scientific source implementation;
3. keep the same canonical numerical phase grouping;
4. do not move/replace large checkpoint identity files unless their identity
   semantics are understood;
5. do not use `--force` simply because a stage failed;
6. inspect the original failure first;
7. preserve `output/processing/sequential`;
8. preserve `output/phase_correction`;
9. preserve K-state/effective-K arrays and exact support caches.

Performance scheduling should be tested independently before changing a running
checkpointed production job.

---

## 20. Output structure

Typical output tree:

```text
output/
├── logs/
├── qa/
├── runtime/
├── phase_correction/
│   ├── geometry_cache/
│   ├── failed_tiles/
│   └── ...
├── processing/
│   ├── ds_statistics/
│   ├── exact_support_cache/
│   ├── sequential/
│   │   ├── checkpoints/
│   │   └── ...
│   ├── network/
│   ├── network_inversion/
│   └── canonical_phase_parallel_autotune.json
└── products/
```

Exact products depend on enabled modules and optional corrections.

---

## 21. Performance guidance for large scenes

For large scenes, the main performance rules are:

- keep canonical phase grouping at the validated size;
- use spatial parallelism rather than over-parallelizing every temporal pair;
- reuse persistent GAMMA geometry caches;
- reuse exact SHP support;
- use sequential Phase Linking;
- use runtime autotuning;
- enable one-ahead prefetch only after stability testing on the target machine;
- avoid unnecessary full-scene cache regeneration;
- keep checkpoint cadence large enough to avoid excessive memmap flushes;
- keep BLAS/Numba/process oversubscription under control.

On a 32-logical-CPU machine, the validated performance fallback is currently:

```text
Phase source      : GAMMA
Canonical cell    : 128 x 256
Spatial workers   : 6
Pair workers      : 3
GAMMA max         : 18 processes
PL workers        : hardware/runtime tuned
Prefetch          : 1 in the performance configuration
```

Actual optimal settings depend on CPU, RAM, filesystem and stack size.

---

## 22. Troubleshooting

### GAMMA command not found

```bash
which phase_sim_orb_pt
```

Load the GAMMA environment and rerun:

```bash
pypsds doctor \
  --config pypsds.yaml
```

### `phase_sim_orb_pt` return code `-11`

On Linux, a negative subprocess return code represents termination by signal.
`-11` corresponds to SIGSEGV.

The current performance implementation retries transient GAMMA failures.
Persistent failures preserve the final failed tile bundle under:

```text
output/phase_correction/failed_tiles/
```

### Sequential checkpoint fingerprint changed

Do not immediately delete checkpoints.

Check first whether:

- source code changed;
- configuration changed;
- input/checkpoint file identities changed;
- large files were moved/copied;
- runtime parameters included in the checkpoint fingerprint changed.

Use a fresh sequential recomputation only when intentional.

### Non-contiguous checkpoint

For legacy runs, first audit:

```bash
python tools/repair_empty_sequential_checkpoints.py \
  --processing-dir /path/to/output/processing \
  --stage 0
```

### Low CPU utilization

Check:

```text
runtime.phase_link_prefetch_tiles
runtime autotune profile
canonical spatial/pair workers
PL workers
filesystem throughput
geometry-cache hit rate
```

A low average load can occur when GAMMA streaming and EMI/SCM are executed
serially instead of overlapped.

### Out of disk space

Large scenes can create very large persistent arrays. Check before a fresh
Phase-Linking run:

```bash
df -h /path/to/project
du -sh output/*
```

Do not delete checkpoint arrays without understanding their role.

---

## 23. Updating an existing project

A newly generated config always uses the currently installed default template:

```bash
pypsds init /new/project
```

For an existing project, `--force` replaces the complete configuration:

```bash
pypsds init /existing/project --force
```

Back up the existing project configuration first:

```bash
cp pypsds.yaml pypsds.yaml.backup
```

Then compare old and new configurations rather than blindly discarding
project-specific values.

---

## 24. Development

Install editable development dependencies:

```bash
python -m pip install -e ".[dev,products]"
```

Run the complete test suite:

```bash
pytest -q
```

Run syntax checks:

```bash
python -m py_compile \
  pypsds/gamma/phase_correction.py \
  pypsds/phase_linking/phase_source.py \
  pypsds/phase_linking/sequential_multistage.py \
  pypsds/phase_linking/tile_prefetch.py
```

Check repository diff:

```bash
git diff --check
git status --short
```

---

## 25. Minimal production command sequence

For a new project:

```bash
# 1. install
conda create -n pypsds python=3.12 pip -y
conda activate pypsds

git clone https://github.com/songzwgithub/test-ds.git
cd test-ds
python -m pip install ".[products]"

# 2. initialize
pypsds init /path/to/project
cd /path/to/project

# 3. edit pypsds.yaml
#    - reference_date
#    - paths if non-standard
#    - ROI if required
#    - correction/reference settings

# 4. validate
pypsds config-check --config pypsds.yaml
pypsds doctor --config pypsds.yaml
pypsds plan --config pypsds.yaml

# 5. optional runtime tuning
pypsds tune --config pypsds.yaml

# 6. dry run
pypsds run --config pypsds.yaml --dry-run

# 7. production
pypsds run --config pypsds.yaml
```

For a normal checkpoint restart:

```bash
pypsds run \
  --config pypsds.yaml \
  --from-stage phase_linking
```

No `--force` is required for normal resume.

---

## License

Apache-2.0.
