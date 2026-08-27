# pyPSDS-GAMMA

pyPSDS-GAMMA is a portable PS/DS-InSAR production pipeline for GAMMA-coregistered RSLC stacks. It integrates PS/DS selection, statistically homogeneous pixel (SHP) identification, robust Phase Linking, temporal/spatial network quality control, point-graph phase unwrapping, time-series inversion, optional atmospheric/SCLA/SCN corrections, final LOS displacement, and full-resolution point products.

The public production workflow is organized into **9 logical modules**. Internally, the package retains **38 checkpoint/output-contract stages** for resumability, failure isolation, and reproducible release validation.

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

Acquisition count, IFG count, point count, geometric reference, radar wavelength, multilook factors, spatial extent, CPU count, and available memory are resolved from the current project. Moving to another region, stack, or Linux server does not require editing Python source.

## Version 1.2.0

v1.2.0 keeps the validated scientific path while consolidating the user-facing workflow and reducing redundant Phase-Linking computation.

Key production changes include:

- 9 public logical processing modules while retaining 38 internal stages;
- cgroup-aware CPU/RAM planning and machine-local Phase-Linking autotuning;
- exact packed Rayleigh-GLRT SHP support caching;
- sequential robust EMI Phase Linking with threshold-Cholesky fast path and conservative EVD fallback;
- specialized all-pairs full-span coherence;
- allocation-light temporal-coherence evaluation;
- temporal canonical-cell reuse across sequential ministacks;
- fused post-Phase-Linking row-band processing;
- GAMMA subprocess timeout and resource budgeting;
- one-ahead GAMMA prefetch retained as an **opt-in** feature, with the production default **disabled**.

The frozen reference Phase-Linking workload (`600 × 2000 × 38`) decreased from approximately **438.4 s** to **302.5 s** while preserving the frozen science counts. Further eigensolver-level optimization is intentionally deferred for v1.2.0.

See `RELEASE_NOTES_v1.2.0.md` and `PERFORMANCE_BASELINE_v1.2.0.md` for the frozen release baseline.

## Scientific conventions

- Final LOS sign: **positive toward the satellite**.
- LOS conversion: `d_LOS = +lambda/(4*pi) * phi_final`.
- `lambda` is derived from `radar_frequency` in the current reference RSLC parameter file.
- The canonical temporal datum is the first acquisition of the finalized acquisition order.
- The GAMMA geometric/coregistration reference is independent of the temporal datum.
- Spatial referencing uses the configured reference-point set and epoch-wise median.

## Requirements

- Linux; Ubuntu 22.04/24.04 recommended.
- Python >= 3.11.
- Licensed GAMMA installation for GAMMA-backed operations.
- Required GAMMA commands must be available in `PATH` when their stages are used, including `SLC2pt`, `data2pt`, `phase_sim_orb_pt`, `base_calc`, and `base_orbit`.

Core dependencies are installed with the package.

Optional product dependencies:

```bash
pip install ".[products]"
```

The `products` extra provides pandas/Parquet/GeoPackage support. CSV remains the dependency-light tabular output.

## Installation

```bash
cd /path/to/pyPSDS-GAMMA
python -m pip install .
```

Development/tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Create a project

```bash
mkdir -p ~/insar/my_project
pypsds init ~/insar/my_project
cd ~/insar/my_project
```

Recommended layout:

```text
my_project/
├── pypsds.yaml
├── RSLC/
│   ├── YYYYMMDD.rslc
│   ├── YYYYMMDD.rslc.par
│   └── ...
├── RSLC_tab
├── DEM_prep/
├── GACOS/              # optional
└── output/
```

The conventional names above are auto-discovered. Configure `paths:` explicitly only when a project uses a different layout.

## Input contract

### RSLC stack

`RSLC_tab` defines the current coregistered stack. Each acquisition must have readable RSLC data and a matching GAMMA parameter file. Acquisition dates and raster dimensions are resolved from the current project rather than hard-coded in the package.

### Geometry

Configure the actual GAMMA coregistration/geometric reference date. It does not have to be the first temporal acquisition.

Radar-coordinate height and its matching geometry parameter file are required by GAMMA geometric phase-correction operations.

### GACOS

GACOS is optional. When enabled, correction files are matched to the current acquisition dates. With `strict_dates: true`, a missing required date is an error.

## Main configuration

Portable path defaults:

```yaml
paths:
  work_dir: .
  data_dir: .
  rslc_dir: null
  rslc_tab: null
  output_dir: output
  dem_dir: null
  gacos_dir: null
  scratch_dir: output/.scratch
  products_dir: output/products
```

Hardware-aware runtime:

```yaml
runtime:
  cpu: auto
  memory_fraction: 0.85
  io_workers: auto
  autotune:
    enabled: true
    sample_points: 16384
    repeats: 2
  phase_link_prefetch_tiles: 0
```

Use a CPU cap on a shared server, for example `cpu: 16`. Runtime settings change scheduling and resource use; they are not scientific thresholds.

### PS/DS and SHP defaults

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

### Phase Linking defaults

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

The solver-aware SHP policy distinguishes numerical support required by the sequential state from formal DS eligibility. Performance scheduling does not change these scientific definitions.

### Geometric phase correction

```yaml
phase_correction:
  enabled: true
  geometric_reference_date: YYYYMMDD
  command_timeout_seconds: 300.0
```

Use `geometric_reference_date: auto` only when the first temporal acquisition genuinely matches the GAMMA coregistration reference.

### Temporal network

```yaml
network:
  max_temporal_baseline_days: 72
  max_perpendicular_baseline_m: 200.0
  target_connections_each_side: 3
```

### Spatial reference

```yaml
reference:
  method: radar_window
  radar_window:
    center_row: 1000
    center_col: 2000
    half_row: 10
    half_col: 15
    min_points: 100
  statistic: median
```

Choose a physically stable reference region.

## Optional corrections

Atmospheric correction disabled:

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

SCLA:

```yaml
corrections:
  scla:
    mode: stamps
    backend: stamps
```

SCLA uses the finalized network. Missing GAMMA baseline models can be regenerated with `base_orbit`. Multilook factors are inferred from the current project geometry; the package does not assume a fixed 4:1 ratio.

SCN:

```yaml
corrections:
  scn:
    mode: stamps
    backend: stamps
    temporal_window_days: 365.0
    wavelength_m: 100.0
    radius_factor: 4.0
    cell_size_m: 200.0
```

## Preflight

For each new project:

```bash
pypsds config-check --config pypsds.yaml
pypsds doctor --config pypsds.yaml
pypsds plan --config pypsds.yaml
pypsds run --config pypsds.yaml --dry-run
```

List public modules:

```bash
pypsds modules
```

List internal stages:

```bash
pypsds run --config pypsds.yaml --list-stages
```

## Run

### Complete pipeline

```bash
pypsds run --config pypsds.yaml
```

### Run one logical module

```bash
pypsds run \
  --config pypsds.yaml \
  --module phase_linking
```

### Run a logical module interval

```bash
pypsds run \
  --config pypsds.yaml \
  --from-module shp \
  --to-module unwrap
```

The module interface is the recommended production interface.

Internal stage selectors remain available for advanced checkpoint recovery and debugging:

```bash
pypsds run \
  --config pypsds.yaml \
  --from-stage atmosphere_correction \
  --to-stage point_products
```

Do not mix module selectors and stage selectors in the same command.

## Public module mapping

| Module | Internal stages | Purpose |
|---|---:|---|
| `data_ps` | 2 | stack/phase-source preparation, statistics, PS candidates |
| `shp` | 1 | exact statistically homogeneous pixel support |
| `phase_linking` | 1 | bounded-memory covariance/coherence + sequential robust Phase Linking |
| `ps_ds` | 3 | DS selection, PS finalization, unified point stack |
| `network_qc` | 14 | temporal/spatial network construction and pre-unwrap QC |
| `unwrap` | 9 | unwrapping and post-unwrap closure/conflict/integer-candidate validation |
| `timeseries` | 1 | acquisition-domain time-series inversion |
| `corrections` | 5 | geometry, reference, atmosphere, SCLA, SCN |
| `products` | 2 | final LOS solution and point-product export |

Detailed mapping: `docs/PIPELINE_MODULES.md`.

## Internal 38-stage pipeline

The 38 stages remain checkpoint/output-contract units. They are not the primary user-facing workflow abstraction.

| # | Stage |
|---:|---|
|1|ds_statistics|
|2|phase_cache|
|3|exact_support_cache|
|4|phase_linking|
|5|ds_selection|
|6|ps_finalize|
|7|point_stack|
|8|network_prepare|
|9|network_build|
|10|network_cycle_quality|
|11|network_finalize|
|12|virtual_ifg_quality|
|13|spatial_graph_quality|
|14|spatial_bridge_quality|
|15|spatial_component_quality|
|16|spatial_anchor_quality|
|17|spatial_anchor_summary|
|18|spatial_local_graph_quality|
|19|spatial_graph|
|20|spatial_gradient_quality|
|21|unwrap_policy|
|22|unwrap|
|23|unwrap_severity_quality|
|24|unwrap_conflict_quality|
|25|unwrap_acquisition_quality|
|26|temporal_closure|
|27|temporal_integer_candidate|
|28|temporal_candidate_spatial_quality|
|29|unwrap_signature_quality|
|30|unwrap_finalize|
|31|timeseries_inversion|
|32|point_geometry|
|33|reference|
|34|atmosphere_correction|
|35|scla|
|36|scn|
|37|final_los|
|38|point_products|

## Phase-Linking execution architecture

### Runtime autotuning

The cgroup-aware planner first defines a safe CPU/RAM envelope. The runtime autotuner then benchmarks the real sequential threshold-Cholesky EMI solver on a bounded real-data sample and chooses a worker/chunk/batch schedule inside that envelope.

Autotuning is enabled by default. A normal production run creates or refreshes the runtime profile immediately before Phase Linking after the exact SHP support cache is available.

Profiles are reused only when the software version, CPU model, effective CPU allocation, Python/NumPy environment, NumPy build identity, and solver dimension match.

Explicit calibration:

```bash
pypsds tune --config /path/to/project/pypsds.yaml
```

Manual `runtime.phase_link_workers`, `runtime.phase_link_chunk_size`, or `runtime.phase_link_batch_size` values take precedence.

Runtime tuning changes scheduling only. Rayleigh-GLRT support, the sequential temporal plan, EMI/EVD mathematics, and DS thresholds are unchanged.

### GAMMA streaming and temporal canonical-cell reuse

GAMMA-backed Phase Linking uses a canonical spatial correction grid. Sequential ministacks correct only their required acquisition subsets.

When memory allows, corrected temporal pieces for the same canonical spatial cell are retained and later composed into the full-date cell used by post-Phase-Linking quality/fallback/PS-fill processing. Composition is accepted only when the temporal pieces exactly cover the requested date sequence and their geometry-valid masks are identical; otherwise the code falls back to the normal GAMMA full-date correction path.

The cache is explicitly bounded by available-memory policy. Small and medium scenes may retain all temporal canonical pieces; larger scenes remain bounded by the LRU/memory budget.

### Post-Phase-Linking row-band fusion

For GAMMA streaming, post-Phase-Linking processing is fused by row band. A full-date `PhaseTile` is shared by:

1. sequential full-span temporal/pair-coherence quality;
2. original-support full-SCM fallback;
3. PS linked-phase fill.

This avoids repeated full-date phase-source work.

### One-ahead tile prefetch policy

One-ahead GAMMA tile prefetch is implemented as an **explicit opt-in execution path**. It is **not the production default**.

```yaml
runtime:
  phase_link_prefetch_tiles: 0
```

The queue depth is fixed to one when enabled. Repeated real-data tests showed that the asynchronous canonical streaming path could stall non-deterministically after GAMMA subprocess completion, so synchronous streaming remains the default for v1.2.0.

GAMMA timeout protection, CPU/process budgeting, canonical caching, and post-PL fusion remain active regardless of the prefetch default.

## Full production validation

The v1.2.0 production tree completed the full `data_ps → products` workflow on the reference `600 × 2000 × 38` workload.

```text
formal DS                    : 1,077,566
sequential route             : 1,075,120
combined PL valid            : 1,077,566
combined TC >= 0.80          : 863,969
strict unwrap-valid points   : 881,315 / 881,516 (99.977198%)
strict-mask closure bad pts  : 0
strict-mask bad occurrences  : 0
Tree vs full-L2 RMS diff     : 1.108e-06 rad
final LOS                    : PASS
point products               : PASS
full run manifest            : created
```

The sparse temporal integer candidate remains quality-controlled: a candidate that removes temporal closure residuals is not applied when counterfactual spatial validation would create new SAFE spatial conflicts.

## Performance baseline

Frozen v1.2.0 Phase-Linking reference:

```text
stage seconds       : 232.158
post-PL fused total : 58.004
total wall          : 302.462
module wall         : 304.61
```

Earlier reference wall time was approximately `438.432 s`, corresponding to roughly a **31% reduction** while preserving the frozen scientific counts.

Performance optimization is frozen for v1.2.0. The remaining dominant numerical cost is the EMI fast-path eigensolver and is intentionally left unchanged in this release.

## Main outputs

Final phase/LOS:

```text
output/processing/final_los/
├── acquisition_phase_final_rad.npy
├── los_displacement_toward_satellite_m.npy
├── los_displacement_toward_satellite_mm.npy
└── final_los_manifest.json
```

Point products:

```text
output/products/
├── los_velocity_toward_satellite_mm_per_year.npy
├── los_cumulative_toward_satellite_mm.npy
├── linear_residual_rms_mm.npy
├── velocity_slope_standard_error_mm_per_year.npy
├── time_axis_contract.npz
├── point_products_manifest.json
├── point_products.csv
└── point_products.parquet
```

CSV and Parquet are the primary point tables in the validated production workflow.

### Optional GeoPackage export

GeoPackage export is optional and requires `geopandas`. If `geopandas` is not installed, pyPSDS-GAMMA skips only the GeoPackage export; CSV/Parquet and the point-products stage remain valid.

The velocity-slope standard error is the temporal OLS regression standard error, not total InSAR/geodetic uncertainty.

## Move to another region

Do not edit source code. Create a new project, provide the new RSLC stack and DEM/radar geometry, set the actual geometric reference, choose a stable spatial reference region, optionally provide GACOS, then run `config-check`, `doctor`, `plan`, and the production workflow.

## Move to another stack or sensor

The pipeline does not assume a fixed acquisition count. Final LOS wavelength is derived from the current RSLC `radar_frequency`, so final conversion is not tied to a fixed Sentinel-1 wavelength.

Sensor-specific preprocessing required to create a GAMMA-compatible coregistered RSLC stack remains outside this package.

## Move to another server

Install Python/package and GAMMA, copy or mount the project data, update only filesystem paths if required, and run `doctor`/`plan`.

`cpu: auto` follows CPU resources available to the process, including detected CPU affinity/cgroup restrictions. Memory planning is bounded by host and detected cgroup availability.

## Release validation

Inspect package release identity:

```bash
python tools/release_gate.py identity
```

Run the complete release gate:

```bash
python tools/release_gate.py all --config /path/to/project/pypsds.yaml
```

Individual gates:

```bash
python tools/release_gate.py tests
python tools/release_gate.py wheel
python tools/release_gate.py contract --config /path/to/project/pypsds.yaml
```

## Troubleshooting

- **GAMMA command missing:** load the GAMMA environment and rerun `doctor`.
- **GAMMA command timeout:** inspect the phase-correction GAMMA log and `phase_correction.command_timeout_seconds`.
- **Multiple geometry candidates:** configure the desired geometry explicitly.
- **SCLA baseline missing:** verify network dates and that `base_orbit` is available.
- **GACOS date mismatch:** provide products for every required acquisition or intentionally disable strict correction.
- **Too few reference points:** choose or enlarge a physically stable reference region.
- **Out of memory:** reduce CPU cap/memory fraction; do not change scientific PS/DS thresholds merely to fit hardware.
- **GeoPackage skipped:** install `.[products]` or `geopandas`; CSV/Parquet remain valid.

## License

Apache-2.0.
