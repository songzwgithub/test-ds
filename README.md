# pyPSDS-GAMMA

pyPSDS-GAMMA is a portable PS/DS-InSAR production pipeline for GAMMA coregistered RSLC stacks. It combines PS/DS selection, statistical homogeneous pixels, robust phase linking, temporal-network quality control, point-graph phase unwrapping, optional atmospheric/SCLA/SCN corrections, final LOS displacement, and full-resolution point products.

The production chain contains **38 stages**. Acquisition count, IFG count, point count, geometric master, radar wavelength, multilook factors, spatial extent, CPU count and available memory are resolved from the current project. Moving to another region, stack, or Linux server does not require editing Python source.

## Scientific conventions

- Final LOS sign: **positive toward the satellite**.
- LOS conversion: `d_LOS = +lambda/(4*pi) * phi_final`.
- `lambda` is derived from `radar_frequency` in the current reference RSLC parameter file.
- Canonical temporal datum is the first acquisition of the finalized acquisition order.
- Geometric/coregistration reference is independent of the temporal datum.
- Spatial referencing uses the configured reference-point set and epoch-wise median.

## Requirements

- Linux; Ubuntu 22.04/24.04 recommended.
- Python >= 3.11.
- Licensed GAMMA installation for GAMMA-backed operations.
- GAMMA commands such as `SLC2pt`, `data2pt`, `phase_sim_orb_pt`, `base_calc`, and `base_orbit` must be available in `PATH` when their stages are used.

Core dependencies are installed with the package. Optional Parquet/GeoPackage products use:

```bash
pip install ".[products]"
```

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

The conventional project names above are auto-discovered. Set `paths:` explicitly only when the project uses a different layout.

## Input contract

### RSLC

`RSLC_tab` defines the current coregistered stack. Each acquisition must have readable RSLC data and a matching GAMMA parameter file. Dates and raster dimensions are read from the current project.

### Geometry

Set the actual GAMMA coregistration/geometric reference date. It does not have to be the first acquisition. Radar-coordinate height and its matching parameter file are required by the geometric phase-correction operations.

### GACOS

GACOS is optional. When enabled, correction files are matched to the acquisition dates of the current stack. With `strict_dates: true`, a missing date is an error.

## Main configuration

Portable path defaults:

```yaml
paths:
  work_dir: .
  data_dir: .
  rslc_dir: null       # auto-discover RSLC_cropped or RSLC
  rslc_tab: null       # auto-discover RSLC_tab or one matching *RSLC*tab*
  output_dir: output
  dem_dir: null        # auto-discover DEM_prep, DEM, or dem
  gacos_dir: null      # optional; auto-discover GACOS or gacos
  scratch_dir: output/.scratch
  products_dir: output/products
```

Hardware-aware runtime:

```yaml
runtime:
  cpu: auto
  memory_fraction: 0.85
  io_workers: auto
```

Use a CPU cap on a shared server, e.g. `cpu: 16`. Hardware settings change execution only; they are not scientific parameters.

PS/DS defaults include amplitude-dispersion PS selection, Rayleigh-GLRT SHP identification, robust EMI phase linking, and temporal-coherence DS selection.

Geometric reference:

```yaml
phase_correction:
  enabled: true
  geometric_reference_date: YYYYMMDD
```

Use `auto` only when the first temporal acquisition is genuinely the GAMMA coregistration reference.

Temporal network:

```yaml
network:
  max_temporal_baseline_days: 72
  max_perpendicular_baseline_m: 200.0
  target_connections_each_side: 3
```

Spatial reference example:

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

Atmosphere disabled:

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

SCLA uses the current finalized network. Missing GAMMA baseline models can be regenerated with `base_orbit`. Multilook factors are inferred from current RSLC and radar-geometry dimensions; no fixed 4:1 assumption is used.

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

Spatial radius is `wavelength_m * radius_factor`. Point count and neighbor census are generated from the current point stack.

## Preflight

For every new project:

```bash
pypsds config-check --config pypsds.yaml
pypsds doctor --config pypsds.yaml
pypsds plan --config pypsds.yaml
pypsds run --config pypsds.yaml --dry-run
```

List stages:

```bash
pypsds run --config pypsds.yaml --list-stages
```

## Run

Complete pipeline:

```bash
pypsds run --config pypsds.yaml
```

Selected interval:

```bash
pypsds run --config pypsds.yaml --from-stage atmosphere_correction --to-stage point_products
```

## 38 production stages

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
└── point_products_manifest.json
```

CSV is the dependency-free default table. Install `.[products]` and add `parquet`/`geopackage` to `products.point.formats` when required.

The velocity slope standard error is the temporal OLS regression standard error, not total InSAR/geodetic uncertainty.

## Move to another region

Do not edit source code. Create a new project, provide the new RSLC stack/DEM geometry, set the actual geometric reference, choose a stable reference region, optionally provide GACOS, run `config-check`, `doctor`, and `plan`, then process. Acquisition/IFG/point counts and spatial extent are dynamic.

## Move to another stack or sensor

The pipeline does not assume a fixed acquisition count. Final LOS wavelength is derived from the current RSLC `radar_frequency`, so the final conversion is not limited to one Sentinel-1 wavelength. Sensor-specific preprocessing before a GAMMA-compatible coregistered RSLC stack remains outside this package.

## Move to another server

Install Python/package and GAMMA, copy/mount data, update only filesystem paths if needed, and run `doctor`/`plan`. `cpu: auto` follows the CPU resources available to the process, including detected affinity/cgroup restrictions; memory planning is bounded by host and detected cgroup availability. No source edit is required.

## Release validation

Inspect the release identity resolved from the package:

```bash
python tools/release_gate.py identity
```

Run the complete release gate:

```bash
python tools/release_gate.py all --config /path/to/project/pypsds.yaml
```

Individual gates remain available:

```bash
python tools/release_gate.py tests
python tools/release_gate.py wheel
python tools/release_gate.py contract --config /path/to/project/pypsds.yaml
```

The release gate derives the package version and complete production-stage sequence from the authoritative package sources. Version numbers and stage counts are not duplicated in the gate.

## Troubleshooting

- **GAMMA command missing:** load the GAMMA environment and rerun `doctor`.
- **Multiple geometry candidates:** configure the desired geometry explicitly.
- **SCLA baseline missing:** verify network dates and that `base_orbit` is available.
- **GACOS date mismatch:** provide products for every required acquisition or disable strict correction intentionally.
- **Too few reference points:** choose/enlarge a physically stable reference region.
- **Out of memory:** reduce CPU cap/memory fraction; do not change scientific PS/DS thresholds merely to fit hardware.

## License

Apache-2.0.
