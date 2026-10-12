# Phase-linked PS/DS multi-resolution support audit

**Read-only diagnostic. It does not perform phase unwrapping, resample actual phase, or write any production outputs.**

The input is `output/processing/point_phase_stack/phase_rad.npy`, which contains the **wrapped** phase of 75 phase-linked acquisitions sampled at 26.4 million PS and DS points. It is not GAMMA's original `.diff` and not an already-unwrapped LOS product. PS=1, DS=2 are read from `point_type.npy`.

For 50, 100 and 200 metre *nominal radar-coordinate* grids, it independently reports PS-only, DS-only and mixed spatial occupancy, observed four-neighbor edges, disconnected components, singleton components, and what coverage each point type adds. Cells without real points remain empty. The radar grid spacing is derived from the actual GAMMA reference RSLC `.par` file using the existing pyPSDS geometry adapter. At 200 m the occupied footprint is explicitly checked against `stamps3d_unwrap/coarse_node_grid.npy` if present.

For one temporal network edge it also evaluates three estimators separately for each point type:

- Mean acquisition phasor at i: `sum(exp(1j * phase_i))`
- Mean acquisition phasor at j: `sum(exp(1j * phase_j))`
- Direct point IFG mean: `sum(exp(1j * (phase_j-phase_i)))`

The wrapped phase of the difference of acquisition means **generally differs** from the circular mean of point IFG phases. This quantifies the possible distortion induced by the coarse-grid estimator. Low direct IFG resultant length is a sign of phasor disagreement **not a calibrated interferometric coherence**. Selecting a pair with the first reference acquisition would make this discrepancy trivially zero because all point phases at the reference are zero; the CLI automatically chooses a pair without that reference when possible.

## Run from the repository

```bash
cd /mnt/ningbo_process/ds/test-ds-main
conda activate pypsds
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python -m pypsds.unwrap.psds_resolution_audit \
  --config /mnt/ningbo_process/ds/pypsds.yaml \
  --grid-sizes-m 50 100 200 \
  --output-dir /mnt/ningbo_process/ds/output/diagnostics/psds_resolution_audit
```

For **quick topology-only diagnostics** before the more expensive point-phase streaming read:

```bash
python -m pypsds.unwrap.psds_resolution_audit \
  --config /mnt/ningbo_process/ds/pypsds.yaml \
  --grid-sizes-m 50 100 200 --no-phase \
  --output-dir /mnt/ningbo_process/ds/output/diagnostics/psds_resolution_support_only
```

This is a streaming Numba audit on 26 million points; it can take several minutes and reads part of an approximately 8 GB array once sequentially. It never loads all 75-point phase histories into memory at once. Only the `summary.json` output is written in the requested diagnostics directory.

## Interpretation

The most important comparisons are `cells_added_by_ds_over_ps_only`, `groups.combined.occupied_cells`, `groups.combined.component_count`, `groups.combined.observed_four_neighbor_edges`, and `phase_aggregation_audit.combined.aggregation_discrepancy_rad_p50_p90_p95_p99`. Check the PS/DS individual phase aggregation as well.

Do not pick the finest grid merely because it increases the occupied-cell count. Physical separations between occupied nodes and component-wise integer gauges remain unobservable without further independent constraints. This audit informs later high-resolution Whirlwind/SNAPHU and native sparse point-graph evaluations; it does not rank their actual unwrapping accuracy.
