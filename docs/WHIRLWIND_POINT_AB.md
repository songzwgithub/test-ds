# Whirlwind/SNAPHU comparison on phase-linked PS/DS data

The two-dimensional data are reconstructed from `stamps3d_unwrap/coarse_complex_mean_phase_rad.npy` (nodes x 75 acquisition dates) and `stamps3d_unwrap/coarse_node_grid.npy` (observed grid cells x/y, -1 for missing). Each chosen temporal edge's wrapped IFG is `exp(1j * (phase[:, j] - phase[:, i]))` **on occupied nodes only**. There is no original GAMMA IFG input and no filling across blank cells.

This is an A/B **solver experiment**, not an equivalent statistical reproduction of an original IFG. The quality input is either a uniform constant or the geometric mean of two per-acquisition complex-mean resultant lengths. Neither is independently calibrated interferometric sample coherence. The `--nlooks` numerical setting is a sensitivity-test assumption, not a measured equivalent look count, so it is inappropriate to assert probabilistic likelihood calibration or physical accuracy based on a lower solver cost.

For `--audit-only`, no Whirlwind installation or solver parameters are needed:

```bash
python -m pypsds.unwrap.whirlwind_point_ab \
  --node-phase /mnt/ningbo_process/ds/output/processing/stamps3d_unwrap/coarse_complex_mean_phase_rad.npy \
  --node-grid /mnt/ningbo_process/ds/output/processing/stamps3d_unwrap/coarse_node_grid.npy \
  --dates /mnt/ningbo_process/ds/output/processing/point_phase_stack/dates.txt \
  --itab /mnt/ningbo_process/ds/output/processing/network/network.itab \
  --pair-index 0 \
  --output-dir /mnt/ningbo_process/ds/output/diagnostics/whirlwind_point_ab/pair000 \
  --audit-only
```

To run *both* solvers on identical data, append the numerical assumptions `--nlooks 1 --wavelength-m <read from acquisition metadata>` and optionally `--baseline-dir /mnt/ningbo_process/ds/output/processing/stamps3d_statcost_ab`; omit `--audit-only`. Whirlwind must be installed. Both methods receive identical complex wrapped input, mask and quality surrogate. Neither method is permitted to bridge or interpolate missing regions. A single pair cannot recover a 75-acquisition time series. The next step is temporal network synchronization and independent PS-only comparisons, while preserving arbitrary integer gauges per disconnected spatial component.

Outputs are under the experiment directory, including `input_audit.json`, `summary.json`, `whirlwind_node_ifg_unwrapped_rad.npy`, `whirlwind_node_conncomp.npy`, `snaphu_node_ifg_unwrapped_rad.npy` and `snaphu_node_conncomp.npy`. No production arrays are modified.
