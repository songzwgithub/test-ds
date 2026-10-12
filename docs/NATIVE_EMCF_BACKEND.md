# Native PS/DS Spurt EMCF – single ROI island pilot

This branch is **not production**, **not a final 75-acquisition phase time series**, and does **not modify the existing** `stamps3d_unwrap`, `stamps3d_statcost_ab`, `point_phase_stack`, `final_los_tp_la`, or `gis_tp_la` products.

Scientific design: read native `point_phase_stack/phase_rad.npy`, `rows.npy`, `cols.npy`, `point_type.npy` with exact IDs. Build a Delaunay triangulation on real radar coordinates with nominal spacing, reject triangles with any edge exceeding `--max-edge-m`, and solve **only one connected, hole-free, consistently oriented triangle island**. Never fill gaps, bridge invalid areas, or copy a coarse integer history onto native points.

The MCF core is the **official** `spurt.workflows.emcf._solver.EMCFSolver` plus Spurt's `ORMCFSolver`, `Hop3Graph` and `Irreg3DInput`. Spurt expects `[acquisitions, points]` and produces `[temporal-ifg edges, points]`, not `[points, acquisitions]`. The original input stack uses `[points, acquisitions]` and is transposed. Spatial node indices remain aligned with original native point IDs; all spatial cycles are oriented consistently. Input phase convention and output wrap congruence are checked.

Scientific limitations:

* The 40 m bound uses nominal radar pixel spacings, not surveyed ground distances; a short Delaunay edge may still cross an actual water body or a physical displacement discontinuity.
* No temporal coherence weights or calibrated per-edge statistical costs are implemented; Spurt's built-in distance cost is a computational pilot, not a physical likelihood.
* The solver uses a Hop3 time graph instead of the old `network.itab`; compare time-network differences, do not equate the output IFGs.
* Disconnected triangle islands and holes are **not repaired** by interpolation; a non-manifold or hole-containing island is skipped. Isolated native points do not appear in an EMCF output.
* Spurt's IFG phase is reanchored with the observed *wrapped* reference-point IFG, just as in Spurt's standard workflow. Its temporal integer gauge can be unresolved. Report time-triangle closure, but do not invert these IFGs blindly or interpret them as independently referenced LOS displacement.
* Successful phase modulo parity and low closure counts are necessary consistency checks, not proof of true deformation or the absence of common-mode aliases. Holdout edges and independent PS-only comparisons are next.

## 1. Audit the exact native PS/DS ROI

Run in original pypsds environment (Spurt dependency is **not** needed for geometry audit):

```bash
cd /mnt/ningbo_process/ds/test-ds-main
conda activate pypsds
ROOT=/mnt/ningbo_process/ds
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m pypsds.unwrap.native_emcf_backend \
  --config "$ROOT/pypsds.yaml" \
  --row0 7155 --col0 14125 --height 1024 --width 1024 \
  --subset combined --max-edge-m 40 \
  --output-dir "$ROOT/output/diagnostics/native_emcf_roi_topology" \
  --audit-only
```

Read `geometry.islands_first_20` from `summary.json`. Choose a rank with `solver_eligible: true`, and inspect size and `max_edge_m` before running MCF. The largest island is not guaranteed to be hole-free.

## 2. Install Spurt safely in a cloned environment

Spurt uses OR-Tools; install a pinned official source snapshot to avoid unexpected dependency changes to the main pypsds production environment:

```bash
conda create -y -n pypsds-emcf --clone pypsds
conda activate pypsds-emcf
python -m pip install 'spurt @ git+https://github.com/isce-framework/spurt.git@4be99dec066e41567b7adfae12fe3d8b03a4a5de'
python -c 'import spurt, ortools; print("Spurt import PASS", spurt.__file__)'
cd /mnt/ningbo_process/ds/test-ds-main
python -m pytest -q tests/test_native_emcf_backend.py
```

Dependency installation and real Spurt integration were **not executable in the authoring sandbox**, which lacks Spurt and OR-Tools. Unit tests use a stub solver to test adapter contracts, not EMCF physics. The first server run is a real integration test.

## 3. Run a single eligible island; do not launch whole-scene production

```bash
ROOT=/mnt/ningbo_process/ds
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m pypsds.unwrap.native_emcf_backend \
  --config "$ROOT/pypsds.yaml" \
  --row0 7155 --col0 14125 --height 1024 --width 1024 \
  --subset combined --max-edge-m 40 \
  --island-rank 0 --max-run-points 60000 \
  --workers 4 --links-per-batch 1024 \
  --output-dir "$ROOT/output/diagnostics/native_emcf_roi_island0"
```

**Change `--island-rank` if 0 is not solver eligible.** Choose a new empty output directory on every attempt. Never use `--force`, never write to existing processing directories.

Output:

* `unwrapped_ifg_phase_rad.npy`: `[time-graph IFG, selected native point]`, relative spatially and anchored to observed wrapped temporal IFG at first point
* `time_ifg_pairs_0based.npy`: [time-graph IFG, 2] (Hop3 links)
* `original_point_ids.npy`, `original_radar_rows.npy`, `original_radar_cols.npy`, `original_point_type.npy`: original point identity preserved
* `space_graph_links.npy`, `space_graph_cycles.npy`: observed-node edges and valid triangle loops
* `summary.json`: topology, time closure diagnostics, modulo parity, exact Spurt version, and explicit nonproduction status

Do not pass outputs to the existing LOS/GIS modules yet. Need external PS-only validation, inter-island gauge registration, time-series inversion, and distinct output contracts.
