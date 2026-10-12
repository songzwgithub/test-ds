# Original PS/DS point-graph integer branch (local validation)

This module is an alternative to the **coarse-grid** route, grounded in the documented separation between phase-linked wrapped observation generation and subsequent unwrapping in MiaplPy and Dolphin. It is newly implemented for **this pyPSDS repository**, not copied from, or claimed to be an implementation of, either upstream unwrapper.

### Operations completely bypassed

* NO 200 m complex-mean node phase; uses each point's original `PointPhaseStack/phase_rad.npy` values (`[original_point,75]`, wrapped).
* NO nearest-neighbor filling of empty radar pixels; a graph edge always connects **two actual input point IDs**.
* NO integer history broadcast from a coarse node to all its underlying points. Integer histories are estimated per original point, within its **own connected graph component**.

### What it actually does

1. Selects original PS/DS point IDs within a user-specified radar-coordinate ROI, preserving the exact IDs, point types, rows/cols, and all 75 phase-linked acquisition phases; no downsampling or circular averaging.
2. Proposes a bounded kNN graph using **nominal** azimuth/ground-range pixel spacings read from the reference GAMMA RSLC `.par` file. Each edge connects real measured pixels, but geometric proximity is **not proof** of physically smooth displacement.
3. For each edge, temporally unwraps the wrapped **spatial phase-gradient** sequence. Rejects edges with observed frame-to-frame wrapped steps near π. Unobserved true steps > π remain **fundamentally ambiguous**.
4. Holds out 20% of the temporally accepted edges (deterministic seed). Solves two alternative maximum-weight spanning forests over the remaining edges using the repository's existing `native_integer_constraints.py`. Computes training/heldout edge integer closure and 2-core/two-forest agreement. A heldout edge is not a distinct independent sensor; it is still derived from the same wrapped measurements.
5. Saves an isolated **local point integer candidate** and its per-component gauge-free phase time series. NEVER writes to existing PointPhaseStack, `stamps3d_unwrap`, `stamps3d_statcost_ab`, or final LOS products.

**Limitations:** This is a single-tile (ROI) graph candidate, not a whole-scene production phase unwrapping pipeline. The independent per-component time-varying integer gauge is not known from wrapped data. Global mosaic registration, overlapping tile reconciliation, real PS-only comparisons, independent monitor validation and trustworthy physical edge constraints must be implemented and passed before production use. Even a spatial loop with zero integer mismatch is **not proof** of true deformation in the presence of shared aliasing. The max-distance parameter has units based on **nominal radar** spacing rather than measured geodesic separation.

### Install

```bash
cd /mnt/ningbo_process/ds/test-ds-main
conda activate pypsds
mkdir -p /tmp/psds_native_point
unzip -oq /mnt/ningbo_process/ds/native_point_graph_integration.zip -d /tmp/psds_native_point
python /tmp/psds_native_point/install_native_point_graph.py --check --repo .
python /tmp/psds_native_point/install_native_point_graph.py --apply --repo .
python -m pytest -q
git diff --check
git add pypsds/unwrap/native_point_graph.py tests/test_native_point_graph.py docs/NATIVE_POINT_GRAPH.md
git commit -m "feat: add original PS-DS point graph integer candidate branch"
git push origin main
```

### Representative radar ROI: near rows 7667 / cols 14638

The ROI below is a **starting test location** from your past radar-coordinate audit, **not** confirmed to be one of the observed block boundaries. Set a more relevant ROI if you have precise anomaly coordinates. Audit first; use different output directories for each experiment (the module refuses nonempty outputs).

```bash
ROOT=/mnt/ningbo_process/ds
cd "$ROOT/test-ds-main"

OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python -m pypsds.unwrap.native_point_graph \
  --config "$ROOT/pypsds.yaml" \
  --row0 7155 --col0 14125 --height 1024 --width 1024 \
  --subset combined --max-points 400000 \
  --neighbors 8 --max-distance-m 40 \
  --output-dir "$ROOT/output/diagnostics/native_point_graph_roi_audit" \
  --audit-only
```

Then run with a **new** output folder:

```bash
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python -m pypsds.unwrap.native_point_graph \
  --config "$ROOT/pypsds.yaml" \
  --row0 7155 --col0 14125 --height 1024 --width 1024 \
  --subset combined --max-points 400000 \
  --neighbors 8 --max-distance-m 40 --max-step-rad 2.65 \
  --output-dir "$ROOT/output/diagnostics/native_point_graph_roi_candidate"
```

The optional `--baseline-point-phase <full-shape-original-point-unwrapped.npy>` compares existing point-level unwrapped phase against the same native training and heldout integer edge histories. Do not pass the 147,971-node coarse array here: it must be `[26,389,105,75]` and wrapped-congruent with PointPhaseStack. If the exact file is unknown, **omit the flag** and first inspect output manifests.

### Main outputs

* `summary.json`: accepted/rejected edges, component count, two-core two-forest consistency, training and heldout edge closure, PS–PS/DS–DS/PS–DS edge support.
* `original_point_ids.npy`, `original_radar_rows.npy`, `original_radar_cols.npy`, `original_point_type.npy`: original point identity and coordinates.
* `cycles_primary.npy`, `cycles_secondary.npy`: candidate integer time histories for each selected real point; arbitrary reference per component and acquisition.
* `candidate_point_edge_ids.npy`, `candidate_point_edge_cycles.npy`, `candidate_point_edge_train.npy`, `candidate_point_edge_heldout.npy`: original-point graph observation constraints and diagnostic split.
* `local_relative_unwrapped_phase_rad.npy`: per-point wrapped phase + 2π × candidate integers; NOT globally aligned or validated displacement.

### Interpreting results

If the graph fragments into small components or most heldout edges cannot be compared within the same training component, the local phase is underconstrained. That is a **valid result** and must not be “fixed” by invented connections. If the spatial integer candidates agree on heldout edges, run an independent PS-only/GNSS/mapping comparison before accepting the branch, and check localized fault boundaries rather than just global percentages.
