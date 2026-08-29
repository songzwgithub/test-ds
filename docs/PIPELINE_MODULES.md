# pyPSDS-GAMMA production modules

Public production workflow:

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

Internal stages remain checkpoint/output-contract units. They are not the primary user-facing workflow interface.

## Module mapping

### data_ps
- ds_statistics
- phase_cache

### shp
- exact_support_cache

### phase_linking
- phase_linking

Covariance/coherence estimation is fused with Phase Linking in bounded memory. No full-scene covariance product is materialized.

### ps_ds
- ds_selection
- ps_finalize
- point_stack

### network_qc
- network_prepare
- network_build
- network_cycle_quality
- network_finalize
- virtual_ifg_quality
- spatial_graph_quality
- spatial_bridge_quality
- spatial_component_quality
- spatial_anchor_quality
- spatial_anchor_summary
- spatial_local_graph_quality
- spatial_graph
- spatial_gradient_quality
- unwrap_policy

### unwrap
- unwrap
- unwrap_severity_quality
- unwrap_conflict_quality
- unwrap_acquisition_quality
- temporal_closure
- temporal_integer_candidate
- temporal_candidate_spatial_quality
- unwrap_signature_quality
- unwrap_finalize

### timeseries
- point_geometry
- residual_ramp
- timeseries_inversion

Residual ramp is estimated in the final unwrapped IFG domain. All final strict PS participate with equal total base weight per 2-km metric cell. Degree-1 IFG slopes are fitted with Huber IRLS and projected onto the connected acquisition network before subtraction. The fitted intercept is not removed.

### corrections
- reference
- atmosphere_correction
- scla
- scn

### products
- final_los
- point_products

## CLI

```bash
pypsds modules

pypsds run \
  --config project.yaml \
  --module phase_linking

pypsds run \
  --config project.yaml \
  --from-module shp \
  --to-module unwrap
```

Internal stage selectors remain available for advanced checkpoint recovery/debugging.

### GAMMA tile prefetch

For GAMMA streaming, the production Phase Linking path uses a bounded
one-ahead tile prefetch pipeline.

While the CPU evaluates exact SHP support, coherence/covariance, EMI,
temporal coherence and compression for the current tile, one background
reader may read and phase-correct the next real-acquisition tile.

The queue depth is intentionally fixed to one:

```text
current tile:
    SHP -> coherence/covariance -> EMI -> compression

in parallel:
    read + phase-correct next GAMMA tile
```

This changes execution scheduling only. The SHP definition, covariance
mathematics, sequential temporal plan, EMI estimator and DS selection
criteria are unchanged.

For mmap/cache-backed phase sources, the existing synchronous path is
retained because the operating-system page cache already provides the
appropriate read-ahead behavior.

### Fused post-Phase-Linking row bands

For GAMMA streaming, the post-Phase-Linking path is physically fused
by row band. One full-date corrected `PhaseTile` is produced and reused
immediately by three consumers:

1. sequential full-span temporal/pair coherence quality;
2. original-support full-SCM fallback;
3. PS linked-phase fill.

The tile is released before the next row band. This avoids repeated
full-date GAMMA phase correction while keeping memory independent of
the total scene size.

The non-GAMMA mmap/cache backend retains the validated legacy
orchestration.

### Phase-Linking prefetch production policy

One-ahead GAMMA Phase-Linking prefetch remains available as an explicit runtime option,
but the production default is disabled (`runtime.phase_link_prefetch_tiles: 0`).
Repeated real-data runs showed non-deterministic stalls in the asynchronous canonical
streaming path after GAMMA subprocess completion. GAMMA timeout protection, resource
budgeting, and post-Phase-Linking row-band fusion remain active.
