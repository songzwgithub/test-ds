# pyPSDS-GAMMA v1.1 authoritative migration baseline

These files are the frozen implementations selected for migration into the v1.1 production package.

They are reference sources only and must not be imported directly by the runtime package.

## Production sources

| Component | Frozen source | Canonical migration name |
|---|---|---|
| `point_geometry` | `files/gacos/p15_3a_v4b_finalize__1a252a48fb.py` | `production/geometry/point_geolocation.py` |
| `incidence_geometry` | `files/geometry/p15_3h_fast_incidence_final__14d749c8b9.py` | `production/geometry/incidence.py` |
| `gacos` | `files/gacos/p15_5a_fast_gacos_production__785a545075.py` | `production/corrections/gacos.py` |
| `scla_baseline` | `files/scla/p15_5b3b_regenerate_missing_bases__c943160b85.py` | `production/corrections/scla_baseline.py` |
| `scla_k` | `files/gacos/p15_5b4_stamps_final_pass_k__435ca04ba3.py` | `production/corrections/scla_k.py` |
| `scla_c` | `files/scla/p15_5b5_stamps_final_pass_c__967d439c53.py` | `production/corrections/scla_c.py` |
| `pre_scn` | `files/scla/p15_5b6_materialize_pre_scn__685f3ce84b.py` | `production/corrections/pre_scn.py` |
| `scn` | `files/scn/p15_6b2_stage8_scn_production__df55f1ac3a.py` | `production/corrections/scn.py` |
| `los_timeseries` | `files/los_timeseries/p15_6c_finalize_los_timeseries__42b48c5df8.py` | `production/inversion/los_timeseries.py` |
| `point_metrics` | `files/point_products/p15_7a_final_point_products__d865468fdf.py` | `production/products/point_metrics.py` |
| `point_geocoding` | `files/geocoding/p15_7b2_point_geocoding__f7222964ac.py` | `production/products/point_geocoding.py` |
| `point_packaging` | `files/geocoding/p15_7c_point_packaging__73b9b4ca1e.py` | `production/products/point_packaging.py` |
| `point_attributes` | `files/geocoding/p15_8_point_attribute_enrichment__62e595bd33.py` | `production/products/point_attributes.py` |
| `delivery_qc` | `files/delivery/p15_9_delivery_qc__6147c99c2d.py` | `production/quality/delivery_qc.py` |

## Migration order

1. point geometry and incidence
2. GACOS atmospheric correction
3. SCLA baseline / K / C / pre-SCN
4. SCN
5. final LOS time series
6. point metrics
7. point geocoding and packaging
8. delivery QC

## Explicit exclusions

- `pystamps_stale_sitepackages` — Stale external/site-packages reference implementation; not the final validated pyPSDS production source.
- `benchmark_` — Benchmark only; never a production pipeline implementation.
- `smoke` — Development/smoke validation only.
- `p15_7b2_radar_rasterize` — Rejected science-product route because it forces full-resolution PS/DS points into multilooked radar geometry.
- `cog_rasterizer` — Optional visualization route only; not primary scientific output.

## Rule

A production module may replace a frozen source only after numerical regression against the validated current-project outputs.
