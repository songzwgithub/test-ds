# pyPSDS-GAMMA v1.1 Geometry upstream producer audit

The currently frozen point-geolocation implementation is a finalizer: it consumes pre-existing geometry binaries. This audit searches the migration snapshot for their upstream producers.

## Authoritative Geometry file literals

### point_geolocation

| Line | Literal | File tokens |
|---:|---|---|
| 11 | `/home/ubuntu/Downloads/RSLC/20151212.rslc.par` | `20151212.rslc.par` |
| 351 | `gacos_geometry_manifest.json` | `gacos_geometry_manifest.json` |
| 39 | `rows.npy` | `rows.npy` |
| 44 | `cols.npy` | `cols.npy` |
| 256 | `strict_point_ids.npy` | `strict_point_ids.npy` |
| 261 | `radar_row.npy` | `radar_row.npy` |
| 266 | `radar_col.npy` | `radar_col.npy` |
| 271 | `longitude_deg.npy` | `longitude_deg.npy` |
| 276 | `latitude_deg.npy` | `latitude_deg.npy` |
| 281 | `incidence_angle_stamps_deg.npy` | `incidence_angle_stamps_deg.npy` |
| 286 | `valid_gacos_geometry_mask.npy` | `valid_gacos_geometry_mask.npy` |
| 33 | `strict_point_ids.npy` | `strict_point_ids.npy` |

### incidence

| Line | Literal | File tokens |
|---:|---|---|
| 15 | `/home/ubuntu/Downloads/RSLC/20151212.rslc.par` | `20151212.rslc.par` |
| 20 | `incidence_gamma_compatible_fast_rad.npy` | `incidence_gamma_compatible_fast_rad.npy` |
| 25 | `fast_incidence_manifest.json` | `fast_incidence_manifest.json` |
| 90 | `longitude_deg.npy` | `longitude_deg.npy` |
| 98 | `latitude_deg.npy` | `latitude_deg.npy` |
| 106 | `radar_row.npy` | `radar_row.npy` |

## Highest-ranked upstream candidates

| Rank | Score | Candidate | Writes | GAMMA | Penalties |
|---:|---:|---|---|---|---|
| 1 | 271 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3a_v3_stamps_point_geometry__8eae26b2b5.py` | `np.save×7, .tofile(×1, write_text(×1` | `data2pt×14, incidence_angle×3` | `smoke` |
| 2 | 268 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3a_v4__717f6a84ad.py` | `np.save×7, .tofile(×1, write_text(×1` | `data2pt×12, incidence_angle×2` | `smoke` |
| 3 | 193 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/scn/10a5c_batch_context_repeatability_v09__2acd2cada8.py` | `np.save×4, .tofile(×4, write_text(×1, open(×2` | `data2pt×14` | `audit` |
| 4 | 174 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3a_build_radar_point_geolocation__74f26bba1f.sh` | `np.save×9, np.memmap×3, write_text(×4, open(×1` | `gc_map×10, incidence_angle×4` | `smoke, audit` |
| 5 | 159 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3a_v4b_finalize__1a252a48fb.py` | `np.save×7, write_text(×1` | `data2pt×2, incidence_angle×2` | `smoke` |
| 6 | 140 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3a_v2_build_radar_point_geolocation__a8eab49a3b.sh` | `np.save×9, np.memmap×3, write_text(×4, open(×1` | `gc_map×7, incidence_angle×2` | `smoke, audit` |
| 7 | 132 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3a_v2_build_radar_point_geolocation__13c1cf02e8.py` | `np.save×9, np.memmap×3, write_text(×4, open(×1` | `gc_map×5, incidence_angle×2` | `smoke, audit` |
| 8 | 123 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3e_gamma_truth__f3f0131084.sh` | `` | `data2pt×5, gc_map×5, gc_map2×5, geocode×5` | `` |
| 9 | 89 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/10r3_reuse_pystamps_geometry_bperp_v09__9cdec4e691.py` | `np.save×10, np.memmap×1, write_text(×2, open(×1` | `` | `` |
| 10 | 72 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/10r5a1_fix_stage7_boundary_xy_v09__ac207b122b.py` | `np.save×4, write_text(×2` | `` | `audit` |
| 11 | 70 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/10r3b_raw_grid_geometry_base_coverage_audit__0e818c0bf7.py` | `np.save×6, write_text(×3, open(×1` | `` | `audit` |
| 12 | 67 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/p15_3h_fast_incidence_final__14d749c8b9.py` | `np.save×1, write_text(×1` | `data2pt×1, gc_map×1, gc_map2×1, geocode×1` | `benchmark` |
| 13 | 57 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/scn/gamma_stage1__da5b056b83.py` | `np.save×1, write_text(×8` | `incidence_angle×9` | `` |
| 14 | 50 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/scla/10r4a_stage7_bridge_contract_preflight_v09__114ceace0b.py` | `np.save×3, write_text(×1, open(×1` | `` | `` |
| 15 | 47 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/scla/10r4b_build_pystamps_stage7_bridge_v09__8e2ff9b795.py` | `np.memmap×3, write_text(×1, open(×1` | `` | `` |
| 16 | 44 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_4_fast_gacos_smoke__988d8de739.py` | `np.save×5, write_text(×1, open(×1` | `` | `benchmark, smoke, audit` |
| 17 | 37 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/point_products/p15_7b2_radar_rasterize__b0f17431ac.py` | `.tofile(×3, open(×1` | `` | `` |
| 18 | 36 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/point_products/apply_reference__0ce08ccedd.py` | `np.save×5, write_text(×1, open(×1` | `` | `` |
| 19 | 36 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/point_products/apply_reference__aa7c691444.py` | `np.save×5, write_text(×1, open(×1` | `` | `` |
| 20 | 36 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/scn/benchmark_phase_sim_parallel__da3f347760.py` | `.tofile(×3, write_text(×1, open(×1` | `data2pt×3` | `benchmark` |
| 21 | 34 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/p15_3g_incidence_definition__7866f415c2.py` | `` | `` | `` |
| 22 | 32 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geocoding/p15_8_point_attribute_enrichment__62e595bd33.py` | `open(×1` | `` | `` |
| 23 | 32 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geocoding/run_p15_8_point_attribute_enrichment__c847ca01b6.sh` | `open(×1` | `` | `` |
| 24 | 32 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/p15_3d_fast_incidence__507edab363.py` | `np.save×1` | `look_vector×1` | `benchmark` |
| 25 | 31 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/run_v11_migration_source_freeze__5dfd2b016d.sh` | `write_text(×3, open(×1` | `` | `smoke` |
| 26 | 30 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/point_products/09c_apply_reference_region_v09__54d3ec05e0.py` | `np.save×5, write_text(×1, open(×1` | `` | `audit` |
| 27 | 29 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p14_final_product_audit__cb7eb52310.sh` | `write_text(×2, open(×1` | `` | `audit` |
| 28 | 27 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geocoding/p15_7b2_point_geocoding__f7222964ac.py` | `open(×1` | `geocode×1` | `` |
| 29 | 27 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geocoding/run_p15_7b2_point_geocoding__a14c54d54a.sh` | `open(×1` | `geocode×1` | `` |
| 30 | 26 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/p15_5b3_stamps_baseline_source_audit__d63c14895e.py` | `write_text(×1` | `` | `audit` |
| 31 | 26 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/p15_7b0_geocoding_contract_audit__6b9b655473.py` | `write_text(×1` | `` | `audit` |
| 32 | 24 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/p15_3f_zero_doppler_fast__becab4f81a.py` | `np.save×2` | `` | `benchmark, audit` |
| 33 | 23 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_2_scla_residual_dem_audit__7ce3663ee4.sh` | `write_text(×2` | `incidence_angle×1` | `audit` |
| 34 | 23 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/p15_6a_stamps_xy_grid_audit__ee51f32f9b.py` | `np.save×2, write_text(×1` | `` | `audit` |
| 35 | 22 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/scn/p15_6b0_exact_xy_neighbor_census__d6529e8b88.py` | `np.save×2, write_text(×1` | `` | `benchmark, audit` |
| 36 | 21 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_5c_scla_spatial_regularization__72bf99005f.py` | `np.save×1, write_text(×1, open(×1` | `incidence_angle×1` | `smoke, audit` |
| 37 | 21 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/point_products/pipeline__02cd1f5b68.py` | `write_text(×2, open(×3` | `` | `benchmark` |
| 38 | 21 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/point_products/pipeline__2f826aeb77.py` | `write_text(×2, open(×3` | `` | `benchmark` |
| 39 | 21 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/point_products/pipeline__70e9578d45.py` | `write_text(×2, open(×3` | `` | `benchmark` |
| 40 | 21 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/point_products/pipeline_before_p8g2_final_20260820_164604__f9ea1aa749.py` | `write_text(×2, open(×3` | `` | `benchmark` |

## data2pt candidates

| Score | Candidate | Exact file hits |
|---:|---|---|
| 271 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3a_v3_stamps_point_geometry__8eae26b2b5.py` | `20151212.rslc.par×1, cols.npy×1, gacos_geometry_manifest.json×1, latitude_deg.npy×1, longitude_deg.npy×1, radar_col.npy×1, radar_row.npy×1, rows.npy×1, strict_point_ids.npy×2, valid_gacos_geometry_mask.npy×1` |
| 268 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3a_v4__717f6a84ad.py` | `20151212.rslc.par×1, cols.npy×1, gacos_geometry_manifest.json×1, latitude_deg.npy×1, longitude_deg.npy×1, radar_col.npy×1, radar_row.npy×1, rows.npy×1, strict_point_ids.npy×2, valid_gacos_geometry_mask.npy×1` |
| 193 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/scn/10a5c_batch_context_repeatability_v09__2acd2cada8.py` | `cols.npy×1, rows.npy×1, strict_point_ids.npy×2` |
| 159 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3a_v4b_finalize__1a252a48fb.py` | `20151212.rslc.par×1, cols.npy×1, gacos_geometry_manifest.json×1, incidence_angle_stamps_deg.npy×1, latitude_deg.npy×1, longitude_deg.npy×1, radar_col.npy×1, radar_row.npy×1, rows.npy×1, strict_point_ids.npy×2, valid_gacos_geometry_mask.npy×1` |
| 123 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/gacos/p15_3e_gamma_truth__f3f0131084.sh` | `20151212.rslc.par×1` |
| 67 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/geometry/p15_3h_fast_incidence_final__14d749c8b9.py` | `20151212.rslc.par×1, fast_incidence_manifest.json×1, incidence_gamma_compatible_fast_rad.npy×1, latitude_deg.npy×1, longitude_deg.npy×1, radar_row.npy×1` |
| 36 | `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/migration_sources/snapshot_20260825_094037/files/scn/benchmark_phase_sim_parallel__da3f347760.py` | `` |

## Required decision

Before Geometry is migrated into the runtime package, the complete producer chain for longitude/latitude (and any required point height) must be identified and frozen. A finalizer alone is not portable to a new study area.
