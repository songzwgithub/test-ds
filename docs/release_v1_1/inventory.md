# pyPSDS-GAMMA v1.1 release inventory

- Repository: `/home/ubuntu/software/pyPSDS-GAMMA-v1.0`
- Project name: `pypsds-gamma`
- Current version: `1.0.0`
- Python requirement: `>=3.11`
- Text/source files audited: **177**
- Development-name hits: **112**
- Absolute-path hits: **0**
- Project-specific path hits: **0**

## Current CLI entry points

| Command | Python entry |
|---|---|
| `pypsds` | `pypsds.cli:main` |

## Suggested functional domains

| Domain | Files |
|---|---:|
| `inversion` | 1 |
| `network` | 19 |
| `phase_linking` | 21 |
| `preparation` | 14 |
| `products` | 1 |
| `quality` | 2 |
| `selection` | 119 |

## Source/module inventory

| File | Lines | Suggested domain | Dev hits | Hard paths |
|---|---:|---|---:|---:|
| `.github/workflows/ci.yml` | 23 | `selection` | 1 | 0 |
| `.p11a_backup_20260823_182743/pypsds/cli.py` | 176 | `preparation` | 0 | 0 |
| `.p11a_backup_20260823_182743/pypsds/phase_linking/sequential_production.py` | 3422 | `phase_linking` | 2 | 0 |
| `.p11a_backup_20260823_182743/pypsds/pipeline.py` | 1865 | `quality` | 0 | 0 |
| `.p11a_backup_20260823_182743/pypsds/runtime.py` | 242 | `phase_linking` | 0 | 0 |
| `.p11a_backup_20260823_182743/pypsds/stages/run_phase_linking.py` | 1274 | `selection` | 1 | 0 |
| `.p11b1_backup_20260823_185416/pypsds/gamma/phase_correction.py` | 939 | `selection` | 1 | 0 |
| `.p11b1_backup_20260823_185416/pypsds/phase_linking/phase_source.py` | 1199 | `selection` | 0 | 0 |
| `.p11b1_backup_20260823_185416/pypsds/stages/run_phase_linking.py` | 1324 | `selection` | 1 | 0 |
| `.p11b3_backup_20260823_190806/pypsds/gamma/phase_correction.py` | 1364 | `selection` | 2 | 0 |
| `.p11b3_backup_20260823_190806/pypsds/stages/run_phase_linking.py` | 1357 | `selection` | 2 | 0 |
| `.p11b3v2_backup_20260823_191338/pypsds/gamma/phase_correction.py` | 1364 | `selection` | 2 | 0 |
| `.p11b3v2_backup_20260823_191338/pypsds/stages/run_phase_linking.py` | 1357 | `selection` | 2 | 0 |
| `.p11d1_backup_20260823_194422/pypsds/phase_linking/emi_threshold.py` | 957 | `phase_linking` | 0 | 0 |
| `.p11d4_backup_20260823_214228/pypsds/runtime.py` | 423 | `phase_linking` | 4 | 0 |
| `.p11e3_backup_20260823_221506/pypsds/phase_linking/emi_threshold.py` | 957 | `phase_linking` | 0 | 0 |
| `.p11f1_backup_20260823_224042/pypsds/phase_linking/emi.py` | 280 | `phase_linking` | 0 | 0 |
| `.p12_1a_backup_20260824_082243/pypsds/stages/build_exact_support_cache.py` | 1187 | `selection` | 0 | 0 |
| `.p12_1a_backup_20260824_082402/pypsds/stages/build_exact_support_cache.py` | 1187 | `selection` | 0 | 0 |
| `.p12_1a_backup_20260824_082534/pypsds/stages/build_exact_support_cache.py` | 1187 | `selection` | 0 | 0 |
| `.p13_sg2_backup_20260824_095330/pypsds/stages/build_spatial_graph.py` | 856 | `network` | 0 | 0 |
| `CHANGELOG.md` | 28 | `selection` | 2 | 0 |
| `README.md` | 67 | `selection` | 2 | 0 |
| `TEST_STATUS.txt` | 16 | `selection` | 1 | 0 |
| `THIRD_PARTY_NOTICES.md` | 13 | `selection` | 0 | 0 |
| `config/pypsds.yaml` | 110 | `selection` | 0 | 0 |
| `config/pypsds_template.yaml` | 110 | `selection` | 0 | 0 |
| `docs/DS_PRODUCTION_FREEZE_P9.md` | 131 | `selection` | 12 | 0 |
| `docs/PRODUCTION_STATUS.md` | 7 | `products` | 1 | 0 |
| `docs/REPRODUCIBILITY.md` | 19 | `selection` | 0 | 0 |
| `docs/release/README.md` | 3 | `selection` | 0 | 0 |
| `install.sh` | 16 | `selection` | 0 | 0 |
| `pyproject.toml` | 29 | `selection` | 0 | 0 |
| `pypsds/__init__.py` | 10 | `selection` | 0 | 0 |
| `pypsds/cli.py` | 228 | `preparation` | 0 | 0 |
| `pypsds/config.py` | 85 | `selection` | 0 | 0 |
| `pypsds/context.py` | 164 | `preparation` | 0 | 0 |
| `pypsds/corrections/__init__.py` | 1 | `selection` | 0 | 0 |
| `pypsds/filtering/__init__.py` | 5 | `selection` | 0 | 0 |
| `pypsds/filtering/goldstein.py` | 371 | `selection` | 0 | 0 |
| `pypsds/gamma/__init__.py` | 25 | `preparation` | 0 | 0 |
| `pypsds/gamma/binary.py` | 146 | `preparation` | 0 | 0 |
| `pypsds/gamma/geometry.py` | 72 | `preparation` | 0 | 0 |
| `pypsds/gamma/par.py` | 118 | `preparation` | 0 | 0 |
| `pypsds/gamma/phase_correction.py` | 1874 | `selection` | 4 | 0 |
| `pypsds/gamma/stack.py` | 69 | `preparation` | 0 | 0 |
| `pypsds/io/__init__.py` | 11 | `selection` | 0 | 0 |
| `pypsds/io/arrays.py` | 87 | `selection` | 0 | 0 |
| `pypsds/manifest.py` | 98 | `selection` | 0 | 0 |
| `pypsds/network/__init__.py` | 1 | `selection` | 0 | 0 |
| `pypsds/phase_linking/__init__.py` | 25 | `phase_linking` | 0 | 0 |
| `pypsds/phase_linking/coherence.py` | 77 | `preparation` | 0 | 0 |
| `pypsds/phase_linking/compression.py` | 379 | `preparation` | 0 | 0 |
| `pypsds/phase_linking/emi.py` | 280 | `phase_linking` | 0 | 0 |
| `pypsds/phase_linking/emi_fast.py` | 696 | `phase_linking` | 0 | 0 |
| `pypsds/phase_linking/emi_threshold.py` | 957 | `phase_linking` | 0 | 0 |
| `pypsds/phase_linking/full_scm_points.py` | 617 | `phase_linking` | 0 | 0 |
| `pypsds/phase_linking/fullspan_quality.py` | 647 | `selection` | 0 | 0 |
| `pypsds/phase_linking/phase_source.py` | 1326 | `selection` | 1 | 0 |
| `pypsds/phase_linking/reliability_qa.py` | 920 | `selection` | 0 | 0 |
| `pypsds/phase_linking/sequential_multistage.py` | 2851 | `selection` | 1 | 0 |
| `pypsds/phase_linking/sequential_phase_writer.py` | 358 | `phase_linking` | 0 | 0 |
| `pypsds/phase_linking/sequential_plan_executor.py` | 482 | `selection` | 0 | 0 |
| `pypsds/phase_linking/sequential_production.py` | 3459 | `phase_linking` | 2 | 0 |
| `pypsds/phase_linking/sequential_routing.py` | 287 | `selection` | 0 | 0 |
| `pypsds/phase_linking/shp_coherence_bitset.py` | 500 | `selection` | 0 | 0 |
| `pypsds/phase_linking/shp_exact_packed.py` | 531 | `selection` | 0 | 0 |
| `pypsds/phase_linking/shp_policy.py` | 186 | `selection` | 0 | 0 |
| `pypsds/phase_linking/shp_vectorized_exact.py` | 359 | `selection` | 0 | 0 |
| `pypsds/phase_linking/state_domain.py` | 584 | `selection` | 0 | 0 |
| `pypsds/phase_linking/streaming_quality.py` | 217 | `phase_linking` | 0 | 0 |
| `pypsds/phase_linking/support_cache.py` | 817 | `selection` | 0 | 0 |
| `pypsds/phase_linking/temporal_plan.py` | 769 | `selection` | 0 | 0 |
| `pypsds/pipeline.py` | 1967 | `quality` | 0 | 0 |
| `pypsds/points/__init__.py` | 13 | `selection` | 0 | 0 |
| `pypsds/points/stack.py` | 93 | `selection` | 0 | 0 |
| `pypsds/progress.py` | 667 | `selection` | 0 | 0 |
| `pypsds/project.py` | 180 | `preparation` | 0 | 0 |
| `pypsds/quality/__init__.py` | 1 | `selection` | 0 | 0 |
| `pypsds/resources/__init__.py` | 1 | `selection` | 0 | 0 |
| `pypsds/resources/default_config.yaml` | 108 | `selection` | 0 | 0 |
| `pypsds/runtime.py` | 473 | `phase_linking` | 5 | 0 |
| `pypsds/selection/__init__.py` | 21 | `selection` | 0 | 0 |
| `pypsds/selection/center_prior.py` | 120 | `preparation` | 0 | 0 |
| `pypsds/selection/ps.py` | 41 | `selection` | 0 | 0 |
| `pypsds/selection/shp.py` | 70 | `selection` | 0 | 0 |
| `pypsds/stages/__init__.py` | 1 | `selection` | 0 | 0 |
| `pypsds/stages/apply_reference.py` | 1043 | `selection` | 3 | 0 |
| `pypsds/stages/assess_local_spatial_graph.py` | 900 | `selection` | 0 | 0 |
| `pypsds/stages/assess_network_cycle_quality.py` | 906 | `network` | 0 | 0 |
| `pypsds/stages/assess_spatial_anchor_quality.py` | 788 | `network` | 3 | 0 |
| `pypsds/stages/assess_spatial_bridge_quality.py` | 596 | `network` | 0 | 0 |
| `pypsds/stages/assess_spatial_components.py` | 739 | `selection` | 3 | 0 |
| `pypsds/stages/assess_spatial_graph_quality.py` | 736 | `selection` | 0 | 0 |
| `pypsds/stages/assess_spatial_phase_gradient.py` | 900 | `network` | 9 | 0 |
| `pypsds/stages/assess_temporal_integer_closure.py` | 1450 | `network` | 0 | 0 |
| `pypsds/stages/assess_unwrap_acquisition_quality.py` | 958 | `selection` | 0 | 0 |
| `pypsds/stages/assess_unwrap_conflicts.py` | 1117 | `network` | 1 | 0 |
| `pypsds/stages/assess_unwrap_severity.py` | 1218 | `selection` | 0 | 0 |
| `pypsds/stages/assess_unwrap_signature_feasibility.py` | 1239 | `selection` | 0 | 0 |
| `pypsds/stages/assess_virtual_ifg_quality.py` | 1109 | `network` | 1 | 0 |
| `pypsds/stages/benchmark_adaptive_ifg_filter.py` | 1461 | `network` | 7 | 0 |
| `pypsds/stages/build_ds_statistics.py` | 1346 | `selection` | 0 | 0 |
| `pypsds/stages/build_exact_support_cache.py` | 1604 | `selection` | 1 | 0 |
| `pypsds/stages/build_moraine_center_prior.py` | 113 | `selection` | 0 | 0 |
| `pypsds/stages/build_phase_cache.py` | 122 | `selection` | 0 | 0 |
| `pypsds/stages/build_phase_linking_reliability_qa.py` | 994 | `selection` | 1 | 0 |
| `pypsds/stages/build_point_phase_stack.py` | 1013 | `selection` | 0 | 0 |
| `pypsds/stages/build_safe_fragment_integer_quality.py` | 1698 | `network` | 0 | 0 |
| `pypsds/stages/build_spatial_graph.py` | 1044 | `selection` | 0 | 0 |
| `pypsds/stages/build_temporal_integer_candidates.py` | 1518 | `network` | 0 | 0 |
| `pypsds/stages/build_temporal_network.py` | 795 | `selection` | 0 | 0 |
| `pypsds/stages/finalize_ps_geometry.py` | 437 | `selection` | 0 | 0 |
| `pypsds/stages/finalize_single_ifg_solution.py` | 1153 | `network` | 0 | 0 |
| `pypsds/stages/finalize_temporal_network.py` | 854 | `network` | 0 | 0 |
| `pypsds/stages/finalize_unwrap_policy.py` | 701 | `network` | 0 | 0 |
| `pypsds/stages/finalize_unwrap_solution.py` | 1245 | `network` | 0 | 0 |
| `pypsds/stages/invert_timeseries.py` | 1189 | `network` | 3 | 0 |
| `pypsds/stages/prepare_temporal_network.py` | 2093 | `preparation` | 1 | 0 |
| `pypsds/stages/run_ds_tiled.py` | 2755 | `selection` | 4 | 0 |
| `pypsds/stages/run_phase_linking.py` | 1365 | `selection` | 2 | 0 |
| `pypsds/stages/select_ds.py` | 57 | `selection` | 0 | 0 |
| `pypsds/stages/summarize_spatial_anchor_quality.py` | 194 | `network` | 0 | 0 |
| `pypsds/stages/unwrap_all_ifgs.py` | 1404 | `network` | 2 | 0 |
| `pypsds/stages/validate_temporal_integer_candidates.py` | 1630 | `selection` | 0 | 0 |
| `pypsds/timeseries/__init__.py` | 1 | `selection` | 0 | 0 |
| `pypsds/unwrap/__init__.py` | 1 | `selection` | 0 | 0 |
| `pypsds_gamma.egg-info/SOURCES.txt` | 126 | `selection` | 0 | 0 |
| `pypsds_gamma.egg-info/dependency_links.txt` | 1 | `selection` | 0 | 0 |
| `pypsds_gamma.egg-info/entry_points.txt` | 2 | `selection` | 0 | 0 |
| `pypsds_gamma.egg-info/requires.txt` | 9 | `selection` | 0 | 0 |
| `pypsds_gamma.egg-info/top_level.txt` | 1 | `selection` | 0 | 0 |
| `tests/test_adaptive_filter_production_decision.py` | 61 | `selection` | 0 | 0 |
| `tests/test_compression.py` | 198 | `preparation` | 0 | 0 |
| `tests/test_ds_production_freeze_p9.py` | 247 | `selection` | 0 | 0 |
| `tests/test_emi_threshold.py` | 175 | `phase_linking` | 0 | 0 |
| `tests/test_full_scm_points.py` | 416 | `selection` | 0 | 0 |
| `tests/test_fullspan_quality.py` | 365 | `selection` | 0 | 0 |
| `tests/test_goldstein_filter.py` | 212 | `selection` | 0 | 0 |
| `tests/test_phase_linking_reliability_qa.py` | 202 | `phase_linking` | 0 | 0 |
| `tests/test_pipeline_sequential_dispatch.py` | 225 | `selection` | 0 | 0 |
| `tests/test_portable_distribution.py` | 38 | `selection` | 0 | 0 |
| `tests/test_production_config.py` | 142 | `selection` | 0 | 0 |
| `tests/test_production_core.py` | 186 | `phase_linking` | 0 | 0 |
| `tests/test_reference_point_ids_dispatch.py` | 66 | `selection` | 0 | 0 |
| `tests/test_sequential_phase_sink.py` | 164 | `selection` | 0 | 0 |
| `tests/test_sequential_phase_writer.py` | 289 | `inversion` | 0 | 0 |
| `tests/test_sequential_plan_executor.py` | 493 | `selection` | 0 | 0 |
| `tests/test_sequential_routing.py` | 176 | `selection` | 0 | 0 |
| `tests/test_sequential_stage_writer_integration.py` | 558 | `phase_linking` | 0 | 0 |
| `tests/test_shp_solver_aware_policy.py` | 93 | `selection` | 0 | 0 |
| `tests/test_state_domain.py` | 116 | `selection` | 0 | 0 |
| `tests/test_support_cache.py` | 70 | `selection` | 0 | 0 |
| `tests/test_temporal_plan.py` | 234 | `selection` | 0 | 0 |
| `tools/analyze_temporal_scaling.py` | 273 | `selection` | 0 | 0 |
| `tools/audit_compression_state_drop_impact.py` | 1203 | `selection` | 3 | 0 |
| `tools/audit_compression_state_fixed_point.py` | 734 | `selection` | 2 | 0 |
| `tools/audit_dense_compression_center_eligibility.py` | 989 | `selection` | 1 | 0 |
| `tools/audit_k24_core_phase_stability.py` | 1294 | `phase_linking` | 1 | 0 |
| `tools/audit_local_state_rescue_threshold.py` | 1391 | `selection` | 2 | 0 |
| `tools/audit_lowk_state_phase_stability.py` | 1317 | `phase_linking` | 1 | 0 |
| `tools/audit_phase_correction_tiling.py` | 728 | `selection` | 0 | 0 |
| `tools/audit_sequential_compression_coverage.py` | 595 | `selection` | 0 | 0 |
| `tools/audit_sequential_phase_coverage.py` | 599 | `selection` | 1 | 0 |
| `tools/audit_stage_contracts.py` | 229 | `selection` | 0 | 0 |
| `tools/audit_temporal_planner.py` | 388 | `selection` | 0 | 0 |
| `tools/benchmark_emi_backends.py` | 1818 | `selection` | 3 | 0 |
| `tools/benchmark_exact_support_cache.py` | 1306 | `selection` | 2 | 0 |
| `tools/benchmark_phase_sim_parallel.py` | 897 | `selection` | 1 | 0 |
| `tools/benchmark_shp_backends.py` | 1038 | `selection` | 2 | 0 |
| `tools/benchmark_shp_vectorized_exact.py` | 579 | `selection` | 0 | 0 |
| `tools/build_stage_contract_inventory.py` | 1777 | `selection` | 0 | 0 |
| `tools/export_contract_review_bundle.py` | 240 | `selection` | 0 | 0 |
| `tools/freeze_stage_output_contracts.py` | 805 | `network` | 0 | 0 |
| `tools/release_gate.py` | 90 | `selection` | 1 | 0 |
| `tools/tune_canonical_phase_parallel.py` | 772 | `selection` | 1 | 0 |
| `tools/tune_ds_emi_cpu.py` | 747 | `selection` | 1 | 0 |

## Generated/build artifacts found at repository root

| Path | Type | Recommended release action |
|---|---|---|
| `build` | directory | `exclude_from_release` |
| `pypsds_gamma.egg-info` | directory | `exclude_from_release` |
| `SLC2pt.log` | file | `exclude_from_release` |
| `TEST_STATUS.txt` | file | `review_or_move_to_docs` |

## v1.1 naming recommendation

Do **not** introduce a second numeric stage hierarchy such as `01_`, `02_`, ... at the package/module level.

Recommended public functional domains:

- `preparation` — input discovery, radar/DEM geometry
- `selection` — PS/DS candidate and SHP selection
- `phase_linking` — covariance/EMI/phase optimization
- `network` — interferometric network and graph operations
- `unwrapping` — phase unwrapping
- `correction` — GACOS, SCLA/residual DEM, SCN
- `inversion` — acquisition phase, time series, LOS/velocity
- `products` — point products, Parquet, GeoPackage, quicklooks
- `quality` — validation, manifests, diagnostics

Existing internal stage identifiers may remain where they are part of the computational dependency graph, but development identifiers such as `P15-6B2`, `smoke`, `benchmark`, and prototype-version names should not appear in the public CLI, configuration schema, or final output names.

## Next migration gate

Before moving any file, define a v1.1 configuration schema and a regression baseline from the current frozen production run. Every migrated module must reproduce that baseline before the old implementation is retired.

