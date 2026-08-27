## 1.2.0

- Made one-ahead GAMMA Phase-Linking prefetch opt-in (production default off); timeout protection, resource budgeting, and post-PL fusion remain active.

- Fused GAMMA post-Phase-Linking row-band processing so full-span quality, full-SCM fallback, and PS phase fill share one full-date corrected PhaseTile per band instead of repeating phase-source reads.

- Added bounded one-ahead GAMMA tile prefetch for sequential Phase Linking, overlapping next-tile I/O/phase correction with current SHP/coherence/EMI/compression while keeping at most one extra real-acquisition tile in memory.

- Added the nine-module production workflow interface (`data_ps`, `shp`, `phase_linking`, `ps_ds`, `network_qc`, `unwrap`, `timeseries`, `corrections`, `products`) while retaining internal stage checkpoints.

- Added bounded real-data runtime autotuning for sequential threshold-Cholesky EMI.
- Runtime tuning selects Phase Linking worker/chunk/batch scheduling without changing scientific parameters.
- Runtime profiles are hardware/BLAS/solver-dimension keyed and clamped by the cgroup-aware safe resource plan.
- Production runs automatically create or reuse the runtime profile before Phase Linking.
- Added the public `pypsds tune --config ...` command.
- Removed stale hard-coded pipeline/run-manifest version strings.
- The validated DS scientific defaults remain unchanged.

# Changelog

## 1.1.0

- Added the validated public tail stages: atmospheric correction, SCLA, SCN, final LOS, and point products.
- Promoted the production pipeline from 33 to 38 package-contained stages.
- Kept atmospheric correction optional with a canonical corrected-phase ownership contract.
- Migrated the validated dynamic SCLA/SCN/final-LOS/point-metric runtime adapters without changing the frozen numerical cores.
- Added full-resolution point products as the primary scientific delivery format.
- Added permanent v1.1 stage-contract, wheel-content, and installed-package release gates.
- Updated the public package version to 1.1.0.

## 1.0.0

- Started the formal pyPSDS-GAMMA production release.
- Removed version-specific names from the active Python package.
- Established schema_version=1 configuration.
- Added the `pypsds` command-line interface.
- Added automatic CPU/RAM runtime planning.
- Added stage-signature and manifest infrastructure.
- Promoted validated high-performance GLRT/coherence/robust-EMI kernels.
- Established mmap-first canonical array I/O.
- Preserved the frozen pre-v1 implementation only as development parity
  material.
- SCLA remains disabled by default pending positive held-out validation.

- Finalized sequential ministack Phase Linking with K>=24 state continuity and K>=48 formal DS eligibility.
- Added threshold-Cholesky EMI with conservative fallback to the validated eigendecomposition implementation.
- Added exact packed Rayleigh-GLRT support caching.
- Added crash-safe sequential tile checkpoints and Step04 completion fast-resume.
- Added canonical Gamma streaming for sequential Phase Linking.
- Made the full corrected-YXT cache optional for sequential production through `PYPSDS_PHASE_SOURCE=auto|gamma|cache`.
- Preserved the corrected-YXT cache path for explicit cache mode and legacy `full_scm`.
- Replaced experiment-step script prefixes with semantic production command names.
- Packaged all production stages inside `pypsds` for normal wheel installs.
- Removed machine/study-area paths, fixed acquisition counts and reference coordinates from distributed configuration.
- Added `pypsds init`, hardware/acquisition-aware planning, portable release gates and installed-wheel smoke tests.
- Removed current-branch historical development trees; Git history remains the provenance archive.
