# Changelog

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
