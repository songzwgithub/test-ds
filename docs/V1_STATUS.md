# pyPSDS-GAMMA v1.0 status

## Frozen science

- PS ADI threshold: 0.25
- formal DS domain: geometry-valid non-PS pixels
- Rayleigh GLRT: 11 x 23, alpha = 0.005
- formal minimum SHP: 48
- sequential state minimum SHP: 24
- temporal strategy: sequential
- beta: 0.0
- EMI target: 0.99
- EMI backend: threshold Cholesky
- EVD fallback: enabled
- final DS TC threshold: 0.80
- PS priority: enabled

## Production pipeline

- [x] DS statistics
- [x] conditional phase cache
- [x] exact support cache
- [x] sequential Phase Linking
- [x] full-SCM fallback
- [x] PS/DS selection
- [x] PointPhaseStack
- [x] temporal network
- [x] spatial graph
- [x] spatial unwrapping
- [x] network time-series inversion
- [x] spatial reference

## Large-scene architecture

`PYPSDS_PHASE_SOURCE=auto|gamma|cache` is supported.

With `auto`, an existing corrected-YXT cache is reused. When it does not
exist, canonical Gamma streaming is used and full-YXT construction is
skipped.

Gamma streaming has exact parity with the frozen corrected-YXT Phase Linking
baseline.

Status: **v1.0 numerical and production architecture frozen**.
