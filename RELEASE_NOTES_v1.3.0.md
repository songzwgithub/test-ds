# pyPSDS-GAMMA v1.3.0

v1.3.0 adds the formal ground-deformation monitoring layer while preserving
the frozen PS/DS, Phase-Linking, and graph-unwrapping scientific core.

The public workflow remains 9 logical modules and 38 internal checkpoint
stages.

## Added

- Conservative feasible weighted-L2 temporal-network inversion.
- Automatic fallback to the exact ordinary-L2 result when strict-network
  residuals are below the numerical floor.
- Per-IFG residual scales, WLS weights, acquisition-phase covariance, and
  formal network-inversion uncertainty.
- Automatic stable relative reference-region selection when no explicit
  point set or radar window is supplied.
- Composite formal velocity uncertainty with temporal-regression,
  network-inversion, and reference-datum components reported separately.
- Annual LOS velocity products.
- Monitoring point CSV, QA JSON, and self-contained QA HTML.
- Optional engineering figures, GeoJSON, Shapefile, and non-interpolated
  GeoTIFF products.
- Optional LOS-to-vertical products using per-point incidence angles.
- `pypsds-decompose` for matched ascending/descending East/Up velocity
  decomposition.

## Scientific safeguards

The installer records and verifies SHA256 identities for the frozen
Phase-Linking and graph-unwrapping source files.

Weighted inversion uses global per-IFG residual scales from the strict domain.
When the median residual is at or below the configured numerical floor, the
ordinary-L2 acquisition phase is preserved exactly.

The reported formal monitoring uncertainty is not a complete bound on
systematic geodetic error. Correlated atmospheric, orbital, geocoding, and
deformation-model errors can remain.

Automatic reference selection establishes a relative InSAR datum; it cannot
prove zero physical deformation.

LOS-to-vertical products assume negligible horizontal deformation.
Two-track East/Up decomposition assumes the North component is zero.
