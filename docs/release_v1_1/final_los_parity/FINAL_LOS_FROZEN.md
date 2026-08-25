# Final LOS v1.1 Frozen Contract

Status: FROZEN

Production domain:

- points: 881315
- acquisitions: 38
- temporal reference: 20141006
- geometric master: 20151212
- spatial reference: median of 607 frozen reference points
- LOS positive direction: toward satellite
- wavelength: 0.055465759531382094 m
- velocity year: 365.25 days

Final time-series products:

- acquisition_phase_final_rad.npy
  - 881315 x 38
  - float32

- los_displacement_toward_satellite_m.npy
  - 881315 x 38
  - float32

- los_displacement_toward_satellite_mm.npy
  - 881315 x 38
  - float32

Final point products:

- los_velocity_toward_satellite_mm_per_year.npy
  - 881315
  - float32

- los_cumulative_toward_satellite_mm.npy
  - 881315
  - float32

- linear_residual_rms_mm.npy
  - 881315
  - float32

- velocity_slope_standard_error_mm_per_year.npy
  - 881315
  - float32

Validation:

- authoritative vs migrated final phase: byte exact
- authoritative vs migrated LOS m: byte exact
- authoritative vs migrated LOS mm: byte exact
- authoritative vs migrated velocity: byte exact
- authoritative vs migrated cumulative: byte exact
- authoritative vs migrated residual RMS: byte exact
- authoritative vs migrated velocity SE: byte exact
- time_axis_contract.npz: all arrays exact
- scientific manifest fields: exact

Geometry owner:

- processing/point_geometry

Upstream frozen:

- Geometry
- GACOS
- SCLA
- pre-SCN
- SCN

Pipeline remains unchanged at 33 stages.
