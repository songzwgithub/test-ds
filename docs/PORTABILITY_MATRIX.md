# Portability matrix

Release: **pyPSDS-GAMMA 1.1.0**

Production stages: **38**

This matrix validates package portability, project-path discovery, geometry/reference portability, temporal planning across acquisition counts, hardware-aware runtime replanning, wheel installation, the current production dry-run, and the dynamic stage-output contract.

| Check | Status | Detail |
|---|---|---|
| Installed wheel identity | PASS | version=1.1.0, stages=38 |
| Project A conventional auto-discovery | PASS | RSLC / RSLC_tab / DEM_prep |
| Project B alternate auto-discovery | PASS | RSLC_cropped precedence / glob tab / DEM / gacos |
| Project C explicit external-path override | PASS |  |
| Ambiguous RSLC_tab hard failure | PASS |  |
| GACOS disabled without GACOS directory | PASS |  |
| GACOS enabled path discovery | PASS | lowercase gacos directory accepted |
| Geometry A dynamic dimensions/reference/multilook | PASS | date=20200101, geometry=160x120, looks=4:2 |
| Geometry B dynamic dimensions/reference/multilook | PASS | date=20220415, geometry=200x160, looks=4:3 |
| Scientific reference choices remain explicit | PASS |  |
| Variable acquisition-count temporal plans | PASS | 12,19,20,37,38,39,57,83 |
| Machine S runtime adaptation | PASS | cpu=4, RAM=6 GiB, workers=4 |
| Machine M runtime adaptation | PASS | cpu=8, RAM=24 GiB, workers=8 |
| Machine L runtime adaptation | PASS | cpu=32, RAM=64 GiB, workers=32 |
| Machine XL runtime adaptation | PASS | cpu=64, RAM=128 GiB, workers=64 |
| Explicit CPU cap respected | PASS | available=16, requested=3, effective=3 |
| Machine S + Project A contract | PASS |  |
| Machine L + Project A contract | PASS |  |
| Machine L + Project B contract | PASS |  |

## Scope

- The synthetic project cases validate configuration, filesystem discovery, geometry dimensions, geometric reference dates, multilook inference, GACOS path handling, and temporal/runtime planning.
- The existing production project is additionally validated with `config-check`, `plan`, the complete 38-stage dry-run, and the dynamic output-contract gate.
- Synthetic project cases do not claim numerical InSAR equivalence because they intentionally contain no SAR signal.
- Scientific reference selection remains an explicit project decision and is not auto-selected for portability.
- The clean-wheel test isolates the pyPSDS-GAMMA package installation from the source tree; validated dependencies are shared from the host environment.

## Result

PASS: 19 / 19 synthetic portability checks.

Current production dry-run: **PASS**.  Dynamic stage-output contract: **PASS**.  Source test gate: **PASS**.
