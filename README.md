# pyPSDS-GAMMA

pyPSDS-GAMMA is a production PS/DS InSAR workflow for GAMMA RSLC stacks.

## Installation

Python 3.11 or 3.12 is supported. Required GAMMA executables must be in `PATH`.

```bash
python -m pip install .
pypsds --version
```

## Create a project

```bash
pypsds init /path/to/project
cd /path/to/project
```

The generated `pypsds.yaml` uses project-relative paths and contains no study-area coordinates or machine-specific paths. Configure the actual GAMMA co-registration reference date and the study-area radar reference window before production.

```bash
pypsds doctor --config pypsds.yaml
pypsds plan --config pypsds.yaml
pypsds run --config pypsds.yaml --dry-run
pypsds run --config pypsds.yaml
```

## Frozen scientific defaults

- formal DS: geometry-valid non-PS;
- Rayleigh GLRT: 11 x 23, alpha 0.005;
- formal SHP threshold K >= 48;
- sequential-state threshold K >= 24;
- ministack size 19;
- EMI beta 0, target eigenvalue 0.99;
- threshold-Cholesky EMI with EVD fallback;
- exact static support cache;
- final DS temporal coherence >= 0.80;
- PS priority.

Study-area paths, co-registration dates, acquisition counts and reference coordinates are not distributed as production defaults.

See `docs/REPRODUCIBILITY.md`.

A normal wheel contains all 32 production stages. Bitwise identity across every CPU/BLAS implementation is not claimed; numerical reproducibility is validated by explicit configuration and regression gates.
