# Production status

Current release: **pyPSDS-GAMMA 1.1.0**

The production pipeline contains **38 package-contained semantic stages**, from `ds_statistics` through `point_products`.

## Release validation

The release gate validates:

- source compilation and source tests;
- non-editable wheel construction;
- isolated wheel installation and package import;
- importability of all 38 stage modules;
- clean-project initialization;
- portable configuration without machine-specific absolute paths;
- the dynamic 38-stage output contract.

## Portability

Project input locations can be explicitly configured or discovered from conventional project layouts. The public project template does not encode a specific study-area filesystem.

Runtime resource planning uses the resources available to the current process rather than assuming one server configuration. CPU planning respects process affinity and detected cgroup CPU quota/cpuset restrictions. Memory planning is bounded by host `MemAvailable` and finite cgroup v1/v2 memory limits when present.

Scientific choices remain explicit. In particular, the geometric/coregistration reference and the deformation reference definition are not silently inferred merely for filesystem portability.

## Reference-case regression

The validated reference case produced:

| Quantity | Value |
|---|---:|
| formal DS | 1,077,566 |
| sequential | 1,075,120 |
| full-SCM fallback | 2,446 |
| phase-linking valid | 1,077,566 |
| final DS with temporal coherence >= 0.80 | 863,969 |

The portability refactor preserved the validated phase-linking arrays exactly for the reference case.

Bitwise identity across arbitrary CPU, BLAS, LAPACK, operating-system, and hardware implementations is not claimed. Scientific reproducibility is enforced through explicit project configuration, fixed scientific defaults, source/wheel validation, stage-output contracts, and numerical/reference regression checks.

Development snapshots, migration backups, and temporary audit trees are not part of the release tree. Git history and external release archives provide development provenance.
