# pyPSDS-GAMMA v1.2.0

## Release scope

v1.2.0 exposes 9 public logical modules while preserving all 38 internal
checkpoint/output-contract stages and the frozen 108 output-contract edges.

Public modules:

1. data_ps
2. shp
3. phase_linking
4. ps_ds
5. network_qc
6. unwrap
7. timeseries
8. corrections
9. products

## Phase Linking science

The release retains the validated production path:

- exact two-Rayleigh GLRT SHP support
- solver-aware SHP support
- sequential robust EMI
- ministack size 19
- maximum 5 compressed states
- threshold-Cholesky EMI fast path
- exact robust EMI/EVD fallback
- exact support cache
- full-SCM fallback
- full-span temporal quality
- TC threshold 0.80
- EVD acceptance retained

Performance changes do not relax the scientific selection or solver rules.

## Production optimizations retained

- cgroup-aware runtime planning
- machine-local Phase Linking autotuning
- persistent exact support cache
- threshold-Cholesky EMI backend
- resource-safe GAMMA process budgeting
- GAMMA subprocess timeout
- one-ahead prefetch implementation retained but production default OFF
- grouped crash-safe tile checkpointing
- post-Phase-Linking row-band fusion
- specialized all-pairs full-span coherence
- allocation-light sequential temporal coherence
- temporal canonical-cell reuse across sequential ministacks
- post-PL cache-capacity invariant

Temporary profiling code and rejected experimental kernels are excluded from
the release source.

## Frozen Phase Linking science baseline

Reference production workload:

- scene: 600 x 2000
- acquisitions: 38
- formal DS: 1,077,566

Frozen results:

- sequential route: 1,075,120
- sequential PL valid: 1,075,120
- EMI all stages: 1,073,390
- at least one EVD stage: 1,730
- sequential invalid: 0
- full-SCM fallback: 2,446
- combined PL valid: 1,077,566
- combined TC >= 0.80: 863,969

## Frozen performance baseline

Latest clean Phase Linking benchmark:

- sequential stage seconds: 232.158 s
- post-PL fused total: 58.004 s
- total wall: 302.462 s
- phase_linking module wall: 304.61 s

The earlier reference wall time was approximately 438.432 s, for an overall
reduction of about 31% while preserving the frozen science counts.

## Performance freeze

Performance optimization is frozen for v1.2.0. Eigensolver-level optimization
is intentionally deferred because expected end-to-end gains are small relative
to numerical-risk and maintenance cost.
