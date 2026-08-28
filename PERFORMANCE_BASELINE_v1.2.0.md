# pyPSDS-GAMMA v1.2.0 Performance Baseline

## Reference workload

- effective CPU: 32
- scene: 600 x 2000
- acquisitions: 38
- formal DS: 1,077,566
- sequential Phase Linking: M19 + max5 compressed states

## Frozen science counts

```text
formal DS          : 1,077,566
sequential route   : 1,075,120
PL valid           : 1,075,120
EMI all stages     : 1,073,390
>=1 EVD stage      : 1,730
invalid            : 0
full-SCM fallback  : 2,446
combined PL valid  : 1,077,566
combined TC>=0.80  : 863,969
```

## Frozen performance

```text
stage seconds      : 232.158
post-PL fused total: 58.004
total wall         : 302.462
module wall        : 304.61
```

## Post-PL cache state

```text
LRU                : 80 -> 80
cache misses       : 0
composed hits      : 56
cache hit rate     : 100%
raw read seconds   : 0.000
correction seconds : 0.000
```

## Decision

v1.2.0 performance is frozen at this point. Remaining dominant cost lies in
the EMI fast-path eigensolver and is not modified further in this release.
