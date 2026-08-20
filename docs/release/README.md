# Release evidence

This directory contains construction-time contract snapshots, parity reports,
and release-review evidence.

Older records may list `processing/cache/phase_corrected_yxt.npy` as a
mandatory Phase Linking input. This describes the earlier cache-based
implementation.

Current sequential production supports:

- `auto`: existing corrected-YXT cache, otherwise Gamma streaming;
- `gamma`: canonical Gamma streaming;
- `cache`: corrected-YXT cache.

Historical JSON/TXT evidence is intentionally preserved unchanged.
