# pyPSDS-GAMMA 1.0

CPU/RAM-oriented PS/DS InSAR processing for GAMMA co-registered RSLC stacks.

## Validated production configuration

The v1.0 production path uses:

- amplitude-dispersion PS candidates with ADI <= 0.25;
- geometry-valid non-PS pixels as the formal DS center domain;
- Dolphin-compatible Rayleigh GLRT SHP support;
- 11 x 23 GLRT support with alpha = 0.005;
- formal SHP eligibility K >= 48;
- sequential state continuity K >= 24;
- sequential Phase Linking;
- beta = 0;
- EMI target eigenvalue mu = 0.99;
- threshold-Cholesky EMI with EVD fallback;
- exact static SHP support caching;
- final DS temporal coherence TC >= 0.80;
- PS-priority PointPhaseStack.

The historical Moraine center prior remains available for compatibility and
diagnostic work. It is not a formal production DS eligibility gate.

## Phase source

Sequential production does not require a full-scene `phase_corrected_yxt.npy`.

Phase-source selection is controlled by:

```bash
PYPSDS_PHASE_SOURCE=auto
PYPSDS_PHASE_SOURCE=gamma
PYPSDS_PHASE_SOURCE=cache
```

`auto` is the default.

- `auto`: reuse an existing corrected-YXT cache; otherwise use Gamma streaming.
- `gamma`: use canonical Gamma streaming and skip full-YXT construction.
- `cache`: use/build the full corrected-YXT cache.

The validated Gamma streaming implementation uses a canonical 128 x 256
correction grid with bounded caching.

For sequential Gamma production the pipeline retains `exact_support_cache`
but conditionally skips `phase_cache`. The legacy `full_scm` path retains
the corrected-YXT cache backend.

## Large-scene execution

Sequential production uses bounded working sets:

- sequential Phase Linking: tile + exact halo;
- full-span quality: row band + exact halo;
- full-SCM fallback: row band + exact halo;
- PS phase fill: row band;
- phase correction: canonical Gamma streaming.

## Resume

Sequential stages provide crash-safe tile checkpoints. Step04 also provides
a completion manifest and fast resume when its source/input fingerprint
remains valid.

## Configuration

`config/pypsds.yaml` (schema version 1)

## CLI

```bash
pypsds --version
pypsds config-check --config config/pypsds.yaml
pypsds doctor --config config/pypsds.yaml
python -m pypsds.cli run --config config/pypsds.yaml --list-stages
python -m pypsds.cli run --config config/pypsds.yaml --dry-run
python -m pypsds.cli run --config config/pypsds.yaml
```

Force Gamma streaming:

```bash
PYPSDS_PHASE_SOURCE=gamma \
python -m pypsds.cli run --config config/pypsds.yaml
```

## Release evidence

Historical files under `docs/release` and `docs/development` are retained as
construction/parity evidence. Some older snapshots describe
`phase_corrected_yxt.npy` as mandatory; that reflects the earlier full-cache
architecture and is not the current sequential-production contract.

Version: **1.0.0**
