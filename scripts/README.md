# Production scripts

The active production pipeline uses semantic script names. Historical
experiment-step prefixes such as `08u`, `u34`, and `u35` are not production
interfaces.

Use the public CLI instead of invoking numbered development scripts directly:

```bash
pypsds --version
python -m pypsds.cli run --config config/pypsds.yaml --list-stages
python -m pypsds.cli run --config config/pypsds.yaml --dry-run
python -m pypsds.cli run --config config/pypsds.yaml
```

The authoritative ordered stage registry is `pypsds.pipeline.STAGES`.
