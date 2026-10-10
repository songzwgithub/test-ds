# Production repository integration: Whirlwind / SNAPHU same-input benchmark

Create-only files:

- `pypsds/unwrap/whirlwind_ab.py`
- `tests/test_whirlwind_ab.py`

This benchmark uses **real complex interferogram and its actual sample coherence** at identical GAMMA radar coordinates and the same byte mask. It deliberately does **not** substitute pyPSDS temporal coherence or coarse-node quality for true interferometric coherence. Effective coherence looks and radar wavelength are mandatory inputs.

It runs both two-dimensional solvers, disables Whirlwind's synthetic bridging, saves their separate connected-component labels and reports cycle differences aligned only within shared component intersections. Neither output is considered ground truth and nothing in production is overwritten.

See the accompanying response for setup and exact invocation. Run the `--audit-only` mode before invoking solvers.
