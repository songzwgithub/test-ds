# Temporal planner provenance

The pyPSDS-GAMMA temporal/ministack planning model was informed by
the ministack and compressed-SLC bookkeeping concepts in Dolphin:

- project: isce-framework/dolphin
- source file: src/dolphin/stack.py
- reference commit:
  c2f7c24a055f3de2be13e5fda717baa03cbabad2
- upstream license: BSD-3-Clause OR Apache-2.0

pyPSDS-GAMMA does not import Dolphin and the implementation in
`pypsds/phase_linking/temporal_plan.py` is an independent lightweight
dataclass implementation for GAMMA/pyPSDS processing.

U3.1 provides planning and metadata only. It does not yet implement
M < N sequential phase linking or compressed-SLC generation.

## Compressed-SLC projection

The pyPSDS compressed-SLC mathematical implementation was informed by:

- isce-framework/dolphin
- src/dolphin/phase_link/_compress.py
- src/dolphin/workflows/single.py
- commit c2f7c24a055f3de2be13e5fda717baa03cbabad2
- upstream license: BSD-3-Clause OR Apache-2.0

The local implementation is independent and does not import Dolphin.

A key implementation constraint is preserved: phase referencing is applied
to the complete ministack phase vector before previously compressed layers
are excluded from the new compressed-SLC projection.
