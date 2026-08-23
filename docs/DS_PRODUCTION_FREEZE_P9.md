# DS production freeze after P9A-P9F

## Frozen default

The default DS path remains:

1. Rayleigh GLRT SHP support, alpha = 0.005.
2. Solver-aware SHP support policy with a base 11 x 23 window and base formal K = 48.
3. Sequential robust EMI:
   - ministack size = 19
   - maximum compressed inputs = 5
   - base state K = 24
   - target eigenvalue = 0.99
   - EVD fallback enabled
4. Full-SCM fallback for formal DS centers not routed through sequential PL.
5. Final full-span temporal coherence gate:
   - TC >= 0.80
   - pair coherence >= 0.0 (finite-value guard in practice)
   - EVD accepted.

## P9 decisions

P9A:
- Added solver-aware SHP/state/full-SCM rank safeguards.
- The N=38 reference case remained exact-array identical.

P9B/C:
- Added connected-SHP, phase similarity, closure and CRLB diagnostics.
- They remain QA products and are not hard DS gates.

P9D:
- Connected K >= 48 would remove a non-trivial number of current high-TC DS.
- Similarity, closure and CRLB are substantially correlated with TC.
- No additional hard gate was justified.

P9E:
- Positive zero-correlation thresholds did not improve the common full-pair comparison.
- Finite baseline-lag policies degraded the common full-pair comparison and increased EVD use.
- Production therefore retains:
  - zero_correlation_threshold = 0
  - baseline_lag = None.

P9F:
- M15 / max-compressed 10 was inferior to the current default.
- M24 / max-compressed 5 moved the solution closer to raw full-SCM and slightly improved TC,
  but at a large runtime cost.
- M19 / max-compressed 5 remains the default.
- M24 / max-compressed 5 is retained only as a high-accuracy benchmark profile.

## Adaptive filtering insertion point

Adaptive **interferogram** filtering must not be inserted into the amplitude-GLRT/SHP
or covariance/phase-linking inputs after this freeze.

The planned position is:

    phase_linking
      -> ds_selection
      -> ps_finalize
      -> point_stack
      -> network_prepare/build/quality
      -> network_finalize
      -> virtual_ifg_quality              # raw / unfiltered QA
      -> spatial graph / gradient quality # raw / unfiltered QA
      -> unwrap_policy
      -> [adaptive_ifg_filter]            # optional, unwrap aid only
      -> unwrap(filtered integer estimate)
      -> transfer integer cycles to original unfiltered IFG
      -> network inversion

Why:

- Goldstein/Werner-type adaptive phase filters operate on wrapped interferograms.
- The finalized temporal network is known at this point, so only interferograms that will
  actually be used need filtering.
- Filtering remains upstream of spatial unwrapping, where noise/residue reduction is useful.
- The unfiltered linked phase, TC, SHP support and PL QA remain available for DS selection
  and scientific provenance.
- Both filtered and unfiltered products must be retained during validation.

This is different from **adaptive multilooking**. Adaptive multilooking belongs before
coherence estimation / phase linking. The current pyPSDS path already performs SHP-based
statistical averaging for covariance estimation; adding another adaptive multilook there
would change the PL estimator and requires a separate scientific benchmark.


### P10 placement clarification

After inspecting the actual point-graph unwrapping path, the adaptive filter hook
is deliberately placed **after `unwrap_policy` and before `unwrap`**.

The raw virtual-IFG closure and raw spatial-gradient QA remain unfiltered.
The filtered wrapped IFG is used only to estimate integer cycles during
unwrapping.  Those integer cycles must then be transferred back to the original
unfiltered wrapped IFG.  This mirrors the MiaplPy `removeFilter` concept and
prevents filter-induced phase bias from entering the final time-series values.

## P10 adaptive-filter decision

P10A and P10B tested a Dolphin-style Goldstein/Werner raster interferogram
filter as an optional aid to the existing irregular PS/DS point-graph unwrap.

P10A used 12 representative temporal-network IFGs and showed only modest
aggregate smoothing. At alpha=0.30 the median reduction in local
`|gradient| > pi/2` observations was about 3.64%, while the median q95 phase
change on high-TC DS points was about 0.160 rad.

P10B then tested the only scientifically acceptable integration semantics:

    filtered wrapped IFG
        -> estimate integer cycles
        -> discard filtered phase
        -> original wrapped IFG + 2*pi*k

This actual unwrap test rejected both weak candidate strengths. Alpha=0.15
failed the safe-fragment solver on 12/12 representative IFGs, and alpha=0.30
also failed on 12/12. The production solver is therefore not relaxed or
modified to accommodate raster filtering.

Frozen decision:

- `adaptive_filter.enabled = false`
- Production uses the original unfiltered point-graph unwrap.
- Goldstein filtering remains benchmark/experimental code only.
- Raw virtual-IFG closure and spatial-gradient QA remain unchanged.
- Revisit only if a mature point-graph-compatible filtering/regularization
  method is available, or if a separate raster-unwrapping backend is added.

This is a compatibility result for the current pyPSDS point-graph unwrap
architecture, not a claim that Goldstein/Werner filtering is generally invalid
for raster InSAR unwrapping.
