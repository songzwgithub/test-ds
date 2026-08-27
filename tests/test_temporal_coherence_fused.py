from __future__ import annotations
import numpy as np
from pypsds.phase_linking.emi import (
    image_pairs,
    temporal_coherence,
    temporal_coherence_fused,
)


def _case(seed, *, B, N):
    rng = np.random.default_rng(seed)
    pairs = image_pairs(N)
    Q = pairs.shape[0]

    coh = (
        rng.normal(size=(B, Q))
        + 1j * rng.normal(size=(B, Q))
    ).astype(np.complex64)
    coh *= np.float32(0.35)

    phase = np.exp(
        1j * rng.normal(size=(B, N))
    ).astype(np.complex64)

    if B >= 8:
        coh[1, 0] = np.complex64(np.nan + 1j * np.nan)
        coh[2, 1] = np.complex64(0.0 + 0.0j)
        phase[3, 0] = np.complex64(np.nan + 1j * np.nan)

    return coh, phase, pairs


def _compare(case):
    coh, phase, pairs = case
    ref = temporal_coherence(coh, phase, pairs)
    got = temporal_coherence_fused(coh, phase, pairs)
    np.testing.assert_allclose(
        got,
        ref,
        rtol=2e-6,
        atol=2e-6,
        equal_nan=True,
    )


def test_fused_tc_matches_n19():
    _compare(_case(1901, B=512, N=19))


def test_fused_tc_matches_n20():
    _compare(_case(2001, B=512, N=20))


def test_fused_tc_threshold_classification_stable():
    coh, phase, pairs = _case(20260827, B=4096, N=20)
    ref = temporal_coherence(coh, phase, pairs)
    got = temporal_coherence_fused(coh, phase, pairs)
    np.testing.assert_array_equal(got >= 0.80, ref >= 0.80)
