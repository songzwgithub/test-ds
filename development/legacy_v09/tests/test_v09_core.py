from __future__ import annotations

import numpy as np

from pypsds.ds.shp_dolphin import glrt_statistic, glrt_threshold
from pypsds.ds.moraine_ks import moraine_ks_d_sorted
from pypsds.ds.covariance_pc_v09 import compressed_coherence_pc_v09
from pypsds.ds.phase_link_v09 import (
    image_pairs,
    robust_emi_batch_v09,
    temporal_coherence_v09,
)


def test_glrt_equal_scale_zero():
    stat = glrt_statistic(np.array([2.0]), np.array([2.0]), nslc=38)
    assert np.allclose(stat, 0.0)
    assert glrt_threshold(0.005) > 0


def test_moraine_ks_basic():
    a = np.arange(38, dtype=np.float32)
    b = a.copy()
    c = a + 10.0
    assert moraine_ks_d_sorted(a, b) == 0.0
    assert moraine_ks_d_sorted(a, c) > 0.0


def test_candidate_coherence_kernel_perfect_support():
    H, W, N = 5, 5, 4
    phase = np.exp(1j * np.array([0.0, 0.2, -0.4, 0.7], dtype=np.float32))
    x = np.empty((H, W, N), np.complex64)
    for r in range(H):
        for c in range(W):
            x[r, c] = phase * np.complex64(2.0 + 0.1 * r + 0.05 * c)
    rows = np.array([2], np.int32)
    cols = np.array([2], np.int32)
    support = np.ones((1, 3, 3), bool)
    pairs = image_pairs(N)
    coh = compressed_coherence_pc_v09(x, rows, cols, support, pairs[:, 0], pairs[:, 1])
    assert coh.shape == (1, len(pairs))
    assert np.allclose(np.abs(coh), 1.0, atol=1e-5)


def test_phase_link_temporal_coherence_perfect():
    N = 6
    pairs = image_pairs(N)
    true_phase = np.exp(1j * np.linspace(0.0, 1.0, N)).astype(np.complex64)
    coh = (true_phase[pairs[:, 0]] * np.conj(true_phase[pairs[:, 1]]))[None, :]
    phase, est, emi, evd, gmin = robust_emi_batch_v09(
        coh,
        n_images=N,
        pairs=pairs,
        beta=0.05,
        gamma_jitter=1e-6,
        emi_mu=0.99,
        reference_idx=0,
    )
    tc = temporal_coherence_v09(coh, phase, pairs)
    assert est[0] in (0, 1)
    assert np.isfinite(tc[0])
    assert tc[0] > 0.999
