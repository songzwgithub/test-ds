import numpy as np

import pypsds.phase_linking.state_domain as sd


def _fake_glrt(
    ctx,
    rows,
    cols,
    *,
    alpha,
    nslc,
    block_size,
):
    """
    Deterministic 3x3 all-true SHP support.

    This isolates and tests the fixed-point state-domain
    pruning semantics independently from the GLRT kernel.
    """

    b = len(rows)

    support = np.ones(
        (b, 3, 3),
        dtype=np.bool_,
    )

    K = np.full(
        b,
        9,
        dtype=np.int16,
    )

    return support, K


def test_fixed_point_state_domain(monkeypatch):

    monkeypatch.setattr(
        sd,
        "glrt_support_vectorized_exact",
        _fake_glrt,
    )

    valid_nonps = np.zeros(
        (7, 7),
        dtype=np.bool_,
    )

    # Stable three-pixel horizontal component.
    valid_nonps[3, 2] = True
    valid_nonps[3, 3] = True
    valid_nonps[3, 4] = True

    # Isolated state pixel: effective K = 1,
    # therefore removed for threshold=2.
    valid_nonps[1, 5] = True

    original_K = np.full(
        (7, 7),
        -1,
        dtype=np.int16,
    )

    original_K[
        valid_nonps
    ] = 9

    state, history = (
        sd.build_fixed_point_state_core(
            ctx=None,
            valid_nonps=valid_nonps,
            original_K=original_K,
            threshold=2,
            alpha=0.005,
            ndate=38,
            batch=16,
            support_block=16,
            half_row=1,
            half_col=1,
        )
    )

    expected = np.zeros(
        (7, 7),
        dtype=np.bool_,
    )

    expected[3, 2] = True
    expected[3, 3] = True
    expected[3, 4] = True

    assert np.array_equal(
        state,
        expected,
    )

    assert len(history) == 2

    assert history[0]["before"] == 4
    assert history[0]["removed"] == 1
    assert history[0]["after"] == 3

    assert history[1]["before"] == 3
    assert history[1]["removed"] == 0
    assert history[1]["after"] == 3


def test_production_alias():

    assert (
        sd.build_fixed_point_state_core
        is
        sd.fixed_point_core
    )
