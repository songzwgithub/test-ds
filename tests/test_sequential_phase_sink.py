import numpy as np

from pypsds.phase_linking.sequential_multistage import (
    reference_real_phase,
)


def _unit_phase(angle):
    return np.exp(
        1j
        *
        np.asarray(
            angle,
            dtype=np.float32,
        )
    ).astype(
        np.complex64
    )


def test_stage0_reference_real_phase():

    # Stage 0:
    # [real0, real1, real2]
    # reference is real0.
    ph = _unit_phase(
        [
            [0.3, 0.8, -0.2],
            [-1.0, 0.4, 1.2],
        ]
    )

    got = reference_real_phase(
        ph,
        first_real_idx=0,
        reference_idx=0,
    )

    expected = (
        ph
        *
        np.conj(
            ph[:, 0][:, None]
        )
    ).astype(
        np.complex64
    )

    np.testing.assert_allclose(
        got,
        expected,
        rtol=0,
        atol=2e-7,
    )

    # Global first acquisition becomes phase origin.
    np.testing.assert_allclose(
        got[:, 0],
        np.ones(
            2,
            dtype=np.complex64,
        ),
        rtol=0,
        atol=2e-7,
    )


def test_later_stage_drops_compressed_inputs():

    # Later stage:
    #
    # [compressed0, real19, real20, real21]
    #
    # latest compressed input is the stage reference.
    ph = _unit_phase(
        [
            [0.2, 0.5, 0.9, -0.1],
            [-0.7, 0.2, 1.1, 1.4],
        ]
    )

    got = reference_real_phase(
        ph,
        first_real_idx=1,
        reference_idx=0,
    )

    expected_full = (
        ph
        *
        np.conj(
            ph[:, 0][:, None]
        )
    )

    expected = expected_full[
        :,
        1:
    ].astype(
        np.complex64
    )

    assert got.shape == (
        2,
        3,
    )

    np.testing.assert_allclose(
        got,
        expected,
        rtol=0,
        atol=2e-7,
    )


def test_reference_real_phase_m19_shapes():

    rng = np.random.default_rng(
        123
    )

    # M19 stage 0 -> solver 19 -> emit 19.
    ph0 = _unit_phase(
        rng.uniform(
            -np.pi,
            np.pi,
            size=(7, 19),
        )
    )

    out0 = reference_real_phase(
        ph0,
        first_real_idx=0,
        reference_idx=0,
    )

    assert out0.shape == (
        7,
        19,
    )

    # M19 stage 1:
    # one compressed + 19 real = solver size 20.
    ph1 = _unit_phase(
        rng.uniform(
            -np.pi,
            np.pi,
            size=(7, 20),
        )
    )

    out1 = reference_real_phase(
        ph1,
        first_real_idx=1,
        reference_idx=0,
    )

    assert out1.shape == (
        7,
        19,
    )

    assert out0.dtype == np.complex64
    assert out1.dtype == np.complex64
